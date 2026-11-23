#!/usr/bin/env python3
"""Preflight аренды GPU: «можно / нельзя» и одна точная следующая команда.

Зачем
=====

Оплаченный час GPU уходит на мелочи: CPU-сборка torch, чужой процесс на
карте, отсутствующий ``bitsandbytes`` (и AdamW на 29,95 ГиБ вместо Adam8bit
на 21,32), не те данные, не тот конфиг, кеш HF без весов. Скрипт проверяет
всё это ДО запуска обучения, ничего не устанавливает, не скачивает и не
обучает, и печатает «можно» либо «нельзя» с перечнем того, чего не хватает.

Что проверяется
===============

* Python 3.10–3.12 (под другие версии может не быть колёс torch/bitsandbytes);
* ``torch`` и ``torch.cuda.is_available()``, имя GPU, полная и свободная VRAM,
  поддержка bf16, CUDA в сборке torch;
* оценка памяти из конфига — той же ``estimate_memory()``, что печатает
  ``python3 -m qat.train --dry-run`` (без torch и без загрузки весов);
* влезает ли оценка (21,32 ГиБ для 1.5B) в карту и сколько осталось запаса;
* данные: подготовленный split ``data/train.jsonl`` (3607 пар, под
  ``max_steps: 350``), затем ``runs/mixture/train.jsonl`` (микс 67,5 тыс.)
  или ``distill/out/*.jsonl``: размер, число пар, поля ``question``/``answer``
  и сколько эпох дают шаги конфига;
* отложенная выборка для val loss (``val.jsonl`` рядом с train, т.е. ``data/val.jsonl``);
* ``transformers``, ``bitsandbytes``, ``accelerate``, ``numpy``, ``pyyaml``;
  если чего-то нет в venv, но есть ``.pylibs``, проба повторяется с
  ``PYTHONPATH=.pylibs``;
* кеш Hugging Face (``HF_HOME``/``HF_HUB_CACHE``/``~/.cache/huggingface``)
  и наличие в нём весов базовой модели;
* свободное место на диске (рунбук: ≥ 40 ГиБ на один прогон).

Сеть не используется: если весов в кеше нет, скрипт предупреждает и даёт
команду ``curl``, которой это проверяется. По желанию — ``--check-network``:
один TCP-коннект к huggingface.co:443, и только тогда, когда весов нет в кеше.

Коды возврата
=============

* 0 — «можно» (предупреждения допустимы, они помечены ``[warn]``);
* 1 — «нельзя»: хотя бы одна проверка ``[FAIL]``.

Примеры::

    python3 tools/rental_preflight.py
    python3 tools/rental_preflight.py --data runs/mixture/train.jsonl
    python3 tools/rental_preflight.py --config qat/configs/qwen3b.yaml \\
        --check-network

Только стандартная библиотека: torch/transformers проверяются в отдельном
процессе, чтобы падение импорта не роняло сам preflight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "qat" / "configs" / "qwen1.5b.yaml"
RUNBOOK = ROOT / "docs" / "rental-runbook.md"

#: Оценка пика для 1.5B в пилотном конфиге (рунбук §1, задача 0047).
EXPECTED_TOTAL_GIB = 21.32
#: Допуск сверки оценки с рунбуком: конфиг мог быть изменён намеренно.
EXPECTED_TOLERANCE_GIB = 0.05
#: Меньше этого запаса — советовать micro_batch 2 (рунбук §12).
MIN_MARGIN_GIB = 1.0
#: Диск: 40 ГиБ на прогон (рунбук §2), ниже 20 — не влезет (два чекпоинта ~10 ГБ).
DISK_OK_GIB = 40.0
DISK_FAIL_GIB = 20.0
#: Поддерживаемые версии Python (qat/requirements-qat.txt).
PYTHON_MIN = (3, 10)
PYTHON_MAX_EXCL = (3, 13)
#: Примерный размер весов Qwen2.5-1.5B (bf16 safetensors).
MODEL_WEIGHTS_GIB = 3.1
#: Сколько шагов делает `scripts/rental_run.sh` по умолчанию.
DEFAULT_STEPS = 100
#: Полный пилот: 350 шагов из конфига (docs/qat.md §10.4), время и цена — рунбук §10.
PILOT_STEPS = 350
PILOT_HOURS = 2.1
PILOT_RUB = (192, 256)

SYMBOL = {"ok": "[ok  ]", "warn": "[warn]", "fail": "[FAIL]", "info": "[info]"}

#: Маркер JSON-строки в выводе дочерних проб.
_MARKER = "PREFLIGHT_JSON "

#: Проба стека: версии модулей и состояние CUDA. torch импортируется последним:
#: если он падает, остальные версии всё равно попадут в отчёт.
_PROBE = r'''
import json

report = {"modules": {}, "torch": None}


def _err(exc):
    return f"{type(exc).__name__}: {exc}"


for name in ("yaml", "numpy", "accelerate", "transformers", "bitsandbytes", "torch"):
    try:
        module = __import__(name)
    except BaseException as exc:  # noqa: BLE001 - нужен любой сбой импорта
        report["modules"][name] = {"ok": False, "error": _err(exc)}
        continue
    report["modules"][name] = {
        "ok": True,
        "version": str(getattr(module, "__version__", "") or ""),
    }
    if name != "torch":
        continue
    info = {"version": str(getattr(module, "__version__", "") or "")}
    try:
        info["cuda_available"] = bool(module.cuda.is_available())
        info["cuda_version"] = str(module.version.cuda) if module.version.cuda else None
        if info["cuda_available"]:
            props = module.cuda.get_device_properties(0)
            free, total = module.cuda.mem_get_info(0)
            info["device_name"] = str(props.name)
            info["total_bytes"] = int(total)
            info["free_bytes"] = int(free)
            try:
                info["bf16"] = bool(module.cuda.is_bf16_supported())
            except BaseException:
                info["bf16"] = None
            info["capability"] = list(module.cuda.get_device_capability(0))
        else:
            info["device_name"] = None
    except BaseException as exc:  # noqa: BLE001
        info["error"] = _err(exc)
    report["torch"] = info

print("PREFLIGHT_JSON " + json.dumps(report, ensure_ascii=False))
'''

#: Оценка памяти ровно тем кодом, что и `qat.train --dry-run`, но без torch:
#: `qat.config` — чистая арифметика над конфигом.
_ESTIMATE = r'''
import json
import sys

from qat.config import estimate_memory, load_config

cfg = load_config(sys.argv[1])
est = estimate_memory(cfg)
payload = {
    "model": est["model"],
    "params": est["params"],
    "total_gib": est["total_gib"],
    "margin_gib": est["margin_gib"],
    "budget_gib": est["budget_gib"],
    "fits_24gb": est["fits_24gb"],
    "master_dtype": est["master_dtype"],
    "compute_dtype": est["compute_dtype"],
    "optimizer": est["optimizer"],
    "grad_dtype": est["grad_dtype"],
    "micro_batch_size": est["micro_batch_size"],
    "grad_accum_steps": cfg.train.grad_accum_steps,
    "max_seq_len": est["max_seq_len"],
    "components_gib": est["components_gib"],
    "revision": cfg.model.revision,
    "data_path": cfg.data.path,
    "max_steps": cfg.train.max_steps,
    "quantize_embeddings": cfg.quant.quantize_embeddings,
}
print("PREFLIGHT_JSON " + json.dumps(payload, ensure_ascii=False))
'''


# ---------------------------------------------------------------------------
# Мелкие помощники
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """Одна проверка: статус, заголовок, подробность и подсказки."""

    section: str
    status: str
    title: str
    detail: str = ""
    hints: tuple[str, ...] = ()


@dataclass
class Report:
    """Набор проверок и печать «можно/нельзя»."""

    checks: list[Check] = field(default_factory=list)

    def add(
        self,
        section: str,
        status: str,
        title: str,
        detail: str = "",
        *hints: str,
    ) -> None:
        self.checks.append(Check(section, status, title, detail, tuple(hints)))

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if check.status == "fail"]

    @property
    def warnings(self) -> list[Check]:
        return [check for check in self.checks if check.status == "warn"]


def _first_line(text: str | None, limit: int = 200) -> str:
    """Первая непустая строка ошибки, обрезанная по длине."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return ""


def _tail(text: str | None, lines: int = 3) -> str:
    parts = [line for line in (text or "").splitlines() if line.strip()]
    return " / ".join(parts[-lines:])[-400:]


def _display(path: Path | None) -> str:
    """Путь относительно корня репозитория, если это возможно."""
    if path is None:
        return "—"
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (Path.cwd(), ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return (ROOT / path).resolve()


def _human_lines(count: int) -> str:
    return f"{count:,}".replace(",", " ")


def _gib(value: int | float) -> float:
    return float(value) / 1024**3


def _run(
    command: Sequence[str],
    env: dict[str, str] | None = None,
    timeout: float = 240.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _env_with_pylibs(use_pylibs: bool) -> dict[str, str]:
    env = dict(os.environ)
    if use_pylibs:
        pylibs = str(ROOT / ".pylibs")
        old = env.get("PYTHONPATH")
        env["PYTHONPATH"] = pylibs + (os.pathsep + old if old else "")
    return env


def _json_payload(proc: subprocess.CompletedProcess[str]) -> dict[str, Any] | None:
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_MARKER):
            try:
                payload = json.loads(line[len(_MARKER):])
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


def _probe_stack(use_pylibs: bool) -> dict[str, Any]:
    try:
        proc = _run(
            [sys.executable, "-c", _PROBE],
            env=_env_with_pylibs(use_pylibs),
            timeout=240.0,
        )
    except subprocess.TimeoutExpired:
        return {"error": "проба стека не уложилась в 240 с"}
    payload = _json_payload(proc)
    if payload is None:
        return {
            "error": (
                f"проба стека не вернула JSON (код {proc.returncode}): "
                f"{_tail(proc.stderr) or 'пустой вывод'}"
            )
        }
    return payload


MODULE_ORDER = ("yaml", "numpy", "accelerate", "transformers", "bitsandbytes", "torch")


def _module(probe: dict[str, Any], name: str) -> dict[str, Any]:
    modules = probe.get("modules")
    if isinstance(modules, dict):
        entry = modules.get(name)
        if isinstance(entry, dict):
            return entry
    return {"ok": False, "error": "нет данных пробы"}


def _ok_count(probe: dict[str, Any]) -> int:
    return sum(1 for name in MODULE_ORDER if _module(probe, name).get("ok"))


def _git_head() -> str:
    try:
        proc = _run(["git", "rev-parse", "--short", "HEAD"], timeout=5.0)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------


def _count_lines(path: Path) -> int:
    total = 0
    with path.open("rb") as handle:
        for _ in handle:
            total += 1
    return total


def _first_row(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    return None, f"первая строка не JSON: {exc}"
                if not isinstance(row, dict):
                    return None, "первая строка — не JSON-объект"
                return row, ""
    except OSError as exc:
        return None, f"не читается: {exc}"
    return None, "файл пуст"


def _row_schema(row: dict[str, Any]) -> str:
    missing = [
        key for key in ("question", "answer")
        if not isinstance(row.get(key), str) or not row.get(key, "").strip()
    ]
    if missing:
        return "нет полей " + ", ".join(missing)
    return "поля question/answer на месте"


def _detect_data(explicit: str | None) -> tuple[Path | None, str]:
    """Выбрать датасет: явный --data, split 0092, микс, затем distill/out.

    ``data/train.jsonl`` — split 0092 (3607 пар train + 401 val): именно под него
    в конфиге стоят ``max_steps: 350`` (3 эпохи). ``runs/mixture`` — микс на
    67,5 тыс. пар: на нём 350 шагов лишь ~0,17 эпохи, поэтому это данные для
    экспериментов, а не для пилота из конфига.
    """
    if explicit:
        path = _resolve_path(explicit)
        return (path if path.is_file() else None), "--data"
    candidates: list[tuple[Path, str]] = [
        # Приоритет — микс: пилот должен мерить то, на чём будем жить (код + бизнес
        # + общие + поведение), а не только код. 350 шагов на 67,5 тыс. — 0,16 эпохи,
        # но 11 тыс. УНИКАЛЬНЫХ пар; на 3,6 тыс. это были бы 3 эпохи повторения.
        (ROOT / "runs" / "mixture" / "train.jsonl", "микс 75 тыс. (код+бизнес+общие+поведение)"),
        (ROOT / "data" / "train.jsonl", "split 0092 (3607 пар кода, 3 эпохи — для дешёвого сигнала)"),
        (ROOT / "distill" / "out" / "code_pairs.jsonl", "distill/out (реальные пары 0045)"),
    ]
    for path, label in candidates:
        if path.is_file():
            return path, label
    out_dir = ROOT / "distill" / "out"
    others = sorted(out_dir.glob("*.jsonl")) if out_dir.is_dir() else []
    real = [path for path in others if path.name != "pairs.jsonl"]
    if real:
        return real[0], "distill/out (первый найденный)"
    if others:
        return others[0], "distill/out/pairs.jsonl (мок)"
    return None, "не найден"


def _detect_val(explicit: str | None, data: Path | None) -> tuple[Path | None, str]:
    """val — только рядом с train: на чужой выборке val loss бессмыслен."""
    if explicit:
        path = _resolve_path(explicit)
        return (path if path.is_file() else None), "--val"
    if data is not None:
        sibling = data.with_name("val.jsonl")
        if sibling.is_file():
            return sibling, "рядом с train"
    return None, "не найден"


# ---------------------------------------------------------------------------
# HF-кеш и сеть
# ---------------------------------------------------------------------------


def _hf_hubs() -> list[Path]:
    hubs: list[Path] = []
    env_hub = os.environ.get("HF_HUB_CACHE")
    if env_hub:
        hubs.append(Path(env_hub).expanduser())
    env_home = os.environ.get("HF_HOME")
    if env_home:
        hubs.append(Path(env_home).expanduser() / "hub")
    hubs.append(Path.home() / ".cache" / "huggingface" / "hub")
    hubs.append(ROOT / ".hfcache" / "hub")
    unique: list[Path] = []
    for hub in hubs:
        if hub not in unique:
            unique.append(hub)
    return unique


def _snapshot_files(model_dir: Path) -> list[Path]:
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        return []
    return sorted(snapshots.glob("*/*"))


def _find_model_cache(model: str) -> tuple[Path | None, Path | None, str]:
    """Вернуть (hub, snapshot, причина), где лежат веса модели."""
    repo_dir = "models--" + model.replace("/", "--")
    seen: list[str] = []
    for hub in _hf_hubs():
        seen.append(str(hub))
        model_dir = hub / repo_dir
        if not model_dir.is_dir():
            continue
        files = _snapshot_files(model_dir)
        weights = [
            path for path in files
            if path.suffix in {".safetensors", ".bin"} and path.is_file()
        ]
        if weights:
            snapshot = weights[0].parent
            return hub, snapshot, ""
        if files:
            return hub, None, (
                f"в снапшоте {files[0].parent} нет .safetensors/.bin"
            )
    return None, None, "нет каталога " + repo_dir + " в " + ", ".join(seen)


def _network_ok(
    host: str = "huggingface.co",
    port: int = 443,
    timeout: float = 3.0,
) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"{host}:{port} доступен"
    except OSError as exc:
        return False, f"{host}:{port} недоступен ({type(exc).__name__}: {exc})"


# ---------------------------------------------------------------------------
# Проверки
# ---------------------------------------------------------------------------


def _check_python(report: Report) -> None:
    section = "окружение"
    report.add(
        section,
        "info",
        "корень репозитория",
        str(ROOT),
    )
    head = _git_head()
    if head:
        report.add(
            section,
            "info",
            f"HEAD {head}",
            "запишите в отчёт: прогон должен быть с этого коммита (рунбук §3)",
        )
    pylibs = ROOT / ".pylibs"
    if pylibs.is_dir():
        report.add(
            section,
            "info",
            ".pylibs найден",
            "если torch/transformers нет в venv, проба повторится с PYTHONPATH=.pylibs",
        )
    else:
        report.add(
            section,
            "info",
            ".pylibs нет",
            "на узле зависимости из venv: python3 -m pip install -r qat/requirements-qat.txt",
        )

    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    supported = PYTHON_MIN <= (sys.version_info.major, sys.version_info.minor) < PYTHON_MAX_EXCL
    if supported:
        report.add(section, "ok", f"Python {version}", "в диапазоне 3.10–3.12")
    else:
        report.add(
            section,
            "fail",
            f"Python {version} — не 3.10–3.12",
            "под другую версию может не быть колёс torch/bitsandbytes",
            "поставьте Python 3.11/3.12 и venv (рунбук §2)",
        )


def _check_stack(report: Report, probe: dict[str, Any], used_pylibs: bool) -> None:
    section = "GPU и CUDA"
    if probe.get("error"):
        report.add(
            section,
            "fail",
            "проба стека не удалась",
            str(probe["error"]),
            "запустите вручную: python3 -c \"import torch; print(torch.__version__)\"",
        )
        return
    if used_pylibs:
        report.add(
            section,
            "info",
            "использован PYTHONPATH=.pylibs",
            "venv на узле предпочтительнее; .pylibs — запасной вариант",
        )
    torch_module = _module(probe, "torch")
    torch_info = probe.get("torch") if isinstance(probe.get("torch"), dict) else {}

    if not torch_module.get("ok"):
        detail = _first_line(torch_module.get("error")) or "нет модуля torch"
        report.add(
            section,
            "fail",
            "torch не импортируется",
            detail,
            "на узле: python3 -m pip install -r qat/requirements-qat.txt",
            "нужна CUDA-сборка: "
            "python3 -c \"import torch; print(torch.version.cuda, torch.cuda.is_available())\"",
        )
        return

    version = torch_module.get("version") or "?"
    report.add(section, "ok", f"torch {version}", "модуль импортируется")

    cuda_version = torch_info.get("cuda_version")
    if not cuda_version:
        report.add(
            section,
            "fail",
            "torch собран без CUDA",
            f"torch {version}, torch.version.cuda = None — это CPU-колесо",
            "переставьте torch под драйвер узла (рунбук §2): "
            "python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124",
        )
        return

    if torch_info.get("error"):
        report.add(
            section,
            "fail",
            "CUDA-проверка упала",
            _first_line(torch_info.get("error")),
            "проверьте драйвер: nvidia-smi",
        )
        return

    if not torch_info.get("cuda_available"):
        report.add(
            section,
            "fail",
            "CUDA недоступна",
            f"torch {version} (CUDA {cuda_version}), но torch.cuda.is_available() == False",
            "проверьте драйвер и что карта видна: nvidia-smi",
            "в контейнере нужен --gpus all / правильный runtime",
        )
        return

    name = torch_info.get("device_name") or "GPU"
    total = float(torch_info.get("total_bytes") or 0)
    free = float(torch_info.get("free_bytes") or 0)
    report.add(
        section,
        "ok",
        f"CUDA доступна: {name}",
        f"torch {version}, CUDA {cuda_version}, VRAM {_gib(total):.2f} ГиБ "
        f"(свободно {_gib(free):.2f} ГиБ)",
    )

    capability = torch_info.get("capability")
    if capability:
        report.add(section, "info", f"compute capability {tuple(capability)}", "4090 — (8, 9)")

    bf16 = torch_info.get("bf16")
    if bf16 is None:
        report.add(
            section,
            "warn",
            "bf16 не проверен",
            "torch.cuda.is_bf16_supported() не ответил",
            "конфиг требует amp_dtype bfloat16: проверьте вручную",
        )
    elif not bf16:
        report.add(
            section,
            "fail",
            "bf16 не поддерживается",
            "конфиг требует train.amp_dtype: bfloat16",
            "на этой карте нужен другой amp_dtype или другая карта",
        )


def _check_config(
    report: Report,
    args: argparse.Namespace,
    probe_env: dict[str, str],
) -> dict[str, Any]:
    section = "конфиг и оценка памяти"
    config = _resolve_path(args.config)
    if not config.is_file():
        report.add(
            section,
            "fail",
            f"конфиг не найден: {_display(config)}",
            "проверьте путь --config",
            "по умолчанию нужен qat/configs/qwen1.5b.yaml",
        )
        return {}

    try:
        proc = _run(
            [sys.executable, "-c", _ESTIMATE, str(config)],
            env=probe_env,
            timeout=60.0,
        )
    except subprocess.TimeoutExpired:
        report.add(section, "fail", "оценка памяти не уложилась в 60 с", str(config))
        return {}

    estimate = _json_payload(proc)
    if estimate is None:
        report.add(
            section,
            "fail",
            "оценка памяти не считается",
            _tail(proc.stderr) or f"код {proc.returncode}",
            "проверьте конфиг: python3 -m qat.train --config "
            f"{_display(config)} --dry-run",
        )
        return {}

    total = float(estimate["total_gib"])
    margin = float(estimate["margin_gib"])
    report.add(
        section,
        "ok",
        f"dry-run оценивает {total:.2f} ГиБ",
        f"{estimate['model']}, мастер {estimate['master_dtype']}, "
        f"{estimate['optimizer']}, градиенты {estimate['grad_dtype']}, "
        f"батч {estimate['micro_batch_size']} × {estimate['max_seq_len']}; "
        f"запас {margin:+.2f} ГиБ до {estimate['budget_gib']:.0f} ГиБ "
        f"(estimate_memory — та же функция, что у --dry-run)",
    )

    if abs(total - EXPECTED_TOTAL_GIB) > EXPECTED_TOLERANCE_GIB:
        if "1.5B" in str(estimate.get("model", "")):
            report.add(
                section,
                "warn",
                f"оценка {total:.2f} ГиБ ≠ 21,32 из рунбука",
                "конфиг изменён: другой micro_batch/seq_len/master_dtype "
                "(рунбук §4: правится конфиг, а не карта)",
            )
        else:
            report.add(
                section,
                "warn",
                f"это не 1.5B-конфиг: {estimate.get('model')}",
                f"preflight сверяет с 21,32 ГиБ только 1.5B; здесь {total:.2f} ГиБ",
            )

    if not estimate.get("fits_24gb", total <= 24.0):
        report.add(
            section,
            "fail",
            f"оценка {total:.2f} ГиБ НЕ влезает в 24 ГиБ",
            "OOM на первом шаге до оплаты дешевле поймать здесь",
            "micro_batch_size: 2 (затем 1) в копии конфига (рунбук §12)",
        )

    max_steps = estimate.get("max_steps")
    if isinstance(max_steps, int) and max_steps != PILOT_STEPS:
        report.add(
            section,
            "info",
            f"train.max_steps = {max_steps}",
            "переопределяется --max-steps в scripts/rental_run.sh",
        )

    if estimate.get("revision") == "main":
        report.add(
            section,
            "warn",
            "model.revision = main",
            "прогон не воспроизведётся: зафиксируйте commit модели в конфиге",
            "рунбук §14 / комментарий в qat/configs/qwen1.5b.yaml",
        )

    report.add(
        section,
        "info",
        "квантование",
        f"эмбеддинги {estimate.get('quantize_embeddings')}, "
        f"data.path в конфиге {estimate.get('data_path')}",
    )
    return estimate


def _check_memory_fit(report: Report, estimate: dict[str, Any], probe: dict[str, Any]) -> None:
    """Сверить оценку из конфига с реальной VRAM (если карта видна)."""
    section = "конфиг и оценка памяти"
    torch_info = probe.get("torch") if isinstance(probe.get("torch"), dict) else {}
    if not estimate or not torch_info.get("cuda_available"):
        return
    total_vram = _gib(torch_info.get("total_bytes") or 0)
    free_vram = _gib(torch_info.get("free_bytes") or 0)
    total = float(estimate["total_gib"])
    if total_vram <= 0:
        return
    if total > total_vram:
        report.add(
            section,
            "fail",
            f"нужно {total:.2f} ГиБ, на карте {total_vram:.2f} ГиБ",
            "оценка уже не влезает в физическую память карты",
        )
        return
    spare = total_vram - total
    if free_vram < total:
        report.add(
            section,
            "fail",
            f"карта занята: свободно {free_vram:.2f} ГиБ, нужно {total:.2f}",
            "на карте уже есть чужой/старый процесс — он съест прогон",
            "остановите старый процесс (nvidia-smi) или пересоздайте инстанс",
        )
    elif spare < MIN_MARGIN_GIB:
        report.add(
            section,
            "warn",
            f"запас {spare:+.2f} ГиБ до объёма карты",
            "меньше 1 ГиБ: фрагментация и fused-буферы могут дать OOM",
            "micro_batch_size: 2 в копии конфига (рунбук §12)",
        )
    else:
        report.add(
            section,
            "ok",
            f"влезает: {total:.2f} из {total_vram:.2f} ГиБ, запас {spare:+.2f}",
            "оценка не знает про чужие процессы и фрагментацию — запас и есть страховка",
        )


def _check_libraries(report: Report, probe: dict[str, Any], used_pylibs: bool) -> None:
    section = "библиотеки GPU-стека"
    if probe.get("error"):
        report.add(
            section,
            "info",
            "библиотеки не проверены",
            "проба стека не вернула данные (см. [FAIL] выше)",
        )
        return
    where = " (из .pylibs)" if used_pylibs else ""
    hints = ("на узле: python3 -m pip install -r qat/requirements-qat.txt",)

    transformers = _module(probe, "transformers")
    if transformers.get("ok"):
        report.add(section, "ok", f"transformers {transformers.get('version') or '?'}{where}")
    else:
        report.add(
            section,
            "fail",
            "transformers не импортируется",
            _first_line(transformers.get("error")) or "нет модуля",
            *hints,
        )

    bitsandbytes = _module(probe, "bitsandbytes")
    if bitsandbytes.get("ok"):
        report.add(section, "ok", f"bitsandbytes {bitsandbytes.get('version') or '?'}{where}")
    else:
        report.add(
            section,
            "fail",
            "bitsandbytes нет",
            "без него adam8bit станет AdamW: 29,95 ГиБ на batch 4 — это OOM (рунбук §12)",
            *hints,
        )

    yaml = _module(probe, "yaml")
    if yaml.get("ok"):
        report.add(section, "ok", f"pyyaml {yaml.get('version') or '?'}{where}")
    else:
        report.add(
            section,
            "fail",
            "pyyaml нет",
            "qat.train без него отказывается обучать (проверка зависимостей)",
            *hints,
        )

    numpy = _module(probe, "numpy")
    if numpy.get("ok"):
        report.add(section, "ok", f"numpy {numpy.get('version') or '?'}{where}")
    else:
        report.add(
            section,
            "warn",
            "numpy нет",
            "обучение пройдёт, экспорт .trit — нет (qat/export.py требует numpy)",
            *hints,
        )

    accelerate = _module(probe, "accelerate")
    if accelerate.get("ok"):
        report.add(section, "ok", f"accelerate {accelerate.get('version') or '?'}{where}")
    else:
        report.add(
            section,
            "warn",
            "accelerate нет",
            "transformers может не найти accelerate-пути загрузки",
            *hints,
        )


def _check_data(
    report: Report,
    args: argparse.Namespace,
    estimate: dict[str, Any],
) -> tuple[Path | None, Path | None]:
    section = "данные"
    data, source = _detect_data(args.data)
    if data is None:
        report.add(
            section,
            "fail",
            "датасет не найден",
            "нет data/train.jsonl, runs/mixture/train.jsonl и distill/out/*.jsonl\n"
            f"в {ROOT}",
            "скопируйте данные: rsync -avP data/ user@HOST:~/eva32/data/",
            "или соберите локально: python3 tools/build_mixture.py",
        )
        return None, None

    try:
        size_mib = data.stat().st_size / 1024**2
        lines = _count_lines(data)
    except OSError as exc:
        report.add(section, "fail", f"данные не читаются: {_display(data)}", str(exc))
        return None, None

    report.add(
        section,
        "ok",
        f"данные: {_display(data)}",
        f"{_human_lines(lines)} пар, {size_mib:.1f} МиБ ({source})",
    )

    row, error = _first_row(data)
    if row is None:
        report.add(section, "fail", "данные битые", error, "проверьте файл: head -1 | jq .")
    else:
        schema = _row_schema(row)
        if schema.startswith("нет полей"):
            report.add(
                section,
                "fail",
                "не та схема данных",
                schema,
                "нужны question/answer (рунбук §3)",
            )
        else:
            report.add(section, "ok", "схема данных", schema)

    micro = estimate.get("micro_batch_size")
    accum = estimate.get("grad_accum_steps")
    max_steps = estimate.get("max_steps")
    if (
        lines > 0
        and all(isinstance(value, int) and value > 0 for value in (micro, accum, max_steps))
    ):
        effective = micro * accum
        steps_per_epoch = math.ceil(lines / effective)
        epochs = max_steps / steps_per_epoch
        report.add(
            section,
            "info",
            f"шагов на эпоху: {steps_per_epoch}",
            f"{_human_lines(lines)} пар / батч {effective}; "
            f"{max_steps} шагов из конфига = "
            f"{f'{epochs:.1f}'.replace('.', ',')} эпохи",
        )
        if epochs < 0.5:
            report.add(
                section,
                "warn",
                f"{max_steps} шагов на этих данных = "
                f"{f'{epochs:.2f}'.replace('.', ',')} эпохи",
                "это дым, а не пилот: конфиг рассчитан на 3607 пар (3 эпохи)",
                "для дешёвого сигнала: --data data/train.jsonl (3 эпохи кода)",
            )

    if lines == 0:
        pass
    elif data.name == "pairs.jsonl" and lines <= 1000:
        report.add(
            section,
            "warn",
            f"{_human_lines(lines)} мок-пар",
            "три эпохи на таком объёме — переобучение за первые сто шагов (рунбук §3)",
            "возьмите data/train.jsonl (split 0092) или distill/out/code_pairs.jsonl",
        )
    elif lines < 2000:
        report.add(
            section,
            "warn",
            f"мало пар: {_human_lines(lines)}",
            "для пилота ждут ~3607 пар train (рунбук §3)",
        )

    config_data = str(estimate.get("data_path") or "")
    if config_data and _resolve_path(config_data) != data:
        report.add(
            section,
            "info",
            "data.path в конфиге не совпадает",
            f"в конфиге {config_data}; scripts/rental_run.sh передаст --data явно",
        )

    val, val_source = _detect_val(args.val, data)
    if val is None:
        report.add(
            section,
            "warn",
            "val-выборки нет",
            "val loss будет не на чем считать",
            "нужен val.jsonl рядом с train (data/val.jsonl) или --val PATH",
        )
    else:
        try:
            val_lines = _count_lines(val)
            val_mib = val.stat().st_size / 1024**2
        except OSError as exc:
            report.add(section, "fail", f"val не читается: {_display(val)}", str(exc))
            val = None
        else:
            report.add(
                section,
                "ok",
                f"val: {_display(val)}",
                f"{_human_lines(val_lines)} пар, {val_mib:.1f} МиБ ({val_source})",
            )
    return data, val


def _check_disk_and_cache(
    report: Report,
    args: argparse.Namespace,
    estimate: dict[str, Any],
) -> None:
    section = "диск и кеш HF"
    usage = shutil.disk_usage(ROOT)
    free_gib = _gib(usage.free)
    if free_gib >= DISK_OK_GIB:
        report.add(
            section, "ok", f"свободно {free_gib:.1f} ГиБ",
            "на прогон ждут ≥ 40 ГиБ (рунбук §2)",
        )
    elif free_gib >= DISK_FAIL_GIB:
        report.add(
            section,
            "warn",
            f"свободно {free_gib:.1f} ГиБ",
            "рунбук просит ≥ 40 ГиБ: два чекпоинта ~10 ГБ, веса ~3 ГБ, экспорт ~1 ГБ",
            "освободите место или держите keep_last_checkpoints: 1",
        )
    else:
        report.add(
            section,
            "fail",
            f"мало места: свободно {free_gib:.1f} ГиБ",
            f"нужно ≥ {DISK_FAIL_GIB:.0f} ГиБ только на чекпоинты и веса, "
            f"рекомендуется ≥ {DISK_OK_GIB:.0f} (рунбук §2)",
            "df -h; почистите старые runs/ и кеш HF",
        )

    if not os.access(ROOT, os.W_OK):
        report.add(
            section,
            "fail",
            "нет прав на запись в репозиторий",
            f"{ROOT} только для чтения — runs/ и export/ писать некуда",
        )

    model = str(estimate.get("model") or "Qwen/Qwen2.5-1.5B-Instruct")
    hub, snapshot, reason = _find_model_cache(model)
    hf_home = os.environ.get("HF_HOME")
    if snapshot is not None:
        report.add(
            section,
            "ok",
            f"веса в кеше: {_display(snapshot)}",
            f"HF_HOME={hf_home or 'не задан'} (кеш {_display(hub)})",
        )
    else:
        detail = (
            f"весов {model} нет в кеше: {reason}; "
            f"HF_HOME={hf_home or 'не задан'}"
        )
        hints = (
            f"при первом запуске скачаются ~{MODEL_WEIGHTS_GIB:.1f} ГиБ (5–10 мин аренды)",
            "проверьте сеть: curl -sSI --max-time 5 https://huggingface.co | head -1",
        )
        if not args.check_network:
            report.add(section, "warn", "весов нет в кеше HF", detail, *hints)
        else:
            reachable, net_detail = _network_ok()
            if reachable:
                report.add(
                    section,
                    "warn",
                    "весов нет в кеше HF, но сеть есть",
                    f"{detail}; {net_detail}",
                    f"первый запуск потратит 5–10 мин на скачивание ~{MODEL_WEIGHTS_GIB:.1f} ГиБ",
                )
            else:
                report.add(
                    section,
                    "fail",
                    "весов нет в кеше и сеть недоступна",
                    f"{detail}; {net_detail}",
                    "залейте снапшот Qwen/Qwen2.5-1.5B-Instruct в HF_HOME заранее",
                )

    report.add(
        section,
        "info",
        "TEACHER_* и HF_TOKEN не нужны",
        "дистилляция уже сделана, веса Qwen публичные; ключи не требуются",
    )


def _next_command(
    config: Path | None,
    data: Path | None,
    val: Path | None,
    steps: int = DEFAULT_STEPS,
) -> str:
    parts = ["bash", "scripts/rental_run.sh"]
    if config is not None:
        parts += ["--config", _display(config)]
    if data is not None:
        parts += ["--data", _display(data)]
    if val is not None:
        parts += ["--val", _display(val)]
    if steps != DEFAULT_STEPS:
        parts += ["--steps", str(steps)]
    return " ".join(shlex.quote(part) for part in parts)


# ---------------------------------------------------------------------------
# Печать
# ---------------------------------------------------------------------------


def _print_report(report: Report) -> None:
    print("=== preflight аренды GPU: EVA32 QAT 1.5B ===")
    current = ""
    for check in report.checks:
        if check.section != current:
            current = check.section
            print()
            print(f"-- {current} " + "-" * max(3, 44 - len(current)))
        print(f"{SYMBOL[check.status]} {check.title}")
        if check.detail:
            print(f"       {check.detail}")
        for hint in check.hints:
            print(f"       → {hint}")


def _print_verdict(
    report: Report,
    config: Path | None,
    data: Path | None,
    val: Path | None,
) -> int:
    failures = report.failures
    warnings = report.warnings
    print()
    if failures:
        print(f"ИТОГ: нельзя — не прошло проверок: {len(failures)}")
        for check in failures:
            print(f"  • {check.title}")
        gpu_failed = any(
            check.status == "fail" and ("torch" in check.title or "CUDA" in check.title)
            for check in failures
        )
        if gpu_failed:
            print()
            print("пояснение: рабочего GPU-стека на этой машине нет — обучение здесь")
            print("не запустится (1.5B с fp32-мастером не влезет и на CPU). Preflight")
            print("рассчитан на арендованный узел: там те же проверки должны дать")
            print("«можно». Локально верна только оценка памяти 21,32 ГиБ из конфига.")
        print()
        print("что дальше: исправьте пункты [FAIL] выше (под каждым — команда).")
        print("затем повторите проверку:")
        print(f"  python3 {_display(Path(__file__).resolve())}"
              + (f" --config {_display(config)}" if config else ""))
        print()
        print("после «можно» — прогон:")
        print(f"  {_next_command(config, data, val)}")
        return 1

    print("ИТОГ: можно" + (f" (предупреждений: {len(warnings)} — см. [warn])" if warnings else ""))
    print()
    print("следующая команда (из корня репозитория, внутри tmux — рунбук §5):")
    print(f"  {_next_command(config, data, val)}")
    print()
    print(
        f"полный пилот: {PILOT_STEPS} шагов ≈ "
        f"{str(PILOT_HOURS).replace('.', ',')} ч ≈ "
        f"{PILOT_RUB[0]}–{PILOT_RUB[1]} ₽ при 90–120 ₽/ч (рунбук §10); "
        f"сначала дым {DEFAULT_STEPS} шагов"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 tools/rental_preflight.py",
        description=(
            "Проверка окружения аренды GPU до оплаты: torch/CUDA, память, "
            "данные, конфиг, библиотеки, кеш HF, диск. Печатает «можно/нельзя» "
            "и точную следующую команду, код 0/1."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  python3 tools/rental_preflight.py\n"
            "      проверить умолчания: qwen1.5b.yaml, data/train.jsonl, data/val.jsonl\n\n"
            "  python3 tools/rental_preflight.py --check-network\n"
            "      дополнительно один TCP-коннект к huggingface.co:443, если весов\n"
            "      нет в кеше HF (по умолчанию сеть не трогаем)\n\n"
            "коды возврата: 0 — можно, 1 — нельзя.\n"
            "рунбук: docs/rental-runbook.md §0.1\n"
        ),
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help="QAT-конфиг (по умолчанию qat/configs/qwen1.5b.yaml)",
    )
    parser.add_argument(
        "--data", default=None,
        help="JSONL с парами question/answer (по умолчанию data/train.jsonl — split "
             "0092 под max_steps 350, затем runs/mixture/train.jsonl, затем "
             "distill/out/*.jsonl)",
    )
    parser.add_argument(
        "--val", default=None,
        help="JSONL для val loss (по умолчанию val.jsonl рядом с train)",
    )
    parser.add_argument(
        "--check-network",
        action="store_true",
        help="проверить сеть до huggingface.co, если весов нет в кеше",
    )
    return parser


def _collect(
    args: argparse.Namespace,
) -> tuple[Report, dict[str, Any]]:
    report = Report()
    _check_python(report)

    probe = _probe_stack(False)
    used_pylibs = False
    pylibs = ROOT / ".pylibs"
    if pylibs.is_dir() and _ok_count(probe) < len(MODULE_ORDER):
        fallback = _probe_stack(True)
        if _ok_count(fallback) > _ok_count(probe):
            probe = fallback
            used_pylibs = True
    _check_stack(report, probe, used_pylibs)
    probe_env = _env_with_pylibs(used_pylibs)
    estimate = _check_config(report, args, probe_env)
    if estimate:
        _check_memory_fit(report, estimate, probe)
    _check_libraries(report, probe, used_pylibs)
    data, val = _check_data(report, args, estimate)
    _check_disk_and_cache(report, args, estimate)
    config_path = _resolve_path(args.config)
    context = {
        "config": config_path if config_path.is_file() else None,
        "data": data,
        "val": val,
    }
    return report, context


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report, context = _collect(args)
    except KeyboardInterrupt:
        print("[preflight] прервано", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI не должен пугать трейсбеком
        print(f"[preflight] сбой проверки: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _print_report(report)
    return _print_verdict(report, context["config"], context["data"], context["val"])


if __name__ == "__main__":
    sys.exit(main())
