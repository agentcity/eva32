#!/usr/bin/env python3
"""Раннер эксперимента 0027: эмбеддинги Qwen2.5-1.5B — none / int8 / trit.

Задача 0027 требует трёх QAT-прогонов на одних данных с одинаковым seed и
сравнения качества/размера. Руками это три раза по 3–8 часов с риском
перепутать конфиг; здесь прогоны идут последовательно, каждый в своём каталоге,
а в конце печатается таблица::

    режим | размер экспорта | loss | метрика eval | часы | ₽

Логика обучения НЕ дублируется: раннер правит в копии YAML один ключ
(``quant.quantize_embeddings``) и вызывает существующие модули через
subprocess::

    python3 -m qat.train --config <out>/_configs/<режим>.yaml \\
        --data distill/out/pairs.jsonl --out <out>/<режим> --max-steps N
    python3 -m qat.export --run <out>/<режим> --out <out>/<режим>/export

Запуск::

    python3 tools/embedding_experiment.py --data distill/out/pairs.jsonl \\
        --epochs 1 --max-steps 2000 --out runs/emb --dry-run

``--dry-run`` работает без torch (на машине разработчика) и печатает план трёх
прогонов с оценкой часов и рублей; реальный прогон без GPU-стека завершается
кодом 2 и понятным сообщением.

Эпохи переводятся в шаги: ``ceil(пар × эпохи / (micro_batch × grad_accum))``
(или ровно ``--max-steps``). Шаги у всех трёх прогонов одинаковые, seed — из
конфига (там же ``train.seed``), поэтому сравнение честное. Фактическое число
пар может быть меньше строк JSONL (``data.max_samples`` и
``data.min_answer_tokens`` в конфиге) — это влияет на «сколько эпох реально
увидит модель», но не на число шагов и не на сравнимость режимов.

Метрика eval: ``--eval-cmd`` — шаблон команды с подстановками ``{mode}``,
``{run}``, ``{export}``, ``{out}``, ``{eval_json}``; команда должна записать
JSON в ``{eval_json}`` (схема ``eval/run_eval.py``: ``passed``/``total``).
Без ``--eval-cmd`` раннер ищет готовый ``<out>/eval-<режим>.json``.
Часы — измеренное wall-clock обучение + экспорт; ₽ = часы × ``--rub-per-hour``.

Ограничение (важно для 0027): ``qat.export`` пока кладёт эмбеддинги в
``non_ternary.bin`` fp32-мастер-весами и не квантует их по режиму, поэтому
колонка «размер экспорта» у none/int8/trit совпадёт — раннер предупреждает об
этом. Разницу по размеру даст только реализация экспорта эмбеддингов
(этап 2 задачи 0027); качество и VRAM обучения режимы уже различают.

Только стандартная библиотека. Каталоги прогонов, конфиги и логи — только
внутри ``--out``; файлы вне ``--out`` раннер не пишет.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qat.config import Config as QatConfig  # noqa: E402 - нужны только дефолты
from qat.train import missing_modules  # noqa: E402 - импорт без torch
from qat.util import human_bytes, utc_now, write_json_atomic  # noqa: E402

MODES = ("none", "int8", "trit")
DEFAULT_CONFIG = ROOT / "qat" / "configs" / "qwen1.5b.yaml"
DEFAULT_DATA = ROOT / "distill" / "out" / "pairs.jsonl"
DEFAULT_OUT = ROOT / "runs" / "emb"
DEFAULT_EPOCHS = 3
#: Оценка, если данных на машине нет (0027: ~10 тыс. пар).
DEFAULT_SAMPLES = 10_000
#: docs/qat.md §4: полный прогон 2000 шагов ≈ 5–7 ч; берём 6 ч / 2000 шагов.
DEFAULT_SEC_PER_STEP = 10.8
#: docs/qat.md §9: ориентир 4090 — 90–120 ₽/ч.
DEFAULT_RUB_PER_HOUR = 100.0
#: docs/qat.md §9: ~0,5 ч на установку узла и ~0,5 ч на export/выгрузку.
SETUP_HOURS = 0.5
EXPORT_HOURS = 0.5

TABLE_HEADER = ("режим", "размер экспорта", "loss", "метрика eval", "часы", "₽")
PLAN_HEADER = ("режим", "шагов", "обучение, ч", "экспорт, ч", "всего, ч", "₽")
CONFIG_KEYS = ("micro_batch_size", "grad_accum_steps", "max_steps", "save_every")
_SCALAR_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(-?\d+)\s*$")
_KEY_LINE_RE = re.compile(r"^(\s*)quantize_embeddings\s*:\s*\S*")


# ---------------------------------------------------------------------------
# Планирование: конфиг, данные, шаги, деньги
# ---------------------------------------------------------------------------


def display_path(path: Path) -> str:
    """Путь относительно корня репозитория или cwd, если это короче."""
    for base in (ROOT, Path.cwd()):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def resolve_input_path(value: str) -> Path:
    """Входной путь: абсолютный как есть, относительный — от корня репозитория.

    Так же трактует пути ``qat/config.py::resolve_data_path``; если файла в
    репозитории нет, путь остаётся ``ROOT/<value>`` — его и покажет ошибка.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidate = (ROOT / path)
    if candidate.exists():
        return candidate
    in_cwd = (Path.cwd() / path)
    return in_cwd if in_cwd.exists() else candidate


def resolve_output_path(value: str) -> Path:
    """Выходной каталог: относительный путь — от текущего каталога."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def parse_config_scalars(text: str) -> dict[str, int]:
    """Достать из YAML несколько целых скаляров без pyyaml.

    Ищем строки вида ``ключ: 123`` по всему файлу, комментарии отрезаем.
    Первое вхождение выигрывает: в конфиге эти имена встречаются один раз,
    а повтор в чужой секции не должен незаметно переопределить батч.
    """
    found: dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        match = _SCALAR_RE.match(line)
        if match is None or match.group(1) not in CONFIG_KEYS:
            continue
        key = match.group(1)
        if key not in found:
            found[key] = int(match.group(2))
    return found


def config_values(config_path: Path) -> dict[str, int]:
    """Батч/шаги: значения из YAML поверх дефолтов ``qat.config``."""
    defaults = QatConfig().train
    values = {
        "micro_batch_size": defaults.micro_batch_size,
        "grad_accum_steps": defaults.grad_accum_steps,
        "max_steps": defaults.max_steps,
        "save_every": defaults.save_every,
    }
    values.update(parse_config_scalars(config_path.read_text(encoding="utf-8")))
    return values


@dataclass(frozen=True)
class DataInfo:
    """Что нашли по файлу пар (для плана и для раннего отказа)."""

    path: Path
    exists: bool
    samples: int
    problem: str | None = None


def count_samples(path: Path) -> int:
    """Число непустых строк JSONL — столько пар возьмёт ``qat.data.load_pairs``."""
    count = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def inspect_data(path: Path) -> DataInfo:
    """Существует ли файл, сколько в нём пар и та ли схема первой строки."""
    if not path.exists():
        return DataInfo(path, False, 0, f"файла нет: {display_path(path)}")
    if not path.is_file():
        return DataInfo(path, False, 0, f"это не файл: {display_path(path)}")
    try:
        samples = count_samples(path)
    except OSError as exc:
        return DataInfo(path, False, 0, f"не читается: {exc}")
    if samples == 0:
        return DataInfo(path, True, 0, "файл пуст")
    problem = None
    with path.open(encoding="utf-8") as fh:
        first = next((line for line in fh if line.strip()), "")
    try:
        row = json.loads(first)
    except json.JSONDecodeError as exc:
        problem = f"первая строка не JSON: {exc}"
    else:
        if not isinstance(row, dict) or "question" not in row or "answer" not in row:
            problem = "в первой строке нет полей question/answer"
    return DataInfo(path, True, samples, problem)


def steps_for_run(
    samples: int,
    epochs: int,
    micro_batch: int,
    grad_accum: int,
    max_steps: int | None = None,
) -> int:
    """Сколько шагов сделает прогон: ``--max-steps`` или пары × эпохи / батч."""
    if max_steps is not None:
        return max_steps
    per_step = max(1, micro_batch * grad_accum)
    return max(1, math.ceil(max(0, samples) * max(0, epochs) / per_step))


@dataclass(frozen=True)
class PlanRow:
    """Строка плана: режим, шаги и оценка времени."""

    mode: str
    steps: int
    train_hours: float
    export_hours: float

    @property
    def hours(self) -> float:
        return self.train_hours + self.export_hours

    def rub(self, rub_per_hour: float) -> float:
        return self.hours * rub_per_hour


def make_plan(
    *,
    samples: int,
    epochs: int,
    micro_batch: int,
    grad_accum: int,
    max_steps: int | None,
    sec_per_step: float,
    with_export: bool = True,
    modes: Sequence[str] = MODES,
) -> list[PlanRow]:
    """План трёх прогонов: одинаковые шаги, разные только каталоги/конфиги."""
    steps = steps_for_run(samples, epochs, micro_batch, grad_accum, max_steps)
    train_hours = steps * sec_per_step / 3600.0
    export_hours = EXPORT_HOURS if with_export else 0.0
    return [PlanRow(mode, steps, train_hours, export_hours) for mode in modes]


# ---------------------------------------------------------------------------
# Конфиг: копия YAML с одним другим quantize_embeddings
# ---------------------------------------------------------------------------


def patch_config_text(text: str, mode: str) -> str:
    """Вернуть YAML-текст с ``quant.quantize_embeddings: <mode>``.

    Правка построчная, без pyyaml (на машине разработчика его нет):
    существующий ключ заменяется, отсутствующий вставляется в конец секции
    ``quant:``, а если секции нет — добавляется в конец файла.
    """
    if mode not in MODES:
        raise ValueError(f"неизвестный режим: {mode!r}")
    lines = text.splitlines()

    for index, raw in enumerate(lines):
        code = raw.split("#", 1)[0].rstrip()
        match = _KEY_LINE_RE.match(code)
        if match is not None:
            comment = raw[len(code):].strip()
            line = f"{match.group(1)}quantize_embeddings: {mode}"
            lines[index] = f"{line}  {comment}" if comment else line
            return "\n".join(lines) + "\n"

    quant_start: int | None = None
    for index, raw in enumerate(lines):
        if re.match(r"^quant\s*:\s*(?:#.*)?$", raw):
            quant_start = index
            break

    if quant_start is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(("quant:", f"  quantize_embeddings: {mode}"))
        return "\n".join(lines) + "\n"

    end = len(lines)
    child_indent = "  "
    saw_child = False
    for index in range(quant_start + 1, len(lines)):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw[:1].isspace():
            end = index
            break
        if not saw_child:
            child_indent = raw[: len(raw) - len(raw.lstrip())]
            saw_child = True

    insert_at = end
    while insert_at - 1 > quant_start and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, f"{child_indent}quantize_embeddings: {mode}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Команды subprocess
# ---------------------------------------------------------------------------


def build_train_command(
    *,
    config: Path,
    data: Path,
    out: Path,
    steps: int,
    resume: bool = False,
    extra: Sequence[str] = (),
) -> list[str]:
    """Команда обучения: существующий ``qat.train``, режим — только в конфиге."""
    command = [
        sys.executable, "-m", "qat.train",
        "--config", str(config),
        "--data", str(data),
        "--out", str(out),
        "--max-steps", str(steps),
    ]
    if resume:
        command.append("--resume")
    command.extend(extra)
    return command


#: Режимы, для которых экспорт пока невозможен: нужен контейнерный kind 4 (int8 +
#: масштабы) и чтение в src/vtm/loader.c. Качество при этом меряется на чекпоинте,
#: а размер считается формулой — поэтому прогон не считается проваленным.
EXPORT_DEFERRED: dict[str, str] = {
    "int8": "экспорт int8 отложен: нужен kind 4 в контейнере (задача 0044)",
}


def build_export_command(*, run: Path, export: Path) -> list[str]:
    """Команда экспорта чекпоинта в ``ternary.pt/.trit`` + ``manifest.json``."""
    return [sys.executable, "-m", "qat.export", "--run", str(run), "--out", str(export)]


def run_streamed(
    command: Sequence[str] | str,
    *,
    log_path: Path,
    prefix: str,
    shell: bool = False,
) -> int:
    """Выполнить команду, дублируя stdout/stderr в консоль и в лог.

    Вывод печатается с префиксом режима: по консоли видно, какой из трёх
    прогонов сейчас говорит. Файл лога — только внутри ``--out``.
    """
    import subprocess  # noqa: PLC0415 - нужен только на реальном запуске

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            shell=shell,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(f"[{prefix}] {line}" if line.strip() else line)
                sys.stdout.flush()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except Exception:  # noqa: BLE001 - второй Ctrl-C важнее
                process.kill()
            print(f"\n[emb] прервано пользователем; лог: {display_path(log_path)}")
            return 130
        return process.wait()


# ---------------------------------------------------------------------------
# Сбор результатов: loss, размер экспорта, метрика
# ---------------------------------------------------------------------------


def read_last_loss(log_path: Path) -> float | None:
    """Последний конечный ``loss`` из ``log.jsonl`` прогона (None, если нет)."""
    if not log_path.exists():
        return None
    loss: float | None = None
    try:
        with log_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                value = record.get("loss")
                if isinstance(value, (int, float)) and math.isfinite(value):
                    loss = float(value)
    except OSError:
        return None
    return loss


def read_metric(path: Path) -> tuple[int, int] | None:
    """``passed``/``total`` из JSON ``eval/run_eval.py``; None, если не годится."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    passed, total = payload.get("passed"), payload.get("total")
    if not isinstance(passed, int) or not isinstance(total, int):
        return None
    if total <= 0 or not 0 <= passed <= total:
        return None
    return passed, total


def directory_size(path: Path) -> int | None:
    """Суммарный размер файлов каталога; None, если каталога нет."""
    if not path.is_dir():
        return None
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                continue
    return total


@dataclass
class RunResult:
    """Один режим: где лежит, сколько весит, как учился и что показал eval."""

    mode: str
    out_dir: Path
    status: str = "pending"
    export_dir: Path | None = None
    export_bytes: int | None = None
    loss: float | None = None
    metric: tuple[int, int] | None = None
    hours: float | None = None
    note: str = ""

    def rub(self, rub_per_hour: float) -> float | None:
        return None if self.hours is None else self.hours * rub_per_hour


def format_metric(passed: int, total: int) -> str:
    """``8/10 (80%)`` — компактно и сравнимо между режимами."""
    if total <= 0:
        return "—"
    return f"{passed}/{total} ({100.0 * passed / total:.0f}%)"


def _render_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Выровнять таблицу по ширине колонок: текст слева, числа справа."""
    widths = [len(name) for name in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(row: Sequence[str]) -> str:
        cells = [
            cell.ljust(widths[index]) if index == 0 else cell.rjust(widths[index])
            for index, cell in enumerate(row)
        ]
        return " | ".join(cells).rstrip()

    lines = [render(header)]
    body = [render(row) for row in rows]
    lines.append("-" * max(len(line) for line in lines + body))
    lines.extend(body)
    return lines


def format_table(results: Sequence[RunResult], rub_per_hour: float = DEFAULT_RUB_PER_HOUR) -> str:
    """Таблица сравнения: режим | размер экспорта | loss | метрика | часы | ₽."""
    rows: list[list[str]] = []
    for result in results:
        rub = result.rub(rub_per_hour)
        rows.append([
            result.mode,
            "—" if result.export_bytes is None else human_bytes(result.export_bytes),
            "—" if result.loss is None else f"{result.loss:.4f}",
            "—" if result.metric is None else format_metric(*result.metric),
            "—" if result.hours is None else f"{result.hours:.2f}",
            "—" if rub is None else f"{rub:.0f}",
        ])
    sizes = [r.export_bytes for r in results if r.export_bytes is not None]
    hours = [r.hours for r in results if r.hours is not None]
    rows.append([
        "ИТОГО",
        human_bytes(sum(sizes)) if sizes else "—",
        "—",
        "—",
        f"{sum(hours):.2f}" if hours else "—",
        f"{sum(hours) * rub_per_hour:.0f}" if hours else "—",
    ])
    return "\n".join(_render_table(TABLE_HEADER, rows))


def format_plan(plan: Sequence[PlanRow], rub_per_hour: float = DEFAULT_RUB_PER_HOUR) -> str:
    """План трёх прогонов: шаги, часы обучения/экспорта и деньги."""
    rows = [
        [
            row.mode,
            str(row.steps),
            f"{row.train_hours:.2f}",
            f"{row.export_hours:.2f}",
            f"{row.hours:.2f}",
            f"{row.rub(rub_per_hour):.0f}",
        ]
        for row in plan
    ]
    total_hours = sum(row.hours for row in plan)
    rows.append([
        "ИТОГО",
        str(sum(row.steps for row in plan)),
        f"{sum(row.train_hours for row in plan):.2f}",
        f"{sum(row.export_hours for row in plan):.2f}",
        f"{total_hours:.2f}",
        f"{total_hours * rub_per_hour:.0f}",
    ])
    return "\n".join(_render_table(PLAN_HEADER, rows))


def comparison_notes(results: Sequence[RunResult]) -> list[str]:
    """Сравнение метрик с baseline ``none`` в процентных пунктах (0027)."""
    baseline = next((r for r in results if r.mode == "none" and r.metric), None)
    if baseline is None:
        return []
    passed, total = baseline.metric  # type: ignore[misc]
    base_percent = 100.0 * passed / total
    lines = [f"baseline none: {format_metric(passed, total)}"]
    for result in results:
        if result.mode == "none" or result.metric is None:
            continue
        other_passed, other_total = result.metric
        if other_total != total:
            lines.append(
                f"  {result.mode}: {format_metric(other_passed, other_total)} — "
                f"другой размер набора, проценты не сравнимы"
            )
            continue
        delta = 100.0 * other_passed / other_total - base_percent
        verdict = "в пределах порога 0027 (≤3 п.п.)" if abs(delta) <= 3.0 else "вне порога 0027 (>3 п.п.)"
        lines.append(f"  {result.mode}: {format_metric(other_passed, other_total)} "
                     f"({delta:+.1f} п.п. к none) — {verdict}")
    return lines


# ---------------------------------------------------------------------------
# Реальный прогон
# ---------------------------------------------------------------------------


def _environment_error(missing: Sequence[str]) -> str:
    return (
        "[emb] обучение невозможно: не хватает зависимостей GPU-стека: "
        + ", ".join(missing) + "\n"
        "      QAT запускается только на GPU-машине:\n"
        "          python3 -m pip install -r qat/requirements-qat.txt\n"
        "      На машине разработчика доступны план и тесты:\n"
        "          python3 tools/embedding_experiment.py --dry-run\n"
        "          python3 -m unittest tests.test_embedding_experiment -v"
    )


def _run_dir_ready(run_dir: Path) -> bool:
    """Прогон завершён (DONE) или есть чекпоинт, с которого можно продолжить."""
    if (run_dir / "DONE").exists():
        return True
    return (run_dir / "checkpoints" / "latest.txt").exists()


def _load_previous_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _previous_run(summary: dict[str, Any], mode: str) -> dict[str, Any]:
    for run in summary.get("runs", []):
        if isinstance(run, dict) and run.get("mode") == mode:
            return run
    return {}


def _write_summary(path: Path, args: argparse.Namespace, plan: Sequence[PlanRow],
                   results: Sequence[RunResult], samples: int) -> None:
    payload = {
        "ts": utc_now(),
        "task": "0027",
        "out": str(args.out_dir),
        "config": str(args.config),
        "data": str(args.data),
        "samples": samples,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "steps_per_run": plan[0].steps if plan else 0,
        "sec_per_step": args.sec_per_step,
        "rub_per_hour": args.rub_per_hour,
        "runs": [
            {
                "mode": result.mode,
                "status": result.status,
                "out_dir": str(result.out_dir),
                "export_dir": str(result.export_dir) if result.export_dir else None,
                "export_bytes": result.export_bytes,
                "loss": result.loss,
                "metric": (
                    {"passed": result.metric[0], "total": result.metric[1]}
                    if result.metric else None
                ),
                "hours": result.hours,
                "rub": result.rub(args.rub_per_hour),
                "note": result.note,
            }
            for result in results
        ],
    }
    write_json_atomic(path, payload)


def run_experiment(args: argparse.Namespace) -> int:
    """Последовательно обучить три режима, экспортировать и собрать сравнение."""
    missing = missing_modules()
    if missing:
        print(_environment_error(missing), file=sys.stderr)
        return 2

    try:
        config_text = args.config.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[emb] не читается конфиг {display_path(args.config)}: {exc}", file=sys.stderr)
        return 2
    data_info = inspect_data(args.data)
    if not data_info.exists or data_info.samples == 0:
        print(f"[emb] данные не годятся: {data_info.problem}", file=sys.stderr)
        print("      ожидается JSONL с парами question/answer; "
              "см. docs/qat.md §3", file=sys.stderr)
        return 2
    if data_info.problem:
        print(f"[emb] ВНИМАНИЕ: {data_info.problem} — qat.train может отказаться",
              file=sys.stderr)

    values = config_values(args.config)
    plan = make_plan(
        samples=data_info.samples,
        epochs=args.epochs,
        micro_batch=values["micro_batch_size"],
        grad_accum=values["grad_accum_steps"],
        max_steps=args.max_steps,
        sec_per_step=args.sec_per_step,
        with_export=not args.no_export,
    )

    out_root = args.out_dir
    configs_dir = out_root / "_configs"
    logs_dir = out_root / "_logs"
    previous = _load_previous_summary(out_root / "summary.json")

    print(f"EVA32 · эксперимент 0027 (эмбеддинги: {' / '.join(MODES)})")
    print(f"[emb] каталог: {display_path(out_root)}")
    print(f"[emb] конфиг:  {display_path(args.config)} "
          f"(quantize_embeddings правится в копиях)")
    print(f"[emb] данные:  {display_path(args.data)} — {data_info.samples} пар")
    print(f"[emb] шагов на прогон: {plan[0].steps}, "
          f"батч {values['micro_batch_size']}×{values['grad_accum_steps']}")
    print()

    out_root.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    failed = False

    for index, row in enumerate(plan, start=1):
        mode = row.mode
        run_dir = out_root / mode
        export_dir = run_dir / "export"
        config_path = configs_dir / f"{mode}.yaml"
        result = RunResult(mode=mode, out_dir=run_dir)
        results.append(result)
        head = f"[{index}/{len(plan)}] {mode}"
        print(f"{head}: quantize_embeddings={mode} → {display_path(run_dir)}")

        if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
            result.status = "failed"
            result.note = (f"каталог уже существует: {display_path(run_dir)}; "
                           f"продолжить — --resume, или укажите другой --out")
            print(f"{head}: ОШИБКА: {result.note}", file=sys.stderr)
            failed = True
            if not args.keep_going:
                break
            continue

        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(patch_config_text(config_text, mode), encoding="utf-8")
        except OSError as exc:
            result.status = "failed"
            result.note = f"не записать конфиг {display_path(config_path)}: {exc}"
            print(f"{head}: ОШИБКА: {result.note}", file=sys.stderr)
            failed = True
            if not args.keep_going:
                break
            continue

        finished_before = (run_dir / "DONE").exists()
        resume_run = args.resume and not finished_before and _run_dir_ready(run_dir)
        started = time.monotonic()

        if finished_before:
            print(f"{head}: прогон уже завершён (DONE), обучение пропущено")
            result.status = "done"
        else:
            command = build_train_command(
                config=config_path,
                data=args.data,
                out=run_dir,
                steps=row.steps,
                resume=resume_run,
                extra=args.train_arg,
            )
            print(f"{head}: {' '.join(command)}")
            code = run_streamed(
                command,
                log_path=logs_dir / f"{mode}.train.log",
                prefix=mode,
            )
            if code != 0:
                result.status = "failed"
                result.note = (f"qat.train вернул код {code}; "
                               f"лог: {display_path(logs_dir / f'{mode}.train.log')}")
                print(f"{head}: ОШИБКА: {result.note}", file=sys.stderr)
                failed = True
                if not args.keep_going:
                    # часы и loss уже потраченного прогона всё равно учитываем
                    result.hours = (time.monotonic() - started) / 3600.0
                    result.loss = read_last_loss(run_dir / "log.jsonl")
                    break
            else:
                result.status = "done"

        deferred_note = EXPORT_DEFERRED.get(mode)
        if (not args.no_export and result.status == "done"
                and deferred_note is None):
            command = build_export_command(run=run_dir, export=export_dir)
            print(f"{head}: экспорт → {display_path(export_dir)}")
            code = run_streamed(
                command,
                log_path=logs_dir / f"{mode}.export.log",
                prefix=f"{mode}/export",
            )
            if code != 0:
                result.note = (f"qat.export вернул код {code}; "
                               f"лог: {display_path(logs_dir / f'{mode}.export.log')}")
                print(f"{head}: экспорт не удался: {result.note}", file=sys.stderr)
                failed = True
            else:
                result.export_dir = export_dir
                result.export_bytes = directory_size(export_dir)

        if deferred_note is not None and result.status == "done":
            result.note = deferred_note
            print(f"{head}: {deferred_note}")

        result.hours = (time.monotonic() - started) / 3600.0
        if finished_before:
            old = _previous_run(previous, mode)
            old_hours = old.get("hours")
            result.hours = float(old_hours) if isinstance(old_hours, (int, float)) else None

        result.loss = read_last_loss(run_dir / "log.jsonl")
        export_ok = (args.no_export or result.export_dir is not None
                     or mode in EXPORT_DEFERRED)
        eval_json = out_root / f"eval-{mode}.json"
        if args.eval_cmd and result.status == "done" and export_ok:
            eval_command = args.eval_cmd.format(
                mode=mode,
                run=str(run_dir),
                export=str(result.export_dir or run_dir),
                out=str(out_root),
                eval_json=str(eval_json),
            )
            print(f"{head}: eval: {eval_command}")
            print(f"{head}: eval → {display_path(eval_json)}")
            eval_code = run_streamed(
                eval_command,
                log_path=logs_dir / f"{mode}.eval.log",
                prefix=f"{mode}/eval",
                shell=True,
            )
            if eval_code != 0:
                result.note = (result.note + " " if result.note else "") + (
                    f"eval вернул код {eval_code}; лог: {display_path(logs_dir / f'{mode}.eval.log')}"
                )
        result.metric = read_metric(eval_json)
        print(f"{head}: loss={result.loss if result.loss is not None else '—'}, "
              f"экспорт={'—' if result.export_bytes is None else human_bytes(result.export_bytes)}, "
              f"метрика={format_metric(*result.metric) if result.metric else '—'}, "
              f"часов={(result.hours or 0.0):.2f}")
        print()

    print("Сравнение прогонов")
    print(format_table(results, args.rub_per_hour))
    notes = [line for line in comparison_notes(results)]
    if notes:
        print()
        print("Разница метрик:")
        print("\n".join(notes))
    no_metric = [r.mode for r in results if r.metric is None and r.status == "done"]
    if no_metric:
        print()
        print(f"метрика eval не собрана для: {', '.join(no_metric)} — "
              f"передайте --eval-cmd или положите eval-<режим>.json в "
              f"{display_path(args.out_dir)}")
    sizes = [result.export_bytes for result in results
             if result.export_bytes is not None]
    if len(sizes) >= 2 and len(set(sizes)) == 1:
        print()
        print("ВНИМАНИЕ: размер экспорта одинаков у всех режимов: qat.export пока "
              "пишет эмбеддинги в non_ternary.bin fp32-мастер-весами и не квантует "
              "их по режиму. Разницу по размеру даст этап 2 задачи 0027 (экспорт "
              "эмбеддингов), а не сами прогоны QAT; режимы уже различаются "
              "качеством и памятью при обучении.")
    problems = [r for r in results if r.status == "failed"]
    if problems:
        print()
        for result in problems:
            print(f"  {result.mode}: {result.note}")
    for result in results:
        if result.status == "done" and result.note and result not in problems:
            print(f"  {result.mode}: {result.note}")

    _write_summary(out_root / "summary.json", args, plan, results, data_info.samples)
    print()
    print(f"[emb] итог: {display_path(out_root / 'summary.json')}")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------


def dry_run(args: argparse.Namespace) -> int:
    """Печать плана без torch, сети и обучения; код выхода всегда 0."""
    config_error: str | None = None
    try:
        config_text = args.config.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[emb] ОШИБКА: не читается конфиг {display_path(args.config)}: {exc}",
              file=sys.stderr)
        return 2
    try:
        values = config_values(args.config)
    except (OSError, ValueError) as exc:
        print(f"[emb] не удалось прочитать конфиг ({exc}); беру дефолты qat.config")
        config_error = str(exc)
        defaults = QatConfig().train
        values = {
            "micro_batch_size": defaults.micro_batch_size,
            "grad_accum_steps": defaults.grad_accum_steps,
            "max_steps": defaults.max_steps,
            "save_every": defaults.save_every,
        }

    # Проверяем, что подстановка режима вообще возможна в этом YAML.
    for mode in MODES:
        patched = patch_config_text(config_text, mode)
        if f"quantize_embeddings: {mode}" not in patched:
            print(f"[emb] ОШИБКА: не удалось выставить quantize_embeddings={mode} "
                  f"в {display_path(args.config)}", file=sys.stderr)
            return 2

    data_info = inspect_data(args.data)
    if data_info.exists and data_info.samples > 0:
        samples = data_info.samples
        samples_source = f"{samples} пар ({display_path(args.data)})"
    else:
        samples = args.samples
        samples_source = (f"оценка {samples} пар: {data_info.problem}; "
                          f"реальный запуск без данных невозможен")
    missing = missing_modules()

    plan = make_plan(
        samples=samples,
        epochs=args.epochs,
        micro_batch=values["micro_batch_size"],
        grad_accum=values["grad_accum_steps"],
        max_steps=args.max_steps,
        sec_per_step=args.sec_per_step,
        with_export=not args.no_export,
    )
    total_hours = sum(row.hours for row in plan)
    total_rub = total_hours * args.rub_per_hour

    print("EVA32 · эксперимент 0027: эмбеддинги none / int8 / trit")
    print("ПЛАН (--dry-run): обучение не запускается, каталоги не создаются")
    print()
    print(f"каталог прогонов:  {display_path(args.out_dir)}")
    print(f"конфиг:            {display_path(args.config)} "
          f"(quantize_embeddings подменяется в _configs/<режим>.yaml)")
    print(f"данные:            {samples_source}")
    print(f"эпохи:             {args.epochs}")
    print(f"шагов на прогон:   {plan[0].steps} "
          f"(батч {values['micro_batch_size']}×{values['grad_accum_steps']} = "
          f"{values['micro_batch_size'] * values['grad_accum_steps']} "
          f"последовательностей)")
    print(f"оценка шага:       {args.sec_per_step:.1f} с "
          f"(docs/qat.md: 2000 шагов ≈ 6 ч)")
    print(f"ставка:            {args.rub_per_hour:.0f} ₽/ч "
          f"(docs/qat.md §9: 4090 — 90–120 ₽/ч)")
    if args.max_steps is not None:
        print(f"шаги:              ровно --max-steps {args.max_steps}; --epochs не влияет")
    else:
        print(f"шаги:              ceil({samples} × {args.epochs} / "
              f"{values['micro_batch_size'] * values['grad_accum_steps']})")
    if args.no_export:
        print("экспорт:           выключен (--no-export), часы/₽ — только обучение")
    else:
        print(f"экспорт:           {EXPORT_HOURS:.1f} ч на прогон "
              f"(qat.export + выгрузка, docs/qat.md §9)")
    print(f"torch:             {'нет' if missing else 'есть'}"
          + (f" (не хватает: {', '.join(missing)}) — "
             f"этот режим работает без GPU-стека" if missing else ""))
    if data_info.exists and data_info.problem:
        print(f"данные:            ВНИМАНИЕ: {data_info.problem}")
    print()
    print(format_plan(plan, args.rub_per_hour))
    print()
    print(f"итого прогонов:    {len(plan)} × {plan[0].steps} шагов, "
          f"{total_hours:.2f} ч, {total_rub:.0f} ₽")
    print(f"с подготовкой узла: +{SETUP_HOURS:.1f} ч → "
          f"{total_hours + SETUP_HOURS:.2f} ч, "
          f"{(total_hours + SETUP_HOURS) * args.rub_per_hour:.0f} ₽")
    print("eval не входит в смету: он идёт на своей машине против "
          "поднятого export (docs/qat.md §8).")
    if config_error:
        print(f"ВНИМАНИЕ: конфиг прочитан не полностью ({config_error}); "
              f"батч/шаги взяты из дефолтов qat.config.")
    print()
    print("команды, которые будут выполнены (пример для none):")
    example = plan[0]
    config_path = args.out_dir / "_configs" / f"{example.mode}.yaml"
    run_dir = args.out_dir / example.mode
    print("  " + " ".join(build_train_command(
        config=config_path, data=args.data, out=run_dir, steps=example.steps)))
    if not args.no_export:
        print("  " + " ".join(build_export_command(
            run=run_dir, export=run_dir / "export")))
    print()
    print("реальный запуск — командой без --dry-run (нужен torch на GPU-машине):")
    print(f"  python3 tools/embedding_experiment.py --data {display_path(args.data)} "
          f"--out {display_path(args.out_dir)}"
          + (f" --max-steps {args.max_steps}" if args.max_steps else f" --epochs {args.epochs}"))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 tools/embedding_experiment.py",
        description=(
            "Эксперимент 0027: три QAT-прогона (none/int8/trit) и таблица "
            "«режим | размер экспорта | loss | метрика eval | часы | ₽»."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  python3 tools/embedding_experiment.py --dry-run\n"
            "      план трёх прогонов, часы и деньги; torch не нужен\n\n"
            "  python3 tools/embedding_experiment.py --data distill/out/pairs.jsonl \\\n"
            "      --epochs 1 --max-steps 2000 --out runs/emb\n"
            "      реальные прогоны на GPU-машине\n\n"
            "  python3 tools/embedding_experiment.py --out runs/emb --resume\n"
            "      продолжить после обрыва, завершённые режимы пропустить\n\n"
            "  python3 tools/embedding_experiment.py --out runs/emb \\\n"
            "      --eval-cmd 'python3 eval/run_eval.py --base http://127.0.0.1:8000 \\\n"
            "      --model eva32 --out {eval_json}'\n"
            "      заполнить колонку «метрика eval»\n\n"
            "режим задаётся только копией конфига: qat.train не имеет CLI-флага\n"
            "для quantize_embeddings (правка qat/* вне зоны задачи).\n"
        ),
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="JSONL с парами question/answer (по умолчанию "
                             "distill/out/pairs.jsonl)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="базовый YAML; копии с другим quantize_embeddings "
                             "пишутся в <out>/_configs")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="каталог эксперимента: <out>/<режим>, _configs, _logs "
                             "(по умолчанию runs/emb)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"эпох на прогон для расчёта шагов (по умолчанию "
                             f"{DEFAULT_EPOCHS}); игнорируется при --max-steps")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="ровно столько шагов в каждом прогоне (перебивает --epochs)")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"пар для оценки, если данных нет (по умолчанию "
                             f"{DEFAULT_SAMPLES})")
    parser.add_argument("--dry-run", action="store_true",
                        help="только план: без torch, данных и обучения; код 0")
    parser.add_argument("--resume", action="store_true",
                        help="продолжить незавершённые прогоны, готовые (DONE) пропустить")
    parser.add_argument("--keep-going", action="store_true",
                        help="продолжать остальные режимы после падения одного")
    parser.add_argument("--no-export", action="store_true",
                        help="не запускать qat.export (колонка размера будет «—»)")
    parser.add_argument("--sec-per-step", type=float, default=DEFAULT_SEC_PER_STEP,
                        help=f"оценка секунд на шаг (по умолчанию "
                             f"{DEFAULT_SEC_PER_STEP}: 2000 шагов ≈ 6 ч)")
    parser.add_argument("--rub-per-hour", type=float, default=DEFAULT_RUB_PER_HOUR,
                        help=f"ставка аренды, ₽/ч (по умолчанию "
                             f"{DEFAULT_RUB_PER_HOUR:.0f})")
    parser.add_argument("--eval-cmd", default=None,
                        help="шаблон команды eval с {mode}, {run}, {export}, {out}, "
                             "{eval_json}; JSON пишется в {eval_json}")
    parser.add_argument("--train-arg", action="append", default=[], metavar="ARG",
                        help="дополнительный аргумент qat.train; для флагов — "
                             "через '=': --train-arg=--no-save (можно несколько раз)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.epochs <= 0:
        parser.error("--epochs должен быть > 0")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps должен быть > 0")
    if args.samples <= 0:
        parser.error("--samples должен быть > 0")
    if args.sec_per_step <= 0:
        parser.error("--sec-per-step должен быть > 0")
    if args.rub_per_hour <= 0:
        parser.error("--rub-per-hour должен быть > 0")

    args.config = resolve_input_path(args.config)
    args.data = resolve_input_path(args.data)
    args.out_dir = resolve_output_path(args.out)
    if not args.config.is_file():
        print(f"[emb] ОШИБКА: конфиг не найден: {display_path(args.config)}",
              file=sys.stderr)
        return 2

    if args.dry_run:
        return dry_run(args)
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
