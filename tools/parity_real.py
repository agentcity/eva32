#!/usr/bin/env python3
"""Сверка настоящей Qwen2.5-1.5B-Instruct: transformers ↔ VTM C-рантайм.

Зачем
=====

Задача 0006 («полная модель в ВТМ») до этого проверялась только на
синтетике: C-интерпретатор сходился с Python-эталоном на случайных весах,
но никто не сверял его с настоящей моделью. Этот скрипт запускается на
машине, где есть torch и transformers, и делает полный цикл:

1. ``transformers`` загружает ``Qwen/Qwen2.5-1.5B-Instruct``, токенизирует
   фиксированный промпт и снимает логиты последней позиции;
2. из тех же весов собирается QAT-чекпоинт (``qat.model.apply_ternary``),
   ``python3 -m qat.export`` пишет ``ternary.trit`` + ``non_ternary.bin`` +
   ``manifest.json`` (режим ``--quantize-embeddings none`` по умолчанию);
3. ``tools/vtm_compile.py`` собирает ``.vtm``;
4. тот же промпт исполняется C-рантаймом, снимаются логиты;
5. считается max|Δ| логитов, совпадение argmax и пересечение top-5.

Почему эталонов два
====================

VTM исполняет **тернарные** веса: q/k/v/o/gate/up/down квантуются в
``{-scale, 0, +scale}``. Если сравнивать C с полной точностью HF, мы
измерим ошибку квантования (она большая), а не расхождение рантайма.
Поэтому скрипт делает два forward'а на HF:

* ``hf_quantized`` — та же модель, но с ``apply_ternary`` (forward
  возвращает ровно те де-квантованные веса, которые экспортирует
  ``qat.export``). **Это эталон для порога.** Разница с C — только
  fp16-масштабы, порядок суммирования, RoPE/ATTN и округление eps.
* ``hf_full`` — исходная полная точность. Справочно: показывает цену
  тернаризации; на порог не влияет.

Порог: argmax обязан совпасть, ``max|Δ| < 1e-2`` (ориентир на fp16-масштабы
и fp32-накопление). Полная точность HF почти наверняка даст больше — это
нормально и не является провалом рантайма. Если порог не прошёл, сначала
проверьте ``--scale-dtype float32``: так из замера уходит округление
масштабов до fp16 и остаётся только расхождение самого рантайма.

Почему C-проба, а не только ``vtm run``
=======================================

``src/vtm/vtm run ... --tokens ...`` печатает только сгенерированный токен
(argmax), но не вектор логитов. Для max|Δ| нужны числа. Скрипт делает оба
шага:

* запускает ровно ``vtm run <model.vtm> --tokens ... --max-new 1`` и берёт
  из вывода argmax — это проверка штатного CLI;
* собирает крошечную C-пробу (в ``--workdir``, не в репозитории) и линкует её
  с ``src/vtm/libvtm.a``; проба грузит тот же ``.vtm``, исполняет те же
  токены и печатает тензор ``logits``. Это тот же интерпретатор и тот же
  ``libvtm.a``, что у CLI.

Если C-проба не собирается (нет компилятора/библиотеки), сверка помечается
``partial``: argmax по CLI ещё проверяется, а ``max|Δ|`` — нет.

Артефакты
=========

В ``--workdir`` (по умолчанию ``data/parity_real``, каталог в ``.gitignore``):

* ``qat_run/checkpoints/step_000000/model.pt`` — QAT-чекпоинт из HF-весов;
* ``export/{manifest.json,ternary.trit,non_ternary.bin}`` — ``qat.export``;
* ``model.vtm`` — контейнер;
* ``parity_probe.c`` / ``parity_probe`` — C-проба логитов;
* ``parity.json`` — результат (JSON) + человекочитаемая сводка в stdout.

Коды возврата
=============

* 0 — сверка пройдена (argmax совпал, max|Δ| < 1e-2);
* 1 — сверка провалена или неполная (``partial``);
* 2 — нет зависимостей/инструментов, модель не поддержана, ошибка экспорта.

Без torch скрипт не пытается ничего загружать: понятная ошибка и код 2.
``--dry-run`` работает без torch, проверяет наличие инструментов и печатает
план, код 0.

Ограничения
===========

* ``--prompt`` — сырой текст без chat template. Для Instruct-модели можно
  передать уже отрендеренный шаблон (``<|im_start|>…``) строкой.
* ``rope_scaling``, ``partial_rotary_factor != 1``, sliding window меньше
  длины прогона и не-SiLU активации VTM не поддерживает — скрипт падает с
  объяснением, а не меряет заведомо неверную модель.
* ``RMSN`` кодирует eps как ``imm/1000``: Qwen ``1e-6`` честно округляется
  до 0. Вклад мал, но это известное ограничение ``tools/vtm_compile.py``;
  скрипт пишет предупреждение.
* Скрипт сам не ставит пакеты и не делает HTTP-запросов; transformers берёт
  модель из локального пути/кэша, а по hub id может скачать её на GPU-машине.

Запуск
======

    python3 tools/parity_real.py --dry-run
    python3 tools/parity_real.py --model Qwen/Qwen2.5-1.5B-Instruct
    python3 tools/parity_real.py --prompt "..." --limit-tokens 16 \\
        --quantize-embeddings none --workdir data/parity_real
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_WORKDIR = ROOT / "data" / "parity_real"
DEFAULT_LIMIT_TOKENS = 16
DEFAULT_THRESHOLD = 1e-2
DEFAULT_SCALE_DTYPE = "float16"

EMBED_MODES = ("none", "trit", "int8")
SCALE_DTYPES = ("float16", "float32")
TORCH_DTYPES = ("float32", "bfloat16", "float16")
DEVICES = ("auto", "cpu", "cuda", "mps")

VTM_DIR = ROOT / "src" / "vtm"
VTM_BIN = VTM_DIR / "vtm"
VTM_LIB = VTM_DIR / "libvtm.a"
VTM_INCLUDE = VTM_DIR
VTM_COMPILER = ROOT / "tools" / "vtm_compile.py"

_HUB_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_EMBED_INT8_DEFERRED = (
    "int8-эмбеддинги в контейнер отложены: нужен kind 4 (int8 + масштабы) "
    "и его чтение в src/vtm/loader.c и tools/vtm_ref.py. Пока используйте "
    "none или trit"
)


class ParityError(RuntimeError):
    """Ошибка подготовки/конфигурации: код возврата 2."""


# ---------------------------------------------------------------------------
# Окружение и dry-run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Environment:
    """Что нашлось на машине. Только проверки, без импорта torch."""

    torch: bool
    transformers: bool
    numpy: bool
    compiler_script: bool
    export_module: bool
    vtm_bin: bool
    vtm_lib: bool
    cc: str | None
    model_state: str  # local | hub | missing
    c_include: bool

    def missing_required(self) -> list[str]:
        missing: list[str] = []
        if not self.torch:
            missing.append("torch")
        if not self.transformers:
            missing.append("transformers")
        if not self.numpy:
            missing.append("numpy")
        if not self.compiler_script:
            missing.append("tools/vtm_compile.py")
        if not self.export_module:
            missing.append("qat.export")
        if not self.vtm_bin:
            missing.append("src/vtm/vtm")
        if not self.vtm_lib:
            missing.append("src/vtm/libvtm.a")
        if not self.c_include:
            missing.append("src/vtm/vtm.h")
        if not self.cc:
            missing.append("C-компилятор (cc/clang/gcc)")
        return missing

    @property
    def ready(self) -> bool:
        return not self.missing_required()


def module_available(name: str) -> bool:
    """Есть ли модуль, не импортируя его (dry-run не тянет torch)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def find_c_compiler() -> str | None:
    """Путь к C-компилятору: $CC, затем cc/clang/gcc."""
    candidates: list[str] = []
    env_cc = os.environ.get("CC")
    if env_cc:
        try:
            parts = shlex.split(env_cc)
        except ValueError:
            parts = []
        if parts:
            candidates.append(parts[0])
    candidates.extend(("cc", "clang", "gcc"))
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def local_model_state(model: str) -> str:
    """local — путь существует; hub — id вида org/name; missing — путь не найден."""
    path = Path(model).expanduser()
    if path.exists():
        return "local"
    if _HUB_ID_RE.match(model) and not path.is_absolute():
        return "hub"
    return "missing"


def check_environment(model: str) -> Environment:
    return Environment(
        torch=module_available("torch"),
        transformers=module_available("transformers"),
        numpy=module_available("numpy"),
        compiler_script=VTM_COMPILER.is_file(),
        export_module=(ROOT / "qat" / "export.py").is_file()
        and module_available("qat.export"),
        vtm_bin=VTM_BIN.is_file() and os.access(VTM_BIN, os.X_OK),
        vtm_lib=VTM_LIB.is_file(),
        cc=find_c_compiler(),
        model_state=local_model_state(model),
        c_include=(VTM_INCLUDE / "vtm.h").is_file(),
    )


def _mark(ok: bool, required: bool = True) -> str:
    if ok:
        return "[ok]  "
    return "[нет] " if required else "[warn]"


def dry_run(args: argparse.Namespace) -> int:
    """План без torch: какие инструменты есть и что будет запущено."""
    env = check_environment(args.model)
    workdir = Path(args.workdir).expanduser().resolve()
    context_len = max(8, int(args.limit_tokens))
    run_dir = workdir / "qat_run"
    export_dir = workdir / "export"

    print("=== parity_real: dry-run ===")
    print("torch и веса не загружаются; проверяются только инструменты и план.")
    print()
    print(f"репозиторий: {ROOT}")
    print(f"модель:      {args.model}")
    if env.model_state == "local":
        print("             локальный путь найден")
    elif env.model_state == "hub":
        print("             hub id: на GPU-машине transformers возьмёт кэш/хаб")
    else:
        print("             ВНИМАНИЕ: путь не найден")
    print(f"промпт:      {args.prompt!r}")
    print(f"workdir:     {workdir}")
    print(f"лимит:       {args.limit_tokens} токенов (обрезать промпт)")
    print(f"эмбеддинги:  {args.quantize_embeddings}")
    print(f"масштабы:    {args.scale_dtype}")
    print(f"устройство:  {args.device}, dtype {args.torch_dtype}")
    print()
    print("проверки:")
    print(f"  {_mark(env.torch, required=False)} torch")
    print(f"  {_mark(env.transformers, required=False)} transformers")
    print(f"  {_mark(env.numpy, required=False)} numpy (нужен qat.export)")
    print(f"  {_mark(env.compiler_script)} tools/vtm_compile.py")
    print(f"  {_mark(env.export_module)} qat.export")
    print(f"  {_mark(env.vtm_bin)} src/vtm/vtm (исполняемый)")
    print(f"  {_mark(env.vtm_lib)} src/vtm/libvtm.a")
    print(f"  {_mark(env.c_include)} src/vtm/vtm.h")
    print(f"  {_mark(bool(env.cc))} C-компилятор: {env.cc or 'не найден'}")
    print()
    if env.model_state == "missing":
        print(f"ВНИМАНИЕ: модели {args.model!r} нет; на GPU-машине укажите "
              f"локальный путь или hub id.")
    missing = env.missing_required()
    if missing:
        print("для полного прогона не хватает: " + ", ".join(missing))
        if not env.vtm_bin or not env.vtm_lib:
            print("  C-рантайм/библиотека не собраны: make -C src/vtm")
        if env.model_state == "missing":
            print(f"  нет модели: {args.model}")
    else:
        print("всё для полного прогона на месте (torch/transformers проверяются "
              "по наличию, не импортом).")
    print()
    print("план:")
    print("  1. HF: токенайзер Qwen -> промпт -> ids (<= лимита)")
    print("  2. HF forward (fp32): полные логиты — справочно")
    print("  3. apply_ternary -> HF forward: логиты той же тернарной модели — "
          "эталон для порога")
    print("  4. QAT-чекпоинт в workdir (state_dict + trainer_state + run.json)")
    print("  5. qat.export -> ternary.trit + non_ternary.bin + manifest.json")
    print("  6. tools/vtm_compile.py -> model.vtm")
    print("  7. src/vtm/vtm run ... --tokens ... --max-new 1 -> argmax")
    print("  8. C-проба на libvtm.a -> вектор logits; сравнение max|Δ|/argmax/top-5")
    print("  9. parity.json + сводка; порог: argmax совпал и max|Δ| < 1e-2")
    print()
    print("команды, которые будут выполнены:")
    print("  " + shlex.join(build_export_command(
        sys.executable, run_dir, export_dir,
        context_len=context_len, scale_dtype=args.scale_dtype)))
    print("  " + shlex.join(build_compile_command(
        sys.executable, export_dir / "manifest.json", workdir / "model.vtm")))
    print("  " + shlex.join(build_run_command(
        VTM_BIN, workdir / "model.vtm", [1, 2, 3])))
    cc = env.cc or "cc"
    print("  " + shlex.join(build_probe_compile_command(cc, workdir)))
    print("  " + shlex.join([
        str(workdir / "parity_probe"), str(workdir / "model.vtm"), "<ids>",
    ]))
    print()
    print(f"JSON: {workdir / 'parity.json'}")
    if args.quantize_embeddings == "int8":
        print()
        print("int8: " + _EMBED_INT8_DEFERRED + " (полный прогон завершится кодом 2)")
    return 0


# ---------------------------------------------------------------------------
# CLI и команды
# ---------------------------------------------------------------------------


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось целое > 0: {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"ожидалось целое > 0: {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 tools/parity_real.py",
        description=(
            "Сверка Qwen2.5-1.5B-Instruct: transformers против .vtm в C-рантайме."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  python3 tools/parity_real.py --dry-run\n"
            "  python3 tools/parity_real.py --limit-tokens 16\n"
            "  python3 tools/parity_real.py --model /models/Qwen2.5-1.5B-Instruct "
            "--prompt 'Hello' --workdir /tmp/parity\n\n"
            "Порог: argmax совпал, max|Δ| логитов < 1e-2 на тех же тернарных "
            "весах (полная точность HF — справочно).\n"
        ),
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Qwen/Qwen2.5-1.5B-Instruct или локальный путь (по умолчанию {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help=f"фиксированный промпт (по умолчанию {DEFAULT_PROMPT!r})",
    )
    parser.add_argument(
        "--workdir", default=str(DEFAULT_WORKDIR),
        help=f"каталог артефактов (по умолчанию {DEFAULT_WORKDIR})",
    )

    parser.add_argument(
        "--limit-tokens", type=positive_int, default=DEFAULT_LIMIT_TOKENS,
        help=f"обрезать промпт до N токенов (по умолчанию {DEFAULT_LIMIT_TOKENS})",
    )
    parser.add_argument(
        "--quantize-embeddings", choices=EMBED_MODES, default="none",
        help="режим эмбеддингов: none (fp32), trit; int8 отложен",
    )
    parser.add_argument(
        "--scale-dtype", choices=SCALE_DTYPES, default=DEFAULT_SCALE_DTYPE,
        help=f"тип масштабов тритов в .trit (по умолчанию {DEFAULT_SCALE_DTYPE})",
    )
    parser.add_argument(
        "--torch-dtype", choices=TORCH_DTYPES, default="float32",
        help="dtype HF-модели и мастер-весов (по умолчанию float32)",
    )
    parser.add_argument(
        "--device", choices=DEVICES, default="auto",
        help="устройство HF (auto: cuda -> mps -> cpu)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="без torch: проверить инструменты и напечатать план, код 0",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def token_ids_arg(ids: Sequence[int]) -> str:
    if not ids:
        raise ParityError("пустой список токенов")
    return ",".join(str(int(token)) for token in ids)


def build_export_command(
    python: str,
    run_dir: Path,
    out_dir: Path,
    *,
    context_len: int,
    scale_dtype: str = DEFAULT_SCALE_DTYPE,
) -> list[str]:
    """Команда qat.export.

    ``--quantize-embeddings`` у qat.export нет: режим читается из
    ``quant.quantize_embeddings`` в run.json/config. ``--step 0`` фиксирует
    наш единственный чекпоинт, ``--context-len`` не даёт раздуть KV-кэш.
    """
    return [
        python,
        "-m",
        "qat.export",
        "--run",
        str(run_dir),
        "--out",
        str(out_dir),
        "--step",
        "0",
        "--force",
        "--context-len",
        str(int(context_len)),
        "--scale-dtype",
        str(scale_dtype),
        "--tie-embeddings",
        "auto",
    ]


def build_compile_command(
    python: str, manifest: Path, out_path: Path
) -> list[str]:
    return [
        python,
        str(VTM_COMPILER),
        "--manifest",
        str(manifest),
        "--out",
        str(out_path),
    ]


def build_run_command(
    vtm_bin: Path, vtm_path: Path, token_ids: Sequence[int]
) -> list[str]:
    return [
        str(vtm_bin),
        "run",
        str(vtm_path),
        "--tokens",
        token_ids_arg(token_ids),
        "--max-new",
        "1",
    ]


def build_probe_compile_command(
    cc: str,
    workdir: Path,
    *,
    include_dir: Path = VTM_INCLUDE,
    lib: Path = VTM_LIB,
) -> list[str]:
    return [
        cc,
        "-std=c99",
        "-O2",
        "-Wall",
        "-Wextra",
        "-I",
        str(include_dir),
        "-o",
        str(workdir / "parity_probe"),
        str(workdir / "parity_probe.c"),
        str(lib),
        "-lm",
    ]


# ---------------------------------------------------------------------------
# Разбор вывода
# ---------------------------------------------------------------------------


def _parse_ids(text: str) -> list[int]:
    text = text.strip()
    if not text or text.startswith("("):
        return []
    try:
        return [int(part) for part in text.split()]
    except ValueError as exc:
        raise ParityError(f"не разобрал список токенов: {text!r}") from exc


def parse_vtm_run_output(stdout: str) -> dict[str, Any]:
    """Разобрать ``vtm run``: prompt, вывод, счётчики."""
    result: dict[str, Any] = {
        "prompt": [],
        "generated": [],
        "tensor_count": None,
        "instr_count": None,
        "backend": None,
    }
    for line in stdout.splitlines():
        if line.startswith("prompt:"):
            result["prompt"] = _parse_ids(line.split(":", 1)[1])
        elif line.startswith("вывод:"):
            result["generated"] = _parse_ids(line.split(":", 1)[1])
        elif line.startswith("тензоров:"):
            match = re.search(
                r"тензоров:\s*(\d+),\s*инструкций:\s*(\d+),\s*бэкенд:\s*(\S+)",
                line,
            )
            if match:
                result["tensor_count"] = int(match.group(1))
                result["instr_count"] = int(match.group(2))
                result["backend"] = match.group(3)
    return result


@dataclass
class ProbeResult:
    logits: list[float]
    meta: dict[str, Any]


def parse_probe_output(stdout: str) -> ProbeResult:
    """Разобрать вывод C-пробы: ``# key value`` и по числу на строку."""
    meta: dict[str, Any] = {}
    logits: list[float] = []
    count: int | None = None
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line[1:].strip().split(None, 1)
            if not parts:
                continue
            key = parts[0]
            value = parts[1].strip() if len(parts) > 1 else ""
            if key == "count":
                try:
                    count = int(value)
                except ValueError as exc:
                    raise ParityError(f"проба: плохой # count: {value!r}") from exc
                continue
            if not value:
                meta[key] = value
                continue
            try:
                meta[key] = float(value)
            except ValueError:
                meta[key] = value
            continue
        try:
            logits.append(float(line))
        except ValueError as exc:
            raise ParityError(f"проба: не число в строке {line!r}") from exc
    if count is None:
        raise ParityError("проба: нет строки '# count N'")
    if len(logits) != count:
        raise ParityError(f"проба: count={count}, а чисел {len(logits)}")
    if not logits:
        raise ParityError("проба: пустые логиты")
    return ProbeResult(logits=logits, meta=meta)


def tail(text: str, lines: int = 30) -> str:
    chunk = [line for line in (text or "").strip().splitlines() if line.strip()]
    return "\n".join(chunk[-lines:])


# ---------------------------------------------------------------------------
# Сравнение
# ---------------------------------------------------------------------------


def argmax(values: Sequence[float]) -> int:
    """Первый максимум, как в C и vtm_ref (NaN не побеждает)."""
    if not values:
        raise ValueError("argmax: пустой список")
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def top_indices(values: Sequence[float], k: int = 5) -> list[int]:
    """Top-k индексов: по убыванию значения, при равенстве — по индексу."""
    if not values:
        return []
    take = min(int(k), len(values))
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[:take]


def compare_logits(
    reference: Sequence[float],
    runtime: Sequence[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    if len(reference) != len(runtime):
        raise ParityError(
            f"размеры логитов не совпали: HF {len(reference)}, VTM {len(runtime)}"
        )
    if not reference:
        raise ParityError("пустые логиты")
    max_delta = max(abs(float(a) - float(b)) for a, b in zip(reference, runtime))
    ref_argmax = argmax(reference)
    run_argmax = argmax(runtime)
    ref_top = top_indices(reference, 5)
    run_top = top_indices(runtime, 5)
    overlap = len(set(ref_top) & set(run_top))
    within = max_delta < float(threshold)
    return {
        "max_abs_delta": max_delta,
        "argmax_match": ref_argmax == run_argmax,
        "argmax_reference": ref_argmax,
        "argmax_runtime": run_argmax,
        "top5_reference": ref_top,
        "top5_runtime": run_top,
        "top5_overlap": overlap,
        "top5_size": min(5, len(reference)),
        "threshold": float(threshold),
        "within_threshold": within,
        "passed": bool(ref_argmax == run_argmax and within),
    }


def summarize_logits(values: Sequence[float]) -> dict[str, Any]:
    return {
        "argmax": argmax(values),
        "top5": top_indices(values, 5),
        "count": len(values),
        "first8": [float(value) for value in values[:8]],
        "sum": math.fsum(float(value) for value in values),
    }


def check_runtime_meta(
    meta: dict[str, Any],
    architecture: dict[str, Any],
    logits_count: int,
) -> list[str]:
    """Проверить, что C прочитал meta-JSON, а не взял дефолты."""
    problems: list[str] = []

    def number(key: str) -> float | None:
        value = meta.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    present = number("meta_present")
    if present is None or int(present) != 1:
        problems.append("C не видит meta-JSON контейнера")

    theta = number("meta_rope_theta")
    expected_theta = float(architecture.get("rope_theta", 10000.0))
    if theta is None or not math.isclose(
        theta, expected_theta, rel_tol=1e-9, abs_tol=1e-12
    ):
        problems.append(
            f"rope_theta: C={theta if theta is not None else '?'}, "
            f"HF={expected_theta:g}"
        )

    layout = number("meta_rope_layout")
    expected_layout = 1 if architecture.get("rope_layout") == "interleaved" else 0
    if layout is None or int(layout) != expected_layout:
        problems.append(
            f"rope_layout: C={layout if layout is not None else '?'}, "
            f"ожидался {expected_layout}"
        )

    vocab = number("meta_vocab")
    if vocab is None or int(vocab) != int(logits_count):
        problems.append(
            f"vocab: C={vocab if vocab is not None else '?'}, "
            f"логитов {logits_count}"
        )

    tied = number("meta_tied")
    expected_tied = bool(architecture.get("tie_word_embeddings"))
    if tied is None or bool(int(tied)) != expected_tied:
        problems.append(
            f"tied_embeddings: C={tied if tied is not None else '?'}, "
            f"ожидалось {expected_tied}"
        )
    return problems


# ---------------------------------------------------------------------------
# C-проба: тот же libvtm.a, вектор логитов наружу
# ---------------------------------------------------------------------------


def probe_source() -> str:
    """Исходник крошечной C-пробы (пишется в workdir, не в репозиторий)."""
    return r'''/* parity_probe.c — снять logits последней позиции из .vtm.
 *
 * Собирается tools/parity_real.py и линкуется с src/vtm/libvtm.a.
 * Печатает "# key value" (meta) и по одному float на строку. */
#include "vtm.h"

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>

static int parse_tokens(const char *text, int **out, size_t *out_n) {
    int *tokens = NULL;
    size_t n = 0, cap = 0;
    const char *p = text;

    while (*p != '\0') {
        char *end = NULL;
        long value;
        while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n') p++;
        if (*p == '\0') break;
        value = strtol(p, &end, 10);
        if (end == p || value < 0 || value > 2147483647L) {
            free(tokens);
            return -1;
        }
        if (n == cap) {
            size_t next = cap == 0 ? 16 : cap * 2;
            int *grown = (int *)realloc(tokens, next * sizeof(*tokens));
            if (grown == NULL) {
                free(tokens);
                return -1;
            }
            tokens = grown;
            cap = next;
        }
        tokens[n++] = (int)value;
        p = end;
    }
    *out = tokens;
    *out_n = n;
    return 0;
}

int main(int argc, char **argv) {
    vtm_t *machine;
    const vtm_tensor *tensor;
    int *tokens = NULL;
    size_t n_tokens = 0;
    uint32_t i;
    vtm_status status;

    /* Десятичная точка, а не запятая: иначе парсер Python не прочитает
     * логиты в русской локали. */
    setlocale(LC_NUMERIC, "C");

    if (argc < 3) {
        fprintf(stderr, "parity_probe: нужны model.vtm и список tokens\n");
        return 2;
    }
    if (parse_tokens(argv[2], &tokens, &n_tokens) != 0 || n_tokens == 0) {
        fprintf(stderr, "parity_probe: плохой список токенов\n");
        return 2;
    }

    machine = vtm_load(argv[1]);
    if (machine == NULL) {
        fprintf(stderr, "parity_probe: vtm_load: %s\n", vtm_last_error());
        free(tokens);
        return 1;
    }

    printf("# meta_present %d\n", vtm_meta_present(machine));
    printf("# meta_rope_theta %.17g\n", vtm_meta_rope_theta(machine));
    printf("# meta_rope_layout %d\n", vtm_meta_rope_layout(machine));
    printf("# meta_vocab %u\n", (unsigned)vtm_meta_vocab_size(machine));
    printf("# meta_tied %d\n", vtm_meta_tied_embeddings(machine));

    for (i = 0; i < (uint32_t)n_tokens; i++) {
        vtm_set_position(machine, (int)i);
        vtm_set_token(machine, tokens[i]);
        status = vtm_run(machine, NULL, NULL);
        if (status != VTM_OK) {
            fprintf(stderr, "parity_probe: прогон %u: %s: %s\n",
                    i, vtm_strerror(status), vtm_error_message(machine));
            free(tokens);
            vtm_free(machine);
            return 1;
        }
    }

    tensor = vtm_tensor_by_name(machine, "logits");
    if (tensor == NULL) {
        fprintf(stderr, "parity_probe: в контейнере нет тензора logits\n");
        free(tokens);
        vtm_free(machine);
        return 1;
    }
    if (tensor->kind != VTM_TENSOR_F32 || tensor->f32 == NULL) {
        fprintf(stderr, "parity_probe: logits не fp32 (kind=%u)\n",
                (unsigned)tensor->kind);
        free(tokens);
        vtm_free(machine);
        return 1;
    }

    printf("# count %u\n", (unsigned)tensor->n_elements);
    for (i = 0; i < tensor->n_elements; i++) {
        printf("%.9g\n", tensor->f32[i]);
    }

    free(tokens);
    vtm_free(machine);
    return 0;
}
'''


def run_c_probe(
    workdir: Path,
    vtm_path: Path,
    token_ids: Sequence[int],
    cc: str,
) -> ProbeResult:
    source_path = workdir / "parity_probe.c"
    try:
        source_path.write_text(probe_source(), encoding="utf-8")
    except OSError as exc:
        raise ParityError(f"не записался {source_path}: {exc}") from exc
    binary = workdir / "parity_probe"
    compile_cmd = build_probe_compile_command(cc, workdir)
    proc = run_command(compile_cmd)
    if proc.returncode != 0:
        raise ParityError(
            f"C-проба не собралась (код {proc.returncode}):\n{tail(proc.stderr)}"
        )
    run_cmd = [str(binary), str(vtm_path), token_ids_arg(token_ids)]
    proc = run_command(run_cmd)
    if proc.returncode != 0:
        raise ParityError(
            f"C-проба не отработала (код {proc.returncode}):\n{tail(proc.stderr)}"
        )
    return parse_probe_output(proc.stdout)


# ---------------------------------------------------------------------------
# HF: загрузка, forward, тернаризация, сохранение QAT-чекпоинта
# ---------------------------------------------------------------------------


def resolve_device(torch: Any, name: str) -> Any:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ParityError("--device cuda, но CUDA недоступна")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ParityError("--device mps, но MPS недоступна")
    return device


def load_hf_model(model_name: str, dtype_name: str) -> Any:
    """Загрузить CausalLM; совместимость transformers 4.x/5.x."""
    import torch  # noqa: PLC0415 - только в не-dry-run
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    dtype = getattr(torch, dtype_name)
    attempts = (
        {"torch_dtype": dtype, "attn_implementation": "eager"},
        {"dtype": dtype, "attn_implementation": "eager"},
        {"dtype": dtype},
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=False, **kwargs
            )
        except (TypeError, ImportError) as exc:
            last_error = exc
            continue
    raise ParityError(
        f"не удалось загрузить HF-модель {model_name!r}: {last_error}"
    )


def tokenize_prompt(tokenizer: Any, prompt: str, limit: int) -> tuple[list[int], bool]:
    encoded = tokenizer(prompt, add_special_tokens=False)
    ids = [int(token) for token in encoded["input_ids"]]
    truncated = len(ids) > int(limit)
    return ids[: int(limit)], truncated


def hf_logits(torch: Any, model: Any, token_ids: Sequence[int], device: Any) -> list[float]:
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    return [float(value) for value in logits[0, -1].detach().float().cpu().tolist()]


def hf_tie_broken(model: Any) -> bool:
    """config.tie_word_embeddings=true, но lm_head и embed — разные тензоры.

    qat.export при tied пишет lm_head как ссылку на таблицу эмбеддингов.
    Если HF на самом деле считает отдельный lm_head, C и HF разойдутся не
    из-за рантайма, а из-за разных весов; лучше остановиться сразу.
    """
    try:
        if not bool(getattr(model.config, "tie_word_embeddings", False)):
            return False
        embed = model.get_input_embeddings()
        head = model.get_output_embeddings()
        if embed is None or head is None:
            return False
        return int(embed.weight.data_ptr()) != int(head.weight.data_ptr())
    except Exception:  # noqa: BLE001 - не наша забота, если API другой
        return False


def check_model_supported(config: Any, context_len: int | None = None) -> None:
    """VTM умеет Qwen/Llama-декодер; всё остальное — явная ошибка.

    Sliding window допускается, только если окно не меньше длины прогона:
    тогда маска ни на что не влияет. Иначе VTM посчитает полное внимание и
    молча разойдётся с HF.
    """
    problems: list[str] = []

    def cfg(name: str, default: Any = None) -> Any:
        if isinstance(config, dict):
            return config.get(name, default)
        return getattr(config, name, default)

    hidden_act = str(cfg("hidden_act", "silu")).lower()
    if hidden_act not in ("silu", "swish"):
        problems.append(f"hidden_act={hidden_act!r}, VTM умеет только silu")

    rope_scaling = cfg("rope_scaling")
    if rope_scaling:
        problems.append(f"rope_scaling={rope_scaling!r} не поддержан")

    partial = cfg("partial_rotary_factor", 1.0)
    try:
        partial_value = float(partial) if partial is not None else 1.0
    except (TypeError, ValueError):
        partial_value = 0.0
    if not math.isclose(partial_value, 1.0, rel_tol=0.0, abs_tol=1e-12):
        problems.append(
            f"partial_rotary_factor={partial_value:g}: VTM вращает всю голову"
        )

    if bool(cfg("use_sliding_window", False)):
        window = cfg("sliding_window", None)
        try:
            window_value = int(window) if window is not None else None
        except (TypeError, ValueError):
            window_value = None
        if (
            window_value is None
            or context_len is None
            or window_value < int(context_len)
        ):
            problems.append(
                "use_sliding_window=true и окно "
                f"{window_value if window_value is not None else '?'} < "
                f"длины прогона {context_len if context_len is not None else '?'}: "
                "VTM не умеет sliding window"
            )

    heads = cfg("num_attention_heads")
    kv_heads = cfg("num_key_value_heads", heads)
    try:
        heads = int(heads)
        kv_heads = int(kv_heads) if kv_heads is not None else heads
    except (TypeError, ValueError):
        problems.append("не удалось прочитать num_attention_heads/num_key_value_heads")
    else:
        if heads <= 0 or kv_heads <= 0:
            problems.append(f"плохие heads={heads}, kv_heads={kv_heads}")
        elif heads % kv_heads:
            problems.append(f"n_heads={heads} не делится на n_kv_heads={kv_heads}")

    head_dim = cfg("head_dim")
    if head_dim is not None:
        try:
            if int(head_dim) % 2:
                problems.append(f"head_dim={head_dim} нечётный: RoPE требует чётный")
        except (TypeError, ValueError):
            problems.append(f"плохой head_dim={head_dim!r}")

    if problems:
        raise ParityError(
            "VTM не поддерживает эту модель: " + "; ".join(problems)
        )


def build_architecture(config: Any, context_len: int) -> dict[str, Any]:
    """Архитектура в терминах manifest/ModelConfig (не HF)."""

    def cfg(name: str, default: Any = None) -> Any:
        if isinstance(config, dict):
            return config.get(name, default)
        return getattr(config, name, default)

    hidden = int(cfg("hidden_size"))
    heads = int(cfg("num_attention_heads"))
    kv_heads = cfg("num_key_value_heads", heads)
    kv_heads = int(kv_heads) if kv_heads is not None else heads
    head_dim = cfg("head_dim")
    head_dim = int(head_dim) if head_dim is not None else hidden // heads
    max_context = cfg("max_position_embeddings", context_len)
    return {
        "hidden": hidden,
        "n_heads": heads,
        "n_kv_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate": int(cfg("intermediate_size")),
        "vocab": int(cfg("vocab_size")),
        "n_layers": int(cfg("num_hidden_layers")),
        "context_len": int(context_len),
        "max_context": int(max_context) if max_context else int(context_len),
        "rope_theta": float(cfg("rope_theta", 10000.0) or 10000.0),
        "rms_norm_eps": float(cfg("rms_norm_eps", 1e-6)),
        "tie_word_embeddings": bool(cfg("tie_word_embeddings", False)),
        "rope_layout": "half",
        "hidden_act": str(cfg("hidden_act", "silu")),
        "source": "transformers",
    }


def check_manifest_architecture(
    manifest_arch: dict[str, Any],
    architecture: dict[str, Any],
    *,
    context_len: int,
    embed_mode: str,
) -> list[str]:
    """Сверить architecture манифеста с тем, что мы считали в HF.

    qat.export собирает её из state_dict и run.json; если он понял конфиг
    иначе, .vtm будет не той моделью, и сравнивать логиты бессмысленно.
    """
    expected = {
        "hidden": architecture["hidden"],
        "n_heads": architecture["n_heads"],
        "n_kv_heads": architecture["n_kv_heads"],
        "head_dim": architecture["head_dim"],
        "intermediate": architecture["intermediate"],
        "vocab": architecture["vocab"],
        "n_layers": architecture["n_layers"],
        "context_len": int(context_len),
        "tie_word_embeddings": architecture["tie_word_embeddings"],
        "embed": embed_mode,
    }
    mismatches = [
        f"{key}: манифест {manifest_arch.get(key)!r}, ожидалось {value!r}"
        for key, value in expected.items()
        if manifest_arch.get(key) != value
    ]
    for key, value in (
        ("rope_theta", architecture["rope_theta"]),
        ("rms_norm_eps", architecture["rms_norm_eps"]),
    ):
        got = manifest_arch.get(key)
        try:
            ok = got is not None and math.isclose(
                float(got), float(value), rel_tol=1e-9, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            ok = False
        if not ok:
            mismatches.append(f"{key}: манифест {got!r}, ожидалось {value!r}")
    return mismatches


def hf_config_dict(config: Any) -> dict[str, Any]:
    """HF-конфиг для run.json: qat.export читает из него n_heads/rope/eps."""
    if hasattr(config, "to_dict"):
        data = config.to_dict()
    elif isinstance(config, dict):
        data = dict(config)
    elif hasattr(config, "__dict__"):
        data = {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_")
        }
    else:  # pragma: no cover - защита от неожиданного типа
        raise ParityError(f"не умею сериализовать HF-конфиг {type(config).__name__}")
    if not isinstance(data, dict):
        raise ParityError("HF-конфиг не словарь")
    return {str(key): _jsonable(value) for key, value in data.items()}


def apply_hf_quantization(model: Any, mode: str) -> tuple[Any, list[dict[str, Any]], Any]:
    """Тернаризовать HF-модель тем же кодом, что QAT-обучение/экспорт."""
    from qat.config import QuantConfig  # noqa: PLC0415
    from qat.model import apply_embedding_quant, apply_ternary  # noqa: PLC0415

    quant = QuantConfig(quantize_embeddings=mode)
    report = apply_ternary(model, quant)
    embed_report = None
    if mode != "none":
        embed_report = apply_embedding_quant(model, quant)
    return quant, report, embed_report


def retie_quantized_lm_head(torch: Any, model: Any) -> bool:
    """tied lm_head должен считать по квантованной таблице, как C.

    ``apply_embedding_quant`` оборачивает ``embed_tokens.weight``
    параметризацией; ``lm_head.weight`` остаётся старым Parameter, то есть
    полной точностью. Тогда HF-эталон и .vtm разошлись бы по lm_head, а не
    по рантайму. Подменяем вес lm_head текущим (уже квантованным) весом
    эмбеддингов. qat.export при tie всё равно игнорирует lm_head и пишет
    ссылку на таблицу.
    """
    head = model.get_output_embeddings()
    embed = model.get_input_embeddings()
    if head is None or embed is None or not hasattr(head, "weight"):
        return False
    # detach без clone: таблица уже посчитана параметризацией, копия на
    # 151936×1536 fp32 (≈0,9 ГиБ) здесь лишняя.
    head.weight = torch.nn.Parameter(embed.weight.detach())
    return True


def build_run_config(args: argparse.Namespace, quant: Any) -> dict[str, Any]:
    """Секция config для run.json/trainer_state: её читает qat.export."""
    return {
        "model": {
            "name_or_path": args.model,
            "revision": None,
            "trust_remote_code": False,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": "eager",
        },
        "quant": {
            "block_size": int(quant.block_size),
            "scale_dtype": args.scale_dtype,
            "target_suffixes": list(quant.target_suffixes),
            "exclude_suffixes": list(quant.exclude_suffixes),
            "quantize_embeddings": quant.quantize_embeddings,
        },
        "data": {
            "path": "",
            "max_seq_len": int(args.limit_tokens),
            "max_samples": 0,
            "min_answer_tokens": 1,
            "use_teacher_logits": False,
            "kl_temperature": 1.0,
            "kl_weight": 0.0,
            "kl_chunk_size": 64,
        },
        "train": {
            "max_steps": 0,
            "save_optimizer": False,
            "seed": 0,
            "gradient_checkpointing": False,
            "device": args.device,
        },
    }


def build_run_meta(
    *,
    model_name: str,
    config: dict[str, Any],
    hf_architecture: dict[str, Any],
    vocab: int,
    num_params: int,
    ternary_params: int,
    ternary_report: Sequence[dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    return {
        "format": "eva32.qat.run",
        "format_version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "config": config,
        "config_path": "tools/parity_real.py",
        "model": {
            "name_or_path": model_name,
            "revision": None,
            "vocab_size": int(vocab),
            "num_params": int(num_params),
            "ternary_params": int(ternary_params),
        },
        "quant": {
            "block_size": int(config["quant"]["block_size"]),
            "scale_dtype": config["quant"]["scale_dtype"],
            "target_suffixes": list(config["quant"]["target_suffixes"]),
            "modules": list(ternary_report),
            "modules_count": len(ternary_report),
        },
        "device": device,
        "status": "parity-export",
        # Именно HF-конфиг: qat.export.infer_architecture ждёт hidden_size,
        # num_attention_heads, rope_theta и т.п. Манифестную архитектуру он
        # соберёт сам.
        "architecture": hf_architecture,
    }


def save_qat_checkpoint(
    torch: Any,
    state: dict[str, Any],
    run_dir: Path,
    *,
    config: dict[str, Any],
    step: int = 0,
) -> Path:
    """Сохранить state_dict в структуру runs/<run>/checkpoints/step_000000."""
    ckpt = run_dir / "checkpoints" / f"step_{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model_tmp = ckpt / ".model.pt.tmp"
    torch.save(state, model_tmp)
    os.replace(model_tmp, ckpt / "model.pt")
    trainer_state = {
        "format_version": 1,
        "step": int(step),
        "samples_seen": 0,
        "max_steps": 0,
        "config": config,
        "created_at": utc_now(),
        "dry_run": False,
        "status": "parity-real",
    }
    state_tmp = ckpt / ".trainer_state.pt.tmp"
    torch.save(trainer_state, state_tmp)
    os.replace(state_tmp, ckpt / "trainer_state.pt")
    (run_dir / "latest.txt").write_text(f"step_{step:06d}\n", encoding="utf-8")
    return ckpt


# ---------------------------------------------------------------------------
# JSON/результат/сводка
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """Привести к JSON: Path, tuple, NaN/Inf и torch-подобные объекты."""
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(data), handle, ensure_ascii=False, indent=2,
                  allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def make_result(
    *,
    model: str,
    prompt: str,
    token_ids: Sequence[int],
    limit_tokens: int,
    quantize_embeddings: str,
    scale_dtype: str,
    device: str,
    torch_dtype: str,
    context_len: int,
    workdir: Path,
    artifacts: dict[str, Any],
    reference_quantized: dict[str, Any],
    reference_full: dict[str, Any] | None,
    runtime: dict[str, Any],
    comparison: dict[str, Any],
    full_precision: dict[str, Any] | None,
    quantization_effect: dict[str, Any] | None,
    warnings: Sequence[str],
    timings: dict[str, float],
    status: str,
    error: str | None = None,
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "format": "eva32.parity.real",
        "format_version": 1,
        "created_at": utc_now(),
        "model": model,
        "prompt": prompt,
        "token_ids": list(token_ids),
        "token_count": len(token_ids),
        "limit_tokens": int(limit_tokens),
        "context_len": int(context_len),
        "quantize_embeddings": quantize_embeddings,
        "scale_dtype": scale_dtype,
        "device": device,
        "torch_dtype": torch_dtype,
        "workdir": str(workdir),
        "artifacts": artifacts,
        "reference": {
            "quantized": reference_quantized,
            "full_precision": reference_full,
            "note": (
                "Порог считается по hf_quantized: forward с теми же "
                "тернарными весами, что экспортированы в .vtm. "
                "full_precision — справочно, это цена тернаризации."
            ),
        },
        "runtime": runtime,
        "comparison": comparison,
        "full_precision": full_precision,
        "quantization_effect": quantization_effect,
        "threshold": {
            "max_abs_delta": DEFAULT_THRESHOLD,
            "rule": "argmax должен совпасть; max|Δ| логитов < 1e-2",
            "reference": "hf_quantized",
        },
        "passed": status == "passed",
        "status": status,
        "error": error,
        "warnings": list(warnings),
        "timings": {key: round(float(value), 3) for key, value in timings.items()},
        "argv": list(argv or []),
    }


def _short(text: str, limit: int = 120) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _fmt_delta(value: Any) -> str:
    if value is None:
        return "не измерен"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _verdict(result: dict[str, Any]) -> str:
    status = result.get("status")
    if status == "passed":
        return "ПРОЙДЕНО"
    if status == "partial":
        return "ЧАСТИЧНО — порог по max|Δ| не проверен"
    return "ПРОВАЛЕНО"


def write_failure_json(
    path: Path,
    *,
    args: argparse.Namespace,
    token_ids: Sequence[int],
    workdir: Path,
    context_len: int,
    stage: str,
    error: str,
    returncode: int | None = None,
    stdout: str = "",
    stderr: str = "",
    warnings: Sequence[str] = (),
    timings: dict[str, float] | None = None,
) -> None:
    """Минимальный parity.json, если прогон умер до сравнения логитов."""
    write_json_atomic(path, {
        "format": "eva32.parity.real",
        "format_version": 1,
        "created_at": utc_now(),
        "model": args.model,
        "prompt": args.prompt,
        "token_ids": list(token_ids),
        "token_count": len(token_ids),
        "context_len": int(context_len),
        "quantize_embeddings": args.quantize_embeddings,
        "scale_dtype": args.scale_dtype,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "workdir": str(workdir),
        "stage": stage,
        "returncode": returncode,
        "error": error,
        "stdout_tail": tail(stdout),
        "stderr_tail": tail(stderr),
        "status": "failed",
        "passed": False,
        "warnings": list(warnings),
        "timings": {
            key: round(float(value), 3) for key, value in (timings or {}).items()
        },
    })


def format_summary(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=== parity_real: Qwen HF ↔ VTM C ===")
    lines.append(f"модель:  {result['model']}")
    lines.append(f"промпт:  {_short(result['prompt'])}")
    lines.append(
        f"токены:  {result['token_count']} (лимит {result['limit_tokens']}): "
        f"{result['token_ids']}"
    )
    lines.append(
        f"экспорт: эмбеддинги={result['quantize_embeddings']}, "
        f"масштабы={result['scale_dtype']}, context={result['context_len']}"
    )
    lines.append(f"расчёт:  {result['device']}, {result['torch_dtype']}")
    lines.append("")

    reference = result["reference"]["quantized"]
    runtime = result["runtime"]
    comparison = result["comparison"]
    lines.append("эталон: HF с теми же тернарными весами, что в .vtm")
    lines.append(
        f"  argmax: HF {reference.get('argmax')} / VTM {runtime.get('argmax')} — "
        f"{'совпал' if comparison.get('argmax_match') else 'НЕ совпал'}"
    )
    lines.append(
        f"  max|Δ| логитов: {_fmt_delta(comparison.get('max_abs_delta'))} "
        f"(порог {result['threshold']['max_abs_delta']:.0e})"
    )
    if comparison.get("top5_overlap") is not None:
        lines.append(
            f"  top-5: {comparison['top5_overlap']}/"
            f"{comparison.get('top5_size', 5)} общих"
        )
    if comparison.get("method") == "cli-only":
        lines.append("  C-логиты: не измерены (C-проба недоступна)")
    if comparison.get("meta_ok") is False:
        lines.append("  meta: C не подтвердил rope_theta/vocab/tied — см. warnings")
    if comparison.get("cli_argmax_match") is False:
        lines.append("  CLI: сгенерированный токен не совпал с argmax пробы")
    lines.append(f"  ИТОГ: {_verdict(result)}")

    full_precision = result.get("full_precision")
    if full_precision:
        lines.append("")
        lines.append("справочно (полная точность HF, не порог):")
        lines.append(
            f"  max|Δ| {_fmt_delta(full_precision.get('max_abs_delta'))}; "
            f"argmax {'совпал' if full_precision.get('argmax_match') else 'НЕ совпал'}; "
            f"top-5 {full_precision.get('top5_overlap')}/"
            f"{full_precision.get('top5_size', 5)}"
        )
    quantization_effect = result.get("quantization_effect")
    if quantization_effect:
        lines.append(
            f"цена тернаризации (HF fp32 -> HF тернарные): "
            f"max|Δ| {_fmt_delta(quantization_effect.get('max_abs_delta'))}; "
            f"argmax {'совпал' if quantization_effect.get('argmax_match') else 'НЕ совпал'}"
        )

    warnings = result.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("предупреждения:")
        for warning in warnings:
            lines.append(f"  - {warning}")

    lines.append("")
    lines.append(f"JSON: {result['artifacts'].get('json')}")
    lines.append(f".vtm: {result['artifacts'].get('vtm')}")
    if result.get("timings"):
        timings = ", ".join(
            f"{key}={value:.1f}s" for key, value in result["timings"].items()
        )
        lines.append(f"время: {timings}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path = ROOT,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    print("[parity_real] $ " + shlex.join([str(part) for part in cmd]), flush=True)
    try:
        return subprocess.run(
            [str(part) for part in cmd],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise ParityError(f"команда не найдена: {cmd[0]}") from exc


def _print_command_failure(what: str, proc: subprocess.CompletedProcess[str]) -> None:
    print(
        f"[parity_real] {what}: код {proc.returncode}",
        file=sys.stderr,
    )
    if proc.stdout.strip():
        print("--- stdout ---", file=sys.stderr)
        print(tail(proc.stdout), file=sys.stderr)
    if proc.stderr.strip():
        print("--- stderr ---", file=sys.stderr)
        print(tail(proc.stderr), file=sys.stderr)


def _missing_torch_message() -> str:
    return (
        "[parity_real] нет torch: сверка с настоящими весами невозможна.\n"
        "Запустите на GPU-машине, где есть torch, transformers и numpy\n"
        "(см. qat/requirements-qat.txt). Без них работает только план:\n"
        "    python3 tools/parity_real.py --dry-run"
    )


def _missing_transformers_message() -> str:
    return (
        "[parity_real] нет transformers: нужен HF-эталон (AutoModelForCausalLM\n"
        "и AutoTokenizer). Установите GPU-стек на машине прогона\n"
        "(см. qat/requirements-qat.txt)."
    )


def _missing_c_runtime_message(env: Environment) -> str:
    lines = ["[parity_real] C-рантайм не собран:"]
    if not env.vtm_bin:
        lines.append("  нет src/vtm/vtm (исполняемый файл)")
    if not env.vtm_lib:
        lines.append("  нет src/vtm/libvtm.a")
    if not env.c_include:
        lines.append("  нет src/vtm/vtm.h")
    lines.append("Соберите его:")
    lines.append("    make -C src/vtm")
    return "\n".join(lines)


def _check_preflight(args: argparse.Namespace, env: Environment) -> None:
    if not env.torch:
        raise ParityError(_missing_torch_message())
    if not env.transformers:
        raise ParityError(_missing_transformers_message())
    if args.quantize_embeddings == "int8":
        raise ParityError(_EMBED_INT8_DEFERRED)
    if not env.vtm_bin or not env.vtm_lib or not env.c_include:
        raise ParityError(_missing_c_runtime_message(env))
    if not env.cc:
        raise ParityError(
            "[parity_real] не найден C-компилятор (cc/clang/gcc): C-проба "
            "логитов не соберётся.\nУстановите компилятор (или задайте $CC) и "
            "повторите; сам рантайм собирается через make -C src/vtm."
        )
    if not env.compiler_script:
        raise ParityError(
            f"[parity_real] нет компилятора VTM: {VTM_COMPILER}"
        )
    if not env.export_module:
        raise ParityError(
            "[parity_real] нет qat.export: нужен файл qat/export.py в репозитории"
        )
    if not env.numpy:
        raise ParityError(
            "[parity_real] нет numpy: qat.export требует numpy. Установите "
            "GPU-стек (см. qat/requirements-qat.txt)."
        )
    if env.model_state == "missing":
        raise ParityError(
            f"[parity_real] модели нет: {args.model}\n"
            "Укажите локальный путь или hub id (например, "
            "Qwen/Qwen2.5-1.5B-Instruct)."
        )


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    stage = started
    timings: dict[str, float] = {}
    warnings: list[str] = []

    env = check_environment(args.model)
    _check_preflight(args, env)

    workdir = Path(args.workdir).expanduser().resolve()
    try:
        workdir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ParityError(f"не создаётся workdir {workdir}: {exc}") from exc
    try:
        free_bytes = shutil.disk_usage(workdir).free
    except OSError:
        free_bytes = None
    if free_bytes is not None and free_bytes < 8 * 2**30:
        warnings.append(
            f"в {workdir} свободно {free_bytes / 2**30:.1f} ГиБ: для "
            "model.pt + ternary.pt + non_ternary.bin + .vtm на 1.5B нужно "
            "~10–16 ГиБ, прогон может упасть по диску"
        )
    run_dir = workdir / "qat_run"
    export_dir = workdir / "export"
    vtm_path = workdir / "model.vtm"
    result_path = workdir / "parity.json"

    vtm_sources = [
        VTM_INCLUDE / name
        for name in ("vtm.h", "loader.c", "interp.c", "backend_scalar.c", "trit.c")
    ]
    if VTM_LIB.is_file() and VTM_BIN.is_file():
        try:
            lib_mtime = VTM_LIB.stat().st_mtime
            bin_mtime = VTM_BIN.stat().st_mtime
            newer_than_lib = [
                path.name for path in vtm_sources
                if path.is_file() and path.stat().st_mtime > lib_mtime
            ]
            newer_than_bin = [
                path.name for path in vtm_sources
                if path.is_file() and path.stat().st_mtime > bin_mtime
            ]
        except OSError:
            newer_than_lib = []
            newer_than_bin = []
        if newer_than_lib or newer_than_bin:
            warnings.append(
                "сборка src/vtm старше исходников ("
                + ", ".join(sorted(set(newer_than_lib + newer_than_bin))[:4])
                + "): перед прогоном стоит сделать make -C src/vtm"
            )

    print("[parity_real] 1/8: HF-токенайзер и модель")
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - сломанный GPU-стек
        raise ParityError(f"не импортируется torch/transformers: {exc}") from exc

    device = resolve_device(torch, args.device)
    print(f"[parity_real] устройство: {device}, dtype {args.torch_dtype}")
    if args.torch_dtype != "float32":
        warnings.append(
            f"HF-эталон считается в {args.torch_dtype}, а не fp32: "
            "max|Δ| может вырасти из-за низкой точности самого эталона; "
            "для чистого порога используйте --torch-dtype float32."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=False
        )
    except Exception as exc:  # noqa: BLE001 - сеть/кэш/битый токенайзер
        raise ParityError(f"не загрузился токенайзер {args.model!r}: {exc}") from exc
    try:
        token_ids, truncated = tokenize_prompt(
            tokenizer, args.prompt, args.limit_tokens
        )
    except Exception as exc:  # noqa: BLE001 - битый/неожиданный токенайзер
        raise ParityError(
            f"не удалось токенизировать промпт {args.prompt!r}: {exc}"
        ) from exc
    del tokenizer
    if not token_ids:
        raise ParityError(
            f"промпт {args.prompt!r} дал пустой список токенов — сверять нечего"
        )
    if truncated:
        warnings.append(
            f"промпт обрезан до {args.limit_tokens} токенов (--limit-tokens)"
        )
    context_len = max(8, len(token_ids))
    print(f"[parity_real] токенов: {len(token_ids)} {token_ids}; context_len={context_len}")

    try:
        model = load_hf_model(args.model, args.torch_dtype)
        model.eval()
        if hf_tie_broken(model):
            raise ParityError(
                "config.tie_word_embeddings=true, но lm_head.weight и "
                "model.embed_tokens.weight — разные тензоры: qat.export "
                "запишет tied-таблицу, а HF-эталон посчитает отдельный lm_head. "
                "Загрузите модель с tie_word_embeddings=true/false согласованно."
            )
        model.to(device)
    except ParityError:
        raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise ParityError(
                "HF-модель не поместилась в память: попробуйте "
                "--device cpu или --torch-dtype bfloat16.\n" + str(exc)
            ) from exc
        raise ParityError(f"HF-модель не загрузилась: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - сеть/кэш/битый конфиг
        raise ParityError(f"HF-модель не загрузилась: {exc}") from exc

    check_model_supported(model.config, context_len)
    hf_architecture = hf_config_dict(model.config)
    architecture = build_architecture(model.config, context_len)
    if context_len > architecture["max_context"]:
        raise ParityError(
            f"промпт {len(token_ids)} токенов длиннее max_position_embeddings="
            f"{architecture['max_context']}"
        )
    vocab = architecture["vocab"]
    kv_dim = architecture["n_kv_heads"] * architecture["head_dim"]
    kv_bytes = architecture["n_layers"] * context_len * kv_dim * 2 * 4
    if kv_bytes > 2 * 2**30:
        raise ParityError(
            f"context_len={context_len} даёт KV-кэш {kv_bytes / 2**30:.1f} ГиБ; "
            "контейнер .vtm не влезет в лимит загрузчика 4 ГиБ. "
            "Уменьшите --limit-tokens."
        )
    if args.quantize_embeddings == "none":
        embed_bytes = vocab * architecture["hidden"] * 4
        if embed_bytes > 512 * 1024 * 1024:
            warnings.append(
                f"fp32-эмбеддинги: {embed_bytes / 2**30:.1f} ГиБ в non_ternary.bin; "
                "tools/vtm_compile.py держит их списком float — на 1.5B это "
                "может потребовать ~8–10 ГиБ RAM. Если не хватит памяти, "
                "запустите с --quantize-embeddings trit (таблица уйдёт в "
                "ternary.trit)."
            )

    print("[parity_real] 2/8: forward полной точности (справочно)")
    try:
        full_logits = hf_logits(torch, model, token_ids, device)
    except Exception as exc:  # noqa: BLE001 - ошибка HF/OOM/формы
        raise ParityError(f"HF forward (full) упал: {exc}") from exc
    if len(full_logits) != vocab:
        raise ParityError(
            f"HF вернул {len(full_logits)} логитов, а vocab={vocab}"
        )
    timings["hf_full_forward"] = time.monotonic() - stage
    stage = time.monotonic()
    num_params = int(sum(parameter.numel() for parameter in model.parameters()))

    print("[parity_real] 3/8: тернаризация HF тем же кодом, что QAT")
    try:
        quant, ternary_report, _embed_report = apply_hf_quantization(
            model, args.quantize_embeddings
        )
        if (
            args.quantize_embeddings == "trit"
            and bool(getattr(model.config, "tie_word_embeddings", False))
        ):
            if retie_quantized_lm_head(torch, model):
                warnings.append(
                    "tied lm_head пересобран по квантованной таблице "
                    "эмбеддингов, чтобы HF-эталон совпал с tied-графом .vtm."
                )
            else:
                warnings.append(
                    "не удалось пересобрать tied lm_head по квантованной "
                    "таблице: HF-эталон может считать lm_head в полной точности."
                )
    except Exception as exc:  # noqa: BLE001 - QAT-код может отказать
        raise ParityError(f"не удалось тернаризовать модель: {exc}") from exc
    try:
        quantized_logits = hf_logits(torch, model, token_ids, device)
    except Exception as exc:  # noqa: BLE001 - ошибка HF/OOM/формы
        raise ParityError(f"HF forward (ternary) упал: {exc}") from exc
    if len(quantized_logits) != vocab:
        raise ParityError(
            f"HF ternary вернул {len(quantized_logits)} логитов, а vocab={vocab}"
        )
    timings["hf_ternary_forward"] = time.monotonic() - stage
    stage = time.monotonic()

    rms_eps = float(architecture["rms_norm_eps"])
    if int(round(rms_eps * 1000.0)) == 0 and rms_eps > 0:
        warnings.append(
            f"RMSN eps={rms_eps:g} кодируется в vtm_compile.py как imm=0 "
            "(округление до тысячных): C считает eps=0. Вклад в логиты мал, "
            "но это известное ограничение."
        )

    run_config = build_run_config(args, quant)
    ternary_params = int(sum(int(item["params"]) for item in ternary_report))

    print("[parity_real] 4/8: QAT-чекпоинт")
    try:
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        raw_state = model.state_dict()
    except RuntimeError as exc:
        raise ParityError(f"не удалось перенести модель на CPU: {exc}") from exc
    # qat.export._to_numpy_bytes умеет только fp32/fp16; bf16 через .numpy()
    # падает. Мастер-веса для экспорта всё равно нужны в fp32 — приводим.
    skipped_buffers = {
        key for key in raw_state if "rotary_emb" in key or "inv_freq" in key
    }
    if skipped_buffers:
        warnings.append(
            "из state_dict убраны служебные буферы RoPE: "
            + ", ".join(sorted(skipped_buffers)[:4])
        )
    try:
        state = {
            key: value.detach().to(torch.float32)
            for key, value in raw_state.items()
            if key not in skipped_buffers
        }
    except Exception as exc:  # noqa: BLE001 - неожиданный dtype/устройство
        raise ParityError(f"не удалось привести state_dict к fp32: {exc}") from exc
    del raw_state
    run_meta = build_run_meta(
        model_name=args.model,
        config=run_config,
        hf_architecture=hf_architecture,
        vocab=vocab,
        num_params=num_params,
        ternary_params=ternary_params,
        ternary_report=ternary_report,
        device=str(device),
    )
    try:
        write_json_atomic(run_dir / "run.json", run_meta)
    except OSError as exc:
        raise ParityError(f"не записался {run_dir / 'run.json'}: {exc}") from exc
    try:
        ckpt = save_qat_checkpoint(torch, state, run_dir, config=run_config)
    except Exception as exc:  # noqa: BLE001 - диск/сериализация
        raise ParityError(f"не сохранился QAT-чекпоинт: {exc}") from exc
    del state
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    timings["checkpoint"] = time.monotonic() - stage
    stage = time.monotonic()
    print(f"[parity_real] чекпоинт: {ckpt}")

    print("[parity_real] 5/8: qat.export -> ternary.trit + non_ternary.bin")
    print("[parity_real] квантование ~1.5B весов может занять несколько минут")
    export_cmd = build_export_command(
        sys.executable,
        run_dir,
        export_dir,
        context_len=context_len,
        scale_dtype=args.scale_dtype,
    )
    proc = run_command(export_cmd)
    if proc.returncode != 0:
        _print_command_failure("qat.export", proc)
        timings["export"] = time.monotonic() - stage
        try:
            write_failure_json(
                result_path, args=args, token_ids=token_ids, workdir=workdir,
                context_len=context_len, stage="qat.export",
                error=f"qat.export: код {proc.returncode}",
                returncode=proc.returncode, stdout=proc.stdout,
                stderr=proc.stderr, warnings=warnings, timings=timings,
            )
        except OSError as exc:
            print(f"[parity_real] не записался {result_path}: {exc}",
                  file=sys.stderr)
        return 2
    if proc.stdout.strip():
        print(tail(proc.stdout, 40))
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ParityError(f"qat.export не написал {manifest_path}")
    try:
        manifest_arch = json.loads(
            manifest_path.read_text(encoding="utf-8")
        ).get("architecture") or {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ParityError(f"не читается {manifest_path}: {exc}") from exc
    if not isinstance(manifest_arch, dict):
        raise ParityError(
            f"{manifest_path}: architecture должна быть объектом, "
            f"а не {type(manifest_arch).__name__}"
        )
    mismatches = check_manifest_architecture(
        manifest_arch,
        architecture,
        context_len=context_len,
        embed_mode=args.quantize_embeddings,
    )
    if mismatches:
        raise ParityError(
            "qat.export записал не ту архитектуру: " + "; ".join(mismatches)
        )
    timings["export"] = time.monotonic() - stage
    stage = time.monotonic()

    print("[parity_real] 6/8: tools/vtm_compile.py -> .vtm")
    print("[parity_real] Python-компилятор разбирает .trit; на 1.5B это может "
          "занять десятки минут и ~8–10 ГиБ RAM")
    compile_cmd = build_compile_command(
        sys.executable, manifest_path, vtm_path
    )
    proc = run_command(compile_cmd)
    if proc.returncode != 0:
        _print_command_failure("vtm_compile", proc)
        timings["compile"] = time.monotonic() - stage
        try:
            write_failure_json(
                result_path, args=args, token_ids=token_ids, workdir=workdir,
                context_len=context_len, stage="vtm_compile",
                error=f"vtm_compile: код {proc.returncode}",
                returncode=proc.returncode, stdout=proc.stdout,
                stderr=proc.stderr, warnings=warnings, timings=timings,
            )
        except OSError as exc:
            print(f"[parity_real] не записался {result_path}: {exc}",
                  file=sys.stderr)
        return 2
    if proc.stdout.strip():
        print(tail(proc.stdout, 40))
    if not vtm_path.is_file():
        raise ParityError(f"vtm_compile не написал {vtm_path}")
    timings["compile"] = time.monotonic() - stage
    stage = time.monotonic()

    print("[parity_real] 7/8: C-рантайм, argmax через vtm run")
    print("[parity_real] C-рантайм скалярный: прогон промпта может занять минуты")
    cli_cmd = build_run_command(VTM_BIN, vtm_path, token_ids)
    proc = run_command(cli_cmd)
    if proc.returncode != 0:
        _print_command_failure("vtm run", proc)
        if "версия" in (proc.stderr or "").lower():
            print(
                "[parity_real] C-рантайм устарел относительно формата .vtm: "
                "make -C src/vtm",
                file=sys.stderr,
            )
        print(
            "[parity_real] если контейнер новее бинарника — "
            "make -C src/vtm",
            file=sys.stderr,
        )
        timings["c_cli"] = time.monotonic() - stage
        try:
            write_failure_json(
                result_path, args=args, token_ids=token_ids, workdir=workdir,
                context_len=context_len, stage="vtm run",
                error=f"vtm run: код {proc.returncode}",
                returncode=proc.returncode, stdout=proc.stdout,
                stderr=proc.stderr, warnings=warnings, timings=timings,
            )
        except OSError as exc:
            print(f"[parity_real] не записался {result_path}: {exc}",
                  file=sys.stderr)
        return 1
    if proc.stdout.strip():
        print(tail(proc.stdout, 20))
    cli = parse_vtm_run_output(proc.stdout)
    if not cli["generated"]:
        print(
            "[parity_real] vtm run не выбрал токен (нет SAMP/пустой вывод); "
            "проверьте .vtm",
            file=sys.stderr,
        )
        timings["c_cli"] = time.monotonic() - stage
        try:
            write_failure_json(
                result_path, args=args, token_ids=token_ids, workdir=workdir,
                context_len=context_len, stage="vtm run",
                error="vtm run не выбрал токен (нет SAMP/пустой вывод)",
                returncode=proc.returncode, stdout=proc.stdout,
                stderr=proc.stderr, warnings=warnings, timings=timings,
            )
        except OSError as exc:
            print(f"[parity_real] не записался {result_path}: {exc}",
                  file=sys.stderr)
        return 1
    timings["c_cli"] = time.monotonic() - stage
    stage = time.monotonic()

    print("[parity_real] 8/8: C-проба логитов (libvtm.a)")
    probe: ProbeResult | None = None
    probe_error: str | None = None
    try:
        probe = run_c_probe(workdir, vtm_path, token_ids, env.cc or "cc")
    except ParityError as exc:
        probe_error = str(exc)
        warnings.append(f"C-проба логитов не удалась: {exc}")
    if probe is not None and len(probe.logits) != vocab:
        probe_error = (
            f"C-проба вернула {len(probe.logits)} логитов, а vocab={vocab}"
        )
        warnings.append(probe_error)
        probe = None
    timings["c_probe"] = time.monotonic() - stage
    stage = time.monotonic()

    reference_quantized = summarize_logits(quantized_logits)
    reference_full = summarize_logits(full_logits)
    quantization_effect = compare_logits(
        full_logits, quantized_logits, DEFAULT_THRESHOLD
    )
    quantization_effect["informational"] = True

    if probe is not None:
        runtime_summary = summarize_logits(probe.logits)
        comparison = compare_logits(
            quantized_logits, probe.logits, DEFAULT_THRESHOLD
        )
        comparison["method"] = "c-probe"
        meta_problems = check_runtime_meta(
            probe.meta, architecture, len(probe.logits)
        )
        cli_argmax_match = int(cli["generated"][0]) == int(runtime_summary["argmax"])
        comparison["meta_ok"] = not meta_problems
        comparison["cli_argmax_match"] = cli_argmax_match
        comparison["passed"] = bool(
            comparison["passed"] and not meta_problems and cli_argmax_match
        )
        full_precision = compare_logits(
            full_logits, probe.logits, DEFAULT_THRESHOLD
        )
        full_precision["informational"] = True
        warnings.extend(meta_problems)
        if not cli_argmax_match:
            warnings.append(
                "vtm run выбрал не тот токен, что argmax C-пробы: "
                f"{cli['generated'][0]} против {runtime_summary['argmax']}"
            )
        status = "passed" if comparison["passed"] else "failed"
        runtime = {
            "kind": "vtm-c",
            "argmax": runtime_summary["argmax"],
            "top5": runtime_summary["top5"],
            "count": runtime_summary["count"],
            "first8": runtime_summary["first8"],
            "sum": runtime_summary["sum"],
            "cli": {
                "returncode": proc.returncode,
                "prompt": cli["prompt"],
                "generated": cli["generated"],
                "backend": cli["backend"],
                "tensor_count": cli["tensor_count"],
                "instr_count": cli["instr_count"],
            },
            "meta": probe.meta,
            "meta_problems": meta_problems,
            "probe": {"error": None},
        }
    else:
        runtime_summary = {
            "argmax": int(cli["generated"][0]),
            "top5": [],
            "count": None,
            "first8": [],
            "sum": None,
        }
        comparison = {
            "method": "cli-only",
            "max_abs_delta": None,
            "argmax_match": int(cli["generated"][0])
            == int(reference_quantized["argmax"]),
            "argmax_reference": int(reference_quantized["argmax"]),
            "argmax_runtime": int(cli["generated"][0]),
            "top5_reference": reference_quantized["top5"],
            "top5_runtime": [],
            "top5_overlap": None,
            "top5_size": 5,
            "threshold": DEFAULT_THRESHOLD,
            "within_threshold": None,
            "passed": False,
            "meta_ok": None,
            "cli_argmax_match": int(cli["generated"][0])
            == int(reference_quantized["argmax"]),
        }
        full_precision = None
        status = "partial"
        runtime = {
            "kind": "vtm-c-cli-only",
            "argmax": int(cli["generated"][0]),
            "top5": [],
            "count": None,
            "first8": [],
            "sum": None,
            "cli": {
                "returncode": proc.returncode,
                "prompt": cli["prompt"],
                "generated": cli["generated"],
                "backend": cli["backend"],
                "tensor_count": cli["tensor_count"],
                "instr_count": cli["instr_count"],
            },
            "meta": {},
            "meta_problems": [],
            "probe": {"error": probe_error},
        }

    artifacts = {
        "run_json": str(run_dir / "run.json"),
        "model_pt": str(ckpt / "model.pt"),
        "manifest": str(manifest_path),
        "ternary_trit": str(export_dir / "ternary.trit"),
        "non_ternary": str(export_dir / "non_ternary.bin"),
        "vtm": str(vtm_path),
        "probe_c": str(workdir / "parity_probe.c"),
        "probe_bin": str(workdir / "parity_probe"),
        "json": str(result_path),
    }
    timings["total"] = time.monotonic() - started
    result = make_result(
        model=args.model,
        prompt=args.prompt,
        token_ids=token_ids,
        limit_tokens=args.limit_tokens,
        quantize_embeddings=args.quantize_embeddings,
        scale_dtype=args.scale_dtype,
        device=str(device),
        torch_dtype=args.torch_dtype,
        context_len=context_len,
        workdir=workdir,
        artifacts=artifacts,
        reference_quantized=reference_quantized,
        reference_full=reference_full,
        runtime=runtime,
        comparison=comparison,
        full_precision=full_precision,
        quantization_effect=quantization_effect,
        warnings=warnings,
        timings=timings,
        status=status,
        error=probe_error,
        argv=sys.argv,
    )
    try:
        write_json_atomic(result_path, result)
    except OSError as exc:
        raise ParityError(f"не записался {result_path}: {exc}") from exc
    print()
    print(format_summary(result))
    print()

    if status == "passed":
        return 0
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:  # argparse уже напечатал ошибку
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        if args.dry_run:
            return dry_run(args)
        return run(args)
    except ParityError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[parity_real] прервано", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
