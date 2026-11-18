#!/usr/bin/env python3
"""Профиль полного прохода VTM: куда уходит время.

Знаем скорость сэмплинга (`tests/test_vtm_sampling.py`, 0051/0059), знаем
скорость TMV (`src/vtm/bench_backend`, `docs/vtm-perf.md`), но не знаем, на что
уходит время в **полном проходе** модели: TMV, attention, RMSN, сэмплинг или
загрузка контейнера. Без этого непонятно, что оптимизировать первым.

Что делает скрипт:

1. гоняет **настоящий** ``vtm run`` (CLI) на контейнерах разного размера:
   контекст 128/512/2048 × генерация 1/32/128 × бэкенд scalar/simd;
2. считает мс/токен, токенов/с, мс/шаг (полный проход) и wall-время, отдельно
   отдаёт «вне CL» — запуск процесса и загрузку контейнера (таймер CLI идёт
   после ``vtm_load``);
3. **разбивает время по операциям**. Источник — трасса интерпретатора (задача
   0082: ``vtm run --trace-summary`` или ``--trace``, проверяется через
   ``vtm --help``): интерпретатор снимает ``clock_gettime(CLOCK_MONOTONIC)``
   вокруг каждой инструкции, так что доли — **реальный замер**, а не оценка.
   Если сборка трассы не умеет (старый бинарник), остаётся **оценка через
   сравнение конфигураций**: в байткод копируются инструкции одного класса
   (TMV, ATTN, RMSN, SAMP, KVW), и прирост времени на копию даёт вклад класса в
   проход. Оценка шумит (копии меняют кэш-поведение, диспетчеризация уходит в
   остаток), поэтому она — резерв и перекрёстная проверка; метод честно
   помечен в JSON (``breakdown.method``) и в отчёте;
4. тайминги сценариев снимаются **без** трассировки (хук замедляет прогон), а
   доли — отдельным прогоном с трассой; при наличии trace оценка дублированием
   всё равно считается и лежит в ``breakdown.dup_estimate``;
5. ``--json`` — машинный отчёт, иначе таблица; ``--budget-ms-token`` включает
   сравнение с бюджетом (превышение — код возврата 1).

Почему контекст 2048 работает, хотя у демо-модели KV-кэш на 8 позиций:
``prepare_container`` пересобирает контейнер с тем же графом/весами, но с
KV-кэшем нужной длины (``l0.kcache``/``l0.vcache``), и при необходимости
добавляет ``SAMP`` перед ``HALT`` (у синтетики компилятора его нет). Это
честная копия модели, а не другой рантайм: меняется только размер кэша.
Если в контейнере есть блок ``integrity`` (sha256, задача 0066), копия его
пересчитывает — иначе загрузчик отверг бы файл как повреждённый.

Запуск::

    python3 tools/bench_vtm.py                    # demo, полная матрица, таблица
    python3 tools/bench_vtm.py --json             # то же в JSON
    python3 tools/bench_vtm.py --synthetic-size 256
    python3 tools/bench_vtm.py --contexts 128,2048 --gen 1,32 \
        --backends simd --budget-ms-token 33.3 --json
    python3 tools/bench_vtm.py --breakdown-contexts 128,2048 --dup-copies 4

Ограничения (важно для чтения чисел):

* контейнеры синтетические/демо: 1 слой, GQA 4:1, hidden = vocab = 128…256.
  Доля attention и SAMP здесь **выше**, чем у настоящей Qwen-1.5B (у неё на
  токен ~1.5 млрд MAC TMV против ~6 млн у attention). Тренды сравнимы,
  абсолютные доли — нет;
* таймер CLI — ``clock()`` (процессорное время), поэтому «вне CL» включает
  старт процесса и загрузку, а не только I/O;
* разбивка дублированием шумит на быстрых операциях: абсолютная ошибка часов
  ~0,5 мс на прогон, поэтому мелкие классы (KVW, SAMP на vocab=128) видны
  хуже крупных; для них есть ``--dup-copies``;
* тесты (``tests/test_bench_vtm.py``) не требуют C: CLI подменяется
  mock-раннером.

Код возврата: 0 — замеры сняты, бюджет не превышен (или не задан); 1 —
превышен бюджет ``--budget-ms-token``; 2 — нет бинарника/контейнера, CLI упал
или вывод не разобран.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import vtm_ref  # noqa: E402

#: Пути по умолчанию. Абсолютные, чтобы скрипт работал из любого каталога.
VTM_BIN_DEFAULT = os.path.join(ROOT, "src", "vtm", "vtm")
DEMO_DIR = os.path.join(ROOT, "demo")
DEMO_MODEL = os.path.join(DEMO_DIR, "model.vtm")
DEMO_SCRIPT = os.path.join(ROOT, "scripts", "demo_ternary_model.py")
COMPILER = os.path.join(ROOT, "tools", "vtm_compile.py")

REPORT_VERSION = 1

DEFAULT_CONTEXTS = (128, 512, 2048)
DEFAULT_GEN = (1, 32, 128)
DEFAULT_BACKENDS = ("scalar", "simd")
DEFAULT_REPEAT = 3
DEFAULT_DUP_COPIES = 8
DEFAULT_BREAKDOWN_REPEAT = 2
DEFAULT_SYNTHETIC_SIZE = 256
DEFAULT_SYNTHETIC_SEED = 7
DEFAULT_SYNTHETIC_MODE = "attn"
DEFAULT_TIMEOUT_S = 600.0

#: Допустимые имена бэкендов CLI (``--backend``).
BACKEND_CHOICES = ("auto", "simd", "scalar", "lut", "direct")

#: Классы операций для разбивки: имя в отчёте -> opcode-имена.
CLUSTERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("TMV", ("TMV",)),
    ("MM", ("MM",)),
    ("ATTN", ("ATTN",)),
    ("RMSN", ("RMSN",)),
    ("SAMP", ("SAMP",)),
    ("KVW", ("KVW",)),
    ("прочее", ("EMB", "ROPE", "SM", "SILU", "MUL", "ADD", "CPY", "SCAL")),
)

#: HALT в классы не входит: он ничего не считает, его время — в остатке.
NON_CLUSTER_OPS = frozenset({"HALT"})

#: Комментарий к методу разбивки, одинаковый в JSON и в таблице.
BREAKDOWN_NOTE = (
    "оценка через сравнение конфигураций: инструкции класса копируются в байткод "
    "(+N копий), прирост времени на прогон делится на копии и на число проходов; "
    "копии меняют кэш-поведение, неучтённое — в остатке"
)
TRACE_NOTE = (
    "разбивка из трассы интерпретатора (задача 0082): время каждой инструкции — "
    "clock_gettime(CLOCK_MONOTONIC), интерпретатор суммирует по опкодам; это "
    "реальный замер, а не оценка"
)

#: Ключ меты с sha256-блоком (задача 0066). Производные контейнеры нельзя
#: писать со старым блоком: хеш данных после правки кэшей другой.
INTEGRITY_KEY = "integrity"


class BenchError(RuntimeError):
    """Ошибка бенчмарка: нет бинарника, CLI упал, вывод не разобран."""


# ---------------------------------------------------------------------------
# Разбор вывода `vtm run`
# ---------------------------------------------------------------------------


@dataclass
class RunOutput:
    """Разобранный вывод одного ``vtm run``."""

    time_ms: float
    generated: list[int] = field(default_factory=list)
    prompt: list[int] = field(default_factory=list)
    tokens_per_s: float | None = None
    backend: str | None = None
    tensor_count: int | None = None
    instr_count: int | None = None
    raw: str = field(default="", repr=False)


_RE_COUNTERS = re.compile(
    r"тензоров:\s*(\d+)\s*,\s*инструкций:\s*(\d+)\s*,\s*бэкенд:\s*(\S+)"
)
# "время: 1.234 мс" / "время: 1,234 мс, 32 токен(ов), 1234.5 ток/с"
_RE_TIME = re.compile(
    r"время:\s*(-?\d+(?:[.,]\d+)?)\s*мс"
    r"(?:\s*,\s*(\d+)\s*ток(?:ен\(ов\)|ена|ен)?"
    r"(?:\s*,\s*(-?\d+(?:[.,]\d+)?)\s*ток/с)?)?"
)


def _to_float(text: str) -> float:
    return float(text.replace(",", "."))


def _parse_ids(text: str) -> list[int]:
    value = text.strip()
    if not value or value.startswith("("):
        return []
    try:
        return [int(part) for part in value.split()]
    except ValueError as exc:
        raise BenchError(f"vtm run: не список токенов в строке {value!r}") from exc


def parse_run_output(stdout: str) -> RunOutput:
    """Разобрать stdout ``vtm run``: токены, счётчики, время, бэкенд.

    Формат — тот, что печатает ``cmd_run``: строки ``prompt:``, ``вывод:``,
    ``тензоров: N, инструкций: M, бэкенд: X`` и ``время: T мс, K токен(ов),
    V ток/с``. Текст ошибки на русском, поэтому разбор — по префиксам, а не по
    позициям.
    """
    if not isinstance(stdout, str):
        raise BenchError("vtm run: ожидался текст, а не байты")
    result = RunOutput(time_ms=float("nan"), raw=stdout)
    have_time = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("prompt:"):
            result.prompt = _parse_ids(stripped.split(":", 1)[1])
        elif stripped.startswith("вывод:"):
            result.generated = _parse_ids(stripped.split(":", 1)[1])
        elif stripped.startswith("тензоров:"):
            match = _RE_COUNTERS.search(stripped)
            if match:
                result.tensor_count = int(match.group(1))
                result.instr_count = int(match.group(2))
                result.backend = match.group(3)
        elif stripped.startswith("время:"):
            match = _RE_TIME.search(stripped)
            if match:
                result.time_ms = _to_float(match.group(1))
                if match.group(3) is not None:
                    result.tokens_per_s = _to_float(match.group(3))
                have_time = True
    if not have_time:
        raise BenchError(
            "vtm run: в выводе нет строки «время: … мс» — это не вывод vtm run?"
        )
    return result


# ---------------------------------------------------------------------------
# Команды CLI
# ---------------------------------------------------------------------------


def build_run_command(
    vtm_bin: str,
    model: str,
    token_ids: Sequence[int],
    max_new: int,
    *,
    backend: str | None = None,
    trace: bool = False,
    trace_summary: bool = False,
) -> list[str]:
    """Собрать команду ``vtm run`` (без текстового режима и токенизатора).

    ``trace`` — построчная трасса (``--trace``), ``trace_summary`` — сводка по
    опкодам (``--trace-summary``, задача 0082). Для разбивки нужен ровно один
    из них: сводку интерпретатор считает сам и печатает одну строку на опкод.
    """
    if not token_ids:
        raise BenchError("build_run_command: пустой промпт")
    if max_new < 0:
        raise BenchError("build_run_command: max_new < 0")
    if backend is not None and backend not in BACKEND_CHOICES:
        raise BenchError(f"build_run_command: неизвестный бэкенд {backend!r}")
    if trace and trace_summary:
        raise BenchError("build_run_command: --trace и --trace-summary взаимоисключающие")
    command = [
        str(vtm_bin),
        "run",
        str(model),
        "--tokens",
        ",".join(str(int(token)) for token in token_ids),
        "--max-new",
        str(int(max_new)),
    ]
    if backend is not None:
        command += ["--backend", backend]
    if trace:
        command.append("--trace")
    if trace_summary:
        command.append("--trace-summary")
    return command


def build_inspect_command(vtm_bin: str, model: str) -> list[str]:
    """Команда ``vtm inspect`` — карточка контейнера и прокси времени загрузки."""
    return [str(vtm_bin), "inspect", str(model)]


def build_synthetic_command(
    python: str,
    compiler: str,
    out: str,
    *,
    size: int,
    seed: int = DEFAULT_SYNTHETIC_SEED,
    mode: str = DEFAULT_SYNTHETIC_MODE,
) -> list[str]:
    """Команда сборки синтетики: ``vtm_compile.py --synthetic --size … --out …``.

    Других флагов у компилятора для синтетики нет: ``--seed``, ``--size``,
    ``--mode`` (``attn``/``mlp``), ``--out``. ``context_len`` в синтетике
    зашит (8), поэтому длинный контекст получается расширением KV-кэша в
    :func:`prepare_container`.
    """
    return [
        str(python),
        str(compiler),
        "--synthetic",
        "--size",
        str(int(size)),
        "--seed",
        str(int(seed)),
        "--mode",
        str(mode),
        "--out",
        str(out),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_vtm.py",
        description=(
            "Профиль полного прохода VTM: контекст × генерация × бэкенд, "
            "мс/токен, ток/с, разбивка по операциям."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python3 tools/bench_vtm.py\n"
            "  python3 tools/bench_vtm.py --json\n"
            "  python3 tools/bench_vtm.py --synthetic-size 256 --contexts 128,512\n"
            "  python3 tools/bench_vtm.py --budget-ms-token 33.3 --json\n"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"контейнер .vtm (по умолчанию {os.path.relpath(DEMO_MODEL, ROOT)}; "
        "если нет — генерируется scripts/demo_ternary_model.py)",
    )
    parser.add_argument(
        "--synthetic-size",
        type=int,
        default=None,
        help="собрать синтетику через tools/vtm_compile.py --synthetic "
        f"(hidden, по умолчанию {DEFAULT_SYNTHETIC_SIZE}); контейнер demo не нужен",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=DEFAULT_SYNTHETIC_SEED,
        help=f"seed синтетики (по умолчанию {DEFAULT_SYNTHETIC_SEED})",
    )
    parser.add_argument(
        "--synthetic-mode",
        choices=("attn", "mlp"),
        default=DEFAULT_SYNTHETIC_MODE,
        help=f"режим синтетики (по умолчанию {DEFAULT_SYNTHETIC_MODE})",
    )
    parser.add_argument(
        "--vtm",
        default=VTM_BIN_DEFAULT,
        help=f"бинарник vtm (по умолчанию {os.path.relpath(VTM_BIN_DEFAULT, ROOT)})",
    )
    parser.add_argument(
        "--contexts",
        type=_int_list_arg,
        default=list(DEFAULT_CONTEXTS),
        help=f"длины промпта через запятую (по умолчанию {_join(DEFAULT_CONTEXTS)})",
    )
    parser.add_argument(
        "--gen",
        type=_int_list_arg,
        default=list(DEFAULT_GEN),
        help=f"сколько токенов генерировать (по умолчанию {_join(DEFAULT_GEN)})",
    )
    parser.add_argument(
        "--backends",
        type=str,
        default=",".join(DEFAULT_BACKENDS),
        help=f"бэкенды через запятую (по умолчанию {_join(DEFAULT_BACKENDS)})",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=DEFAULT_REPEAT,
        help=f"повторов на сценарий, берётся медиана (по умолчанию {DEFAULT_REPEAT})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="напечатать JSON вместо таблицы (валидный, для машин)",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="дополнительно записать JSON в файл",
    )
    parser.add_argument(
        "--budget-ms-token",
        type=float,
        default=None,
        help="бюджет на шаг генерации, мс/шаг (по умолчанию не задан); "
        "превышение — код возврата 1",
    )
    parser.add_argument(
        "--no-breakdown",
        action="store_true",
        help="не считать разбивку по операциям (быстрее)",
    )
    parser.add_argument(
        "--breakdown-contexts",
        type=_int_list_arg,
        default=None,
        help="контексты для разбивки (по умолчанию 128 и 512; можно 128,2048)",
    )
    parser.add_argument(
        "--breakdown-gen",
        type=int,
        default=1,
        help="генерация в замере разбивки (по умолчанию 1 — чистая цена прохода)",
    )
    parser.add_argument(
        "--breakdown-backends",
        default=None,
        help="бэкенды для разбивки (по умолчанию те же, что в сценариях)",
    )
    parser.add_argument(
        "--dup-copies",
        type=int,
        default=DEFAULT_DUP_COPIES,
        help=f"сколько копий инструкций добавлять в замер (по умолчанию "
        f"{DEFAULT_DUP_COPIES})",
    )
    parser.add_argument(
        "--breakdown-repeat",
        type=int,
        default=DEFAULT_BREAKDOWN_REPEAT,
        help=f"повторов на точку разбивки (по умолчанию {DEFAULT_BREAKDOWN_REPEAT})",
    )
    parser.add_argument(
        "--trace",
        choices=("auto", "off", "on"),
        default="auto",
        help="использовать `vtm run --trace`, если сборка его умеет "
        "(по умолчанию auto: проверка через vtm --help)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"таймаут одного запуска, с (по умолчанию {DEFAULT_TIMEOUT_S:g})",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="куда класть производные контейнеры (по умолчанию временный каталог)",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="не удалять временный каталог с производными контейнерами",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="seed выбора токенов промпта (по умолчанию 7)",
    )
    return parser


def _join(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in values)


def _int_list_arg(text: str) -> list[int]:
    """Разобрать ``128,512,2048``; пустые элементы и пробелы допускаются."""
    values: list[int] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"ожидалось целое число, получено {part!r}"
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"значение должно быть > 0: {value}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("пустой список")
    return values


def _backend_list_arg(text: str) -> list[str]:
    values = [part.strip() for part in str(text).replace(" ", "").split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("пустой список бэкендов")
    for value in values:
        if value not in BACKEND_CHOICES:
            raise argparse.ArgumentTypeError(
                f"бэкенд {value!r} вне {', '.join(BACKEND_CHOICES)}"
            )
    return values


# ---------------------------------------------------------------------------
# Запуск процесса
# ---------------------------------------------------------------------------

Runner = Callable[[Sequence[str], float | None], "subprocess.CompletedProcess[str]"]


def default_runner(
    command: Sequence[str], timeout: float | None
) -> "subprocess.CompletedProcess[str]":
    """Настоящий запуск. Тесты подменяют ``subprocess.run`` или сам runner."""
    return subprocess.run(
        list(command), capture_output=True, text=True, timeout=timeout
    )


def run_cli(
    command: Sequence[str],
    *,
    runner: Runner | None = None,
    timeout: float | None = None,
) -> "subprocess.CompletedProcess[str]":
    """Запустить команду и проверить код возврата.

    ``runner`` инжектируется, чтобы тесты обходились без C-рантайма. Ошибки
    процесса превращаются в :class:`BenchError` с хвостом stderr — бенчмарк не
    должен падать трейсбеком на первом неожиданном коде.
    """
    run = default_runner if runner is None else runner
    shown = " ".join(str(part) for part in command)
    try:
        proc = run(list(command), timeout)
    except subprocess.TimeoutExpired as exc:
        raise BenchError(f"таймаут {timeout:g} с: {shown}") from exc
    except OSError as exc:
        raise BenchError(f"не удалось запустить {shown}: {exc}") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        tail = "; ".join(stderr[-3:]) if stderr else "без stderr"
        raise BenchError(f"код {proc.returncode}: {shown}: {tail}")
    return proc


def run_and_parse(
    command: Sequence[str],
    *,
    runner: Runner | None = None,
    timeout: float | None = None,
) -> tuple[RunOutput, float]:
    """Запустить ``vtm run`` и вернуть (разобранный вывод, wall-время в мс)."""
    started = time.perf_counter()
    proc = run_cli(command, runner=runner, timeout=timeout)
    wall_ms = (time.perf_counter() - started) * 1000.0
    return parse_run_output(proc.stdout), wall_ms


def detect_trace(vtm_bin: str, *, runner: Runner | None = None) -> bool:
    """Есть ли в сборке ``--trace``: ищем флаг в общем ``vtm --help``.

    ``vtm run --help`` в текущей сборке — ошибка аргументов, поэтому общий
    help — единственный надёжный способ не ловить ложный код 2 на каждом
    запуске.
    """
    return "--trace" in _help_text(vtm_bin, runner=runner)


def detect_trace_summary(vtm_bin: str, *, runner: Runner | None = None) -> bool:
    """Есть ли ``--trace-summary`` (задача 0082).

    Сводка по опкодам — та же длительность инструкций, что у ``--trace``, но
    интерпретатор суммирует её сам: на прогоне с предзаполнением это в сотни
    раз меньше вывода, поэтому разбивка предпочитает её. Старые сборки (и
    mock-раннер в тестах) знают только ``--trace`` — тогда остаётся он.
    """
    return "--trace-summary" in _help_text(vtm_bin, runner=runner)


def _help_text(vtm_bin: str, *, runner: Runner | None = None) -> str:
    """Вывод ``vtm --help`` ("" — если бинарника нет или он не ответил)."""
    if not os.path.exists(vtm_bin):
        return ""
    try:
        proc = run_cli([str(vtm_bin), "--help"], runner=runner, timeout=30.0)
    except BenchError:
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------


def summarize(samples: Sequence[float]) -> dict[str, Any]:
    """Медиана/среднее/min/max по повторам. Медиана — основная цифра."""
    if not samples:
        raise BenchError("summarize: пустой список замеров")
    values = [float(value) for value in samples]
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


# ---------------------------------------------------------------------------
# Контейнеры
# ---------------------------------------------------------------------------


@dataclass
class ContainerInfo:
    """Структура контейнера, нужная для сценариев и разбивки."""

    path: str
    context: int
    vocab: int
    tensors: int
    instructions: int
    file_bytes: int
    ops: dict[str, int]
    meta: dict[str, Any] = field(default_factory=dict, repr=False)

    def as_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "context": self.context,
            "vocab": self.vocab,
            "tensors": self.tensors,
            "instructions": self.instructions,
            "file_bytes": self.file_bytes,
            "ops": dict(sorted(self.ops.items())),
        }


def container_context(vtm: "vtm_ref.VTM") -> int:
    """Длина KV-кэша контейнера: ``dims[0]`` тензора ``*cache``."""
    for tensor in vtm.tensors:
        if tensor.name.endswith("cache") and len(tensor.dims) == 2:
            return int(tensor.dims[0])
    for holder in (vtm.meta.get("model"), vtm.meta):
        if isinstance(holder, Mapping):
            value = holder.get("context_len")
            if isinstance(value, int) and value > 0:
                return int(value)
    return 0


def container_vocab(vtm: "vtm_ref.VTM") -> int:
    """Размер словаря: meta -> ``emb.w`` -> ``lm_head.w``."""
    meta = vtm.meta if isinstance(vtm.meta, Mapping) else {}
    model_meta = meta.get("model") if isinstance(meta.get("model"), Mapping) else {}
    candidates = (
        meta.get("vocab_size"),
        meta.get("vocab"),
        model_meta.get("vocab"),
        model_meta.get("vocab_size"),
    )
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return int(value)
    for tensor in vtm.tensors:
        if tensor.name.endswith("emb.w") or tensor.name == "lm_head.w":
            if len(tensor.dims) >= 2 and int(tensor.dims[0]) > 0:
                return int(tensor.dims[0])
    raise BenchError(
        f"не удалось определить словарь контейнера {getattr(vtm, 'path', '?')}: "
        "нет vocab в meta и нет emb.w/lm_head.w"
    )


def _info_from_vtm(path: str, vtm: "vtm_ref.VTM") -> ContainerInfo:
    return ContainerInfo(
        path=path,
        context=container_context(vtm),
        vocab=container_vocab(vtm),
        tensors=len(vtm.tensors),
        instructions=len(vtm.program),
        file_bytes=os.path.getsize(path),
        ops=vtm.op_counts(),
        meta=dict(vtm.meta) if isinstance(vtm.meta, Mapping) else {},
    )


def read_container_info(path: str) -> ContainerInfo:
    """Прочитать контейнер Python-эталоном (без C): размеры, ops, словарь."""
    try:
        vtm = vtm_ref.read_vtm(path)
    except (OSError, ValueError) as exc:
        raise BenchError(f"контейнер {path}: {exc}") from exc
    return _info_from_vtm(path, vtm)


def token_ids(count: int, vocab: int, seed: int = 7) -> list[int]:
    """Детерминированный промпт длины ``count`` (id < vocab).

    Шаг 7 взаимно прост с типичными размерами словаря (128/256), поэтому
    токены не вырождаются в один и тот же id: ``(i*7+1) mod vocab``.
    """
    if count <= 0:
        raise BenchError("token_ids: count <= 0")
    if vocab <= 0:
        raise BenchError("token_ids: vocab <= 0")
    step = 7 + (seed % 2) * 2  # 7 или 9 — оба нечётные
    return [((index * step) + 1) % vocab for index in range(count)]


def _cache_tensors(vtm: "vtm_ref.VTM") -> list["vtm_ref.Tensor"]:
    return [
        tensor
        for tensor in vtm.tensors
        if tensor.name.endswith("cache") and len(tensor.dims) == 2
    ]


def add_samp_if_missing(vtm: "vtm_ref.VTM") -> bool:
    """Добавить ``SAMP`` (argmax) перед ``HALT``, если его нет.

    У демо-модели SAMP добавляет сам скрипт, а синтетика компилятора его не
    содержит: без SAMP ``vtm run --max-new N`` печатает «программа не вызвала
    SAMP; генерировать нечего». Возвращает True, если контейнер изменён.
    """
    if int(vtm_ref.Op.SAMP) in {int(instr.op) for instr in vtm.program}:
        return False
    if not vtm.program or int(vtm.program[-1].op) != int(vtm_ref.Op.HALT):
        raise BenchError("add_samp_if_missing: программа не кончается HALT")
    logits_index = None
    for instr in reversed(vtm.program):
        if int(instr.op) in (int(vtm_ref.Op.TMV), int(vtm_ref.Op.MM)):
            logits_index = int(instr.dst)
            break
    if logits_index is None:
        raise BenchError("add_samp_if_missing: нет TMV/MM перед HALT — нечего сэмплить")
    token_index = len(vtm.tensors)
    used = {tensor.name for tensor in vtm.tensors}
    name = "token" if "token" not in used else "bench.token"
    vtm.tensors.append(vtm_ref.Tensor.f32(name, (1,), [0.0]))
    halt = vtm.program.pop()
    vtm.program.append(
        vtm_ref.Instr(
            op=int(vtm_ref.Op.SAMP),
            flags=0,
            dst=token_index,
            a=logits_index,
            b=0,
            c=0,
            imm=0,
        )
    )
    vtm.program.append(halt)
    return True


def write_container(
    path: str,
    vtm: "vtm_ref.VTM",
    *,
    program: Sequence["vtm_ref.Instr"] | None = None,
    integrity: bool | None = None,
) -> None:
    """Записать производный контейнер, не сломав блок целостности (0066).

    ``write_vtm`` с версии 0066 умеет ``integrity=``: по умолчанию блок
    пишется только при ``VTM_WRITE_HASH=1``. Если в исходной мете блок был, его
    нельзя просто скопировать — после правки KV-кэша/байткода хеш другой,
    загрузчик отвергнет контейнер. Поэтому: есть параметр — пересчитываем
    (``integrity=True``); нет (старый ``vtm_ref``) — вырезаем блок, чтобы
    контейнер остался читаемым без ложного «повреждён».
    """
    meta = dict(vtm.meta) if isinstance(vtm.meta, Mapping) else {}
    code = list(vtm.program) if program is None else list(program)
    supported = "integrity" in inspect.signature(vtm_ref.write_vtm).parameters
    inherit = INTEGRITY_KEY in meta
    if integrity is None:
        integrity = True if inherit else None
    if supported:
        if not integrity:
            # Явный отказ или контейнер без блока: старый блок не копируем.
            meta.pop(INTEGRITY_KEY, None)
        vtm_ref.write_vtm(
            path, vtm.tensors, code, meta,
            integrity=True if integrity else None,
        )
    else:
        meta.pop(INTEGRITY_KEY, None)
        vtm_ref.write_vtm(path, vtm.tensors, code, meta)


def prepare_container(
    src: str,
    context: int,
    workdir: str,
    *,
    runner: Runner | None = None,
    timeout: float | None = DEFAULT_TIMEOUT_S,
) -> str:
    """Копия контейнера с KV-кэшем ``context`` и (при необходимости) SAMP.

    Демо/синтетика имеют KV-кэш на 8 позиций, поэтому контекст 128/512/2048
    недостижим: ``KVW`` падает с «позиция 8 вне ctx=8». Копия меняет только
    ``dims`` кэшей и их нули — граф, веса и meta остаются прежними. Если
    контейнер уже подходит, возвращается исходный путь (в отчёте видно, что
    замер шёл на нём).
    """
    try:
        vtm = vtm_ref.read_vtm(src)
    except (OSError, ValueError) as exc:
        raise BenchError(f"контейнер {src}: {exc}") from exc
    ops = vtm.op_counts()
    if container_context(vtm) >= context and "SAMP" in ops:
        return src
    changed = add_samp_if_missing(vtm)
    if container_context(vtm) < context:
        caches = _cache_tensors(vtm)
        if not caches:
            raise BenchError(
                f"prepare_container: в {src} нет тензоров *cache — контекст "
                f"{context} недостижим"
            )
        for tensor in caches:
            kv_dim = int(tensor.dims[1])
            tensor.dims = (int(context), kv_dim)
            tensor.data = [0.0] * (int(context) * kv_dim)
        changed = True
    if not changed:
        return src
    target_dir = os.path.join(workdir, f"ctx{context}")
    os.makedirs(target_dir, exist_ok=True)
    name = os.path.basename(src)
    target = os.path.join(target_dir, name)
    write_container(target, vtm)
    prepared = read_container_info(target)
    if prepared.context < context:
        raise BenchError(
            f"prepare_container: после расширения контекст {prepared.context} "
            f"< {context}"
        )
    return target


def ensure_demo_model(
    path: str = DEMO_MODEL,
    *,
    python: str | None = None,
    runner: Runner | None = None,
    timeout: float | None = DEFAULT_TIMEOUT_S,
) -> str:
    """Демо-контейнер: если нет — собрать ``scripts/demo_ternary_model.py``."""
    if os.path.exists(path):
        return path
    python = python or sys.executable
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    command = [python, DEMO_SCRIPT, "--out", out_dir]
    run_cli(command, runner=runner, timeout=timeout)
    if not os.path.exists(path):
        raise BenchError(f"demo-скрипт не создал {path}")
    return path


def ensure_synthetic(
    workdir: str,
    *,
    size: int,
    seed: int = DEFAULT_SYNTHETIC_SEED,
    mode: str = DEFAULT_SYNTHETIC_MODE,
    python: str | None = None,
    runner: Runner | None = None,
    timeout: float | None = DEFAULT_TIMEOUT_S,
) -> str:
    """Собрать синтетику ``tools/vtm_compile.py --synthetic`` в workdir."""
    python = python or sys.executable
    os.makedirs(workdir, exist_ok=True)
    out = os.path.join(workdir, f"synthetic-{size}-seed{seed}-{mode}.vtm")
    if os.path.exists(out):
        return out
    command = build_synthetic_command(
        python, COMPILER, out, size=size, seed=seed, mode=mode
    )
    run_cli(command, runner=runner, timeout=timeout)
    if not os.path.exists(out):
        raise BenchError(f"компилятор не создал {out}")
    return out


def inspect_container(
    vtm_bin: str,
    model: str,
    *,
    runner: Runner | None = None,
    timeout: float | None = 60.0,
) -> dict[str, Any]:
    """``vtm inspect``: счётчики из шапки и wall-время (прокси загрузки).

    Таймер ``vtm run`` стартует после ``vtm_load``, поэтому время загрузки
    контейнера в него не входит. ``vtm inspect`` грузит контейнер и печатает
    карточку, ничего не считая, — его wall-время и есть «старт процесса +
    загрузка» (плюс печать таблицы тензоров).
    """
    command = build_inspect_command(vtm_bin, model)
    started = time.perf_counter()
    proc = run_cli(command, runner=runner, timeout=timeout)
    wall_ms = (time.perf_counter() - started) * 1000.0
    text = proc.stdout or ""
    match = _RE_COUNTERS.search(text)
    result: dict[str, Any] = {
        "wall_ms": wall_ms,
        "tensors": None,
        "instructions": None,
        "backend": None,
    }
    if match:
        result["tensors"] = int(match.group(1))
        result["instructions"] = int(match.group(2))
        result["backend"] = match.group(3)
    return result


# ---------------------------------------------------------------------------
# Сценарии
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    context: int
    gen: int
    backend: str

    @property
    def forwards(self) -> int:
        """Сколько раз вызывается ``vtm_run``: prefill + генерация."""
        return self.context + self.gen - 1

    def label(self) -> str:
        return f"ctx={self.context} gen={self.gen} {self.backend}"


def iter_scenarios(
    contexts: Sequence[int], gens: Sequence[int], backends: Sequence[str]
) -> list[Scenario]:
    """Декартово произведение контекстов, генераций и бэкендов."""
    return [
        Scenario(context=int(context), gen=int(gen), backend=str(backend))
        for context in contexts
        for gen in gens
        for backend in backends
    ]


def measure_scenario(
    prepared: str,
    scenario: Scenario,
    *,
    vtm_bin: str,
    vocab: int,
    repeat: int,
    seed: int = 7,
    runner: Runner | None = None,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    trace: bool = False,
) -> dict[str, Any]:
    """Прогнать сценарий ``repeat`` раз, вернуть медианы и производные метрики."""
    if repeat <= 0:
        raise BenchError("measure_scenario: repeat <= 0")
    tokens = token_ids(scenario.context, vocab, seed)
    command = build_run_command(
        vtm_bin,
        prepared,
        tokens,
        scenario.gen,
        backend=scenario.backend,
        trace=trace,
    )
    time_samples: list[float] = []
    wall_samples: list[float] = []
    last: RunOutput | None = None
    for _ in range(repeat):
        output, wall_ms = run_and_parse(command, runner=runner, timeout=timeout)
        if len(output.generated) != scenario.gen:
            raise BenchError(
                f"{scenario.label()}: сгенерировано {len(output.generated)} "
                f"вместо {scenario.gen} токенов (вывод: {output.generated[:8]})"
            )
        time_samples.append(output.time_ms)
        wall_samples.append(wall_ms)
        last = output
    assert last is not None
    time_stats = summarize(time_samples)
    wall_stats = summarize(wall_samples)
    # Накладные расходы считаем по парам (wall - время CL), а не как разность
    # медиан: так шум запуска не складывается с шумом самих замеров.
    overheads = [wall - measured for wall, measured in zip(wall_samples, time_samples)]
    forwards = scenario.forwards
    median_ms = time_stats["median"]
    return {
        "context": scenario.context,
        "gen": scenario.gen,
        "backend_requested": scenario.backend,
        "backend": last.backend or scenario.backend,
        "forwards": forwards,
        "generated": scenario.gen,
        "time_ms": time_stats,
        "wall_ms": wall_stats,
        "overhead_ms": statistics.median(overheads),
        "overhead_min_ms": min(overheads),
        "ms_per_token": median_ms / scenario.gen,
        "tokens_per_s": (scenario.gen * 1000.0 / median_ms) if median_ms > 0 else None,
        "ms_per_step": median_ms / forwards if forwards > 0 else None,
        "samples_ms": time_samples,
        "samples_wall_ms": wall_samples,
    }


# ---------------------------------------------------------------------------
# Разбивка по операциям
# ---------------------------------------------------------------------------


def duplicate_instructions(
    program: Sequence["vtm_ref.Instr"],
    op_names: Iterable[str],
    copies: int,
) -> list["vtm_ref.Instr"]:
    """Вернуть программу, где инструкции классов ``op_names`` повторены.

    ``copies`` дополнительных копий вставляются сразу после оригинала; порядок
    и все поля сохраняются. Так класс операции получает ``copies`` лишних
    проходов на каждый ``vtm_run`` — разница времени и есть его вклад
    (в пределах шума и кэш-эффектов).
    """
    if copies < 0:
        raise BenchError("duplicate_instructions: copies < 0")
    wanted = {str(name) for name in op_names}
    result: list["vtm_ref.Instr"] = []
    for instr in program:
        result.append(instr)
        name = vtm_ref.OP_NAMES.get(int(instr.op))
        if name in wanted:
            result.extend([instr] * copies)
    return result


def parse_trace_output(text: str) -> dict[str, float] | None:
    """Разобрать ``--trace``/``--trace-summary`` в ``{OP: мс}``.

    Формат 0082: строка начинается с имени opcode, время — последнее число
    с единицей (``TMV pc=3 seq=12 out=128 dur=3500.0 ns``; ``TMV 24 0.129000 ms
    94.4%``). Разбор при этом терпимый, потому что старые/сторонние сборки
    могут писать иначе: сначала JSON (``{"TMV": ms}``, ``{"ops": {...}}`` или
    список ``{"op","ms"}``), затем строки вида ``TMV ... 12.3 мс`` /
    ``ATTN: 4.5 ms``. Ничего не разобралось — ``None``: лучше честно откатиться
    на оценку, чем показывать выдуманные числа.
    """
    if not text or not text.strip():
        return None
    stripped = text.strip()
    if stripped[:1] in "{[":
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            parsed = _trace_from_json(payload)
            if parsed:
                return parsed
    # Строка trace: имя операции в начале, число с единицей времени — последнее
    # в строке. Между ними может быть счётчик («TMV  8 x 3.500 ms»), поэтому
    # середина не ограничена, а число берётся непосредственно перед единицей.
    pattern = re.compile(
        r"^\s*(?:\[[^\]]*\]\s*)?(?P<op>[A-Z]{2,6})\b"
        r"[^\n]*?(?P<value>[-+]?\d+(?:[.,]\d+)?)\s*"
        r"(?P<unit>мс|ms|µs|us|нс|ns|с|s)\b"
    )
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        op = match.group("op")
        if op not in vtm_ref.OP_NAMES.values() or op in NON_CLUSTER_OPS:
            continue
        unit = match.group("unit")
        value = _to_float(match.group("value"))
        if unit in ("µs", "us"):
            value /= 1000.0
        elif unit in ("нс", "ns"):
            value /= 1_000_000.0
        elif unit in ("с", "s"):
            value *= 1000.0
        totals[op] = totals.get(op, 0.0) + value
    return totals or None


def _trace_from_json(payload: Any) -> dict[str, float] | None:
    """Достать ``{OP: мс}`` из возможных JSON-форм trace."""
    totals: dict[str, float] = {}

    def take(name: Any, value: Any) -> None:
        if not isinstance(name, str) or name.upper() not in vtm_ref.OP_NAMES.values():
            return
        if name.upper() in NON_CLUSTER_OPS:
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            totals[name.upper()] = totals.get(name.upper(), 0.0) + float(value)

    if isinstance(payload, Mapping):
        ops = payload.get("ops")
        if isinstance(ops, Mapping):
            payload = ops
        for key, value in payload.items():
            take(key, value)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                name = item.get("op") or item.get("name")
                value = item.get("ms", item.get("time_ms"))
                if value is None:
                    value = item.get("us")
                    if isinstance(value, (int, float)):
                        value = float(value) / 1000.0
                take(name, value)
    return totals or None


def measure_breakdown(
    prepared: str,
    *,
    context: int,
    gen: int,
    backend: str,
    vtm_bin: str,
    info: ContainerInfo,
    copies: int,
    repeat: int,
    workdir: str,
    runner: Runner | None = None,
    timeout: float | None = DEFAULT_TIMEOUT_S,
    seed: int = 7,
    trace: bool = False,
) -> dict[str, Any]:
    """Вклад классов операций в полный проход (оценка дублированием)."""
    scenario = Scenario(context=context, gen=gen, backend=backend)
    base = measure_scenario(
        prepared,
        scenario,
        vtm_bin=vtm_bin,
        vocab=info.vocab,
        repeat=repeat,
        seed=seed,
        runner=runner,
        timeout=timeout,
        trace=trace,
    )
    base_ms = base["time_ms"]["median"]
    forwards = base["forwards"]
    vtm = vtm_ref.read_vtm(prepared)
    rows: list[dict[str, Any]] = []
    covered_ms = 0.0
    absent: list[str] = []
    for cluster, op_names in CLUSTERS:
        count = sum(info.ops.get(name, 0) for name in op_names)
        if count == 0:
            absent.append(cluster)
            continue
        variant = duplicate_instructions(vtm.program, op_names, copies)
        variant_path = os.path.join(
            workdir, f"dup-{cluster}-{context}-{gen}-{backend}.vtm"
        )
        write_container(variant_path, vtm, program=variant)
        variant_scenario = Scenario(context=context, gen=gen, backend=backend)
        measured = measure_scenario(
            variant_path,
            variant_scenario,
            vtm_bin=vtm_bin,
            vocab=info.vocab,
            repeat=repeat,
            seed=seed,
            runner=runner,
            timeout=timeout,
            trace=trace,
        )
        variant_ms = measured["time_ms"]["median"]
        total_ms = (variant_ms - base_ms) / copies
        covered_ms += total_ms
        rows.append(
            {
                "cluster": cluster,
                "ops": sorted(op_names),
                "instructions": count,
                "total_ms": total_ms,
                "ms_per_step": total_ms / forwards if forwards else None,
                "share_pct": (total_ms / base_ms * 100.0) if base_ms > 0 else None,
                "base_time_ms": base_ms,
                "variant_time_ms": variant_ms,
                "copies": copies,
                # Отрицательный вклад = копии не дороже шума часов (KVW/SAMP
                # на маленьком словаре). В JSON остаётся сырое число, в
                # таблице показывается «≈0».
                "within_noise": total_ms <= 0.0,
            }
        )
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    residual_ms = base_ms - covered_ms
    return {
        "method": "duplication",
        "note": BREAKDOWN_NOTE,
        "scenario": {
            "context": context,
            "gen": gen,
            "backend": base["backend"],
            "forwards": forwards,
            "generated": gen,
        },
        "base_time_ms": base_ms,
        "base_ms_per_step": base["ms_per_step"],
        "copies": copies,
        "repeat": repeat,
        "rows": rows,
        "absent_clusters": absent,
        "covered_ms": covered_ms,
        "residual_ms": residual_ms,
        "residual_share_pct": (residual_ms / base_ms * 100.0) if base_ms > 0 else None,
    }


def breakdown_from_trace(
    trace_measured: Mapping[str, Any],
    *,
    context: int,
    gen: int,
    backend: str,
    forwards: int,
) -> list[dict[str, Any]]:
    """Строки разбивки из trace: ``{OP: мс}`` -> классы и доли.

    Доля считается от суммы измеренных времён инструкций, поэтому доли
    складываются в 100%. Насколько эта сумма покрывает базу без трассировки
    (цикл диспетчеризации, замедление от хука) — поле ``covered_pct_of_base``
    у элемента разбивки.
    """
    totals = {str(name).upper(): float(value) for name, value in trace_measured.items()}
    total_ms = sum(totals.values())
    rows: list[dict[str, Any]] = []
    for cluster, op_names in CLUSTERS:
        value = sum(totals.get(name, 0.0) for name in op_names)
        if value <= 0:
            continue
        rows.append(
            {
                "cluster": cluster,
                "ops": sorted(op_names),
                "instructions": None,
                "total_ms": value,
                "ms_per_step": value / forwards if forwards else None,
                "share_pct": (value / total_ms * 100.0) if total_ms > 0 else None,
                "copies": 0,
            }
        )
    rows.sort(key=lambda row: row["total_ms"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Отчёт и таблицы
# ---------------------------------------------------------------------------


def _ms(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1000.0:
        return f"{value:.0f}"
    if abs(value) >= 100.0:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def _num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}%"


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    aligns: Sequence[str] | None = None,
) -> str:
    """Текстовая таблица: ширина по содержимому, выравнивание по колонкам."""
    header_cells = [str(cell) for cell in headers]
    body = [[str(cell) for cell in row] for row in rows]
    columns = len(header_cells)
    for row in body:
        if len(row) != columns:
            raise BenchError(
                f"format_table: в строке {len(row)} колонок вместо {columns}"
            )
    if aligns is None:
        aligns = ["<"] * columns
    widths = [len(header_cells[index]) for index in range(columns)]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = [
        "  ".join(
            f"{header_cells[index]:{aligns[index]}{widths[index]}}"
            for index in range(columns)
        ).rstrip(),
        "  ".join("-" * widths[index] for index in range(columns)),
    ]
    for row in body:
        lines.append(
            "  ".join(
                f"{row[index]:{aligns[index]}{widths[index]}}"
                for index in range(columns)
            ).rstrip()
        )
    return "\n".join(lines)


SCENARIO_HEADERS = (
    "контекст",
    "ген",
    "бэкенд",
    "шагов",
    "время, мс",
    "мс/ток",
    "ток/с",
    "мс/шаг",
    "wall, мс",
    "вне CL, мс",
    "бюджет",
)
SCENARIO_ALIGNS = ("<", ">", "<", ">", ">", ">", ">", ">", ">", ">", "<")


def scenario_table(rows: Sequence[Mapping[str, Any]], budget: float | None) -> str:
    """Таблица сценариев. «вне CL» = wall - время CLI (старт + загрузка)."""
    cells: list[list[str]] = []
    for row in rows:
        if budget is None or row.get("budget_ok") is None:
            budget_cell = "—"
        else:
            budget_cell = "ok" if row["budget_ok"] else "ПРЕВЫШЕН"
        cells.append(
            [
                str(row["context"]),
                str(row["gen"]),
                str(row.get("backend") or row.get("backend_requested") or "?"),
                str(row["forwards"]),
                _ms(row["time_ms"]["median"]),
                _ms(row["ms_per_token"]),
                _num(row["tokens_per_s"]),
                _ms(row["ms_per_step"]),
                _ms(row["wall_ms"]["median"]),
                _ms(row["overhead_ms"]),
                budget_cell,
            ]
        )
    return format_table(SCENARIO_HEADERS, cells, aligns=SCENARIO_ALIGNS)


BREAKDOWN_HEADERS = ("операция", "инстр.", "всего, мс", "мс/шаг", "доля")
BREAKDOWN_ALIGNS = ("<", ">", ">", ">", ">")


def breakdown_table(rows: Sequence[Mapping[str, Any]], residual: float | None) -> str:
    """Таблица разбивки: класс операции -> мс за прогон, мс/шаг, доля."""
    cells: list[list[str]] = []
    for row in rows:
        noise = bool(row.get("within_noise"))
        cells.append(
            [
                str(row["cluster"]),
                "—" if row.get("instructions") is None else str(row["instructions"]),
                "≈0" if noise else _ms(row.get("total_ms")),
                "≈0" if noise else _ms(row.get("ms_per_step")),
                "≈0" if noise else _pct(row.get("share_pct")),
            ]
        )
    if residual is not None:
        cells.append(["остаток", "—", _ms(residual), "—", "—"])
    return format_table(BREAKDOWN_HEADERS, cells, aligns=BREAKDOWN_ALIGNS)


def _cluster_verdict(breakdown_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Самый дорогой класс — кандидат на оптимизацию №1."""
    if not breakdown_rows:
        return None
    top = max(breakdown_rows, key=lambda row: row.get("total_ms") or 0.0)
    return {
        "cluster": top["cluster"],
        "share_pct": top.get("share_pct"),
        "ms_per_step": top.get("ms_per_step"),
        "total_ms": top.get("total_ms"),
    }


#: Классы, которые вообще зависят от бэкенда (остальные — скалярные op'ы).
BACKEND_SENSITIVE_CLUSTERS = ("TMV", "MM", "RMSN")


def _crosscheck_breakdown(
    breakdown: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Сверка разбивки с разницей scalar-simd.

    Разница времени scalar и simd — это ровно цена TMV+MM+RMSN (остальные
    операции в обоих бэкендах скалярные). Если сумма этих классов из разбивки
    совпадает с разницей, оценка дублированием непротиворечива.
    """
    by_context: dict[int, dict[str, Mapping[str, Any]]] = {}
    for item in breakdown:
        scenario = item.get("scenario") or {}
        context = scenario.get("context")
        backend = scenario.get("backend") or ""
        if context is None or item.get("method") != "duplication":
            continue
        by_context.setdefault(int(context), {})[str(backend)] = item
    for context in sorted(by_context, reverse=True):
        entries = by_context[context]
        slow = next((item for name, item in entries.items() if "scalar" in name), None)
        fast = next((item for name, item in entries.items() if "simd" in name), None)
        if slow is None or fast is None:
            continue
        slow_step = slow["base_time_ms"] / slow["scenario"]["forwards"]
        fast_step = fast["base_time_ms"] / fast["scenario"]["forwards"]
        delta = slow_step - fast_step

        def claimed(item: Mapping[str, Any]) -> float:
            total = 0.0
            for row in item.get("rows", []):
                if row.get("cluster") in BACKEND_SENSITIVE_CLUSTERS:
                    total += row.get("ms_per_step") or 0.0
            return total

        estimate = claimed(slow) - claimed(fast)
        return {
            "context": context,
            "scalar_ms_per_step": slow_step,
            "simd_ms_per_step": fast_step,
            "backend_diff_ms_per_step": delta,
            "breakdown_delta_ms_per_step": estimate,
            "relative_diff_pct": (
                abs(estimate - delta) / delta * 100.0 if delta else None
            ),
            "note": "разница scalar-simd = цена TMV+MM+RMSN; сверка с разбивкой",
        }
    return None


def build_report(
    *,
    model: ContainerInfo,
    source_model: str,
    cli: Mapping[str, Any],
    config: Mapping[str, Any],
    scenarios: Sequence[Mapping[str, Any]],
    breakdown: Sequence[Mapping[str, Any]],
    load: Mapping[str, Any] | None,
    budget_ms_token: float | None,
    notes: Sequence[str],
) -> dict[str, Any]:
    """Собрать машинный отчёт (стабильная схема, version=1)."""
    best = None
    if scenarios:
        best = min(scenarios, key=lambda row: row.get("ms_per_step") or float("inf"))
    verdict = None
    verdicts: list[dict[str, Any]] = []
    # Кандидат №1 — самый дорогой класс на максимальном контексте: на длинном
    # контексте доли меняются (attention растёт), и именно там больно. Считаем
    # для каждого бэкенда: у scalar и simd «первое место» может различаться.
    usable = [
        item
        for item in breakdown
        if item.get("scenario") and item.get("rows")
    ]
    if usable:
        max_context = max(int(item["scenario"].get("context") or 0) for item in usable)
        for item in usable:
            if int(item["scenario"].get("context") or 0) != max_context:
                continue
            candidate = _cluster_verdict(item["rows"])
            if candidate is None:
                continue
            candidate = dict(candidate)
            candidate["context"] = item["scenario"].get("context")
            candidate["backend"] = item["scenario"].get("backend")
            candidate["base_ms_per_step"] = item.get("base_ms_per_step")
            verdicts.append(candidate)
        verdicts.sort(key=lambda entry: entry.get("base_ms_per_step") or float("inf"))
        if verdicts:
            verdict = verdicts[0]
    budget_ok: bool | None = None
    exceeded = [row for row in scenarios if row.get("budget_ok") is False]
    if budget_ms_token is not None:
        budget_ok = not exceeded
    return {
        "tool": "bench_vtm",
        "version": REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "model": {**model.as_json(), "source": source_model},
        "cli": dict(cli),
        "config": dict(config),
        "scenarios": list(scenarios),
        "breakdown": list(breakdown),
        "crosscheck": _crosscheck_breakdown(breakdown),
        "load": dict(load or {}),
        "budget_ms_token": budget_ms_token,
        # Бюджет сравнивается с ценой одного прохода (мс/шаг), а не с
        # «мс/ток» из CLI: у «мс/ток» при gen=1 весь prefill в одном токене.
        "budget_metric": "ms_per_step" if budget_ms_token is not None else None,
        "budget_ok": budget_ok,
        "best_scenario": best,
        "verdict": verdict,
        "verdicts": verdicts,
        "notes": list(notes),
    }


def format_report(report: Mapping[str, Any]) -> str:
    """Человекочитаемый отчёт: карточка модели, таблицы, вывод и бюджет."""
    model = report["model"]
    cli = report["cli"]
    budget = report.get("budget_ms_token")
    lines: list[str] = []
    lines.append("VTM: профиль полного прохода")
    lines.append(
        f"  модель   : {model['path']} ({model['file_bytes']:,} Б, "
        f"{model['tensors']} тензоров, {model['instructions']} инструкций, "
        f"ctx={model['context']}, vocab={model['vocab']})"
    )
    if model.get("source") and model["source"] != model["path"]:
        lines.append(f"  исходник : {model['source']} (контекст расширен для замеров)")
    ops = ", ".join(f"{name}×{count}" for name, count in model["ops"].items())
    lines.append(f"  операции : {ops}")
    lines.append(
        f"  vtm      : {cli.get('vtm_bin')}  "
        f"--trace: {'есть' if cli.get('trace') else 'нет'}, "
        f"--trace-summary: {'есть' if cli.get('trace_summary') else 'нет'}"
    )
    host = report.get("host", {})
    lines.append(
        f"  стенд    : {host.get('machine')}, {host.get('platform')}, "
        f"python {host.get('python')}, cpu {host.get('cpu_count')}"
    )
    config = report["config"]
    lines.append(
        f"  повторов : {config.get('repeat')}, генерация: "
        f"{_join(config.get('gen', []))}, контексты: "
        f"{_join(config.get('contexts', []))}"
    )
    load = report.get("load") or {}
    if load.get("inspect_wall_ms") is not None:
        lines.append(
            f"  загрузка : vtm inspect ≈ {_ms(load['inspect_wall_ms'])} мс wall "
            "(процесс + контейнер, вне таймера CLI)"
        )
    lines.append("")
    lines.append("Сценарии (медиана повторов):")
    lines.append(scenario_table(report["scenarios"], budget))
    lines.append(
        "  мс/ток = время/сгенерированные (включая prefill), "
        "мс/шаг = время/вызовы vtm_run — цена одного прохода"
    )
    lines.append("")
    for item in report["breakdown"]:
        scenario = item["scenario"]
        # У trace-разбивки method — реальный замер; остальное — оценка.
        method = item["method"] if item["method"] == "trace" else item["method"] + " (оценка)"
        lines.append(
            f"Разбивка, ctx={scenario['context']} gen={scenario['gen']} "
            f"{scenario['backend']}: метод — {method}"
        )
        lines.append(breakdown_table(item["rows"], item.get("residual_ms")))
        lines.append(f"  {item['note']}")
        coverage = item.get("covered_pct_of_base")
        lines.append(
            f"  покрыто: {_ms(item.get('covered_ms'))} мс из "
            f"{_ms(item.get('base_time_ms'))} мс"
            + (f" ({coverage:.1f}% базы)" if coverage is not None else "")
            + f", остаток {_pct(item.get('residual_share_pct'))}"
        )
        lines.append("")
    crosscheck = report.get("crosscheck")
    if crosscheck:
        lines.append(
            "сверка: разница scalar-simd "
            f"{_ms(crosscheck['backend_diff_ms_per_step'])} мс/шаг, "
            "сумма TMV+MM+RMSN из разбивки "
            f"{_ms(crosscheck['breakdown_delta_ms_per_step'])} мс/шаг "
            f"(расхождение {_pct(crosscheck['relative_diff_pct'])}, "
            f"ctx={crosscheck['context']})"
        )
        lines.append("")
    if not report["breakdown"]:
        lines.append("Разбивка отключена (--no-breakdown).")
        lines.append("")
    for note in report.get("notes", []):
        lines.append(f"примечание: {note}")
    verdicts = report.get("verdicts") or []
    for index, verdict in enumerate(verdicts):
        prefix = "оптимизировать первым: " if index == 0 else "                     "
        lines.append(
            f"{prefix}{verdict['cluster']} — {_pct(verdict.get('share_pct'))} "
            f"времени ({_ms(verdict.get('ms_per_step'))} мс/шаг на ctx="
            f"{verdict.get('context')}, {verdict.get('backend')})"
        )
    if budget is None:
        lines.append(
            "бюджет: не задан — сравнение не выполняется (--budget-ms-token N)"
        )
    else:
        lines.append(
            f"бюджет: {float(budget):.3f} мс/шаг — "
            + ("ok" if report.get("budget_ok") else "ПРЕВЫШЕН")
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _resolve_model(
    args: argparse.Namespace,
    workdir: str,
    *,
    runner: Runner | None,
) -> tuple[str, bool]:
    """Модель для замеров и флаг «это синтетика».

    Приоритет: ``--synthetic-size`` (сборка компилятором) -> ``--model`` ->
    demo (с генерацией скриптом, если файла нет).
    """
    if args.synthetic_size is not None:
        if args.synthetic_size <= 0 or args.synthetic_size % 8:
            raise BenchError("--synthetic-size должен быть > 0 и кратен 8")
        path = ensure_synthetic(
            workdir,
            size=args.synthetic_size,
            seed=args.synthetic_seed,
            mode=args.synthetic_mode,
            runner=runner,
            timeout=args.timeout,
        )
        return path, True
    if args.model:
        path = os.path.abspath(args.model)
        if not os.path.exists(path):
            raise BenchError(f"контейнер не найден: {path}")
        return path, False
    return ensure_demo_model(runner=runner, timeout=args.timeout), False


def _prepare_workdir(args: argparse.Namespace) -> tuple[str, bool]:
    if args.workdir:
        os.makedirs(args.workdir, exist_ok=True)
        return os.path.abspath(args.workdir), False
    return tempfile.mkdtemp(prefix="vtm-bench-"), True


def main(argv: Sequence[str] | None = None, *, runner: Runner | None = None) -> int:
    """Точка входа CLI. ``runner`` — для тестов (подмена subprocess)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    workdir = None
    temporary = False
    try:
        workdir, temporary = _prepare_workdir(args)

        vtm_bin = args.vtm
        if not os.path.exists(vtm_bin):
            raise BenchError(
                f"нет бинарника {vtm_bin}: соберите `make -C src/vtm` "
                "или укажите --vtm"
            )
        if not os.access(vtm_bin, os.X_OK):
            raise BenchError(f"{vtm_bin} не исполняемый")

        try:
            backends = _backend_list_arg(args.backends)
        except argparse.ArgumentTypeError as exc:
            raise BenchError(str(exc)) from exc
        contexts = list(args.contexts)
        gens = list(args.gen)
        if args.repeat <= 0:
            raise BenchError("--repeat должен быть > 0")
        if args.breakdown_gen <= 0:
            raise BenchError("--breakdown-gen должен быть > 0")
        if args.dup_copies <= 0:
            raise BenchError("--dup-copies должен быть > 0")
        if args.budget_ms_token is not None and args.budget_ms_token <= 0:
            raise BenchError("--budget-ms-token должен быть > 0")

        source_model, synthetic = _resolve_model(args, workdir, runner=runner)
        info = read_container_info(source_model)
        # Длина промпта — это контекст сценария, но генерация пишет ещё
        # gen-1 позиций в KV-кэш, поэтому ёмкость кэша = context + gen - 1.
        # Берём максимум по сценариям и по разбивке, иначе замер разбивки с
        # большей генерацией упрётся в «позиция вне ctx».
        def capacity(context: int, gen: int) -> int:
            return int(context) + int(gen) - 1

        max_gen = max(gens)
        if not args.no_breakdown:
            max_gen = max(max_gen, args.breakdown_gen)

        prepared_by_context: dict[int, str] = {}
        for context in contexts:
            prepared_by_context[context] = prepare_container(
                source_model,
                capacity(context, max_gen),
                workdir,
                runner=runner,
                timeout=args.timeout,
            )
        # Готовим контейнеры и для разбивки: её контексты могут не входить в
        # сценарную матрицу.
        breakdown_contexts = list(
            args.breakdown_contexts
            or [context for context in (128, 512) if context in contexts]
            or contexts[:1]
        )
        for context in breakdown_contexts:
            if context not in prepared_by_context:
                prepared_by_context[context] = prepare_container(
                    source_model,
                    capacity(context, max_gen),
                    workdir,
                    runner=runner,
                    timeout=args.timeout,
                )

        detected_trace = args.trace != "off" and detect_trace(vtm_bin, runner=runner)
        if args.trace == "on" and not detected_trace:
            raise BenchError(
                "--trace on, но эта сборка vtm не умеет --trace "
                "(проверено через vtm --help); используйте --trace auto/off"
            )
        trace_available = args.trace == "on" or (args.trace == "auto" and detected_trace)
        # Сводка (--trace-summary, задача 0082) — предпочтительный источник:
        # интерпретатор суммирует сам, вывод не растёт с числом инструкций.
        summary_available = trace_available and detect_trace_summary(
            vtm_bin, runner=runner
        )

        notes: list[str] = []
        if synthetic:
            notes.append(
                f"синтетика vtm_compile.py --synthetic --size {args.synthetic_size} "
                f"--seed {args.synthetic_seed} --mode {args.synthetic_mode}: "
                "1 слой, доля attention выше, чем у Qwen-1.5B"
            )
        if args.no_breakdown:
            notes.append("разбивка отключена (--no-breakdown): таблица без долей")
        elif summary_available:
            notes.append(
                "разбивка — реальный замер: vtm run --trace-summary, сумма "
                "длительностей инструкций по опкодам (CLOCK_MONOTONIC), "
                "не оценка; оценка дублированием остаётся для сверки"
            )
        elif trace_available:
            notes.append(
                "разбивка — реальный замер: vtm run --trace, длительности "
                "инструкций по опкодам (оценка дублированием — для сверки)"
            )
        else:
            notes.append(
                "в этой сборке vtm нет --trace (проверено через vtm --help): "
                "разбивка — оценка дублированием инструкций"
            )
        largest_capacity = max(
            capacity(context, max_gen) for context in set(contexts) | set(breakdown_contexts)
        )
        if info.context < largest_capacity:
            notes.append(
                f"контейнер имеет ctx={info.context}; для замеров собраны копии "
                f"с KV-кэшем до {largest_capacity} "
                "(промпт + генерация; граф и веса те же)"
            )

        load = None
        try:
            inspected = inspect_container(
                vtm_bin, prepared_by_context[contexts[0]], runner=runner
            )
            if inspected.get("tensors") is not None:
                load = {
                    "inspect_wall_ms": inspected["wall_ms"],
                    "tensors": inspected["tensors"],
                    "instructions": inspected["instructions"],
                    "backend": inspected["backend"],
                    "model": prepared_by_context[contexts[0]],
                }
        except BenchError:
            load = None

        scenarios: list[dict[str, Any]] = []
        for scenario in iter_scenarios(contexts, gens, backends):
            prepared = prepared_by_context[scenario.context]
            measured = measure_scenario(
                prepared,
                scenario,
                vtm_bin=vtm_bin,
                vocab=info.vocab,
                repeat=args.repeat,
                seed=args.seed,
                runner=runner,
                timeout=args.timeout,
                # Трасса замедляет прогон, поэтому тайминги сценариев снимаются
                # без неё: доли берёт отдельный прогон с --trace-summary.
                trace=False,
            )
            if args.budget_ms_token is not None:
                measured["budget_ok"] = (
                    measured["ms_per_step"] is not None
                    and measured["ms_per_step"] <= args.budget_ms_token
                )
            else:
                measured["budget_ok"] = None
            scenarios.append(measured)

        breakdown: list[dict[str, Any]] = []
        if not args.no_breakdown:
            if args.breakdown_backends:
                try:
                    breakdown_backends = _backend_list_arg(args.breakdown_backends)
                except argparse.ArgumentTypeError as exc:
                    raise BenchError(str(exc)) from exc
            else:
                breakdown_backends = backends
            for context in breakdown_contexts:
                prepared = prepared_by_context[context]
                for backend in breakdown_backends:
                    item = measure_breakdown(
                        prepared,
                        context=context,
                        gen=args.breakdown_gen,
                        backend=backend,
                        vtm_bin=vtm_bin,
                        info=info,
                        copies=args.dup_copies,
                        repeat=args.breakdown_repeat,
                        workdir=workdir,
                        runner=runner,
                        timeout=args.timeout,
                        seed=args.seed,
                        trace=False,
                    )
                    if trace_available:
                        item = _try_trace_breakdown(
                            item,
                            prepared=prepared,
                            context=context,
                            gen=args.breakdown_gen,
                            backend=backend,
                            vtm_bin=vtm_bin,
                            vocab=info.vocab,
                            repeat=args.breakdown_repeat,
                            seed=args.seed,
                            runner=runner,
                            timeout=args.timeout,
                            summary=summary_available,
                        )
                    breakdown.append(item)

        report = build_report(
            model=info,
            source_model=source_model,
            cli={
                "vtm_bin": vtm_bin,
                "trace": trace_available,
                "trace_summary": summary_available,
                "trace_mode": args.trace,
            },
            config={
                "contexts": contexts,
                "gen": gens,
                "backends": backends,
                "repeat": args.repeat,
                "seed": args.seed,
                "breakdown_contexts": breakdown_contexts,
                "breakdown_gen": args.breakdown_gen,
                "breakdown_backends": args.breakdown_backends or backends,
                "dup_copies": args.dup_copies,
                "breakdown_repeat": args.breakdown_repeat,
                "budget_ms_token": args.budget_ms_token,
                "synthetic": (
                    {
                        "size": args.synthetic_size,
                        "seed": args.synthetic_seed,
                        "mode": args.synthetic_mode,
                    }
                    if synthetic
                    else None
                ),
            },
            scenarios=scenarios,
            breakdown=breakdown,
            load=load,
            budget_ms_token=args.budget_ms_token,
            notes=notes,
        )
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.json_out:
            try:
                with open(args.json_out, "w", encoding="utf-8") as handle:
                    handle.write(text + "\n")
            except OSError as exc:
                raise BenchError(
                    f"не удалось записать --json-out {args.json_out}: {exc}"
                ) from exc
        if args.json:
            print(text)
        else:
            print(format_report(report))
        if args.budget_ms_token is not None and report["budget_ok"] is False:
            return 1
        return 0
    except BenchError as exc:
        print(f"bench_vtm: {exc}", file=sys.stderr)
        return 2
    finally:
        if temporary and workdir and not args.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def _try_trace_breakdown(
    item: Mapping[str, Any],
    *,
    prepared: str,
    context: int,
    gen: int,
    backend: str,
    vtm_bin: str,
    vocab: int,
    repeat: int,
    seed: int,
    runner: Runner | None,
    timeout: float | None,
    summary: bool = False,
) -> dict[str, Any]:
    """Пересчитать разбивку из trace, если его вывод действительно разобрался.

    ``summary`` — сборка умеет ``--trace-summary`` (0082): тогда берётся
    сводка, иначе построчный ``--trace``. В обоих случаях это реальные
    длительности инструкций, а не оценка.
    """
    trace = None
    mode = "summary" if summary else "lines"
    try:
        tokens = token_ids(context, vocab, seed)
        command = build_run_command(
            vtm_bin,
            prepared,
            tokens,
            gen,
            backend=backend,
            trace=not summary,
            trace_summary=summary,
        )
        collected: list[dict[str, float]] = []
        for _ in range(repeat):
            proc = run_cli(command, runner=runner, timeout=timeout)
            # --trace/--trace-summary печатаются в stderr (диагностика), но
            # старые/сторонние сборки могли писать в stdout — смотрим оба.
            parsed = parse_trace_output(proc.stdout) or parse_trace_output(
                proc.stderr or ""
            )
            if parsed:
                collected.append(parsed)
        if collected:
            merged: dict[str, float] = {}
            for parsed in collected:
                for name, value in parsed.items():
                    merged[name] = merged.get(name, 0.0) + value
            for name in merged:
                merged[name] /= len(collected)
            trace = merged
    except BenchError:
        trace = None
    # Trace может печатать среднее на вызов, а не сумму за прогон. Признак
    # суммы — его итог близок к времени CLI (в разы); иначе честнее откатиться.
    base_ms = item.get("base_time_ms")
    if trace and base_ms:
        trace_total = sum(trace.values())
        if trace_total <= 0 or not (0.3 * base_ms <= trace_total <= 3.0 * base_ms):
            trace = None
    if not trace:
        result = dict(item)
        result["trace"] = {"available": True, "parsed": False, "mode": mode}
        result["note"] = (
            item.get("note", BREAKDOWN_NOTE)
            + "; трассировка есть, но её вывод не разобран или не сходится с "
            "временем CLI — оставлена оценка дублированием"
        )
        return result
    forwards = context + gen - 1
    # Доли — от измеренной суммы времён инструкций (складываются в 100%).
    # Сколько эта сумма покрывает от базы без трассировки, видно отдельно:
    # расхождение — цикл диспетчеризации плюс замедление от самого хука.
    rows = breakdown_from_trace(
        trace,
        context=context,
        gen=gen,
        backend=backend,
        forwards=forwards,
    )
    covered_ms = sum(trace.values())
    covered_pct = (covered_ms / float(base_ms) * 100.0) if base_ms else None
    result = dict(item)
    result.update(
        {
            "method": "trace",
            "method_estimation": False,
            "note": TRACE_NOTE + (" (--trace-summary)" if summary else " (--trace)"),
            "rows": rows,
            "trace_totals_ms": trace,
            "trace": {"available": True, "parsed": True, "mode": mode},
            "covered_ms": covered_ms,
            "covered_pct_of_base": covered_pct,
            "residual_ms": 0.0,
            "residual_share_pct": 0.0,
            "dup_estimate": {
                "base_time_ms": item.get("base_time_ms"),
                "rows": item.get("rows"),
            },
        }
    )
    return result


if __name__ == "__main__":
    sys.exit(main())
