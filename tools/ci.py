#!/usr/bin/env python3
"""Единая проверка репозитория EVA32 — локально и в GitHub Actions (задача 0102).

Зачем: до этой задачи проверки запускались вручную и по одной (`make test`,
`python3 -m unittest`, `make -C src/vtm test`, ...), а регрессии ловились
глазами — за сессию чистый клон был красным трижды.

Порядок шагов (он важен):

  1. C: `make -C src/vtm test` — сборка с нуля и C-тесты. Сначала чистим
     артефакты, чтобы проверка ловила ошибки сборки, а не жила на старых .o.
     C идёт до Python: часть Python-тестов компилирует пробники против
     `src/vtm/libvtm.a` и запускает `src/vtm/vtm`.
  2. Python: `python3 -m unittest discover -s tests` — весь набор.
  3. Скорость: perf smoke (задача 0117) — короткий замер `vtm run` на
     `demo/model.vtm` (5 прогонов по 2048 токенов, ~0,5 с), медиана токенов/с
     против `tests/fixtures/perf_baseline.json`, допуск ±25 %. Это защита от
     регрессий скорости: корректность ловят шаги 1–2, а «стало втрое
     медленнее» до сих пор замечали только руками.
     Эталон хранит дату, машину (ОС/архитектура/CPU), версию бенчмарка, sha256
     модели и протокол замера. Если на этой машине эталона нет (другая ОС,
     архитектура или CPU), шаг **скипается с предупреждением**, а не краснеет:
     чужой процессор — не регрессия. Красный — только когда эталон для этой
     машины есть, а замер вылез за допуск (медленнее или быстрее — в обе
     стороны: «быстрее» значит, что эталон устарел).
  4. Реестр задач: `python3 tools/check_tasks.py`.
  5. Документация VTM: `python3 tools/check_vtm_docs.py`.
  6. Справочник команд: `python3 tools/check_commands_doc.py` — страница
     `docs/commands.md` существует, дата на месте, упомянуты все плагины из
     `config/plugins.json` и ключевые фразы, страница есть в индексе `docs/`.
  7. Чистота: `make -C src/vtm clean` и ещё раз Python-набор. Ни один тест не
     должен требовать собранных артефактов: на чистом клоне набор обязан
     проходить (пусть и со скипами), а не падать.

Только стандартная библиотека, без сети и установки пакетов.

Запуск:
    python3 tools/ci.py                 # все проверки
    python3 tools/ci.py -v              # показывать полный вывод команд
    python3 tools/ci.py --skip-c        # без C (нет make)
    python3 tools/ci.py --skip-perf     # без замера скорости
    python3 tools/ci.py --no-artifact-check   # без шага 6
    python3 tools/ci.py --record-perf-baseline  # перезаписать эталон скорости

Перезапись эталона (после осознанного изменения производительности или смены
машины) — только на незагруженной машине и только вместе с осознанным
изменением кода; шаг записи печатает медиану и разброс.

Код выхода: 0 — всё зелёное, 1 — есть упавшая проверка, 130 — Ctrl-C.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VTM_DIR = ROOT / "src" / "vtm"
VTM_MAKEFILE = VTM_DIR / "Makefile"

# Артефакты `make -C src/vtm test` (см. src/vtm/Makefile). После clean их быть
# не должно — иначе проверка на чистом клоне ничего не проверяет.
C_ARTIFACTS = (
    "libvtm.a",
    "vtm",
    "test_vtm",
    "test_vtm_api",
    "test_vtm_session",
    "test_sha256",
    "bench_backend",
)

PYTHON = sys.executable or "python3"

# --- perf smoke (задача 0117) ---------------------------------------------
#: Эталон скорости: дата, машина, версия бенчмарка, sha256 модели, медиана.
PERF_BASELINE = ROOT / "tests" / "fixtures" / "perf_baseline.json"
PERF_VTM_BIN = VTM_DIR / "vtm"
PERF_MODEL = ROOT / "demo" / "model.vtm"

#: Формат файла эталона. Меняется при смене структуры JSON (не протокола).
PERF_BASELINE_FORMAT = 1
#: Версия самого бенчмарка. Меняется при смене протокола замера (модель,
#: число токенов, прогрев, способ съёма времени) — старый эталон тогда
#: несравним и шаг скипается, пока эталон не перезапишут.
PERF_BENCHMARK_VERSION = 1

#: Протокол замера: один prompt-токен, генерация 2048 токенов, 5 прогонов.
#: 2048 токенов — это ~40 мс работы; на 5 прогонах разброс ~2 %, чего с запасом
#: хватает для допуска ±25 %. Меньше брать нельзя: таймер и старт процесса
#: начинают шуметь сильнее допуска.
PERF_TOKENS = 1
PERF_MAX_NEW = 2048
PERF_REPEATS = 5
PERF_TOLERANCE_PCT = 25.0
#: Зависший `vtm run` не должен вешать CI (нормальный прогон — доли секунды).
PERF_TIMEOUT_S = 10.0
PERF_TITLE = "Скорость: demo/model.vtm (perf smoke)"

_RAN_RE = re.compile(r"^Ran (\d+) tests? in ", re.MULTILINE)
_SKIPPED_RE = re.compile(r"^OK \(skipped=(\d+)\)", re.MULTILINE)
_FAILED_RE = re.compile(r"^FAILED \((.+)\)", re.MULTILINE)

#: Строка `vtm run --stats` (stderr): «stats: токенов=N, время=X с, токенов/с=Y».
_PERF_STATS_RE = re.compile(
    r"stats:\s*токенов=(\d+),\s*время=([0-9.]+)\s*с,\s*токенов/с=([0-9.]+)"
)

# Сколько строк вывода показывать при падении (полный вывод — через -v).
FAIL_TAIL_LINES = 160


@dataclass
class Outcome:
    """Итог одного шага."""

    title: str
    status: str  # "ok" | "skip" | "fail"
    seconds: float = 0.0
    detail: str = ""
    output: str = ""

    @property
    def symbol(self) -> str:
        return {"ok": "OK", "skip": "ПРОПУСК", "fail": "FAIL"}[self.status]


# ---------------------------------------------------------------------------
# Запуск команд
# ---------------------------------------------------------------------------


def run_command(
    command: list[str],
    *,
    cwd: Path = ROOT,
    verbose: bool = False,
) -> tuple[int, float, str]:
    """Запустить команду. Возвращает (код, секунды, объединённый вывод).

    stderr сливаем в stdout, чтобы в отчёте не терялся порядок строк.
    """
    env = os.environ.copy()
    # Тесты и проверки печатают по-русски; на ubuntu-latest локаль бывает C,
    # и без этого дочерний Python может упасть на кодировке вывода.
    env.setdefault("PYTHONIOENCODING", "utf-8")

    started = time.monotonic()
    if verbose:
        proc = subprocess.run(command, cwd=str(cwd), env=env)
        output = ""
    else:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = proc.stdout or ""
    return proc.returncode, time.monotonic() - started, output


def show_output(output: str, limit: int = FAIL_TAIL_LINES) -> None:
    """Напечатать хвост вывода упавшей команды."""
    lines = output.splitlines()
    if not lines:
        return
    if len(lines) > limit:
        print(f"... (первые {len(lines) - limit} строк не показаны, полный вывод — с -v)")
        lines = lines[-limit:]
    print("\n".join(lines))


def python_summary(output: str) -> str:
    """Короткая сводка из вывода unittest: тесты, скипы, упавшие."""
    parts: list[str] = []
    ran = _RAN_RE.search(output)
    if ran:
        parts.append(f"тестов {ran.group(1)}")
    skipped = _SKIPPED_RE.search(output)
    if skipped and int(skipped.group(1)):
        parts.append(f"пропущено {skipped.group(1)}")
    failed = _FAILED_RE.search(output)
    if failed:
        parts.append(f"упали: {failed.group(1)}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Шаги
# ---------------------------------------------------------------------------


def banner(index: int, total: int, title: str) -> None:
    print(f"\n== [{index}/{total}] {title} ==")
    sys.stdout.flush()


def report(outcome: Outcome) -> None:
    """Напечатать строку итога шага и, при падении, хвост вывода."""
    suffix = f": {outcome.detail}" if outcome.detail else ""
    if outcome.seconds:
        print(f"-- {outcome.symbol} ({outcome.seconds:.1f}s){suffix}")
    else:
        print(f"-- {outcome.symbol}{suffix}")
    if outcome.status == "fail" and outcome.output:
        print("---- вывод ----")
        show_output(outcome.output)
    sys.stdout.flush()


def which_make() -> str | None:
    return shutil.which("make")


def c_toolchain_problem() -> str | None:
    """Причина пропустить C-шаги или None, если всё на месте."""
    if which_make() is None:
        return "make не найден в PATH"
    if not VTM_MAKEFILE.is_file():
        return "нет src/vtm/Makefile"
    return None


def remaining_artifacts() -> list[Path]:
    """Собранные артефакты, которые остались в src/vtm после clean."""
    found = [VTM_DIR / name for name in C_ARTIFACTS if (VTM_DIR / name).exists()]
    found += sorted(VTM_DIR.glob("*.o"))
    return found


def step_c_tests(verbose: bool) -> Outcome:
    """Шаг 1: сборка с нуля и C-тесты."""

    def failure(detail: str, output: str, seconds: float) -> Outcome:
        return Outcome("C: make -C src/vtm test", "fail", seconds, detail, output)

    rc, seconds, output = run_command(
        ["make", "-C", "src/vtm", "clean"], verbose=verbose
    )
    if rc != 0:
        return failure("make clean не отработал", output, seconds)

    rc, seconds, output = run_command(
        ["make", "-C", "src/vtm", "test"], verbose=verbose
    )
    if rc != 0:
        return failure("сборка или C-тесты упали", output, seconds)
    return Outcome(
        "C: make -C src/vtm test",
        "ok",
        seconds,
        "сборка с нуля и C-тесты пройдены",
        output,
    )


def step_python(verbose: bool) -> Outcome:
    """Шаг 2: весь Python-набор (после C-сборки)."""
    rc, seconds, output = run_command(
        [PYTHON, "-m", "unittest", "discover", "-s", "tests"], verbose=verbose
    )
    if rc != 0:
        return Outcome(
            "Python: unittest discover",
            "fail",
            seconds,
            python_summary(output) or "набор упал",
            output,
        )
    return Outcome(
        "Python: unittest discover",
        "ok",
        seconds,
        python_summary(output) or "набор пройден",
        output,
    )


def step_script(verbose: bool, title: str, script: str) -> Outcome:
    """Шаг-скрипт: tools/check_tasks.py, check_vtm_docs.py, check_commands_doc.py."""
    rc, seconds, output = run_command([PYTHON, script], verbose=verbose)
    if rc != 0:
        return Outcome(title, "fail", seconds, "проверка вернула ошибку", output)
    return Outcome(title, "ok", seconds, "ошибок нет", output)


def step_artifact_free(verbose: bool) -> Outcome:
    """Шаг 7: Python-набор не требует собранных артефактов."""
    title = "Python без артефактов (после make clean)"

    rc, seconds, output = run_command(
        ["make", "-C", "src/vtm", "clean"], verbose=verbose
    )
    if rc != 0:
        return Outcome(title, "fail", seconds, "make clean не отработал", output)

    leftover = remaining_artifacts()
    if leftover:
        names = ", ".join(str(path.relative_to(ROOT)) for path in leftover)
        return Outcome(
            title,
            "fail",
            seconds,
            f"make clean не удалил: {names}",
            output,
        )

    rc, seconds, output = run_command(
        [PYTHON, "-m", "unittest", "discover", "-s", "tests"], verbose=verbose
    )
    detail = python_summary(output)
    if rc != 0:
        return Outcome(
            title,
            "fail",
            seconds,
            "набор упал после make clean (тесты обязаны проходить или "
            "скипать): " + (detail or "см. FAILED выше"),
            output,
        )
    return Outcome(title, "ok", seconds, detail or "набор пройден (или скипнут)")


# ---------------------------------------------------------------------------
# Шаг perf smoke (задача 0117)
# ---------------------------------------------------------------------------

# Что именно меряем: `vtm run demo/model.vtm --tokens 1 --max-new 2048 --stats`
# печатает по монотонным часам время самой генерации (без старта процесса и
# загрузки контейнера) — это и есть «токенов/с» из эталона.


class PerfError(RuntimeError):
    """Замер невозможен: не запустился `vtm run`, упал или вывод не разобран."""


def relative_path(path: Path) -> str:
    """Путь для сообщений: относительно корня, если он внутри корня."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    """sha256 файла потоком (модель небольшая, но читаем без чтения целиком)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_model() -> str:
    """Человеческое имя CPU: на macOS/Linux sysctl//proc/cpuinfo, иначе platform.

    Имя важно: на одном и том же arm64 эталон с M1 Pro и M4 Max сравнивать
    нельзя, поэтому CPU — часть «машины» в эталоне.
    """
    raw = ""
    if sys.platform == "darwin":
        try:
            raw = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            raw = ""
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                key, _, value = line.partition(":")
                if key.strip().lower() == "model name":
                    raw = value
                    break
        except OSError:
            raw = ""
    if not raw.strip():
        raw = platform.processor() or ""
    return " ".join(raw.split())[:120]


def perf_machine() -> dict[str, object]:
    """Машина, на которой снят замер: ОС, архитектура, CPU, число ядер.

    Для сравнения важны os/arch/cpu; cores и python — справочно (в решении о
    скипе не участвуют).
    """
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu": _cpu_model(),
        "cores": os.cpu_count() or 0,
        "python": platform.python_version(),
    }


def load_perf_baseline(path: Path | None = None) -> dict:
    """Прочитать эталон. Нет файла или битый JSON — ValueError."""
    path = Path(path) if path else PERF_BASELINE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"не читается {relative_path(path)}: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{relative_path(path)}: не JSON ({exc.msg}, строка {exc.lineno})"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"{relative_path(path)}: ожидался объект JSON")
    return data


def perf_skip_reason(
    baseline: Mapping[str, object],
    *,
    machine: Mapping[str, object] | None = None,
    model: Path | None = None,
) -> str | None:
    """Почему эталон не для этой машины (или None, если сравнивать можно).

    Скип — не ошибка: на чужой ОС/архитектуре/CPU скорость другая, и красный
    шаг там означал бы «у вас другой компьютер», а не регрессию. Поэтому все
    несовпадения метаданных ведут к предупреждению, а не к падению.
    """
    if not isinstance(baseline, Mapping):
        raise ValueError("эталон должен быть объектом JSON")

    fmt = baseline.get("format")
    if fmt != PERF_BASELINE_FORMAT:
        return (
            f"эталон формата {fmt!r}, ожидается {PERF_BASELINE_FORMAT} — "
            "перезапишите его: python3 tools/ci.py --record-perf-baseline"
        )

    version = baseline.get("benchmark_version")
    if version != PERF_BENCHMARK_VERSION:
        return (
            f"эталон снят версией бенчмарка {version!r}, сейчас "
            f"{PERF_BENCHMARK_VERSION} — нужен новый замер: "
            "python3 tools/ci.py --record-perf-baseline"
        )

    machine = dict(machine or perf_machine())
    saved = baseline.get("machine")
    if not isinstance(saved, Mapping):
        raise ValueError("в эталоне нет объекта machine")
    for key, human in (("os", "ОС"), ("arch", "архитектура")):
        was, now = saved.get(key), machine.get(key)
        if was != now:
            return (
                f"эталон снят на другой машине ({human}: {was!r}, здесь "
                f"{now!r}) — на этой машине эталона нет"
            )
    saved_cpu = str(saved.get("cpu") or "").strip()
    now_cpu = str(machine.get("cpu") or "").strip()
    if saved_cpu and now_cpu and saved_cpu != now_cpu:
        return (
            f"эталон снят на другом CPU ({saved_cpu!r}, здесь {now_cpu!r}) — "
            "на этой машине эталона нет"
        )

    params = baseline.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("в эталоне нет объекта params")
    if params.get("tokens") != PERF_TOKENS or params.get("max_new") != PERF_MAX_NEW:
        return (
            "эталон снят по другому протоколу "
            f"(tokens={params.get('tokens')!r}, max_new={params.get('max_new')!r}; "
            f"сейчас tokens={PERF_TOKENS}, max_new={PERF_MAX_NEW})"
        )

    model = Path(model) if model else PERF_MODEL
    saved_model = baseline.get("model")
    if not isinstance(saved_model, Mapping):
        raise ValueError("в эталоне нет объекта model")
    saved_sha = str(saved_model.get("sha256") or "").strip()
    if saved_sha and model.is_file():
        actual_sha = sha256_file(model)
        if actual_sha != saved_sha:
            return (
                f"эталон снят на другой demo-модели (sha256 {saved_sha[:12]}…, "
                f"здесь {actual_sha[:12]}…) — перезапишите: "
                "python3 tools/ci.py --record-perf-baseline"
            )
    return None


def perf_baseline_metric(baseline: Mapping[str, object]) -> tuple[float, float]:
    """(медиана токенов/с, допуск в %) из эталона; ValueError — эталон битый."""
    metric = baseline.get("metric")
    if not isinstance(metric, Mapping):
        raise ValueError("в эталоне нет объекта metric")
    median = metric.get("median")
    if isinstance(median, bool) or not isinstance(median, (int, float)):
        raise ValueError(f"metric.median не число: {median!r}")
    if median <= 0:
        raise ValueError(f"metric.median должен быть > 0, а он {median!r}")
    tolerance = baseline.get("tolerance_pct", PERF_TOLERANCE_PCT)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError(f"tolerance_pct не число: {tolerance!r}")
    if not 0 < tolerance < 100:
        raise ValueError(f"tolerance_pct вне (0, 100): {tolerance!r}")
    return float(median), float(tolerance)


def compare_perf(
    measured: float, baseline: float, tolerance_pct: float
) -> tuple[str, str]:
    """Сравнить медиану с эталоном. Возвращает (status, detail).

    Допуск двусторонний (±25 %): медленнее — регрессия, быстрее — эталон
    устарел и его надо перезаписать. Иначе «дырка» вверх накапливалась бы:
    каждая следующая правка могла бы отъесть по 20 % и остаться незамеченной.
    """
    delta = (measured - baseline) / baseline * 100.0
    window = (
        f"медиана {measured:.0f} ток/с, эталон {baseline:.0f} ток/с, "
        f"допуск ±{tolerance_pct:.0f}%"
    )
    if delta < -tolerance_pct:
        return (
            "fail",
            f"медленнее эталона на {abs(delta):.1f}% ({window}) — проверьте, "
            "что машина не загружена, и профилируйте (docs/vtm-perf.md)",
        )
    if delta > tolerance_pct:
        return (
            "fail",
            f"быстрее эталона на {delta:.1f}% ({window}) — эталон устарел: "
            "python3 tools/ci.py --record-perf-baseline",
        )
    return "ok", f"{window}, отклонение {delta:+.1f}%"


def measure_perf_once(
    *,
    binary: Path | None = None,
    model: Path | None = None,
    max_new: int | None = None,
) -> float:
    """Один замер: токенов/с из `vtm run --stats`. Бросает PerfError."""
    binary = Path(binary) if binary else PERF_VTM_BIN
    model = Path(model) if model else PERF_MODEL
    max_new = int(max_new if max_new is not None else PERF_MAX_NEW)
    command = [
        str(binary),
        "run",
        str(model),
        "--tokens",
        str(PERF_TOKENS),
        "--max-new",
        str(max_new),
        "--stats",
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=PERF_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise PerfError(
            f"{binary.name} run не уложился в {PERF_TIMEOUT_S:.0f} с"
        ) from exc
    except OSError as exc:
        raise PerfError(f"не запустить {binary}: {exc}") from exc
    if proc.returncode != 0:
        tail = " / ".join((proc.stderr or proc.stdout or "").splitlines()[-4:])
        raise PerfError(f"{binary.name} run вернул {proc.returncode}: {tail}")
    match = _PERF_STATS_RE.search(proc.stderr or "")
    if not match:
        tail = " / ".join((proc.stderr or proc.stdout or "").splitlines()[-4:])
        raise PerfError(f"в выводе нет строки stats: {tail}")
    tokens = int(match.group(1))
    rate = float(match.group(3))
    if tokens != max_new:
        raise PerfError(f"сгенерировано {tokens} токенов вместо {max_new}")
    if rate <= 0:
        raise PerfError(f"неположительная скорость: {rate}")
    return rate


def measure_perf(
    *,
    repeats: int | None = None,
    binary: Path | None = None,
    model: Path | None = None,
) -> list[float]:
    """Серия замеров токенов/с (по умолчанию 5 прогонов)."""
    repeats = int(repeats if repeats is not None else PERF_REPEATS)
    if repeats < 1:
        raise ValueError(f"repeats должен быть ≥ 1, а он {repeats}")
    return [
        measure_perf_once(binary=binary, model=model) for _ in range(repeats)
    ]


def _perf_samples_detail(samples: list[float]) -> str:
    """Короткая строка с замерами — для -v и для записи эталона."""
    return ", ".join(f"{sample:.0f}" for sample in samples)


def perf_baseline_data(
    samples: list[float],
    *,
    machine: Mapping[str, object] | None = None,
    model: Path | None = None,
    repeats: int | None = None,
) -> dict:
    """Собрать JSON эталона из серии замеров (дата, машина, версия, модель)."""
    model = Path(model) if model else PERF_MODEL
    info = dict(machine or perf_machine())
    median = float(statistics.median(samples))
    spread = (
        100.0 * (max(samples) - min(samples)) / median if len(samples) > 1 else 0.0
    )
    return {
        "format": PERF_BASELINE_FORMAT,
        "benchmark": "vtm run demo/model.vtm (decode, --stats)",
        "benchmark_version": PERF_BENCHMARK_VERSION,
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tolerance_pct": PERF_TOLERANCE_PCT,
        "machine": {
            "os": info.get("os"),
            "arch": info.get("arch"),
            "cpu": info.get("cpu"),
            "cores": info.get("cores"),
            "python": info.get("python") or platform.python_version(),
        },
        "model": {
            "path": relative_path(model),
            "bytes": model.stat().st_size,
            "sha256": sha256_file(model),
        },
        "params": {
            "tokens": PERF_TOKENS,
            "max_new": PERF_MAX_NEW,
            "repeats": int(repeats if repeats is not None else PERF_REPEATS),
            "backend": "auto",
        },
        "metric": {
            "name": "tokens_per_second",
            "unit": "ток/с",
            "median": round(median, 1),
            "samples": [round(sample, 1) for sample in samples],
            "spread_pct": round(spread, 2),
        },
    }


def write_perf_baseline(data: Mapping[str, object], path: Path | None = None) -> Path:
    """Записать эталон (родительский каталог создаётся)."""
    path = Path(path) if path else PERF_BASELINE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def step_perf(
    verbose: bool = False,
    *,
    baseline_path: Path | None = None,
    binary: Path | None = None,
    model: Path | None = None,
    measure: Callable[[], list[float]] | None = None,
    machine: Mapping[str, object] | None = None,
) -> Outcome:
    """Шаг perf smoke: сравнить медиану токенов/с с эталоном.

    Параметры-инъекции (baseline_path/binary/model/measure/machine) нужны
    тестам: без них шаг ходит в `src/vtm/vtm` и в репозиторный эталон.
    """
    started = time.monotonic()
    baseline_path = Path(baseline_path) if baseline_path else PERF_BASELINE
    binary = Path(binary) if binary else PERF_VTM_BIN
    model = Path(model) if model else PERF_MODEL

    def finish(status: str, detail: str, output: str = "") -> Outcome:
        return Outcome(
            PERF_TITLE, status, time.monotonic() - started, detail, output
        )

    # Без собранного бинаря и модели мерить нечего: это не регрессия, а
    # незапущенный шаг 1 (например, `--skip-c` на чистом клоне).
    if not model.is_file():
        return finish(
            "skip",
            f"нет {relative_path(model)} — соберите C-часть "
            "(`make -C src/vtm test`)",
        )
    if measure is None and not binary.is_file():
        return finish(
            "skip",
            f"нет {relative_path(binary)} — C-часть не собрана, мерить нечего",
        )

    if not baseline_path.is_file():
        return finish(
            "skip",
            f"предупреждение: нет эталона {relative_path(baseline_path)} — на "
            "этой машине он не снят, шаг пропущен. Снять: "
            "python3 tools/ci.py --record-perf-baseline",
        )

    try:
        baseline = load_perf_baseline(baseline_path)
        reason = perf_skip_reason(baseline, machine=machine, model=model)
        if reason:
            return finish("skip", f"предупреждение: {reason} — шаг пропущен")
        baseline_median, tolerance = perf_baseline_metric(baseline)
    except ValueError as exc:
        return finish(
            "fail",
            f"эталон {relative_path(baseline_path)} повреждён: {exc}",
        )

    try:
        samples = measure() if measure is not None else measure_perf()
    except PerfError as exc:
        return finish("fail", f"замер не удался: {exc}")
    if not samples:
        return finish("fail", "замер не дал ни одного значения")
    if verbose:
        print(f"  замеры (ток/с): {_perf_samples_detail(samples)}")
        sys.stdout.flush()

    status, detail = compare_perf(
        float(statistics.median(samples)), baseline_median, tolerance
    )
    return finish(status, detail)


def step_perf_record(
    verbose: bool = False,
    *,
    baseline_path: Path | None = None,
    binary: Path | None = None,
    model: Path | None = None,
    machine: Mapping[str, object] | None = None,
    repeats: int | None = None,
    measure: Callable[[], list[float]] | None = None,
) -> Outcome:
    """Шаг `--record-perf-baseline`: снять замер и перезаписать эталон."""
    started = time.monotonic()
    baseline_path = Path(baseline_path) if baseline_path else PERF_BASELINE
    binary = Path(binary) if binary else PERF_VTM_BIN
    model = Path(model) if model else PERF_MODEL

    def finish(status: str, detail: str) -> Outcome:
        return Outcome(
            PERF_TITLE, status, time.monotonic() - started, detail
        )

    if not model.is_file():
        return finish(
            "skip",
            f"нет {relative_path(model)} — соберите C-часть "
            "(`make -C src/vtm test`)",
        )
    if measure is None and not binary.is_file():
        return finish(
            "skip", f"нет {relative_path(binary)} — C-часть не собрана"
        )

    try:
        samples = (
            measure()
            if measure is not None
            else measure_perf(repeats=repeats)
        )
    except PerfError as exc:
        return finish("fail", f"замер не удался: {exc}")
    if not samples:
        return finish("fail", "замер не дал ни одного значения")

    data = perf_baseline_data(samples, machine=machine, model=model, repeats=repeats)
    write_perf_baseline(data, baseline_path)
    if verbose:
        print(f"  замеры (ток/с): {_perf_samples_detail(samples)}")
        sys.stdout.flush()
    metric = data["metric"]
    return finish(
        "ok",
        f"эталон перезаписан: {relative_path(baseline_path)}, медиана "
        f"{metric['median']:.0f} ток/с, разброс {metric['spread_pct']:.1f}%",
    )


# ---------------------------------------------------------------------------
# Итог
# ---------------------------------------------------------------------------


def print_summary(outcomes: list[Outcome], total_seconds: float) -> None:
    print("\n== ci: итог ==")
    width = max(len(outcome.title) for outcome in outcomes)
    for outcome in outcomes:
        detail = f"  {outcome.detail}" if outcome.detail else ""
        print(
            f"  {outcome.symbol:<8} {outcome.title:<{width}} "
            f"{outcome.seconds:5.1f}s{detail}"
        )

    failed = [outcome for outcome in outcomes if outcome.status == "fail"]
    skipped = [outcome for outcome in outcomes if outcome.status == "skip"]
    print()
    if failed:
        word = "проверка" if len(failed) == 1 else "проверки"
        print(
            f"итого {total_seconds:.1f}s: {len(failed)} {word} красн"
            f"{'ая' if len(failed) == 1 else 'ые'} — см. вывод выше"
        )
        return
    note = f", {len(skipped)} пропущено" if skipped else ""
    print(
        f"итого {total_seconds:.1f}s: все {len(outcomes)} проверок зелёные{note}"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Единая проверка EVA32: C-тесты, Python-тесты, perf smoke на "
            "demo-модели, реестр задач, документация VTM и чистота набора без "
            "собранных артефактов."
        ),
        epilog=(
            "Начинает с `make -C src/vtm clean`, поэтому после прогона "
            "C-артефакты удалены: для работы нужен `make -C src/vtm`."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="показывать полный вывод команд (по умолчанию — только сводка)",
    )
    parser.add_argument(
        "--skip-c",
        action="store_true",
        help="не собирать и не запускать C-тесты (например, если нет make)",
    )
    parser.add_argument(
        "--no-artifact-check",
        action="store_true",
        help="не проверять, что Python-набор проходит без собранных артефактов",
    )
    parser.add_argument(
        "--skip-perf",
        action="store_true",
        help="не мерить скорость (perf smoke на demo/model.vtm)",
    )
    parser.add_argument(
        "--record-perf-baseline",
        action="store_true",
        help=(
            "снять замер скорости и перезаписать tests/fixtures/perf_baseline.json "
            "вместо сравнения с ним"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()

    if args.skip_c:
        make_problem = "отключено флагом --skip-c"
    else:
        make_problem = c_toolchain_problem()

    check_artifacts = not args.no_artifact_check
    check_perf = not args.skip_perf
    total = 5 + (1 if check_perf else 0) + (1 if check_artifacts else 0)

    outcomes: list[Outcome] = []
    index = 0

    # 1. C: сборка и тесты.
    index += 1
    banner(index, total, "C: make -C src/vtm test")
    if make_problem:
        outcome = Outcome("C: make -C src/vtm test", "skip", 0.0, make_problem)
    else:
        outcome = step_c_tests(args.verbose)
    outcomes.append(outcome)
    report(outcome)
    c_green = outcome.status == "ok"

    # 2. Python-набор (после C-сборки).
    index += 1
    banner(index, total, "Python: python3 -m unittest discover -s tests")
    outcome = step_python(args.verbose)
    outcomes.append(outcome)
    report(outcome)

    # 3. Скорость: короткий замер demo-модели (задача 0117).
    if check_perf:
        index += 1
        banner(index, total, PERF_TITLE)
        if args.record_perf_baseline:
            outcome = step_perf_record(args.verbose)
        else:
            outcome = step_perf(args.verbose)
        outcomes.append(outcome)
        report(outcome)

    # 4. Реестр задач.
    index += 1
    banner(index, total, "Реестр задач: tools/check_tasks.py")
    outcome = step_script(args.verbose, "tasks: check_tasks.py", "tools/check_tasks.py")
    outcomes.append(outcome)
    report(outcome)

    # 5. Документация VTM.
    index += 1
    banner(index, total, "Документация VTM: tools/check_vtm_docs.py")
    outcome = step_script(
        args.verbose, "docs: check_vtm_docs.py", "tools/check_vtm_docs.py"
    )
    outcomes.append(outcome)
    report(outcome)

    # 6. Справочник команд: страница «что умеет, на какие фразы, как отвечает»
    # обязана упоминать все включённые плагины и ключевые фразы. Без этого
    # шага документ тихо отстаёт от кода, а проверка «источник правды — код»
    # держится только на внимательности.
    index += 1
    banner(index, total, "Справочник команд: tools/check_commands_doc.py")
    outcome = step_script(
        args.verbose,
        "docs: check_commands_doc.py",
        "tools/check_commands_doc.py",
    )
    outcomes.append(outcome)
    report(outcome)

    # 7. Чистота: набор не должен зависеть от собранных артефактов.
    if check_artifacts:
        index += 1
        banner(index, total, "Python без артефактов (после make clean)")
        if make_problem:
            outcome = Outcome(
                "Python без артефактов (после make clean)",
                "skip",
                0.0,
                "нет C-сборки — чистить нечего",
            )
        elif not c_green:
            outcome = Outcome(
                "Python без артефактов (после make clean)",
                "skip",
                0.0,
                "C-сборка красная — проверка чистоты не показательной",
            )
        else:
            outcome = step_artifact_free(args.verbose)
        outcomes.append(outcome)
        report(outcome)

    total_seconds = time.monotonic() - started
    print_summary(outcomes, total_seconds)
    return 1 if any(o.status == "fail" for o in outcomes) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nпрервано", file=sys.stderr)
        sys.exit(130)
