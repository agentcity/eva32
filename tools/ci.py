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
  3. Реестр задач: `python3 tools/check_tasks.py`.
  4. Документация VTM: `python3 tools/check_vtm_docs.py`.
  5. Чистота: `make -C src/vtm clean` и ещё раз Python-набор. Ни один тест не
     должен требовать собранных артефактов: на чистом клоне набор обязан
     проходить (пусть и со скипами), а не падать.

Только стандартная библиотека, без сети и установки пакетов.

Запуск:
    python3 tools/ci.py                 # все проверки
    python3 tools/ci.py -v              # показывать полный вывод команд
    python3 tools/ci.py --skip-c        # без C (нет make)
    python3 tools/ci.py --no-artifact-check   # без шага 5

Код выхода: 0 — всё зелёное, 1 — есть упавшая проверка, 130 — Ctrl-C.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
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

_RAN_RE = re.compile(r"^Ran (\d+) tests? in ", re.MULTILINE)
_SKIPPED_RE = re.compile(r"^OK \(skipped=(\d+)\)", re.MULTILINE)
_FAILED_RE = re.compile(r"^FAILED \((.+)\)", re.MULTILINE)

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
    """Шаг-скрипт: tools/check_tasks.py или tools/check_vtm_docs.py."""
    rc, seconds, output = run_command([PYTHON, script], verbose=verbose)
    if rc != 0:
        return Outcome(title, "fail", seconds, "проверка вернула ошибку", output)
    return Outcome(title, "ok", seconds, "ошибок нет", output)


def step_artifact_free(verbose: bool) -> Outcome:
    """Шаг 5: Python-набор не требует собранных артефактов."""
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
            "Единая проверка EVA32: C-тесты, Python-тесты, реестр задач, "
            "документация VTM и чистота набора без собранных артефактов."
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()

    if args.skip_c:
        make_problem = "отключено флагом --skip-c"
    else:
        make_problem = c_toolchain_problem()

    check_artifacts = not args.no_artifact_check
    total = 5 if check_artifacts else 4

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

    # 3. Реестр задач.
    index += 1
    banner(index, total, "Реестр задач: tools/check_tasks.py")
    outcome = step_script(args.verbose, "tasks: check_tasks.py", "tools/check_tasks.py")
    outcomes.append(outcome)
    report(outcome)

    # 4. Документация VTM.
    index += 1
    banner(index, total, "Документация VTM: tools/check_vtm_docs.py")
    outcome = step_script(
        args.verbose, "docs: check_vtm_docs.py", "tools/check_vtm_docs.py"
    )
    outcomes.append(outcome)
    report(outcome)

    # 5. Чистота: набор не должен зависеть от собранных артефактов.
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
