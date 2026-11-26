#!/usr/bin/env python3
"""Паритет холодного экспорта и урон от третизации без QAT (задача 0136).

Зачем
=====

Холодный экспорт (``tools/cold_export.py``, задача 0134) делает из обычной
HF-модели тернарный контейнер **без обучения**. После него нужны ответы на
два разных вопроса, и путать их нельзя:

1. **Рантайм.** Совпадает ли C-интерпретатор с HF на **тех же** тритах?
   Если нет — числа качества ничего не значат: мы меряем баг, а не
   квантизацию. Это ``max|Δ|`` логитов и argmax на 3–5 промптах.
2. **Качество.** Сколько стоит сама третизация без QAT? Это perplexity
   одной и той же модели до и после третизации на фиксированном тексте.

Оба ответа считаются здесь, из одного и того же контейнера.

Почему эталонов два
===================

``tools/parity_real.py`` уже решает пункт 1 для QAT-чекпоинта и объясняет,
почему сравнивать C с полной точностью нельзя: тогда ``max|Δ|`` меряет
ошибку квантования (десятки единиц), а не расхождение рантайма. Здесь та же
схема, но источник весов — готовая HF-модель:

* ``hf_ternary`` — HF с ``qat.model.apply_ternary`` (та же функция, которой
  тернаризуют экспорт и обучение). **Это эталон для порога.** Разница с C —
  только порядок суммирования, fp32-масштабы и округление eps RMSNorm;
* ``hf_original`` — модель в ``--dtype`` (по умолчанию bf16). Справочно:
  именно она показывает цену третизации. На порог не влияет.

Порог 1e-2 и почему именно он
=============================

Формулировка «порог 1e-2, потому что третизация» неточна, и это стоит
зафиксировать. Триты на обеих сторонах **одни и те же** — их пишет
``qat.ternary`` (absmax на блок 128, ``round_half_away``, масштаб fp32).
Поэтому остаточное расхождение C↔HF — не ошибка квантования, а арифметика:

* скалярный C копит блок в double, HF — в fp32/bf16 своим порядком;
* масштабы тритов в контейнере fp32 (``--scale-dtype float32``);
* eps RMSNorm в контейнере кодируется как ``imm/1000`` и для Qwen
  ``1e-6`` превращается в 0 (известное ограничение ``tools/vtm_compile.py``).

Наблюдённые величины: fp32-эталон даёт ``max|Δ| ~ 1e-6..1e-5``, bf16-эталон
— до ``~1e-2`` просто потому, что bf16 хранит ~3 десятичных знака. Порог
1e-2 — это «разные ответы», а не «идеальное совпадение»: он на порядок ниже
типичного отрыва top1–top2, а **настоящий критерий — совпадение argmax**.
Поэтому отчёт печатает и отрыв лидера от второго кандидата, и отношение
``отрыв / (2·max|Δ|)``: если argmax сменился при отрыве, который в десятки
раз больше расхождения, это баг ядра, а не допуск.

Perplexity: что именно меряется
===============================

Метрика — **PPL по токенам ответа** на фиксированных парах «вопрос →
ответ»: промпт замаскирован (``-100``), считается только ответ вместе с
EOS. Это та же метрика, что у ``tools/eval_torch.py`` и ``qat/data.py``,
поэтому «до/после» сравнимо с обучением, а не с посторонним числом.

Текст фиксирован: первые ``--ppl-lines`` строк ``--ppl-data`` (JSONL с
``question``/``answer``). Никакой выборки по seed — файл и число строк
попадают в отчёт вместе с sha256, поэтому прогон воспроизводим.

Perplexity считается трижды:

* ``hf_original`` — исходная модель в ``--dtype``;
* ``hf_ternary`` — та же модель после ``apply_ternary``;
* ``vtm_c`` — C-рантайм на готовом ``.vtm`` (проба ниже). Третье число —
  проверка, что PPL C и HF совпадают: если ``hf_ternary`` и ``vtm_c``
  разошлись, качество померяно не на том артефакте.

C-пробы
=======

``src/vtm/vtm run`` печатает только выбранный токен, а для ``max|Δ|`` и PPL
нужны числа. Поэтому скрипт собирает в ``--workdir`` крошечную C-пробу и
линкует её с ``src/vtm/libvtm.a`` — тем же интерпретатором, что у CLI.
Проба умеет два режима:

* ``logits`` — по списку токенов печатает вектор логитов последней позиции;
* ``ppl`` — по последовательности с длиной промпта печатает NLL токенов
  ответа (log-sum-exp в double), то есть PPL считает сам C-рантайм, а не
  Python по его логитам.

Откуда берётся ``.vtm``
=======================

Три варианта, в порядке предпочтения:

1. ``--vtm PATH`` — готовый контейнер (так работает связка с
   ``tools/cold_export.py``: он пишет ``manifest.json`` + ``ternary.trit``,
   из них собирается контейнер);
2. ``--manifest PATH`` — манифест холодного экспорта; скрипт сам зовёт
   ``tools/vtm_compile.py`` и получает ``.vtm`` в ``--workdir``;
3. ``--export`` — **временный** путь, пока ``tools/cold_export.py`` не
   готов: собрать ``.trit`` из HF-весов через ``qat.export`` (математика та
   же: ``qat.ternary``, блок 128, fp32-масштабы) и скомпилировать. Это не
   замена холодному экспорту: ``qat.export`` требует чекпоинт в формате QAT,
   который здесь собирается из ``state_dict`` на лету.

Демо-модели (``--make-demo``)
=============================

Скачивать Qwen нельзя (нет сети в этой среде), а обученных **не-QAT** весов
в репозитории нет: ``runs/tiny_kl/teacher_model`` — мастер-веса QAT-прогона,
их полная точность ничего не значит. Поэтому ``--make-demo`` обучает две
крошечные Qwen2 на ``data/train.jsonl`` без сети:

* ``plain`` — обычное обучение (никакой тернаризации), это «bf16-оригинал»;
* ``qat`` — те же данные, тот же seed инициализации, тот же порядок батчей,
  но forward сразу по тритам (``apply_ternary``, backward через STE).

Это контрольный A/B: одинаковые данные, одинаковая инициализация, одинаковое
число шагов, разница только в QAT. Демо-числа показывают механизм, а не качество Qwen; числа для
0.5B/3B появятся, когда будет ``tools/cold_export.py`` и веса.

Запуск
======

::

    # паритет + PPL на готовом контейнере
    PYTHONPATH=.pylibs python3.13 tools/parity_cold.py \\
        --model Qwen/Qwen2.5-0.5B --vtm export/model.vtm \\
        --ppl-data data/train.jsonl --ppl-lines 200

    # холодный экспорт: из HF-весов собрать .vtm и сразу сверить
    PYTHONPATH=.pylibs python3.13 tools/parity_cold.py \\
        --model /models/Qwen2.5-3B --export --embed-mode trit

    # из манифеста cold_export.py
    PYTHONPATH=.pylibs python3.13 tools/parity_cold.py \\
        --model /models/Qwen2.5-0.5B --manifest export/manifest.json

    # демо без сети: обучить plain+QAT и получить все числа
    PYTHONPATH=.pylibs python3.13 tools/parity_cold.py \\
        --model runs/tiny_kl/teacher_model --make-demo data/parity_cold/demo-a

Коды возврата: 0 — паритет пройден (argmax совпал, ``max|Δ| < порога``);
1 — не пройден; 2 — нет зависимостей/инструментов или плохие аргументы.
``--dry-run`` работает без torch: печатает план и наличие инструментов.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
#: ``tools`` — не пакет (namespace package), поэтому корень нужен в sys.path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
#: Локальные зависимости (torch/transformers для CPU) — только если их ещё
#: нет: stdlib должен побеждать, иначе .pylibs подменит системные модули.

from tools import parity_real as pr  # noqa: E402 - ленивых тяжёлых импортов там нет

def _enable_local_pylibs() -> None:
    """Добавить ``.pylibs`` в путь — только при запуске скриптом.

    Импорт модуля из тестов не должен менять ``sys.path``: иначе каталог
    ``.pylibs`` (собранный под 3.13) попадает в интерпретатор 3.14, ``import
    torch`` падает и навсегда травит ``sys.modules`` — после этого чужие тесты
    видят «torch есть» и падают вместо скипа.
    """
    path = ROOT / ".pylibs"
    if path.is_dir() and str(path) not in sys.path:
        sys.path.append(str(path))


VTM_DIR = ROOT / "src" / "vtm"
VTM_LIB = VTM_DIR / "libvtm.a"
VTM_INCLUDE = VTM_DIR
VTM_COMPILER = ROOT / "tools" / "vtm_compile.py"

#: Порог паритета: «разные ответы», а не «идеальное совпадение» (см. докстринг).
DEFAULT_THRESHOLD = 1e-2
#: dtype исходной модели. bf16 — так модели и лежат в safetensors.
DEFAULT_DTYPE = "bfloat16"
#: Триты считаются по fp32-мастер-весам (как ``qat.export``), поэтому эталон
#: тернарной модели — fp32, независимо от ``--dtype``.
DEFAULT_REFERENCE_DTYPE = "float32"
DEFAULT_SCALE_DTYPE = "float32"
DEFAULT_EMBED_MODE = "trit"
DEFAULT_CONTEXT_LEN = 512
DEFAULT_WORKDIR = ROOT / "data" / "parity_cold"
DEFAULT_PPL_DATA = ROOT / "data" / "train.jsonl"
DEFAULT_PPL_LINES = 200
DEFAULT_PPL_MAX_TOKENS = 128
DEFAULT_LIMIT_TOKENS = 48
#: Формат промпта пары: он же в PPL, он же в демо-обучении.
PROMPT_PREFIX = "Вопрос: "
PROMPT_SUFFIX = "\nОтвет:"

#: Промпты по умолчанию: разные длины и типы (факт, инструкция, код, рассуждение).
DEFAULT_PROMPTS = (
    "Столица Франции — это",
    "Как приготовить борщ? Опиши шаги коротко.",
    "Напиши на Python функцию, которая считает сумму чисел от 1 до n.",
    "Почему небо голубое?",
    "Объясни простыми словами, что такое тернарное квантование и зачем оно нужно.",
)

#: Сколько строк демо-модель видит на обучение и с какого места берётся текст
#: для PPL: они обязаны не пересекаться, иначе PPL меряет запоминание.
DEMO_TRAIN_OFFSET = 1000
DEFAULT_DEMO_PAIRS = 1200
DEFAULT_DEMO_STEPS = 200
DEFAULT_DEMO_QAT_STEPS = 150
DEFAULT_DEMO_SEED = 42
DEFAULT_DEMO_LR = 3e-3
DEFAULT_DEMO_QAT_LR = 1e-3
DEFAULT_DEMO_BATCH = 8



class ColdParityError(RuntimeError):
    """Ошибка прогона, которую видно пользователю без трейсбека."""


# ---------------------------------------------------------------------------
# Аргументы
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parity_cold.py",
        description=(
            "Паритет C↔HF на тритах холодного экспорта и perplexity "
            "«bf16 → триты» на фиксированном тексте."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  PYTHONPATH=.pylibs python3.13 tools/parity_cold.py "
            "--model Qwen/Qwen2.5-0.5B --vtm export/model.vtm\n"
            "  PYTHONPATH=.pylibs python3.13 tools/parity_cold.py "
            "--model /models/Qwen2.5-3B --export --embed-mode trit --ppl-lines 300\n"
            "  PYTHONPATH=.pylibs python3.13 tools/parity_cold.py "
            "--model runs/tiny_kl/teacher_model --make-demo data/parity_cold/demo-a\n"
        ),
    )
    parser.add_argument(
        "--model",
        default=str(ROOT / "runs" / "tiny_kl" / "teacher_model"),
        help="HF-каталог/id: токенизатор, конфиг и эталон (по умолчанию демо-модель репозитория)",
    )
    parser.add_argument("--vtm", default=None, help="готовый контейнер .vtm")
    parser.add_argument(
        "--manifest",
        default=None,
        help="manifest.json холодного экспорта: .vtm соберётся через vtm_compile.py",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="собрать .vtm из --model через qat.export (временный путь, пока нет cold_export.py)",
    )
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR),
                        help=f"куда писать .vtm/пробы/JSON (по умолчанию {DEFAULT_WORKDIR})")
    parser.add_argument("--json", default=None, help="куда записать результат (по умолчанию WORKDIR/parity.json)")
    parser.add_argument("--prompts-file", default=None,
                        help="файл с промптами, по одному на строку (по умолчанию 5 встроенных)")
    parser.add_argument("--prompt", action="append", default=None,
                        help="промпт (можно повторять; заменяет встроенный набор)")
    parser.add_argument("--limit-tokens", type=int, default=DEFAULT_LIMIT_TOKENS,
                        help="обрезать промпт до N токенов")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE,
                        choices=("bfloat16", "float16", "float32"),
                        help="dtype исходной HF-модели (bf16 — как в safetensors)")
    parser.add_argument("--reference-dtype", default=DEFAULT_REFERENCE_DTYPE,
                        choices=("bfloat16", "float16", "float32"),
                        help="dtype HF-эталона тритов (fp32: триты считаются по fp32-мастерам)")
    parser.add_argument("--embed-mode", default=DEFAULT_EMBED_MODE,
                        choices=("none", "trit"),
                        help="квантовать ли таблицу словаря (как quant.quantize_embeddings)")
    parser.add_argument("--scale-dtype", default=DEFAULT_SCALE_DTYPE,
                        choices=("float32", "float16"),
                        help="dtype масштабов тритов при --export/--manifest")
    parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN,
                        help="context_len контейнера при --export (должен быть >= самой длинной PPL-последовательности)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="порог max|Δ| логитов (по умолчанию 1e-2)")
    parser.add_argument("--ppl-data", default=str(DEFAULT_PPL_DATA),
                        help="JSONL с question/answer для perplexity")
    parser.add_argument("--ppl-lines", type=int, default=DEFAULT_PPL_LINES,
                        help="сколько первых строк файла взять (200–500)")
    parser.add_argument("--ppl-max-tokens", type=int, default=DEFAULT_PPL_MAX_TOKENS,
                        help="обрезать пару (промпт+ответ) до N токенов")
    parser.add_argument("--skip-parity", action="store_true", help="только perplexity")
    parser.add_argument("--skip-ppl", action="store_true", help="только паритет")
    parser.add_argument("--skip-original", action="store_true",
                        help="не мерить исходную модель (для QAT-модели её полная точность не значит ничего)")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"),
                        help="устройство HF (cpu: C-рантайм всё равно на CPU)")
    parser.add_argument("--cc", default=None, help="C-компилятор (по умолчанию $CC/cc/clang/gcc)")
    parser.add_argument("--make-demo", default=None, metavar="DIR",
                        help="обучить демо-модель (plain + QAT) в DIR и выйти")
    parser.add_argument("--demo-preset", default="teacher", choices=("teacher", "small"),
                        help="teacher — конфиг из --model; small — своя маленькая форма")
    parser.add_argument("--demo-pairs", type=int, default=DEFAULT_DEMO_PAIRS,
                        help="пар на обучение демо-модели")
    parser.add_argument("--demo-steps", type=int, default=DEFAULT_DEMO_STEPS,
                        help="шагов обычного обучения (без QAT)")
    parser.add_argument("--demo-qat-steps", type=int, default=DEFAULT_DEMO_QAT_STEPS,
                        help="шагов QAT-обучения с того же старта (A/B против plain)")
    parser.add_argument("--demo-lr", type=float, default=DEFAULT_DEMO_LR,
                        help="lr обычного обучения демо-модели")
    parser.add_argument("--demo-qat-lr", type=float, default=DEFAULT_DEMO_QAT_LR,
                        help="lr QAT-дообучения демо-модели")
    parser.add_argument("--demo-seed", type=int, default=DEFAULT_DEMO_SEED)
    parser.add_argument("--dry-run", action="store_true", help="план без загрузки весов")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


# ---------------------------------------------------------------------------
# Окружение и подпроцессы
# ---------------------------------------------------------------------------


@dataclass
class Environment:
    torch: bool
    transformers: bool
    numpy: bool
    export_module: bool
    compiler_script: bool
    vtm_lib: bool
    c_include: bool
    cc: str | None

    def problems(self, *, need_torch: bool = True) -> list[str]:
        problems: list[str] = []
        if need_torch and not self.torch:
            problems.append(
                "нет torch: запустите под python3.13 с PYTHONPATH=.pylibs "
                "(см. qat/requirements-qat.txt)"
            )
        if need_torch and not self.transformers:
            problems.append("нет transformers (нужен для HF-эталона)")
        if not self.compiler_script:
            problems.append(f"нет {VTM_COMPILER}")
        if not self.vtm_lib:
            problems.append("нет src/vtm/libvtm.a: соберите `make -C src/vtm`")
        if not self.c_include:
            problems.append("нет src/vtm/vtm.h")
        if not self.cc:
            problems.append("не найден C-компилятор ($CC/cc/clang/gcc)")
        return problems


def check_environment() -> Environment:
    return Environment(
        torch=pr.module_available("torch"),
        transformers=pr.module_available("transformers"),
        numpy=pr.module_available("numpy"),
        export_module=(ROOT / "qat" / "export.py").is_file()
        and pr.module_available("qat.export"),
        compiler_script=VTM_COMPILER.is_file(),
        vtm_lib=VTM_LIB.is_file(),
        c_include=(VTM_INCLUDE / "vtm.h").is_file(),
        cc=pr.find_c_compiler(),
    )


def subprocess_env() -> dict[str, str]:
    """PYTHONPATH для дочерних процессов: репозиторий + .pylibs + текущий."""
    env = dict(os.environ)
    parts = [str(ROOT)]
    if (ROOT / ".pylibs").is_dir():
        parts.append(str(ROOT / ".pylibs"))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def run_command(cmd: Sequence[str], *, what: str) -> subprocess.CompletedProcess:
    """Запустить подпроцесс и вернуть результат, не теряя хвост ошибки."""
    try:
        proc = subprocess.run(
            list(cmd), capture_output=True, text=True, env=subprocess_env(), cwd=str(ROOT)
        )
    except OSError as exc:
        raise ColdParityError(f"{what}: не запускается {cmd[0]!r}: {exc}") from exc
    if proc.returncode != 0:
        raise ColdParityError(
            f"{what}: код {proc.returncode}\n{pr.tail(proc.stderr or proc.stdout, 30)}"
        )
    return proc


def sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Текст и последовательности (чистые функции — их проверяют тесты)
# ---------------------------------------------------------------------------


def read_text_pairs(path: str | Path, limit: int, *, offset: int = 0) -> list[tuple[str, str]]:
    """Прочитать JSONL и вернуть пары (question, answer) с ``offset``.

    Пустые строки и строки без непустого question/answer пропускаются: файл
    данных пишут разные инструменты, и падать на одной битой строке незачем.
    """
    path = Path(path)
    if not path.is_file():
        raise ColdParityError(f"нет файла данных: {path}")
    pairs: list[tuple[str, str]] = []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ColdParityError(f"{path}:{lineno}: битый JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ColdParityError(f"{path}:{lineno}: ожидался объект JSON")
            question = record.get("question")
            answer = record.get("answer")
            if not (isinstance(question, str) and question.strip()):
                skipped += 1
                continue
            if not (isinstance(answer, str) and answer.strip()):
                skipped += 1
                continue
            pairs.append((question.strip(), answer.strip()))
            if limit and len(pairs) >= int(offset) + int(limit):
                break
    if offset:
        pairs = pairs[int(offset):]
    if not pairs:
        raise ColdParityError(f"в {path} нет пригодных пар question/answer")
    return pairs


def pair_prompt_text(question: str) -> str:
    return f"{PROMPT_PREFIX}{question}{PROMPT_SUFFIX}"


def build_example(
    tokenizer: Any,
    question: str,
    answer: str,
    *,
    max_tokens: int,
    vocab: int | None = None,
) -> dict[str, Any] | None:
    """Пара → ``{ids, prompt_len, answer_tokens, truncated}`` или None.

    Промпт — ``PROMPT_PREFIX + вопрос + PROMPT_SUFFIX``; ответ — токены ответа
    с EOS в конце, как в ``qat/data.py``. Обрезка сохраняет EOS: без него
    модель не учится останавливаться, и PPL ответа теряет смысл.
    """
    prompt_ids = [
        int(token)
        for token in tokenizer(pair_prompt_text(question), add_special_tokens=False)["input_ids"]
    ]
    answer_ids = [
        int(token) for token in tokenizer(answer, add_special_tokens=False)["input_ids"]
    ]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and (not answer_ids or answer_ids[-1] != int(eos_id)):
        answer_ids.append(int(eos_id))
    if not answer_ids or not prompt_ids:
        return None
    truncated = False
    if len(prompt_ids) + len(answer_ids) > int(max_tokens):
        keep = int(max_tokens) - len(prompt_ids)
        if keep <= 0:
            return None
        answer_ids = answer_ids[:keep]
        if eos_id is not None:
            answer_ids[-1] = int(eos_id)
        truncated = True
    ids = prompt_ids + answer_ids
    if vocab is not None and any(token < 0 or token >= int(vocab) for token in ids):
        return None
    return {
        "ids": ids,
        "prompt_len": len(prompt_ids),
        "answer_tokens": len(answer_ids),
        "truncated": truncated,
    }


def build_examples(
    tokenizer: Any,
    pairs: Sequence[tuple[str, str]],
    *,
    max_tokens: int,
    vocab: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Все пары в примеры + число отброшенных (длинные/вне словаря)."""
    examples: list[dict[str, Any]] = []
    dropped = 0
    for question, answer in pairs:
        example = build_example(
            tokenizer, question, answer, max_tokens=max_tokens, vocab=vocab
        )
        if example is None:
            dropped += 1
            continue
        examples.append(example)
    return examples, dropped


def sequence_file_text(examples: Sequence[dict[str, Any]]) -> str:
    """Формат входа C-пробы: ``prompt_len id,id,...`` построчно."""
    lines = []
    for example in examples:
        ids = ",".join(str(int(token)) for token in example["ids"])
        lines.append(f"{int(example['prompt_len'])} {ids}")
    return "\n".join(lines) + ("\n" if lines else "")


def token_file_text(sequences: Sequence[Sequence[int]]) -> str:
    """Формат входа C-пробы для логитов: только id, по строке на промпт."""
    return "\n".join(",".join(str(int(token)) for token in ids) for ids in sequences) + "\n"


def perplexity(nll_sum: float, tokens: int) -> float | None:
    """``exp(средний NLL)``; None, если считать не из чего."""
    if tokens <= 0 or not math.isfinite(nll_sum):
        return None
    try:
        return math.exp(min(700.0, nll_sum / tokens))
    except OverflowError:  # pragma: no cover - защита от экзотических float
        return None


def token_nll(logits: Sequence[float], target: int) -> float:
    """NLL одного токена: ``logsumexp(logits) - logits[target]`` (double)."""
    if not logits:
        raise ColdParityError("пустой вектор логитов")
    if not 0 <= int(target) < len(logits):
        raise ColdParityError(f"target={target} вне словаря {len(logits)}")
    best = max(float(value) for value in logits)
    total = math.fsum(math.exp(float(value) - best) for value in logits)
    return best + math.log(total) - float(logits[int(target)])


def parse_logits_probe_output(text: str) -> dict[str, Any]:
    """Разобрать вывод C-пробы ``logits``: meta, argmax и векторы по промптам."""
    meta: dict[str, Any] = {}
    argmaxes: dict[int, int] = {}
    counts: dict[int, int] = {}
    logits: dict[int, list[float]] = {}
    pending: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line[1:].split()
            if not parts:
                continue
            key = parts[0]
            if key == "seq" and len(parts) >= 3:
                index = int(parts[1])
                argmaxes[index] = int(parts[3]) if len(parts) >= 4 and parts[2] == "argmax" else -1
            elif key == "logits" and len(parts) >= 3:
                pending = int(parts[1])
                counts[pending] = int(parts[2])
            elif key == "meta_json":
                # Храним строкой: в meta уже лежат числовые поля (vocab и
                # т.п.), и разбор вложенного JSON не должен их затирать.
                meta["meta_json"] = line[len("# meta_json"):].strip()
            else:
                try:
                    meta[key] = float(parts[1])
                except (IndexError, ValueError):
                    meta[key] = " ".join(parts[1:])
            continue
        if pending is None:
            raise ColdParityError("проба logits: значения без заголовка '# logits'")
        values = [float(value) for value in line.split()]
        if len(values) != counts[pending]:
            raise ColdParityError(
                f"проба logits: seq {pending} — {len(values)} значений, ожидалось "
                f"{counts[pending]}"
            )
        logits[pending] = values
        if argmaxes.get(pending, -1) < 0 and values:
            argmaxes[pending] = pr.argmax(values)
        pending = None
    if not logits:
        raise ColdParityError("проба logits: ни одного вектора логитов")
    return {"meta": meta, "argmax": argmaxes, "logits": logits}


def parse_ppl_probe_output(text: str) -> dict[str, Any]:
    """Разобрать вывод C-пробы ``ppl``: NLL по последовательностям и итог."""
    meta: dict[str, Any] = {}
    sequences: dict[int, dict[str, float]] = {}
    total_tokens: int | None = None
    total_nll: float | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line[1:].split() if line.startswith("#") else line.split()
        if not line.startswith("#"):
            raise ColdParityError(f"проба ppl: неожиданная строка {line!r}")
        if not parts:
            continue
        key = parts[0]
        if key == "seq" and len(parts) >= 6:
            sequences[int(parts[1])] = {
                "tokens": int(parts[3]),
                "nll": float(parts[5]),
            }
        elif key == "total_tokens" and len(parts) >= 2:
            total_tokens = int(parts[1])
        elif key == "total_nll" and len(parts) >= 2:
            total_nll = float(parts[1])
        else:
            try:
                meta[key] = float(parts[1])
            except (IndexError, ValueError):
                meta[key] = " ".join(parts[1:])
    if total_tokens is None or total_nll is None:
        raise ColdParityError("проба ppl: нет итогов total_tokens/total_nll")
    return {
        "meta": meta,
        "sequences": sequences,
        "tokens": total_tokens,
        "nll_sum": total_nll,
    }


def gap_ratio(reference: Sequence[float], runtime: Sequence[float],
              max_delta: float) -> float | None:
    """Отрыв top1–top2 против ``2·max|Δ|``: во сколько раз ответ устойчив.

    Смена argmax арифметически возможна, только если отрыв лидера от второго
    меньше ``2·d``. Отношение много больше единицы — вердикт «ответ не мог
    перевернуться», а не «повезло».
    """
    if not reference or not runtime:
        return None
    ordered = sorted(range(len(reference)), key=lambda i: (-reference[i], i))
    if len(ordered) < 2 or max_delta <= 0:
        return None
    gap = float(reference[ordered[0]]) - float(reference[ordered[1]])
    return gap / (2.0 * float(max_delta))


# ---------------------------------------------------------------------------
# C-проба: логиты и perplexity из libvtm.a
# ---------------------------------------------------------------------------


def probe_source() -> str:
    """Исходник C-пробы (пишется в workdir, не в репозиторий).

    Два режима: ``logits`` (вектор последней позиции по каждому промпту) и
    ``ppl`` (NLL токенов ответа; log-sum-exp в double). Оба идут через тот
    же ``vtm_run``, что и CLI, поэтому меряют сам интерпретатор.
    """
    return r'''/* cold_parity_probe.c — снять логиты и PPL из .vtm.
 *
 * Собирается tools/parity_cold.py и линкуется с src/vtm/libvtm.a.
 *   cold_parity_probe <model.vtm> logits <файл>   # по промпту в строке
 *   cold_parity_probe <model.vtm> ppl <файл>      # "prompt_len id,id,..."
 * Печатает "# key value" и данные; числа — с точкой (LC_NUMERIC=C). */
#include "vtm.h"

#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_ids(const char *text, int **out, size_t *out_n) {
    int *tokens = NULL;
    size_t n = 0, cap = 0;
    const char *p = text;
    while (*p != '\0') {
        char *end = NULL;
        long value;
        while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
        if (*p == '\0') break;
        value = strtol(p, &end, 10);
        if (end == p || value < 0 || value > 2147483647L) {
            free(tokens);
            return -1;
        }
        if (n == cap) {
            size_t next = cap == 0 ? 16 : cap * 2;
            int *grown = (int *)realloc(tokens, next * sizeof(*tokens));
            if (grown == NULL) { free(tokens); return -1; }
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

/* Один проход: спозиционировать токены последовательности. */
static vtm_status run_tokens(vtm_t *m, const int *ids, size_t n, uint32_t vocab) {
    size_t i;
    for (i = 0; i < n; i++) {
        vtm_status status;
        if (ids[i] < 0 || (uint32_t)ids[i] >= vocab) {
            fprintf(stderr, "токен %d вне словаря %u\n", ids[i], (unsigned)vocab);
            return VTM_E_ARG;
        }
        vtm_set_position(m, (int)i);
        vtm_set_token(m, ids[i]);
        status = vtm_run(m, NULL, NULL);
        if (status != VTM_OK) {
            fprintf(stderr, "прогон: %s: %s\n", vtm_strerror(status),
                    vtm_error_message(m));
            return status;
        }
    }
    return VTM_OK;
}

static const vtm_tensor *read_logits(vtm_t *m, uint32_t vocab) {
    const vtm_tensor *tensor = vtm_tensor_by_name(m, "logits");
    if (tensor == NULL) {
        fprintf(stderr, "в контейнере нет тензора logits\n");
        return NULL;
    }
    if (tensor->kind != VTM_TENSOR_F32 || tensor->f32 == NULL) {
        fprintf(stderr, "logits не fp32 (kind=%u)\n", (unsigned)tensor->kind);
        return NULL;
    }
    if (tensor->n_elements != vocab) {
        fprintf(stderr, "логитов %u, а словарь %u\n",
                (unsigned)tensor->n_elements, (unsigned)vocab);
        return NULL;
    }
    return tensor;
}

static int mode_logits(vtm_t *m, FILE *fh, uint32_t vocab) {
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    int index = 0;
    while ((len = getline(&line, &cap, fh)) > 0) {
        int *ids = NULL;
        size_t n = 0, i;
        const vtm_tensor *tensor;
        int best = 0;
        if (line[0] == '#' || (line[0] == '\n' && len == 1)) continue;
        if (parse_ids(line, &ids, &n) != 0) {
            fprintf(stderr, "плохой список токенов в строке %d\n", index + 1);
            free(line);
            return 2;
        }
        if (n == 0) { free(ids); index++; continue; }
        vtm_reset(m);
        if (run_tokens(m, ids, n, vocab) != VTM_OK) { free(ids); free(line); return 1; }
        tensor = read_logits(m, vocab);
        if (tensor == NULL) { free(ids); free(line); return 1; }
        for (i = 1; i < tensor->n_elements; i++) {
            if (tensor->f32[i] > tensor->f32[best]) best = (int)i;
        }
        printf("# seq %d argmax %d\n", index, best);
        printf("# logits %d %u\n", index, (unsigned)tensor->n_elements);
        for (i = 0; i < tensor->n_elements; i++) {
            printf(i + 1 == tensor->n_elements ? "%.9g\n" : "%.9g ", tensor->f32[i]);
        }
        free(ids);
        ++index;
    }
    free(line);
    return 0;
}

static int mode_ppl(vtm_t *m, FILE *fh, uint32_t vocab) {
    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    int index = 0;
    double total_nll = 0.0;
    unsigned long long total_tokens = 0;
    while ((len = getline(&line, &cap, fh)) > 0) {
        int *ids = NULL;
        size_t n = 0, i;
        long prompt_len = 0;
        char *space;
        double seq_nll = 0.0;
        unsigned long long seq_tokens = 0;
        if (line[0] == '#' || (line[0] == '\n' && len == 1)) continue;
        space = strchr(line, ' ');
        if (space == NULL) {
            fprintf(stderr, "строка %d: жду 'prompt_len id,id,...'\n", index + 1);
            free(line);
            return 2;
        }
        *space = '\0';
        prompt_len = strtol(line, NULL, 10);
        if (parse_ids(space + 1, &ids, &n) != 0) {
            fprintf(stderr, "плохой список токенов в строке %d\n", index + 1);
            free(line);
            return 2;
        }
        if (prompt_len <= 0 || (size_t)prompt_len >= n) {
            fprintf(stderr, "строка %d: prompt_len %ld при %zu токенах\n",
                    index + 1, prompt_len, n);
            free(ids);
            free(line);
            return 2;
        }
        vtm_reset(m);
        for (i = 0; i < n; i++) {
            const vtm_tensor *tensor;
            vtm_set_position(m, (int)i);
            vtm_set_token(m, ids[i]);
            if (vtm_run(m, NULL, NULL) != VTM_OK) {
                fprintf(stderr, "прогон: %s\n", vtm_error_message(m));
                free(ids);
                free(line);
                return 1;
            }
            if (i + 1 < (size_t)prompt_len || i + 1 >= n) continue;
            tensor = read_logits(m, vocab);
            if (tensor == NULL) { free(ids); free(line); return 1; }
            {
                int target = ids[i + 1];
                float best = tensor->f32[0];
                double sum = 0.0;
                uint32_t k;
                for (k = 1; k < tensor->n_elements; k++) {
                    if (tensor->f32[k] > best) best = tensor->f32[k];
                }
                for (k = 0; k < tensor->n_elements; k++) {
                    sum += exp((double)tensor->f32[k] - (double)best);
                }
                seq_nll += (double)best + log(sum) - (double)tensor->f32[target];
                seq_tokens++;
            }
        }
        printf("# seq %d tokens %llu nll %.10g\n", index, seq_tokens, seq_nll);
        total_nll += seq_nll;
        total_tokens += seq_tokens;
        free(ids);
        ++index;
    }
    free(line);
    printf("# total_tokens %llu\n", total_tokens);
    printf("# total_nll %.17g\n", total_nll);
    return 0;
}

int main(int argc, char **argv) {
    vtm_t *machine;
    FILE *fh;
    uint32_t vocab;
    int rc;

    setlocale(LC_NUMERIC, "C");
    if (argc < 4) {
        fprintf(stderr, "нужны: <model.vtm> logits|ppl <файл>\n");
        return 2;
    }
    machine = vtm_load(argv[1]);
    if (machine == NULL) {
        fprintf(stderr, "vtm_load: %s\n", vtm_last_error());
        return 1;
    }
    vocab = vtm_meta_vocab_size(machine);
    printf("# vocab %u\n", (unsigned)vocab);
    if (vtm_meta_json(machine) != NULL && vtm_meta_json(machine)[0] != '\0') {
        printf("# meta_json %s\n", vtm_meta_json(machine));
    }
    fh = fopen(argv[3], "r");
    if (fh == NULL) {
        fprintf(stderr, "не открывается %s\n", argv[3]);
        vtm_free(machine);
        return 2;
    }
    if (strcmp(argv[2], "logits") == 0) {
        rc = mode_logits(machine, fh, vocab);
    } else if (strcmp(argv[2], "ppl") == 0) {
        rc = mode_ppl(machine, fh, vocab);
    } else {
        fprintf(stderr, "неизвестный режим: %s (logits|ppl)\n", argv[2]);
        rc = 2;
    }
    fclose(fh);
    vtm_free(machine);
    return rc;
}
'''


def probe_binary_path(workdir: Path) -> Path:
    return workdir / "cold_parity_probe"


def ensure_probe_built(workdir: Path, cc: str) -> Path:
    """Собрать C-пробу (кеш по mtime исходника) и вернуть путь к бинарнику."""
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / "cold_parity_probe.c"
    text = probe_source()
    if not source.is_file() or source.read_text(encoding="utf-8") != text:
        source.write_text(text, encoding="utf-8")
    binary = probe_binary_path(workdir)
    if binary.is_file() and binary.stat().st_mtime > source.stat().st_mtime:
        return binary
    cmd = [
        cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-D_GNU_SOURCE",
        "-I", str(VTM_INCLUDE), "-o", str(binary), str(source),
        str(VTM_LIB),
        # На Darwin libvtm.a содержит backend_metal.o (Metal/Foundation).
        *(["-framework", "Metal", "-framework", "Foundation"]
          if sys.platform == "darwin" else []),
        "-lm",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ColdParityError(
            "C-проба не собралась:\n" + pr.tail(proc.stderr, 20)
        )
    return binary


def run_probe(workdir: Path, vtm_path: Path, mode: str, payload: str,
              cc: str) -> str:
    """Собрать пробу (если нужно) и прогнать её на ``vtm_path``."""
    binary = ensure_probe_built(workdir, cc)
    payload_path = workdir / f"probe_input_{mode}.txt"
    payload_path.write_text(payload, encoding="utf-8")
    proc = subprocess.run(
        [str(binary), str(vtm_path), mode, str(payload_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise ColdParityError(
            f"C-проба ({mode}) упала с кодом {proc.returncode}:\n"
            + pr.tail(proc.stderr, 20)
        )
    return proc.stdout


def c_last_logits(workdir: Path, vtm_path: Path, sequences: Sequence[Sequence[int]],
                  cc: str) -> dict[str, Any]:
    if not sequences:
        raise ColdParityError("C-проба: пустой список промптов")
    output = run_probe(workdir, vtm_path, "logits", token_file_text(sequences), cc)
    parsed = parse_logits_probe_output(output)
    for index, ids in enumerate(sequences):
        if index not in parsed["logits"]:
            raise ColdParityError(f"C-проба: нет логитов для промпта {index}")
        if len(parsed["logits"][index]) != parsed["meta"].get(
            "vocab", len(parsed["logits"][index])
        ):
            raise ColdParityError(
                f"C-проба: промпт {index}: логитов {len(parsed['logits'][index])}, "
                f"ожидалось vocab={parsed['meta'].get('vocab')}"
            )
    parsed["context_len"] = _meta_context_len(parsed["meta"])
    return parsed


def c_answer_ppl(workdir: Path, vtm_path: Path, examples: Sequence[dict[str, Any]],
                 cc: str) -> dict[str, Any]:
    if not examples:
        raise ColdParityError("C-проба: нечего мерять (нет примеров)")
    output = run_probe(workdir, vtm_path, "ppl", sequence_file_text(examples), cc)
    parsed = parse_ppl_probe_output(output)
    parsed["perplexity"] = perplexity(parsed["nll_sum"], int(parsed["tokens"]))
    parsed["context_len"] = _meta_context_len(parsed["meta"])
    return parsed


def _meta_context_len(meta: dict[str, Any]) -> int | None:
    """Достать context_len контейнера из meta-JSON (если он там есть)."""
    payload = meta.get("meta_json")
    if isinstance(payload, str):
        try:
            meta = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if isinstance(meta, dict):
        model = meta.get("model")
        if isinstance(model, dict):
            try:
                return int(model.get("context_len"))
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# HF: модель, логиты, perplexity
# ---------------------------------------------------------------------------


def resolve_device(torch: Any, name: str) -> Any:
    if name == "auto":
        return torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    return torch.device(name)


def load_hf(dtype_name: str, model: str) -> tuple[Any, Any, Any]:
    """Загрузить (torch, tokenizer, model) и вернуть их вызывающему."""
    try:
        import torch  # noqa: PLC0415 - только в прогоне
        from transformers import AutoTokenizer  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - сломанный GPU-стек
        raise ColdParityError(f"не импортируется torch/transformers: {exc}") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001 - сеть/кэш/битый токенайзер
        raise ColdParityError(f"не загрузился токенизатор {model!r}: {exc}") from exc
    try:
        model_obj = pr.load_hf_model(model, dtype_name)
    except Exception as exc:  # noqa: BLE001 - сеть/кэш/битый конфиг/OOM
        raise ColdParityError(f"не загрузилась HF-модель {model!r}: {exc}") from exc
    model_obj.eval()
    return torch, tokenizer, model_obj


def hf_last_logits(torch: Any, model: Any, ids: Sequence[int]) -> list[float]:
    with torch.no_grad():
        output = model(
            input_ids=torch.tensor([list(ids)], dtype=torch.long),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
            use_cache=False,
        )
    logits = output.logits if hasattr(output, "logits") else output[0]
    return [float(value) for value in logits[0, -1].detach().float().cpu().tolist()]


def hf_answer_ppl(torch: Any, model: Any, examples: Sequence[dict[str, Any]],
                  *, progress: bool = True) -> dict[str, Any]:
    """PPL по токенам ответа: промпт замаскирован, как в eval_torch/qat.data."""
    torch_mod = torch
    total_nll = 0.0
    total_tokens = 0
    per_sequence: list[float] = []
    keep = torch_mod.no_grad()
    with keep:
        for index, example in enumerate(examples):
            ids = list(example["ids"])
            input_ids = torch_mod.tensor([ids], dtype=torch_mod.long)
            logits = model(input_ids=input_ids, use_cache=False).logits[0].float()
            if logits.size(-1) <= max(ids):
                raise ColdParityError(
                    f"логитов {logits.size(-1)}, а токен {max(ids)} вне словаря"
                )
            logprobs = torch_mod.log_softmax(logits[:-1], dim=-1)
            targets = torch_mod.tensor(ids[1:], dtype=torch_mod.long)
            nll = -logprobs.gather(1, targets.view(-1, 1)).squeeze(1)
            start = int(example["prompt_len"]) - 1
            seq_nll = float(nll[start:].sum())
            total_nll += seq_nll
            total_tokens += len(ids) - 1 - start
            per_sequence.append(seq_nll)
            if progress and (index + 1) % 20 == 0:
                print(f"  perplexity: {index + 1}/{len(examples)}",
                      file=sys.stderr, flush=True)
    return {
        "nll_sum": total_nll,
        "tokens": total_tokens,
        "per_sequence_nll": per_sequence,
        "perplexity": perplexity(total_nll, total_tokens),
    }


def apply_ternary_reference(torch: Any, model: Any, embed_mode: str) -> dict[str, Any]:
    """Тернаризовать HF тем же кодом, что экспорт/обучение, и вернуть отчёт."""
    try:
        from qat.model import apply_embedding_quant, apply_ternary  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise ColdParityError(f"не импортируется qat.model: {exc}") from exc
    from qat.config import QuantConfig  # noqa: PLC0415

    quant = QuantConfig(quantize_embeddings=embed_mode)
    report = apply_ternary(model, quant)
    embed_report = apply_embedding_quant(model, quant) if embed_mode != "none" else None
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    if embed_mode != "none" and tied:
        pr.retie_quantized_lm_head(torch, model)
    return {
        "modules": len(report),
        "parameters": report,
        "ternary_params": int(sum(int(item["params"]) for item in report)),
        "embed": embed_report,
        "tied": tied,
    }


# ---------------------------------------------------------------------------
# Сборка .vtm: qat.export (временный путь) или vtm_compile по манифесту
# ---------------------------------------------------------------------------


def export_vtm_from_hf(args: argparse.Namespace, workdir: Path) -> tuple[Path, dict[str, Any]]:
    """Собрать .vtm из HF-весов: qat.export → vtm_compile.

    Это временный путь (``--export``): ``qat.export`` берёт чекпоинт QAT, а
    мы собираем его из ``state_dict`` HF-модели с уже навешенной
    параметризацией. Математика та же, что у холодного экспорта
    (``qat.ternary``: absmax на блок 128, fp32-масштабы), но это **не**
    ``tools/cold_export.py`` и не заменяет его.
    """
    from types import SimpleNamespace  # noqa: PLC0415

    if not pr.module_available("qat.export") or not pr.module_available("numpy"):
        raise ColdParityError(
            "--export требует qat.export и numpy (GPU-стек): "
            "PYTHONPATH=.pylibs python3.13 (см. qat/requirements-qat.txt). "
            "Без них укажите готовый --vtm или --manifest"
        )
    torch, _tokenizer, model = load_hf("float32", args.model)
    report = apply_ternary_reference(torch, model, args.embed_mode)
    run_dir = workdir / "qat_run"
    export_dir = workdir / "export"
    vtm_path = workdir / "model.vtm"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    if export_dir.exists():
        shutil.rmtree(export_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    context_len = max(int(args.context_len), int(args.ppl_max_tokens), 8)
    fake = SimpleNamespace(
        model=args.model,
        torch_dtype="float32",
        device="cpu",
        scale_dtype=args.scale_dtype,
        limit_tokens=context_len,
    )
    from qat.config import QuantConfig  # noqa: PLC0415
    quant = QuantConfig(quantize_embeddings=args.embed_mode)
    run_config = pr.build_run_config(fake, quant)
    raw_state = model.state_dict()
    state = {
        key: value.detach().to(torch.float32)
        for key, value in raw_state.items()
        if "rotary_emb" not in key and "inv_freq" not in key
    }
    del raw_state
    config = model.config
    hf_architecture = pr.hf_config_dict(config)
    architecture = pr.build_architecture(config, context_len)
    run_meta = pr.build_run_meta(
        model_name=args.model,
        config=run_config,
        hf_architecture=hf_architecture,
        vocab=architecture["vocab"],
        num_params=int(sum(parameter.numel() for parameter in model.parameters())),
        ternary_params=int(report["ternary_params"]),
        ternary_report=report["parameters"],
        device="cpu",
    )
    pr.write_json_atomic(run_dir / "run.json", run_meta)
    pr.save_qat_checkpoint(torch, state, run_dir, config=run_config)
    del state, model
    gc.collect()

    export_cmd = pr.build_export_command(
        sys.executable, run_dir, export_dir,
        context_len=context_len, scale_dtype=args.scale_dtype,
    )
    run_command(export_cmd, what="qat.export")
    manifest = export_dir / "manifest.json"
    if not manifest.is_file():
        raise ColdParityError(f"qat.export не написал {manifest}")
    compile_manifest(manifest, vtm_path)
    return vtm_path, {
        "mode": "export",
        "manifest": str(manifest),
        "ternary_trit": str(export_dir / "ternary.trit"),
        "non_ternary": str(export_dir / "non_ternary.bin"),
        "context_len": context_len,
        "warnings": [
            "контейнер собран через qat.export из HF-весов (--export): это "
            "временный путь, пока tools/cold_export.py не готов"
        ],
    }


def compile_manifest(manifest: str | Path, out_path: str | Path) -> Path:
    """Собрать ``.vtm`` из манифеста холодного экспорта."""
    manifest = Path(manifest)
    out_path = Path(out_path)
    if not manifest.is_file():
        raise ColdParityError(f"нет манифеста: {manifest}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = pr.build_compile_command(sys.executable, manifest, out_path)
    run_command(cmd, what="vtm_compile")
    if not out_path.is_file():
        raise ColdParityError(f"vtm_compile не написал {out_path}")
    return out_path


def resolve_vtm(args: argparse.Namespace, workdir: Path) -> tuple[Path, dict[str, Any]]:
    """Понять, откуда взять ``.vtm``: --vtm, --manifest или --export."""
    if args.vtm:
        path = Path(args.vtm).expanduser().resolve()
        if not path.is_file():
            raise ColdParityError(f"нет контейнера: {path}")
        return path, {"mode": "given", "source": str(path)}
    if args.manifest:
        path = compile_manifest(args.manifest, workdir / "model.vtm")
        return path, {
            "mode": "manifest",
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "source": str(path),
        }
    if args.export:
        return export_vtm_from_hf(args, workdir)
    raise ColdParityError(
        "не откуда взять .vtm: укажите --vtm (готовый контейнер), "
        "--manifest (манифест холодного экспорта) или --export (собрать из "
        "--model через qat.export; временный путь, пока нет cold_export.py)"
    )


# ---------------------------------------------------------------------------
# Сравнение и отчёт
# ---------------------------------------------------------------------------


def compare_parity(
    reference: Sequence[float],
    runtime: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """max|Δ|, argmax, top-5 и устойчивость ответа к расхождению."""
    result = pr.compare_logits(reference, runtime, threshold)
    result["gap_ratio"] = gap_ratio(reference, runtime, result["max_abs_delta"])
    return result


def summarize_parity(prompts: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Сводка по промптам: худший max|Δ|, совпадения argmax/top-5, вердикт."""
    if not prompts:
        return {"prompts": 0, "passed": False, "verdict": "нет промптов"}
    worst = max(prompts, key=lambda item: item["comparison"]["max_abs_delta"])
    argmax_matches = sum(1 for item in prompts if item["comparison"]["argmax_match"])
    top5_matches = sum(
        1 for item in prompts
        if item["comparison"]["top5_overlap"] == item["comparison"]["top5_size"]
    )
    within = sum(1 for item in prompts if item["comparison"]["within_threshold"])
    gaps = [
        item["comparison"]["gap_ratio"] for item in prompts
        if item["comparison"]["gap_ratio"] is not None
    ]
    passed = argmax_matches == len(prompts) and within == len(prompts)
    return {
        "prompts": len(prompts),
        "max_abs_delta": worst["comparison"]["max_abs_delta"],
        "worst_prompt": worst["index"],
        "mean_abs_delta": math.fsum(
            item["comparison"]["max_abs_delta"] for item in prompts
        ) / len(prompts),
        "argmax_matches": argmax_matches,
        "top5_matches": top5_matches,
        "within_threshold": within,
        "threshold": float(threshold),
        "min_gap_ratio": min(gaps) if gaps else None,
        "threshold_margin": (
            float(threshold) / worst["comparison"]["max_abs_delta"]
            if worst["comparison"]["max_abs_delta"] > 0 else None
        ),
        "passed": bool(passed),
        "verdict": (
            f"ПРОЙДЕН: argmax {argmax_matches}/{len(prompts)}, max|Δ| "
            f"{worst['comparison']['max_abs_delta']:.3g} < {threshold:g}"
            if passed else
            f"ПРОВАЛЕН: argmax {argmax_matches}/{len(prompts)}, max|Δ| "
            f"{worst['comparison']['max_abs_delta']:.3g} (порог {threshold:g})"
        ),
    }


def summarize_ppl(result: dict[str, Any]) -> dict[str, Any]:
    """Сравнить PPL: исходная, триты в HF, триты в C."""
    original = result.get("hf_original")
    hf_ternary = result.get("hf_ternary")
    c_ternary = result.get("vtm_c")
    summary: dict[str, Any] = {
        "lines": result.get("lines"),
        "tokens": result.get("tokens"),
        "original": original.get("perplexity") if original else None,
        "hf_ternary": hf_ternary.get("perplexity") if hf_ternary else None,
        "c_ternary": c_ternary.get("perplexity") if c_ternary else None,
    }
    if summary["original"] and summary["hf_ternary"]:
        summary["damage_ratio"] = summary["hf_ternary"] / summary["original"]
        summary["damage_percent"] = 100.0 * (summary["damage_ratio"] - 1.0)
    else:
        summary["damage_ratio"] = None
        summary["damage_percent"] = None
    if summary["hf_ternary"] and summary["c_ternary"] and summary["c_ternary"] > 0:
        summary["c_vs_hf_relative"] = abs(summary["c_ternary"] - summary["hf_ternary"]) / summary["c_ternary"]
    else:
        summary["c_vs_hf_relative"] = None
    return summary


def format_report(result: dict[str, Any]) -> str:
    """Человекочитаемая сводка: таблица паритета и «было/стало» по PPL."""
    lines: list[str] = []
    model = result.get("model", {})
    lines.append(f"=== parity_cold: {model.get('name', '?')} ===")
    lines.append(f"модель:      {model.get('name')} (dtype {model.get('dtype')})")
    vtm = result.get("vtm", {})
    lines.append(
        f"контейнер:   {vtm.get('path')} ({vtm.get('size_bytes', 0) / 1024:.1f} КиБ, "
        f"{str(vtm.get('sha256') or '?')[:12]}…, режим {vtm.get('mode')})"
    )
    lines.append(
        f"эталон:      HF {result.get('reference_dtype')} + apply_ternary "
        f"(embed={result.get('embed_mode')}), порог {result.get('threshold'):g}"
    )
    parity = result.get("parity")
    if parity:
        lines.append("")
        lines.append("паритет C ↔ HF на одних тритах:")
        lines.append(
            f"  {'промпт':<34} {'ток':>4} {'argmax':>9} {'max|Δ|':>10} "
            f"{'top5':>6} {'отрыв/2Δ':>10}"
        )
        for item in result["prompts"]:
            text = item["text"]
            short = text if len(text) <= 32 else text[:31] + "…"
            comparison = item["comparison"]
            gap = comparison["gap_ratio"]
            lines.append(
                f"  {short:<34} {item['tokens']:>4} "
                f"{comparison['argmax_reference']:>4}/{comparison['argmax_runtime']:<4} "
                f"{comparison['max_abs_delta']:>10.3g} "
                f"{comparison['top5_overlap']}/{comparison['top5_size']:<4} "
                f"{(f'{gap:.0f}×' if gap is not None else '—'):>10}"
            )
        lines.append("")
        if parity.get("skipped"):
            lines.append(f"итог: {parity.get('verdict', 'паритет не проверялся')}")
        else:
            lines.append(
                f"итог: {parity['verdict']}; top-5 {parity.get('top5_matches', 0)}/"
                f"{parity.get('prompts', 0)}; худший отрыв/2Δ "
                + (f"{parity['min_gap_ratio']:.0f}×"
                   if parity.get("min_gap_ratio") else "—")
            )
        if parity.get("threshold_margin"):
            lines.append(
                f"запас до порога: {parity['threshold_margin']:.0f}× "
                f"(порог {parity['threshold']:g} при худшем max|Δ| "
                f"{parity['max_abs_delta']:.3g})"
            )
        effect = result.get("quantization_effect")
        if effect:
            lines.append(
                "справочно (это цена третизации, а не рантайма): "
                f"bf16-оригинал против тритов max|Δ| {effect['max_abs_delta']:.3g}, "
                f"argmax совпал {effect['argmax_matches']}/{effect['prompts']}"
            )
    ppl = result.get("perplexity")
    if ppl:
        lines.append("")
        lines.append(
            f"perplexity (ответ+EOS, {ppl.get('lines')} строк {ppl.get('data')}, "
            f"{ppl.get('tokens')} токенов):"
        )
        lines.append(f"  {'вариант':<28} {'PPL':>10} {'Δ':>10}")
        if ppl.get("hf_original"):
            value = ppl["hf_original"]["perplexity"]
            lines.append(f"  {'bf16-оригинал':<28} {value:>10.4f} {'—':>10}")
        if ppl.get("hf_ternary"):
            value = ppl["hf_ternary"]["perplexity"]
            delta = (
                f"×{value / ppl['hf_original']['perplexity']:.2f}"
                if ppl.get("hf_original") and ppl["hf_original"].get("perplexity")
                else "—"
            )
            lines.append(f"  {'триты, HF (эталон)':<28} {value:>10.4f} {delta:>10}")
        if ppl.get("vtm_c"):
            value = ppl["vtm_c"]["perplexity"]
            delta = (
                f"×{value / ppl['hf_ternary']['perplexity']:.4f}"
                if ppl.get("hf_ternary") and ppl["hf_ternary"].get("perplexity")
                else "—"
            )
            lines.append(f"  {'триты, C (.vtm)':<28} {value:>10.4f} {delta:>10}")
        summary = ppl.get("summary") or {}
        if summary.get("damage_percent") is not None:
            lines.append(
                f"урон от третизации без QAT: +{summary['damage_percent']:.0f}% "
                f"perplexity (×{summary['damage_ratio']:.2f})"
            )
        if summary.get("c_vs_hf_relative") is not None:
            lines.append(
                f"сверка PPL C↔HF: расхождение {summary['c_vs_hf_relative']:.2e} "
                "относительных"
            )
    warnings = result.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("предупреждения:")
        lines.extend(f"  * {warning}" for warning in warnings)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Демо-модели: обучение без сети (plain + QAT)
# ---------------------------------------------------------------------------


def demo_config(args: argparse.Namespace) -> Any:
    """Конфиг демо-модели: из ``--model`` (teacher) или своя маленькая форма."""
    from transformers import AutoConfig  # noqa: PLC0415

    config = AutoConfig.from_pretrained(args.model)
    if args.demo_preset == "small":
        config.hidden_size = 128
        # 384: input down_proj обязан делиться на block_size 128 (иначе
        # qat.export отказывается писать .trit — row-padding формата).
        config.intermediate_size = 384
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.num_key_value_heads = 1
        config.head_dim = 32
        config.layer_types = ["full_attention"] * 2
        config.max_window_layers = 2
    config.use_cache = False
    return config


def _collate_batch(torch: Any, examples: Sequence[dict[str, Any]]) -> tuple[Any, Any, Any]:
    """Батч с padding: маскируем промпт и pad, как qat.data.collate_samples."""
    width = max(len(example["ids"]) for example in examples)
    input_ids = torch.zeros((len(examples), width), dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    for index, example in enumerate(examples):
        ids = list(example["ids"])
        row = torch.tensor(ids, dtype=torch.long)
        input_ids[index, :len(ids)] = row
        attention_mask[index, :len(ids)] = 1
        label_row = row.clone()
        label_row[:int(example["prompt_len"]) - 1] = -100
        labels[index, :len(ids)] = label_row
    return input_ids, attention_mask, labels


def train_steps(
    torch: Any,
    model: Any,
    examples: Sequence[dict[str, Any]],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    label: str,
) -> dict[str, Any]:
    """Обычный цикл обучения: ``model(input_ids, labels=...)`` даёт CE-лосс."""
    import torch.nn.functional as F  # noqa: PLC0415

    if not examples:
        raise ColdParityError(f"{label}: нет обучающих примеров")
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(lr), betas=(0.9, 0.95), eps=1e-8,
    )
    rng = random.Random(seed)
    order: list[int] = []
    history: list[float] = []
    started = time.time()
    for step in range(1, int(steps) + 1):
        batch = []
        while len(batch) < batch_size:
            if not order:
                order = list(range(len(examples)))
                rng.shuffle(order)
            batch.append(examples[order.pop()])
        input_ids, attention_mask, labels = _collate_batch(torch, batch)
        output = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            use_cache=False,
        )
        loss = output.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()
        value = float(loss.item())
        history.append(value)
        if step % 50 == 0 or step == int(steps):
            print(
                f"  {label}: шаг {step}/{steps}, loss {value:.4f}, "
                f"{time.time() - started:.0f} с",
                file=sys.stderr, flush=True,
            )
    model.eval()
    return {
        "steps": int(steps),
        "lr": float(lr),
        "first_loss": history[0] if history else None,
        "last_loss": history[-1] if history else None,
        "mean_last10": math.fsum(history[-10:]) / max(1, len(history[-10:])),
        "seconds": time.time() - started,
    }


def strip_parametrizations(model: Any) -> int:
    """Снять триты и оставить мастер-веса: так их ждёт экспорт/паритет."""
    from torch.nn.utils.parametrize import remove_parametrizations  # noqa: PLC0415

    removed = 0
    for _name, module in model.named_modules():
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is not None and "weight" in parametrizations:
            remove_parametrizations(module, "weight", leave_parametrized=False)
            removed += 1
    return removed


def make_demo(args: argparse.Namespace) -> int:
    """Обучить демо-модель: plain (bf16-оригинал) и QAT с того же старта.

    Это контрольный A/B, а не «дообучение»: обе модели стартуют с одинаковой
    инициализацией (один ``--demo-seed``), видят одни и те же батчи в одном
    порядке и делают одинаковое число шагов. Разница ровно одна — во второй
    модели forward идёт по тритам (``apply_ternary`` + STE), поэтому её
    мастер-веса изначально приспособлены к квантизации. Иначе «QAT лучше»
    означало бы ещё и «больше шагов обучения».

    Сеть не нужна: конфиг и токенизатор берутся из ``--model``, данные — из
    ``--ppl-data``. Строки обучения (``DEMO_TRAIN_OFFSET``) не пересекаются
    с текстом для PPL: иначе PPL меряет запоминание, а не качество.
    """
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 - нет GPU-стека
        raise ColdParityError(
            f"--make-demo требует torch/transformers: {exc}. Запустите "
            "PYTHONPATH=.pylibs python3.13 tools/parity_cold.py --make-demo …"
        ) from exc

    from qat.config import QuantConfig  # noqa: PLC0415
    from qat.model import apply_embedding_quant, apply_ternary  # noqa: PLC0415

    out_dir = Path(args.make_demo).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = demo_config(args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001
        raise ColdParityError(f"не загрузился токенизатор {args.model!r}: {exc}") from exc
    pairs = read_text_pairs(args.ppl_data, args.demo_pairs, offset=DEMO_TRAIN_OFFSET)
    examples, dropped = build_examples(
        tokenizer, pairs, max_tokens=args.ppl_max_tokens, vocab=int(config.vocab_size)
    )
    print(
        f"[demo] пар на обучение: {len(examples)} (отброшено {dropped}), "
        f"форма hidden={config.hidden_size} layers={config.num_hidden_layers} "
        f"heads={config.num_attention_heads}/{config.num_key_value_heads}",
        file=sys.stderr,
    )
    torch.manual_seed(args.demo_seed)
    model = AutoModelForCausalLM.from_config(config)
    plain_stats = train_steps(
        torch, model, examples,
        steps=args.demo_steps, batch_size=DEFAULT_DEMO_BATCH,
        lr=args.demo_lr, seed=args.demo_seed, label="plain",
    )
    plain_dir = out_dir / "plain"
    model.save_pretrained(str(plain_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(plain_dir))
    print(f"[demo] plain: {plain_dir} (loss {plain_stats['last_loss']:.3f})",
          file=sys.stderr)

    # QAT-модель: тот же seed инициализации и те же батчи, но forward по
    # тритам с первого шага. Никакого «дообучения» поверх plain: иначе
    # выигрыш QAT был бы неотличим от лишних шагов обучения.
    torch.manual_seed(args.demo_seed)
    model = AutoModelForCausalLM.from_config(config)
    quant = QuantConfig(quantize_embeddings=args.embed_mode)
    apply_ternary(model, quant)
    if args.embed_mode != "none":
        apply_embedding_quant(model, quant)
    qat_stats = train_steps(
        torch, model, examples,
        steps=args.demo_qat_steps, batch_size=DEFAULT_DEMO_BATCH,
        lr=args.demo_qat_lr, seed=args.demo_seed, label="qat",
    )
    removed = strip_parametrizations(model)
    qat_dir = out_dir / "qat"
    model.save_pretrained(str(qat_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(qat_dir))
    print(f"[demo] qat: {qat_dir} (мастер-весов {removed} тензоров снято с тритов)",
          file=sys.stderr)
    meta = {
        "format": "eva32.parity_cold.demo",
        "format_version": 1,
        "created_at": pr.utc_now(),
        "source_model": args.model,
        "data": str(Path(args.ppl_data).resolve()),
        "train_offset": DEMO_TRAIN_OFFSET,
        "pairs": len(examples),
        "max_tokens": args.ppl_max_tokens,
        "quant": {"embed_mode": args.embed_mode, "block_size": int(quant.block_size),
                  "scale_dtype": quant.scale_dtype},
        "architecture": {
            "hidden": int(config.hidden_size),
            "intermediate": int(config.intermediate_size),
            "layers": int(config.num_hidden_layers),
            "heads": int(config.num_attention_heads),
            "kv_heads": int(config.num_key_value_heads),
            "vocab": int(config.vocab_size),
        },
        "plain": plain_stats,
        "qat": qat_stats,
        "dirs": {"plain": str(plain_dir), "qat": str(qat_dir)},
    }
    pr.write_json_atomic(out_dir / "demo.json", meta)
    print(format_demo_report(meta))
    return 0


def format_demo_report(meta: dict[str, Any]) -> str:
    architecture = meta["architecture"]
    lines = [
        f"=== демо-модель: {meta['dirs']['plain']} ===",
        f"архитектура: hidden {architecture['hidden']}, слоёв {architecture['layers']}, "
        f"голов {architecture['heads']}/{architecture['kv_heads']}, "
        f"словарь {architecture['vocab']}",
        f"обучение: plain {meta['plain']['steps']} шагов "
        f"(loss {meta['plain']['last_loss']:.3f}), QAT {meta['qat']['steps']} шагов "
        f"(loss {meta['qat']['last_loss']:.3f}), embed={meta['quant']['embed_mode']}",
        "",
        "дальше:",
        f"  plain: --model {meta['dirs']['plain']} --export --ppl-lines 200",
        f"  qat:   --model {meta['dirs']['qat']} --export --skip-original --ppl-lines 200",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# План и прогон
# ---------------------------------------------------------------------------


def collect_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        return [str(item) for item in args.prompt if str(item).strip()]
    if args.prompts_file:
        path = Path(args.prompts_file)
        if not path.is_file():
            raise ColdParityError(f"нет файла промптов: {path}")
        prompts = [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not prompts:
            raise ColdParityError(f"{path}: пустой список промптов")
        return prompts
    return list(DEFAULT_PROMPTS)


def dry_run(args: argparse.Namespace) -> int:
    env = check_environment()
    workdir = Path(args.workdir).expanduser().resolve()
    print("=== parity_cold: dry-run ===")
    print("веса не загружаются; проверяются инструменты и план.")
    print()
    print(f"репозиторий: {ROOT}")
    print(f"модель:      {args.model}")
    print(f"workdir:     {workdir}")
    if args.vtm:
        print(f"контейнер:   {args.vtm}")
    elif args.manifest:
        print(f"контейнер:   из манифеста {args.manifest} (vtm_compile.py)")
    elif args.export:
        print("контейнер:   --export: qat.export из весов, context_len "
              f"{args.context_len}")
    else:
        print("контейнер:   НЕ УКАЗАН (--vtm | --manifest | --export)")
    print(f"промптов:    {len(collect_prompts(args))}, лимит {args.limit_tokens} токенов")
    print(f"perplexity:  {args.ppl_data}, строк {args.ppl_lines}, "
          f"лимит {args.ppl_max_tokens} токенов")
    print(f"dtype:       оригинал {args.dtype}, эталон тритов {args.reference_dtype}, "
          f"embed={args.embed_mode}, порог {args.threshold:g}")
    print()
    print("инструменты:")
    print(f"  {'[ok]  ' if env.torch else '[нет] '} torch")
    print(f"  {'[ok]  ' if env.transformers else '[нет] '} transformers")
    print(f"  {'[ok]  ' if env.export_module else '[нет] '} qat.export (нужен для --export)")
    print(f"  {'[ok]  ' if env.compiler_script else '[нет] '} tools/vtm_compile.py")
    print(f"  {'[ok]  ' if env.vtm_lib else '[нет] '} src/vtm/libvtm.a")
    print(f"  {'[ok]  ' if env.c_include else '[нет] '} src/vtm/vtm.h")
    print(f"  {'[ok]  ' if env.cc else '[нет] '} C-компилятор: {env.cc or 'не найден'}")
    problems = env.problems(need_torch=not args.skip_parity or not args.skip_ppl)
    if problems:
        print()
        print("чего не хватает для полного прогона:")
        for problem in problems:
            print(f"  * {problem}")
    return 0


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    env = check_environment()
    warnings: list[str] = []
    need_torch = not args.skip_parity or not args.skip_ppl
    problems = env.problems(need_torch=need_torch)
    if problems:
        raise ColdParityError("\n".join(["прогон невозможен:"] + [
            f"  * {problem}" for problem in problems
        ]))
    if not (args.vtm or args.manifest or args.export):
        raise ColdParityError(
            "не откуда взять .vtm: --vtm | --manifest | --export"
        )

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    json_path = Path(args.json).expanduser().resolve() if args.json else workdir / "parity.json"

    print("[parity_cold] 1/6: токенизатор и промпты", file=sys.stderr)
    torch, tokenizer, model = load_hf(args.dtype, args.model)
    device = resolve_device(torch, args.device)
    model.to(device)
    vocab = int(getattr(model.config, "vocab_size", 0) or 0)
    if not vocab:
        raise ColdParityError("у модели нет vocab_size в конфиге")
    prompts = collect_prompts(args)
    prompt_ids: list[list[int]] = []
    for prompt in prompts:
        ids = [
            int(token) for token in tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ][: int(args.limit_tokens)]
        if not ids:
            raise ColdParityError(f"промпт {prompt!r} дал пустой список токенов")
        if any(token < 0 or token >= vocab for token in ids):
            raise ColdParityError(
                f"промпт {prompt!r}: токен вне словаря модели ({vocab})"
            )
        prompt_ids.append(ids)
    print(f"[parity_cold] промптов: {len(prompt_ids)}, токенов: "
          f"{[len(ids) for ids in prompt_ids]}", file=sys.stderr)

    print("[parity_cold] 2/6: контейнер .vtm", file=sys.stderr)
    vtm_path, vtm_meta = resolve_vtm(args, workdir)
    warnings.extend(vtm_meta.get("warnings") or [])

    result: dict[str, Any] = {
        "format": "eva32.cold.parity",
        "format_version": 1,
        "created_at": pr.utc_now(),
        "model": {"name": args.model, "dtype": args.dtype,
                  "reference_dtype": args.reference_dtype,
                  "vocab": vocab, "device": str(device)},
        "reference_dtype": args.reference_dtype,
        "embed_mode": args.embed_mode,
        "scale_dtype": args.scale_dtype,
        "threshold": float(args.threshold),
        "vtm": {
            "path": str(vtm_path),
            "mode": vtm_meta.get("mode"),
            "size_bytes": vtm_path.stat().st_size,
            "sha256": sha256_file(vtm_path),
            "context_len": vtm_meta.get("context_len"),
        },
        "warnings": warnings,
        "prompts": [],
        "timings": {},
    }
    stage = time.monotonic()

    # --- 3/6: исходная модель (bf16) --------------------------------------
    hf_original_logits: list[list[float]] = []
    hf_original_ppl: dict[str, Any] | None = None
    examples: list[dict[str, Any]] = []
    longest = 0
    if not args.skip_ppl:
        print("[parity_cold] 3/6: текст для perplexity", file=sys.stderr)
        pairs = read_text_pairs(args.ppl_data, args.ppl_lines)
        examples, dropped = build_examples(
            tokenizer, pairs, max_tokens=args.ppl_max_tokens, vocab=vocab
        )
        if dropped:
            warnings.append(
                f"perplexity: отброшено {dropped} пар (длиннее {args.ppl_max_tokens} "
                "токенов или с токенами вне словаря)"
            )
        if not examples:
            raise ColdParityError("perplexity: после обрезки не осталось пар")
        longest = max(len(example["ids"]) for example in examples)
        if longest > int(args.ppl_max_tokens):
            raise ColdParityError("perplexity: пример длиннее --ppl-max-tokens")
    if not args.skip_original:
        print(f"[parity_cold] 3/6: HF-оригинал ({args.dtype})", file=sys.stderr)
        for ids in prompt_ids:
            hf_original_logits.append(hf_last_logits(torch, model, ids))
        if examples:
            hf_original_ppl = hf_answer_ppl(torch, model, examples)
            print(f"  PPL bf16: {hf_original_ppl['perplexity']:.4f} "
                  f"({hf_original_ppl['tokens']} токенов)", file=sys.stderr)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result["timings"]["hf_original"] = time.monotonic() - stage
    stage = time.monotonic()

    # --- 4/6: HF-эталон тритов -------------------------------------------
    print(f"[parity_cold] 4/6: HF-эталон тритов ({args.reference_dtype})",
          file=sys.stderr)
    torch_ref, _tok, model_ref = load_hf(args.reference_dtype, args.model)
    model_ref.to(device)
    quant_report = apply_ternary_reference(torch_ref, model_ref, args.embed_mode)
    result["ternary"] = {
        "modules": quant_report["modules"],
        "ternary_params": quant_report["ternary_params"],
        "embed": quant_report["embed"],
        "tied": quant_report["tied"],
    }
    hf_ternary_logits = [hf_last_logits(torch_ref, model_ref, ids) for ids in prompt_ids]
    hf_ternary_ppl = None
    if examples:
        hf_ternary_ppl = hf_answer_ppl(torch_ref, model_ref, examples)
        print(f"  PPL триты HF: {hf_ternary_ppl['perplexity']:.4f} "
              f"({hf_ternary_ppl['tokens']} токенов)", file=sys.stderr)
    del model_ref
    gc.collect()
    result["timings"]["hf_ternary"] = time.monotonic() - stage
    stage = time.monotonic()

    # --- 5/6: C-проба ------------------------------------------------------
    print("[parity_cold] 5/6: C-проба (logits + ppl)", file=sys.stderr)
    c_logits = None
    if not args.skip_parity:
        c_logits = c_last_logits(workdir, vtm_path, prompt_ids, env.cc or "cc")
    c_context = c_logits.get("context_len") if c_logits is not None else None
    if c_context and examples and max(len(ex["ids"]) for ex in examples) > c_context:
        warnings.append(
            f"PPL-последовательность длиннее context_len контейнера ({c_context}): "
            "C завернёт KV-кольцо и разойдётся с HF. Пересоберите контейнер с "
            "--context-len >= --ppl-max-tokens"
        )
    c_ppl = None
    if examples:
        c_ppl = c_answer_ppl(workdir, vtm_path, examples, env.cc or "cc")
        print(f"  PPL триты C: {c_ppl['perplexity']:.4f} "
              f"({c_ppl['tokens']} токенов)", file=sys.stderr)
        if c_context and longest > c_context:
            warnings.append(
                f"C-проба: longest={longest} > context_len={c_context}"
            )
    result["timings"]["c_probe"] = time.monotonic() - stage

    # --- 6/6: сравнение и отчёт -------------------------------------------
    print("[parity_cold] 6/6: сравнение", file=sys.stderr)
    for index, prompt in enumerate(prompts if c_logits is not None else ()):
        ternary_logits = hf_ternary_logits[index]
        runtime_logits = c_logits["logits"][index]
        if len(ternary_logits) != len(runtime_logits):
            warnings.append(
                f"промпт {index}: логитов HF {len(ternary_logits)}, C "
                f"{len(runtime_logits)} — контейнер не той модели?"
            )
            continue
        comparison = compare_parity(ternary_logits, runtime_logits, args.threshold)
        item: dict[str, Any] = {
            "index": index,
            "text": prompt,
            "tokens": len(prompt_ids[index]),
            "comparison": comparison,
            "c_meta": {key: value for key, value in c_logits["meta"].items()
                       if key != "meta_json"},
        }
        if hf_original_logits:
            original = pr.compare_logits(
                hf_ternary_logits[index], hf_original_logits[index], args.threshold
            )
            item["original_vs_ternary"] = original
            item["hf_original_argmax"] = original["argmax_runtime"]
        result["prompts"].append(item)
    if c_logits is not None:
        parity_summary = summarize_parity(result["prompts"], args.threshold)
    else:
        parity_summary = {
            "prompts": 0,
            "passed": True,
            "skipped": True,
            "verdict": "паритет не проверялся (--skip-parity)",
        }
    result["parity"] = parity_summary
    if hf_original_logits and c_logits is not None:
        effect = [
            pr.compare_logits(hf_ternary_logits[i], hf_original_logits[i], args.threshold)
            for i in range(len(prompts))
        ]
        result["quantization_effect"] = {
            "max_abs_delta": max(item["max_abs_delta"] for item in effect),
            "argmax_matches": sum(1 for item in effect if item["argmax_match"]),
            "prompts": len(effect),
            "informational": True,
            "note": "цена третизации (HF-оригинал против тритов), не порог рантайма",
        }
    if examples:
        ppl_result: dict[str, Any] = {
            "data": str(Path(args.ppl_data).resolve()),
            "data_sha256": sha256_file(Path(args.ppl_data)),
            "lines": len(examples),
            "tokens": None,
            "max_tokens": args.ppl_max_tokens,
            "prompt_format": f"{PROMPT_PREFIX}…{PROMPT_SUFFIX}",
        }
        if hf_original_ppl is not None:
            ppl_result["hf_original"] = hf_original_ppl
            ppl_result["tokens"] = hf_original_ppl["tokens"]
        if hf_ternary_ppl is not None:
            ppl_result["hf_ternary"] = hf_ternary_ppl
            ppl_result["tokens"] = ppl_result["tokens"] or hf_ternary_ppl["tokens"]
        if c_ppl is not None:
            ppl_result["vtm_c"] = {
                "perplexity": c_ppl["perplexity"],
                "tokens": c_ppl["tokens"],
                "nll_sum": c_ppl["nll_sum"],
            }
        ppl_result["summary"] = summarize_ppl(ppl_result)
        result["perplexity"] = ppl_result
        if c_ppl is not None and hf_ternary_ppl is not None:
            relative = ppl_result["summary"]["c_vs_hf_relative"]
            if relative is not None and relative > 1e-3:
                warnings.append(
                    f"PPL C≠HF: расхождение {relative:.2e} относительных — "
                    "проверьте context_len контейнера и масштабы"
                )
    result["timings"]["total"] = time.monotonic() - started
    status = "passed" if parity_summary.get("passed", False) else "failed"
    ppl_summary = (result.get("perplexity") or {}).get("summary") or {}
    ppl_relative = ppl_summary.get("c_vs_hf_relative")
    if ppl_relative is not None and ppl_relative > 1e-3:
        status = "failed"
    if parity_summary.get("skipped") and ppl_relative is None:
        status = "partial"
    result["status"] = status
    pr.write_json_atomic(json_path, result)
    print()
    print(format_report(result))
    print()
    print(f"результат: {json_path}")
    return 0 if result["status"] == "passed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:  # argparse уже напечатал ошибку
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        if args.make_demo:
            return make_demo(args)
        if args.dry_run:
            return dry_run(args)
        return run(args)
    except ColdParityError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[parity_cold] прервано", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
