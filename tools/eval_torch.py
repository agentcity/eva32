#!/usr/bin/env python3
"""Замер качества модели на torch: golden-проверки, perplexity, длина ответа.

Скрипт запускается на арендованной GPU-машине после QAT (``none``/``int8``/
``trit``) и отвечает на вопрос «стало ли хуже по ответам», а не только по loss:

* **accuracy** — доля вопросов ``eval/golden.jsonl``, прошедших поведенческие
  проверки ``check()`` из ``eval/run_eval.py`` (must/any/forbid/regex/
  forbid_regex/no_prompt_shingle);
* **perplexity** — по токенам ответа на отложенных парах ``question → answer``
  (``--ppl-data``), разметка как в обучении ``qat/data.py``: вопрос в chat-
  шаблоне, label = -100 на промпте, EOS входит в ответ;
* **avg_len** — средняя длина сгенерированного ответа в токенах.

Генерация детерминированная (greedy), поэтому прогоны ``none``/``int8``/``trit``
сравниваются между собой, а не с шумом сэмплинга.

Запуск::

    python3 tools/eval_torch.py --dry-run        # без torch: план и golden
    python3 tools/eval_torch.py --model Qwen/Qwen2.5-1.5B-Instruct \\
        --golden eval/golden.jsonl --ppl-data distill/out/pairs.jsonl \\
        --out runs/eval.json

Чекпоинт QAT можно передать каталогом прогона или каталогом шага::

    python3 tools/eval_torch.py --model runs/pilot1 --out runs/eval-pilot1.json
    python3 tools/eval_torch.py --model runs/pilot1/checkpoints/step_002000 \\
        --out runs/eval-step2000.json

``--model`` — либо Hugging Face id/локальный каталог HF, либо run-каталог QAT
(``checkpoints/step_*``), либо сам каталог чекпоинта. Для чекпоинта базовая
модель и параметры квантования читаются из ``trainer_state.pt``.

Без torch ``--dry-run`` работает всегда, обычный запуск печатает понятную
ошибку и возвращает 2. Обучение не запускается; сеть скрипт не использует
(кроме обычной загрузки весов transformers — её отключает
``--local-files-only``).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_GOLDEN = ROOT / "eval" / "golden.jsonl"
DEFAULT_SYSTEM_PROMPT = ROOT / "prompts" / "system.ru.md"
DEFAULT_OUT = Path("runs") / "eval.json"

#: Сколько пар брать на perplexity по умолчанию (можно изменить --ppl-limit).
DEFAULT_PPL_LIMIT = 200
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_SEQ_LEN = 2048

#: Правила golden, которые понимает ``eval/run_eval.py::check``.
KNOWN_CHECKS = ("must", "any", "forbid", "regex", "forbid_regex", "no_prompt_shingle")
LIST_CHECKS = ("must", "any", "forbid", "regex", "forbid_regex")

#: Потолок позиций в одном forward для perplexity: логиты [B, L, vocab]
#: в fp32 иначе съедают память на длинных парах.
PPL_TOKEN_BUDGET = 4096
PPL_LOGIT_CHUNK = 128

REQUIRED_MODULES = (("torch", "torch"), ("transformers", "transformers"))

_REGEX_FLAGS = re.IGNORECASE | re.UNICODE


class EvalError(RuntimeError):
    """Прогон невозможен: плохие данные, модель или аргументы."""


# ---------------------------------------------------------------------------
# CLI и проверка окружения
# ---------------------------------------------------------------------------


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось целое число, получено {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"ожидалось число > 0, получено {value}")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось целое число, получено {text!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"ожидалось число >= 0, получено {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 tools/eval_torch.py",
        description=(
            "Замер на torch: accuracy по eval/golden.jsonl, perplexity на парах "
            "distill-формата и средняя длина ответа."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  python3 tools/eval_torch.py --dry-run\n"
            "      без torch: прочитать golden и напечатать план\n\n"
            "  python3 tools/eval_torch.py --model Qwen/Qwen2.5-1.5B-Instruct \\\n"
            "      --golden eval/golden.jsonl --ppl-data distill/out/pairs.jsonl \\\n"
            "      --out runs/eval.json\n"
            "      базовая модель: accuracy, perplexity, средняя длина\n\n"
            "  python3 tools/eval_torch.py --model runs/pilot1 \\\n"
            "      --ppl-data distill/out/pairs.jsonl --out runs/eval-trit.json\n"
            "      последний чекпоинт QAT-прогона (база и quant — из trainer_state.pt)\n\n"
            "  python3 tools/eval_torch.py --model runs/pilot1 --limit 10 \\\n"
            "      --max-new-tokens 128 --batch-size 4 --ppl-limit 50\n"
            "      быстрая проверка на десяти вопросах\n\n"
            "зависимости GPU-машины:\n"
            "  python3 -m pip install -r qat/requirements-qat.txt\n\n"
            "тесты без torch:\n"
            "  python3 -m unittest tests.test_eval_torch -v\n"
        ),
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"HF id/каталог или run-каталог QAT (по умолчанию {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--golden", default=str(DEFAULT_GOLDEN),
        help="JSONL с вопросами и checks (по умолчанию eval/golden.jsonl)",
    )
    parser.add_argument(
        "--ppl-data", default=None,
        help="JSONL с парами question/answer для perplexity (например distill/out/pairs.jsonl)",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="куда записать JSON с метриками (по умолчанию runs/eval.json)",
    )
    parser.add_argument(
        "--limit", type=_non_negative_int, default=0,
        help="взять первые N вопросов golden; 0 — все",
    )
    parser.add_argument(
        "--ppl-limit", type=_non_negative_int, default=DEFAULT_PPL_LIMIT,
        help=f"сколько пар брать на perplexity; 0 — все (по умолчанию {DEFAULT_PPL_LIMIT})",
    )
    parser.add_argument(
        "--max-new-tokens", type=_positive_int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"лимит новых токенов на ответ (по умолчанию {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--batch-size", type=_positive_int, default=DEFAULT_BATCH_SIZE,
        help=f"размер батча генерации (по умолчанию {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-seq-len", type=_positive_int, default=DEFAULT_MAX_SEQ_LEN,
        help=f"максимум токенов пары для perplexity (по умолчанию {DEFAULT_MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--device", default="auto",
        help="auto|cuda|cuda:0|mps|cpu (по умолчанию auto: cuda → mps → cpu)",
    )
    parser.add_argument(
        "--local-files-only", action="store_true",
        help="не обращаться к Hugging Face: брать веса только из локального кеша",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="без torch: проверить golden и напечатать план, exit 0",
    )
    parser.add_argument(
        "--traceback", action="store_true",
        help="показать трейсбек при неожиданной ошибке",
    )
    return parser


def missing_modules() -> list[str]:
    """Каких зависимостей GPU-стека не хватает (без импорта тяжёлых модулей)."""
    missing: list[str] = []
    for module, package in REQUIRED_MODULES:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(package)
    return missing


def print_missing_dependencies(missing: Sequence[str]) -> None:
    print("[eval] не хватает зависимостей: " + ", ".join(missing), file=sys.stderr)
    print(file=sys.stderr)
    print("Замер с perplexity и генерацией запускается только там, где есть torch.",
          file=sys.stderr)
    print("На GPU-машине:", file=sys.stderr)
    print("    python3 -m pip install -r qat/requirements-qat.txt", file=sys.stderr)
    print(file=sys.stderr)
    print("Без torch доступен план (golden читается, модель не грузится):", file=sys.stderr)
    print("    python3 tools/eval_torch.py --dry-run", file=sys.stderr)
    print("    python3 -m unittest tests.test_eval_torch -v", file=sys.stderr)


# ---------------------------------------------------------------------------
# Чтение golden и отложенных пар
# ---------------------------------------------------------------------------


def _strip(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def load_golden(path: str | Path) -> list[dict[str, Any]]:
    """Прочитать ``eval/golden.jsonl`` и проверить формат checks.

    Ошибки формата — с путём и номером строки: чинить набор на GPU-машине
    дорого, поэтому кривой golden должен падать до загрузки модели.
    """
    path = Path(path)
    if not path.exists():
        raise EvalError(f"golden не найден: {path}")
    rows: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalError(f"{path}:{lineno}: невалидный JSON: {exc}") from exc
            _validate_golden_row(row, path, lineno)
            row_id = row["id"]
            if row_id in seen_ids:
                raise EvalError(
                    f"{path}:{lineno}: повтор id {row_id!r} "
                    f"(уже в строке {seen_ids[row_id]})"
                )
            seen_ids[row_id] = lineno
            rows.append(row)
    if not rows:
        raise EvalError(f"{path}: пустой набор")
    return rows


def _validate_golden_row(row: Any, path: Path, lineno: int) -> None:
    where = f"{path}:{lineno}"
    if not isinstance(row, dict):
        raise EvalError(f"{where}: ожидался объект JSON")
    if not _strip(row.get("id")):
        raise EvalError(f"{where}: пустой id")
    if not _strip(row.get("question")):
        raise EvalError(f"{where}: пустой question")
    if "context" in row and row["context"] is not None and not isinstance(row["context"], str):
        raise EvalError(f"{where}: context должен быть строкой")
    checks = row.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise EvalError(f"{where}: checks должен быть непустым объектом")
    # Неизвестные ключи не роняют набор: run_eval.py их игнорирует,
    # а план dry-run перечислит их отдельно.
    for key in LIST_CHECKS:
        value = checks.get(key)
        if value is None:
            continue
        if not isinstance(value, list) or not value:
            raise EvalError(f"{where}: checks.{key} должен быть непустым списком")
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise EvalError(f"{where}: checks.{key}: ожидались непустые строки")
        if key.endswith("regex"):
            for pattern in value:
                try:
                    re.compile(pattern, _REGEX_FLAGS)
                except re.error as exc:
                    raise EvalError(
                        f"{where}: checks.{key}: некорректный regex {pattern!r}: {exc}"
                    ) from exc
    shingle = checks.get("no_prompt_shingle")
    if shingle is not None:
        if isinstance(shingle, bool):
            pass
        elif not (isinstance(shingle, int) and shingle >= 2):
            raise EvalError(
                f"{where}: checks.no_prompt_shingle должен быть true/false или числом >= 2"
            )


def check_rule_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Сколько строк используют каждое правило (для плана dry-run)."""
    counts = {key: 0 for key in KNOWN_CHECKS}
    for row in rows:
        for key in (row.get("checks") or {}):
            if key in counts:
                counts[key] += 1
    return counts


def unknown_check_keys(rows: Iterable[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(set(row.get("checks") or {}) - set(KNOWN_CHECKS))
    return sorted(keys)


def load_pairs(path: str | Path, limit: int = 0) -> dict[str, Any]:
    """Прочитать пары ``question → answer`` для perplexity.

    Возвращает ``{"rows", "total", "skipped"}``. Пустые question/answer
    пропускаются (как в ``qat/data.py``), битый JSON — ошибка с номером строки.
    ``limit`` — сколько пар взять; выборка равномерная по файлу, чтобы не
    мерить только начало (там пары одного тарифа).
    """
    path = Path(path)
    if not path.exists():
        raise EvalError(f"файл пар для perplexity не найден: {path}")
    rows: list[dict[str, Any]] = []
    skipped = 0
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalError(f"{path}:{lineno}: невалидный JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise EvalError(f"{path}:{lineno}: ожидался объект JSON")
            question = _strip(row.get("question"))
            answer = _strip(row.get("answer"))
            if not question or not answer:
                skipped += 1
                continue
            rows.append({"question": question, "answer": answer})
    if not rows:
        raise EvalError(f"{path}: нет пригодных пар question/answer")
    return {
        "rows": sample_evenly(rows, limit),
        "total": len(rows),
        "skipped": skipped,
    }


def sample_evenly(items: Sequence[Any], limit: int) -> list[Any]:
    """Взять ``limit`` элементов равномерно по списку (детерминированно).

    ``limit <= 0`` или ``limit >= len(items)`` — вернуть всё как есть.
    """
    values = list(items)
    if limit <= 0 or limit >= len(values):
        return values
    if limit == 1:
        return [values[0]]
    step = (len(values) - 1) / (limit - 1)
    return [values[round(index * step)] for index in range(limit)]


# ---------------------------------------------------------------------------
# Промпты: как в eval/run_eval.py и qat/data.py
# ---------------------------------------------------------------------------


def row_content(row: dict[str, Any]) -> str:
    """Содержимое user-сообщения ровно как в ``eval/run_eval.py::build_payload``."""
    content = row["question"]
    if row.get("context"):
        content = f"Контекст:\n{row['context']}\n\nВопрос: {content}"
    return content


def build_messages(system: str, row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": row_content(row)},
    ]


def render_prompt(tokenizer: Any, system: str, row: dict[str, Any]) -> str:
    """Chat-шаблон с system-промптом; запасной вариант — без шаблона."""
    messages = build_messages(system, row)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (
        f"Система: {system}\n\n"
        f"Пользователь: {row_content(row)}\n\n"
        "Ассистент:"
    )


def pair_prompt_ids(tokenizer: Any, question: str) -> list[int]:
    """Промпт пары для perplexity — как ``qat/data.py::PairDataset._prompt_ids``."""
    messages = [{"role": "user", "content": question}]
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        return [int(token) for token in ids]
    text = f"Пользователь: {question}\nАссистент:"
    return [int(token) for token in tokenizer(text, add_special_tokens=False).input_ids]


def pair_answer_ids(tokenizer: Any, answer: str) -> list[int]:
    """Токены ответа с EOS: ровно то, на чём учили в ``qat/data.py``."""
    ids = [int(token) for token in tokenizer(answer, add_special_tokens=False).input_ids]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and (not ids or ids[-1] != eos_id):
        ids = ids + [int(eos_id)]
    return ids


def tokenize_pair(
    tokenizer: Any, row: dict[str, Any], max_seq_len: int
) -> dict[str, Any] | None:
    """Пара в ``input_ids``/``labels``; ``None`` — если длиннее лимита."""
    prompt_ids = pair_prompt_ids(tokenizer, row["question"])
    answer_ids = pair_answer_ids(tokenizer, row["answer"])
    if not answer_ids:
        return None
    if len(prompt_ids) + len(answer_ids) > max_seq_len:
        return None
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
        "answer_tokens": len(answer_ids),
    }


# ---------------------------------------------------------------------------
# Модель: спецификация, загрузка, устройство
# ---------------------------------------------------------------------------


def _checkpoint_spec(path: Path) -> dict[str, Any]:
    if not (path / "model.pt").exists():
        raise EvalError(f"{path}: нет model.pt — это не каталог чекпоинта QAT")
    if not (path / "trainer_state.pt").exists():
        raise EvalError(
            f"{path}: нет trainer_state.pt — не из чего взять базовую модель "
            f"и параметры квантования"
        )
    return {"kind": "checkpoint", "checkpoint": str(path), "name": None}


def _latest_checkpoint(run_dir: Path) -> Path | None:
    root = run_dir / "checkpoints"
    if not root.is_dir():
        return None
    found: list[tuple[int, Path]] = []
    for path in root.glob("step_*"):
        if path.is_dir() and path.name[5:].isdigit():
            found.append((int(path.name[5:]), path))
    if not found:
        return None
    return max(found, key=lambda item: item[0])[1]


def resolve_model(path: str | Path) -> dict[str, Any]:
    """Понять, что передали в ``--model``: HF id, HF-каталог или чекпоинт QAT.

    Чекпоинт определяется до импорта torch: каталог с ``model.pt`` и
    ``trainer_state.pt``, run-каталог с ``checkpoints/step_*`` или сам шаг.
    """
    raw = str(path)
    candidate = Path(path).expanduser()
    if not candidate.exists():
        # Не локальный путь — значит HF id вроде Qwen/Qwen2.5-1.5B-Instruct.
        return {"kind": "hf", "name": raw, "local": False}
    candidate = candidate.resolve()
    if candidate.is_file():
        if candidate.name == "model.pt":
            return _checkpoint_spec(candidate.parent)
        raise EvalError(
            f"{candidate}: ожидался каталог модели/чекпоинта или Hugging Face id, "
            f"а не файл"
        )
    if (candidate / "model.pt").exists() or (candidate / "trainer_state.pt").exists():
        return _checkpoint_spec(candidate)
    latest = _latest_checkpoint(candidate)
    if latest is not None:
        return _checkpoint_spec(latest)
    if (candidate / "checkpoints").is_dir():
        raise EvalError(
            f"{candidate}: в checkpoints нет целых step_*; прогон не сохранил чекпоинт"
        )
    if (candidate / "config.json").exists():
        return {"kind": "hf", "name": str(candidate), "local": True}
    if (candidate / "latest.txt").exists():
        raise EvalError(
            f"{candidate}: latest.txt есть, но в checkpoints нет целых step_*"
        )
    raise EvalError(
        f"{candidate}: не похоже ни на run QAT (checkpoints/step_*), "
        f"ни на HF-модель (config.json)"
    )


def describe_model_spec(spec: dict[str, Any]) -> str:
    if spec["kind"] == "checkpoint":
        return (
            f"чекпоинт QAT {spec['checkpoint']}; базовая модель и quant — "
            f"из trainer_state.pt (нужен torch)"
        )
    if spec.get("local"):
        return f"локальный каталог HF {spec['name']} (config.json найден, сеть не нужна)"
    return (
        f"HF id {spec['name']}; transformers берёт веса из локального кеша, "
        f"а при их отсутствии скачает (--local-files-only это запретит)"
    )


def resolve_device(torch: Any, name: str) -> Any:
    """auto → cuda → mps → cpu; явное имя проверяется на доступность."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(name)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise EvalError(f"--device {name!r}: неизвестное устройство ({exc})") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise EvalError(f"--device {name}: CUDA недоступна на этой машине")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise EvalError(f"--device {name}: MPS недоступна на этой машине")
    return device


def load_validators() -> Any:
    """Загрузить ``eval/run_eval.py`` по пути и вернуть модуль проверок.

    ``eval`` — не пакет, поэтому импортируем файл явно. Так accuracy считается
    ровно той же функцией ``check``, что и в живом прогоне эндпоинта.
    """
    path = ROOT / "eval" / "run_eval.py"
    if not path.exists():
        raise EvalError(f"не найден {path}: без него не работают проверки golden")
    spec = importlib.util.spec_from_file_location("eva32_eval_run_eval", path)
    if spec is None or spec.loader is None:
        raise EvalError(f"не удалось загрузить {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_system_prompt(path: str | Path = DEFAULT_SYSTEM_PROMPT) -> str:
    path = Path(path)
    if not path.exists():
        raise EvalError(f"не найден системный промпт: {path}")
    return path.read_text(encoding="utf-8").strip()


def _from_pretrained(cls: Any, name: str, **kwargs: Any) -> Any:
    """from_pretrained с совместимостью torch_dtype → dtype (transformers 5)."""
    try:
        return cls.from_pretrained(name, **kwargs)
    except TypeError:
        if "torch_dtype" not in kwargs:
            raise
        kwargs = dict(kwargs)
        kwargs["dtype"] = kwargs.pop("torch_dtype")
        return cls.from_pretrained(name, **kwargs)


def _hub_revision(name: str, revision: str | None) -> str | None:
    """Ревизия нужна только для hub-модели; локальному каталогу она мешает."""
    if not revision:
        return None
    try:
        if Path(name).expanduser().exists():
            return None
    except OSError:  # pragma: no cover - экзотические имена путей
        return None
    return revision


def _load_base_model(
    torch: Any,
    name: str,
    *,
    revision: str | None,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    from transformers import AutoModelForCausalLM  # noqa: PLC0415 - тяжёлый импорт

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float32,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    revision = _hub_revision(name, revision)
    if revision:
        kwargs["revision"] = revision
    try:
        model = _from_pretrained(AutoModelForCausalLM, name, **kwargs)
    except (OSError, ValueError) as exc:
        hint = ""
        if local_files_only:
            hint = "\n       --local-files-only: весов нет в кеше, скачайте их заранее"
        raise EvalError(f"не удалось загрузить модель {name}: {exc}{hint}") from exc
    return model


def load_tokenizer(
    *,
    name: str,
    revision: str | None,
    trust_remote_code: bool,
    local_files_only: bool,
) -> Any:
    from transformers import AutoTokenizer  # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    revision = _hub_revision(name, revision)
    if revision:
        kwargs["revision"] = revision
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    except (OSError, ValueError) as exc:
        raise EvalError(f"не удалось загрузить токенизатор {name}: {exc}") from exc
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise EvalError(f"у токенизатора {name} нет ни pad_token, ни eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _quant_config_from_state(state: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Собрать QuantConfig из trainer_state.pt (или дефолт для старых чекпоинтов)."""
    from qat.config import QuantConfig  # noqa: PLC0415

    saved = (state.get("config") or {}).get("quant") or {}
    kwargs: dict[str, Any] = {}
    if "block_size" in saved:
        kwargs["block_size"] = int(saved["block_size"])
    if "scale_dtype" in saved:
        kwargs["scale_dtype"] = str(saved["scale_dtype"])
    if saved.get("target_suffixes"):
        kwargs["target_suffixes"] = tuple(str(x) for x in saved["target_suffixes"])
    if saved.get("exclude_suffixes"):
        kwargs["exclude_suffixes"] = tuple(str(x) for x in saved["exclude_suffixes"])
    if "quantize_embeddings" in saved:
        kwargs["quantize_embeddings"] = str(saved["quantize_embeddings"])
    quant = QuantConfig(**kwargs)
    return quant, {
        "block_size": quant.block_size,
        "quantize_embeddings": quant.quantize_embeddings,
    }


def load_model(
    torch: Any,
    spec: dict[str, Any],
    device: Any,
    *,
    local_files_only: bool,
) -> tuple[Any, dict[str, Any]]:
    """Загрузить HF-модель или чекпоинт QAT и вернуть (model, info)."""
    if spec["kind"] == "hf":
        name = spec["name"]
        model = _load_base_model(
            torch, name, revision=None, trust_remote_code=False,
            local_files_only=local_files_only,
        )
        info: dict[str, Any] = {
            "model": name,
            "base_model": name,
            "revision": None,
            "trust_remote_code": False,
            "checkpoint": None,
            "step": None,
            "quant": None,
            "train_data": None,
        }
    else:
        from qat.checkpoint import (  # noqa: PLC0415
            STATE_FORMAT_VERSION,
            load_trainer_state,
            restore_model,
        )
        from qat.model import apply_embedding_quant, apply_ternary  # noqa: PLC0415

        checkpoint = Path(spec["checkpoint"])
        state = load_trainer_state(checkpoint)
        if state.get("format_version") != STATE_FORMAT_VERSION:
            raise EvalError(
                f"{checkpoint}/trainer_state.pt: версия формата "
                f"{state.get('format_version')}, код ждёт {STATE_FORMAT_VERSION}"
            )
        saved = state.get("config") or {}
        saved_model = saved.get("model") or {}
        name = saved_model.get("name_or_path")
        if not name:
            raise EvalError(
                f"{checkpoint}/trainer_state.pt: нет config.model.name_or_path — "
                f"не знаю, какую базовую модель восстанавливать"
            )
        revision = saved_model.get("revision") or None
        trust = bool(saved_model.get("trust_remote_code", False))
        quant, quant_info = _quant_config_from_state(state)
        print(f"[eval] чекпоинт: {checkpoint} (шаг {state.get('step', '?')})")
        print(f"[eval] базовая модель: {name}"
              + (f" (revision {revision})" if revision else ""))
        print(f"[eval] quant: block_size={quant.block_size}, "
              f"embeddings={quant.quantize_embeddings}")
        if state.get("dry_run"):
            print("[eval] ВНИМАНИЕ: чекпоинт помечен dry_run — метрики не про обучение")
        model = _load_base_model(
            torch, str(name), revision=revision, trust_remote_code=trust,
            local_files_only=local_files_only,
        )
        try:
            reports = apply_ternary(model, quant)
            embeddings = apply_embedding_quant(model, quant)
        except (ValueError, RuntimeError) as exc:
            raise EvalError(f"не удалось навесить квантование чекпоинта: {exc}") from exc
        if embeddings:
            print(f"[eval] эмбеддинги: {embeddings['mode']}, блок "
                  f"{embeddings['block_size']}")
        restore_model(model, checkpoint, device)
        print(f"[eval] веса загружены: тернарных Linear {len(reports)}")
        train_data = (saved.get("data") or {}).get("path")
        info = {
            "model": str(name),
            "base_model": str(name),
            "revision": revision,
            "trust_remote_code": trust,
            "checkpoint": str(checkpoint),
            "step": state.get("step"),
            "quant": quant_info,
            "train_data": train_data,
        }
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    config = getattr(model, "config", None)
    if config is not None:
        config.use_cache = True
    return model, info


# ---------------------------------------------------------------------------
# Генерация и проверки golden
# ---------------------------------------------------------------------------


def _generated_token_count(ids: Iterable[int], eos_ids: set[int], pad_id: int | None) -> int:
    """Сколько токенов модель реально сгенерировала (до EOS, без padding)."""
    count = 0
    for value in ids:
        token = int(value)
        if token in eos_ids:
            break
        if pad_id is not None and token == pad_id:
            continue
        count += 1
    return count


def generate_answers(
    torch: Any,
    model: Any,
    tokenizer: Any,
    validator: Any,
    rows: Sequence[dict[str, Any]],
    system: str,
    device: Any,
    *,
    max_new_tokens: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Greedy-генерация и проверка каждого ответа ``run_eval.check``."""
    context = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    if context and max_new_tokens >= context:
        raise EvalError(
            f"--max-new-tokens {max_new_tokens} >= окна модели {context}: "
            f"промпт не поместится"
        )
    max_prompt_len = max(1, context - max_new_tokens) if context else 4096
    pad_id = int(tokenizer.pad_token_id)
    eos_id = tokenizer.eos_token_id
    eos_ids = {int(eos_id)} if eos_id is not None else set()
    previous_side = tokenizer.padding_side
    previous_truncation = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    results: list[dict[str, Any]] = []
    try:
        for start in range(0, len(rows), batch_size):
            batch = list(rows[start:start + batch_size])
            prompts = [render_prompt(tokenizer, system, row) for row in batch]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_len = int(encoded["input_ids"].shape[1])
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": pad_id,
            }
            if eos_id is not None:
                generate_kwargs["eos_token_id"] = int(eos_id)
            with torch.inference_mode():
                output = model.generate(**encoded, **generate_kwargs)
            new_ids = output[:, prompt_len:]
            sequences = new_ids.tolist()
            texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            for row, text, ids in zip(batch, texts, sequences):
                answer = text.strip()
                try:
                    ok, problems = validator.check(
                        answer, row.get("checks", {}), system
                    )
                except Exception as exc:  # noqa: BLE001 - одна строка не роняет прогон
                    ok, problems = False, [f"ошибка проверки: {exc}"]
                results.append({
                    "id": row["id"],
                    "ok": bool(ok),
                    "problems": list(problems),
                    "answer": answer,
                    "answer_tokens": _generated_token_count(ids, eos_ids, pad_id),
                    "answer_chars": len(answer),
                })
            print(f"[eval] ответы: {len(results)}/{len(rows)}",
                  file=sys.stderr, flush=True)
    finally:
        tokenizer.padding_side = previous_side
        tokenizer.truncation_side = previous_truncation
    return results


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


def compute_perplexity(nll_sum: float, tokens: int) -> float | None:
    """``exp(средний NLL)``; None, если токенов нет или сумма не конечна."""
    if tokens <= 0 or not math.isfinite(nll_sum):
        return None
    try:
        return math.exp(min(700.0, nll_sum / tokens))
    except OverflowError:  # pragma: no cover - защита от экзотических float
        return None


def _ppl_batches(
    samples: Sequence[dict[str, Any]], batch_size: int, token_budget: int
) -> Iterable[list[dict[str, Any]]]:
    """Батчи с ограничением ``batch × длина``: длинные пары идут по одной."""
    batch: list[dict[str, Any]] = []
    longest = 0
    for sample in samples:
        length = len(sample["input_ids"])
        if batch and (
            len(batch) >= batch_size
            or (len(batch) + 1) * max(longest, length) > token_budget
        ):
            yield batch
            batch, longest = [], 0
        batch.append(sample)
        longest = max(longest, length)
    if batch:
        yield batch


def evaluate_perplexity(
    torch: Any,
    model: Any,
    tokenizer: Any,
    pairs: Sequence[dict[str, Any]],
    device: Any,
    *,
    batch_size: int,
    max_seq_len: int,
) -> dict[str, Any]:
    """Perplexity по токенам ответа; промпт замаскирован ``-100``."""
    import torch.nn.functional as F  # noqa: PLC0415

    samples: list[dict[str, Any]] = []
    skipped = 0
    for row in pairs:
        sample = tokenize_pair(tokenizer, row, max_seq_len)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)
    if skipped:
        print(f"[eval] perplexity: пропущено {skipped} пар длиннее {max_seq_len}",
              file=sys.stderr)
    if not samples:
        return {
            "perplexity": None,
            "nll_sum": None,
            "tokens": 0,
            "pairs": 0,
            "skipped": skipped,
        }

    # Сортировка по длине уменьшает padding; на perplexity порядок не влияет.
    samples.sort(key=lambda item: len(item["input_ids"]))
    pad_id = int(tokenizer.pad_token_id)
    vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)
    total_nll = 0.0
    total_tokens = 0
    done = 0
    with torch.inference_mode():
        for batch in _ppl_batches(samples, batch_size, PPL_TOKEN_BUDGET):
            width = max(len(sample["input_ids"]) for sample in batch)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
            labels = torch.full((len(batch), width), -100, dtype=torch.long)
            for index, sample in enumerate(batch):
                length = len(sample["input_ids"])
                input_ids[index, :length] = torch.tensor(sample["input_ids"])
                attention_mask[index, :length] = 1
                labels[index, :length] = torch.tensor(sample["labels"])
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
            if not vocab_size:
                vocab_size = int(logits.size(-1))
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            batch_tokens = int((shift_labels != -100).sum().item())
            expected_tokens = sum(
                int(sample["answer_tokens"]) for sample in batch
            )
            if batch_tokens != expected_tokens:
                # Prompt-часть маскирована -100, поэтому предсказываемых
                # токенов ровно столько, сколько токенов в ответах.
                raise EvalError(
                    f"perplexity: меток {batch_tokens}, токенов ответа "
                    f"{expected_tokens} — ошибка разметки"
                )
            for start in range(0, shift_labels.size(1), PPL_LOGIT_CHUNK):
                part_logits = shift_logits[:, start:start + PPL_LOGIT_CHUNK, :]
                part_labels = shift_labels[:, start:start + PPL_LOGIT_CHUNK]
                if part_logits.dtype != torch.float32:
                    part_logits = part_logits.float()
                loss = F.cross_entropy(
                    part_logits.reshape(-1, part_logits.size(-1)),
                    part_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_nll += float(loss.item())
            total_tokens += batch_tokens
            done += len(batch)
            print(f"[eval] perplexity: {done}/{len(samples)}",
                  file=sys.stderr, flush=True)
            # Явно освобождаем логиты: иначе они доживают до forward
            # следующего батча и память удваивается.
            del logits, shift_logits, shift_labels
    return {
        "perplexity": compute_perplexity(total_nll, total_tokens),
        "nll_sum": total_nll,
        "tokens": total_tokens,
        "pairs": len(samples),
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Сводка и вывод
# ---------------------------------------------------------------------------


def summarize(
    rows: Sequence[dict[str, Any]],
    *,
    perplexity: float | None = None,
    ppl_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Посчитать accuracy и среднюю длину по записям (чистая функция)."""
    total = len(rows)
    passed = sum(1 for row in rows if row.get("ok"))
    lengths = [int(row.get("answer_tokens") or 0) for row in rows]
    chars = [int(row.get("answer_chars") or 0) for row in rows]
    stats = ppl_stats or {}
    return {
        "accuracy": round(passed / total, 6) if total else 0.0,
        "passed": passed,
        "total": total,
        "perplexity": None if perplexity is None else round(float(perplexity), 6),
        "avg_len": round(sum(lengths) / total, 3) if total else 0.0,
        "avg_len_chars": round(sum(chars) / total, 3) if total else 0.0,
        "generated_tokens": sum(lengths),
        "ppl_pairs": int(stats.get("pairs") or 0),
        "ppl_tokens": int(stats.get("tokens") or 0),
        "ppl_skipped": int(stats.get("skipped") or 0),
    }


def format_report(result: dict[str, Any]) -> str:
    """Человекочитаемый отчёт: провалы по id, затем одна итоговая строка."""
    lines: list[str] = []
    for row in result.get("rows") or []:
        if row.get("ok"):
            continue
        problems = "; ".join(row.get("problems") or []) or "проверка не прошла"
        lines.append(f"FAIL {row.get('id')}: {problems}")
    if lines:
        lines.append("")

    perplexity = result.get("perplexity")
    if perplexity is None:
        ppl_text = "perplexity n/a"
    else:
        ppl_text = (
            f"perplexity {float(perplexity):.3f} "
            f"({int(result.get('ppl_pairs') or 0)} пар, "
            f"{int(result.get('ppl_tokens') or 0)} токенов)"
        )
    accuracy = float(result.get("accuracy") or 0.0)
    lines.append(
        f"accuracy {int(result.get('passed') or 0)}/{int(result.get('total') or 0)} "
        f"= {accuracy * 100:.1f}% | {ppl_text} | "
        f"avg_len {float(result.get('avg_len') or 0.0):.1f} ток | "
        f"model {result.get('model') or '—'} -> {result.get('out') or '—'}"
    )
    return "\n".join(lines)


def format_dry_run(plan: dict[str, Any]) -> str:
    """План прогона без torch: что прочитано и что будет запущено."""
    lines = [
        "eval_torch dry-run: torch не импортируется, модель не загружается, "
        "сеть не используется.",
    ]
    golden_line = f"golden: {plan['golden']} — {plan['golden_rows']} вопросов"
    if plan.get("limit") and plan["golden_rows"] < plan["golden_total"]:
        golden_line += f" (--limit: первые {plan['golden_rows']} из {plan['golden_total']})"
    lines.append(golden_line)
    counts = plan.get("check_counts") or {}
    parts = [f"{key} {counts[key]}" for key in KNOWN_CHECKS if counts.get(key)]
    if parts:
        lines.append("правила golden: " + " · ".join(parts))
    unknown = plan.get("unknown_checks") or []
    if unknown:
        lines.append(
            "внимание: неизвестные checks " + ", ".join(unknown)
            + " — eval/run_eval.py их игнорирует"
        )
    lines.append("модель: " + plan["model"])
    lines.append("устройство: " + plan["device_line"])
    lines.append(
        f"генерация: greedy (do_sample=False), max_new_tokens="
        f"{plan['max_new_tokens']}, batch_size={plan['batch_size']}"
    )
    if plan.get("ppl_line"):
        lines.append("perplexity: " + plan["ppl_line"])
    else:
        lines.append(
            "perplexity: не будет посчитана — --ppl-data не задан "
            "(в JSON будет null)"
        )
    lines.append(
        f"план: {plan['golden_rows']} ответов, {plan['ppl_pairs']} пар perplexity "
        f"→ {plan['out']}"
    )
    lines.append(
        "проверки считаются eval/run_eval.py::check; perplexity — по токенам "
        "ответа, как в qat/data.py"
    )
    return "\n".join(lines)


def write_json(path: str | Path, obj: Any) -> Path:
    """Записать JSON (с созданием каталога) атомарно через .tmp."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


# ---------------------------------------------------------------------------
# Прогон
# ---------------------------------------------------------------------------


def _check_train_data_overlap(info: dict[str, Any], ppl_path: str | None) -> None:
    """Предупредить, если perplexity считается на данных обучения."""
    if not ppl_path or not info.get("train_data"):
        return
    train = Path(str(info["train_data"])).expanduser()
    if not train.is_absolute():
        train = ROOT / train
    try:
        same = train.resolve() == Path(ppl_path).expanduser().resolve()
    except OSError:
        same = False
    if same:
        print(
            "[eval] ВНИМАНИЕ: --ppl-data совпадает с данными обучения чекпоинта; "
            "perplexity будет оптимистичной — для честного замера нужен отложенный файл",
            file=sys.stderr,
        )


def run_evaluation(
    rows: Sequence[dict[str, Any]],
    spec: dict[str, Any],
    pairs: Sequence[dict[str, Any]] | None,
    *,
    device_name: str,
    max_new_tokens: int,
    batch_size: int,
    max_seq_len: int,
    local_files_only: bool,
    system: str,
    validators: Any,
    ppl_path: str | None = None,
) -> dict[str, Any]:
    """Тяжёлая часть: загрузка модели, ответы, perplexity. Нужен torch."""
    import torch  # noqa: PLC0415 - тяжёлый импорт только здесь

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = resolve_device(torch, device_name)
    device_label = str(device)
    if device.type == "cuda":
        device_label += f" ({torch.cuda.get_device_name(device)})"
    print(f"[eval] устройство: {device_label}")

    model, info = load_model(
        torch, spec, device, local_files_only=local_files_only
    )
    tokenizer = load_tokenizer(
        name=info["base_model"],
        revision=info.get("revision"),
        trust_remote_code=bool(info.get("trust_remote_code", False)),
        local_files_only=local_files_only,
    )
    _check_train_data_overlap(info, ppl_path)

    print(f"[eval] golden: {len(rows)} вопросов, систему берём из prompts/system.ru.md")
    results = generate_answers(
        torch, model, tokenizer, validators, rows, system, device,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
    )

    ppl_stats: dict[str, Any] | None = None
    if pairs:
        print(f"[eval] perplexity: {len(pairs)} пар, max_seq_len={max_seq_len}",
              file=sys.stderr)
        ppl_stats = evaluate_perplexity(
            torch, model, tokenizer, pairs, device,
            batch_size=batch_size, max_seq_len=max_seq_len,
        )
    summary = summarize(
        results,
        perplexity=(ppl_stats or {}).get("perplexity"),
        ppl_stats=ppl_stats,
    )
    summary.update({
        "model": info["model"],
        "base_model": info["base_model"],
        "checkpoint": info.get("checkpoint"),
        "step": info.get("step"),
        "quant": info.get("quant"),
        "device": device_label,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "rows": list(results),
    })
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        all_rows = load_golden(args.golden)
    except (EvalError, OSError) as exc:
        print(f"[eval] {exc}", file=sys.stderr)
        return 2
    rows = all_rows[: args.limit] if args.limit else all_rows
    if not rows:
        print("[eval] после --limit не осталось вопросов", file=sys.stderr)
        return 2

    try:
        spec = resolve_model(args.model)
    except (EvalError, OSError) as exc:
        print(f"[eval] {exc}", file=sys.stderr)
        return 2

    pairs_meta: dict[str, Any] | None = None
    if args.ppl_data:
        try:
            pairs_meta = load_pairs(args.ppl_data, args.ppl_limit)
        except (EvalError, OSError) as exc:
            print(f"[eval] {exc}", file=sys.stderr)
            return 2

    out_path = Path(args.out).expanduser()

    if args.dry_run:
        ppl_line = ""
        ppl_pairs = 0
        if pairs_meta is not None:
            ppl_pairs = len(pairs_meta["rows"])
            ppl_line = (
                f"{args.ppl_data} — {ppl_pairs} пар из {pairs_meta['total']}"
                f" (равномерно; --ppl-limit {args.ppl_limit})"
                if args.ppl_limit
                else f"{args.ppl_data} — все {ppl_pairs} пар"
            )
            if pairs_meta["skipped"]:
                ppl_line += f", пропущено строк без question/answer: {pairs_meta['skipped']}"
        plan = {
            "golden": str(args.golden),
            "golden_rows": len(rows),
            "golden_total": len(all_rows),
            "limit": args.limit,
            "check_counts": check_rule_counts(rows),
            "unknown_checks": unknown_check_keys(rows),
            "model": describe_model_spec(spec),
            "device_line": (
                f"{args.device} (cuda → mps → cpu выберется при запуске)"
                if args.device == "auto"
                else f"{args.device} (доступность проверится при запуске)"
            ),
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "ppl_line": ppl_line,
            "ppl_pairs": ppl_pairs,
            "out": str(out_path),
        }
        print(format_dry_run(plan))
        return 0

    missing = missing_modules()
    if missing:
        print_missing_dependencies(missing)
        return 2

    try:
        validators = load_validators()
        system = read_system_prompt()
        result = run_evaluation(
            rows,
            spec,
            pairs_meta["rows"] if pairs_meta else None,
            device_name=args.device,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            local_files_only=args.local_files_only,
            system=system,
            validators=validators,
            ppl_path=args.ppl_data,
        )
    except EvalError as exc:
        print(f"[eval] {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"[eval] ошибка ввода-вывода: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[eval] прервано", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI не должен пугать трейсбеком
        print(f"[eval] ошибка прогона: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.traceback:
            import traceback  # noqa: PLC0415

            traceback.print_exc()
        return 1

    result["golden"] = str(args.golden)
    result["ppl_data"] = args.ppl_data
    result["out"] = str(out_path)
    try:
        write_json(out_path, result)
    except OSError as exc:
        print(f"[eval] не удалось записать {out_path}: {exc}", file=sys.stderr)
        return 1
    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
