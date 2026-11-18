#!/usr/bin/env python3
"""Скачать внешние датасеты и привести к нашему формату `{question, answer}`.

Зачем: у нас 4008 своих пар по программированию, но бизнесу нужны ещё код, общие
инструкции и сценарии. Публичные наборы лежат в разных форматах и лицензиях —
этот инструмент берёт только разрешённые лицензии и приводит всё к одному виду.

    PYTHONPATH=.pylibs python3.13 tools/pull_datasets.py --source russian_code_qa --limit 500
    PYTHONPATH=.pylibs python3.13 tools/pull_datasets.py --list

Политика лицензий: пускаем MIT, Apache-2.0, CC-BY-4.0, CC-BY-SA-4.0; не пускаем
NC (некоммерческие) — продукт коммерческий. Решение зашито в ALLOWED_LICENSES.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from distill import quality  # noqa: E402 — фильтр качества из общего модуля

OUT_DIR = ROOT / "distill" / "pulled"

#: Лицензии, которые можно использовать в коммерческом продукте.
ALLOWED_LICENSES = {"mit", "apache-2.0", "cc-by-4.0", "cc-by-sa-4.0", "odc-by", "unknown"}
#: Явно запрещённые (NC и прочее, что нельзя в продукт).
DENIED_LICENSES = {"cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0", "gpl-3.0"}


@dataclass(frozen=True)
class Source:
    """Описание источника: что качать и как из него достать пару."""

    key: str
    repo: str
    license: str
    language: str
    kind: str  # code | general | business | behavior
    fields: tuple[str, ...]  # (question, answer) в терминах датасета
    note: str
    extra: dict[str, Any] = field(default_factory=dict)
    raw_url: str = ""  # прямой JSONL-файл, если datasets-хаб качает его медленно


SOURCES: dict[str, Source] = {
    "russian_code_qa": Source(
        key="russian_code_qa",
        repo="Den4ikAI/russian_code_qa",
        license="mit",
        language="ru",
        kind="code",
        fields=("question", "code"),
        note="русские вопросы + сниппет кода, MIT, крупный",
    ),
    "ru_codefeedback": Source(
        key="ru_codefeedback",
        repo="mizinovmv/ru_codefeedback",
        license="apache-2.0",
        language="ru",
        kind="code",
        fields=("ru_query", "answer"),
        note="русский Code-Feedback: Q&A + unit-тесты, apache-2.0",
    ),
    "russian_instructions": Source(
        key="russian_instructions",
        repo="Lev384501/russian_instructions_2_cleaned",
        license="mit",
        language="ru",
        kind="general",
        fields=("messages",),
        note="очищенные русские инструкции, MIT, chat-формат",
        raw_url=("https://huggingface.co/datasets/Lev384501/"
                 "russian_instructions_2_cleaned/resolve/main/dataset.jsonl"),
    ),
    "easy_instructions": Source(
        key="easy_instructions",
        repo="attn-signs/russian-easy-instructions",
        license="apache-2.0",
        language="ru",
        kind="general",
        fields=("messages",),
        note="базовые знания и короткие ответы, apache-2.0",
    ),
    "magicoder": Source(
        key="magicoder",
        repo="ise-uiuc/Magicoder-OSS-Instruct-75K",
        license="mit",
        language="en",
        kind="code",
        fields=("problem", "solution"),
        note="английский код, MIT — переводить учителем",
    ),
    "codealpaca": Source(
        key="codealpaca",
        repo="sahil2801/CodeAlpaca-20k",
        license="cc-by-4.0",
        language="en",
        kind="code",
        fields=(),
        note="английский код, cc-by; поля instruction/input/output",
    ),
}

_CYR = re.compile(r"[а-яА-ЯёЁ]")
_LINK = re.compile(r"https?://|www\.")


def _check_license(src: Source, actual: str | None) -> str:
    """Проверить лицензию: вернуть решение или упасть."""
    lic = (actual or src.license or "unknown").lower().strip()
    if lic in DENIED_LICENSES:
        raise SystemExit(
            f"{src.repo}: лицензия {lic} запрещена для коммерческого продукта"
        )
    if lic not in ALLOWED_LICENSES:
        raise SystemExit(
            f"{src.repo}: лицензия {lic!r} неизвестна политике; "
            f"разрешены {sorted(ALLOWED_LICENSES)}"
        )
    return lic


def _messages_to_pair(messages: Any) -> tuple[str, str] | None:
    """chat-формат `[{role, content}, ...]` → (вопрос, ответ)."""
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    user = next(
        (m.get("content") for m in messages
         if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    assistant = next(
        (m.get("content") for m in messages
         if isinstance(m, dict) and m.get("role") == "assistant"),
        None,
    )
    if isinstance(user, str) and isinstance(assistant, str) and user.strip() and assistant.strip():
        return user, assistant
    return None


#: Пары полей, которые встречаются в публичных датасетах, в порядке предпочтения.
_PAIR_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("ru_query", "answer"),
    ("question", "answer"),
    ("instruction", "output"),
    ("problem", "solution"),
    ("prompt", "response"),
    ("question", "code"),
)


def _field_pair(row: dict[str, Any], src: Source) -> tuple[str, str] | None:
    """Достать пару из строки. Схемы у наборов разные — распознаём по колонкам."""
    if "messages" in row:
        return _messages_to_pair(row.get("messages"))
    # CodeAlpaca держит вопрос в instruction + input.
    if "instruction" in row and "output" in row:
        instruction = str(row.get("instruction") or "").strip()
        extra = str(row.get("input") or "").strip()
        output = str(row.get("output") or "").strip()
        if not instruction or not output:
            return None
        question = f"{instruction}\n\n{extra}" if extra else instruction
        return question, _wrap_code(output)
    for q_key, a_key in _PAIR_CANDIDATES:
        if q_key in row and a_key in row:
            q = str(row.get(q_key) or "").strip()
            a = str(row.get(a_key) or "").strip()
            if not q or not a:
                return None
            return q, _wrap_code(a)
    return None


def _wrap_code(answer: str) -> str:
    """Голый сниппет кода обернуть в блок — так модель учится отвечать кодом."""
    if answer.startswith("```") or "\n" not in answer:
        return answer
    looks_code = any(t in answer for t in ("def ", "import ", "SELECT ", "function ", "class ", "(){", "=>"))
    if looks_code and not _CYR.search(answer):
        return f"```\n{answer.strip()}\n```"
    return answer


def _clean(text: str, *, max_chars: int = 8000) -> str:
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text[:max_chars]


def _iter_raw_jsonl(src: Source, limit: int) -> Iterator[dict[str, Any]]:
    """Потоковое чтение прямого JSONL по URL — без Xet и без полной загрузки."""
    import urllib.request  # noqa: PLC0415

    with urllib.request.urlopen(src.raw_url, timeout=120) as resp:
        emitted = 0
        for raw in resp:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            pair = _field_pair(row, src)
            if pair is None:
                continue
            question, answer = _clean(pair[0]), _clean(pair[1])
            if len(question) < 10 or len(answer) < 20:
                continue
            if _LINK.search(answer) or not quality.accept_pair(question, answer):
                continue
            yield {
                "question": question,
                "answer": answer,
                "meta": {
                    "source": src.key,
                    "repo": src.repo,
                    "license": src.license,
                    "language": src.language,
                    "kind": src.kind,
                    "field": "pulled",
                },
            }
            emitted += 1
            if emitted >= limit:
                return


def iter_pairs(src: Source, limit: int) -> Iterator[dict[str, Any]]:
    """Читать датасет и отдавать нормализованные пары."""
    if src.raw_url and limit > 0:
        yield from _iter_raw_jsonl(src, limit)
        return
    try:
        from datasets import load_dataset  # noqa: PLC0415 — тяжёлый импорт
    except ImportError as exc:  # pragma: no cover - без библиотеки не работаем
        raise SystemExit(
            "нет библиотеки datasets: PYTHONPATH=.pylibs python3.13 ..."
        ) from exc

    streaming = limit > 0
    ds = load_dataset(src.repo, split="train", streaming=streaming)
    actual = None
    try:
        info = ds.info  # DatasetInfo
        actual = getattr(info, "license", None) or (info.card_metadata or {}).get("license")
    except Exception:  # noqa: BLE001 — метаданные не критичны, берём из SOURCES
        actual = None
    license_used = _check_license(src, actual)

    emitted = 0
    for row in ds:
        pair = _field_pair(row, src)
        if pair is None:
            continue
        question, answer = _clean(pair[0]), _clean(pair[1])
        min_answer = 8 if src.kind == "code" else 20
        if len(question) < 10 or len(answer) < min_answer:
            continue
        if _LINK.search(answer):
            continue
        if not quality.accept_pair(question, answer):
            continue
        # Русскость проверяем по вопросу; ответ может быть кодом без кириллицы.
        if src.language == "ru" and not _CYR.search(question):
            continue
        if src.kind != "code" and src.language == "ru" and len(_CYR.findall(answer)) < 20:
            continue
        yield {
            "question": question,
            "answer": answer,
            "meta": {
                "source": src.key,
                "repo": src.repo,
                "license": license_used,
                "language": src.language,
                "kind": src.kind,
                "field": "pulled",
            },
        }
        emitted += 1
        if limit and emitted >= limit:
            break


def _dedupe_key(question: str) -> str:
    return re.sub(r"[^а-яa-z0-9]+", "", question.lower())[:200]


def pull(key: str, *, limit: int, out: pathlib.Path) -> int:
    src = SOURCES[key]
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for pair in iter_pairs(src, limit):
            k = _dedupe_key(pair["question"])
            if not k or k in seen:
                continue
            seen.add(k)
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
            written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES))
    parser.add_argument("--list", action="store_true", help="показать источники")
    parser.add_argument("--limit", type=int, default=0, help="0 — все строки")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    if args.list or not args.source:
        print(f"{'ключ':22s} {'лицензия':12s} {'язык':5s} {'вид':9s} репозиторий")
        for s in SOURCES.values():
            print(f"{s.key:22s} {s.license:12s} {s.language:5s} {s.kind:9s} {s.repo}")
        return 0

    out = args.out or (OUT_DIR / f"{args.source}.jsonl")
    n = pull(args.source, limit=args.limit, out=out)
    print(f"{args.source}: {n} пар → {out}")
    if n == 0:
        print("предупреждение: ни одной пары не прошло фильтры", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
