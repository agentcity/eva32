#!/usr/bin/env python3
"""Подготовка данных для QAT (задача 0092): очистка, train/val, предтокенизация.

Что делает:

1. Читает ``distill/out/code_pairs.jsonl`` (поля ``question``/``answer``/``meta``),
   выбрасывает пустые и битые строки.
2. Дедуплицирует по нормализованному вопросу: NFKC, нижний регистр, пунктуация
   заменяется пробелом, пробелы схлопываются. Из одинаковых вопросов остаётся
   первый — порядок входа детерминирован.
3. Делит пары на train/val 90/10. Номер половины задаёт SHA-256 от
   ``seed + нормализованный вопрос``, а не порядок строк: разбиение не зависит
   ни от версии Python (у ``random.shuffle`` свои гарантии), ни от порядка строк
   в исходном файле. Пересечение вопросов train и val проверяется и обязано
   быть нулевым.
4. Считает токены тем самым BPE из ``tokenizer.json``, который использует
   ``src/vtm`` (тот же файл, тот же byte-level BPE и GPT-2 pre-tokenizer).
   Python-обёртки над ``src/vtm`` в репозитории нет, поэтому BPE разобран
   напрямую из JSON; паритет с ``src/vtm/vtm-tok`` проверяется тестом.
5. Пишет ``data/train.jsonl``, ``data/val.jsonl`` и ``data/stats.json``:
   число пар и токенов, распределение длин (p50/p95/max), сколько пар длиннее
   ``max_seq_len`` (в трейнере ``qat/data.py`` такие пары отбрасываются), оценку
   шагов для 1.5B (micro-batch 4) и 3B (micro-batch 2) на 1–3 эпохах.

Границы: скрипт ничего не обучает, не импортирует torch и не ходит в сеть.

Запуск::

    python3 tools/prepare_qat_data.py
    python3 tools/prepare_qat_data.py --tokenizer /path/Qwen2.5-1.5B-Instruct/tokenizer.json
    python3 tools/prepare_qat_data.py --reference-tokenizer ~/.cache/.../gpt2/tokenizer.json

Оговорка про токенизатор: в репозитории лежит только фикстура
``tests/vtm/fixtures/tokenizer.json`` (287 токенов, создана для тестов
в задаче 0012). Реальный ``tokenizer.json`` Qwen в репозиторий не кладут
(объём ~11 МБ), поэтому числа на фикстуре — верхняя граница: каждый байт
кириллицы там почти всегда отдельный токен. Для планирования GPU-прогона
перезапустите скрипт с реальным ``tokenizer.json`` Qwen2.5 через ``--tokenizer``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PAIRS = ROOT / "distill" / "out" / "code_pairs.jsonl"
DEFAULT_OUT_DIR = ROOT / "data"
FIXTURE_TOKENIZER = ROOT / "tests" / "vtm" / "fixtures" / "tokenizer.json"

# Контекст и разбиение из постановки задачи.
MAX_SEQ_LEN = 2048
VAL_FRACTION = 0.1
SEED = 42

# Профили из qat/configs/*.yaml: micro_batch_size и grad_accum_steps.
# Эффективный батч у обоих 32, поэтому шагов одинаково, а micro-batch разный.
PROFILES: dict[str, dict[str, Any]] = {
    "qwen1.5b": {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "micro_batch": 4,
        "grad_accum": 8,
    },
    "qwen3b": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "micro_batch": 2,
        "grad_accum": 16,
    },
}
EPOCHS = (1, 2, 3)

# max_steps из qat/configs/*.yaml: нужен только для предупреждения, хватает ли
# 1–3 эпох, чтобы его набрать.
CONFIG_MAX_STEPS = 2000

# Qwen2.5-Instruct: qat/data.py зовёт apply_chat_template с одним сообщением
# user и add_generation_prompt=True. Шаблон Qwen2.5 при отсутствии system
# подставляет стандартную системную строку — она здесь и воспроизведена.
# .format() не используется: в тексте вопроса могут быть фигурные скобки.
QWEN_CHAT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "{question}"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)

_VALID_PROMPT_STYLES = ("qwen", "plain")
_WS_RE = re.compile(r"\s+")


class DataError(RuntimeError):
    """Данные не читаются или не годятся для QAT."""


class TokenizerError(RuntimeError):
    """tokenizer.json не читается или использует неподдержанную функцию."""


# ---------------------------------------------------------------------------
# Очистка и разбиение
# ---------------------------------------------------------------------------


def normalize_question(text: str) -> str:
    """Ключ дедупликации: регистр, пунктуация и пробелы не различаются.

    NFKC приводит типографские варианты к одной форме, casefold — к нижнему
    регистру (в том числе для «Ё»). Пунктуация заменяется пробелом, а не
    удаляется: иначе «тест?» и «тест» склеились бы с «тест»+«...», а слова —
    друг с другом.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    chars: list[str] = []
    for ch in folded:
        if unicodedata.category(ch).startswith("P"):
            chars.append(" ")
        else:
            chars.append(ch)
    return _WS_RE.sub(" ", "".join(chars)).strip()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pairs(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Прочитать JSONL, почистить и дедуплицировать по нормализованному вопросу."""
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"файл пар не найден: {path}\n"
            "соберите пары: make distill-mock (проверка) или make distill (учитель)"
        )
    stats: dict[str, Any] = {
        "path": display_path(path),
        "sha256": file_sha256(path),
        "rows": 0,
        "skipped_invalid": 0,
        "skipped_empty": 0,
        "duplicates_removed": 0,
        "pairs": 0,
        "invalid_lines": [],
    }
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            stats["rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["skipped_invalid"] += 1
                if len(stats["invalid_lines"]) < 10:
                    stats["invalid_lines"].append(f"{lineno}: {exc.msg}")
                continue
            if not isinstance(row, dict):
                stats["skipped_invalid"] += 1
                if len(stats["invalid_lines"]) < 10:
                    stats["invalid_lines"].append(f"{lineno}: ожидался объект JSON")
                continue
            question = row.get("question")
            answer = row.get("answer")
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(answer, str)
                or not answer.strip()
            ):
                stats["skipped_empty"] += 1
                continue
            question = question.strip()
            answer = answer.strip()
            key = normalize_question(question)
            if not key:
                stats["skipped_empty"] += 1
                continue
            if key in seen:
                stats["duplicates_removed"] += 1
                continue
            seen.add(key)
            record: dict[str, Any] = {
                "question": question,
                "answer": answer,
                "_key": key,
            }
            meta = row.get("meta")
            if isinstance(meta, dict):
                record["meta"] = meta
            pairs.append(record)

    stats["pairs"] = len(pairs)
    if not pairs:
        raise DataError(f"в {path} нет пригодных пар question/answer")
    return pairs, stats


def _split_digest(question_key: str, seed: int) -> str:
    """Детерминированный «жребий»: SHA-256(seed + нормализованный вопрос)."""
    return hashlib.sha256(f"{seed}\x00{question_key}".encode("utf-8")).hexdigest()


def _question_key(row: dict[str, Any]) -> str:
    key = row.get("_key")
    if isinstance(key, str):
        return key
    return normalize_question(str(row["question"]))


def split_pairs(
    pairs: Sequence[dict[str, Any]],
    *,
    seed: int = SEED,
    val_fraction: float = VAL_FRACTION,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Разбить пары на train/val по хешу вопроса; пересечение — ошибка."""
    if not 0.0 < val_fraction < 1.0:
        raise DataError(f"val_fraction должен быть в (0, 1), получено {val_fraction}")
    total = len(pairs)
    if total < 2:
        raise DataError(f"для train/val нужно минимум 2 пары, есть {total}")
    n_val = max(1, min(total - 1, int(round(total * val_fraction))))

    question_keys = [_question_key(row) for row in pairs]
    ranked = sorted(
        range(total),
        key=lambda i: (_split_digest(question_keys[i], seed), i),
    )
    val_indices = set(ranked[:n_val])
    train = [row for i, row in enumerate(pairs) if i not in val_indices]
    val = [row for i, row in enumerate(pairs) if i in val_indices]

    train_keys = {question_keys[i] for i, row in enumerate(pairs) if i not in val_indices}
    val_keys = {question_keys[i] for i, row in enumerate(pairs) if i in val_indices}
    overlap_normalized = len(train_keys & val_keys)
    overlap_raw = len({row["question"] for row in train} & {row["question"] for row in val})
    if overlap_normalized or overlap_raw:
        raise DataError(
            "утечка train/val: "
            f"нормализованных вопросов {overlap_normalized}, "
            f"точных {overlap_raw}"
        )

    stats = {
        "method": "sha256(seed + '\\x00' + normalize_question(question))",
        "seed": seed,
        "val_fraction": val_fraction,
        "pairs": total,
        "train_pairs": len(train),
        "val_pairs": len(val),
        "train_fraction": round(len(train) / total, 6),
        "val_actual_fraction": round(len(val) / total, 6),
        "overlap_normalized": overlap_normalized,
        "overlap_raw": overlap_raw,
    }
    return train, val, stats


# ---------------------------------------------------------------------------
# BPE-токенизатор: прямой разбор tokenizer.json, как в src/vtm/tokenizer.cpp
# ---------------------------------------------------------------------------

# Классы символов из src/vtm/tokenizer.cpp (приближение \p{L}, \p{N}, \s):
# полноценные Unicode-таблицы в рантайм не тянутся, набор диапазонов совпадает
# с C++, чтобы Python-счётчик и vtm-tok давали одинаковые id.
_SPACE_CODEPOINTS = frozenset(
    [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680]
    + list(range(0x2000, 0x200B))
    + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
)

_LETTER_RANGES = (
    (0x41, 0x5A), (0x61, 0x7A),
    (0xAA, 0xAA), (0xB5, 0xB5), (0xBA, 0xBA),
    (0xC0, 0x2AF), (0x370, 0x3FF), (0x400, 0x52F), (0x531, 0x58F),
    (0x5D0, 0x5EA), (0x620, 0x64A), (0x900, 0x97F), (0xE01, 0xE5B),
    (0x10A0, 0x10FF), (0x1E00, 0x1FFF), (0x2C60, 0x2C7F),
    (0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
    (0xA640, 0xA69F), (0xAC00, 0xD7AF), (0xF900, 0xFAFF),
    (0xFF21, 0xFF3A), (0xFF41, 0xFF5A), (0x20000, 0x2FA1F),
)

_DIGIT_RANGES = (
    (0x30, 0x39), (0x660, 0x669), (0x6F0, 0x6F9),
    (0x966, 0x96F), (0xFF10, 0xFF19),
)

_APOSTROPHE_SUFFIXES = ("s", "t", "re", "ve", "m", "ll", "d")


def _is_space(ch: str) -> bool:
    return ord(ch) in _SPACE_CODEPOINTS


def _is_letter(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _LETTER_RANGES)


def _is_digit(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _DIGIT_RANGES)


def _is_other(ch: str) -> bool:
    return not _is_space(ch) and not _is_letter(ch) and not _is_digit(ch)


def _bytes_to_unicode() -> tuple[dict[int, str], dict[str, int]]:
    """Таблица GPT-2/Qwen: байт -> Unicode-символ byte-level словаря."""
    bs = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    used = set(bs)
    cs = list(bs)
    extra = 0
    for byte in range(256):
        if byte not in used:
            bs.append(byte)
            cs.append(256 + extra)
            extra += 1
    byte_to_char = {b: chr(c) for b, c in zip(bs, cs)}
    char_to_byte = {chr(c): b for b, c in zip(bs, cs)}
    return byte_to_char, char_to_byte


_BYTE_TO_CHAR, _CHAR_TO_BYTE = _bytes_to_unicode()


def _unicode_normalize(text: str, spec: Any) -> str:
    """Подмножество normalizer из tokenizer.json (у GPT-2/Qwen его нет)."""
    if spec is None:
        return text
    if isinstance(spec, list):
        for step in spec:
            text = _unicode_normalize(text, step)
        return text
    if not isinstance(spec, dict):
        raise TokenizerError(f"normalizer: ожидался объект или null, есть {type(spec).__name__}")
    kind = spec.get("type")
    form = {"NFC": "NFC", "NFD": "NFD", "NFKC": "NFKC", "NFKD": "NFKD"}.get(kind or "")
    if form:
        return unicodedata.normalize(form, text)
    if kind == "Lowercase":
        return text.lower()
    if kind == "Prepend":
        return str(spec.get("prepend", "")) + text
    if kind == "Strip":
        return text.strip(str(spec.get("strip", "")) or None)
    if kind == "Replace":
        pattern = spec.get("pattern")
        content = spec.get("content", "")
        if isinstance(pattern, dict) and "String" in pattern:
            return text.replace(str(pattern["String"]), str(content))
        raise TokenizerError("normalizer Replace: поддерживается только pattern.String")
    if kind == "Sequence":
        for step in spec.get("normalizers", []):
            text = _unicode_normalize(text, step)
        return text
    raise TokenizerError(f"normalizer {kind!r} не поддерживается")


def _parse_pre_tokenizer(spec: Any) -> dict[str, Any]:
    """ByteLevel pre-tokenizer (или Sequence, где ByteLevel есть)."""
    result = {"byte_level": False, "use_regex": False, "add_prefix_space": False}
    if spec is None:
        return result
    if not isinstance(spec, dict):
        raise TokenizerError("pre_tokenizer: ожидался объект или null")
    kind = spec.get("type")
    if kind == "ByteLevel":
        result["byte_level"] = True
        result["use_regex"] = bool(spec.get("use_regex", True))
        result["add_prefix_space"] = bool(spec.get("add_prefix_space", False))
        return result
    if kind == "Sequence":
        for child in spec.get("pretokenizers", []):
            parsed = _parse_pre_tokenizer(child)
            if parsed["byte_level"]:
                return parsed
        return result
    raise TokenizerError(
        f"pre_tokenizer {kind!r} не поддерживается: счёт токенов разошёлся бы с vtm"
    )


class BPETokenizer:
    """Минимальный byte-level BPE из tokenizer.json (vocab + merges + added_tokens).

    Повторяет ``src/vtm/tokenizer.cpp``: GPT-2 regex-разбиение, поиск added_tokens
    подстрокой до BPE, слияния по рангу (при равном ранге — левое вхождение).
    ``decode`` не реализован: для подсчёта длин он не нужен.
    """

    def __init__(
        self,
        *,
        vocab: dict[str, int],
        merges: Iterable[Any],
        added_tokens: Iterable[dict[str, Any]] = (),
        normalizer: Any = None,
        byte_level: bool = True,
        use_regex: bool = True,
        add_prefix_space: bool = False,
        eos_token: str | None = None,
        source: str = "<memory>",
    ) -> None:
        self.source = source
        self.vocab = dict(vocab)
        self.merge_rank: dict[str, int] = {}
        self.merge_count = 0
        for rank, item in enumerate(merges):
            a, b = self._parse_merge(item)
            self.merge_rank.setdefault(a + b, rank)
            self.merge_count += 1
        self.added: list[tuple[str, int]] = []
        self.special_ids: set[int] = set()
        for token in added_tokens or ():
            if not isinstance(token, dict):
                continue
            content = token.get("content")
            token_id = token.get("id")
            if not isinstance(content, str) or not isinstance(token_id, int):
                continue
            self.added.append((content, token_id))
            if token.get("special"):
                self.special_ids.add(token_id)
        # Длинные added_tokens проверяются раньше: совпадение longest-match.
        self.added.sort(key=lambda pair: (-len(pair[0]), pair[1]))
        self._normalizer_spec = normalizer
        self.byte_level = byte_level
        self.use_regex = use_regex
        self.add_prefix_space = add_prefix_space
        self.eos_id = self._resolve_eos(eos_token)

    # -- разбор файла ---------------------------------------------------

    @staticmethod
    def _parse_merge(item: Any) -> tuple[str, str]:
        if isinstance(item, str):
            parts = item.split(" ")
            if len(parts) != 2:
                raise TokenizerError(f"merge {item!r}: ожидались два токена через пробел")
            return parts[0], parts[1]
        if isinstance(item, (list, tuple)) and len(item) == 2:
            return str(item[0]), str(item[1])
        raise TokenizerError(f"merge {item!r}: неподдерживаемый формат")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        path = Path(path)
        if not path.exists():
            raise TokenizerError(f"tokenizer.json не найден: {path}")
        try:
            with path.open(encoding="utf-8") as fh:
                root = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise TokenizerError(f"{path}: не читается tokenizer.json: {exc}") from exc
        if not isinstance(root, dict):
            raise TokenizerError(f"{path}: ожидался объект JSON")
        model = root.get("model")
        if not isinstance(model, dict):
            raise TokenizerError(f"{path}: нет model")
        model_type = model.get("type")
        # В старых tokenizer.json (например, gpt2) model.type не записан:
        # наличие vocab/merges и есть BPE.
        if model_type is not None and model_type != "BPE":
            raise TokenizerError(f"{path}: model.type={model_type!r}, поддерживается только 'BPE'")
        vocab = model.get("vocab")
        if not isinstance(vocab, dict):
            raise TokenizerError(f"{path}: нет model.vocab")
        merges = model.get("merges") or []
        if not isinstance(merges, list):
            raise TokenizerError(f"{path}: model.merges должен быть списком")
        pre = _parse_pre_tokenizer(root.get("pre_tokenizer"))
        return cls(
            vocab=vocab,
            merges=merges,
            added_tokens=root.get("added_tokens") or [],
            normalizer=root.get("normalizer"),
            byte_level=pre["byte_level"],
            use_regex=pre["use_regex"],
            add_prefix_space=pre["add_prefix_space"],
            eos_token=model.get("eos_token") if isinstance(model.get("eos_token"), str) else None,
            source=str(path),
        )

    # -- свойства -------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int | None:
        if token in self.vocab:
            return self.vocab[token]
        for content, token_id in self.added:
            if content == token:
                return token_id
        return None

    def _resolve_eos(self, eos_token: str | None) -> int | None:
        if eos_token:
            found = self.token_to_id(eos_token)
            if found is not None:
                return found
        for content, token_id in self.added:
            low = content.lower()
            if "im_end" in low or "endoftext" in low or "eot_id" in low or "eos" in low:
                return token_id
        return None

    # -- кодирование ----------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """id токенов; added_tokens ищутся в исходном тексте, до нормализации."""
        if not text:
            return []
        out: list[int] = []
        start = 0
        pos = 0
        while pos < len(text):
            hit = self._match_added(text, pos)
            if hit is not None:
                content, token_id = hit
                if pos > start:
                    out.extend(self._encode_segment(text[start:pos]))
                out.append(token_id)
                pos += len(content)
                start = pos
            else:
                pos += 1
        if start < len(text):
            out.extend(self._encode_segment(text[start:]))
        return out

    def _match_added(self, text: str, pos: int) -> tuple[str, int] | None:
        for content, token_id in self.added:
            if text.startswith(content, pos):
                return content, token_id
        return None

    def _encode_segment(self, segment: str) -> list[int]:
        if self.add_prefix_space and segment and not segment.startswith(" "):
            segment = " " + segment
        return self._encode_ordinary(_unicode_normalize(segment, self._normalizer_spec))

    def _encode_ordinary(self, text: str) -> list[int]:
        if not text:
            return []
        if not self.use_regex:
            return self._bpe(text)
        ids: list[int] = []
        for piece in self.pre_tokenize(text):
            ids.extend(self._bpe(piece))
        return ids

    def pre_tokenize(self, text: str) -> list[str]:
        """GPT-2 regex-разбиение — порт ``Tokenizer::pre_tokenize`` из C++."""
        pieces: list[str] = []
        n = len(text)
        i = 0
        while i < n:
            ch = text[i]

            # 's|'t|'re|'ve|'m|'ll|'d
            if ch == "'":
                matched = False
                for suffix in _APOSTROPHE_SUFFIXES:
                    if text.startswith(suffix, i + 1):
                        pieces.append(text[i:i + 1 + len(suffix)])
                        i += 1 + len(suffix)
                        matched = True
                        break
                if matched:
                    continue

            lead_space = ch == " "
            start_after_space = i + 1 if lead_space else i

            if start_after_space < n:
                nxt = text[start_after_space]
                if _is_letter(nxt) or _is_digit(nxt) or _is_other(nxt):
                    if _is_letter(nxt):
                        end = self._consume_run(text, start_after_space, _is_letter)
                    elif _is_digit(nxt):
                        end = self._consume_run(text, start_after_space, _is_digit)
                    else:
                        end = self._consume_run(text, start_after_space, _is_other)
                    pieces.append(text[i:end])
                    i = end
                    continue

            if not lead_space:
                if _is_letter(ch):
                    end = self._consume_run(text, i, _is_letter)
                    pieces.append(text[i:end])
                    i = end
                    continue
                if _is_digit(ch):
                    end = self._consume_run(text, i, _is_digit)
                    pieces.append(text[i:end])
                    i = end
                    continue
                if not _is_space(ch):
                    end = self._consume_run(text, i, _is_other)
                    pieces.append(text[i:end])
                    i = end
                    continue

            if _is_space(ch):
                k = i
                last_start = i
                while k < n and _is_space(text[k]):
                    last_start = k
                    k += 1
                if k < n and last_start > i:
                    # \s+(?!\S): всё, кроме последнего пробела.
                    pieces.append(text[i:last_start])
                    i = last_start
                else:
                    pieces.append(text[i:k])
                    i = k
                continue

            # Некорректная/неизвестная ситуация: тянем «прочее» до пробела,
            # буквы или цифры (в C++ эта ветка — для битого UTF-8).
            k = i
            while k < n and not _is_space(text[k]) and not _is_letter(text[k]) and not _is_digit(text[k]):
                k += 1
            if k == i:
                k = i + 1
            pieces.append(text[i:k])
            i = k
        return pieces

    @staticmethod
    def _consume_run(text: str, pos: int, predicate) -> int:
        while pos < len(text) and predicate(text[pos]):
            pos += 1
        return pos

    def _bpe(self, piece: str) -> list[int]:
        symbols = [_BYTE_TO_CHAR[byte] for byte in piece.encode("utf-8")]
        while len(symbols) > 1:
            best_rank: int | None = None
            best_i = -1
            for i in range(len(symbols) - 1):
                rank = self.merge_rank.get(symbols[i] + symbols[i + 1])
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i < 0:
                break
            left = symbols[best_i]
            right = symbols[best_i + 1]
            merged: list[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i + 1 < len(symbols)
                    and symbols[i] == left
                    and symbols[i + 1] == right
                ):
                    merged.append(left + right)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged

        ids: list[int] = []
        for symbol in symbols:
            token_id = self.vocab.get(symbol)
            if token_id is not None:
                ids.append(token_id)
                continue
            # unk_token в GPT-2/Qwen нет: символ обязан быть в словаре, иначе
            # словарь неполный (например, отсутствуют byte-level токены).
            raise TokenizerError(
                f"{self.source}: символа нет в словаре (unk_token не задан): "
                f"{symbol!r}"
            )
        return ids

    def info(self) -> dict[str, Any]:
        return {
            "source": display_path(self.source),
            "model_type": "BPE",
            "vocab_size": self.vocab_size,
            "merges": self.merge_count,
            "added_tokens": len(self.added),
            "special_tokens": len(self.special_ids),
            "byte_level": self.byte_level,
            "use_regex": self.use_regex,
            "add_prefix_space": self.add_prefix_space,
            "eos_id": self.eos_id,
        }


# ---------------------------------------------------------------------------
# Токенизация пар и статистика
# ---------------------------------------------------------------------------


def render_prompt(question: str, style: str = "qwen") -> str:
    if style == "plain":
        return question
    if style == "qwen":
        return QWEN_CHAT_TEMPLATE.replace("{question}", question)
    raise DataError(f"неизвестный prompt_style {style!r}, допустимо {_VALID_PROMPT_STYLES}")


def count_pair(
    tokenizer: Any,
    question: str,
    answer: str,
    *,
    prompt_style: str = "qwen",
    max_seq_len: int = MAX_SEQ_LEN,
) -> dict[str, Any]:
    """Длины одной пары так, как их считает qat/data.py: prompt + answer + eos."""
    prompt_ids = tokenizer.encode(render_prompt(question, prompt_style))
    answer_ids = tokenizer.encode(answer)
    eos_id = getattr(tokenizer, "eos_id", None)
    if eos_id is not None and (not answer_ids or answer_ids[-1] != eos_id):
        answer_ids = answer_ids + [eos_id]
    full = len(prompt_ids) + len(answer_ids)
    return {
        "prompt": len(prompt_ids),
        "answer": len(answer_ids),
        "full": full,
        "over_context": full > max_seq_len,
        "answer_over_context": len(answer_ids) > max_seq_len,
    }


def percentile(sorted_values: Sequence[int], p: float) -> int:
    """Nearest-rank перцентиль: для n значений p50 — медиана снизу."""
    if not sorted_values:
        return 0
    rank = max(1, math.ceil(p / 100.0 * len(sorted_values)))
    return int(sorted_values[min(rank, len(sorted_values)) - 1])


def distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "p50": 0, "p95": 0, "min": 0, "max": 0, "mean": 0.0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "min": int(ordered[0]),
        "max": int(ordered[-1]),
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def histogram(values: Sequence[int], width: int, limit: int) -> dict[str, int]:
    """Компактная гистограмма длин: бакеты по width, последний — «limit и выше»."""
    buckets: dict[str, int] = {}
    for value in values:
        if value >= limit:
            key = f"{limit}+"
        else:
            low = (value // width) * width
            key = f"{low}-{low + width - 1}"
        buckets[key] = buckets.get(key, 0) + 1
    ordered_keys = sorted(
        buckets,
        key=lambda key: (int(key[:-1]) if key.endswith("+") else int(key.split("-")[0])),
    )
    return {key: buckets[key] for key in ordered_keys}


def tokenize_split(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    prompt_style: str = "qwen",
    max_seq_len: int = MAX_SEQ_LEN,
) -> dict[str, Any]:
    full_lengths: list[int] = []
    answer_lengths: list[int] = []
    prompt_tokens = 0
    answer_tokens = 0
    over_context = 0
    answer_over_context = 0
    for row in rows:
        counts = count_pair(
            tokenizer,
            row["question"],
            row["answer"],
            prompt_style=prompt_style,
            max_seq_len=max_seq_len,
        )
        full_lengths.append(counts["full"])
        answer_lengths.append(counts["answer"])
        prompt_tokens += counts["prompt"]
        answer_tokens += counts["answer"]
        over_context += int(counts["over_context"])
        answer_over_context += int(counts["answer_over_context"])

    pairs = len(rows)
    full_total = prompt_tokens + answer_tokens
    return {
        "pairs": pairs,
        "tokens": {
            "full": full_total,
            "prompt": prompt_tokens,
            "answer": answer_tokens,
        },
        "mean_full_tokens_per_pair": round(full_total / pairs, 2) if pairs else 0.0,
        "length": {
            "full": distribution(full_lengths),
            "answer": distribution(answer_lengths),
        },
        "histogram_full": {"width": 512, "limit": 4096, "buckets": histogram(full_lengths, 512, 4096)},
        "over_context": {
            "max_seq_len": max_seq_len,
            "count": over_context,
            "fraction": round(over_context / pairs, 6) if pairs else 0.0,
            "answer_only_count": answer_over_context,
        },
    }


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-numerator // denominator)


def steps_for_pairs(
    train_pairs: int,
    val_pairs: int,
    *,
    profiles: dict[str, dict[str, Any]] | None = None,
    epochs: Sequence[int] = EPOCHS,
) -> dict[str, Any]:
    """Оценка шагов: steps = пары × эпохи / (micro_batch × grad_accum)."""
    profiles = profiles or PROFILES
    result: dict[str, Any] = {
        "formula": "steps = ceil(pairs * epochs / (micro_batch * grad_accum))",
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "profiles": {},
    }
    for name, profile in profiles.items():
        micro_batch = int(profile["micro_batch"])
        grad_accum = int(profile["grad_accum"])
        effective = micro_batch * grad_accum
        epoch_rows: dict[str, Any] = {}
        for epoch in epochs:
            used = train_pairs * epoch
            epoch_rows[str(epoch)] = {
                "steps": ceil_div(used, effective),
                "micro_batches": ceil_div(used, micro_batch),
            }
        result["profiles"][name] = {
            "model": profile["model"],
            "micro_batch": micro_batch,
            "grad_accum": grad_accum,
            "effective_batch": effective,
            "steps_per_epoch": ceil_div(train_pairs, effective),
            "val_batches_per_epoch": ceil_div(val_pairs, effective),
            "epochs": epoch_rows,
        }
    return result


def tokenizer_stats(tokenizer: BPETokenizer, path: str | Path, role: str) -> dict[str, Any]:
    info = tokenizer.info()
    info["role"] = role
    info["sha256"] = file_sha256(path)
    info["is_repo_fixture"] = Path(path).resolve() == FIXTURE_TOKENIZER.resolve()
    if info["is_repo_fixture"]:
        info["note"] = (
            "в репозитории только тестовая фикстура 0012 (287 токенов): "
            "кириллица кодируется побайтно, числа — верхняя граница. "
            "Для планирования перезапустите с реальным tokenizer.json Qwen2.5 "
            "(--tokenizer PATH)"
        )
    elif role == "reference":
        info["note"] = (
            "справочный токенизатор из --reference-tokenizer, не Qwen2.5: "
            "нужен только для порядка величин, в шагах и train/val не участвует"
        )
    return info


# ---------------------------------------------------------------------------
# Запись и CLI
# ---------------------------------------------------------------------------


def display_path(path: str | Path) -> str:
    original = Path(path)
    resolved = original.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except (ValueError, OSError):
        # Вне репозитория: для HF-кэша оставляем имя модели, иначе — каталог+файл.
        # Берём исходный путь, а не resolved: tokenizer.json в снапшоте — симлинк
        # на blobs/<sha>, и после resolve() имя файла потерялось бы.
        for part in original.parts:
            if part.startswith("models--"):
                return f"{part[len('models--'):]}/{original.name}"
        return f"{original.parent.name}/{original.name}" if original.parent.name else original.name


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    lines = []
    for row in rows:
        record = {key: value for key, value in row.items() if not key.startswith("_")}
        lines.append(json.dumps(record, ensure_ascii=False))
    write_text_atomic(path, "\n".join(lines) + ("\n" if lines else ""))


def default_tokenizer_path() -> Path:
    env = os.environ.get("QAT_TOKENIZER") or os.environ.get("VTM_TOKENIZER")
    if env:
        return Path(env).expanduser()
    for candidate in (ROOT / "tokenizer.json", FIXTURE_TOKENIZER):
        if candidate.exists():
            return candidate
    raise TokenizerError(
        "tokenizer.json не найден: положите файл в корень репозитория, "
        "задайте QAT_TOKENIZER/VTM_TOKENIZER или --tokenizer PATH"
    )


def prepare(
    *,
    pairs_path: str | Path = DEFAULT_PAIRS,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    tokenizer_path: str | Path | None = None,
    reference_tokenizer_path: str | Path | None = None,
    seed: int = SEED,
    val_fraction: float = VAL_FRACTION,
    max_seq_len: int = MAX_SEQ_LEN,
    prompt_style: str = "qwen",
    epochs: Sequence[int] = EPOCHS,
    quiet: bool = False,
) -> dict[str, Any]:
    if prompt_style not in _VALID_PROMPT_STYLES:
        raise DataError(f"prompt_style {prompt_style!r}: допустимо {_VALID_PROMPT_STYLES}")
    if max_seq_len <= 0:
        raise DataError(f"max_seq_len должен быть > 0, получено {max_seq_len}")
    epochs = tuple(int(epoch) for epoch in epochs)
    if not epochs or any(epoch <= 0 for epoch in epochs):
        raise DataError(f"epochs должны быть положительными, получено {list(epochs)}")

    pairs, source = load_pairs(pairs_path)
    train, val, split = split_pairs(pairs, seed=seed, val_fraction=val_fraction)

    primary_path = Path(tokenizer_path).expanduser() if tokenizer_path else default_tokenizer_path()
    primary = BPETokenizer.load(primary_path)

    tokenization: dict[str, Any] = {
        "max_seq_len": max_seq_len,
        "prompt_style": prompt_style,
        "prompt_template": QWEN_CHAT_TEMPLATE if prompt_style == "qwen" else "{question}",
        "splits": {
            "train": tokenize_split(train, primary, prompt_style=prompt_style, max_seq_len=max_seq_len),
            "val": tokenize_split(val, primary, prompt_style=prompt_style, max_seq_len=max_seq_len),
        },
    }
    tokenizers: dict[str, Any] = {"primary": tokenizer_stats(primary, primary_path, "primary")}

    reference_stats: dict[str, Any] | None = None
    if reference_tokenizer_path:
        reference_path = Path(reference_tokenizer_path).expanduser()
        reference = BPETokenizer.load(reference_path)
        reference_stats = {
            "tokenizer": tokenizer_stats(reference, reference_path, "reference"),
            "splits": {
                "train": tokenize_split(
                    train, reference, prompt_style=prompt_style, max_seq_len=max_seq_len
                ),
                "val": tokenize_split(
                    val, reference, prompt_style=prompt_style, max_seq_len=max_seq_len
                ),
            },
        }
        tokenization["reference"] = reference_stats

    usable_train = tokenization["splits"]["train"]["pairs"] - tokenization["splits"]["train"]["over_context"]["count"]
    steps = {
        "raw": steps_for_pairs(len(train), len(val), epochs=epochs),
        "usable": {"primary": steps_for_pairs(usable_train, len(val), epochs=epochs)},
    }
    if reference_stats is not None:
        ref_usable = (
            reference_stats["splits"]["train"]["pairs"]
            - reference_stats["splits"]["train"]["over_context"]["count"]
        )
        steps["usable"]["reference"] = steps_for_pairs(ref_usable, len(val), epochs=epochs)

    warnings: list[str] = []
    if tokenizers["primary"]["is_repo_fixture"]:
        warnings.append(
            "токенизатор — тестовая фикстура 0012, а не Qwen2.5: токены и доля пар "
            "длиннее 2048 завышены (кириллица кодируется побайтно). Перезапустите "
            "с --tokenizer /path/Qwen2.5-*/tokenizer.json"
        )
    dropped_fraction = (
        1.0 - usable_train / len(train) if train else 0.0
    )
    if dropped_fraction > 0:
        warnings.append(
            f"{len(train) - usable_train} из {len(train)} пар train длиннее "
            f"{max_seq_len} токенов ({dropped_fraction:.1%}) и будут отброшены "
            "qat/data.py (PairDataset.dropped_long)"
        )
    max_epoch = max(epochs)
    max_steps_at_last_epoch = steps["raw"]["profiles"]["qwen1.5b"]["epochs"][str(max_epoch)]["steps"]
    if max_steps_at_last_epoch < CONFIG_MAX_STEPS:
        warnings.append(
            f"train.max_steps={CONFIG_MAX_STEPS} из конфигов: {max_epoch} эпох(и) дают "
            f"только {max_steps_at_last_epoch} шагов; для {CONFIG_MAX_STEPS} шагов нужно "
            f"~{ceil_div(CONFIG_MAX_STEPS * steps['raw']['profiles']['qwen1.5b']['effective_batch'], len(train))} "
            "эпох — либо уменьшать max_steps"
        )
    if reference_stats is not None:
        warnings.append(
            "reference-токенизатор посчитан дополнительно и помечен: "
            "в stats.json это не Qwen2.5, а сторонний BPE для порядка величин"
        )

    stats: dict[str, Any] = {
        "schema": 1,
        "generator": "tools/prepare_qat_data.py",
        "source": source,
        "split": split,
        "tokenizer": tokenizers,
        "tokenization": tokenization,
        "steps": steps,
        "warnings": warnings,
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_text_atomic(out / "stats.json", json.dumps(stats, ensure_ascii=False, indent=2) + "\n")

    if not quiet:
        print_summary(stats)
    return stats


def print_summary(stats: dict[str, Any]) -> None:
    source = stats["source"]
    split = stats["split"]
    tok = stats["tokenizer"]["primary"]
    train = stats["tokenization"]["splits"]["train"]
    val = stats["tokenization"]["splits"]["val"]
    print(f"[qat-data] источник: {source['path']} ({source['rows']} строк)")
    print(
        f"[qat-data] очищено: битых {source['skipped_invalid']}, "
        f"пустых {source['skipped_empty']}, дублей {source['duplicates_removed']} "
        f"→ {split['pairs']} пар"
    )
    print(
        f"[qat-data] split seed={split['seed']}: train {split['train_pairs']} / "
        f"val {split['val_pairs']}, пересечение {split['overlap_normalized']}"
    )
    print(
        f"[qat-data] токенизатор: {tok['source']} (vocab {tok['vocab_size']}), "
        f"контекст {stats['tokenization']['max_seq_len']}"
    )
    for name, part in (("train", train), ("val", val)):
        dist = part["length"]["full"]
        print(
            f"[qat-data] {name}: токенов {part['tokens']['full']} "
            f"(в ответах {part['tokens']['answer']}), "
            f"p50 {dist['p50']} / p95 {dist['p95']} / max {dist['max']}, "
            f">{stats['tokenization']['max_seq_len']} токенов: "
            f"{part['over_context']['count']} пар"
        )
    for name, profile in stats["steps"]["raw"]["profiles"].items():
        rows = ", ".join(
            f"{epoch}э={data['steps']}" for epoch, data in profile["epochs"].items()
        )
        print(
            f"[qat-data] шаги {name} (micro {profile['micro_batch']} × "
            f"accum {profile['grad_accum']} = {profile['effective_batch']}): {rows}"
        )
    if "reference" in stats["tokenization"]:
        ref = stats["tokenization"]["reference"]["splits"]["train"]
        dist = ref["length"]["full"]
        print(
            f"[qat-data] reference (не Qwen): токенов {ref['tokens']['full']}, "
            f"p50 {dist['p50']} / p95 {dist['p95']} / max {dist['max']}, "
            f">{stats['tokenization']['max_seq_len']} токенов: "
            f"{ref['over_context']['count']} пар"
        )
    for warning in stats["warnings"]:
        print(f"[qat-data] ВНИМАНИЕ: {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 tools/prepare_qat_data.py",
        description=(
            "Очистка, дедупликация, train/val 90/10 и предтокенизация пар "
            "distill/out/code_pairs.jsonl для QAT."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  python3 tools/prepare_qat_data.py\n"
            "  python3 tools/prepare_qat_data.py --tokenizer "
            "/path/Qwen2.5-1.5B-Instruct/tokenizer.json\n"
            "  python3 tools/prepare_qat_data.py --out-dir /tmp/qat-data --max-seq-len 1024\n"
        ),
    )
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS),
                        help=f"входной JSONL (по умолчанию {display_path(DEFAULT_PAIRS)})")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="каталог результатов (по умолчанию data/)")
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer.json (по умолчанию $QAT_TOKENIZER, ./tokenizer.json, "
                             "фикстура tests/vtm/fixtures/tokenizer.json)")
    parser.add_argument("--reference-tokenizer", default=None,
                        help="дополнительный tokenizer.json для справочного расчёта; "
                             "помечается в stats.json как reference")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"seed разбиения (по умолчанию {SEED})")
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION,
                        help=f"доля val (по умолчанию {VAL_FRACTION})")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                        help=f"контекст обучения (по умолчанию {MAX_SEQ_LEN})")
    parser.add_argument("--prompt-style", choices=_VALID_PROMPT_STYLES, default="qwen",
                        help="qwen — chat-шаблон Qwen2.5 с system (как qat/data.py), "
                             "plain — только вопрос")
    parser.add_argument("--epochs", type=int, nargs="+", default=list(EPOCHS),
                        help=f"эпохи для оценки шагов (по умолчанию {list(EPOCHS)})")
    parser.add_argument("--quiet", action="store_true", help="не печатать сводку")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prepare(
            pairs_path=args.pairs,
            out_dir=args.out_dir,
            tokenizer_path=args.tokenizer,
            reference_tokenizer_path=args.reference_tokenizer,
            seed=args.seed,
            val_fraction=args.val_fraction,
            max_seq_len=args.max_seq_len,
            prompt_style=args.prompt_style,
            epochs=args.epochs,
            quiet=args.quiet,
        )
    except (DataError, TokenizerError, OSError) as exc:
        print(f"[qat-data] ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
