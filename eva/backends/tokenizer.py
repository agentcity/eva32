"""BPE-токенизатор из ``tokenizer.json`` — чистый Python, без внешних зависимостей.

Зачем: C-ABI ВТМ (``src/vtm/api.h``) работает с ``int32_t`` id и не знает про
текст. Хосту нужен слой «строка ↔ id», и он не должен требовать сборки
C++-моста к ``tokenizer.hpp`` или запуска процесса ``src/vtm/vtm``. Здесь
реализован тот же алгоритм для формата HuggingFace:

* ``model.type == "BPE"``, ``model.vocab``, ``model.merges`` (пары через пробел);
* byte-level отображение GPT-2 (256 байт ↔ символы) и GPT-2 regex-разбиение
  (порт ``Tokenizer::pre_tokenize`` из ``src/vtm/tokenizer.cpp``);
* ``added_tokens`` (спецтокены ищутся в исходном тексте, самый длинный матч);
* нормализатор ``null`` или ``NFC``;
* ``decode`` с накоплением неполных UTF-8-байт (``errors="replace"``).

Публичный API::

    tok = BPETokenizer.from_file("tokenizer.json")
    ids = tok.encode("привет")
    text = tok.decode(ids)

    dec = tok.stream_decoder()       # для стриминга по одному токену
    for token_id in ids:
        chunk = dec.push(token_id)   # может быть "", если байт неполный
    tail = dec.finish()

Только стандартная библиотека: никакой сети, ``pip`` и ``regex``.
"""

from __future__ import annotations

import codecs
import json
import unicodedata
from typing import Any, Iterable, Sequence

__all__ = ["BPETokenizer", "StreamDecoder", "TokenizerError"]


class TokenizerError(ValueError):
    """``tokenizer.json`` не разобран или токена нет в словаре."""


# ---------------------------------------------------------------------------
# GPT-2: байт -> unicode-символ и обратно
# ---------------------------------------------------------------------------

def _bytes_to_unicode() -> dict[int, str]:
    """Таблица GPT-2: печатные байты остаются собой, остальные -> U+0100+."""
    used = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    codepoints = list(used)
    extra = 0
    for byte in range(256):
        if byte not in used:
            used.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {byte: chr(cp) for byte, cp in zip(used, codepoints)}


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {ord(ch): byte for byte, ch in _BYTE_ENCODER.items()}


# ---------------------------------------------------------------------------
# Классы символов: точный порт is_space_cp/is_letter_cp/is_digit_cp
# ---------------------------------------------------------------------------

def _is_space_cp(cp: int) -> bool:
    return (
        cp in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2028,
               0x2029, 0x202F, 0x205F, 0x3000)
        or 0x2000 <= cp <= 0x200A
    )


def _is_letter_cp(cp: int) -> bool:
    if 0x41 <= cp <= 0x5A or 0x61 <= cp <= 0x7A:
        return True
    if cp in (0xAA, 0xB5, 0xBA):
        return True
    for lo, hi in (
        (0xC0, 0x2AF), (0x370, 0x3FF), (0x400, 0x52F), (0x531, 0x58F),
        (0x5D0, 0x5EA), (0x620, 0x64A), (0x900, 0x97F), (0xE01, 0xE5B),
        (0x10A0, 0x10FF), (0x1E00, 0x1FFF), (0x2C60, 0x2C7F), (0x3040, 0x30FF),
        (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xA640, 0xA69F), (0xAC00, 0xD7AF),
        (0xF900, 0xFAFF), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A),
        (0x20000, 0x2FA1F),
    ):
        if lo <= cp <= hi:
            return True
    return False


def _is_digit_cp(cp: int) -> bool:
    return (
        0x30 <= cp <= 0x39
        or 0x660 <= cp <= 0x669
        or 0x6F0 <= cp <= 0x6F9
        or 0x966 <= cp <= 0x96F
        or 0xFF10 <= cp <= 0xFF19
    )


def _is_other_cp(cp: int) -> bool:
    return not (_is_space_cp(cp) or _is_letter_cp(cp) or _is_digit_cp(cp))


# ---------------------------------------------------------------------------
# Одноразовые константы GPT-2 regex
# ---------------------------------------------------------------------------

_SUFFIXES = ("s", "t", "re", "ve", "m", "ll", "d")


# ---------------------------------------------------------------------------
# Токенизатор
# ---------------------------------------------------------------------------

class BPETokenizer:
    """BPE из ``tokenizer.json`` формата HuggingFace (``model.type == "BPE"``).

    Объект неизменяемый после сборки, ``encode``/``decode`` потокобезопасны
    для чтения (внутренние словари только читаются).
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: Iterable[Any],
        *,
        added_tokens: Iterable[dict[str, Any]] = (),
        byte_level: bool = True,
        use_regex: bool = True,
        add_prefix_space: bool = False,
        normalizer: str | None = None,
        unk_id: int = -1,
        eos_id: int = -1,
        source: str = "",
    ) -> None:
        self.source = source
        self.byte_level = bool(byte_level)
        self.use_regex = bool(use_regex)
        self.add_prefix_space = bool(add_prefix_space)
        self.normalizer = normalizer
        self.unk_id = int(unk_id)
        self.eos_id = int(eos_id)

        self._token_to_id: dict[str, int] = {str(k): int(v) for k, v in vocab.items()}
        self._id_to_token: dict[int, str] = {i: t for t, i in self._token_to_id.items()}
        self._merge_rank: dict[tuple[str, str], int] = {}
        for rank, entry in enumerate(merges):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                left, right = str(entry[0]), str(entry[1])
            else:
                text = str(entry)
                parts = text.split(" ")
                if len(parts) != 2:
                    raise TokenizerError(
                        f"{source}: model.merges[{rank}] = {text!r}: ожидались две "
                        "части через пробел"
                    )
                left, right = parts
            merged = left + right
            # В C++ мерж без результата в словаре пропускается (с предупреждением).
            if merged not in self._token_to_id:
                continue
            self._merge_rank.setdefault((left, right), rank)

        self._added: list[tuple[str, int, bool]] = []
        self._added_ids: set[int] = set()
        self._eos_content: str | None = None
        for item in added_tokens:
            content = str(item.get("content", ""))
            try:
                token_id = int(item.get("id", -1))
            except (TypeError, ValueError):
                continue
            if not content or token_id < 0:
                continue
            special = bool(item.get("special", False))
            self._added.append((content, token_id, special))
            self._added_ids.add(token_id)
            # Спецтокенов нет в model.vocab, но decode обязан их знать.
            if self._id_to_token.get(token_id, content) != content:
                raise TokenizerError(
                    f"{source}: added_token {content!r} id={token_id} конфликтует "
                    f"с токеном {self._id_to_token[token_id]!r}"
                )
            self._id_to_token[token_id] = content
            if eos_id >= 0 and token_id == eos_id:
                self._eos_content = content
        # Длинные спецтокены — первыми: match_added ищет самый длинный матч.
        self._added.sort(key=lambda item: len(item[0]), reverse=True)
        self._added_max_len = max((len(c) for c, _i, _s in self._added), default=0)

    # -- сборка ------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str) -> "BPETokenizer":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except OSError as exc:
            raise TokenizerError(f"tokenizer.json: {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise TokenizerError(f"tokenizer.json: {path}: невалидный JSON: {exc}") from exc
        return cls.from_json(data, source=path)

    @classmethod
    def from_json(cls, data: dict[str, Any], *, source: str = "") -> "BPETokenizer":
        if not isinstance(data, dict):
            raise TokenizerError(f"{source}: tokenizer.json — ожидался объект")
        model = data.get("model")
        if not isinstance(model, dict):
            raise TokenizerError(f"{source}: нет объекта model")
        model_type = str(model.get("type") or "")
        if model_type.upper() != "BPE":
            raise TokenizerError(
                f"{source}: model.type={model_type!r}, поддерживается только BPE"
            )
        vocab = model.get("vocab")
        if not isinstance(vocab, dict) or not vocab:
            raise TokenizerError(f"{source}: model.vocab пуст или не объект")
        merges = model.get("merges")
        if merges is None:
            merges = []
        if not isinstance(merges, list):
            raise TokenizerError(f"{source}: model.merges должен быть массивом")

        pre = data.get("pre_tokenizer")
        decoder = data.get("decoder")
        byte_level = cls._has_byte_level(pre) or cls._has_byte_level(decoder)
        use_regex = cls._byte_level_flag(pre, "use_regex", True)
        add_prefix_space = cls._byte_level_flag(pre, "add_prefix_space", False)

        normalizer = cls._describe_normalizer(data.get("normalizer"), source)

        tokenizer = cls(
            vocab,
            merges,
            added_tokens=data.get("added_tokens") or [],
            byte_level=byte_level,
            use_regex=use_regex,
            add_prefix_space=add_prefix_space,
            normalizer=normalizer,
            source=source,
        )
        # model.unk_token/eos_token — строка из словаря или числовой id
        # (как resolve_model_token() в C++).
        tokenizer.unk_id = tokenizer._resolve_model_token(model.get("unk_token"))
        tokenizer.eos_id = tokenizer._resolve_model_token(model.get("eos_token"))
        if tokenizer.eos_id < 0:
            # У Qwen (и др.) eos в tokenizer.json не задан: он лежит рядом в
            # tokenizer_config.json как строка "<|im_end|>". Без этого генерация
            # не останавливается и модель зацикливается до max_tokens.
            tokenizer.eos_id = tokenizer._eos_from_config(source)
        if tokenizer.eos_id >= 0:
            tokenizer._eos_content = tokenizer._id_to_token.get(tokenizer.eos_id)
        if bool(model.get("ignore_merges", False)):
            tokenizer._merge_rank.clear()
        return tokenizer


    def _eos_from_config(self, source: Any) -> int:
        """EOS из соседнего ``tokenizer_config.json`` (Qwen: ``<|im_end|>``).

        В ``tokenizer.json`` у Qwen ``model.eos_token`` пуст, поэтому без этого
        чтения ``eos_id`` остаётся -1 и генерация идёт до лимита токенов,
        повторяя один и тот же текст.
        """
        import json as _json
        import os as _os
        from pathlib import Path as _Path

        try:
            path = _Path(str(source)).resolve().parent / "tokenizer_config.json"
            if not path.is_file():
                return -1
            config = _json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return -1
        raw = config.get("eos_token_id")
        if raw is None:
            raw = config.get("eos_token")
        if isinstance(raw, list) and raw:
            raw = raw[0]
        return self._resolve_model_token(raw)

    def _resolve_model_token(self, raw: Any) -> int:
        """Строка словаря/added_token или числовой id -> id (или -1)."""
        if raw is None or isinstance(raw, bool):
            return -1
        if isinstance(raw, (int, float)):
            token_id = int(raw)
            return token_id if token_id in self._id_to_token else -1
        if isinstance(raw, str):
            token_id = self._token_to_id.get(raw)
            if token_id is not None:
                return token_id
            for content, added_id, _special in self._added:
                if content == raw:
                    return added_id
        return -1

    @staticmethod
    def _has_byte_level(node: Any) -> bool:
        if isinstance(node, dict):
            if node.get("type") == "ByteLevel":
                return True
            return any(BPETokenizer._has_byte_level(v) for v in node.values())
        if isinstance(node, list):
            return any(BPETokenizer._has_byte_level(v) for v in node)
        return False

    @staticmethod
    def _byte_level_flag(node: Any, key: str, default: bool) -> bool:
        if isinstance(node, dict):
            if node.get("type") == "ByteLevel" and key in node:
                return bool(node[key])
            for value in node.values():
                found = BPETokenizer._byte_level_flag(value, key, default)
                if found != default:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = BPETokenizer._byte_level_flag(value, key, default)
                if found != default:
                    return found
        return default

    @staticmethod
    def _describe_normalizer(node: Any, source: str) -> str | None:
        if node is None:
            return None
        if isinstance(node, dict):
            kind = str(node.get("type") or "")
            if kind.lower() == "nfc":
                return "NFC"
            raise TokenizerError(
                f"{source}: нормализатор {kind!r} не поддержан (null или NFC)"
            )
        raise TokenizerError(f"{source}: normalizer должен быть объектом или null")

    # -- свойства ----------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    @property
    def eos_token(self) -> str | None:
        return self._eos_content

    @property
    def added_tokens(self) -> tuple[tuple[str, int, bool], ...]:
        return tuple(self._added)

    def token(self, token_id: int) -> str:
        try:
            return self._id_to_token[int(token_id)]
        except KeyError as exc:
            raise TokenizerError(f"decode: неизвестный token id = {token_id}") from exc

    def id_of(self, content: str) -> int | None:
        return self._token_to_id.get(content)

    # -- encode ------------------------------------------------------------

    def encode(self, text: str, *, allow_special_tokens: bool = True) -> list[int]:
        """Строка -> id. Спецтокены из ``added_tokens`` распознаются в тексте."""
        if not text:
            return []
        source = str(text)
        ids: list[int] = []
        start = 0
        pos = 0
        while pos < len(source):
            match = self._match_added(source, pos, allow_special_tokens)
            if match is not None:
                content, token_id = match
                if pos > start:
                    ids.extend(self._encode_segment(source[start:pos]))
                ids.append(token_id)
                pos += len(content)
                start = pos
            else:
                pos += 1
        if start < len(source):
            ids.extend(self._encode_segment(source[start:]))
        return ids

    def _encode_segment(self, segment: str) -> list[int]:
        if self.add_prefix_space and segment and segment[0] != " ":
            segment = " " + segment
        segment = self._normalize(segment)
        return self._encode_ordinary(segment)

    def _normalize(self, text: str) -> str:
        if self.normalizer == "NFC":
            return unicodedata.normalize("NFC", text)
        return text

    def _match_added(
        self, text: str, pos: int, allow_special: bool
    ) -> tuple[str, int] | None:
        if not self._added:
            return None
        if self._added_max_len == 0 or pos + 1 > len(text):
            return None
        for content, token_id, special in self._added:
            if special and not allow_special:
                continue
            if len(content) > len(text) - pos:
                continue
            if text.startswith(content, pos):
                return content, token_id
        return None

    def _encode_ordinary(self, text: str) -> list[int]:
        if not text:
            return []
        if not self.use_regex:
            return self._bpe(text)
        ids: list[int] = []
        for piece in self._pre_tokenize(text):
            ids.extend(self._bpe(piece))
        return ids

    def _pre_tokenize(self, text: str) -> list[str]:
        """Порт ``Tokenizer::pre_tokenize``: GPT-2 regex, приближение \\p{L}."""
        pieces: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            cp = ord(ch)

            if ch == "'":
                matched = False
                for suffix in _SUFFIXES:
                    if text.startswith(suffix, i + 1):
                        pieces.append(text[i:i + 1 + len(suffix)])
                        i += 1 + len(suffix)
                        matched = True
                        break
                if matched:
                    continue

            lead_space = cp == 0x20
            start_after_space = i + 1 if lead_space else i

            if start_after_space < n:
                nc = ord(text[start_after_space])
                next_letter = _is_letter_cp(nc)
                next_digit = _is_digit_cp(nc)
                next_other = not (
                    _is_space_cp(nc) or _is_letter_cp(nc) or _is_digit_cp(nc)
                )
                if next_letter or next_digit or next_other:
                    if next_letter:
                        end = self._consume(text, start_after_space, _is_letter_cp)
                    elif next_digit:
                        end = self._consume(text, start_after_space, _is_digit_cp)
                    else:
                        end = self._consume(text, start_after_space, _is_other_cp)
                    pieces.append(text[i:end])
                    i = end
                    continue

            if not lead_space:
                if _is_letter_cp(cp):
                    end = self._consume(text, i, _is_letter_cp)
                    pieces.append(text[i:end])
                    i = end
                    continue
                if _is_digit_cp(cp):
                    end = self._consume(text, i, _is_digit_cp)
                    pieces.append(text[i:end])
                    i = end
                    continue
                if not _is_space_cp(cp):
                    end = self._consume(text, i, _is_other_cp)
                    pieces.append(text[i:end])
                    i = end
                    continue

            if _is_space_cp(cp):
                k = i
                last_start = i
                while k < n and _is_space_cp(ord(text[k])):
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

            # Недостижимо для str (все символы — валидные code point), но
            # оставляем безопасный шаг, чтобы цикл не завис.
            pieces.append(ch)
            i += 1
        return pieces

    @staticmethod
    def _consume(text: str, pos: int, pred) -> int:
        n = len(text)
        while pos < n and pred(ord(text[pos])):
            pos += 1
        return pos

    def _bpe(self, piece: str) -> list[int]:
        raw = piece.encode("utf-8", "surrogatepass")
        if not self.byte_level:
            symbols = [raw.decode("utf-8", "replace")]
        else:
            symbols = [_BYTE_ENCODER[byte] for byte in raw]
        while len(symbols) > 1:
            best_rank: int | None = None
            best_pair: tuple[str, str] | None = None
            for left, right in zip(symbols, symbols[1:]):
                rank = self._merge_rank.get((left, right))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, (left, right)
            if best_pair is None:
                break
            merged = best_pair[0] + best_pair[1]
            out: list[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i + 1 < len(symbols)
                    and symbols[i] == best_pair[0]
                    and symbols[i + 1] == best_pair[1]
                ):
                    out.append(merged)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            symbols = out
        ids: list[int] = []
        for symbol in symbols:
            token_id = self._token_to_id.get(symbol)
            if token_id is not None:
                ids.append(token_id)
                continue
            if self.unk_id >= 0:
                ids.append(self.unk_id)
                continue
            raise TokenizerError(
                f"BPE: символа нет в словаре (нет unk_token): {symbol!r}; "
                "словарь неполный?"
            )
        return ids

    # -- decode ------------------------------------------------------------

    def token_bytes(self, token_id: int) -> bytes | None:
        """Байты обычного токена; ``None`` — токен-спецтокен (не byte-level)."""
        token_id = int(token_id)
        if token_id in self._added_ids:
            return None
        content = self._id_to_token.get(token_id)
        if content is None:
            raise TokenizerError(f"decode: неизвестный token id = {token_id}")
        if not self.byte_level:
            return content.encode("utf-8", "surrogatepass")
        out = bytearray()
        for ch in content:
            byte = _BYTE_DECODER.get(ord(ch))
            if byte is None:
                out.extend(ch.encode("utf-8", "surrogatepass"))
            else:
                out.append(byte)
        return bytes(out)

    def decode(self, ids: Sequence[int]) -> str:
        """id -> строка. Неполные UTF-8-последовательности -> U+FFFD."""
        out: list[str] = []
        pending = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if token_id in self._added_ids:
                if pending:
                    out.append(pending.decode("utf-8", "replace"))
                    pending.clear()
                out.append(self._id_to_token[token_id])
                continue
            data = self.token_bytes(token_id)
            if data is None:  # pragma: no cover — added уже обработан выше
                continue
            pending.extend(data)
        if pending:
            out.append(pending.decode("utf-8", "replace"))
        return "".join(out)

    def stream_decoder(self) -> "StreamDecoder":
        """Инкрементальный декодер для стриминга по одному токену."""
        return StreamDecoder(self)

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab={self.vocab_size}, merges={len(self._merge_rank)}, "
            f"added={len(self._added)}, source={self.source!r})"
        )


class StreamDecoder:
    """Инкрементальный decode: ``push(id) -> str``, затем ``finish()``.

    Склейка всех ``push`` + ``finish`` равна ``tokenizer.decode(ids)``.
    Пустая строка — токен дал только часть UTF-8-символа (байты накопятся).
    """

    def __init__(self, tokenizer: BPETokenizer) -> None:
        self._tokenizer = tokenizer
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._finished = False

    def push(self, token_id: int) -> str:
        if self._finished:
            raise TokenizerError("stream_decoder: push после finish()")
        data = self._tokenizer.token_bytes(token_id)
        if data is None:
            # Спецтокен выводится как есть. Накопленный неполный UTF-8-хвост
            # сбрасывается с заменой — как utf8_sanitize() в C++.
            flushed = self._decoder.decode(b"", final=True)
            self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
            return flushed + self._tokenizer.token(int(token_id))
        return self._decoder.decode(data, final=False)

    def finish(self) -> str:
        if self._finished:
            return ""
        self._finished = True
        return self._decoder.decode(b"", final=True)

    def __enter__(self) -> "StreamDecoder":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.finish()
