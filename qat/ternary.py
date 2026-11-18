"""Тернарное квантование и упаковка тритов.

Здесь две части, разделённые осознанно:

* **чистая математика** (списки float/int) — работает без torch и numpy.
  Её используют тесты, проверка формата и экспорт метаданных;
* **torch-путь** — функции с суффиксом ``_torch``. ``import torch`` происходит
  только внутри них, поэтому ``python -m qat.train --help`` и юнит-тесты
  не требуют GPU-стека.

Правила квантования (зафиксированы в задаче, не пересматриваются):

* блок — 128 элементов вдоль входной размерности (последней оси веса);
* ``scale = absmax(блок)``; у полностью нулевого блока ``scale = 0``,
  все триты нулевые (деления на ноль не происходит);
* ``trit = clamp(round_half_away(w / scale), -1, +1)``;
* ``w_tern = trit * scale``;
* округление — «половина от нуля» (0.5 -> 1, -0.5 -> -1), чтобы результат
  не зависел от чётности и совпадал у чистого Python и torch.

STE (straight-through estimator) реализован в :func:`ternary_ste_torch`:
вперёд идёт квантованный вес, назад — тождественный градиент.

Упаковка — 20 тритов в 32-битном слове, как в ``docs/trit-format.md`` и
``tools/trit_codec.py``; логика переписана здесь, чтобы не импортировать
``tools`` (и чтобы не тянуть зависимости). Триты кодируются цифрами
``-1 -> 0``, ``0 -> 1``, ``+1 -> 2``, слово = Σ digit_i · 3^i.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - нужно только для аннотаций
    import torch

#: Блок квантования: absmax ищется по 128 элементам вдоль входной размерности.
BLOCK_SIZE = 128
#: Тритов в одном 32-битном слове (3^20 < 2^32).
TRITS_PER_WORD = 20
#: Слов в блоке формата: 8 слов = 160 тритов = 32 байта.
WORDS_PER_STORAGE_BLOCK = 8
TRITS_PER_STORAGE_BLOCK = TRITS_PER_WORD * WORDS_PER_STORAGE_BLOCK
TRIT_VALUES = (-1, 0, 1)
MAX_WORD = 3 ** TRITS_PER_WORD - 1
MAGIC = b"TRT1"
VERSION = 1
FLAG_SCALE_F32 = 0x01

_HEADER = struct.Struct("<4sBBQI")  # magic, version, flags, n_trits, n_scale_blocks
_TO_DIGIT = {-1: 0, 0: 1, 1: 2}
_FROM_DIGIT = {0: -1, 1: 0, 2: 1}
_SCALE_CODE = {"float16": "e", "float32": "f"}
_POW3 = [3 ** k for k in range(TRITS_PER_WORD)]


class TritError(ValueError):
    """Ошибка данных тритов, тритовой упаковки или формата."""


class MissingTorchError(RuntimeError):
    """Torch не установлен, а вызвана функция GPU-пути."""


# ---------------------------------------------------------------------------
# Квантование: чистая математика
# ---------------------------------------------------------------------------


def round_half_away(value: float) -> int:
    """Округлить к ближайшему целому, половину — от нуля.

    ``round()`` в Python и ``torch.round()`` округляют половину к чётному
    (0.5 -> 0, 1.5 -> 2). Для квантования это неявно меняет порог, поэтому
    здесь и в torch-пути используется одно и то же правило.
    """
    if not math.isfinite(value):
        raise TritError(f"нельзя округлить неконечное значение: {value!r}")
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _check_block_size(block_size: int) -> None:
    if not isinstance(block_size, int) or block_size <= 0:
        raise TritError(f"block_size должен быть положительным целым, получено {block_size!r}")


def n_blocks(n_values: int, block_size: int = BLOCK_SIZE) -> int:
    """Сколько блоков масштаба нужно на ``n_values`` элементов."""
    _check_block_size(block_size)
    if n_values < 0:
        raise TritError("число элементов не может быть отрицательным")
    return (n_values + block_size - 1) // block_size


def _as_floats(values: Iterable[float]) -> list[float]:
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise TritError(f"ожидались числа, получено {values!r}") from exc
    for v in out:
        if not math.isfinite(v):
            raise TritError(f"вес не конечен: {v!r}; проверьте данные и инициализацию")
    return out


def _check_scales(scales: Sequence[float], expected: int) -> list[float]:
    out = [float(s) for s in scales]
    if len(out) != expected:
        raise TritError(f"масштабов {len(out)}, а блоков {expected}")
    for s in out:
        if not math.isfinite(s) or s < 0:
            raise TritError(f"масштаб должен быть конечным и >= 0, получено {s!r}")
    return out


def _check_trit(trit: int) -> int:
    if trit not in TRIT_VALUES:
        raise TritError(f"трит должен быть -1, 0 или +1, получено {trit!r}")
    return trit


def block_scales(values: Iterable[float], block_size: int = BLOCK_SIZE) -> list[float]:
    """absmax по каждому блоку вдоль последней оси (для плоского списка)."""
    _check_block_size(block_size)
    vals = _as_floats(values)
    scales: list[float] = []
    for start in range(0, len(vals), block_size):
        block = vals[start:start + block_size]
        scales.append(max(abs(v) for v in block) if block else 0.0)
    return scales


def quantize_value(value: float, scale: float) -> int:
    """Один вес в один трит при заданном масштабе."""
    if not math.isfinite(value) or not math.isfinite(scale):
        raise TritError("вес и масштаб должны быть конечными")
    if scale < 0:
        raise TritError(f"масштаб не может быть отрицательным: {scale!r}")
    if scale == 0.0:
        return 0
    return max(-1, min(1, round_half_away(value / scale)))


def quantize_with_scale(
    values: Iterable[float],
    scales: Sequence[float],
    block_size: int = BLOCK_SIZE,
) -> list[int]:
    """Квантовать значения по уже известным масштабам блоков.

    Нужна для проверки границ (например, `1.5 * scale -> +1`): внутри блока
    с absmax-масштабом отношение не превышает 1, поэтому граничные значения
    проверяются через явно заданный масштаб.
    """
    _check_block_size(block_size)
    vals = _as_floats(values)
    scs = _check_scales(scales, n_blocks(len(vals), block_size))
    return [quantize_value(v, scs[i // block_size]) for i, v in enumerate(vals)]


def quantize(
    values: Iterable[float],
    block_size: int = BLOCK_SIZE,
) -> tuple[list[int], list[float]]:
    """Квантовать плоский список: вернуть (триты, масштабы блоков).

    ``scale = absmax`` блока; если блок нулевой — масштаб 0 и триты 0.
    """
    vals = _as_floats(values)
    scales = block_scales(vals, block_size)
    return quantize_with_scale(vals, scales, block_size), scales


def dequantize(
    trits: Iterable[int],
    scales: Sequence[float],
    block_size: int = BLOCK_SIZE,
) -> list[float]:
    """Восстановить веса: ``w = trit * scale`` (масштаб — на блок)."""
    _check_block_size(block_size)
    ts = [_check_trit(t) for t in trits]
    scs = _check_scales(scales, n_blocks(len(ts), block_size))
    return [t * scs[i // block_size] for i, t in enumerate(ts)]


# ---------------------------------------------------------------------------
# Упаковка 20 тритов в 32-битное слово
# ---------------------------------------------------------------------------


def words20(trits: Iterable[int]) -> list[int]:
    """Упаковать триты в слова; хвост дополняется нулевыми тритами."""
    ts = [_check_trit(t) for t in trits]
    pad = (-len(ts)) % TRITS_PER_WORD
    ts.extend([0] * pad)
    out: list[int] = []
    for start in range(0, len(ts), TRITS_PER_WORD):
        word = 0
        for k, trit in enumerate(ts[start:start + TRITS_PER_WORD]):
            word += _TO_DIGIT[trit] * _POW3[k]
        out.append(word)
    return out


def pack20(trits: Iterable[int]) -> bytes:
    """20 тритов -> 4 байта little-endian; неполное слово добивается нулями."""
    return b"".join(word.to_bytes(4, "little") for word in words20(trits))


def unpack20_words(words: Iterable[int], count: int) -> list[int]:
    """Распаковать слова в триты, оставив первые ``count`` штук."""
    if count < 0:
        raise TritError("count не может быть отрицательным")
    out: list[int] = []
    for word in words:
        value = int(word)
        if value < 0 or value > MAX_WORD:
            raise TritError(f"слово испорчено: {value} вне диапазона 3^20")
        for _ in range(TRITS_PER_WORD):
            out.append(_FROM_DIGIT[value % 3])
            value //= 3
    if count > len(out):
        raise TritError(f"запрошено {count} тритов, а в данных {len(out)}")
    return out[:count]


def unpack20(data: bytes, count: int) -> list[int]:
    """Распаковать байты, проверив длину, диапазон слов и ``count``."""
    if len(data) % 4:
        raise TritError("длина данных должна быть кратна 4 байтам")
    words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
    return unpack20_words(words, count)


# ---------------------------------------------------------------------------
# Формат сегмента TRT1 (docs/trit-format.md)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TritSegment:
    """Разобранный заголовок сегмента — для диагностики экспорта."""

    n_trits: int
    n_scale_blocks: int
    n_words: int
    n_storage_blocks: int
    scale_dtype: str


def _storage_words(n_trits: int) -> int:
    blocks = (n_trits + TRITS_PER_STORAGE_BLOCK - 1) // TRITS_PER_STORAGE_BLOCK
    return blocks * WORDS_PER_STORAGE_BLOCK


def encode_segment(
    trits: Iterable[int],
    scales: Sequence[float],
    scale_dtype: str = "float16",
    scale_block: int = BLOCK_SIZE,
) -> bytes:
    """Собрать сегмент формата TRT1: заголовок + масштабы + слова.

    ``n_blocks`` в заголовке — число групп масштаба (по умолчанию 128 весов).
    Слова добиваются до 8 штук на 160-тритовый блок: так данные выровнены по
    32 байта, как требует docs/trit-format.md. Размер группы масштаба —
    открытый вопрос §8 документа; он пишется в manifest, чтобы читатель
    не угадывал и передал тот же ``scale_block``.
    """
    if scale_dtype not in _SCALE_CODE:
        raise TritError(f"scale_dtype должен быть float16 или float32, получено {scale_dtype!r}")
    ts = [_check_trit(t) for t in trits]
    n_trits = len(ts)
    expected_scales = n_blocks(n_trits, scale_block)
    scs = _check_scales(scales, expected_scales)

    words = words20(ts)
    words.extend([0] * (_storage_words(n_trits) - len(words)))
    flags = FLAG_SCALE_F32 if scale_dtype == "float32" else 0
    header = _HEADER.pack(MAGIC, VERSION, flags, n_trits, len(scs))
    try:
        scales_bytes = struct.pack(f"<{len(scs)}{_SCALE_CODE[scale_dtype]}", *scs)
    except (OverflowError, struct.error) as exc:
        raise TritError(f"масштаб не помещается в {scale_dtype}: {exc}") from exc
    words_bytes = b"".join(word.to_bytes(4, "little") for word in words)
    return header + scales_bytes + words_bytes


def decode_segment(data: bytes, scale_block: int = BLOCK_SIZE) -> tuple[list[int], list[float]]:
    """Разобрать сегмент TRT1, собранный :func:`encode_segment`.

    Проверяются magic/version/флаги/длины/диапазон слов — как в §5
    docs/trit-format.md. Размер группы масштаба передаётся явно: заголовок
    TRT1 его не хранит, источник правды — manifest экспорта.
    """
    if len(data) < _HEADER.size:
        raise TritError("сегмент короче заголовка TRT1")
    magic, version, flags, n_trits, n_scale = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise TritError(f"чужой magic: {magic!r}")
    if version != VERSION:
        raise TritError(f"неизвестная версия формата: {version}")
    if flags & ~FLAG_SCALE_F32:
        raise TritError(f"неизвестные флаги: 0x{flags:02x}")
    scale_dtype = "float32" if flags & FLAG_SCALE_F32 else "float16"
    expected_scales = n_blocks(n_trits, scale_block)
    if n_scale != expected_scales:
        raise TritError(f"в заголовке {n_scale} блоков масштаба, ожидалось {expected_scales}")
    if n_scale * TRITS_PER_STORAGE_BLOCK < n_trits:
        raise TritError("заголовок врёт: масштабов не хватает на все триты")

    scale_size = 2 if scale_dtype == "float16" else 4
    n_words = _storage_words(n_trits)
    expected_len = _HEADER.size + n_scale * scale_size + n_words * 4
    if len(data) != expected_len:
        raise TritError(f"длина сегмента {len(data)}, ожидалось {expected_len}")

    offset = _HEADER.size
    scales = list(struct.unpack_from(f"<{n_scale}{_SCALE_CODE[scale_dtype]}", data, offset))
    offset += n_scale * scale_size
    words = [int.from_bytes(data[offset + 4 * i: offset + 4 * i + 4], "little")
             for i in range(n_words)]
    trits = unpack20_words(words, n_trits)
    return trits, [float(s) for s in scales]


def segment_info(data: bytes, scale_block: int = BLOCK_SIZE) -> TritSegment:
    """Прочитать только заголовок — для логов и проверок экспорта."""
    if len(data) < _HEADER.size:
        raise TritError("сегмент короче заголовка TRT1")
    magic, version, flags, n_trits, n_scale = _HEADER.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION:
        raise TritError("чужой формат или версия")
    return TritSegment(
        n_trits=n_trits,
        n_scale_blocks=n_scale,
        n_words=_storage_words(n_trits),
        n_storage_blocks=(n_trits + TRITS_PER_STORAGE_BLOCK - 1) // TRITS_PER_STORAGE_BLOCK,
        scale_dtype="float32" if flags & FLAG_SCALE_F32 else "float16",
    )


# ---------------------------------------------------------------------------
# Torch-путь: подключается лениво
# ---------------------------------------------------------------------------


def _require_torch():
    """Импортировать torch или объяснить, что делать на машине без GPU."""
    try:
        import torch  # noqa: PLC0415 - ленивый импорт здесь принципиален
    except ImportError as exc:
        raise MissingTorchError(
            "для QAT нужен PyTorch. Установите зависимости GPU-машины:\n"
            "    python3 -m pip install -r qat/requirements-qat.txt"
        ) from exc
    return torch


def _quantize_last_dim(
    weight: "torch.Tensor",
    block_size: int,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Общая сердцевина: (dequantized, trits int8, scales) для последней оси."""
    torch = _require_torch()
    _check_block_size(block_size)
    if weight.ndim < 1:
        raise TritError("вес должен иметь хотя бы одно измерение")
    original_shape = tuple(weight.shape)
    in_features = original_shape[-1]
    if in_features == 0:
        raise TritError("пустой вес квантовать нечего")

    w = weight.detach().float()
    nb = n_blocks(in_features, block_size)
    pad = nb * block_size - in_features
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    blocks = w.reshape(-1, nb, block_size)
    scales = blocks.abs().amax(dim=-1)  # [rows, nb]
    safe = torch.where(scales > 0, scales, torch.ones_like(scales))
    ratio = blocks / safe.unsqueeze(-1)
    q = torch.sign(ratio) * torch.floor(ratio.abs() + 0.5)
    q = q.clamp(-1.0, 1.0)
    q = torch.where((scales > 0).unsqueeze(-1), q, torch.zeros_like(q))
    deq = (q * scales.unsqueeze(-1)).reshape(w.shape)[..., :in_features]
    trits = q.to(torch.int8).reshape(w.shape)[..., :in_features]
    scales = scales.reshape(original_shape[:-1] + (nb,))
    return deq.reshape(original_shape), trits.reshape(original_shape), scales


def ternary_weight_torch(weight: "torch.Tensor", block_size: int = BLOCK_SIZE) -> "torch.Tensor":
    """Квантованный вес без графа (для forward/экспорта)."""
    return _quantize_last_dim(weight, block_size)[0]


def ternary_ste_torch(weight: "torch.Tensor", block_size: int = BLOCK_SIZE) -> "torch.Tensor":
    """Тернарный вес со STE: forward = w_tern, backward = dL/dw.

    В задаче формула записана как ``w_tern + (w - w_tern).detach()``. В прямом
    проходе она даёт ``w_tern + w - w_tern = w``, то есть квантование в forward
    не попадает. Эквивалентная по смыслу запись STE:

        ``w + (w_tern - w).detach()``

    forward = ``w_tern``, производная по ``w`` = 1. Именно она здесь и
    используется; ``w_tern`` считается без графа (round/clamp всё равно
    почти всюду имеют нулевую производную).
    """
    torch = _require_torch()
    with torch.no_grad():
        w_tern = ternary_weight_torch(weight, block_size).detach()
    w_tern = w_tern.to(weight.dtype)
    return weight + (w_tern - weight).detach()


def quantize_torch(
    weight: "torch.Tensor",
    block_size: int = BLOCK_SIZE,
) -> dict[str, object]:
    """Квантовать тензор для экспорта: триты int8 и масштабы блоков.

    Возвращает словарь:

    * ``trits`` — int8 того же shape, значения -1/0/+1;
    * ``scales`` — float32, shape ``weight.shape[:-1] + (n_blocks,)``;
    * ``n_scale_blocks`` — число групп масштаба на строку;
    * ``block_size`` — размер группы.
    """
    torch = _require_torch()
    with torch.no_grad():
        deq, trits, scales = _quantize_last_dim(weight, block_size)
        # Проверка прямо в момент квантования: триты и масштабы обязаны
        # воспроизводить dequantized бит-в-бит (в float32).
        in_features = trits.shape[-1]
        rows = trits.numel() // in_features
        expanded = scales.reshape(rows, -1).repeat_interleave(block_size, dim=1)
        restored = trits.reshape(rows, in_features).float() * expanded[:, :in_features]
        if not torch.equal(restored, deq.reshape(rows, in_features)):
            raise TritError("внутренняя ошибка квантования: dequantize не совпал")
    return {
        "trits": trits.contiguous(),
        "scales": scales.contiguous(),
        "n_scale_blocks": int(scales.shape[-1]),
        "block_size": block_size,
    }


def pack20_torch(trits: "torch.Tensor", chunk_words: int = 1 << 20) -> "torch.Tensor":
    """Упаковать тензор тритов в int64-слова по 20 тритов (векторно, chunked).

    Для 1.5B-модели чистый Python был бы слишком медленным, поэтому экспорт
    использует эту функцию. Результат — слова в диапазоне ``0..3^20-1``;
    хвост добит нулевыми тритами.
    """
    torch = _require_torch()
    if chunk_words <= 0:
        raise TritError("chunk_words должен быть положительным")
    flat = trits.detach().reshape(-1).to(torch.int64)
    if flat.numel() and (flat.min().item() < -1 or flat.max().item() > 1):
        raise TritError("в тензоре есть значения вне {-1, 0, +1}")
    pad = (-flat.numel()) % TRITS_PER_WORD
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.int64, device=flat.device)])
    pow3 = torch.tensor(_POW3, dtype=torch.int64, device=flat.device)
    words: list[torch.Tensor] = []
    step = chunk_words * TRITS_PER_WORD
    for start in range(0, flat.numel(), step):
        group = flat[start:start + step].reshape(-1, TRITS_PER_WORD) + 1
        words.append((group * pow3).sum(dim=1))
    if not words:
        return torch.empty(0, dtype=torch.int64, device=flat.device)
    return torch.cat(words)
