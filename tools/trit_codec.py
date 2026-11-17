#!/usr/bin/env python3
"""Упаковка тритов {-1,0,+1} — прототип формата.

Три режима:
  pack20/unpack20 — 20 тритов в 32-битное слово: 1.60 бита/трит, нативное выравнивание
                    (ОСНОВНОЙ формат, см. docs/trit-format.md)
  pack5/unpack5   — 5 тритов в байт (3^5=243 <= 256): те же 1.60 бита, но без выравнивания
  pack8/unpack8   — 8 тритов + 3 бита метаданных в 16-бит слово: 2.00 бита/трит
                    (только если 3 бита несут масштаб блока)

Триты кодируются цифрами: -1 -> 0, 0 -> 1, +1 -> 2.
"""
from __future__ import annotations
import random

DIGITS = (-1, 0, 1)
_TO_DIGIT = {-1: 0, 0: 1, 1: 2}
_FROM_DIGIT = {0: -1, 1: 0, 2: 1}


def pack20(trits: list[int]) -> bytes:
    """20 тритов в 32-битное слово. Неполный блок дополняется нулевыми тритами.

    3^20 = 3_486_784_401 <= 2^32, поэтому слово несёт 20 тритов без потерь.
    """
    pad = (-len(trits)) % 20
    trits = list(trits) + [0] * pad
    out = bytearray()
    for i in range(0, len(trits), 20):
        v = 0
        for k, t in enumerate(trits[i:i + 20]):
            v += _TO_DIGIT[t] * (3 ** k)
        out += v.to_bytes(4, "little")
    return bytes(out)


def unpack20(data: bytes, count: int) -> list[int]:
    if len(data) % 4:
        raise ValueError("длина данных должна быть кратна 4")
    out: list[int] = []
    for i in range(0, len(data), 4):
        v = int.from_bytes(data[i:i + 4], "little")
        if v > 3 ** 20 - 1:
            raise ValueError("слово испорчено: вне диапазона 3^20")
        for _ in range(20):
            out.append(_FROM_DIGIT[v % 3])
            v //= 3
    return out[:count]


def pack5(trits: list[int]) -> bytes:
    """Неполный блок дополняется нулевыми тритами — как в реальных форматах."""
    pad = (-len(trits)) % 5
    trits = list(trits) + [0] * pad
    out = bytearray()
    for i in range(0, len(trits), 5):
        v = 0
        for k, t in enumerate(trits[i:i + 5]):
            v += _TO_DIGIT[t] * (3 ** k)
        out.append(v)              # максимум 3^5-1 = 242 < 256
    return bytes(out)


def unpack5(data: bytes, count: int) -> list[int]:
    out: list[int] = []
    for b in data:
        v = b
        for _ in range(5):
            out.append(_FROM_DIGIT[v % 3])
            v //= 3
    return out[:count]


def pack8(trits: list[int], meta: int = 0, meta_bits: int = 3) -> bytes:
    """8 тритов (13 бит) + meta_bits метаданных в 16-бит слово."""
    pad = (-len(trits)) % 8
    trits = list(trits) + [0] * pad
    if not 0 <= meta < (1 << meta_bits):
        raise ValueError("meta не влезает в отведённые биты")
    out = bytearray()
    for i in range(0, len(trits), 8):
        v = 0
        for k, t in enumerate(trits[i:i + 8]):
            v += _TO_DIGIT[t] * (3 ** k)   # максимум 3^8-1 = 6560
        word = v | (meta << 13)
        out += word.to_bytes(2, "little")
    return bytes(out)


def unpack8(data: bytes, count: int, meta_bits: int = 3) -> list[int]:
    out: list[int] = []
    if len(data) % 2:
        raise ValueError("длина данных должна быть чётной")
    for i in range(0, len(data), 2):
        word = int.from_bytes(data[i:i + 2], "little")
        v = word & 0x1FFF               # младшие 13 бит — данные
        meta = word >> 13
        if meta >= (1 << meta_bits):
            raise ValueError("слово испорчено: метаданные вне диапазона")
        for _ in range(8):
            out.append(_FROM_DIGIT[v % 3])
            v //= 3
    return out[:count]


def _selftest() -> None:
    random.seed(1)

    # 1) исчерпывающая проверка pack5: все 243 комбинации
    for code in range(243):
        v, digs = code, []
        for _ in range(5):
            digs.append(_FROM_DIGIT[v % 3]); v //= 3
        assert unpack5(pack5(digs), 5) == digs, code

    # 2) round-trip на больших векторах
    for n in (8, 40, 160, 1024):
        t = [random.choice(DIGITS) for _ in range(n)]
        assert unpack5(pack5(t), n) == t, n
        assert unpack8(pack8(t), n) == t, n

    # 3) pack20: граничные векторы и round-trip
    assert unpack20(pack20([1] * 20), 20) == [1] * 20
    assert unpack20(pack20([-1] * 20), 20) == [-1] * 20
    assert unpack20(pack20([1, -1] * 10), 20) == [1, -1] * 10
    for n in (20, 40, 160, 1000):
        t = [random.choice(DIGITS) for _ in range(n)]
        assert unpack20(pack20(t), n) == t, n
    # максимальное слово не переполняет 32 бита
    assert 3 ** 20 - 1 < 2 ** 32

    # 4) метаданные переживают упаковку
    t = [random.choice(DIGITS) for _ in range(16)]
    blob = pack8(t, meta=5)
    assert unpack8(blob, 16) == t
    assert (int.from_bytes(blob[:2], "little") >> 13) == 5

    # 5) плотность
    n = 4096
    t = [random.choice(DIGITS) for _ in range(n)]
    print(f"{n} тритов: pack5 {len(pack5(t))} Б ({(len(pack5(t))*8)/n:.2f} б/трит), "
          f"pack8 {len(pack8(t))} Б ({(len(pack8(t))*8)/n:.2f} б/трит), "
          f"pack20 {len(pack20(t))} Б ({(len(pack20(t))*8)/n:.2f} б/трит)")
    print("самопроверка: ok")


if __name__ == "__main__":
    _selftest()
