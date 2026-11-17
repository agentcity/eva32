#!/usr/bin/env python3
"""Упаковка тритов {-1,0,+1} — прототип формата.

Два режима:
  pack5/unpack5  — 5 тритов в байт (3^5=243 <= 256): 1.60 бита/трит
  pack8/unpack8  — 8 тритов + 3 бита метаданных в 16-бит слово: 2.00 бита/трит
                   (но 3 свободных бита на блок можно занять масштабом)

Триты кодируются цифрами: -1 -> 0, 0 -> 1, +1 -> 2.
"""
from __future__ import annotations
import random

DIGITS = (-1, 0, 1)
_TO_DIGIT = {-1: 0, 0: 1, 1: 2}
_FROM_DIGIT = {0: -1, 1: 0, 2: 1}


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

    # 3) метаданные переживают упаковку
    t = [random.choice(DIGITS) for _ in range(16)]
    blob = pack8(t, meta=5)
    assert unpack8(blob, 16) == t
    assert (int.from_bytes(blob[:2], "little") >> 13) == 5

    # 4) плотность
    n = 4096
    t = [random.choice(DIGITS) for _ in range(n)]
    print(f"{n} тритов: pack5 {len(pack5(t))} Б ({(len(pack5(t))*8)/n:.2f} б/трит), "
          f"pack8 {len(pack8(t))} Б ({(len(pack8(t))*8)/n:.2f} б/трит)")
    print("самопроверка: ok")


if __name__ == "__main__":
    _selftest()
