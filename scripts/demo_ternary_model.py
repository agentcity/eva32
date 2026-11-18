#!/usr/bin/env python3
"""Демонстрационная троичная модель: настоящий .trit, настоящий .vtm, прогон в ВТМ.

Веса случайные — модель ничего не знает и по-русски не отвечает. Цель другая:
показать, что путь «триты на диске → контейнер → исполнение» работает на реальных
файлах, а не только в тестах.

    python3 scripts/demo_ternary_model.py --out demo/
    ./src/vtm/vtm run demo/model.vtm --tokens 1,2,3 --max-new 1
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import vtm_ref  # noqa: E402
from tools.trit_codec import pack20, unpack20  # noqa: E402
from tools.vtm_compile import Op, make_synthetic  # noqa: E402

TRIT_KIND = 0
TRITS_PER_WORD = 20
WORDS_PER_STORAGE_BLOCK = 8
TRITS_PER_STORAGE_BLOCK = TRITS_PER_WORD * WORDS_PER_STORAGE_BLOCK  # 160


def _trits_of(tensor) -> list[int]:
    """Достать триты из троичного тензора (у него они в поле `trits`)."""
    raw = getattr(tensor, "trits", None)
    if raw is None:
        raw = getattr(tensor, "data", []) or []
    return [max(-1, min(1, int(round(float(v))))) for v in raw]


def _scales_of(tensor) -> list[float]:
    return [float(x) for x in (getattr(tensor, "scales", None) or [])]


def write_trit_file(path: str, tensors: list[tuple[str, list[int], list[float]]]) -> int:
    """Собрать `.trit`: подряд TRT1-сегменты, по одному на тензор."""
    blob = bytearray()
    for _name, trits, tensor_scales in tensors:
        n = len(trits)
        n_scale = (n + 127) // 128
        words_needed = (n + TRITS_PER_STORAGE_BLOCK - 1) // TRITS_PER_STORAGE_BLOCK
        padded = trits + [0] * (words_needed * TRITS_PER_STORAGE_BLOCK - n)
        # Масштабы настоящие: по одному на блок 128 (как в QAT-экспорте).
        scales = (tensor_scales + [tensor_scales[-1] if tensor_scales else 1.0] * n_scale)[:n_scale]
        blob += b"TRT1"
        blob += struct.pack("<BBQI", 1, 1, n, n_scale)   # версия, флаги(f32), n_trits, n_scale
        blob += struct.pack(f"<{n_scale}f", *scales)
        for i in range(0, len(padded), TRITS_PER_WORD):
            words = pack20(padded[i:i + TRITS_PER_WORD])
            blob += words
    with open(path, "wb") as fh:
        fh.write(blob)
    return len(blob)


def read_trit_file(path: str) -> list[tuple[str, list[int]]]:
    """Прочитать `.trit` обратно — доказательство, что файл настоящий."""
    raw = open(path, "rb").read()
    pos, out = 0, []
    while pos < len(raw):
        if raw[pos:pos + 4] != b"TRT1":
            raise ValueError(f"сегмент {len(out)}: нет магии TRT1")
        _ver, _flags, n_trits, n_scale = struct.unpack_from("<BBQI", raw, pos + 4)
        pos += 4 + 2 + 8 + 4
        pos += n_scale * 4
        words_needed = (n_trits + TRITS_PER_STORAGE_BLOCK - 1) // TRITS_PER_STORAGE_BLOCK
        n_words = words_needed * WORDS_PER_STORAGE_BLOCK
        payload = raw[pos:pos + n_words * 4]
        pos += n_words * 4
        out.append((f"seg{len(out)}", unpack20(payload, n_trits)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "demo"))
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = make_synthetic(seed=7, size=args.size, mode="attn")
    # Синтетика считает логиты, но не выбирает токен. Добавляем SAMP перед HALT,
    # чтобы модель действительно «генерировала», а не только считала.
    if model.program and model.program[-1].op == int(Op.HALT):
        logits_reg = model.program[-2].dst
        token_tensor = vtm_ref.Tensor.f32("token", (1,), [0.0])
        token_idx = len(model.tensors)
        model.tensors.append(token_tensor)
        halt = model.program.pop()
        model.program.append(vtm_ref.Instr(op=int(Op.SAMP), flags=0, dst=token_idx,
                                           a=logits_reg, b=0, c=0, imm=0))
        model.program.append(halt)
    trit_tensors = [(t.name, _trits_of(t), _scales_of(t)) for t in model.tensors
                    if getattr(t, "kind", None) == TRIT_KIND]
    fp32_params = sum(len(getattr(t, "data", []) or []) for t in model.tensors
                      if getattr(t, "kind", None) != TRIT_KIND)
    trit_params = sum(len(tr) for _n, tr, _s in trit_tensors)
    scale_count = sum(len(_s) for _n, _tr, _s in trit_tensors)

    trit_path = os.path.join(args.out, "ternary.trit")
    size_trit = write_trit_file(trit_path, trit_tensors)

    vtm_path = os.path.join(args.out, "model.vtm")
    vtm_ref.write_vtm(vtm_path, model.tensors, model.program, model.meta)
    size_vtm = os.path.getsize(vtm_path)

    back = read_trit_file(trit_path)
    same = all(a == b for (_n, a, _s), (_n2, b) in zip(trit_tensors, back))

    print("демонстрационная троичная модель")
    print(f"  тритов в весах      : {trit_params:,} ({len(trit_tensors)} тензоров)")
    print(f"  fp32-параметров     : {fp32_params:,} (эмбеддинги, norm)")
    print(f"  масштабов на диске  : {scale_count:,} (один на блок 128)")
    print(f"  ternary.trit        : {size_trit:,} байт")
    print(f"  model.vtm           : {size_vtm:,} байт "
          f"({len(model.tensors)} тензоров, {len(model.program)} инструкций)")
    fp32_bytes = trit_params * 4
    print(f"  fp32-эквивалент этих весов: {fp32_bytes:,} байт")
    print(f"  сжатие по файлу      : {fp32_bytes / size_trit:.1f}× против fp32 "
          f"(теория по тритам: {fp32_bytes / (trit_params * 1.585 / 8):.1f}×, "
          f"разница — масштабы)")
    print(f"  .trit читается обратно и совпадает: {'да' if same else 'НЕТ'}")
    print()
    print("прогон:")
    print(f"  ./src/vtm/vtm run {os.path.relpath(vtm_path, ROOT)} --tokens 1,2,3 --max-new 1")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
