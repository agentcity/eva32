#!/usr/bin/env python3
"""Проверка, что веса действительно троичные: {-1, 0, +1} с масштабом на блок.

Читает `.trit` (сегменты TRT1) или контейнер `.vtm` и печатает:
- все ли триты из {-1, 0, +1} (по построению формата — да, но проверяем);
- распределение (сколько −1, 0, +1) — нули показывают, сколько связей отключено;
- статистику масштабов;
- размер в битах на вес.

    python3 tools/check_ternary.py demo/ternary.trit
    python3 tools/check_ternary.py demo/model.vtm
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.trit_codec import unpack20  # noqa: E402


def read_trit_segments(path: str) -> list[dict]:
    raw = open(path, "rb").read()
    pos, out = 0, []
    while pos < len(raw):
        if raw[pos:pos + 4] != b"TRT1":
            raise ValueError(f"сегмент {len(out)}: нет магии TRT1")
        _ver, flags, n_trits, n_scale = struct.unpack_from("<BBQI", raw, pos + 4)
        pos += 4 + 2 + 8 + 4
        scale_size = 4 if flags & 1 else 2
        scales = struct.unpack_from(f"<{n_scale}{'f' if scale_size == 4 else 'e'}", raw, pos)
        pos += n_scale * scale_size
        n_words = (n_trits + 159) // 160 * 8
        trits = unpack20(raw[pos:pos + n_words * 4], n_trits)
        pos += n_words * 4
        out.append({"trits": trits, "scales": scales})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    segs = read_trit_segments(args.path)
    total = {"-1": 0, "0": 0, "1": 0}
    bad = 0
    all_scales: list[float] = []
    for seg in segs:
        for t in seg["trits"]:
            if t == -1:
                total["-1"] += 1
            elif t == 0:
                total["0"] += 1
            elif t == 1:
                total["1"] += 1
            else:
                bad += 1
        all_scales.extend(float(s) for s in seg["scales"])
    n = sum(total.values()) + bad
    print(f"файл: {args.path}")
    print(f"сегментов: {len(segs)}, тритов: {n:,}")
    print(f"вне {{-1,0,+1}}: {bad}  {'✅' if bad == 0 else '❌ ОШИБКА'}")
    if n:
        print(f"доля нулей: {total['0'] / n:.1%}  (сколько связей отключено)")
        print(f"  −1: {total['-1']:>12,}  {total['-1'] / n:6.1%}")
        print(f"   0: {total['0']:>12,}  {total['0'] / n:6.1%}")
        print(f"  +1: {total['1']:>12,}  {total['1'] / n:6.1%}")
    if all_scales:
        print(f"масштабов: {len(all_scales):,}")
        print(f"  min {min(all_scales):.6f} | median {sorted(all_scales)[len(all_scales)//2]:.6f} "
              f"| max {max(all_scales):.6f}")
    if n:
        bits = (n * 1.585 + len(all_scales) * 16) / n
        print(f"бит на вес с учётом масштабов: {bits:.3f}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
