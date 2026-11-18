#!/usr/bin/env python3
"""Собрать обучающую смесь из нескольких банков пар с весами по слоям.

Зачем: код, бизнес, общие инструкции и поведение — разные слои, и пропорция между
ними влияет на модель сильнее, чем суммарный объём. Смесь собирается детерминированно
(seed), с дедупликацией по вопросу и разбиением train/val без утечки.

    python3 tools/build_mixture.py --total 90000 --out runs/mixture
    python3 tools/build_mixture.py --total 5000 --out runs/mixture_small --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Слои смеси: имя → (доля, файлы-кандидаты по убыванию приоритета).
LAYERS: dict[str, tuple[float, list[str]]] = {
    "code": (0.40, [
        "distill/out/code_pairs.jsonl",
        "distill/pulled/ru_codefeedback.jsonl",
        "distill/pulled/russian_code_qa.jsonl",
        "distill/pulled/magicoder_ru.jsonl",
    ]),
    "business": (0.30, [
        "distill/out/business_pairs.jsonl",
        "distill/out/pairs.jsonl",
    ]),
    "general": (0.20, [
        "distill/pulled/russian_instructions.jsonl",
        "distill/pulled/easy_instructions.jsonl",
    ]),
    "behavior": (0.10, [
        "distill/out/behavior_pairs.jsonl",
    ]),
}


@dataclass
class Pair:
    question: str
    answer: str
    layer: str
    source: str


def _key(question: str) -> str:
    """Ключ дедупликации: без регистра, пунктуации и лишних пробелов."""
    return re.sub(r"[^а-яa-z0-9]+", "", question.lower())[:200]


def read_pairs(path: pathlib.Path, layer: str) -> list[Pair]:
    if not path.exists():
        return []
    out: list[Pair] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or row.get("response") or "").strip()
            if not question or not answer:
                continue
            source = (row.get("meta") or {}).get("source", path.stem)
            out.append(Pair(question, answer, layer, source))
    return out


def take_layer(
    layer: str, need: int, rng: random.Random, seen: set[str]
) -> tuple[list[Pair], dict[str, int]]:
    """Взять `need` пар слоя, начиная с приоритетных файлов, без дублей."""
    out: list[Pair] = []
    used: dict[str, int] = {}
    for rel in LAYERS[layer][1]:
        path = ROOT / rel
        pairs = read_pairs(path, layer)
        if not pairs:
            continue
        rng.shuffle(pairs)
        taken = 0
        for pair in pairs:
            if len(out) >= need:
                break
            k = _key(pair.question)
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(pair)
            taken += 1
        if taken:
            used[rel] = taken
        if len(out) >= need:
            break
    return out, used


def build(total: int, *, seed: int = 42) -> tuple[list[Pair], dict[str, dict]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    chosen: list[Pair] = []
    stats: dict[str, dict] = {}

    # Первый проход — по долям; если слой не набрал, остаток добираем позже.
    shortfall = 0
    for layer, (share, _files) in LAYERS.items():
        need = int(total * share)
        pairs, used = take_layer(layer, need, rng, seen)
        chosen.extend(pairs)
        stats[layer] = {"нужно": need, "взято": len(pairs), "файлы": used}
        shortfall += max(0, need - len(pairs))

    # Второй проход — добрать недостающее из самых наполненных слоёв.
    if shortfall:
        for layer in ("code", "general", "business", "behavior"):
            if shortfall <= 0:
                break
            pairs, used = take_layer(layer, shortfall, rng, seen)
            chosen.extend(pairs)
            stats[layer]["добрано"] = len(pairs)
            for rel, n in used.items():
                stats[layer]["файлы"][rel] = stats[layer]["файлы"].get(rel, 0) + n
            shortfall -= len(pairs)

    rng.shuffle(chosen)
    return chosen, stats


def split(pairs: list[Pair], val_ratio: float = 0.10) -> tuple[list[Pair], list[Pair]]:
    n_val = max(1, int(len(pairs) * val_ratio)) if pairs else 0
    return pairs[n_val:], pairs[:n_val]


def _write(path: pathlib.Path, pairs: list[Pair]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps({
                "question": pair.question,
                "answer": pair.answer,
                "meta": {"layer": pair.layer, "source": pair.source},
            }, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--total", type=int, default=90000)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "runs" / "mixture")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    pairs, stats = build(args.total, seed=args.seed)
    train, val = split(pairs, args.val_ratio)

    report = {
        "total": len(pairs),
        "train": len(train),
        "val": len(val),
        "seed": args.seed,
        "layers": stats,
        "доли_факт": {
            layer: round(len([p for p in pairs if p.layer == layer]) / len(pairs), 3)
            if pairs else 0.0
            for layer in LAYERS
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    train_sha = _write(args.out / "train.jsonl", train)
    val_sha = _write(args.out / "val.jsonl", val)
    (args.out / "stats.json").write_text(
        json.dumps({**report, "train_sha256": train_sha, "val_sha256": val_sha},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"записано: {args.out}/train.jsonl ({len(train)}), val.jsonl ({len(val)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
