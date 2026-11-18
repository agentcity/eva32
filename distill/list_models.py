#!/usr/bin/env python3
"""Показать доступные модели у учителя — чтобы не угадывать имя в конфиге.

    python3 distill/list_models.py            # только deepseek
    python3 distill/list_models.py --all      # все
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill.teacher import load_env  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--filter", default="deepseek")
    args = ap.parse_args()

    load_env()
    base = os.environ.get("TEACHER_BASE", "").rstrip("/")
    key = os.environ.get("TEACHER_KEY", "")
    if not base or not key:
        print("нужны TEACHER_BASE и TEACHER_KEY в .env (см. .env.example)", file=sys.stderr)
        return 2

    req = urllib.request.Request(base + "/v1/models",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.reason}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"сеть: {exc.reason}", file=sys.stderr)
        return 1

    ids = [m.get("id", "") for m in data.get("data", [])]
    shown = ids if args.all else [i for i in ids if args.filter.lower() in i.lower()]
    for i in sorted(shown):
        print(i)
    print(f"\nвсего моделей: {len(ids)}, показано: {len(shown)}", file=sys.stderr)
    return 0 if shown else 1


if __name__ == "__main__":
    sys.exit(main())
