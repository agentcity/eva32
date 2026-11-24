#!/usr/bin/env python3
"""Декодирование ответов из готового ``.vtm`` через chat-template HF (задача 0142).

Зачем
=====

``src/vtm/vtm run --text --chat`` умеет и токенизацию, и чат-шаблон, но
собранный в C шаблон — это **реконструкция** шаблона Qwen2.5. Здесь наоборот:
промпт собирает ``AutoTokenizer.apply_chat_template(..., add_generation_prompt=True,
tokenize=False)`` из того же ``tokenizer_config.json``, что лежит рядом с HF-моделью,
затем ids уходят в ``vtm run --tokens``, а сгенерированные id декодируются тем
же токенизатором. Так ответ — это то, что модель видит в проде, а не то, что
мы думаем о её шаблоне.

Ничего не пишет в репозиторий, кроме ``--json-out``: это инструмент отчёта,
а не часть рантайма. Запуск — под GPU-стеком (нужен transformers)::

    PYTHONPATH=.pylibs python3.13 tools/cold_bench.py \\
        --vtm runs/cold/qwen3b-trit/model.vtm \\
        --tokenizer runs/cold/qwen3b/tokenizer.json \\
        --prompt "Столица Франции — это" --max-new 40

Формат ``--vtm``/``--tokens`` — как у ``src/vtm/vtm``: контейнер VTM v0.2.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VTM = ROOT / "src" / "vtm" / "vtm"

DEFAULT_PROMPTS = (
    "Столица Франции — это",
    "Объясни в двух предложениях, что такое тернарное квантование.",
    "Напиши на Python функцию, которая считает сумму чисел от 1 до n.",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cold_bench.py",
        description="chat-template HF -> ids -> vtm run --tokens -> текст ответа",
    )
    parser.add_argument("--vtm", required=True, help="готовый контейнер .vtm")
    parser.add_argument("--tokenizer", required=True,
                        help="каталог модели или tokenizer.json с chat_template")
    parser.add_argument("--prompt", action="append", default=None,
                        help="промпт (можно повторять; заменяет встроенный набор)")
    parser.add_argument("--system", default=None, help="system-сообщение чата")
    parser.add_argument("--max-new", type=int, default=48,
                        help="сколько токенов генерировать (по умолчанию 48)")
    parser.add_argument("--max-prompt-tokens", type=int, default=512,
                        help="обрезать промпт до N токенов (контекст контейнера)")
    parser.add_argument("--timeout", type=float, default=None,
                        help="таймаут генерации, секунды (vtm run --timeout)")
    parser.add_argument("--vtm-bin", default=str(DEFAULT_VTM),
                        help="бинарник интерпретатора")
    parser.add_argument("--json-out", default=None, help="куда записать результат")
    return parser


def chat_prompt_text(tokenizer: Any, prompt: str, system: str | None) -> str:
    """Текст промпта ровно так, как его собирает HF chat template."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if not isinstance(text, str) or not text:
        raise SystemExit("apply_chat_template вернул не строку — нет chat_template?")
    return text


def prompt_ids(tokenizer: Any, text: str, limit: int) -> list[int]:
    ids = [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]
    if limit and len(ids) > int(limit):
        ids = ids[-int(limit):]
    if not ids:
        raise SystemExit("промпт дал пустой список токенов")
    return ids


def parse_vtm_output(text: str) -> list[int]:
    """Достать сгенерированные id из строки ``вывод:  1 2 3``."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("вывод:"):
            payload = stripped[len("вывод:"):].strip()
            return [int(value) for value in payload.split()] if payload else []
    raise SystemExit(f"в выводе vtm нет строки 'вывод:':\n{text}")


def run_vtm(binary: str, vtm: Path, ids: Sequence[int], max_new: int,
            timeout: float | None) -> tuple[list[int], str, float]:
    cmd = [binary, "run", str(vtm), "--tokens", ",".join(str(token) for token in ids),
           "--max-new", str(int(max_new))]
    if timeout:
        cmd += ["--timeout", str(float(timeout))]
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - started
    if proc.returncode != 0:
        raise SystemExit(
            f"vtm run: код {proc.returncode}\n{proc.stderr.strip()[-2000:]}"
        )
    return parse_vtm_output(proc.stdout), proc.stdout, elapsed


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError:
        print("нужен transformers: PYTHONPATH=.pylibs python3.13 tools/cold_bench.py …",
              file=sys.stderr)
        return 2
    vtm = Path(args.vtm).expanduser().resolve()
    if not vtm.is_file():
        print(f"нет контейнера: {vtm}", file=sys.stderr)
        return 2
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=False)
    prompts = [str(item) for item in (args.prompt or DEFAULT_PROMPTS)]

    results: list[dict[str, Any]] = []
    for prompt in prompts:
        text = chat_prompt_text(tokenizer, prompt, args.system)
        ids = prompt_ids(tokenizer, text, args.max_prompt_tokens)
        generated, _stdout, elapsed = run_vtm(
            args.vtm_bin, vtm, ids, args.max_new, args.timeout
        )
        answer = tokenizer.decode(generated, skip_special_tokens=True)
        record = {
            "prompt": prompt,
            "prompt_tokens": len(ids),
            "prompt_text": text,
            "generated_tokens": generated,
            "answer": answer,
            "seconds": round(elapsed, 3),
            "tokens_per_second": round(len(generated) / elapsed, 2) if elapsed else None,
        }
        results.append(record)
        print(f"=== {prompt} ===")
        print(f"промпт: {len(ids)} токенов; ответ: {len(generated)} токенов "
              f"за {elapsed:.1f} с")
        print(answer)
        print()

    payload = {
        "format": "eva32.cold.decode",
        "format_version": 1,
        "vtm": str(vtm),
        "vtm_size_bytes": vtm.stat().st_size,
        "tokenizer": str(Path(args.tokenizer).expanduser().resolve()),
        "max_new": int(args.max_new),
        "system": args.system,
        "prompts": results,
    }
    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
        print(f"результат: {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
