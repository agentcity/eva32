#!/usr/bin/env python3
"""Инспекция и замер MoE-программы VTM (задача 0172).

Две команды:

`check model.vtm [--json]`
    Разобрать контейнер и проверить MoE-часть программы: у каждой инструкции
    MOE — роутер, три банка (смежность c/c+1/c+2 — часть ABI, см. `vtm.h`),
    формы, kind, top-k. Заодно печатает, сколько байт экспертов читается на
    токен и какую долю весов это составляет: по этим числам видно, почему MoE
    «летает» на 16 ГиБ — модель 19 ГиБ, а на токен читается ~1,2 ГБ (если
    page cache не удержал горячие банки).

`measure model.vtm --tokens 1,2,3 --max-new 10`
    Прогнать `src/vtm/vtm run` под `/usr/bin/time -l` и собрать ток/с, RSS и
    страничные отказы. Счётчики кэша экспертов (hits/misses) — из stderr при
    `VTM_MOE_TRACE=1`; кэш включается `VTM_MOE_CACHE_MB`, упреждение —
    `VTM_MOE_MADVISE=0`.

Запуск:
    python3 tools/moe_program.py check /tmp/moe/slice.vtm
    python3 tools/moe_program.py measure /tmp/moe/slice.vtm --tokens 1,2,3 \
        --max-new 8 --env VTM_MOE_CACHE_MB=1024
"""
from __future__ import annotations

import argparse
import json
import os
import contextlib
import io
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vtm_ref  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VTM_BIN = ROOT / "src" / "vtm" / "vtm"

#: Блоки kind 6/7: 256 значений, 144 байта у Q4_K, 210 у Q6_K.
BLOCK_VALUES = vtm_ref.Q4K_BLOCK_VALUES
BLOCK_BYTES = {vtm_ref.KIND_Q4K: vtm_ref.Q4K_BLOCK_BYTES,
               vtm_ref.KIND_Q6K: vtm_ref.Q6K_BLOCK_BYTES}


class ProgramError(ValueError):
    """Программа не MoE/не сходится — текст для человека."""


def _bank_bytes(tensor: vtm_ref.Tensor) -> int:
    block = BLOCK_BYTES.get(tensor.kind)
    if block is None:
        raise ProgramError(
            f"банк {tensor.name!r}: kind {tensor.kind_name}, ожидался Q4_K/Q6_K"
        )
    return tensor.n_elements // BLOCK_VALUES * block


def check(path: str | Path) -> dict[str, Any]:
    """Проверить MoE-инструкции контейнера. Вернуть сводку (для --json)."""
    vtm = vtm_ref.read_vtm(path)
    n_instr = {int(instr.op): 0 for instr in vtm.program}
    moe: list[dict[str, Any]] = []
    layers: dict[str, dict[str, Any]] = {}
    for index, instr in enumerate(vtm.program):
        n_instr[int(instr.op)] = n_instr.get(int(instr.op), 0) + 1
        if int(instr.op) != vtm_ref.Op.MOE:
            continue
        router = vtm.tensors[int(instr.a)]
        x = vtm.tensors[int(instr.b)]
        base = int(instr.c)
        if base + 2 >= len(vtm.tensors):
            raise ProgramError(
                f"инструкция {index}: банки c/c+1/c+2 = {base} вне таблицы "
                f"({len(vtm.tensors)} тензоров)"
            )
        gate, up, down = (vtm.tensors[base + delta] for delta in (0, 1, 2))
        if len(router.dims) != 2:
            raise ProgramError(f"MOE #{index}: роутер {router.name!r} не 2-D")
        n_experts, hidden = router.dims
        # gate/up — [n_experts, inter, hidden], down — [n_experts, hidden, inter].
        inter = int(gate.dims[1])
        for label, bank, rows, cols in (
            ("gate", gate, inter, hidden),
            ("up", up, inter, hidden),
            ("down", down, hidden, inter),
        ):
            if len(bank.dims) != 3:
                raise ProgramError(
                    f"MOE #{index}: {label}-банк {bank.name!r} не 3-D"
                )
            if tuple(bank.dims) != (n_experts, rows, cols):
                raise ProgramError(
                    f"MOE #{index}: {label}-банк {bank.name!r}: dims "
                    f"{tuple(bank.dims)}, ожидались {(n_experts, rows, cols)}"
                )
        top_k = int(instr.imm)
        if not 0 < top_k <= n_experts:
            raise ProgramError(
                f"MOE #{index}: top_k={top_k} вне 1..{n_experts}"
            )
        expert_bytes = (
            _bank_bytes(gate) // n_experts
            + _bank_bytes(up) // n_experts
            + _bank_bytes(down) // n_experts
        )
        # Имя банка вида "l3.gate_bank" -> слой "l3" (для группировки в отчёте).
        layer = gate.name.split(".")[0] if "." in gate.name else f"#{index}"
        entry = {
            "instruction": index,
            "layer": layer,
            "router": router.name,
            "banks": [gate.name, up.name, down.name],
            "n_experts": int(n_experts),
            "top_k": top_k,
            "hidden": int(hidden),
            "expert_inter": int(inter),
            "expert_bytes": int(expert_bytes),
            "token_bytes": int(expert_bytes) * top_k,
            "renorm": not (int(instr.flags) & vtm_ref.IFLAG_MOE_NO_RENORM),
        }
        moe.append(entry)
        layers.setdefault(layer, {"moe": 0, "token_bytes": 0})
        layers[layer]["moe"] += 1
        layers[layer]["token_bytes"] += entry["token_bytes"]

    total_bank_bytes = sum(
        _bank_bytes(tensor) for tensor in vtm.tensors
        if tensor.kind in BLOCK_BYTES
    )
    token_bytes = sum(entry["token_bytes"] for entry in moe)
    names = {int(op): name for op, name in vtm_ref.OP_NAMES.items()}
    summary = {
        "path": str(path),
        "tensors": len(vtm.tensors),
        "instructions": len(vtm.program),
        "opcodes": {names.get(op, str(op)): count
                    for op, count in sorted(n_instr.items())},
        "moe_layers": len(moe),
        "moe": moe,
        "bank_bytes": int(total_bank_bytes),
        "token_expert_bytes": int(token_bytes),
        "token_expert_share": (
            token_bytes / total_bank_bytes if total_bank_bytes else 0.0
        ),
        "meta": {
            "n_layers": vtm.meta.get("n_layers"),
            "moe": vtm.meta.get("moe"),
            "n_experts": vtm.meta.get("n_experts"),
            "top_k": vtm.meta.get("top_k"),
            "expert_intermediate": vtm.meta.get("expert_intermediate"),
            "qk_norm": vtm.meta.get("qk_norm"),
            "context_len": vtm.meta.get("context_len"),
        },
    }
    return summary


def human_bytes(size: float) -> str:
    value = float(size)
    for unit in ("Б", "КиБ", "МиБ", "ГиБ", "ТиБ"):
        if value < 1024.0 or unit == "ТиБ":
            return f"{value:.1f} {unit}" if unit != "Б" else f"{int(value)} Б"
        value /= 1024.0
    return f"{int(size)} Б"


def print_check(summary: dict[str, Any]) -> None:
    print(f"контейнер: {summary['path']}")
    print(
        f"тензоров {summary['tensors']}, инструкций {summary['instructions']}, "
        f"MoE-слоёв {summary['moe_layers']}"
    )
    print("опкоды: " + ", ".join(
        f"{name}×{count}" for name, count in summary["opcodes"].items()
    ))
    meta = summary["meta"]
    print(
        f"мета: n_layers={meta['n_layers']} moe={meta['moe']} "
        f"n_experts={meta['n_experts']} top_k={meta['top_k']} "
        f"expert_inter={meta['expert_intermediate']} qk_norm={meta['qk_norm']} "
        f"ctx={meta['context_len']}"
    )
    for entry in summary["moe"][:6]:
        print(
            f"  {entry['layer']:>4}: инструкция #{entry['instruction']:<4} "
            f"экспертов {entry['n_experts']:>4} top-{entry['top_k']:<3} "
            f"[{entry['expert_inter']}×{entry['hidden']}] "
            f"на токен {human_bytes(entry['token_bytes'])}"
        )
    if len(summary["moe"]) > 6:
        print(f"  … ещё {len(summary['moe']) - 6} слоёв")
    print(
        f"банки kind 6/7: {human_bytes(summary['bank_bytes'])}; "
        f"на токен читается {human_bytes(summary['token_expert_bytes'])} "
        f"({summary['token_expert_share'] * 100:.1f} % банков)"
    )


_RSS_RE = re.compile(r"(\d+)\s+maximum resident set size")
_FAULTS_RE = re.compile(r"(\d+)\s+page reclaims")
_TIME_RE = re.compile(r"время:\s+([\d.]+)\s+мс")
_TOKS_RE = re.compile(r",\s+(\d+)\s+токен\(ов\),\s+([\d.]+)\s+ток/с")
_MOE_RE = re.compile(
    r"VTM MOE: hits=(\d+) misses=(\d+) cache=(\d+) КиБ advised=(\d+)"
)


def measure(
    path: str | Path,
    tokens: Sequence[int],
    max_new: int = 8,
    *,
    env: dict[str, str] | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Прогнать модель и собрать ток/с, RSS, page faults, счётчики кэша."""
    binary = VTM_BIN
    if not binary.is_file():
        raise ProgramError(
            f"нет {binary} — соберите: make -C src/vtm"
        )
    command = [
        "/usr/bin/time", "-l", str(binary), "run", str(path),
        "--tokens", ",".join(str(int(token)) for token in tokens),
        "--max-new", str(int(max_new)), "--stats",
    ]
    run_env = dict(os.environ)
    run_env.setdefault("VTM_MOE_TRACE", "1")
    if env:
        run_env.update({key: str(value) for key, value in env.items()})
    proc = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, env=run_env
    )
    out = proc.stdout
    err = proc.stderr
    result: dict[str, Any] = {
        "returncode": proc.returncode,
        "command": command,
        "env": {key: run_env[key] for key in ("VTM_MOE_CACHE_MB",
                                              "VTM_MOE_MADVISE",
                                              "VTM_MOE_TRACE")
                if key in run_env},
    }
    match = _TIME_RE.search(err) or _TIME_RE.search(out)
    if match:
        result["ms"] = float(match.group(1))
    match = _TOKS_RE.search(err) or _TOKS_RE.search(out)
    if match:
        result["generated"] = int(match.group(1))
        result["tok_per_s"] = float(match.group(2))
    match = _RSS_RE.search(err)
    if match:
        result["rss_bytes"] = int(match.group(1))
    match = _FAULTS_RE.search(err)
    if match:
        result["page_reclaims"] = int(match.group(1))
    hits = misses = cache_kib = advised = 0
    for line in err.splitlines():
        match = _MOE_RE.search(line)
        if match:
            hits += int(match.group(1))
            misses += int(match.group(2))
            cache_kib = max(cache_kib, int(match.group(3)))
            advised += int(match.group(4))
    if hits or misses:
        result["cache"] = {
            "hits": hits,
            "misses": misses,
            "hit_rate": hits / (hits + misses) if hits + misses else 0.0,
            "cache_kib": cache_kib,
            "advised": advised,
        }
    result["stdout"] = out[-4000:]
    result["stderr"] = err[-4000:]
    return result


def print_measure(result: dict[str, Any]) -> None:
    if result["returncode"] != 0:
        print(f"прогон упал (код {result['returncode']})", file=sys.stderr)
        print(result["stderr"][-2000:], file=sys.stderr)
        return
    speed = result.get("tok_per_s")
    print(f"время: {result.get('ms', 0):.1f} мс, "
          f"{result.get('generated', 0)} токен(ов), "
          f"{(f'{speed:.2f} ток/с') if speed else '—'}")
    if "rss_bytes" in result:
        print(f"RSS: {human_bytes(result['rss_bytes'])}")
    if "page_reclaims" in result:
        print(f"page reclaims: {result['page_reclaims']}")
    cache = result.get("cache")
    if cache:
        print(
            f"кэш экспертов: hits={cache['hits']} misses={cache['misses']} "
            f"({cache['hit_rate'] * 100:.1f} % попаданий), "
            f"{cache['cache_kib']} КиБ, madvise {cache['advised']}"
        )


# ---------------------------------------------------------------------------
# Паритет с ollama по logprobs
# ---------------------------------------------------------------------------


def _log_softmax(logits: Sequence[float]) -> list[float]:
    """log_softmax в double — ровно то, что ollama отдаёт в logprobs."""
    peak = max(logits)
    total = 0.0
    exps = []
    for value in logits:
        item = math.exp(float(value) - float(peak))
        exps.append(item)
        total += item
    shift = math.log(total) + float(peak)
    # log p_i = x_i - (max + log Σ exp(x_j - max)); именно это отдаёт ollama
    # в logprobs (не exp/total — это была бы вероятность, а не логарифм).
    return [float(value) - shift for value in logits]


def _bytes_to_id_map(vocab: dict[str, Any]) -> dict[bytes, int]:
    """Обратная карта «байты токена -> id» (byte-level BPE, как у GPT-2/Qwen)."""
    table = _byte_decoder()
    mapping: dict[bytes, int] = {}
    for index, token in enumerate(vocab["tokens"]):
        mapping.setdefault(_token_bytes(token, table), index)
    return mapping


def _byte_decoder() -> dict[str, int]:
    printable = (list(range(ord("!"), ord("~") + 1))
                 + list(range(ord("¡"), ord("¬") + 1))
                 + list(range(ord("®"), ord("ÿ") + 1)))
    table = {chr(code): code for code in printable}
    extra = 0
    for code in range(256):
        if code not in printable:
            table[chr(256 + extra)] = code
            extra += 1
    return table


def _token_bytes(token: str, table: dict[str, int]) -> bytes:
    out = bytearray()
    for char in token:
        code = table.get(char)
        if code is None:
            out.extend(char.encode("utf-8"))
        else:
            out.append(code)
    return bytes(out)


def parity(
    model: str | Path,
    *,
    ollama_url: str,
    ollama_model: str,
    vocab_path: str | Path,
    prompts: Sequence[str],
    top_logprobs: int = 10,
    tolerance: float = 1e-3,
    workdir: str | Path | None = None,
) -> dict[str, Any]:
    """Сверить логиты VTM с logprobs ollama на тех же промптах.

    Промпт уходит в ollama как текст; из ответа берутся
    `context[:prompt_eval_count]` — точные id, которыми ollama его
    токенизировала, — и те же id прогоняются через C-интерпретатор VTM
    (`tools/parity_real.py`, проба читает тензор `logits`). Сравниваются
    log_softmax(logits) VTM и `top_logprobs` ollama по одним и тем же id:
    так паритет не зависит от того, совпали ли токенизаторы.
    """
    import json as _json
    import urllib.request

    import parity_real

    vocab = _json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    if isinstance(vocab, dict) and "tokens" in vocab:
        tokens = vocab["tokens"]
    else:
        raise ProgramError(f"{vocab_path}: ожидался {{tokens: [...]}}")
    mapping = _bytes_to_id_map({"tokens": tokens})
    cc = parity_real.find_c_compiler()
    if cc is None:
        raise ProgramError("нет C-компилятора — пробу VTM не собрать")
    work = Path(workdir or tempfile.mkdtemp(prefix="moe-parity-"))
    results = []
    for prompt in prompts:
        body = _json.dumps({
            "model": ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 1, "temperature": 0},
            "logprobs": True,
            "top_logprobs": int(top_logprobs),
        }).encode("utf-8")
        request = urllib.request.Request(
            ollama_url.rstrip("/") + "/api/generate",
            data=body, headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                answer = _json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - сеть/JSON: текст для человека
            raise ProgramError(f"ollama не ответила на {prompt!r}: {exc}") from exc
        count = int(answer.get("prompt_eval_count") or 0)
        context = answer.get("context") or []
        if count <= 0 or len(context) < count:
            raise ProgramError(
                f"ollama: в ответе нет context/prompt_eval_count "
                f"({count}, {len(context)}) — нужна версия с logprobs"
            )
        ids = [int(item) for item in context[:count]]
        entry = (answer.get("logprobs") or [None])[0]
        if not entry:
            raise ProgramError("ollama не вернула logprobs[0]")
        expected: dict[int, float] = {}
        unknown = 0
        for item in entry.get("top_logprobs") or [entry]:
            raw = bytes(int(byte) for byte in (item.get("bytes") or []))
            token_id = mapping.get(raw)
            if token_id is None:
                unknown += 1
                continue
            expected[token_id] = float(item["logprob"])
        if not expected:
            raise ProgramError("ни один logprob ollama не сопоставился с vocab")
        # parity_real печатает команды компиляции/прогона в stdout — при
        # --json они сломали бы JSON. Уводим в буфер: пользователю они не
        # нужны, а при ошибке текста хватает в исключении.
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            probe = parity_real.run_c_probe(work, Path(model), ids, cc)
        logp = _log_softmax([float(value) for value in probe.logits])
        deltas = {token_id: logp[token_id] - value
                  for token_id, value in expected.items()}
        worst = max(deltas, key=lambda key: abs(deltas[key]))
        # argmax VTM: сравним с лучшим токеном ollama (тот же, если паритет).
        best_vtm = max(range(len(logp)), key=lambda index: logp[index])
        best_ollama = max(expected, key=lambda key: expected[key])
        results.append({
            "prompt": prompt,
            "tokens": len(ids),
            "ids": ids,
            "compared": len(deltas),
            "unmapped": unknown,
            "max_abs_delta": abs(deltas[worst]),
            "deltas": {int(key): float(value) for key, value in deltas.items()},
            "within_1e3": sum(1 for value in deltas.values()
                              if abs(value) <= 1e-3),
            "within_1e2": sum(1 for value in deltas.values()
                              if abs(value) <= 1e-2),
            "worst_token": worst,
            "worst_delta": deltas[worst],
            "argmax_match": best_vtm == best_ollama,
            "vtm_argmax": best_vtm,
            "ollama_argmax": best_ollama,
            "ollama_logprob": expected[best_ollama],
            "vtm_logprob": logp[best_ollama],
            "ok": abs(deltas[worst]) <= tolerance,
        })
    return {
        "model": str(model),
        "ollama": ollama_url,
        "tolerance": tolerance,
        "prompts": results,
        "ok": all(item["ok"] and item["argmax_match"] for item in results),
    }


def print_parity(result: dict[str, Any]) -> None:
    for item in result["prompts"]:
        status = "OK" if item["ok"] else "ПРЕВЫШЕНИЕ"
        print(
            f"{status}  {item['prompt']!r}: токенов {item['tokens']}, "
            f"сверено {item['compared']}/{item['compared'] + item['unmapped']} "
            f"logprob, max|Δ| = {item['max_abs_delta']:.3e} "
            f"(токен {item['worst_token']}: {item['worst_delta']:+.3e}), "
            f"argmax {'совпал' if item['argmax_match'] else 'РАЗОШЁЛСЯ'} "
            f"({item['vtm_argmax']} против {item['ollama_argmax']}, "
            f"logprob {item['ollama_logprob']:.4f} vs {item['vtm_logprob']:.4f})"
        )
    print("итог: " + ("паритет в пределах порога" if result["ok"]
                      else "порог превышен"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moe_program.py",
        description="Проверка и замер MoE-программы VTM",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("check", help="проверить MoE-инструкции контейнера")
    one.add_argument("model")
    one.add_argument("--json", action="store_true")
    two = sub.add_parser("measure", help="прогнать и замерить ток/с, RSS, faults")
    two.add_argument("model")
    two.add_argument("--tokens", default="1,2,3")
    two.add_argument("--max-new", type=int, default=8)
    two.add_argument("--env", action="append", default=[],
                     metavar="K=V", help="переменная окружения прогона")
    two.add_argument("--json", action="store_true")
    three = sub.add_parser("parity", help="паритет логитов с ollama по logprobs")
    three.add_argument("model")
    three.add_argument("--ollama", default="http://127.0.0.1:11434")
    three.add_argument("--ollama-model", required=True)
    three.add_argument("--vocab", required=True,
                       help="словарь GGUF в JSON (moe_slice.py --vocab-out)")
    three.add_argument("--prompt", action="append", default=[])
    three.add_argument("--top-logprobs", type=int, default=10)
    three.add_argument("--tolerance", type=float, default=1e-3)
    three.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "parity":
        prompts = args.prompt or [
            "The capital of France is",
            "In 2024 the population of Tokyo was about",
            "def fibonacci(n):",
        ]
        result = parity(
            args.model, ollama_url=args.ollama,
            ollama_model=args.ollama_model, vocab_path=args.vocab,
            prompts=prompts, top_logprobs=args.top_logprobs,
            tolerance=args.tolerance,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_parity(result)
        return 0 if result["ok"] else 1
    if args.command == "check":
        summary = check(args.model)
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            print_check(summary)
        return 0
    env: dict[str, str] = {}
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    tokens = [int(part) for part in args.tokens.replace(" ", "").split(",")
              if part]
    result = measure(args.model, tokens, args.max_new, env=env)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_measure(result)
    return 0 if result["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
