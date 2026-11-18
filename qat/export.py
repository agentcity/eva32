#!/usr/bin/env python3
"""Экспорт QAT-чекпоинта в тернарный артефакт.

    python3 -m qat.export --run runs/pilot1 --out export/

Пишет четыре файла:

* ``ternary.pt`` — int8-триты и масштабы по тензорам (основной артефакт);
* ``ternary.trit`` — те же веса, упакованные по 20 тритов в 32-битное слово
  (формат docs/trit-format.md), сегментами подряд;
* ``non_ternary.bin`` — эмбеддинги, финальная норма, lm_head и bias в fp32:
  без них ``tools/vtm_compile.py`` не соберёт полный граф. При
  ``tie_word_embeddings`` lm_head не дублируется, а ссылается на таблицу
  эмбеддингов;
* ``manifest.json`` — ревизия базовой модели, конфиг, шаг, контрольные суммы,
  архитектура для VTM, таблицы смещений тензоров в ``.trit`` и
  ``non_ternary.bin``.

Экспорт не требует transformers: он читает state_dict чекпоинта и квантует
мастер-веса заново тем же кодом, что и обучение (qat.ternary).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import ternary
from .config import PROJECT_ROOT
from .util import human_bytes, human_count, sha256_file, utc_now, write_json_atomic

PARAM_SUFFIX = ".parametrizations.weight.original"
REQUIRED_MODULES = (("torch", "torch"), ("numpy", "numpy"))
#: Нетернарные веса (эмбеддинги, нормы, lm_head, bias) лежат отдельным
#: файлом, чтобы компилятор VTM читал их без torch.
NON_TERNARY_FILE = "non_ternary.bin"
#: 2: добавлены architecture и non_ternary; v1 неполный и компилятором отвергается.
MANIFEST_VERSION = 2

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")
_MATRIX_KEYS = {
    "self_attn.q_proj": "q_proj", "attn.q_proj": "q_proj",
    "attention.q_proj": "q_proj", "q_proj": "q_proj",
    "self_attn.k_proj": "k_proj", "attn.k_proj": "k_proj",
    "attention.k_proj": "k_proj", "k_proj": "k_proj",
    "self_attn.v_proj": "v_proj", "attn.v_proj": "v_proj",
    "attention.v_proj": "v_proj", "v_proj": "v_proj",
    "self_attn.o_proj": "o_proj", "attn.o_proj": "o_proj",
    "attention.o_proj": "o_proj", "o_proj": "o_proj",
    "mlp.gate_proj": "gate_proj", "ffn.gate_proj": "gate_proj",
    "feed_forward.gate_proj": "gate_proj", "gate_proj": "gate_proj",
    "mlp.up_proj": "up_proj", "ffn.up_proj": "up_proj",
    "feed_forward.up_proj": "up_proj", "up_proj": "up_proj",
    "mlp.down_proj": "down_proj", "ffn.down_proj": "down_proj",
    "feed_forward.down_proj": "down_proj", "down_proj": "down_proj",
}
_NORM_KEYS = {
    "input_layernorm": "norm1", "ln1": "norm1", "norm1": "norm1",
    "attn_norm": "norm1",
    "post_attention_layernorm": "norm2", "ln2": "norm2", "norm2": "norm2",
    "ffn_norm": "norm2",
}
_KEY_PRIORITY = {
    "embed": 0, "final_norm": 1, "lm_head": 2,
    "norm": 3, "matrix": 4, "bias": 5,
}


@dataclass(frozen=True)
class WeightRole:
    """Роль имени тензора в state_dict HF/Qwen."""

    kind: str
    layer: int | None
    key: str


def classify_state_key(name: str) -> WeightRole:
    """Разобрать имя тензора: embed | final_norm | lm_head | matrix | bias |
    norm | unknown, номер слоя и короткий ключ (`q_proj`, `q_bias`, `norm1`).
    """
    base = name
    is_bias = name.endswith(".bias")
    if is_bias:
        base = name[:-len(".bias")]
    elif name.endswith(".weight"):
        base = name[:-len(".weight")]
    if base in ("model.embed_tokens", "embed_tokens", "embed", "emb"):
        return WeightRole("embed", None, "embed")
    if base in ("model.norm", "transformer.ln_f", "final_norm", "ln_f", "norm"):
        return WeightRole("final_norm", None, "final_norm")
    if base == "lm_head":
        return WeightRole("lm_head", None, "lm_head")
    match = _LAYER_RE.search(name)
    if not match:
        return WeightRole("unknown", None, base)
    layer = int(match.group(1))
    tail = name[match.end():]
    if tail.endswith(".weight"):
        tail = tail[:-len(".weight")]
    elif tail.endswith(".bias"):
        tail = tail[:-len(".bias")]
    if tail in _NORM_KEYS:
        return WeightRole("norm" if not is_bias else "unknown", layer, _NORM_KEYS[tail])
    key = _MATRIX_KEYS.get(tail)
    if key is None:
        return WeightRole("unknown", layer, tail)
    if is_bias:
        return WeightRole("bias", layer, key[:-len("_proj")] + "_bias")
    return WeightRole("matrix", layer, key)


def _config_value(config: dict[str, Any] | None, *names: str) -> Any:
    if not config:
        return None
    for name in names:
        value = config.get(name)
        if value is not None:
            return value
    return None


def infer_architecture(
    state_shapes: dict[str, Sequence[int]],
    model_config: dict[str, Any] | None = None,
    *,
    tie_hint: bool | None = None,
    overrides: dict[str, Any] | None = None,
    lm_head_trit: bool = False,
) -> dict[str, Any]:
    """Вывести архитектуру для VTM из форм state_dict и HF-конфига.

    Формы дают hidden/vocab/intermediate/q_dim/kv_dim и число слоёв, но не
    делят q_dim на головы. Если `model_config` (HF) не прочитан, `n_heads` и
    `head_dim` обязательны в `overrides` — иначе ошибка с подсказкой.
    """
    overrides = dict(overrides or {})
    shapes = {
        str(name): tuple(int(x) for x in shape)
        for name, shape in state_shapes.items()
        if shape
    }
    layers: dict[int, dict[str, tuple[int, ...]]] = {}
    embed_shape: tuple[int, ...] | None = None
    final_norm_shape: tuple[int, ...] | None = None
    lm_head_shape: tuple[int, ...] | None = None
    for name, shape in shapes.items():
        role = classify_state_key(name)
        if role.kind == "embed":
            embed_shape = shape
        elif role.kind == "final_norm":
            final_norm_shape = shape
        elif role.kind == "lm_head":
            lm_head_shape = shape
        elif role.layer is not None:
            layers.setdefault(role.layer, {})[role.key] = shape

    if not layers:
        raise RuntimeError("в state_dict нет слоёв model.layers.N — это не декодер?")
    n_layers = max(layers) + 1
    if set(layers) != set(range(n_layers)):
        raise RuntimeError(
            f"слои идут не подряд: есть {sorted(layers)}, ожидалось 0..{n_layers - 1}"
        )
    first = layers[0]
    for key in ("q_proj", "k_proj", "v_proj", "gate_proj", "norm1", "norm2"):
        if key not in first:
            raise RuntimeError(f"в слое 0 нет {key}: неподдерживаемая архитектура")
    for index, layer in layers.items():
        for key, shape in layer.items():
            if key in first and shape != first[key]:
                raise RuntimeError(
                    f"слой {index}: {key} формы {shape}, а в слое 0 {first[key]}"
                )

    hidden = int(first["norm1"][0])
    q_dim = int(first["q_proj"][0])
    kv_dim = int(first["k_proj"][0])
    intermediate = int(first["gate_proj"][0])
    if final_norm_shape is not None and int(final_norm_shape[0]) != hidden:
        raise RuntimeError(
            f"model.norm: размер {final_norm_shape[0]}, а hidden={hidden}"
        )
    if embed_shape is not None and int(embed_shape[-1]) != hidden:
        raise RuntimeError(
            f"embed_tokens: последняя размерность {embed_shape[-1]}, "
            f"а hidden={hidden}"
        )
    if embed_shape is not None:
        vocab = int(embed_shape[0])
    else:
        vocab = int(_config_value(model_config, "vocab_size", "vocab") or 0)
    if not vocab:
        raise RuntimeError("не удалось определить vocab: нет model.embed_tokens.weight")

    config = dict(model_config or {})
    cfg_hidden = _config_value(config, "hidden_size", "d_model", "hidden")
    if cfg_hidden is not None and int(cfg_hidden) != hidden:
        raise RuntimeError(
            f"HF-конфиг: hidden_size={cfg_hidden}, а state_dict даёт hidden={hidden}"
        )
    cfg_layers = _config_value(config, "num_hidden_layers", "n_layer", "n_layers")
    if cfg_layers is not None and int(cfg_layers) != n_layers:
        raise RuntimeError(
            f"HF-конфиг: слоёв {cfg_layers}, а state_dict даёт {n_layers}"
        )
    cfg_vocab = _config_value(config, "vocab_size", "vocab")
    if cfg_vocab is not None and int(cfg_vocab) != vocab:
        raise RuntimeError(
            f"HF-конфиг: vocab_size={cfg_vocab}, а state_dict даёт {vocab}"
        )
    cfg_inter = _config_value(config, "intermediate_size", "ffn_dim", "intermediate")
    if cfg_inter is not None and int(cfg_inter) != intermediate:
        raise RuntimeError(
            f"HF-конфиг: intermediate_size={cfg_inter}, "
            f"а state_dict даёт {intermediate}"
        )

    n_heads = overrides.get("n_heads")
    if n_heads is None:
        n_heads = _config_value(config, "num_attention_heads", "n_head", "n_heads")
    head_dim = overrides.get("head_dim")
    if head_dim is None:
        head_dim = _config_value(config, "head_dim")
    if head_dim is None and n_heads:
        if q_dim % int(n_heads):
            raise RuntimeError(f"q_dim={q_dim} не делится на n_heads={n_heads}")
        head_dim = q_dim // int(n_heads)
    if n_heads is None and head_dim:
        if q_dim % int(head_dim):
            raise RuntimeError(f"q_dim={q_dim} не делится на head_dim={head_dim}")
        n_heads = q_dim // int(head_dim)
    if n_heads is None or head_dim is None:
        raise RuntimeError(
            "не удалось определить n_heads/head_dim: нет HF-конфига "
            "(transformers или architecture в run.json). Задайте "
            "--n-heads и --head-dim при экспорте"
        )
    n_heads, head_dim = int(n_heads), int(head_dim)
    if n_heads * head_dim != q_dim:
        raise RuntimeError(
            f"n_heads×head_dim={n_heads * head_dim}, а q_proj даёт q_dim={q_dim}"
        )
    n_kv_heads = overrides.get("n_kv_heads")
    if n_kv_heads is None:
        n_kv_heads = _config_value(config, "num_key_value_heads", "n_head_kv", "n_kv_heads")
    if n_kv_heads is None:
        if kv_dim % head_dim:
            raise RuntimeError(
                f"kv_dim={kv_dim} не делится на head_dim={head_dim}; "
                "задайте --n-kv-heads"
            )
        n_kv_heads = kv_dim // head_dim
    n_kv_heads = int(n_kv_heads)
    if n_kv_heads * head_dim != kv_dim:
        raise RuntimeError(
            f"n_kv_heads×head_dim={n_kv_heads * head_dim}, а k_proj даёт kv_dim={kv_dim}"
        )
    if n_heads % n_kv_heads:
        raise RuntimeError(f"n_heads={n_heads} не делится на n_kv_heads={n_kv_heads}")

    tie_override = overrides.get("tie_word_embeddings")
    if tie_override is None:
        tie_override = _config_value(config, "tie_word_embeddings", "tied_embeddings")
    if tie_override is None:
        if tie_hint is not None:
            tie = bool(tie_hint)
        else:
            tie = lm_head_shape is None
    else:
        tie = bool(tie_override)
    if tie and lm_head_trit:
        raise RuntimeError(
            "lm_head попал в тернарные слои, а tie_word_embeddings=true: "
            "tied-таблица хранится как fp32-эмбеддинг. Исключите lm_head из "
            "quant.target_suffixes или выставьте --tie-embeddings no"
        )

    eps = overrides.get("rms_norm_eps")
    if eps is None:
        eps = _config_value(config, "rms_norm_eps", "layer_norm_eps", "eps")
    rope_theta = overrides.get("rope_theta")
    if rope_theta is None:
        rope_theta = _config_value(config, "rope_theta")
    max_context = _config_value(
        config, "max_position_embeddings", "n_positions", "max_context"
    )
    context_len = overrides.get("context_len")
    if context_len is None:
        context_len = _config_value(config, "context_len")
    if context_len is None:
        # VTM v0.1 держит KV-кэш в контейнере: 32768 токенов на 28 слоёв —
        # это ~1.9 ГиБ нулей. Рабочая длина по умолчанию 2048; больше — через
        # --context-len осознанно.
        context_len = min(int(max_context), 2048) if max_context else 2048
    context_len = int(context_len)

    return {
        "hidden": hidden,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "intermediate": intermediate,
        "vocab": vocab,
        "n_layers": n_layers,
        "context_len": context_len,
        "max_context": int(max_context) if max_context else None,
        "rope_theta": float(rope_theta or 10000.0),
        "rms_norm_eps": float(eps if eps is not None else 1e-6),
        "tie_word_embeddings": tie,
        "lm_head": "tmv" if lm_head_trit else "mm",
    }


def _state_tie_hint(state_dict: dict[str, Any]) -> bool | None:
    """Один ли тензор у embed_tokens и lm_head (Qwen tie по умолчанию)."""
    embed = state_dict.get("model.embed_tokens.weight")
    head = state_dict.get("lm_head.weight")
    if embed is None or head is None:
        return None
    if not (hasattr(embed, "data_ptr") and hasattr(head, "data_ptr")):
        return None
    return int(embed.data_ptr()) == int(head.data_ptr())


def _load_hf_config(model_cfg: dict[str, Any]) -> dict[str, Any] | None:
    """HF-конфиг без жёсткой зависимости: он нужен ради n_heads/rope/eps."""
    name = model_cfg.get("name_or_path")
    if not name:
        return None
    try:
        from transformers import AutoConfig  # noqa: PLC0415 - опционален

        config = AutoConfig.from_pretrained(
            str(name),
            revision=model_cfg.get("revision") or None,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        return config.to_dict()
    except Exception as exc:  # noqa: BLE001 - сеть/кэш/версии падают по-разному
        print(f"[qat.export] ВНИМАНИЕ: HF-конфиг {name} не прочитан ({exc}); "
              f"n_heads/head_dim/rope_theta/eps берутся из --* или state_dict")
        return None


def _write_non_ternary(
    out_dir: Path,
    state_dict: dict[str, Any],
    architecture: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]], int]:
    """Выгрузить эмбеддинги, нормы, lm_head и bias в `<out>/non_ternary.bin`.

    Дедупликация по `data_ptr`: при tie_word_embeddings lm_head и
    embed_tokens — один тензор, в файле он лежит один раз и помечается
    `tied_to`. Offsets/длины пишутся в манифест, чтобы компилятор VTM читал
    файл без torch.
    """
    records: list[tuple[str, Any, WeightRole, int]] = []
    for key in sorted(state_dict):
        value = state_dict[key]
        if not hasattr(value, "numel"):
            continue
        if key.endswith(PARAM_SUFFIX) or ".parametrizations." in key:
            continue
        role = classify_state_key(key)
        if role.kind == "unknown":
            raise RuntimeError(
                f"{key}: компилятор VTM не знает этот тензор; поддержаны "
                "model.embed_tokens/model.norm/lm_head и Qwen/Llama-слои"
            )
        records.append((key, value, role, int(value.data_ptr())))

    groups: dict[int, list[tuple[str, Any, WeightRole, int]]] = {}
    for record in records:
        groups.setdefault(record[3], []).append(record)

    entries: list[dict[str, Any]] = []
    total_params = 0
    tie = bool(architecture.get("tie_word_embeddings"))
    tmp_path = out_dir / ("." + NON_TERNARY_FILE + ".tmp")
    final_path = out_dir / NON_TERNARY_FILE
    with tmp_path.open("wb") as handle:
        offset = 0
        for ptr, group in groups.items():
            group.sort(key=lambda item: (
                _KEY_PRIORITY.get(item[2].kind, 9), item[0],
            ))
            canonical_key, value, role, _ptr = group[0]
            if role.kind == "lm_head" and tie:
                # tied: lm_head — ссылка на embed_tokens, байт не пишем.
                continue
            payload = _to_numpy_bytes(value, "<f4")
            expected = int(value.numel()) * 4
            if len(payload) != expected:
                raise RuntimeError(
                    f"{canonical_key}: {len(payload)} байт, ожидалось {expected}"
                )
            entry = {
                "name": canonical_key,
                "role": role.kind,
                "layer": role.layer,
                "key": role.key,
                "shape": [int(x) for x in value.shape],
                "dtype": "float32",
                "offset": offset,
                "length": len(payload),
                "params": int(value.numel()),
            }
            aliases = [key for key, _v, _r, _p in group[1:] if key != canonical_key]
            if aliases:
                entry["aliases"] = aliases
            handle.write(payload)
            offset += len(payload)
            total_params += int(value.numel())
            entries.append(entry)

    if tie:
        embed = next(
            (entry for entry in entries if entry["role"] == "embed"), None
        )
        if embed is None:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "tie_word_embeddings=true, но model.embed_tokens.weight не найден"
            )
        entries.append({
            "name": "lm_head.weight",
            "role": "lm_head",
            "layer": None,
            "key": "lm_head",
            "shape": list(embed["shape"]),
            "dtype": "float32",
            "tied_to": embed["name"],
            "params": int(embed["params"]),
        })
    elif architecture.get("lm_head") == "tmv":
        # lm_head тернаризован и лежит в ternary.trit, не в non_ternary.bin.
        pass
    else:
        heads = [entry for entry in entries if entry["role"] == "lm_head"]
        if not heads:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "tie_word_embeddings=false, но lm_head.weight не найден"
            )
        if len(heads) > 1:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError("больше одного кандидата в lm_head")
    tmp_path.replace(final_path)
    return final_path, entries, total_params


def _read_run_architecture(run_meta: dict[str, Any]) -> dict[str, Any] | None:
    arch = run_meta.get("architecture")
    return arch if isinstance(arch, dict) else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m qat.export",
        description="Экспорт тернарных весов из QAT-чекпоинта в .pt/.trit + manifest.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  python3 -m qat.export --run runs/pilot1 --out export/\n"
            "  python3 -m qat.export --run runs/pilot1 --step 200 --out export/\n"
            "  python3 -m qat.export --run runs/pilot1/checkpoints/step_000200 --out export/\n\n"
            "тернарный формат описан в docs/trit-format.md; контрольные векторы\n"
            "проверяются тестами: python3 -m unittest discover -s tests -v\n"
        ),
    )
    parser.add_argument("--run", required=True,
                        help="каталог прогона runs/<name> или конкретный checkpoints/step_*")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "export"),
                        help="каталог для ternary.pt, ternary.trit и manifest.json")
    parser.add_argument("--step", type=int, default=None,
                        help="номер шага; по умолчанию последний чекпоинт")
    parser.add_argument("--scale-dtype", choices=("float16", "float32"), default=None,
                        help="тип масштабов (по умолчанию из конфига прогона)")
    parser.add_argument("--force", action="store_true",
                        help="перезаписать существующий экспорт")
    parser.add_argument("--n-heads", type=int, default=None,
                        help="число голов attention, если HF-конфиг недоступен")
    parser.add_argument("--n-kv-heads", type=int, default=None,
                        help="число KV-голов (GQA), если HF-конфиг недоступен")
    parser.add_argument("--head-dim", type=int, default=None,
                        help="размер головы, если HF-конфиг недоступен")
    parser.add_argument("--rope-theta", type=float, default=None,
                        help="theta RoPE (по умолчанию из HF-конфига или 10000)")
    parser.add_argument("--rms-norm-eps", type=float, default=None,
                        help="eps RMSNorm (по умолчанию из HF-конфига или 1e-6)")
    parser.add_argument("--context-len", type=int, default=None,
                        help="длина KV-кэша в контейнере VTM; по умолчанию "
                             "min(max_position_embeddings, 2048): полный "
                             "контекст 32k — это гигабайты нулей")
    parser.add_argument("--tie-embeddings", choices=("auto", "yes", "no"),
                        default="auto",
                        help="tie_word_embeddings: auto — из HF-конфига/state_dict")
    return parser


def missing_modules() -> list[str]:
    missing: list[str] = []
    for module, package in REQUIRED_MODULES:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(package)
    return missing


def print_missing(missing: Sequence[str]) -> None:
    print("[qat.export] не хватает зависимостей: " + ", ".join(missing))
    print()
    print("Экспорт запускается на GPU-машине (нужен только torch/numpy):")
    print("    python3 -m pip install -r qat/requirements-qat.txt")
    print()
    print("Без GPU-стека проверяется только чистый кодек:")
    print("    python3 -m unittest discover -s tests -v")


def _resolve_source(
    source: Path,
    step: int | None,
    checkpoint_mod: Any,
) -> tuple[Path, Path]:
    """Вернуть (каталог прогона, каталог чекпоинта)."""
    source = source.expanduser().resolve()
    if (source / "model.pt").exists():
        ckpt = source
        run_dir = source.parent.parent
        if step is not None and checkpoint_mod.checkpoint_step(source) != step:
            raise RuntimeError(
                f"--step {step}, а указан чекпоинт шага "
                f"{checkpoint_mod.checkpoint_step(source)}"
            )
        return run_dir, ckpt
    if (source / "checkpoints").exists():
        if step is not None:
            ckpt = source / "checkpoints" / f"step_{step:06d}"
            if not ckpt.exists():
                raise RuntimeError(f"чекпоинта шага {step} нет: {ckpt}")
            return source, ckpt
        ckpt = checkpoint_mod.latest_checkpoint(source)
        if ckpt is None:
            raise RuntimeError(f"в {source}/checkpoints нет чекпоинтов")
        return source, ckpt
    raise RuntimeError(
        f"{source} не похож на прогон QAT: нет model.pt или checkpoints/"
    )


def _read_run_meta(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _to_numpy_bytes(tensor: Any, dtype: str) -> bytes:
    """Тензор на CPU -> little-endian байты без зависимости от порядка байт хоста."""
    array = tensor.detach().cpu().numpy()
    return array.astype(dtype, copy=False).tobytes(order="C")


def _check_packing(trits: Any, words: Any, count: int = 160) -> None:
    """Сверить первые слова с чистым Python-кодеком (он же проверен тестами)."""
    flat = trits.reshape(-1)
    take = min(count, int(flat.numel()))
    if take == 0:
        return
    n_words = (take + ternary.TRITS_PER_WORD - 1) // ternary.TRITS_PER_WORD
    blob = _to_numpy_bytes(words[:n_words], "<u4")
    decoded = ternary.unpack20(blob, take)
    expected = [int(x) for x in flat[:take].tolist()]
    if decoded != expected:
        raise RuntimeError("упаковка 20 тритов в слово не сошлась с кодеком")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    missing = missing_modules()
    if missing:
        print_missing(missing)
        return 2

    import numpy  # noqa: F401,PLC0415 - нужен для tensor.numpy(), проверен выше
    import torch  # noqa: PLC0415

    from . import checkpoint as checkpoint_mod  # noqa: PLC0415

    try:
        run_dir, ckpt = _resolve_source(Path(args.run), args.step, checkpoint_mod)
    except RuntimeError as exc:
        print(f"[qat.export] {exc}")
        return 2

    state = checkpoint_mod.load_trainer_state(ckpt)
    run_meta = _read_run_meta(run_dir)
    config = state.get("config") or run_meta.get("config") or {}
    quant_cfg = config.get("quant") or {}
    model_cfg = config.get("model") or {}
    block_size = int(quant_cfg.get("block_size", ternary.BLOCK_SIZE))
    scale_dtype = args.scale_dtype or quant_cfg.get("scale_dtype", "float16")
    if scale_dtype not in ("float16", "float32"):
        print(f"[qat.export] неизвестный --scale-dtype: {scale_dtype}")
        return 2
    step = int(state.get("step", checkpoint_mod.checkpoint_step(ckpt)))
    samples_seen = int(state.get("samples_seen", 0))

    out_dir = Path(args.out).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "ternary.pt"
    trit_path = out_dir / "ternary.trit"
    manifest_path = out_dir / "manifest.json"
    non_ternary_path = out_dir / NON_TERNARY_FILE
    if not args.force and (
        pt_path.exists() or trit_path.exists() or manifest_path.exists()
        or non_ternary_path.exists()
    ):
        print(f"[qat.export] в {out_dir} уже есть экспорт; добавьте --force, чтобы перезаписать")
        return 2

    print(f"[qat.export] чекпоинт: {ckpt}")
    print(f"[qat.export] шаг {step}, обработано последовательностей {samples_seen}")
    print(f"[qat.export] квантование: блок {block_size}, масштабы {scale_dtype}")
    if model_cfg.get("revision") == "main":
        print("[qat.export] ВНИМАНИЕ: в конфиге revision=main; в manifest попадёт "
              "незакреплённая ревизия")

    started = time.time()
    state_dict = checkpoint_mod.load_model_state(ckpt)
    target_keys = sorted(k for k in state_dict if k.endswith(PARAM_SUFFIX))
    if not target_keys:
        print("[qat.export] в чекпоинте нет параметризованных весов — это не QAT-модель?")
        return 2

    expected_modules = {
        item.get("name") for item in (run_meta.get("quant") or {}).get("modules", [])
        if item.get("name")
    }
    found_modules = {key[:-len(PARAM_SUFFIX)] for key in target_keys}
    if expected_modules and expected_modules != found_modules:
        missing_mods = sorted(expected_modules - found_modules)
        extra_mods = sorted(found_modules - expected_modules)
        print("[qat.export] ВНИМАНИЕ: набор тернарных слоёв в чекпоинте и run.json различается")
        if missing_mods:
            print(f"               нет в чекпоинте: {missing_mods[:5]}")
        if extra_mods:
            print(f"               лишние в чекпоинте: {extra_mods[:5]}")

    # --- архитектура и нетернарные веса ------------------------------------
    # Формы state_dict дают hidden/vocab/intermediate и число слоёв; n_heads,
    # head_dim, rope_theta и eps — из HF-конфига (или --* флагов).
    state_shapes = {
        key: tuple(int(x) for x in value.shape)
        for key, value in state_dict.items()
        if hasattr(value, "shape") and len(value.shape) >= 1
    }
    overrides: dict[str, Any] = {}
    for field_name in (
        "n_heads", "n_kv_heads", "head_dim", "rope_theta",
        "rms_norm_eps", "context_len",
    ):
        value = getattr(args, field_name)
        if value is not None:
            overrides[field_name] = value
    if args.tie_embeddings != "auto":
        overrides["tie_word_embeddings"] = args.tie_embeddings == "yes"
    hf_config = _read_run_architecture(run_meta) or _load_hf_config(model_cfg)
    lm_head_trit = any(key.startswith("lm_head.") for key in target_keys)
    architecture = infer_architecture(
        state_shapes,
        hf_config,
        tie_hint=_state_tie_hint(state_dict),
        overrides=overrides,
        lm_head_trit=lm_head_trit,
    )
    print(
        f"[qat.export] архитектура: hidden={architecture['hidden']} "
        f"heads={architecture['n_heads']}/{architecture['n_kv_heads']} "
        f"head_dim={architecture['head_dim']} "
        f"intermediate={architecture['intermediate']} "
        f"vocab={architecture['vocab']} layers={architecture['n_layers']} "
        f"rope_theta={architecture['rope_theta']:g} "
        f"tie={architecture['tie_word_embeddings']} "
        f"lm_head={architecture['lm_head']}"
    )
    non_ternary_path, non_ternary_entries, non_ternary_params = _write_non_ternary(
        out_dir, state_dict, architecture
    )
    print(
        f"[qat.export] нетернарных тензоров: {len(non_ternary_entries)}, "
        f"параметров: {human_count(non_ternary_params)}; "
        f"{non_ternary_path.name}: {human_bytes(non_ternary_path.stat().st_size)}"
    )

    tensors: dict[str, Any] = {}
    manifest_tensors: list[dict[str, Any]] = []
    total_params = 0
    flags = ternary.FLAG_SCALE_F32 if scale_dtype == "float32" else 0
    scale_np_dtype = "<f4" if scale_dtype == "float32" else "<f2"
    torch_scale_dtype = torch.float32 if scale_dtype == "float32" else torch.float16

    tmp_trit = out_dir / ".ternary.trit.tmp"
    with tmp_trit.open("wb") as fh:
        for index, key in enumerate(target_keys, 1):
            name = key[:-len(PARAM_SUFFIX)]
            weight = state_dict.pop(key)
            if weight.ndim < 1:
                raise RuntimeError(f"{name}: неожиданная форма {tuple(weight.shape)}")
            quant = ternary.quantize_torch(weight, block_size)
            trits = quant["trits"].cpu().contiguous()
            scales = quant["scales"].reshape(-1).cpu()
            n_trits = int(trits.numel())
            in_features = int(weight.shape[-1])
            if in_features % block_size != 0:
                raise RuntimeError(
                    f"{name}: входная размерность {in_features} не кратна block_size "
                    f"{block_size}; формат .trit хранит один масштаб на 128 подряд идущих "
                    f"весов и не умеет row-padding. Смените block_size или модель."
                )
            rows = n_trits // in_features
            scale_blocks_per_row = int(quant["n_scale_blocks"])
            total_scale_blocks = int(scales.numel())
            if total_scale_blocks != rows * scale_blocks_per_row:
                raise RuntimeError(
                    f"{name}: масштабов {total_scale_blocks}, ожидалось "
                    f"{rows * scale_blocks_per_row} (строк {rows} × блоков {scale_blocks_per_row})"
                )
            if total_scale_blocks * block_size < n_trits:
                raise RuntimeError(f"{name}: масштабов не хватает на все триты")

            scales_out = scales.to(torch_scale_dtype)
            if not bool(torch.isfinite(scales_out).all()):
                raise RuntimeError(f"{name}: масштаб не помещается в {scale_dtype}")
            scales_bytes = _to_numpy_bytes(scales_out, scale_np_dtype)

            words = ternary.pack20_torch(trits)
            storage_blocks = (n_trits + ternary.TRITS_PER_STORAGE_BLOCK - 1) // ternary.TRITS_PER_STORAGE_BLOCK
            storage_words = storage_blocks * ternary.WORDS_PER_STORAGE_BLOCK
            if int(words.numel()) < storage_words:
                words = torch.cat([
                    words,
                    torch.zeros(storage_words - int(words.numel()), dtype=torch.int64),
                ])
            if int(words.numel()) != storage_words:
                raise RuntimeError(f"{name}: слов {int(words.numel())}, ожидалось {storage_words}")
            if int(words.max().item()) > ternary.MAX_WORD:
                raise RuntimeError(f"{name}: слово вне диапазона 3^20")
            _check_packing(trits, words)
            words_bytes = _to_numpy_bytes(words, "<u4")

            header = struct.pack(
                "<4sBBQI", ternary.MAGIC, ternary.VERSION, flags, n_trits, total_scale_blocks,
            )
            info = ternary.segment_info(header, block_size)
            if info.n_scale_blocks != total_scale_blocks or info.n_words != storage_words:
                raise RuntimeError(f"{name}: заголовок сегмента не сошёлся с раскладкой")
            offset = fh.tell()
            fh.write(header)
            fh.write(scales_bytes)
            fh.write(words_bytes)
            length = fh.tell() - offset
            expected_length = len(header) + len(scales_bytes) + len(words_bytes)
            if length != expected_length:
                raise RuntimeError(f"{name}: длина сегмента {length}, ожидалась {expected_length}")

            shape = [int(x) for x in weight.shape]
            tensors[name] = {
                "trits": trits,
                "scales": scales_out,
                "shape": shape,
                "n_scale_blocks": total_scale_blocks,
                "scale_blocks_per_row": scale_blocks_per_row,
            }
            total_params += int(trits.numel())
            manifest_tensors.append({
                "name": name,
                "shape": shape,
                "params": n_trits,
                "n_trits": n_trits,
                "n_scale_blocks": total_scale_blocks,
                "scale_blocks_per_row": scale_blocks_per_row,
                "n_words": (n_trits + ternary.TRITS_PER_WORD - 1) // ternary.TRITS_PER_WORD,
                "n_storage_blocks": storage_blocks,
                "n_storage_words": storage_words,
                "trit_offset": int(offset),
                "trit_length": int(length),
            })
            if index % 25 == 0 or index == len(target_keys):
                print(f"[qat.export] тензоров: {index}/{len(target_keys)}, "
                      f"параметров: {human_count(total_params)}")

    tmp_trit.replace(trit_path)
    del state_dict  # освобождаем нетернарные веса до torch.save

    payload = {
        "format": "eva32.ternary.pt",
        "format_version": 1,
        "created_at": utc_now(),
        "block_size": block_size,
        "scale_dtype": scale_dtype,
        "packing": {
            "scheme": "trit20",
            "trits_per_word": ternary.TRITS_PER_WORD,
            "words_per_storage_block": ternary.WORDS_PER_STORAGE_BLOCK,
            "storage_block_trits": ternary.TRITS_PER_STORAGE_BLOCK,
            "scale_block": block_size,
            "endianness": "little",
            "reference": "docs/trit-format.md",
        },
        "source": {
            "run": str(run_dir),
            "checkpoint": str(ckpt),
            "step": step,
            "samples_seen": samples_seen,
            "model": model_cfg.get("name_or_path"),
            "revision": model_cfg.get("revision"),
        },
        "tensors": tensors,
    }
    tmp_pt = out_dir / ".ternary.pt.tmp"
    torch.save(payload, tmp_pt)
    tmp_pt.replace(pt_path)
    del payload, tensors

    pt_hash = sha256_file(pt_path)
    trit_hash = sha256_file(trit_path)
    non_ternary_hash = sha256_file(non_ternary_path)
    manifest = {
        "format": "eva32.ternary.manifest",
        "format_version": MANIFEST_VERSION,
        "created_at": utc_now(),
        "training": {
            "run": run_dir.name,
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt),
            "step": step,
            "samples_seen": samples_seen,
            "max_steps": (config.get("train") or {}).get("max_steps"),
            "config": config,
        },
        "base_model": {
            "name_or_path": model_cfg.get("name_or_path"),
            "revision": model_cfg.get("revision"),
        },
        # Всё, что нужно tools/vtm_compile.py, чтобы собрать полный граф без
        # HF/torch: размерности, rope/eps и политика tied-эмбеддингов.
        "architecture": architecture,
        "quantization": {
            "block_size": block_size,
            "scale_dtype": scale_dtype,
            "rule": "scale=absmax(block); trit=clamp(round_half_away(w/scale),-1,1); w=trit*scale",
            "ste": "w + (w_tern - w).detach()",
            "target_suffixes": quant_cfg.get("target_suffixes"),
        },
        "packing": {
            "magic": ternary.MAGIC.decode("ascii"),
            "version": ternary.VERSION,
            "trits_per_word": ternary.TRITS_PER_WORD,
            "words_per_storage_block": ternary.WORDS_PER_STORAGE_BLOCK,
            "storage_block_trits": ternary.TRITS_PER_STORAGE_BLOCK,
            "scale_block": block_size,
            "n_blocks_header": "число групп масштаба (scale_block), а не 160-тритовых блоков",
            "words_padding": "хвост тензора добит нулевыми тритами до 8 слов",
            "endianness": "little",
            "note": (
                "docs/trit-format.md §8 оставлял размер группы масштаба открытым; "
                "здесь он равен block_size из конфига QAT (128). Слова упакованы "
                "по 20 тритов, массив слов выровнен на 32-байтовые блоки по 8 слов. "
                "Сегменты .trit идут подряд в порядке tensors[]."
            ),
        },
        "non_ternary": {
            "stored_in_this_export": True,
            "file": non_ternary_path.name,
            "dtype": "float32",
            "count": len(non_ternary_entries),
            "params": non_ternary_params,
            "source": str(ckpt / "model.pt"),
            "tensors": non_ternary_entries,
            "note": (
                "эмбеддинги, финальная норма, lm_head и bias лежат в "
                "non_ternary.bin (fp32, little-endian), offsets — в tensors[]. "
                "При tie_word_embeddings lm_head не дублируется: запись "
                "ссылается через tied_to на embed_tokens"
            ),
        },
        "files": {
            "ternary.pt": {"bytes": pt_path.stat().st_size, "sha256": pt_hash},
            "ternary.trit": {"bytes": trit_path.stat().st_size, "sha256": trit_hash},
            non_ternary_path.name: {
                "bytes": non_ternary_path.stat().st_size,
                "sha256": non_ternary_hash,
            },
        },
        "totals": {
            "tensors": len(manifest_tensors),
            "params": total_params,
            "bytes_per_param_bits": 32.0 / ternary.TRITS_PER_WORD,
        },
        "tensors": manifest_tensors,
    }
    write_json_atomic(manifest_path, manifest)

    elapsed = time.time() - started
    print(f"[qat.export] тензоров: {len(manifest_tensors)}, "
          f"тернарных параметров: {human_count(total_params)}")
    print(f"[qat.export] {pt_path.name}: {human_bytes(pt_path.stat().st_size)}  "
          f"sha256 {pt_hash[:16]}…")
    print(f"[qat.export] {trit_path.name}: {human_bytes(trit_path.stat().st_size)}  "
          f"sha256 {trit_hash[:16]}…")
    print(f"[qat.export] {non_ternary_path.name}: "
          f"{human_bytes(non_ternary_path.stat().st_size)}  "
          f"sha256 {non_ternary_hash[:16]}…")
    print(f"[qat.export] {manifest_path.name}: {len(manifest_tensors)} записей; "
          f"tie={architecture['tie_word_embeddings']}")
    print(f"[qat.export] готово за {elapsed:.0f} с -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
