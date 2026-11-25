#!/usr/bin/env python3
"""Импорт MoE GGUF (Qwen3-30B-A3B, Qwen2-MoE) -> манифест и веса VTM.

Зачем отдельный импортёр
========================

`tools/gguf_import.py` (0153/0165) знает Qwen2/Llama-раскладку: плотный FFN
(`ffn_gate/up/down`), три матрицы на слой. У MoE FFN другой:

* `ffn_gate_inp` — **роутер** `[n_experts, hidden]`, fp32, на него завязан
  выбор экспертов;
* `ffn_{gate,up,down}_exps` — **банки экспертов**: один тензор на вес, внутри
  `n_experts` матриц подряд (GGUF dims `[hidden, inter, n_experts]`);
* `ffn_{gate,up,down}_shexp` — общий эксперт (у Qwen2-MoE; у Qwen3-30B-A3B
  его нет — `expert_shared_count = 0`);
* `attn_q_norm`/`attn_k_norm` — QK-norm Qwen3, по вектору на `head_dim`.

Поэтому импортёр свой: он переиспользует разбор GGUF и запись сегментов из
`gguf_import`, но план и раскладка — MoE. `gguf_import.py` не меняется, его
тесты остаются в силе.

Что пишется
===========

* `q4.bin` — нативные сегменты kind 6/7 (`Q4K `/`Q6K `, блоки GGML байт-в-байт,
  ADR 0018). Сюда идут матрицы attention, банки экспертов, таблица словаря
  (`token_embd`, обычно Q4_K) и `output` (lm_head, обычно Q6_K). rel-RMS к
  весам GGUF = 0 по построению: блоки копируются, а не переквантуются.
* `non_ternary.bin` — fp32: нормы, QK-нормы и **роутер**. Роутер — fp32
  намеренно: 1 МиБ на слой, а ошибка в логитах роутера меняет top-k, то есть
  какие эксперты вообще считаются.
* `manifest.json` — манифест v2 с `architecture.moe = true`, `n_experts`,
  `top_k`, `expert_intermediate`, `qk_norm`.

Раскладка банка в контейнере
============================

Банк — тензор kind 6/7 с dims `[n_experts, rows, cols]`: эксперт `e` занимает
`rows·cols/256` блоков с начала банка. `rows·cols` обязано делиться на 256
(иначе эксперт разрезал бы блок GGML) — у Qwen3 768·2048 = 6 144 блока ✓.
Для kind 6/7 это не новое ограничение: `rows` и `cols` у K-квантов кратны 256
всегда, а импортёр проверяет это до записи.

CLI
===

    python3 tools/moe_import.py --gguf /tmp/moe/slice.gguf --out /tmp/moe/exp
    python3 tools/moe_import.py --gguf m.gguf --out exp --stats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gguf_import as G  # noqa: E402  (переиспользуем разбор GGUF и запись)

MANIFEST_FORMAT = G.MANIFEST_FORMAT
MANIFEST_VERSION = G.MANIFEST_VERSION
Q4_FILE = "q4.bin"
NON_TERNARY_FILE = "non_ternary.bin"

#: GGUF-суффиксы MoE-слоя -> HF-имена, которые понимает `vtm_compile`.
#: Значение — (HF-суффикс, роль, ключ компилятора).
MOE_LAYER_TENSORS: dict[str, tuple[str, str, str]] = {
    "attn_norm.weight": ("input_layernorm.weight", "norm", "norm1"),
    "ffn_norm.weight": ("post_attention_layernorm.weight", "norm", "norm2"),
    "attn_q.weight": ("self_attn.q_proj.weight", "matrix", "q_proj"),
    "attn_k.weight": ("self_attn.k_proj.weight", "matrix", "k_proj"),
    "attn_v.weight": ("self_attn.v_proj.weight", "matrix", "v_proj"),
    "attn_output.weight": ("self_attn.o_proj.weight", "matrix", "o_proj"),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", "norm", "q_norm"),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", "norm", "k_norm"),
    "ffn_gate_inp.weight": ("mlp.gate.weight", "router", "router"),
    "ffn_gate_exps.weight": ("mlp.experts.gate_proj.weight", "expert_gate",
                             "expert_gate"),
    "ffn_up_exps.weight": ("mlp.experts.up_proj.weight", "expert_up",
                           "expert_up"),
    "ffn_down_exps.weight": ("mlp.experts.down_proj.weight", "expert_down",
                             "expert_down"),
    "ffn_gate_shexp.weight": ("mlp.shared_expert.gate_proj.weight",
                              "shared_gate", "shared_gate"),
    "ffn_up_shexp.weight": ("mlp.shared_expert.up_proj.weight", "shared_up",
                            "shared_up"),
    "ffn_down_shexp.weight": ("mlp.shared_expert.down_proj.weight",
                              "shared_down", "shared_down"),
}

#: Архитектуры, которые этот импортёр понимает. `qwen3moe` — Qwen3-30B-A3B
#: (задача 0172), `qwen2moe` — Qwen2-MoE (общий эксперт). Прочие MoE
#: (`deepseek2`, `mixtral` с другим attention) отвергаются: у них другая
#: механика, и «похожий» импорт дал бы правдоподобно неверные логиты.
MOE_ARCHITECTURES = ("qwen3moe", "qwen2moe")

#: Роли, которые пишутся нативными блоками kind 6/7.
NATIVE_ROLES = ("matrix", "expert_gate", "expert_up", "expert_down",
                "shared_gate", "shared_up", "shared_down", "lm_head", "embed")
#: Роли, которые пишутся fp32 в non_ternary.bin.
FP32_ROLES = ("norm", "router", "final_norm")


class MoeImportError(ValueError):
    """GGUF не MoE или не та архитектура — текст для человека."""


@dataclass
class MoeConfig:
    architecture: str
    name: str
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int          # GGUF feed_forward_length (у MoE не используется)
    expert_inter: int
    shared_inter: int
    n_experts: int
    top_k: int
    vocab: int
    n_layers: int
    context_len: int
    rope_theta: float
    eps: float
    qk_norm: bool
    tie_word_embeddings: bool

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def as_architecture(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            # intermediate у MoE не используется (плотного FFN нет), но
            # LayerConfig требует > 0 — кладём dense-размер из GGUF.
            "intermediate": self.intermediate,
            "vocab": self.vocab,
            "n_layers": self.n_layers,
            "context_len": self.context_len,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.eps,
            "rope_layout": "half",
            "tie_word_embeddings": self.tie_word_embeddings,
            # lm_head — нативная Q4_K/Q6_K-матрица: TMV по kind 6/7.
            "lm_head": "q4",
            # Таблица словаря — kind 6/7 (Q4_K); компайлер берёт вид из
            # тензора, а `embed` в манифесте остаётся "none" (это не
            # триты/int8) — meta контейнера скажет q4_k.
            "embed": "none",
            "moe": True,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "expert_intermediate": self.expert_inter,
            "qk_norm": self.qk_norm,
            "shared_expert_intermediate": self.shared_inter,
        }


def _meta_int(meta: dict[str, Any], key: str, default: int | None = None) -> int:
    if key not in meta or meta[key] is None:
        if default is None:
            raise MoeImportError(f"в метаданных GGUF нет {key!r}")
        return int(default)
    value = meta[key]
    if isinstance(value, (list, tuple)):
        if not value:
            raise MoeImportError(f"метаданные {key}: пустой массив")
        first = int(value[0])
        if any(int(item) != first for item in value):
            raise MoeImportError(
                f"метаданные {key}: значения по слоям разные — поэлементные "
                "head_count/expert_count не поддержаны"
            )
        return first
    return int(value)


def _meta_float(meta: dict[str, Any], key: str, default: float) -> float:
    value = meta.get(key)
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple)):
        return float(_meta_int(meta, key))
    return float(value)


def map_moe_tensor(name: str) -> tuple[str, str, int | None, str] | None:
    """GGUF-имя MoE -> (HF-имя, роль, слой, ключ) или None (не наш тензор)."""
    global_entry = G.GLOBAL_TENSORS.get(name)
    if global_entry is not None:
        hf_name, role, key = global_entry
        return hf_name, role, None, key
    layer, tail = G._split_layer_name(name)
    if layer is None:
        return None
    entry = MOE_LAYER_TENSORS.get(tail)
    if entry is None:
        return None
    hf_tail, role, key = entry
    return f"model.layers.{layer}.{hf_tail}", role, layer, key


def infer_moe_config(
    gguf: G.GgufFile,
    *,
    limit_layers: int | None = None,
    limit_vocab: int | None = None,
) -> MoeConfig:
    """Собрать конфиг MoE из метаданных GGUF и проверить целостность."""
    arch = gguf.metadata.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise MoeImportError("в метаданных нет general.architecture")
    if arch not in MOE_ARCHITECTURES:
        raise MoeImportError(
            f"архитектура {arch!r} не поддержана импортёром MoE "
            f"(поддержаны: {', '.join(MOE_ARCHITECTURES)}). Для плотных "
            "Qwen2/Llama есть tools/gguf_import.py"
        )
    prefix = f"{arch}."
    name = str(gguf.metadata.get("general.name") or gguf.path.name)

    rope_scaling = gguf.metadata.get(f"{prefix}rope.scaling.type")
    if rope_scaling not in (None, "", "none"):
        raise MoeImportError(
            f"rope.scaling.type={rope_scaling!r}: масштабирование RoPE в VTM "
            "не реализовано — импорт остановлен"
        )

    n_layers_source = _meta_int(gguf.metadata, prefix + "block_count")
    hidden = _meta_int(gguf.metadata, prefix + "embedding_length")
    n_heads = _meta_int(gguf.metadata, prefix + "attention.head_count")
    n_kv_heads = _meta_int(gguf.metadata, prefix + "attention.head_count_kv",
                           default=n_heads)
    head_dim = _meta_int(gguf.metadata, prefix + "attention.key_length",
                         default=hidden // n_heads if n_heads else 0)
    intermediate = _meta_int(gguf.metadata, prefix + "feed_forward_length",
                             default=0)
    expert_inter = _meta_int(gguf.metadata, prefix + "expert_feed_forward_length")
    n_experts = _meta_int(gguf.metadata, prefix + "expert_count")
    top_k = _meta_int(gguf.metadata, prefix + "expert_used_count")
    shared_inter = _meta_int(
        gguf.metadata, prefix + "expert_shared_feed_forward_length", default=0
    )
    context_len = _meta_int(gguf.metadata, prefix + "context_length",
                            default=2048)
    rope_theta = _meta_float(gguf.metadata, prefix + "rope.freq_base", 10000.0)
    eps = _meta_float(gguf.metadata,
                      prefix + "attention.layer_norm_rms_epsilon", 1e-6)

    for label, value in (
        ("block_count", n_layers_source), ("embedding_length", hidden),
        ("attention.head_count", n_heads),
        ("attention.head_count_kv", n_kv_heads), ("head_dim", head_dim),
        ("expert_feed_forward_length", expert_inter),
        ("expert_count", n_experts), ("expert_used_count", top_k),
        ("context_length", context_len),
    ):
        if value <= 0:
            raise MoeImportError(f"конфиг: {label}={value} — некорректно")
    if not 0 < top_k <= n_experts:
        raise MoeImportError(
            f"конфиг: expert_used_count={top_k} вне 1..expert_count={n_experts}"
        )
    # У Qwen2.5 hidden == n_heads·head_dim, у Qwen3 — нет: q_proj расширяет
    # hidden (2048) до q_dim = 32·128 = 4096, o_proj сжимает обратно. Поэтому
    # проверяем не hidden, а что q_dim выводится из голов и согласован с
    # тензорами (это проверит plan_moe по формам q_proj/o_proj).
    q_dim = n_heads * head_dim
    if n_heads != q_dim // head_dim:  # pragma: no cover - тавтология, см. ниже
        raise MoeImportError("конфиг: q_dim не делится на head_dim")
    if n_heads % n_kv_heads:
        raise MoeImportError(
            f"конфиг: n_heads={n_heads} не делятся на n_kv_heads={n_kv_heads}"
        )
    if expert_inter % 256 or hidden % 256:
        raise MoeImportError(
            f"конфиг: expert_inter={expert_inter} и hidden={hidden} должны "
            "быть кратны 256 — иначе эксперт не ляжет целыми блоками Q4_K"
        )
    if intermediate <= 0:
        intermediate = expert_inter

    # vocab: из token_embd (ne1) или из метаданных.
    vocab = None
    meta_vocab = gguf.metadata.get(prefix + "vocab_size")
    if meta_vocab is not None:
        vocab = _meta_int(gguf.metadata, prefix + "vocab_size")
    if gguf.has_tensor("token_embd.weight"):
        embed = gguf.tensor("token_embd.weight")
        if len(embed.dims) != 2 or int(embed.dims[0]) != hidden:
            raise MoeImportError(
                f"token_embd.weight: dims={embed.dims}, ожидалось "
                f"[{hidden}, vocab]"
            )
        if vocab is None:
            vocab = int(embed.dims[1])
        elif vocab != int(embed.dims[1]):
            raise MoeImportError(
                f"vocab_size={vocab} не совпал с token_embd ne1={embed.dims[1]}"
            )
    if vocab is None:
        vocab = gguf.array_lengths.get("tokenizer.ggml.tokens")
    if not vocab:
        raise MoeImportError("не удалось определить vocab")

    if limit_layers is not None:
        if not 0 < limit_layers <= n_layers_source:
            raise MoeImportError(
                f"--limit-layers={limit_layers} вне 1..{n_layers_source}"
            )
        n_layers = limit_layers
    else:
        n_layers = n_layers_source
    out_vocab = vocab
    if limit_vocab is not None:
        if not 0 < limit_vocab <= vocab:
            raise MoeImportError(
                f"--limit-vocab={limit_vocab} вне 1..{vocab}"
            )
        out_vocab = limit_vocab

    qk_norm = gguf.has_tensor("blk.0.attn_q_norm.weight")
    if gguf.has_tensor("blk.0.attn_k_norm.weight") != qk_norm:
        raise MoeImportError(
            "есть только одна из attn_q_norm/attn_k_norm — QK-norm "
            "неполный; такой контейнер дал бы правдоподобно неверные логиты"
        )
    tie = not gguf.has_tensor("output.weight")
    if tie:
        raise MoeImportError(
            "output.weight нет (tie_word_embeddings): MoE-импорт пока ждёт "
            "отдельный lm_head; для tied-таблицы нужен путь EMB+MM, которого "
            "у Qwen3-30B-A3B нет"
        )
    return MoeConfig(
        architecture=arch, name=name, hidden=hidden, n_heads=n_heads,
        n_kv_heads=n_kv_heads, head_dim=head_dim, intermediate=intermediate,
        expert_inter=expert_inter, shared_inter=shared_inter,
        n_experts=n_experts, top_k=top_k, vocab=out_vocab,
        n_layers=n_layers, context_len=context_len, rope_theta=rope_theta,
        eps=eps, qk_norm=qk_norm, tie_word_embeddings=False,
    )


@dataclass
class Planned:
    info: G.TensorInfo
    name: str          # HF-имя (его понимает vtm_compile)
    role: str
    layer: int | None
    key: str
    shape: tuple[int, ...]
    target: str        # q4k | q6k | f32


def plan_moe(gguf: G.GgufFile, cfg: MoeConfig) -> list[Planned]:
    """Проверить имена/формы всех тензоров и решить, что куда писать."""
    plans: list[Planned] = []
    unknown: list[str] = []
    seen: set[tuple[str, int | None, str]] = set()
    layer_keys: dict[int, set[str]] = {index: set() for index in range(cfg.n_layers)}

    for info in gguf.tensors:
        if info.type_id not in G.GGUF_TYPES:
            raise MoeImportError(
                f"тензор {info.name!r}: тип {G.type_label(info.type_id)} "
                f"не поддержан (поддержаны: {G.SUPPORTED_TYPE_NAMES})"
            )
        mapped = map_moe_tensor(info.name)
        if mapped is None:
            unknown.append(info.name)
            continue
        name, role, layer, key = mapped
        if layer is not None and layer >= cfg.n_layers:
            continue
        if role == "lm_head" and cfg.tie_word_embeddings:
            continue
        shape = info.hf_shape
        expected = _expected_shape(cfg, role, key, info.name)
        if expected is not None:
            if role in ("embed", "lm_head"):
                if shape[0] < expected[0] or shape[1] != expected[1]:
                    raise MoeImportError(
                        f"{info.name}: форма {shape}, ожидалась не меньше "
                        f"{expected}"
                    )
                shape = (min(shape[0], expected[0]), shape[1])
            elif tuple(shape) != tuple(expected):
                raise MoeImportError(
                    f"{info.name}: форма {shape}, ожидалась {expected}"
                )
        # Целевое хранилище: Q4_K/Q6_K копируются блоками, остальное — fp32.
        if role in NATIVE_ROLES:
            kind = info.spec.kind
            if kind not in ("q4_k", "q6_k"):
                raise MoeImportError(
                    f"{info.name}: тип {info.spec.name}, а MoE-импорт пишет "
                    "матрицы только нативными блоками Q4_K/Q6_K (ADR 0018); "
                    "переквантование в kind 5 для MoE не поддержано — "
                    "возьмите Q4_K_M/Q6_K GGUF"
                )
            target = "q4k" if kind == "q4_k" else "q6k"
            n_values = info.n_elements
            if n_values % G.Q4K_BLOCK_VALUES:
                raise MoeImportError(
                    f"{info.name}: {n_values} значений не кратны блоку GGML "
                    f"{G.Q4K_BLOCK_VALUES}"
                )
        elif role in FP32_ROLES:
            target = "f32"
        else:  # pragma: no cover - роли перечислены выше
            raise MoeImportError(f"{info.name}: роль {role!r}")
        token = (role, layer, key)
        if token in seen:
            raise MoeImportError(f"дубликат тензора по роли {role}/{layer}/{key}")
        seen.add(token)
        if layer is not None:
            layer_keys[layer].add(key)
        plans.append(Planned(info=info, name=name, role=role, layer=layer,
                             key=key, shape=tuple(shape), target=target))

    if unknown:
        preview = ", ".join(sorted(unknown)[:8])
        raise MoeImportError(
            "неизвестные тензоры в GGUF: " + preview
            + ". Импортёр знает Qwen2/3-MoE-раскладку "
            "(ffn_*_exps, ffn_gate_inp, attn_q_norm/k_norm)"
        )
    roles = {(plan.role, plan.layer, plan.key) for plan in plans}
    if ("embed", None, "embed") not in roles:
        raise MoeImportError("в GGUF нет token_embd.weight")
    if ("final_norm", None, "final_norm") not in roles:
        raise MoeImportError("в GGUF нет output_norm.weight")
    if ("lm_head", None, "lm_head") not in roles:
        raise MoeImportError("в GGUF нет output.weight")
    required = {"q_proj", "k_proj", "v_proj", "o_proj", "router",
                "expert_gate", "expert_up", "expert_down", "norm1", "norm2"}
    if cfg.qk_norm:
        required |= {"q_norm", "k_norm"}
    if cfg.shared_inter:
        required |= {"shared_gate", "shared_up", "shared_down"}
    for index in range(cfg.n_layers):
        missing = sorted(required - layer_keys.get(index, set()))
        if missing:
            raise MoeImportError(
                f"слой {index}: нет тензоров {', '.join(missing)} — "
                "манифест был бы неполным"
            )
    order = {"embed": 0, "matrix": 1, "norm": 2, "router": 3,
             "expert_gate": 4, "expert_up": 5, "expert_down": 6,
             "shared_gate": 7, "shared_up": 8, "shared_down": 9,
             "final_norm": 10, "lm_head": 11}
    plans.sort(key=lambda plan: (order.get(plan.role, 99),
                                 -1 if plan.layer is None else plan.layer,
                                 plan.key))
    return plans


def _expected_shape(
    cfg: MoeConfig, role: str, key: str, name: str
) -> tuple[int, ...] | None:
    if role in ("embed", "lm_head"):
        return (cfg.vocab, cfg.hidden)
    if role == "final_norm":
        return (cfg.hidden,)
    if role == "norm":
        return (cfg.head_dim,) if key in ("q_norm", "k_norm") else (cfg.hidden,)
    if role == "router":
        return (cfg.n_experts, cfg.hidden)
    if role in ("expert_gate", "expert_up"):
        return (cfg.n_experts, cfg.expert_inter, cfg.hidden)
    if role == "expert_down":
        return (cfg.n_experts, cfg.hidden, cfg.expert_inter)
    if role in ("shared_gate", "shared_up"):
        return (cfg.shared_inter, cfg.hidden)
    if role == "shared_down":
        return (cfg.hidden, cfg.shared_inter)
    dims = {
        "q_proj": (cfg.q_dim, cfg.hidden),
        "k_proj": (cfg.kv_dim, cfg.hidden),
        "v_proj": (cfg.kv_dim, cfg.hidden),
        "o_proj": (cfg.hidden, cfg.q_dim),
    }
    if key in dims:
        return dims[key]
    raise MoeImportError(f"{name}: неизвестная роль {role}/{key}")


def _native_header(n_values: int, kind: str) -> bytes:
    """18-байтовый заголовок сегмента kind 6/7 (ADR 0018)."""
    magic = G.Q4K_MAGIC if kind == "q4k" else G.Q6K_MAGIC
    n_blocks = n_values // G.Q4K_BLOCK_VALUES
    return G._SEGMENT_HEADER.pack(magic, G.Q4K_VERSION, 0,
                                  n_values & 0xFFFFFFFFFFFFFFFF, n_blocks)


def import_moe(
    gguf_path: str | Path,
    out_dir: str | Path,
    *,
    limit_layers: int | None = None,
    limit_vocab: int | None = None,
    stats: bool = False,
) -> dict[str, Any]:
    """Импортировать MoE GGUF в каталог манифеста. Вернуть манифест."""
    gguf = G.read_gguf(gguf_path)
    cfg = infer_moe_config(gguf, limit_layers=limit_layers,
                           limit_vocab=limit_vocab)
    plans = plan_moe(gguf, cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    q4_sink = G._Sink(out / Q4_FILE)
    fp_sink = G._Sink(out / NON_TERNARY_FILE)
    q4_entries: list[dict[str, Any]] = []
    fp_entries: list[dict[str, Any]] = []
    started = time.time()
    total_values = 0

    try:
        for plan in plans:
            info = plan.info
            n_values = _prod(plan.shape)
            if plan.target in ("q4k", "q6k"):
                q4_sink.align(32)
                offset = q4_sink.offset
                digest = hashlib.sha256()
                q4_sink.write(_native_header(n_values, plan.target))
                copied = 0
                expect = n_values // G.Q4K_BLOCK_VALUES * (
                    G.Q4K_BLOCK_BYTES if plan.target == "q4k"
                    else G.Q6K_BLOCK_BYTES
                )
                for chunk in gguf.iter_tensor_raw_chunks(info):
                    # Срез словаря/слоя: берём только нужный префикс блоков.
                    if copied + len(chunk) > expect:
                        chunk = chunk[: expect - copied]
                    if not chunk:
                        break
                    q4_sink.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                    if copied >= expect:
                        break
                if copied != expect:
                    raise MoeImportError(
                        f"{info.name}: скопировано {copied} байт, ожидалось "
                        f"{expect} — срез/форма не сходятся"
                    )
                q4_entries.append({
                    "name": plan.name,
                    "role": plan.role,
                    "dtype": "q4_k" if plan.target == "q4k" else "q6_k",
                    "kind": plan.target,
                    "shape": list(plan.shape),
                    "gguf_name": info.name,
                    "gguf_type": info.spec.name,
                    "q4_offset": offset,
                    "q4_length": q4_sink.offset - offset,
                    "blocks_sha256": digest.hexdigest(),
                })
            else:
                fp_sink.align(32)
                offset = fp_sink.offset
                values = []
                for chunk in gguf.iter_tensor_chunks(info, n_values):
                    values.extend(chunk)
                if len(values) != n_values:
                    raise MoeImportError(
                        f"{info.name}: значений {len(values)}, ожидалось "
                        f"{n_values}"
                    )
                payload = struct.pack(f"<{n_values}f", *values)
                fp_sink.write(payload)
                fp_entries.append({
                    "name": plan.name,
                    "role": plan.role,
                    "dtype": "float32",
                    "shape": list(plan.shape),
                    "gguf_name": info.name,
                    "offset": offset,
                    "length": fp_sink.offset - offset,
                })
            total_values += n_values
            print(
                f"  {plan.role:13s} {plan.name:52s} {str(plan.shape):24s} "
                f"{plan.target}",
                file=sys.stderr,
            )
    except BaseException:
        q4_sink.abort()
        fp_sink.abort()
        raise
    q4_sink.close()
    fp_sink.close()

    # Компилятор всегда открывает ternary.trit (даже когда тритов нет) —
    # та же конвенция, что у импорта плотных GGUF (0153).
    (out / "ternary.trit").write_bytes(b"")

    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_VERSION,
        "generator": "tools/moe_import.py",
        "source": {
            "gguf": str(Path(gguf_path).name),
            "architecture": cfg.architecture,
            "name": cfg.name,
            "gguf_version": gguf.version,
            "file_type": gguf.metadata.get("general.file_type"),
            "limit_layers": cfg.n_layers,
            "limit_vocab": cfg.vocab,
        },
        "base_model": {"name": cfg.name},
        "architecture": cfg.as_architecture(),
        "files": {
            "ternary.trit": {
                "bytes": 0,
                "sha256": G._sha256_file(out / "ternary.trit"),
            },
            Q4_FILE: {
                "bytes": q4_sink.offset,
                "sha256": G._sha256_file(out / Q4_FILE),
            },
            NON_TERNARY_FILE: {
                "bytes": fp_sink.offset,
                "sha256": G._sha256_file(out / NON_TERNARY_FILE),
            },
        },
        "q4": {"file": Q4_FILE},
        "embeddings": {
            "mode": "q4_k",
            "bytes": next(
                (entry["q4_length"] for entry in q4_entries
                 if entry["role"] == "embed"), 0
            ),
        },
        "tensors": q4_entries,
        "non_ternary": {
            "stored_in_this_export": True,
            "file": NON_TERNARY_FILE,
            "tensors": fp_entries,
        },
        "quantization": {
            "scheme": "ggml-native",
            # block_size здесь — блок тритов VTM (128, его проверяет
            # компилятор), а не блок GGML: Q4_K/Q6_K живут блоками 256
            # значений внутри сегментов kind 6/7.
            "block_size": G.TRIT_SCALE_BLOCK,
            "ggml_block": G.Q4K_BLOCK_VALUES,
            "note": "Q4_K/Q6_K блоки скопированы байт-в-байт (ADR 0018)",
        },
        "stats": {
            "tensors": len(plans),
            "values": total_values,
            "q4_bytes": q4_sink.offset,
            "fp32_bytes": fp_sink.offset,
            "seconds": round(time.time() - started, 2),
            "rel_rms": 0.0,
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def verify_blocks(gguf_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Сверить блоки сегментов `q4.bin` с регионами GGUF по sha256.

    Нативный импорт не переквантовывает веса, поэтому проверка точная:
    sha256 скопированных блоков обязан совпасть с sha256 того же региона
    исходного GGUF. Возвращает сводку (сколько сегментов, байт, расхождений).
    """
    import mmap

    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    gguf = G.read_gguf(gguf_path)
    handle = open(out / Q4_FILE, "rb")
    try:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    except (OSError, ValueError):
        handle.close()
        raise
    checked = 0
    mismatched: list[str] = []
    total = 0
    try:
        for entry in manifest.get("tensors") or []:
            name = str(entry.get("name"))
            info = gguf.tensor(str(entry["gguf_name"]))
            raw = gguf.raw_tensor(info)
            segment = mapped[entry["q4_offset"]:entry["q4_offset"]
                             + entry["q4_length"]]
            blocks = bytes(segment[18:])
            digest = hashlib.sha256(blocks).hexdigest()
            checked += 1
            total += len(blocks)
            if digest != entry.get("blocks_sha256") or blocks != raw:
                mismatched.append(name)
    finally:
        mapped.close()
        handle.close()
    return {"checked": checked, "bytes": total, "mismatched": mismatched}


def _prod(shape: Sequence[int]) -> int:
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moe_import.py",
        description="Импорт MoE GGUF (Qwen3-30B-A3B) в веса VTM",
    )
    parser.add_argument("--gguf", required=True, help="исходный GGUF")
    parser.add_argument("--out", required=True, help="каталог манифеста")
    parser.add_argument("--limit-layers", type=int, default=None,
                        help="взять только первые N слоёв (срез)")
    parser.add_argument("--limit-vocab", type=int, default=None,
                        help="срез таблицы словаря с начала (отладка)")
    parser.add_argument("--stats", action="store_true",
                        help="сверить блоки q4.bin с GGUF по sha256")
    parser.add_argument("--quiet", action="store_true",
                        help="не печатать по тензору")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quiet:
        saved = sys.stderr
        try:
            class _Null:
                def write(self, *_a):  # pragma: no cover
                    return 0
                def flush(self):  # pragma: no cover
                    return None
            sys.stderr = _Null()  # type: ignore[assignment]
            manifest = import_moe(
                args.gguf, args.out, limit_layers=args.limit_layers,
                limit_vocab=args.limit_vocab, stats=args.stats,
            )
        finally:
            sys.stderr = saved
    else:
        manifest = import_moe(
            args.gguf, args.out, limit_layers=args.limit_layers,
            limit_vocab=args.limit_vocab, stats=args.stats,
        )
    if args.stats:
        report = verify_blocks(args.gguf, args.out)
        status = "совпали" if not report["mismatched"] else "РАСХОЖДЕНИЕ"
        print(
            f"сверка блоков с GGUF: {report['checked']} сегментов, "
            f"{report['bytes'] / 1e9:.3f} ГБ — {status}"
            + ("" if not report["mismatched"]
               else ": " + ", ".join(report["mismatched"][:5]))
        )
        if report["mismatched"]:
            return 1
    stats = manifest["stats"]
    print(
        f"импорт: {stats['tensors']} тензоров, {stats['values'] / 1e9:.3f} "
        f"млрд значений, q4.bin {stats['q4_bytes'] / 1e9:.3f} ГБ, "
        f"non_ternary.bin {stats['fp32_bytes'] / 1e6:.1f} МБ, "
        f"{stats['seconds']} с"
    )
    print(f"манифест: {Path(args.out) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
