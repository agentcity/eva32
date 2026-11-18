#!/usr/bin/env python3
"""Случайный паритет C ↔ Python: N случайных архитектур VTM.

Зачем
=====

Паритет `src/vtm/` с `tools/vtm_ref.py` проверялся на фиксированных
конфигурациях (`tools/parity_real.py` — одна настоящая Qwen, `make_synthetic`
— одна синтетическая сетка). Ошибки индексации GQA (`group = n_heads /
n_kv_heads`), раскладки голов в Q/K/V, RoPE и `lm_head` проявляются только на
других соотношениях `heads/kv_heads/head_dim`, а их фиксированные пробы не
покрывают. Этот скрипт гоняет случайные конфигурации и сравнивает числа.

Что делает один прогон
======================

1. `generate_config(seed, index)` — детерминированная случайная архитектура:
   `hidden` 64..256 (кратно `heads*head_dim`), `heads` 1..8, `kv_heads`
   1..heads (делитель heads), `head_dim` 8..64 (чётный: RoPE), слоёв 1..3,
   `vocab` 32..512, `rope_theta` 1e4..1e6 (лог-равномерно), tied/untied,
   блок масштаба тритов 128/160. Зависит только от `(seed, index)` — любую
   конфигурацию можно воспроизвести, не гоняя предыдущие.
2. `build_model` собирает веса (случайные триты с настоящими absmax-масштабами
   на блок) и программу через `tools/vtm_compile.py` (`compile_model`), пишет
   контейнер `.vtm`.
3. Python-эталон: `vtm_ref.execute(vtm, tokens, 0)` — логиты последнего
   прохода.
4. C: `src/vtm/vtm run ... --max-new 1` — штатный CLI (проверяется argmax) и
   маленькая проба на `src/vtm/libvtm.a`, которая печатает сам вектор логитов
   (без неё `max|Δ|` не измерить; так же делает `tools/parity_real.py`).
5. Сравнение: `max|Δ|` логитов, совпадение argmax (Python ↔ C-проба ↔ CLI),
   порог `--tolerance` (по умолчанию 1e-5).

Блок масштаба 128/160
=====================

128 — канон Python-эталона (`vtm_ref.SCALE_BLOCK`), 160 — блок упаковки
`pack20` (`docs/trit-format.md`). C-загрузчик различает их по числу масштабов
в TRT1 и по флагу заголовка `VTM_HEADER_FLAG_SCALE_BLOCK_160` (0x8000);
Python-эталон читает с диска только 128. Поэтому для блока 160:

* тритовый тензор собирается здесь (`trit_tensor`), в обход ограничения
  `vtm_ref.Tensor` (он намеренно принимает только 128), и живёт в памяти —
  `vtm_ref.execute` считает его корректно, потому что берёт
  `tensor.block_size`;
* после записи контейнера в заголовок ставится бит 0x8000, иначе C на
  неоднозначном числе масштабов (например, 200 тритов: ceil(200/128) =
  ceil(200/160) = 2) выберет 128 и молча посчитает другое.

Ограничение: контейнер с блоком 160 Python-эталон с диска не прочитает
(`vtm_ref.read_vtm`), сравнение идёт с моделью в памяти. Это осознанно:
канон Python — 128, а 160 — расширение C.

Токены
======

У каждой конфигурации три токена (`[t0, t1, t2]` из её словаря) — этого
достаточно, чтобы включились причинная маска, KV-кэш и GQA-группировка:
позиция 0 видит только себя, позиции 1–2 — предыдущие. `context_len=8`.

Запуск
======

    python3 tools/parity_random.py                     # 50 конфигураций, seed 0
    python3 tools/parity_random.py --count 10 --seed 7
    python3 tools/parity_random.py --json
    python3 tools/parity_random.py --seed 0 --only 12  # воспроизвести одну

Коды возврата
=============

* 0 — все конфигурации в пороге, argmax и CLI совпали;
* 1 — есть расхождения/ошибки прогона (конфигурация печатается);
* 2 — нет C-рантайма, библиотеки или компилятора (сначала `make -C src/vtm`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for _path in (str(ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tools import parity_real  # noqa: E402
from tools import vtm_compile  # noqa: E402
from tools import vtm_ref  # noqa: E402

TOOL_NAME = "parity_random"
DEFAULT_COUNT = 50
DEFAULT_SEED = 0
DEFAULT_TOLERANCE = 1e-5
DEFAULT_WORKDIR = ROOT / "data" / "parity_random"

#: Сколько токенов прогоняется на конфигурацию (позиции 0..n-1).
PROMPT_TOKENS = 3
#: Длина KV-кэша в контейнере: с запасом больше PROMPT_TOKENS.
CONTEXT_LEN = 8
#: eps в RMSN: компилятор кодирует его как imm/1000, поэтому 1e-3 -> imm=1.
EPS = 1e-3

#: Диапазоны из задачи 0072.
HIDDEN_RANGE = (64, 256)
HEADS_RANGE = (1, 8)
HEAD_DIM_RANGE = (8, 64)
LAYERS_RANGE = (1, 3)
VOCAB_RANGE = (32, 512)
ROPE_THETA_RANGE = (1.0e4, 1.0e6)
SCALE_BLOCKS = (vtm_ref.SCALE_BLOCK, vtm_ref.TRITS_PER_STORAGE_BLOCK)  # 128, 160
ROPE_LAYOUTS = ("half", "interleaved")

#: Бит заголовка `VTM_HEADER_FLAG_SCALE_BLOCK_160` (src/vtm/vtm.h).
HEADER_FLAG_SCALE_BLOCK_160 = 0x8000
#: Смещение поля flags в заголовке `.vtm` (magic 4 + version u16).
HEADER_FLAGS_OFFSET = 6


class ParityRandomError(RuntimeError):
    """Ошибка окружения/подготовки: код возврата 2."""


# ---------------------------------------------------------------------------
# Конфигурации
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomConfig:
    """Одна случайная архитектура и её вход."""

    index: int
    seed: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    n_layers: int
    rope_theta: float
    rope_layout: str
    tie_word_embeddings: bool
    scale_block: int
    context_len: int
    eps: float
    tokens: tuple[int, ...]

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def config_id(self) -> str:
        return f"{self.seed}:{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "seed": self.seed,
            "config_id": self.config_id,
            "hidden": self.hidden,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "q_dim": self.q_dim,
            "kv_dim": self.kv_dim,
            "intermediate": self.intermediate,
            "vocab": self.vocab,
            "n_layers": self.n_layers,
            "rope_theta": self.rope_theta,
            "rope_layout": self.rope_layout,
            "tie_word_embeddings": self.tie_word_embeddings,
            "scale_block": self.scale_block,
            "context_len": self.context_len,
            "eps": self.eps,
            "tokens": list(self.tokens),
        }

    def to_model_config(self) -> vtm_compile.ModelConfig:
        """`ModelConfig` для `vtm_compile.compile_model`.

        `ModelConfig.__post_init__` разрешает только блок 128 (канон
        Python-эталона), поэтому 160 ставится после конструктора: поле
        `block_size` графа — единственное, что видит наш `trit_tensor`.
        """
        cfg = vtm_compile.ModelConfig(
            hidden=self.hidden,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            intermediate=self.intermediate,
            vocab=self.vocab,
            n_layers=self.n_layers,
            context_len=self.context_len,
            eps=self.eps,
            rope_theta=self.rope_theta,
            rope_layout=self.rope_layout,
            tie_word_embeddings=self.tie_word_embeddings,
        )
        if self.scale_block != vtm_ref.SCALE_BLOCK:
            cfg.block_size = int(self.scale_block)
        return cfg


def config_rng(seed: int, index: int) -> random.Random:
    """ГПСЧ конфигурации: зависит только от (seed, index)."""
    return random.Random(f"vtm/parity_random/v1/config:{int(seed)}:{int(index)}")


def weights_rng(seed: int, index: int) -> random.Random:
    """ГПСЧ весов: отдельное пространство имён, чтобы правка генератора
    конфигураций не меняла веса (и наоборот)."""
    return random.Random(f"vtm/parity_random/v1/weights:{int(seed)}:{int(index)}")


def _divisors(value: int) -> list[int]:
    return [d for d in range(1, value + 1) if value % d == 0]


def _sample_architecture(rng: random.Random) -> dict[str, int]:
    """Сэмплировать heads/head_dim/hidden, совместимые с диапазонами.

    `hidden` — кратное `heads*head_dim` в [64, 256]; `head_dim` чётный
    (RoPE), `kv_heads` делит `heads`. Несовместимые комбинации (например,
    8 голов × head_dim 64 = 512 > 256) отбрасываются и сэмплируются заново —
    так все конфигурации допустимы, а не «подогнаны» к одной сетке.
    """
    for _ in range(10_000):
        head_dim = 2 * rng.randint(HEAD_DIM_RANGE[0] // 2, HEAD_DIM_RANGE[1] // 2)
        n_heads = rng.randint(HEADS_RANGE[0], HEADS_RANGE[1])
        q_dim = n_heads * head_dim
        if q_dim > HIDDEN_RANGE[1]:
            continue
        multipliers = [
            m
            for m in range(1, HIDDEN_RANGE[1] // q_dim + 1)
            if HIDDEN_RANGE[0] <= q_dim * m <= HIDDEN_RANGE[1]
        ]
        if not multipliers:
            continue
        hidden = q_dim * rng.choice(multipliers)
        n_kv_heads = rng.choice(_divisors(n_heads))
        return {
            "hidden": hidden,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
        }
    raise ParityRandomError(  # pragma: no cover - недостижимо при диапазонах
        "не удалось сэмплировать архитектуру: диапазоны несовместимы"
    )


def _sample_rope_theta(rng: random.Random) -> float:
    """Лог-равномерно 1e4..1e6, округление до 6 значащих цифр."""
    low, high = math.log10(ROPE_THETA_RANGE[0]), math.log10(ROPE_THETA_RANGE[1])
    value = 10.0 ** rng.uniform(low, high)
    value = float(f"{value:.6g}")
    return min(max(value, ROPE_THETA_RANGE[0]), ROPE_THETA_RANGE[1])


def generate_config(seed: int = DEFAULT_SEED, index: int = 0) -> RandomConfig:
    """Детерминированная случайная конфигурация номер `index` для `seed`."""
    if int(index) < 0:
        raise ValueError(f"index должен быть >= 0, получено {index}")
    seed = int(seed)
    rng = config_rng(seed, index)
    arch = _sample_architecture(rng)
    vocab = rng.randint(VOCAB_RANGE[0], VOCAB_RANGE[1])
    n_layers = rng.randint(LAYERS_RANGE[0], LAYERS_RANGE[1])
    intermediate = arch["hidden"] * rng.choice((2, 3, 4))
    rope_theta = _sample_rope_theta(rng)
    rope_layout = rng.choice(ROPE_LAYOUTS)
    tie = rng.random() < 0.5
    scale_block = rng.choice(SCALE_BLOCKS)
    tokens = tuple(rng.randrange(vocab) for _ in range(PROMPT_TOKENS))
    return RandomConfig(
        index=int(index),
        seed=seed,
        hidden=arch["hidden"],
        n_heads=arch["n_heads"],
        n_kv_heads=arch["n_kv_heads"],
        head_dim=arch["head_dim"],
        intermediate=int(intermediate),
        vocab=int(vocab),
        n_layers=int(n_layers),
        rope_theta=rope_theta,
        rope_layout=rope_layout,
        tie_word_embeddings=bool(tie),
        scale_block=int(scale_block),
        context_len=CONTEXT_LEN,
        eps=EPS,
        tokens=tokens,
    )


def generate_configs(seed: int = DEFAULT_SEED, count: int = DEFAULT_COUNT) -> list[RandomConfig]:
    """`count` конфигураций подряд; каждая не зависит от предыдущих."""
    if int(count) < 0:
        raise ValueError(f"count должен быть >= 0, получено {count}")
    return [generate_config(seed, index) for index in range(int(count))]


def validate_config(config: RandomConfig) -> list[str]:
    """Список нарушений диапазонов задачи (пустой — конфигурация валидна)."""
    problems: list[str] = []
    low, high = HIDDEN_RANGE
    if not low <= config.hidden <= high:
        problems.append(f"hidden={config.hidden} вне {low}..{high}")
    if config.hidden % (config.n_heads * config.head_dim):
        problems.append(
            f"hidden={config.hidden} не кратно heads*head_dim="
            f"{config.n_heads * config.head_dim}"
        )
    if not HEADS_RANGE[0] <= config.n_heads <= HEADS_RANGE[1]:
        problems.append(f"n_heads={config.n_heads} вне {HEADS_RANGE}")
    if not 1 <= config.n_kv_heads <= config.n_heads:
        problems.append(f"n_kv_heads={config.n_kv_heads} вне 1..{config.n_heads}")
    if config.n_heads % config.n_kv_heads:
        problems.append(
            f"n_kv_heads={config.n_kv_heads} не делитель n_heads={config.n_heads}"
        )
    low, high = HEAD_DIM_RANGE
    if not low <= config.head_dim <= high:
        problems.append(f"head_dim={config.head_dim} вне {low}..{high}")
    if config.head_dim % 2:
        problems.append(f"head_dim={config.head_dim} нечётный (RoPE)")
    if not LAYERS_RANGE[0] <= config.n_layers <= LAYERS_RANGE[1]:
        problems.append(f"n_layers={config.n_layers} вне {LAYERS_RANGE}")
    if not VOCAB_RANGE[0] <= config.vocab <= VOCAB_RANGE[1]:
        problems.append(f"vocab={config.vocab} вне {VOCAB_RANGE}")
    if not ROPE_THETA_RANGE[0] <= config.rope_theta <= ROPE_THETA_RANGE[1]:
        problems.append(f"rope_theta={config.rope_theta} вне {ROPE_THETA_RANGE}")
    if config.rope_layout not in ROPE_LAYOUTS:
        problems.append(f"rope_layout={config.rope_layout!r} вне {ROPE_LAYOUTS}")
    if config.scale_block not in SCALE_BLOCKS:
        problems.append(f"scale_block={config.scale_block} вне {SCALE_BLOCKS}")
    if config.intermediate <= 0:
        problems.append(f"intermediate={config.intermediate} <= 0")
    if config.context_len < len(config.tokens):
        problems.append(
            f"context_len={config.context_len} меньше числа токенов "
            f"{len(config.tokens)}"
        )
    if config.context_len <= 0:
        problems.append(f"context_len={config.context_len} <= 0")
    for token in config.tokens:
        if not 0 <= token < config.vocab:
            problems.append(f"токен {token} вне словаря 0..{config.vocab - 1}")
    return problems


# ---------------------------------------------------------------------------
# Тритовые тензоры (включая блок 160, которого нет в vtm_ref.Tensor)
# ---------------------------------------------------------------------------


def trit_tensor(
    name: str,
    dims: Sequence[int],
    values: Iterable[float],
    block_size: int = vtm_ref.SCALE_BLOCK,
    scale_dtype: str = "float32",
) -> vtm_ref.Tensor:
    """Тритовый `Tensor` с масштабом на блок `block_size` (128 или 160).

    Для 128 используется штатный `Tensor.ternary`. Для 160 — тензор
    создаётся через `object.__new__`: `Tensor.__post_init__` намеренно
    запрещает блок, отличный от канона Python (128), а C его поддерживает
    (`VTM_HEADER_FLAG_SCALE_BLOCK_160`). Все поля заполняются и проверяются
    здесь же, чтобы объект вёл себя как обычный `Tensor` (в частности,
    `dense()` и `encode_trit_segment()` читают `block_size`).
    """
    block_size = int(block_size)
    if block_size not in SCALE_BLOCKS:
        raise ValueError(f"block_size={block_size}, допустимо {SCALE_BLOCKS}")
    dims_tuple = tuple(int(d) for d in dims)
    n_values = vtm_ref.prod(dims_tuple)
    trits, scales = vtm_ref.quantize_trits(values, block_size=block_size)
    if len(trits) != n_values:
        raise ValueError(f"тензор {name!r}: {len(trits)} тритов, dims дают {n_values}")
    expected_scales = vtm_ref.n_scale_blocks(n_values, block_size)
    if len(scales) != expected_scales:  # pragma: no cover - защита от рассинхрона
        raise ValueError(
            f"тензор {name!r}: масштабов {len(scales)}, ожидалось {expected_scales}"
        )
    if block_size == vtm_ref.SCALE_BLOCK:
        return vtm_ref.Tensor.ternary(name, dims_tuple, trits, scales, scale_dtype)
    tensor = object.__new__(vtm_ref.Tensor)
    tensor.name = str(name)
    tensor.kind = vtm_ref.KIND_TRITS
    tensor.dims = dims_tuple
    tensor.data = None
    tensor.trits = trits
    tensor.scales = scales
    tensor.scale_dtype = scale_dtype
    tensor.block_size = block_size
    tensor.offset = 0
    tensor.scale_offset = vtm_ref.NO_OFFSET
    return tensor


def mark_scale_block_160(path: str | os.PathLike[str]) -> int:
    """Поставить в заголовке `.vtm` бит «масштаб на 160 тритов».

    Нужно, когда число масштабов неоднозначно: при 200 тритах
    `ceil(200/128) == ceil(200/160) == 2`, и C иначе выберет 128. Бит
    `0x8000` заставляет загрузчик взять 160 (`src/vtm/loader.c`). Python-эталон
    этот бит не ставит и не читает; здесь правится только наш файл.
    """
    target = Path(path)
    data = bytearray(target.read_bytes())
    if len(data) < HEADER_FLAGS_OFFSET + 2:
        raise ValueError(f"{target}: файл короче заголовка VTM")
    if data[:4] != b"VTM1":
        raise ValueError(f"{target}: это не контейнер VTM (magic {data[:4]!r})")
    flags = struct.unpack_from("<H", data, HEADER_FLAGS_OFFSET)[0]
    flags |= HEADER_FLAG_SCALE_BLOCK_160
    struct.pack_into("<H", data, HEADER_FLAGS_OFFSET, flags)
    target.write_bytes(bytes(data))
    return flags


def read_header_flags(path: str | os.PathLike[str]) -> int:
    """Флаги заголовка `.vtm` без полного разбора контейнера (для отчёта/тестов)."""
    data = Path(path).read_bytes()[: HEADER_FLAGS_OFFSET + 2]
    if len(data) < HEADER_FLAGS_OFFSET + 2 or data[:4] != b"VTM1":
        raise ValueError(f"{path}: это не контейнер VTM")
    return struct.unpack_from("<H", data, HEADER_FLAGS_OFFSET)[0]


# ---------------------------------------------------------------------------
# Модель и контейнер
# ---------------------------------------------------------------------------


def build_model(config: RandomConfig) -> vtm_ref.VTM:
    """Собрать VTM в памяти: случайные веса по конфигурации.

    Матрицы слоёв — триты с absmax-масштабами на `config.scale_block`,
    нормы и эмбеддинги — fp32, `lm_head` при untied — плотная fp32-матрица
    (политика QAT: lm_head не тернаризуется), при tied — `MM` по таблице
    эмбеддингов (компилятор сам эмитит `MM`, отдельного тензора нет).
    """
    cfg = config.to_model_config()
    rng = weights_rng(config.seed, config.index)
    hidden, intermediate = cfg.hidden, cfg.intermediate

    def matrix(name: str, rows: int, cols: int) -> vtm_ref.Tensor:
        values = [rng.uniform(-1.0, 1.0) for _ in range(rows * cols)]
        return trit_tensor(name, (rows, cols), values, config.scale_block)

    def norm_vector() -> list[float]:
        return [rng.uniform(0.5, 1.5) for _ in range(hidden)]

    layers: list[dict[str, Any]] = []
    for _ in range(cfg.n_layers):
        layers.append(
            {
                "norm1": norm_vector(),
                "norm2": norm_vector(),
                "q_proj": matrix("q_proj", cfg.q_dim, hidden),
                "k_proj": matrix("k_proj", cfg.kv_dim, hidden),
                "v_proj": matrix("v_proj", cfg.kv_dim, hidden),
                "o_proj": matrix("o_proj", hidden, cfg.q_dim),
                "gate_proj": matrix("gate_proj", intermediate, hidden),
                "up_proj": matrix("up_proj", intermediate, hidden),
                "down_proj": matrix("down_proj", hidden, intermediate),
            }
        )

    weights: dict[str, Any] = {
        "embed": [rng.uniform(-1.0, 1.0) for _ in range(cfg.vocab * hidden)],
        "layers": layers,
        "final_norm": norm_vector(),
    }
    if not cfg.tie_word_embeddings:
        weights["lm_head"] = [
            rng.uniform(-1.0, 1.0) for _ in range(cfg.vocab * hidden)
        ]

    return vtm_compile.compile_model(
        cfg,
        weights,
        meta_extra={
            "scale_block": int(config.scale_block),
            "parity_random": config.to_dict(),
        },
    )


def write_container(
    vtm: vtm_ref.VTM,
    path: str | os.PathLike[str],
    config: RandomConfig,
) -> Path:
    """Записать контейнер; для блока 160 — выставить бит заголовка."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    vtm_ref.write_vtm(target, vtm.tensors, vtm.program, vtm.meta)
    if config.scale_block != vtm_ref.SCALE_BLOCK:
        mark_scale_block_160(target)
    return target


# ---------------------------------------------------------------------------
# Сравнение
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    """Результат сверки двух векторов логитов."""

    max_abs_delta: float
    argmax_reference: int
    argmax_runtime: int
    argmax_match: bool
    within_tolerance: bool
    tolerance: float

    @property
    def passed(self) -> bool:
        return bool(self.argmax_match and self.within_tolerance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs_delta": _json_float(self.max_abs_delta),
            "argmax_reference": self.argmax_reference,
            "argmax_runtime": self.argmax_runtime,
            "argmax_match": self.argmax_match,
            "within_tolerance": self.within_tolerance,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }


def compare_logits(
    reference: Sequence[float],
    runtime: Sequence[float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> Comparison:
    """Сверить логиты Python-эталона и C.

    Порог строгий: `max|Δ| < tolerance` (как в `tools/parity_real.py`).
    argmax — первый максимум; при равенстве значений выигрывает меньший
    индекс (и в C, и в `vtm_ref`).
    """
    if len(reference) != len(runtime):
        raise ValueError(
            f"размеры логитов не совпали: Python {len(reference)}, C {len(runtime)}"
        )
    if not reference:
        raise ValueError("пустые логиты")
    max_delta = max(abs(float(a) - float(b)) for a, b in zip(reference, runtime))
    ref_argmax = parity_real.argmax(reference)
    run_argmax = parity_real.argmax(runtime)
    return Comparison(
        max_abs_delta=max_delta,
        argmax_reference=ref_argmax,
        argmax_runtime=run_argmax,
        argmax_match=ref_argmax == run_argmax,
        within_tolerance=max_delta < float(tolerance),
        tolerance=float(tolerance),
    )


def _json_float(value: float) -> Any:
    return value if math.isfinite(value) else str(value)


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "—"
    if not math.isfinite(value):
        return str(value)
    return f"{value:.3e}"


# ---------------------------------------------------------------------------
# Окружение, C-проба и CLI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Environment:
    vtm_bin: Path
    vtm_lib: Path
    include_dir: Path
    cc: str | None

    @property
    def ready(self) -> bool:
        return bool(
            self.cc
            and self.vtm_bin.is_file()
            and os.access(self.vtm_bin, os.X_OK)
            and self.vtm_lib.is_file()
            and (self.include_dir / "vtm.h").is_file()
        )

    def missing(self) -> list[str]:
        missing: list[str] = []
        if not (self.vtm_bin.is_file() and os.access(self.vtm_bin, os.X_OK)):
            missing.append("src/vtm/vtm (исполняемый)")
        if not self.vtm_lib.is_file():
            missing.append("src/vtm/libvtm.a")
        if not (self.include_dir / "vtm.h").is_file():
            missing.append("src/vtm/vtm.h")
        if not self.cc:
            missing.append("C-компилятор (cc/clang/gcc)")
        return missing


def check_environment() -> Environment:
    return Environment(
        vtm_bin=parity_real.VTM_BIN,
        vtm_lib=parity_real.VTM_LIB,
        include_dir=parity_real.VTM_INCLUDE,
        cc=parity_real.find_c_compiler(),
    )


# Паритет «бит-в-бит» возможен только на скалярном бэкенде: SIMD переупорядочивает
# сложения, и логиты расходятся на ~1e-5 — для проверки корректности это нормально,
# но порог 1e-5 тогда не имеет смысла. Поэтому C-пробы гоняем со скаляром; бэкенд
# можно переопределить переменной VTM_PARITY_BACKEND (например, чтобы измерить
# расхождение SIMD отдельно).
PARITY_BACKEND = os.environ.get("VTM_PARITY_BACKEND", "scalar")


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path = ROOT,
    timeout: float | None = None,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    if not quiet:
        print(
            f"[{TOOL_NAME}] $ " + shlex.join([str(part) for part in cmd]),
            file=sys.stderr,
            flush=True,
        )
    try:
        env = os.environ.copy()
        env["VTM_BACKEND"] = PARITY_BACKEND
        return subprocess.run(
            [str(part) for part in cmd],
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise ParityRandomError(f"команда не найдена: {cmd[0]}") from exc


def _tail(text: str, lines: int = 20) -> str:
    chunk = [line for line in (text or "").strip().splitlines() if line.strip()]
    return "\n".join(chunk[-lines:])


def probe_source() -> str:
    """Исходник C-пробы: логиты последней позиции, 17 значащих цифр.

    Логика та же, что в `tools/parity_real.py` (тот же `libvtm.a`,
    `vtm_set_position`/`vtm_set_token` на каждый токен), но `%.17g` вместо
    `%.9g`: 9 значащих цифр — это точность ~5e-8 для логарифма порядка 1,
    то есть измеренный `max|Δ|` упирался бы в формат вывода, а не в
    расхождение рантайма. 17 цифр возвращают float32 без потерь.
    """
    return r'''/* parity_probe.c — снять logits последней позиции из .vtm.
 *
 * Собирается tools/parity_random.py и линкуется с src/vtm/libvtm.a.
 * Печатает "# key value" (meta) и по одному float (%.17g) на строку. */
#include "vtm.h"

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>

static int parse_tokens(const char *text, int **out, size_t *out_n) {
    int *tokens = NULL;
    size_t n = 0, cap = 0;
    const char *p = text;

    while (*p != '\0') {
        char *end = NULL;
        long value;
        while (*p == ',' || *p == ' ' || *p == '\t' || *p == '\n') p++;
        if (*p == '\0') break;
        value = strtol(p, &end, 10);
        if (end == p || value < 0 || value > 2147483647L) {
            free(tokens);
            return -1;
        }
        if (n == cap) {
            size_t next = cap == 0 ? 16 : cap * 2;
            int *grown = (int *)realloc(tokens, next * sizeof(*tokens));
            if (grown == NULL) {
                free(tokens);
                return -1;
            }
            tokens = grown;
            cap = next;
        }
        tokens[n++] = (int)value;
        p = end;
    }
    *out = tokens;
    *out_n = n;
    return 0;
}

int main(int argc, char **argv) {
    vtm_t *machine;
    const vtm_tensor *tensor;
    int *tokens = NULL;
    size_t n_tokens = 0;
    uint32_t i;
    vtm_status status;

    /* Десятичная точка, а не запятая: иначе парсер Python не прочитает
     * логиты в русской локали. */
    setlocale(LC_NUMERIC, "C");

    if (argc < 3) {
        fprintf(stderr, "parity_probe: нужны model.vtm и список tokens\n");
        return 2;
    }
    if (parse_tokens(argv[2], &tokens, &n_tokens) != 0 || n_tokens == 0) {
        fprintf(stderr, "parity_probe: плохой список токенов\n");
        return 2;
    }

    machine = vtm_load(argv[1]);
    if (machine == NULL) {
        fprintf(stderr, "parity_probe: vtm_load: %s\n", vtm_last_error());
        free(tokens);
        return 1;
    }

    printf("# meta_present %d\n", vtm_meta_present(machine));
    printf("# meta_rope_theta %.17g\n", vtm_meta_rope_theta(machine));
    printf("# meta_rope_layout %d\n", vtm_meta_rope_layout(machine));
    printf("# meta_vocab %u\n", (unsigned)vtm_meta_vocab_size(machine));
    printf("# meta_tied %d\n", vtm_meta_tied_embeddings(machine));

    for (i = 0; i < (uint32_t)n_tokens; i++) {
        vtm_set_position(machine, (int)i);
        vtm_set_token(machine, tokens[i]);
        status = vtm_run(machine, NULL, NULL);
        if (status != VTM_OK) {
            fprintf(stderr, "parity_probe: прогон %u: %s: %s\n",
                    i, vtm_strerror(status), vtm_error_message(machine));
            free(tokens);
            vtm_free(machine);
            return 1;
        }
    }

    tensor = vtm_tensor_by_name(machine, "logits");
    if (tensor == NULL) {
        fprintf(stderr, "parity_probe: в контейнере нет тензора logits\n");
        free(tokens);
        vtm_free(machine);
        return 1;
    }
    if (tensor->kind != VTM_TENSOR_F32 || tensor->f32 == NULL) {
        fprintf(stderr, "parity_probe: logits не fp32 (kind=%u)\n",
                (unsigned)tensor->kind);
        free(tokens);
        vtm_free(machine);
        return 1;
    }

    printf("# count %u\n", (unsigned)tensor->n_elements);
    for (i = 0; i < tensor->n_elements; i++) {
        printf("%.17g\n", tensor->f32[i]);
    }

    free(tokens);
    vtm_free(machine);
    return 0;
}
'''


#: Имя C-пробы: своё (не `parity_probe`), чтобы не столкнуться с артефактами
#: `tools/parity_real.py`, если tools запускают в одном workdir.
PROBE_NAME = "parity_random_probe"


def probe_compile_command(env: Environment, workdir: Path) -> list[str]:
    return [
        env.cc or "cc",
        "-std=c99",
        "-O2",
        "-Wall",
        "-Wextra",
        "-I",
        str(env.include_dir),
        "-o",
        str(workdir / PROBE_NAME),
        str(workdir / f"{PROBE_NAME}.c"),
        str(env.vtm_lib),
        "-lm",
    ]


def ensure_probe(workdir: Path, env: Environment) -> Path:
    """Собрать C-пробу (исходник — в workdir, не в репозитории)."""
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / f"{PROBE_NAME}.c"
    source.write_text(probe_source(), encoding="utf-8")
    binary = workdir / PROBE_NAME
    cmd = probe_compile_command(env, workdir)
    proc = run_command(cmd, quiet=True)
    if proc.returncode != 0:
        raise ParityRandomError(
            "C-проба не собралась (код "
            f"{proc.returncode}):\n{_tail(proc.stderr)}\n{_tail(proc.stdout)}"
        )
    if not binary.is_file():
        raise ParityRandomError(f"C-проба не найдена после сборки: {binary}")
    return binary


def run_probe(
    binary: Path, container: Path, tokens: Sequence[int]
) -> parity_real.ProbeResult:
    token_text = ",".join(str(int(token)) for token in tokens)
    proc = run_command([str(binary), str(container), token_text], quiet=True)
    if proc.returncode != 0:
        raise ParityRandomError(
            f"C-проба не отработала (код {proc.returncode}):\n"
            f"{_tail(proc.stderr)}"
        )
    return parity_real.parse_probe_output(proc.stdout)


def run_cli(vtm_bin: Path, container: Path, tokens: Sequence[int]) -> dict[str, Any]:
    """Штатный `vtm run --max-new 1`: проверяем, что CLI даёт тот же argmax."""
    token_text = ",".join(str(int(token)) for token in tokens)
    cmd = [
        str(vtm_bin),
        "run",
        str(container),
        "--tokens",
        token_text,
        "--max-new",
        "1",
    ]
    proc = run_command(cmd, quiet=True)
    if proc.returncode != 0:
        raise ParityRandomError(
            f"vtm run завершился кодом {proc.returncode}:\n{_tail(proc.stderr)}\n"
            f"{_tail(proc.stdout)}"
        )
    parsed = parity_real.parse_vtm_run_output(proc.stdout)
    if not parsed["generated"]:
        raise ParityRandomError(
            "vtm run не напечатал сгенерированный токен:\n" + _tail(proc.stdout)
        )
    parsed["argmax"] = int(parsed["generated"][0])
    return parsed


# ---------------------------------------------------------------------------
# Результат одного прогона
# ---------------------------------------------------------------------------


@dataclass
class ConfigResult:
    config: RandomConfig
    max_abs_delta: float | None = None
    argmax_reference: int | None = None
    argmax_runtime: int | None = None
    argmax_match: bool = False
    within_tolerance: bool = False
    cli_argmax: int | None = None
    cli_match: bool = False
    meta_problems: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0
    vtm_bytes: int | None = None
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def passed(self) -> bool:
        return bool(
            self.error is None
            and self.argmax_match
            and self.within_tolerance
            and self.cli_match
            and not self.meta_problems
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "max_abs_delta": (
                None if self.max_abs_delta is None else _json_float(self.max_abs_delta)
            ),
            "argmax_reference": self.argmax_reference,
            "argmax_runtime": self.argmax_runtime,
            "argmax_match": self.argmax_match,
            "within_tolerance": self.within_tolerance,
            "cli_argmax": self.cli_argmax,
            "cli_match": self.cli_match,
            "meta_problems": list(self.meta_problems),
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "vtm_bytes": self.vtm_bytes,
            "tolerance": self.tolerance,
            "passed": self.passed,
        }

    def failure_reasons(self) -> list[str]:
        if self.error is not None:
            return [f"ошибка прогона: {self.error}"]
        reasons: list[str] = []
        if not self.argmax_match:
            reasons.append(
                f"argmax: Python {self.argmax_reference}, C {self.argmax_runtime}"
            )
        if not self.within_tolerance:
            reasons.append(
                f"max|Δ| = {_fmt_delta(self.max_abs_delta)} >= "
                f"{self.tolerance:g}"
            )
        if not self.cli_match:
            reasons.append(
                f"CLI argmax {self.cli_argmax} != Python {self.argmax_reference}"
            )
        reasons.extend(self.meta_problems)
        return reasons


def check_config(
    config: RandomConfig,
    *,
    workdir: Path,
    probe: Path,
    env: Environment,
    tolerance: float = DEFAULT_TOLERANCE,
    keep: bool = False,
) -> ConfigResult:
    """Полный цикл для одной конфигурации: контейнер -> Python -> C -> CLI."""
    started = time.perf_counter()
    result = ConfigResult(config=config, tolerance=float(tolerance))
    container = workdir / f"model_{config.index:03d}.vtm"
    try:
        vtm = build_model(config)
        write_container(vtm, container, config)
        result.vtm_bytes = container.stat().st_size

        reference = vtm_ref.execute(vtm, list(config.tokens), 0)
        if reference.logits is None:  # pragma: no cover - защита от графа
            raise ParityRandomError("Python-эталон не дал логитов")
        probe_result = run_probe(probe, container, config.tokens)
        comparison = compare_logits(reference.logits, probe_result.logits, tolerance)
        result.max_abs_delta = comparison.max_abs_delta
        result.argmax_reference = comparison.argmax_reference
        result.argmax_runtime = comparison.argmax_runtime
        result.argmax_match = comparison.argmax_match
        result.within_tolerance = comparison.within_tolerance
        result.meta_problems = parity_real.check_runtime_meta(
            probe_result.meta,
            {
                "rope_theta": config.rope_theta,
                "rope_layout": config.rope_layout,
                "tie_word_embeddings": config.tie_word_embeddings,
            },
            len(reference.logits),
        )

        cli = run_cli(env.vtm_bin, container, config.tokens)
        result.cli_argmax = int(cli["argmax"])
        result.cli_match = result.cli_argmax == comparison.argmax_reference
    except Exception as exc:  # noqa: BLE001 - отчёт должен пережить любую ошибку
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        if result.passed and not keep:
            container.unlink(missing_ok=True)
    return result


# ---------------------------------------------------------------------------
# Сводка и отчёт
# ---------------------------------------------------------------------------


def summarize(
    *,
    seed: int,
    count: int,
    tolerance: float,
    results: Sequence[ConfigResult],
    workdir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Собрать JSON-сводку: counts, худшее расхождение, конфигурации."""
    usable = [item for item in results if item.max_abs_delta is not None]
    worst = max(usable, key=lambda item: item.max_abs_delta, default=None)
    errors = [item for item in results if item.error is not None]
    mismatches = [item for item in results if item.error is None and not item.argmax_match]
    outside = [
        item
        for item in results
        if item.error is None and not item.within_tolerance
    ]
    failed = [item for item in results if not item.passed]
    summary = {
        "tool": TOOL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "count": int(count),
        "tolerance": float(tolerance),
        "workdir": None if workdir is None else str(workdir),
        "total": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "errors": len(errors),
        "argmax_mismatches": len(mismatches),
        "outside_tolerance": len(outside),
        "worst": None if worst is None else _result_brief(worst),
        "failures": [_result_brief(item) for item in failed],
        "configs": [_result_record(item) for item in results],
        "ok": not failed,
    }
    return summary


def _result_brief(result: ConfigResult) -> dict[str, Any]:
    return {
        "index": result.config.index,
        "seed": result.config.seed,
        "config_id": result.config.config_id,
        "max_abs_delta": (
            None if result.max_abs_delta is None else _json_float(result.max_abs_delta)
        ),
        "tolerance": result.tolerance,
        "argmax_reference": result.argmax_reference,
        "argmax_runtime": result.argmax_runtime,
        "argmax_match": result.argmax_match,
        "within_tolerance": result.within_tolerance,
        "cli_argmax": result.cli_argmax,
        "cli_match": result.cli_match,
        "meta_problems": list(result.meta_problems),
        "error": result.error,
        "passed": result.passed,
        "reasons": result.failure_reasons(),
        "config": result.config.to_dict(),
    }


def _result_record(result: ConfigResult) -> dict[str, Any]:
    return {
        "config": result.config.to_dict(),
        "max_abs_delta": (
            None if result.max_abs_delta is None else _json_float(result.max_abs_delta)
        ),
        "argmax_reference": result.argmax_reference,
        "argmax_runtime": result.argmax_runtime,
        "argmax_match": result.argmax_match,
        "within_tolerance": result.within_tolerance,
        "cli_argmax": result.cli_argmax,
        "cli_match": result.cli_match,
        "meta_problems": list(result.meta_problems),
        "error": result.error,
        "elapsed_ms": round(result.elapsed_ms, 1),
        "vtm_bytes": result.vtm_bytes,
        "passed": result.passed,
    }


def format_config(config: RandomConfig) -> str:
    return (
        f"hidden={config.hidden} heads={config.n_heads} kv={config.n_kv_heads} "
        f"hd={config.head_dim} layers={config.n_layers} vocab={config.vocab} "
        f"inter={config.intermediate} theta={config.rope_theta:g} "
        f"rope={config.rope_layout} tied={int(config.tie_word_embeddings)} "
        f"block={config.scale_block}"
    )


def format_config_line(config: RandomConfig) -> str:
    """Однострочная запись для худшей/провалившейся конфигурации."""
    return f"индекс {config.index}, seed {config.seed}: {format_config(config)}"


def reproduction_command(config: RandomConfig) -> str:
    return (
        "python3 tools/parity_random.py "
        f"--seed {config.seed} --only {config.index}"
    )


def format_report(summary: dict[str, Any], *, verbose_configs: bool = True) -> str:
    """Человекочитаемая сводка (в stdout без `--json`)."""
    lines: list[str] = []
    lines.append("=== parity_random: случайный паритет C ↔ Python ===")
    lines.append(
        f"seed: {summary['seed']}, конфигураций: {summary['count']}, "
        f"порог max|Δ|: {summary['tolerance']:g}"
    )
    if summary.get("workdir"):
        lines.append(f"workdir: {summary['workdir']}")
    lines.append("")
    if verbose_configs:
        for index, record in enumerate(summary["configs"], start=1):
            config = record["config"]
            status = "ok " if record["passed"] else "FAIL"
            delta = _fmt_delta(
                None
                if record["max_abs_delta"] is None
                else (
                    record["max_abs_delta"]
                    if isinstance(record["max_abs_delta"], float)
                    else math.inf
                )
            )
            lines.append(
                f"[{index:>3}/{summary['count']}] {status} idx={config['index']:<3} "
                f"hidden={config['hidden']:<3} heads={config['n_heads']} "
                f"kv={config['n_kv_heads']} hd={config['head_dim']:<2} "
                f"layers={config['n_layers']} vocab={config['vocab']:<3} "
                f"theta={config['rope_theta']:g} tied={int(config['tie_word_embeddings'])} "
                f"block={config['scale_block']} max|Δ|={delta}"
            )
        lines.append("")
    lines.append("=== итог ===")
    lines.append(
        f"конфигураций: {summary['total']}, прошли: {summary['passed']}, "
        f"провалились: {summary['failed']}"
    )
    lines.append(f"расхождений argmax: {summary['argmax_mismatches']}")
    lines.append(f"вне порога max|Δ|: {summary['outside_tolerance']}")
    lines.append(f"ошибок прогона: {summary['errors']}")
    worst = summary.get("worst")
    if worst is None:
        lines.append("худшее расхождение: нет данных (все прогоны с ошибкой)")
    else:
        lines.append(
            f"худшее расхождение: max|Δ| = {_fmt_delta(_delta_from_json(worst['max_abs_delta']))} "
            f"(порог {summary['tolerance']:g}), индекс {worst['index']}"
        )
        lines.append(f"  конфигурация: {format_config_from_dict(worst['config'])}")
        lines.append(
            "  воспроизвести: "
            f"python3 tools/parity_random.py --seed {worst['seed']} --only {worst['index']}"
        )
    if summary["failures"]:
        lines.append("")
        lines.append("=== провалившиеся конфигурации ===")
        for record in summary["failures"]:
            config = record["config"]
            lines.append(
                f"idx={config['index']} seed={config['seed']}: "
                f"{format_config_from_dict(config)}"
            )
            for reason in record["reasons"]:
                lines.append(f"  - {reason}")
            lines.append(
                "  воспроизвести: "
                f"python3 tools/parity_random.py --seed {config['seed']} "
                f"--only {config['index']}"
            )
    lines.append("вердикт: " + ("PASS" if summary["ok"] else "FAIL"))
    return "\n".join(lines)


def format_config_from_dict(config: dict[str, Any]) -> str:
    return (
        f"hidden={config['hidden']} heads={config['n_heads']} "
        f"kv={config['n_kv_heads']} hd={config['head_dim']} "
        f"layers={config['n_layers']} vocab={config['vocab']} "
        f"inter={config['intermediate']} theta={config['rope_theta']:g} "
        f"rope={config['rope_layout']} tied={int(config['tie_word_embeddings'])} "
        f"block={config['scale_block']}"
    )


def _delta_from_json(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def format_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)


# ---------------------------------------------------------------------------
# Прогон
# ---------------------------------------------------------------------------


def parse_only(text: str) -> list[int]:
    """`--only 3,1` -> [3, 1]; порядок сохраняется, повторы отбрасываются."""
    indices: list[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            index = int(chunk)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"ожидался индекс конфигурации: {chunk!r}"
            ) from exc
        if index < 0:
            raise argparse.ArgumentTypeError(f"индекс должен быть >= 0: {index}")
        if index not in indices:
            indices.append(index)
    if not indices:
        raise argparse.ArgumentTypeError("--only: пустой список индексов")
    return indices


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось целое > 0: {text!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"ожидалось целое > 0: {text!r}")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось целое >= 0: {text!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"ожидалось целое >= 0: {text!r}")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось число > 0: {text!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"ожидалось число > 0: {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 tools/parity_random.py",
        description=(
            "Случайные архитектуры VTM: C (src/vtm/vtm) против Python-эталона "
            "(tools/vtm_ref.py)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  python3 tools/parity_random.py\n"
            "  python3 tools/parity_random.py --count 10 --seed 7\n"
            "  python3 tools/parity_random.py --json\n"
            "  python3 tools/parity_random.py --seed 0 --only 12\n\n"
            "Конфигурация зависит только от (seed, index): любую можно "
            "воспроизвести через --only.\n"
        ),
    )
    parser.add_argument(
        "--count", type=_positive_int, default=DEFAULT_COUNT,
        help=f"сколько конфигураций прогнать (по умолчанию {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--seed", type=_non_negative_int, default=DEFAULT_SEED,
        help=f"фиксированное зерно генератора (по умолчанию {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--tolerance", type=_positive_float, default=DEFAULT_TOLERANCE,
        help=f"порог max|Δ| логитов (по умолчанию {DEFAULT_TOLERANCE:g})",
    )
    parser.add_argument(
        "--only", type=parse_only, default=None,
        help="индексы конфигураций (например 12 или 3,7); --count игнорируется",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="печатать JSON-сводку в stdout (прогресс — в stderr)",
    )
    parser.add_argument(
        "--workdir", default=str(DEFAULT_WORKDIR),
        help=f"каталог контейнеров и C-пробы (по умолчанию {DEFAULT_WORKDIR})",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="не удалять контейнеры успешных прогонов",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> int:
    env = check_environment()
    missing = env.missing()
    if missing:
        raise ParityRandomError(
            "для прогона не хватает: "
            + ", ".join(missing)
            + "\nC-рантайм и библиотека собираются: make -C src/vtm"
        )

    indices = (
        list(args.only)
        if args.only is not None
        else list(range(int(args.count)))
    )
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{TOOL_NAME}] seed={args.seed}, конфигураций: {len(indices)}, "
        f"порог: {args.tolerance:g}, workdir: {workdir}",
        file=sys.stderr,
        flush=True,
    )
    probe = ensure_probe(workdir, env)
    print(f"[{TOOL_NAME}] C-проба: {probe}", file=sys.stderr, flush=True)

    results: list[ConfigResult] = []
    started = time.perf_counter()
    for position, index in enumerate(indices, start=1):
        config = generate_config(args.seed, index)
        problems = validate_config(config)
        if problems:  # pragma: no cover - генератор обязан давать валидные
            raise ParityRandomError(
                f"невалидная конфигурация idx={index}: " + "; ".join(problems)
            )
        result = check_config(
            config,
            workdir=workdir,
            probe=probe,
            env=env,
            tolerance=args.tolerance,
            keep=bool(args.keep),
        )
        results.append(result)
        status = "ok  " if result.passed else "FAIL"
        delta = _fmt_delta(result.max_abs_delta)
        print(
            f"[{TOOL_NAME}] [{position:>3}/{len(indices)}] {status} idx={index:<3} "
            f"{format_config(config)} max|Δ|={delta} "
            f"({result.elapsed_ms:.0f} мс)",
            file=sys.stderr,
            flush=True,
        )
        if result.error is not None:
            print(
                f"[{TOOL_NAME}]   ошибка: {result.error}",
                file=sys.stderr,
                flush=True,
            )

    summary = summarize(
        seed=args.seed,
        count=len(indices),
        tolerance=args.tolerance,
        results=results,
        workdir=workdir,
    )
    summary["total_ms"] = round((time.perf_counter() - started) * 1000.0, 1)

    if args.json:
        print(format_json(summary))
    else:
        print(format_report(summary, verbose_configs=False))
    print(
        f"[{TOOL_NAME}] готово за {summary['total_ms'] / 1000.0:.1f} с: "
        f"{summary['passed']}/{summary['total']} прошли",
        file=sys.stderr,
        flush=True,
    )
    return 0 if summary["ok"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:  # argparse уже напечатал ошибку
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        return run(args)
    except ParityRandomError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"\n[{TOOL_NAME}] прервано", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
