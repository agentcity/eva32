#!/usr/bin/env python3
"""Смешанная точность при холодном экспорте: политика по группам слоёв (0146).

Зачем
=====

Холодная третизация всех Linear на Qwen2.5-0.5B даёт PPL ×7,3e5 (0143):
текст нечитаем. Но слои не равны: FFN — это 2/3 параметров и он избыточнее,
attention компактнее, таблица словаря вообще особый случай. Гипотеза 0146:
если третить только FFN, а attention (и таблицу) держать точнее, текст может
остаться читаемым при файле в 2–2,5 раза меньше fp16.

Этот модуль экспортирует веса **с разной точностью по группам** и умеет
довести конфигурацию до чисел: экспорт → компиляция → PPL → декод.

Что физически умеет формат (и почему политика такая, а не шире)
==============================================================

Это не выбор инструмента, а свойство контейнера VTM v0.1 и рантайма:

* **Linear (attn/ffn)**: ``trit`` (1,585 бита, TMV по LUT243) либо плотный
  fp32/fp16. Плотный путь в C есть — ``op_tmv`` умеет ``W`` в fp32/fp16/bf16
  (``src/vtm/interp.c``), загрузчик разворачивает fp16/bf16 в fp32-арену
  (``src/vtm/loader.c``), Python-эталон — ``vtm_ref._tmv_dense``. INT8 для
  матриц Linear **не поддерживается**: ``op_tmv`` принимает только
  trit|fp32|fp16|bf16, поэтому «всё int8» собрать не из чего (int8 в формате
  живёт только как сегмент таблицы словаря, kind 4).
* **Таблица словаря (и tied lm_head)**: ``trit``, ``int8`` или ``fp32``.
  fp16 для неё невозможен при ``tie_word_embeddings``: tied-выход считается
  через ``MM`` по той же таблице, и ``vtm_compile.load_export`` отвергает
  F16-таблицу («нужен fp32, int8 или тритовая таблица»). Qwen2.5-0.5B — tied.

Одна честная оговорка про код: ``tools/vtm_compile.py`` (чужая зона) в
``matrix_tensor(dense=False)`` принимает для Linear только триты, хотя
рантайм умеет плотные. Поэтому здесь есть узкий контекстный патч
``allow_dense_linear_weights()``: он разрешает плотный fp-тензор на месте
тритового **только на время компиляции** и возвращает функцию как была.
Правильное место для этого — сам ``vtm_compile.py``; пока зона 0146 его не
трогает.

Политика
========

Строка вида ``embed=int8,attn=fp16,ffn=trit`` плюс необязательные
``first_last=...`` (слои 0 и N-1) и ``layers.N=...`` (конкретный слой,
перекрывает всё). Группы: ``embed`` (таблица + tied lm_head), ``attn``
(q/k/v/o), ``ffn`` (gate/up/down). Точности: Linear — ``trit|fp16|fp32``,
таблица — ``trit|int8|fp32``. Синонимы: ``ternary``, ``half``, ``float16``,
``float32``, ``none`` (=fp32).

Приоритет: ``layers.N`` > ``first_last`` > ``attn``/``ffn``.

Экспорт не дублирует формат: сегменты TRT1 собирает
``cold_export.encode_trit_segment``/``encode_embed_segment`` (тот же
``qat.ternary``, что QAT), плотные веса пишутся в ``non_ternary.bin``
(fp32/fp16 little-endian), манифест — ``qat.export.build_manifest``.
``tools/cold_export.py`` не правится.

Замер
=====

PPL меряется ровно как в 0143 — ``tools/parity_cold.py --ppl-lines 20
--ppl-max-tokens 128 --skip-parity`` (те же 20 строк ``data/train.jsonl``,
PPL по токенам ответа), поэтому числа сравнимы между задачами. bf16-эталон
печатается тем же прогоном. Декод — ``src/vtm/vtm run --text ... --chat``
(чат-шаблон Qwen2.5), 2 промпта, жадная генерация.

Запуск
======

::

    # одна конфигурация: экспорт -> компиляция -> PPL -> декод
    PYTHONPATH=.pylibs python3.13 tools/export_mixed.py \\
        --model .hfcache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<rev> \\
        --out-root export/mixed --name attn-fp16 \\
        --policy embed=int8,attn=fp16,ffn=trit

    # готовый набор конфигураций эксперимента (--suite) и таблица
    PYTHONPATH=.pylibs python3.13 tools/export_mixed.py --model ... \\
        --out-root export/mixed --suite
    python3 tools/export_mixed.py --out-root export/mixed --report

    # только собрать контейнер, без PPL/декода
    PYTHONPATH=.pylibs python3.13 tools/export_mixed.py --model ... \\
        --out-root /tmp/mix --name probe --policy attn=fp16 --no-measure

Коды возврата: 0 — всё, что просили, сделано; 2 — ошибка аргументов/экспорта.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _path in (str(_ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from qat import export as qat_export  # noqa: E402
from qat import ternary  # noqa: E402

try:  # импорт как пакета (tests/*) или запуск как скрипта
    from tools import cold_export, vtm_compile, vtm_ref  # noqa: E402
except ImportError:  # pragma: no cover - `python3 tools/export_mixed.py`
    import cold_export  # type: ignore[no-redef]  # noqa: E402
    import vtm_compile  # type: ignore[no-redef]  # noqa: E402
    import vtm_ref  # type: ignore[no-redef]  # noqa: E402

PROJECT_ROOT = _ROOT
TOOL_VERSION = 1

#: Группы политики.
GROUP_EMBED = "embed"
GROUP_ATTN = "attn"
GROUP_FFN = "ffn"
GROUP_NAMES = (GROUP_EMBED, GROUP_ATTN, GROUP_FFN)
#: Ключи матриц attention и FFN (порядок как в cold_export.TARGET_SUFFIXES).
ATTENTION_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN_KEYS = ("gate_proj", "up_proj", "down_proj")
#: Точности Linear: триты или плотный fp (int8 для Linear рантайм не умеет).
MATRIX_DTYPES = ("trit", "fp16", "fp32")
#: Точности таблицы словаря: fp16 не проходит через tied-MM компилятора.
EMBED_DTYPES = ("trit", "int8", "fp32")
#: Синонимы в политике: то, что удобно писать руками.
DTYPE_ALIASES = {
    "trit": "trit", "trits": "trit", "ternary": "trit", "t1.58": "trit",
    "int8": "int8", "i8": "int8",
    "fp16": "fp16", "float16": "fp16", "half": "fp16", "f16": "fp16",
    "fp32": "fp32", "float32": "fp32", "f32": "fp32",
    "none": "fp32", "dense": "fp32",
}
#: bf16-эталон PPL из 0143 (те же 20 строк, тот же способ замера).
BF16_PPL = 16.13
#: Замер как в 0143: иначе числа несравнимы.
PPL_LINES = 20
PPL_MAX_TOKENS = 128
#: Два промпта декода — один факт и одна инструкция, оба через chat-template.
DECODE_PROMPTS = (
    "Столица Франции — это",
    "Напиши на Python функцию, которая считает сумму чисел от 1 до n.",
)
DEFAULT_DECODE_MAX_NEW = 32
#: Где искать tokenizer.json внутри каталога HF-модели.
TOKENIZER_FILE = "tokenizer.json"
#: Набор конфигураций эксперимента 0146: от эталона к «пределу точности».
#: ``all-trit`` — база 0143, ``attn-fp16`` — гипотеза задачи,
#: ``ffn-fp16`` — обратное распределение (где на самом деле больно),
#: ``first-last-fp16`` — совет «не квантовать крайние слои»,
#: ``all-fp16`` — потолок (проверка, что смешанная сборка вообще честная),
#: ``table-int8`` — вклад таблицы при тритовых Linear.
SUITE: tuple[tuple[str, str], ...] = (
    ("all-trit", "embed=trit,attn=trit,ffn=trit"),
    ("attn-fp16", "embed=int8,attn=fp16,ffn=trit"),
    ("ffn-fp16", "embed=int8,attn=trit,ffn=fp16"),
    ("first-last-fp16", "embed=int8,attn=trit,ffn=trit,first_last=fp16"),
    ("all-fp16", "embed=int8,attn=fp16,ffn=fp16"),
    ("table-int8", "embed=int8,attn=trit,ffn=trit"),
)


class MixedExportError(RuntimeError):
    """Ошибка политики/экспорта/замера: код возврата 2, без трейсбека."""


# ---------------------------------------------------------------------------
# Политика
# ---------------------------------------------------------------------------


def normalize_dtype(text: str, *, group: str) -> str:
    """Привести написание точности к каноническому и проверить группу."""
    key = str(text).strip().lower()
    if key not in DTYPE_ALIASES:
        raise MixedExportError(
            f"точность {text!r}: поддержано {', '.join(sorted(DTYPE_ALIASES))}"
        )
    dtype = DTYPE_ALIASES[key]
    allowed = EMBED_DTYPES if group == GROUP_EMBED else MATRIX_DTYPES
    if dtype not in allowed:
        if group == GROUP_EMBED and dtype == "fp16":
            raise MixedExportError(
                "fp16 для таблицы словаря невозможен: при "
                "tie_word_embeddings=true tied lm_head считается через MM по "
                "этой же таблице, и vtm_compile принимает только fp32, int8 "
                "или триты (Qwen2.5-0.5B — tied). Варианты: embed=trit|int8|fp32"
            )
        raise MixedExportError(
            f"точность {dtype!r} не для группы {group}: поддержано "
            f"{', '.join(allowed)}"
        )
    return dtype


@dataclass
class Policy:
    """Какая точность у какой группы слоёв.

    ``embed`` — таблица словаря и tied lm_head; ``attn`` — q/k/v/o;
    ``ffn`` — gate/up/down. ``first_last`` перекрывает слои 0 и N-1,
    ``layers`` — конкретные слои (перекрывает всё остальное).
    """

    embed: str = "trit"
    attn: str = "trit"
    ffn: str = "trit"
    first_last: str | None = None
    layers: dict[int, str] = field(default_factory=dict)
    text: str = ""

    # -- разбор ------------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> "Policy":
        policy = cls(text=str(text).strip())
        if not policy.text:
            return policy
        seen: set[str] = set()
        for item in policy.text.split(","):
            chunk = item.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise MixedExportError(
                    f"политика {text!r}: элемент {chunk!r} без '='; пример — "
                    "embed=int8,attn=fp16,ffn=trit"
                )
            key, value = (part.strip() for part in chunk.split("=", 1))
            bare = key.lower().replace("-", "_")
            if bare in ("embed", "embeddings", "table", "lm_head"):
                if "embed" in seen:
                    raise MixedExportError(f"политика {text!r}: embed задан дважды")
                seen.add("embed")
                policy.embed = normalize_dtype(value, group=GROUP_EMBED)
            elif bare in ("attn", "attention"):
                if "attn" in seen:
                    raise MixedExportError(f"политика {text!r}: attn задан дважды")
                seen.add("attn")
                policy.attn = normalize_dtype(value, group=GROUP_ATTN)
            elif bare in ("ffn", "mlp"):
                if "ffn" in seen:
                    raise MixedExportError(f"политика {text!r}: ffn задан дважды")
                seen.add("ffn")
                policy.ffn = normalize_dtype(value, group=GROUP_FFN)
            elif bare in ("first_last", "firstlast", "edges", "first_and_last"):
                policy.first_last = normalize_dtype(value, group=GROUP_ATTN)
            elif bare.startswith(("layers.", "layer.")):
                number = bare.split(".", 1)[1]
                if not number.isdigit():
                    raise MixedExportError(
                        f"политика {text!r}: номер слоя в {key!r} не число"
                    )
                index = int(number)
                if index in policy.layers:
                    raise MixedExportError(
                        f"политика {text!r}: слой {index} задан дважды"
                    )
                policy.layers[index] = normalize_dtype(value, group=GROUP_ATTN)
            else:
                raise MixedExportError(
                    f"политика {text!r}: неизвестный ключ {key!r}; поддержано "
                    "embed, attn, ffn, first_last, layers.N"
                )
        return policy

    # -- применение --------------------------------------------------------
    def dtype_for_layer(self, layer: int, group: str, n_layers: int) -> str:
        """Точность группы группы ``group`` у слоя ``layer``."""
        if group == GROUP_EMBED:
            return self.embed
        if layer in self.layers:
            return self.layers[layer]
        if self.first_last is not None and layer in (0, n_layers - 1):
            return self.first_last
        return self.attn if group == GROUP_ATTN else self.ffn

    def group_of(self, key: str) -> str:
        """Группа матрицы Linear по короткому ключу (q_proj, gate_proj, …)."""
        if key in ATTENTION_KEYS:
            return GROUP_ATTN
        if key in FFN_KEYS:
            return GROUP_FFN
        raise MixedExportError(f"матрица {key!r} вне групп attn/ffn")

    def dtype_for_matrix(self, key: str, layer: int, n_layers: int) -> str:
        return self.dtype_for_layer(layer, self.group_of(key), n_layers)

    # -- представление -----------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "embed": self.embed,
            "attn": self.attn,
            "ffn": self.ffn,
            "first_last": self.first_last,
            "layers": {str(k): v for k, v in sorted(self.layers.items())},
        }

    def to_text(self) -> str:
        """Каноническая строка политики (для манифеста и отчётов)."""
        parts = [f"embed={self.embed}", f"attn={self.attn}", f"ffn={self.ffn}"]
        if self.first_last is not None:
            parts.append(f"first_last={self.first_last}")
        for layer, dtype in sorted(self.layers.items()):
            parts.append(f"layers.{layer}={dtype}")
        return ",".join(parts)


# ---------------------------------------------------------------------------
# Плотные (не тритовые) веса
# ---------------------------------------------------------------------------


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "numel") and hasattr(value, "shape")


def encode_dense_payload(weight: Any, dtype: str) -> bytes:
    """Байты плотного веса little-endian: fp32 или fp16 (без torch в тестах)."""
    if dtype not in ("fp32", "fp16"):
        raise MixedExportError(f"плотная выгрузка: dtype={dtype!r}, ждали fp32|fp16")
    n_values = int(weight.numel())
    item, fmt = (4, "f") if dtype == "fp32" else (2, "e")
    if _is_torch_tensor(weight):
        import torch  # noqa: PLC0415 - только на настоящих весах

        out = weight.detach().to("cpu")
        if dtype == "fp32":
            out = out.float()
            payload = qat_export._to_numpy_bytes(out, "<f4")
        else:
            out = out.to(torch.float16)
            if not bool(torch.isfinite(out).all()):
                raise MixedExportError("fp16: в весе есть значения вне диапазона fp16")
            payload = qat_export._to_numpy_bytes(out, "<f2")
        del out
    else:
        values = [float(v) for v in weight]
        if dtype == "fp16":
            bad = [v for v in values if not math.isfinite(v) or abs(v) > 65504.0]
            if bad:
                raise MixedExportError(
                    f"fp16: {len(bad)} значений вне диапазона fp16 (например {bad[0]:g})"
                )
        payload = struct.pack(f"<{len(values)}{fmt}", *values)
    expected = n_values * item
    if len(payload) != expected:
        raise MixedExportError(
            f"плотная выгрузка: {len(payload)} байт, ожидалось {expected}"
        )
    return payload


def _dense_entry(name: str, role: Any, shape: Sequence[int], dtype: str,
                 offset: int, payload: bytes, extra: dict[str, Any] | None = None
                 ) -> dict[str, Any]:
    """Запись ``non_ternary.tensors[]`` для плотной матрицы/таблицы."""
    entry: dict[str, Any] = {
        "name": name,
        "role": role.kind,
        "layer": role.layer,
        "key": role.key,
        "shape": [int(x) for x in shape],
        "dtype": dtype,
        "kind": dtype,
        "storage": "non_ternary.bin",
        "offset": offset,
        "length": len(payload),
        "params": int(math.prod(int(x) for x in shape)),
    }
    if extra:
        entry.update(extra)
    return entry


# ---------------------------------------------------------------------------
# Экспорт
# ---------------------------------------------------------------------------


@dataclass
class MixedExport:
    """Что экспорт положил на диск и чем это оплачено."""

    out_dir: Path
    manifest_path: Path
    trit_path: Path
    non_ternary_path: Path
    manifest: dict[str, Any]
    architecture: dict[str, Any]
    policy: Policy
    elapsed_s: float
    peak_rss_bytes: int
    trit_params: int
    dense_params: int
    group_params: dict[str, int]
    group_bytes: dict[str, int]
    dtype_params: dict[str, int]

    @property
    def export_bytes(self) -> int:
        return self.trit_path.stat().st_size + self.non_ternary_path.stat().st_size


def export_mixed_model(
    source: Any,
    hf_config: dict[str, Any],
    out_dir: Path | str,
    *,
    policy: Policy,
    model_name: str | None = None,
    revision: str | None = None,
    block_size: int = ternary.BLOCK_SIZE,
    scale_dtype: str = "float32",
    embed_scale: str = "block",
    context_window: int = cold_export.DEFAULT_CONTEXT_WINDOW,
    limit_layers: int | None = None,
    overrides: dict[str, Any] | None = None,
    force: bool = False,
    echo: Any = None,
) -> MixedExport:
    """Экспортировать источник весов с по-слойной точностью.

    Один проход по каноническому порядку (``cold_export.canonical_order``):
    матрицы с политикой ``trit`` уходят сегментами TRT1 в ``ternary.trit``,
    с ``fp16``/``fp32`` — плотными записями в ``non_ternary.bin``; таблица
    словаря — ``trit``/``int8``/``fp32``; нормы и bias всегда fp32.
    """
    started = time.time()
    if block_size != ternary.BLOCK_SIZE:
        raise MixedExportError(
            f"--block-size={block_size}: формат VTM v0.1 хранит масштаб только "
            f"на {ternary.BLOCK_SIZE} тритов"
        )
    if scale_dtype not in cold_export.SCALE_DTYPES:
        raise MixedExportError(f"--scale-dtype={scale_dtype}: {cold_export.SCALE_DTYPES}")
    if embed_scale not in cold_export.EMBED_SCALES:
        raise MixedExportError(f"--embed-scale={embed_scale}: {cold_export.EMBED_SCALES}")
    if context_window <= 0:
        raise MixedExportError(f"--context-window={context_window}: нужно > 0")
    cold_export.check_supported_config(hf_config)

    # fp32 таблицы в терминах компилятора — "none" (плотный эмбеддинг).
    arch_embed = "none" if policy.embed == "fp32" else policy.embed
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    shapes = source.shapes()
    architecture = cold_export.infer_export_architecture(
        shapes, hf_config,
        limit_layers=limit_layers,
        context_window=context_window,
        embed_mode=arch_embed,
        overrides=overrides,
    )
    cold_export.check_block_alignment(architecture, block_size, arch_embed, embed_scale)
    n_layers = int(architecture["n_layers"])

    # --- какие ключи берём --------------------------------------------------
    roles: dict[str, Any] = {}
    ignored: list[str] = []
    for key in source.keys():
        if key.endswith(cold_export.IGNORED_KEY_SUFFIXES):
            ignored.append(key)
            continue
        role = cold_export._role_of(key)
        if role.kind == "unknown":
            ignored.append(key)
            continue
        if role.layer is not None and role.layer >= n_layers:
            continue  # обрезано --limit-layers
        roles[key] = role
    if ignored and echo is not None:
        echo(
            f"[mixed] пропущено ключей вне графа VTM: {len(ignored)} "
            f"({', '.join(ignored[:3])}{'…' if len(ignored) > 3 else ''})"
        )
    order = cold_export.canonical_order(roles)
    if not order:
        raise MixedExportError("в шардах не нашлось ни одного тензора Qwen/Llama-слоя")

    trit_path = out_path / "ternary.trit"
    non_ternary_path = out_path / qat_export.NON_TERNARY_FILE
    manifest_path = out_path / cold_export.MANIFEST_FILE
    if not force:
        existing = [
            path.name for path in (trit_path, non_ternary_path, manifest_path)
            if path.exists()
        ]
        if existing:
            raise MixedExportError(
                f"в {out_path} уже есть экспорт ({', '.join(existing)}); "
                "добавьте --force, чтобы перезаписать"
            )

    tmp_trit = out_path / (".ternary.trit.tmp")
    tmp_non = out_path / ("." + qat_export.NON_TERNARY_FILE + ".tmp")
    trit_entries: list[dict[str, Any]] = []
    non_entries: list[dict[str, Any]] = []
    trit_params = 0
    dense_params = 0
    non_params = 0
    group_params: dict[str, int] = {name: 0 for name in GROUP_NAMES}
    group_bytes: dict[str, int] = {name: 0 for name in GROUP_NAMES}
    dtype_params: dict[str, int] = {}
    embed_entry: dict[str, Any] | None = None
    seen_layers: dict[int, set[str]] = {}
    have_embed = False
    have_final_norm = False
    total = len(order)
    step_count = 0

    def _count(dtype: str, params: int, group: str, size: int) -> None:
        dtype_params[dtype] = dtype_params.get(dtype, 0) + params
        group_params[group] += params
        group_bytes[group] += size

    with tmp_trit.open("wb") as trit_handle, tmp_non.open("wb") as non_handle:
        trit_offset = 0
        non_offset = 0
        for index, key in enumerate(order, start=1):
            role = roles[key]
            value = source.tensor(key)
            shape = [int(x) for x in value.shape]
            if role.kind == "matrix":
                dtype = policy.dtype_for_matrix(role.key, int(role.layer), n_layers)
                if dtype == "trit":
                    data, meta = cold_export.encode_trit_segment(
                        value, key, block_size=block_size, scale_dtype=scale_dtype,
                    )
                    meta.update({
                        "name": key,
                        "role": "matrix",
                        "layer": role.layer,
                        "key": role.key,
                        "group": policy.group_of(role.key),
                        "trit_offset": trit_offset,
                        "trit_length": len(data),
                    })
                    trit_handle.write(data)
                    _count("trit", int(meta["params"]), meta["group"], len(data))
                    trit_offset += len(data)
                    trit_entries.append(meta)
                    trit_params += int(meta["params"])
                else:
                    payload = encode_dense_payload(value, dtype)
                    entry = _dense_entry(
                        key, role, shape,
                        "float16" if dtype == "fp16" else "float32",
                        non_offset, payload,
                        {"group": policy.group_of(role.key), "mixed": True},
                    )
                    non_handle.write(payload)
                    non_entries.append(entry)
                    _count(dtype, int(entry["params"]), entry["group"], len(payload))
                    non_offset += len(payload)
                    dense_params += int(entry["params"])
                    non_params += int(entry["params"])
                seen_layers.setdefault(int(role.layer), set()).add(role.key)
            elif role.kind == "embed":
                if policy.embed == "trit":
                    data, meta = cold_export.encode_embed_segment(
                        value, qat_export.EMBED_TENSOR_NAME, mode="trit",
                        scale_mode=embed_scale, block_size=block_size,
                        scale_dtype=scale_dtype,
                    )
                    meta.update({
                        "name": qat_export.EMBED_TENSOR_NAME,
                        "role": "embed",
                        "layer": None,
                        "key": "embed",
                        "group": GROUP_EMBED,
                        "trit_offset": trit_offset,
                        "trit_length": len(data),
                    })
                    trit_handle.write(data)
                    _count("trit", int(meta["params"]), GROUP_EMBED, len(data))
                    trit_offset += len(data)
                    trit_entries.append(meta)
                    trit_params += int(meta["params"])
                    embed_entry = meta
                elif policy.embed == "int8":
                    data, meta = cold_export.encode_embed_segment(
                        value, qat_export.EMBED_TENSOR_NAME, mode="int8",
                        scale_mode=embed_scale, block_size=block_size,
                        scale_dtype=scale_dtype,
                    )
                    non_handle.write(data)
                    entry = {
                        "name": qat_export.EMBED_TENSOR_NAME,
                        "role": "embed",
                        "layer": None,
                        "key": "embed",
                        "shape": shape,
                        "dtype": "int8",
                        "kind": "int8",
                        "storage": "non_ternary.bin",
                        "offset": non_offset,
                        "length": len(data),
                        "params": int(meta["params"]),
                        **{k: v for k, v in meta.items() if k not in ("shape", "kind")},
                    }
                    non_entries.append(entry)
                    _count("int8", int(entry["params"]), GROUP_EMBED, len(data))
                    non_offset += len(data)
                    non_params += int(entry["params"])
                    embed_entry = entry
                else:  # fp32
                    payload = encode_dense_payload(value, "fp32")
                    entry = _dense_entry(
                        qat_export.EMBED_TENSOR_NAME, role, shape, "float32",
                        non_offset, payload, {"group": GROUP_EMBED},
                    )
                    non_handle.write(payload)
                    non_entries.append(entry)
                    _count("fp32", int(entry["params"]), GROUP_EMBED, len(payload))
                    non_offset += len(payload)
                    non_params += int(entry["params"])
                    embed_entry = entry
                have_embed = True
            elif role.kind in ("final_norm", "norm", "bias", "lm_head"):
                if role.kind == "lm_head" and architecture["tie_word_embeddings"]:
                    pass  # tied: байт не пишем, ссылку добавим ниже
                else:
                    payload = encode_dense_payload(value, "fp32")
                    entry = _dense_entry(
                        key, role, shape, "float32", non_offset, payload,
                    )
                    non_handle.write(payload)
                    non_entries.append(entry)
                    non_offset += len(payload)
                    non_params += int(entry["params"])
                    if role.kind == "final_norm":
                        have_final_norm = True
            else:  # pragma: no cover - роли выше перечислены
                raise MixedExportError(f"{key}: неожиданная роль {role.kind!r}")
            step_count += 1
            if echo is not None and (index % 32 == 0 or index == total):
                echo(
                    f"[mixed] {index}/{total} тензоров, тритов {trit_params:,}, "
                    f"плотных параметров {dense_params:,}, "
                    f"RSS {cold_export._human_bytes(cold_export._peak_rss_bytes())}"
                )
            source.release(key)
            del value
    tmp_trit.replace(trit_path)
    tmp_non.replace(non_ternary_path)

    # --- tied lm_head: ссылка на таблицу ------------------------------------
    if architecture["tie_word_embeddings"]:
        if not have_embed or embed_entry is None:
            raise MixedExportError(
                "tie_word_embeddings=true, но model.embed_tokens.weight не найден"
            )
        tied_dtype = {"trit": "trit", "int8": "int8"}.get(policy.embed, "float32")
        non_entries.append({
            "name": "lm_head.weight",
            "role": "lm_head",
            "layer": None,
            "key": "lm_head",
            "shape": list(embed_entry["shape"]),
            "dtype": tied_dtype,
            "tied_to": qat_export.EMBED_TENSOR_NAME,
            "params": int(embed_entry["params"]),
        })

    # --- полнота графа ------------------------------------------------------
    if not have_embed:
        raise MixedExportError(
            "в экспорте нет model.embed_tokens.weight: без таблицы словаря "
            "компилятор VTM не соберёт граф"
        )
    if not have_final_norm:
        raise MixedExportError("в экспорте нет model.norm.weight")
    for layer in range(n_layers):
        present = seen_layers.get(layer, set())
        missing = [key for key in cold_export.TARGET_SUFFIXES if key not in present]
        if missing:
            raise MixedExportError(
                f"слой {layer}: нет матриц {', '.join(missing)} — политика "
                "потеряла тензор, шарды неполные или имена не Qwen/Llama"
            )

    # --- манифест -----------------------------------------------------------
    files = qat_export.WeightFiles(
        non_ternary_path=non_ternary_path,
        non_ternary_entries=non_entries,
        non_ternary_params=non_params,
        trit_path=trit_path,
        tensor_entries=trit_entries,
        trit_params=trit_params,
        payload={},
        embed=embed_entry if policy.embed == "trit" else None,
        embed_mode=arch_embed,
    )
    quant_cfg = {
        "block_size": block_size,
        "scale_dtype": scale_dtype,
        "target_suffixes": list(cold_export.TARGET_SUFFIXES),
        "quantize_embeddings": arch_embed,
        "embed_scale": embed_scale,
        "rule": qat_export.QUANT_RULE,
        "mixed_policy": policy.describe(),
    }
    manifest = qat_export.build_manifest(
        files=files,
        architecture=architecture,
        config={"source": "mixed_export", "model": {"name_or_path": model_name},
                "policy": policy.to_text()},
        quant_cfg=quant_cfg,
        block_size=block_size,
        scale_dtype=scale_dtype,
        run_dir=model_name or "<mixed>",
        ckpt=model_name or "<mixed>",
        step=0,
        samples_seen=0,
        model_cfg={"name_or_path": model_name, "revision": revision},
        file_stats={},
    )
    source_info = source.describe()
    peak_rss = cold_export._peak_rss_bytes()
    elapsed = time.time() - started
    manifest["training"] = {
        "run": None,
        "run_dir": None,
        "checkpoint": None,
        "step": None,
        "samples_seen": None,
        "trained": False,
        "kind": "mixed_export",
        "note": "веса взяты из HF safetensors как есть; обучения не было",
        "config": {"source": "mixed_export", "policy": policy.to_text()},
    }
    manifest["non_ternary"]["source"] = "hf-safetensors"
    manifest["base_model"] = {
        "name_or_path": model_name,
        "revision": revision,
        "dir": source_info.get("dir"),
    }
    manifest["context_window"] = int(architecture["context_len"])
    manifest["source"] = {
        "kind": "mixed_export",
        "tool": "tools/export_mixed.py",
        "tool_version": TOOL_VERSION,
        "task": "0146",
        **source_info,
    }
    manifest["hf_config"] = hf_config

    # Эмбеддинги: qat.export знает только trit/fp32, int8 и посрочные
    # масштабы описываем здесь (как cold_export для 0143).
    embed_dtype = {"trit": "trit", "int8": "int8"}.get(policy.embed, "float32")
    manifest["embeddings"] = {
        "mode": arch_embed,
        "tensor": qat_export.EMBED_TENSOR_NAME,
        "storage": "ternary.trit" if policy.embed == "trit" else qat_export.NON_TERNARY_FILE,
        "dtype": embed_dtype,
        "params": int(embed_entry["params"]) if embed_entry else 0,
        "bytes": int(
            embed_entry.get("trit_length", embed_entry.get("length", 0))
        ) if embed_entry else 0,
        "block_size": block_size,
        "scale_mode": embed_scale,
        "scale_block": (
            vtm_ref.INT8_SCALE_BLOCK if policy.embed == "int8" else block_size
        ),
        "rule": qat_export.QUANT_RULE,
        "policy": policy.embed,
    }
    # Плотные fp16-матрицы лежат в non_ternary.bin: секционная подпись
    # "float32" из build_manifest была бы неправдой.
    dense_dtypes = sorted({str(entry["dtype"]) for entry in non_entries
                           if entry.get("mixed")})
    _dense_matrix_entries = [entry for entry in non_entries if entry.get("mixed")]
    manifest["non_ternary"]["dtype"] = (
        "mixed: float32" + (f" + {', '.join(dense_dtypes)}" if dense_dtypes else "")
    )
    manifest["non_ternary"]["mixed_dtypes"] = dense_dtypes
    manifest["non_ternary"]["note"] = (
        "нормы/bias — fp32; матрицы Linear с политикой fp16/fp32 и таблица "
        "словаря с embed=fp32 — плотными записями (dtype в tensors[]); "
        "тритовые матрицы лежат в ternary.trit"
    )
    manifest["mixed_precision"] = {
        "tool_version": TOOL_VERSION,
        "policy": policy.describe(),
        "policy_text": policy.to_text(),
        "groups": {
            name: {
                "params": int(group_params[name]),
                "bytes": int(group_bytes[name]),
                "share_of_params": round(
                    group_params[name] / max(1, sum(group_params.values())), 4
                ),
            }
            for name in GROUP_NAMES
        },
        "params_by_dtype": {k: int(v) for k, v in sorted(dtype_params.items())},
        "dense_matrix_entries": len(_dense_matrix_entries),
        "supported": {
            "linear": list(MATRIX_DTYPES),
            "embed": list(EMBED_DTYPES),
            "why_no_linear_int8": (
                "op_tmv в src/vtm/interp.c принимает W только как trit|fp32|"
                "fp16|bf16; int8 в формате есть лишь как сегмент таблицы "
                "словаря (kind 4)"
            ),
            "why_embed_no_fp16": (
                "tied lm_head считается через MM по той же таблице, "
                "vtm_compile.load_export принимает fp32/int8/trit"
            ),
        },
        "compiler_shim": (
            "плотные fp-матрицы Linear собираются через "
            "allow_dense_linear_weights(): vtm_compile.matrix_tensor(dense=False) "
            "пропускает для Linear только триты, хотя op_tmv умеет плотный W"
        ),
        "elapsed_s": round(elapsed, 1),
        "memory": {
            "peak_rss_bytes": peak_rss,
            "peak_rss": cold_export._human_bytes(peak_rss),
        },
    }
    qat_export.write_json_atomic(manifest_path, manifest)
    return MixedExport(
        out_dir=out_path,
        manifest_path=manifest_path,
        trit_path=trit_path,
        non_ternary_path=non_ternary_path,
        manifest=manifest,
        architecture=architecture,
        policy=policy,
        elapsed_s=elapsed,
        peak_rss_bytes=peak_rss,
        trit_params=trit_params,
        dense_params=dense_params,
        group_params=group_params,
        group_bytes=group_bytes,
        dtype_params=dtype_params,
    )


# ---------------------------------------------------------------------------
# Компиляция: плотные fp-матрицы Linear
# ---------------------------------------------------------------------------

_DENSE_KINDS = (vtm_ref.KIND_F32, vtm_ref.KIND_F16, vtm_ref.KIND_BF16)


@contextlib.contextmanager
def allow_dense_linear_weights() -> Iterator[None]:
    """Разрешить ``vtm_compile`` класть в Linear плотный fp16/fp32-тензор.

    ``vtm_compile.matrix_tensor(dense=False)`` (его зовёт ``compile_layer``
    для q/k/v/o/gate/up/down) принимает только триты. Рантайм при этом умеет
    плотный ``W``: ``op_tmv`` в C и ``_tmv_dense`` в Python-эталоне. Патч
    узкий (только ``not dense`` + плотный ``vtm_ref.Tensor``), живёт на время
    вызова и возвращает исходную функцию. Правильное место правки —
    ``tools/vtm_compile.py``; зона 0146 его не трогает.
    """
    original = vtm_compile.matrix_tensor

    def patched(graph: Any, name: str, rows: int, cols: int, spec: Any,
                dense: bool = False) -> int:
        if not dense and isinstance(spec, vtm_ref.Tensor) and spec.kind in _DENSE_KINDS:
            if tuple(int(x) for x in spec.dims) != (rows, cols):
                raise ValueError(
                    f"вес {name!r}: форма {tuple(spec.dims)}, ожидалась {(rows, cols)}"
                )
            spec.name = name
            return graph.add(spec)
        return original(graph, name, rows, cols, spec, dense)

    vtm_compile.matrix_tensor = patched  # type: ignore[assignment]
    try:
        yield
    finally:
        vtm_compile.matrix_tensor = original  # type: ignore[assignment]


def compile_mixed(manifest_path: Path | str, out_path: Path | str) -> dict[str, Any]:
    """Собрать ``.vtm`` из смешанного манифеста и вернуть его характеристики."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export = vtm_compile.load_export(manifest_path)
    with allow_dense_linear_weights():
        vtm = vtm_compile.compile_model(export.cfg, export.weights, export.meta)
    vtm_ref.write_vtm(out_path, vtm.tensors, vtm.program, vtm.meta, integrity=True)
    kinds: dict[str, int] = {}
    bytes_by_kind: dict[str, int] = {}
    for tensor in vtm.tensors:
        name = tensor.kind_name
        kinds[name] = kinds.get(name, 0) + 1
        bytes_by_kind[name] = bytes_by_kind.get(name, 0) + vtm_ref.tensor_payload_size(tensor)
    return {
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "tensors": len(vtm.tensors),
        "instructions": len(vtm.program),
        "kinds": kinds,
        "bytes_by_kind": bytes_by_kind,
        "trit_params": int(export.trit_params),
        "fp_params": int(export.fp_params),
    }


# ---------------------------------------------------------------------------
# Замер PPL и декод
# ---------------------------------------------------------------------------


def _tool_env() -> dict[str, str]:
    """PYTHONPATH для дочерних процессов: репозиторий + .pylibs (+ текущий)."""
    env = dict(os.environ)
    parts = [str(_ROOT)]
    if (_ROOT / ".pylibs").is_dir():
        parts.append(str(_ROOT / ".pylibs"))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _tail(text: str, lines: int = 20) -> str:
    chunks = (text or "").strip().splitlines()
    return "\n".join(chunks[-lines:])


def measure_ppl(
    model_dir: Path | str,
    vtm_path: Path | str,
    *,
    workdir: Path | str,
    json_path: Path | str | None = None,
    ppl_lines: int = PPL_LINES,
    ppl_max_tokens: int = PPL_MAX_TOKENS,
    skip_original: bool = False,
    timeout: float | None = None,
    echo: Any = None,
) -> dict[str, Any]:
    """PPL через ``tools/parity_cold.py`` ровно как в 0143.

    Команда: ``--ppl-lines 20 --ppl-max-tokens 128 --skip-parity``. Число,
    которое нас интересует, — ``perplexity.vtm_c.perplexity``: PPL C-рантайма
    на собранном ``.vtm`` (тот же артефакт, что поедет в прод). Код возврата
    parity_cold бывает 1 (hf_ternary-эталон для смешанной модели не совпадает
    с контейнером — это ожидаемо), поэтому смотрим на JSON, а не на код.
    """
    model_path = Path(model_dir).expanduser().resolve()
    vtm_file = Path(vtm_path).expanduser().resolve()
    work_path = Path(workdir).expanduser().resolve()
    work_path.mkdir(parents=True, exist_ok=True)
    json_file = (
        Path(json_path).expanduser().resolve()
        if json_path is not None else work_path / "parity.json"
    )
    cmd = [
        sys.executable, str(_ROOT / "tools" / "parity_cold.py"),
        "--model", str(model_path),
        "--vtm", str(vtm_file),
        "--workdir", str(work_path),
        "--json", str(json_file),
        "--ppl-lines", str(int(ppl_lines)),
        "--ppl-max-tokens", str(int(ppl_max_tokens)),
        "--skip-parity",
    ]
    if skip_original:
        cmd.append("--skip-original")
    if echo is not None:
        echo(f"[mixed] PPL: {' '.join(cmd[1:])}")
    started = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=_tool_env(), cwd=str(_ROOT),
            timeout=timeout,
        )
    except OSError as exc:
        raise MixedExportError(f"parity_cold не запускается: {exc}") from exc
    elapsed = time.time() - started
    if not json_file.is_file():
        raise MixedExportError(
            f"parity_cold не написал {json_file} (код {proc.returncode}):\n"
            + _tail(proc.stderr or proc.stdout)
        )
    payload = json.loads(json_file.read_text(encoding="utf-8"))
    ppl = payload.get("perplexity") or {}
    c_ppl = (ppl.get("vtm_c") or {}).get("perplexity")
    if c_ppl is None:
        raise MixedExportError(
            "parity_cold не посчитал PPL C-рантайма:\n" + _tail(proc.stderr)
        )
    original = (ppl.get("hf_original") or {}).get("perplexity")
    return {
        "ppl": float(c_ppl),
        "bf16_ppl": float(original) if original is not None else None,
        "tokens": int((ppl.get("vtm_c") or {}).get("tokens") or 0),
        "lines": int(ppl.get("lines") or 0),
        "data": ppl.get("data"),
        "data_sha256": ppl.get("data_sha256"),
        "ppl_lines": int(ppl_lines),
        "ppl_max_tokens": int(ppl_max_tokens),
        "hf_ternary_ppl": (ppl.get("hf_ternary") or {}).get("perplexity"),
        "elapsed_s": round(elapsed, 1),
        "json": str(json_file),
        "exit_code": int(proc.returncode),
    }


def find_tokenizer(model_dir: Path | str) -> Path:
    """tokenizer.json в каталоге модели (для ``vtm run --text``)."""
    path = Path(model_dir).expanduser().resolve() / TOKENIZER_FILE
    if not path.is_file():
        raise MixedExportError(f"нет токенизатора: {path}")
    return path


def decode_answers(
    vtm_path: Path | str,
    tokenizer_path: Path | str,
    prompts: Sequence[str] = DECODE_PROMPTS,
    *,
    max_new: int = DEFAULT_DECODE_MAX_NEW,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """Декод промптов через chat-template Qwen2.5 (``vtm run --chat``)."""
    binary = _ROOT / "src" / "vtm" / "vtm"
    if not binary.is_file():
        raise MixedExportError(f"нет C-рантайма: {binary} (соберите src/vtm)")
    answers: list[dict[str, Any]] = []
    for prompt in prompts:
        cmd = [
            str(binary), "run", str(Path(vtm_path).expanduser().resolve()),
            "--text", prompt,
            "--tokenizer", str(Path(tokenizer_path).expanduser().resolve()),
            "--chat", "--max-new", str(int(max_new)),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=str(_ROOT),
            )
        except OSError as exc:
            raise MixedExportError(f"vtm run не запускается: {exc}") from exc
        text = ""
        for line in (proc.stdout or "").splitlines():
            if line.startswith("ответ:"):
                text = line.split(":", 1)[1].strip()
        answers.append({
            "prompt": prompt,
            "answer": text,
            "max_new": int(max_new),
            "exit_code": int(proc.returncode),
            "stderr": _tail(proc.stderr, 4) if proc.returncode else "",
        })
    return answers


# ---------------------------------------------------------------------------
# Прогон конфигурации и таблица
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """Одна измеренная конфигурация."""

    name: str
    policy: Policy
    record: dict[str, Any]


def run_configuration(
    name: str,
    policy: Policy,
    *,
    model_dir: Path | str,
    out_root: Path | str,
    work_root: Path | str,
    force: bool = False,
    recompile: bool = False,
    measure: bool = True,
    decode: bool = True,
    ppl_lines: int = PPL_LINES,
    ppl_max_tokens: int = PPL_MAX_TOKENS,
    skip_original: bool = False,
    decode_max_new: int = DEFAULT_DECODE_MAX_NEW,
    context_window: int = cold_export.DEFAULT_CONTEXT_WINDOW,
    limit_layers: int | None = None,
    echo: Any = None,
) -> dict[str, Any]:
    """Экспорт → компиляция → (PPL + декод) для одной политики."""
    echo = echo or (lambda message: None)
    model_path = Path(model_dir).expanduser().resolve()
    out_dir = Path(out_root).expanduser().resolve() / name
    work_dir = Path(work_root).expanduser().resolve() / name
    echo(f"[mixed] === {name}: {policy.to_text()} ===")

    hf_config = cold_export.load_hf_config(model_path)
    # Переиспользуем готовый экспорт только если политика в манифесте та же:
    # иначе молча померили бы чужую конфигурацию.
    reuse = False
    manifest_file = out_dir / cold_export.MANIFEST_FILE
    if manifest_file.is_file() and not force:
        try:
            previous = json.loads(manifest_file.read_text(encoding="utf-8"))
            reuse = (
                str((previous.get("mixed_precision") or {}).get("policy_text"))
                == policy.to_text()
            )
        except (OSError, json.JSONDecodeError):
            reuse = False
    if reuse:
        echo(f"[mixed] {name}: беру готовый экспорт {out_dir}")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        export = MixedExport(
            out_dir=out_dir,
            manifest_path=manifest_file,
            trit_path=out_dir / "ternary.trit",
            non_ternary_path=out_dir / qat_export.NON_TERNARY_FILE,
            manifest=manifest,
            architecture=manifest.get("architecture") or {},
            policy=policy,
            elapsed_s=0.0,
            peak_rss_bytes=0,
            trit_params=int(sum(e.get("n_trits", 0) for e in manifest.get("tensors") or [])),
            dense_params=int(sum(
                e.get("params", 0)
                for e in (manifest.get("non_ternary") or {}).get("tensors") or []
                if e.get("mixed")
            )),
            group_params={
                key: int(value.get("params", 0))
                for key, value in ((manifest.get("mixed_precision") or {}).get("groups") or {}).items()
            },
            group_bytes={
                key: int(value.get("bytes", 0))
                for key, value in ((manifest.get("mixed_precision") or {}).get("groups") or {}).items()
            },
            dtype_params={
                key: int(value)
                for key, value in ((manifest.get("mixed_precision") or {}).get("params_by_dtype") or {}).items()
            },
        )
    else:
        source = cold_export.open_source(model_path)
        try:
            export = export_mixed_model(
                source, hf_config, out_dir,
                policy=policy,
                model_name=str(model_path),
                context_window=context_window,
                limit_layers=limit_layers,
                force=True,
                echo=echo,
            )
        finally:
            source.close()

    container_path = out_dir / "model.vtm"
    reuse_container = (
        not recompile
        and container_path.is_file()
        and container_path.stat().st_mtime >= export.manifest_path.stat().st_mtime
    )
    if reuse_container:
        echo(f"[mixed] {name}: беру готовый контейнер {container_path}")
        container = {
            "path": str(container_path),
            "size_bytes": container_path.stat().st_size,
            "tensors": None,
            "instructions": None,
            "kinds": None,
            "bytes_by_kind": None,
            "trit_params": export.trit_params,
            "fp_params": export.dense_params,
            "reused": True,
        }
    else:
        container = compile_mixed(export.manifest_path, container_path)
    echo(
        f"[mixed] контейнер {container['size_bytes'] / 2**20:.1f} МиБ, "
        f"тензоров {container['tensors']} ({container['kinds']})"
    )
    record: dict[str, Any] = {
        "name": name,
        "policy": policy.describe(),
        "policy_text": policy.to_text(),
        "out_dir": str(export.out_dir),
        "manifest": str(export.manifest_path),
        "container": container,
        "export_bytes": export.export_bytes,
        "trit_bytes": export.trit_path.stat().st_size,
        "non_ternary_bytes": export.non_ternary_path.stat().st_size,
        "trit_params": export.trit_params,
        "dense_params": export.dense_params,
        "group_params": export.group_params,
        "group_bytes": export.group_bytes,
        "params_by_dtype": export.dtype_params,
        "architecture": {
            key: export.architecture.get(key)
            for key in ("hidden", "n_heads", "n_kv_heads", "head_dim",
                        "intermediate", "vocab", "n_layers",
                        "tie_word_embeddings", "embed")
        },
        "export_s": round(export.elapsed_s, 1),
        "ppl": None,
        "bf16_ppl": None,
        "answers": [],
    }
    if measure:
        record["ppl_result"] = measure_ppl(
            model_path, out_dir / "model.vtm",
            workdir=work_dir / "parity",
            json_path=work_dir / "parity.json",
            ppl_lines=ppl_lines,
            ppl_max_tokens=ppl_max_tokens,
            skip_original=skip_original,
            echo=echo,
        )
        record["ppl"] = record["ppl_result"]["ppl"]
        record["bf16_ppl"] = record["ppl_result"]["bf16_ppl"]
        echo(f"[mixed] PPL {name}: {record['ppl']:.4g}")
    if decode:
        record["answers"] = decode_answers(
            out_dir / "model.vtm",
            find_tokenizer(model_path),
            max_new=decode_max_new,
        )
    return record


def ratio_to_bf16(ppl: float | None, bf16: float | None) -> float | None:
    if ppl is None or not bf16:
        return None
    return float(ppl) / float(bf16)


def format_number(value: float | None) -> str:
    """PPL в человеческом виде: крупные числа — степенью, мелкие — числом."""
    if value is None:
        return "—"
    if value >= 1e4:
        return f"{value:.3g}"
    return f"{value:.4g}"


def format_table(records: Sequence[dict[str, Any]], *,
                 bf16_ppl: float | None = BF16_PPL,
                 answer_chars: int = 90) -> str:
    """Таблица «конфигурация → файл → PPL → ×bf16 → ответ»."""
    lines = [
        "| конфигурация | политика | файл .vtm | PPL | ×bf16 | декод (1-й промпт) |",
        "|---|---|---|---|---|---|",
    ]
    for record in records:
        container = record.get("container") or {}
        size = container.get("size_bytes") or record.get("export_bytes") or 0
        ppl = record.get("ppl")
        reference = record.get("bf16_ppl") or bf16_ppl
        ratio = ratio_to_bf16(ppl, reference)
        answer = ""
        answers = record.get("answers") or []
        if answers:
            answer = (answers[0].get("answer") or "").replace("\n", " ")
            if len(answer) > answer_chars:
                answer = answer[:answer_chars - 1] + "…"
            answer = answer.replace("|", "\\|")
        lines.append(
            f"| {record.get('name')} | `{record.get('policy_text')}` | "
            f"{size / 2**20:.0f} МиБ | {format_number(ppl)} | "
            f"{(f'×{ratio:.3g}' if ratio else '—')} | {answer} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/export_mixed.py",
        description=(
            "Холодный экспорт с по-слойной точностью: политика по группам "
            "embed/attn/ffn (trit | int8 | fp16 | fp32) -> .trit + "
            "non_ternary.bin + manifest.json -> model.vtm -> PPL и декод."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  PYTHONPATH=.pylibs python3.13 tools/export_mixed.py --model <dir> \\\n"
            "      --out-root export/mixed --name attn-fp16 \\\n"
            "      --policy embed=int8,attn=fp16,ffn=trit\n"
            "  PYTHONPATH=.pylibs python3.13 tools/export_mixed.py --model <dir> \\\n"
            "      --out-root export/mixed --suite\n"
            "  python3 tools/export_mixed.py --out-root export/mixed --report\n"
        ),
    )
    parser.add_argument("--model", default=None,
                        help="каталог HF-модели (для --suite/--policy)")
    parser.add_argument("--out-root", default=str(_ROOT / "export" / "mixed"),
                        help="корень каталогов конфигураций (по умолчанию export/mixed)")
    parser.add_argument("--work-root", default=None,
                        help="куда писать JSON/пробы parity_cold (по умолчанию data/parity_cold/mixed)")
    parser.add_argument("--name", default=None, help="имя конфигурации (с --policy)")
    parser.add_argument("--policy", default=None,
                        help="политика: embed=trit|int8|fp32,attn=trit|fp16|fp32,"
                             "ffn=...,first_last=...,layers.N=...")
    parser.add_argument("--suite", action="store_true",
                        help="прогнать набор конфигураций эксперимента 0146")
    parser.add_argument("--only", default=None,
                        help="через запятую: какие конфигурации --suite брать")
    parser.add_argument("--force", action="store_true", help="перезаписать экспорт")
    parser.add_argument("--recompile", action="store_true",
                        help="пересобрать .vtm, даже если он новее манифеста")
    parser.add_argument("--no-measure", action="store_true",
                        help="только экспорт и компиляция, без PPL/декода")
    parser.add_argument("--no-decode", action="store_true", help="без декода")
    parser.add_argument("--ppl-lines", type=int, default=PPL_LINES)
    parser.add_argument("--ppl-max-tokens", type=int, default=PPL_MAX_TOKENS)
    parser.add_argument("--skip-original", action="store_true",
                        help="не мерить bf16-эталон в parity_cold (быстрее)")
    parser.add_argument("--decode-max-new", type=int, default=DEFAULT_DECODE_MAX_NEW)
    parser.add_argument("--context-window", type=int,
                        default=cold_export.DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--limit-layers", type=int, default=None,
                        help="экспортировать первые N слоёв (быстрая проверка)")
    parser.add_argument("--report", action="store_true",
                        help="напечатать таблицу по experiment.json (без экспорта)")
    parser.add_argument("--json-out", default=None,
                        help="куда писать experiment.json (по умолчанию OUT_ROOT/experiment.json)")
    return parser


def experiment_path(out_root: Path | str) -> Path:
    return Path(out_root).expanduser().resolve() / "experiment.json"


def load_experiment(out_root: Path | str) -> dict[str, Any]:
    path = experiment_path(out_root)
    if not path.is_file():
        raise MixedExportError(f"нет {path}: сначала прогоните конфигурации")
    return json.loads(path.read_text(encoding="utf-8"))


def save_experiment(out_root: Path | str, payload: dict[str, Any]) -> Path:
    """Атомарная запись experiment.json (.tmp + rename)."""
    path = experiment_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)
    return path


def _upsert(payload: dict[str, Any], record: dict[str, Any]) -> None:
    records = payload.setdefault("configurations", [])
    for index, existing in enumerate(records):
        if existing.get("name") == record.get("name"):
            records[index] = record
            return
    records.append(record)
    order = {name: index for index, (name, _text) in enumerate(SUITE)}
    records.sort(key=lambda item: order.get(str(item.get("name")), len(order)))


def collect_records(out_root: Path | str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Свести записи из experiment.json и <конфиг>/record.json (без потерь).

    Прогон можно вести несколькими процессами: тогда experiment.json —
    сводка «кто успел», а record.json рядом с каждым экспортом — источник
    правды по своей конфигурации.
    """
    root = Path(out_root).expanduser().resolve()
    merged: dict[str, dict[str, Any]] = {}
    for record in payload.get("configurations") or []:
        merged[str(record.get("name"))] = record
    for path in sorted(root.glob("*/record.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = str(record.get("name"))
        known = merged.get(name)
        # Свежее измерение (PPL есть) важнее старого «только экспорт».
        if known is None or record.get("ppl") is not None:
            merged[name] = record
    order = {name: index for index, (name, _text) in enumerate(SUITE)}
    return sorted(merged.values(), key=lambda item: order.get(str(item.get("name")), len(order)))


def _experiment_defaults(model_dir: Any, args: Any) -> dict[str, Any]:
    model_path = str(Path(model_dir).expanduser().resolve())
    return {
        "format": "eva32.mixed.experiment",
        "format_version": 1,
        "task": "0146",
        "model": model_path,
        "ppl_method": {
            "tool": "tools/parity_cold.py",
            "flags": f"--ppl-lines {args.ppl_lines} --ppl-max-tokens "
                     f"{args.ppl_max_tokens} --skip-parity",
            "data": "data/train.jsonl",
            "bf16_ppl": BF16_PPL,
            "note": "как в 0143: те же 20 строк и способ замера",
        },
        "configurations": [],
    }


def record_experiment(out_root: Path | str, record: dict[str, Any],
                      defaults: dict[str, Any]) -> dict[str, Any]:
    """Прочитать-дописать experiment.json под блокировкой (параллельные прогоны).

    Конфигурации можно гонять несколькими процессами: каждый читает общий
    файл, добавляет свою запись и пишет атомарно, а fcntl.flock не даёт
    потерять чужие записи. Без fcntl (не-POSIX) остаётся атомарная запись:
    худшее — в файле не окажется чужой записи, а свой record.json рядом есть.
    """
    path = experiment_path(out_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+") as lock:
        locked = False
        try:
            import fcntl  # noqa: PLC0415 - только POSIX

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            locked = True
        except (ImportError, OSError):  # pragma: no cover - не-POSIX
            locked = False
        try:
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                payload = dict(defaults)
            _upsert(payload, record)
            if record.get("bf16_ppl") and not payload.get("bf16_ppl"):
                payload["bf16_ppl"] = record["bf16_ppl"]
            # per-config запись рядом с экспортом: источник правды, если
            # experiment.json потеряет строку из-за гонки.
            out_dir = Path(record.get("out_dir") or "")
            if out_dir.is_dir():
                (out_dir / "record.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            save_experiment(out_root, payload)
            return payload
        finally:
            if locked:  # pragma: no cover - отпускается закрытием файла
                try:
                    import fcntl  # noqa: PLC0415

                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    echo = lambda message: print(message, file=sys.stderr, flush=True)  # noqa: E731
    out_root = Path(args.out_root).expanduser().resolve()
    work_root = (
        Path(args.work_root).expanduser().resolve()
        if args.work_root else _ROOT / "data" / "parity_cold" / "mixed"
    )
    try:
        if args.report:
            payload = load_experiment(out_root) if experiment_path(out_root).is_file() else {}
            records = collect_records(out_root, payload)
            if not records:
                raise MixedExportError("нет ни experiment.json, ни <конфиг>/record.json")
            bf16 = payload.get("bf16_ppl") or BF16_PPL
            print(format_table(records, bf16_ppl=bf16))
            return 0

        if not args.model:
            raise MixedExportError("нужен --model (каталог HF-модели)")
        if bool(args.suite) == bool(args.policy):
            raise MixedExportError("нужен ровно один режим: --suite или --policy")
        if args.policy and not args.name:
            raise MixedExportError("с --policy нужен --name")

        if args.suite:
            names = [item.strip() for item in (args.only or "").split(",") if item.strip()]
            configs = [item for item in SUITE if not names or item[0] in names]
            if not configs:
                raise MixedExportError(f"--only={args.only!r}: нет таких конфигураций")
        else:
            configs = [(args.name, args.policy)]

        defaults = _experiment_defaults(args.model, args)

        for name, policy_text in configs:
            policy = Policy.parse(policy_text)
            record = run_configuration(
                name, policy,
                model_dir=args.model,
                out_root=out_root,
                work_root=work_root,
                force=args.force,
                recompile=args.recompile,
                measure=not args.no_measure,
                decode=not args.no_decode and not args.no_measure,
                ppl_lines=args.ppl_lines,
                ppl_max_tokens=args.ppl_max_tokens,
                skip_original=args.skip_original,
                decode_max_new=args.decode_max_new,
                context_window=args.context_window,
                limit_layers=args.limit_layers,
                echo=echo,
            )
            payload = record_experiment(out_root, record, defaults)
            print(format_table([record], bf16_ppl=payload.get("bf16_ppl") or BF16_PPL))
            print()

        if len(configs) > 1:
            payload = load_experiment(out_root)
            records = collect_records(out_root, payload)
            if records:
                print("=== все конфигурации ===")
                print(format_table(records, bf16_ppl=payload.get("bf16_ppl") or BF16_PPL))
        print(f"\nрезультат: {experiment_path(out_root)}")
        return 0
    except MixedExportError as exc:
        print(f"[mixed] {exc}")
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[mixed] прервано", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
