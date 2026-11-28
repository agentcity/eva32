#!/usr/bin/env python3
"""Умное округление при холодном экспорте: AWQ-масштабирование и GPTQ/OBQ.

Зачем
=====

Холодная тритизация Qwen2.5-0.5B (0136/0143) даёт PPL ×6,3e5 от bf16: текст
превращается в мусор. Наивное правило ``trit = round(w / absmax)`` выбирает
ближайший трит по ошибке **веса**, хотя цена ошибки — искажение **выхода
слоя**. Эта задача проверяет два известных улучшения без обучения:

1. **AWQ-масштабирование** (``--round awq``): у части каналов входа
   активации в десятки раз больше остальных, поэтому их вес после деления на
   общий absmax блока обнуляется. Канал ``j`` умножается на ``s_j``, а
   следующий слой — делится на ``s_j`` (произведение то же). Ошибка
   перераспределяется с «важных» каналов на «неважные».
2. **GPTQ/OBQ-округление** (``--round gptq``): триты выбираются по очереди
   (столбец за столбцом) так, чтобы уменьшить ``‖XW − XW_t‖²`` на
   калибровочных активациях, а ошибка уже округлённого столбца вычитается из
   ещё не округлённых через ``H⁻¹`` (``H = XᵀX``) — это линейная
   компенсация, а не обучение.
3. ``--round both`` — оба приёма: сначала масштабы AWQ, затем GPTQ по
   отмасштабированным весам и отмасштабированному Гессиану.

Формат не меняется
==================

Ни .trit, ни non_ternary.bin, ни манифест не получают новых полей, которые
читал бы рантайм: те же сегменты TRT1, те же смещения, те же правила. Меняются
только значения тритов/масштабов (и fp32-веса норм — в них свёрнут масштаб
AWQ). Поэтому:

* ``--round naive`` обязан побайтово совпасть с ``tools/cold_export.py`` —
  это и есть проверка, что новый инструмент пишет тот же формат;
* ``tools/vtm_compile.py`` и C-загрузчик не правятся и не знают, что
  округление было «умным»; отличия видны только в манифесте (блок
  ``smart``) и в содержимом сегментов.

Как устроено
============

1. Базовый экспорт целиком отдан ``tools/cold_export.py`` (сегменты,
   non_ternary.bin, манифест, sha256) — импортируются его функции, файл не
   правится.
2. Для ``awq|gptq|both`` один forward по калибровке (первые
   ``--calib-lines`` пар ``data/train.jsonl``, пропуская текст PPL —
   ``--calib-skip``) снимает с входов всех целевых Linear:
   * ``H = XᵀX`` (Гессиан входа для GPTQ);
   * ``mean|x_j|`` по каналам (статистика активаций для AWQ);
   * первые ``--calib-sample-tokens`` строк ``X`` (для выбора ``alpha``).
3. Масштабы AWQ сворачиваются офлайн в веса — там, где это точное
   тождество, а не приближение:

   * ``q,k,v`` (вход — выход ``input_layernorm``): ``s`` в вес нормы,
     ``1/s`` в столбцы q/k/v;
   * ``gate,up`` (вход — ``post_attention_layernorm``): то же;
   * ``down_proj`` (вход — ``silu(gate)·up``): ``s`` в строки up (произведение
     масштабируется одним множителем), ``1/s`` в столбцы down;
   * ``o_proj`` (вход — выход внимания): ``s`` в строки v и его bias,
     ``1/s`` в столбцы o. У GQA один канал v питает ``n_heads/n_kv_heads``
     каналов выхода внимания, поэтому гранулярность здесь — группа каналов,
     а не один канал (для 0.5B: 128 значений на 896 каналов).

   ``alpha`` в ``s_j = mean|x_j|^α / mean|W_j|^(1−α)`` выбирается на
   калибровке из ``--alphas`` по ошибке выхода слоя после Наивного
   округления (как в AWQ).
4. GPTQ идёт по столбцам блоками по 128 (границы блока совпадают с группой
   масштаба) с замороженным ``absmax`` исходного веса: ``q = clamp(round(w/s),−1,1)``,
   ``err = (w − q·s)/[H⁻¹]_jj``, ``W[:, j+1:] −= err·H⁻¹[j, j+1:]``. Порядок
   столбцов последовательный (без act-order) — детерминированно.
5. Триты/масштабы пишутся в сегменты на месте (смещения те же), после записи
   сегменты перечитываются и сверяются побайтово; sha256 и
   ``zero_fraction`` в манифесте обновляются, в него добавляется блок
   ``smart`` с параметрами прогона.

Память и время
==============

``H`` для ``down_proj`` — 4864² fp32 (94 МиБ) на слой, всего ~2,9 ГиБ на
0.5B; они живут до конца калибровки, поэтому пик RSS ~6 ГиБ. Калибровку
можно один раз сохранить (``--calib-cache``) и переиспользовать между
вариантами округления.

Запуск
======

::

    PYTHONPATH=.pylibs python3.13 tools/export_smart.py \\
        --model .hfcache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/<sha> \\
        --out export/qwen05b-gptq --round gptq --embed int8 --calib-lines 200

Дальше — как обычно: ``tools/vtm_compile.py --manifest …``, PPL —
``tools/parity_cold.py``, декод — ``src/vtm/vtm run``.

Ограничения (честно)
====================

* Это **не** обучение: масштабы и триты подбираются по калибровке, качество
  за её пределами не гарантировано. Калибровка и PPL берутся из одного файла
  (но не из одних строк: ``--calib-skip 20`` исключает текст PPL).
* Масштаб блока остаётся ``absmax`` исходного (и отмасштабированного) веса:
  оптимизируется только выбор трита, а не шаг квантования.
* ``o_proj`` у GQA масштабируется по группам каналов, а не поканально.
* GPTQ без act-order и без повторного выбора масштаба — упрощённый вариант.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import struct
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
from tools import cold_export  # noqa: E402

try:  # импорт как пакета (tests/*, `python3 -m tools.export_smart`)
    from tools import vtm_ref  # noqa: E402
except ImportError:  # pragma: no cover - запуск как `python3 tools/export_smart.py`
    import vtm_ref  # type: ignore[no-redef]  # noqa: E402

PROJECT_ROOT = _ROOT
TOOL_VERSION = 1
#: Варианты выбора трита. ``naive`` — эталон (ровно cold_export), остальные —
#: улучшения по отдельности и вместе.
ROUND_MODES = ("naive", "awq", "gptq", "both")
#: Калибровка: строки data/train.jsonl и обрезка пары в токенах (как PPL).
DEFAULT_CALIB_DATA = PROJECT_ROOT / "data" / "train.jsonl"
DEFAULT_CALIB_LINES = 200
MAX_CALIB_LINES = 400
DEFAULT_CALIB_MAX_TOKENS = 128
#: Первые строки файла — текст PPL (parity_cold --ppl-lines 20): исключаем,
#: чтобы калибровка не была текстом замера.
DEFAULT_CALIB_SKIP = 20
#: Сколько первых токенов входа хранить для выбора alpha (остальное — только
#: статистики: Гессиан от числа токенов не зависит).
DEFAULT_SAMPLE_TOKENS = 256
#: Кандидаты alpha в AWQ: 0 — только магнитуда веса, 1 — только активации.
AWQ_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
#: Нормировка s по геометрическому среднему и зажим: слишком большой разброс
#: масштабов сделал бы fp32-вес нормы бесполезным.
AWQ_SCALE_MIN = 1e-2
AWQ_SCALE_MAX = 1e2
#: Демпфирование Гессиана в GPTQ (доля от среднего диагонального элемента).
GPTQ_DAMP = 0.01
CALIB_CACHE_VERSION = 1


class SmartExportError(RuntimeError):
    """Ошибка умного экспорта, которую видно пользователю без трейсбека."""


# ---------------------------------------------------------------------------
# Выбор трита: общие численные помощники (без torch, тестируются на списках)
# ---------------------------------------------------------------------------


def check_round_mode(mode: str) -> str:
    """Проверить имя варианта округления."""
    if mode not in ROUND_MODES:
        raise SmartExportError(
            f"--round={mode!r}: поддержано {', '.join(ROUND_MODES)}"
        )
    return mode


def check_calib_lines(lines: int) -> int:
    """Калибровка: 1..``MAX_CALIB_LINES`` строк (200–400 — рабочий диапазон)."""
    lines = int(lines)
    if lines <= 0:
        raise SmartExportError(f"--calib-lines={lines}: нужно > 0")
    if lines > MAX_CALIB_LINES:
        raise SmartExportError(
            f"--calib-lines={lines}: больше {MAX_CALIB_LINES} не нужно — "
            "Гессиан стабилизируется на сотнях строк, а память растёт"
        )
    return lines


def parse_alphas(text: str | Sequence[float] | None) -> tuple[float, ...]:
    """Разобрать список alpha для AWQ (по умолчанию ``AWQ_ALPHAS``)."""
    if text is None:
        return AWQ_ALPHAS
    if isinstance(text, str):
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if not parts:
            raise SmartExportError("--alphas: пустой список")
        values = [float(part) for part in parts]
    else:
        values = [float(value) for value in text]
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise SmartExportError(
                f"--alphas: {value} вне [0, 1] — alpha по формуле AWQ там и живёт"
            )
    return tuple(values)


def awq_scale(
    act_absmean: Sequence[float],
    weight_absmean: Sequence[float],
    alpha: float,
    *,
    eps: float = 1e-12,
    clamp: tuple[float, float] = (AWQ_SCALE_MIN, AWQ_SCALE_MAX),
) -> list[float]:
    """Поканальный масштаб AWQ: ``s_j = a_j^α / w_j^(1−α)``, нормированный.

    ``a_j`` — средняя |активация| канала входа, ``w_j`` — средняя |вес| по
    столбцу. Деление на геометрическое среднее держит произведение масштабов
    около единицы (иначе весь слой уехал бы по величине); зажим
    ``[1e-2, 1e2]`` — от вырожденных каналов. Считается в логарифмах: у Qwen
    разброс активаций — десятки порядков, прямое произведение степеней
    упирается в float.
    """
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise SmartExportError(f"alpha={alpha}: нужно 0..1")
    if len(act_absmean) != len(weight_absmean):
        raise SmartExportError(
            f"AWQ: каналов активаций {len(act_absmean)}, весов "
            f"{len(weight_absmean)} — формы не сходятся"
        )
    if not act_absmean:
        raise SmartExportError("AWQ: пустой список каналов")
    logs = [
        alpha * math.log(max(float(a), eps))
        - (1.0 - alpha) * math.log(max(float(w), eps))
        for a, w in zip(act_absmean, weight_absmean)
    ]
    mean = math.fsum(logs) / len(logs)
    low, high = float(clamp[0]), float(clamp[1])
    return [min(max(math.exp(value - mean), low), high) for value in logs]


def kv_channel_map(
    q_dim: int, *, n_heads: int, n_kv_heads: int, head_dim: int
) -> list[int]:
    """Канал входа ``o_proj`` -> канал выхода ``v_proj`` (для GQA).

    Выход внимания склеен по головам запроса (``h*head_dim + d``), а ``v`` —
    по головам ключей. Один канал ``v`` масштабирует сразу
    ``n_heads/n_kv_heads`` каналов выхода внимания, поэтому у ``o_proj``
    поканальный масштаб невозможен: он обязан быть постоянным внутри группы.
    """
    if head_dim <= 0 or n_heads <= 0 or n_kv_heads <= 0:
        raise SmartExportError("kv_channel_map: размеры голов должны быть > 0")
    if n_heads % n_kv_heads:
        raise SmartExportError(
            f"n_heads={n_heads} не делится на n_kv_heads={n_kv_heads}: GQA-групп нет"
        )
    if q_dim != n_heads * head_dim:
        raise SmartExportError(
            f"q_dim={q_dim}, а n_heads*head_dim={n_heads * head_dim}"
        )
    per_kv = n_heads // n_kv_heads
    mapping: list[int] = []
    for channel in range(q_dim):
        head = channel // head_dim
        dim = channel % head_dim
        mapping.append((head // per_kv) * head_dim + dim)
    return mapping


def group_scales(
    per_channel: Sequence[float], mapping: Sequence[int], *, eps: float = 1e-12
) -> tuple[list[float], list[float]]:
    """Свести поканальные масштабы к группам (геометрическое среднее).

    Возвращает ``(по группам, по каналам)``: второе — то, что реально
    домножает столбцы ``o_proj``, первое — то, что домножает строки ``v``.
    """
    if len(per_channel) != len(mapping):
        raise SmartExportError(
            f"групповой масштаб: каналов {len(per_channel)}, карта {len(mapping)}"
        )
    n_groups = max(mapping) + 1 if mapping else 0
    logs: list[list[float]] = [[] for _ in range(n_groups)]
    for channel, group in enumerate(mapping):
        logs[group].append(math.log(max(float(per_channel[channel]), eps)))
    groups = [
        math.exp(math.fsum(values) / len(values)) if values else 1.0
        for values in logs
    ]
    return groups, [groups[group] for group in mapping]


def mean_abs_rows(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Средняя |W| по столбцу (каналу входа) — статистика веса для AWQ."""
    if not matrix:
        raise SmartExportError("mean_abs_rows: пустая матрица")
    width = len(matrix[0])
    sums = [0.0] * width
    for row in matrix:
        if len(row) != width:
            raise SmartExportError("mean_abs_rows: строки разной длины")
        for index, value in enumerate(row):
            sums[index] += abs(float(value))
    return [value / len(matrix) for value in sums]


def average_stats(groups: Sequence[Sequence[float]]) -> list[float]:
    """Поэлементное среднее нескольких статистик каналов (для группы слоёв)."""
    if not groups:
        raise SmartExportError("average_stats: пусто")
    width = len(groups[0])
    for group in groups:
        if len(group) != width:
            raise SmartExportError("average_stats: разная длина")
    return [
        math.fsum(group[index] for group in groups) / len(groups)
        for index in range(width)
    ]


# ---------------------------------------------------------------------------
# Сегмент TRT1 из готовых тритов/масштабов
# ---------------------------------------------------------------------------


def _is_torch_tensor(value: Any) -> bool:
    return cold_export._is_torch_tensor(value)


def _flat_list(value: Any) -> list[Any]:
    """Плоский список из torch-тензора или последовательности."""
    if hasattr(value, "detach"):
        return value.detach().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        flat: list[Any] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                flat.extend(_flat_list(item))
            else:
                flat.append(item)
        return flat
    return list(value)


def encode_trit_segment_explicit(
    trits: Any,
    scales: Any,
    *,
    block_size: int = ternary.BLOCK_SIZE,
    scale_dtype: str = "float32",
) -> tuple[bytes, dict[str, Any]]:
    """Собрать сегмент TRT1 из **уже выбранных** тритов и масштабов.

    Раскладка ровно та же, что у ``cold_export.encode_trit_segment``
    (заголовок ``TRT1``, масштабы, слова по 20 тритов, добивка до 160):
    тест сверяет байты. Поэтому умное округление пишет сегмент на место
    наивного, не меняя ни длины, ни смещений.
    """
    if block_size <= 0:
        raise SmartExportError(f"block_size={block_size}: нужно > 0")
    if scale_dtype not in ("float32", "float16"):
        raise SmartExportError(
            f"scale_dtype={scale_dtype!r}: поддержано float32, float16"
        )
    scale_fmt, torch_dtype = (
        ("<f4", "float32") if scale_dtype == "float32" else ("<f2", "float16")
    )
    torch_path = _is_torch_tensor(trits) or _is_torch_tensor(scales)
    if torch_path:
        import torch  # noqa: PLC0415

        trits_t = (
            trits.detach().reshape(-1).to(torch.int8).cpu()
            if _is_torch_tensor(trits)
            else torch.as_tensor(list(trits), dtype=torch.int8)
        )
        scales_t = (
            scales.detach().reshape(-1).float().cpu()
            if _is_torch_tensor(scales)
            else torch.as_tensor(list(scales), dtype=torch.float32)
        )
        n_trits = int(trits_t.numel())
        if n_trits % block_size:
            raise SmartExportError(
                f"{n_trits} тритов не делятся на block_size={block_size}: "
                "сегмент масштабов не ляжет по строкам"
            )
        total_scale_blocks = n_trits // block_size
        if int(scales_t.numel()) != total_scale_blocks:
            raise SmartExportError(
                f"масштабов {int(scales_t.numel())}, а блоков {total_scale_blocks}"
            )
        scales_out = scales_t.to(getattr(torch, torch_dtype))
        if not bool(torch.isfinite(scales_out).all()):
            raise SmartExportError(f"масштаб не помещается в {scale_dtype}")
        scales_bytes = qat_export._to_numpy_bytes(scales_out, scale_fmt)
        words = ternary.pack20_torch(trits_t)
        words_bytes = qat_export._to_numpy_bytes(words, "<u4")
        del scales_out, words
    else:
        flat_trits = [int(value) for value in _flat_list(trits)]
        flat_scales = [float(value) for value in _flat_list(scales)]
        n_trits = len(flat_trits)
        if n_trits % block_size:
            raise SmartExportError(
                f"{n_trits} тритов не делятся на block_size={block_size}"
            )
        total_scale_blocks = n_trits // block_size
        if len(flat_scales) != total_scale_blocks:
            raise SmartExportError(
                f"масштабов {len(flat_scales)}, а блоков {total_scale_blocks}"
            )
        # Чистый путь: тот же кодировщик, что у qat.ternary (и у cold_export).
        data = ternary.encode_segment(
            flat_trits, flat_scales, scale_dtype=scale_dtype, scale_block=block_size
        )
        info = ternary.segment_info(data, block_size)
        return data, {
            "params": n_trits,
            "n_trits": n_trits,
            "n_scale_blocks": info.n_scale_blocks,
            "n_words": info.n_words,
            "n_storage_blocks": info.n_storage_blocks,
            "n_storage_words": info.n_words,
            "zero_fraction": (
                round(sum(1 for value in flat_trits if value == 0) / n_trits, 6)
                if n_trits else 0.0
            ),
            "scale_blocks_per_row": None,
        }

    storage_blocks = (
        n_trits + ternary.TRITS_PER_STORAGE_BLOCK - 1
    ) // ternary.TRITS_PER_STORAGE_BLOCK
    expected_words = storage_blocks * ternary.WORDS_PER_STORAGE_BLOCK * 4
    if len(words_bytes) > expected_words:  # pragma: no cover - защита
        raise SmartExportError(
            f"слов {len(words_bytes)} байт, а блоков {storage_blocks} даёт "
            f"{expected_words}"
        )
    if len(words_bytes) < expected_words:
        words_bytes += b"\x00" * (expected_words - len(words_bytes))
    flags = ternary.FLAG_SCALE_F32 if scale_dtype == "float32" else 0
    header = struct.pack(
        "<4sBBQI",
        ternary.MAGIC,
        ternary.VERSION,
        flags,
        n_trits,
        total_scale_blocks,
    )
    data = header + scales_bytes + words_bytes
    info = ternary.segment_info(data, block_size)
    if info.n_scale_blocks != total_scale_blocks:
        raise SmartExportError("заголовок сегмента не сошёлся с раскладкой")
    zero_count = 0
    if torch_path:
        zero_count = int((trits_t == 0).sum().item())
    else:
        zero_count = sum(1 for value in flat_trits if value == 0)
    return data, {
        "params": n_trits,
        "n_trits": n_trits,
        "n_scale_blocks": total_scale_blocks,
        "n_words": (n_trits + ternary.TRITS_PER_WORD - 1) // ternary.TRITS_PER_WORD,
        "n_storage_blocks": storage_blocks,
        "n_storage_words": storage_blocks * ternary.WORDS_PER_STORAGE_BLOCK,
        "zero_fraction": round(zero_count / n_trits, 6) if n_trits else 0.0,
        "scale_blocks_per_row": None,
    }


# ---------------------------------------------------------------------------
# Манифест: обновить sha256/разреженность и дописать блок smart
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_smart_manifest(
    manifest: dict[str, Any],
    *,
    round_mode: str,
    file_stats: dict[str, dict[str, Any]],
    zero_by_tensor: dict[str, float],
    zero_counts: dict[str, int],
    info: dict[str, Any],
) -> dict[str, Any]:
    """Вернуть копию манифеста с обновлёнными хешами и блоком ``smart``.

    Смещения, формы, архитектура и записи нетронутых тензоров не меняются:
    сегменты того же размера, формат тот же. Обновляются только
    ``files[].sha256/bytes``, ``integrity.files``, ``sparsity`` и
    ``zero_fraction`` пересчитанных тензоров.
    """
    check_round_mode(round_mode)
    patched = copy.deepcopy(manifest)
    files = patched.setdefault("files", {})
    for name, stats in file_stats.items():
        if name not in files:
            raise SmartExportError(f"манифест: нет записи о файле {name}")
        files[name]["sha256"] = str(stats["sha256"])
        files[name]["bytes"] = int(stats["bytes"])
    integrity = patched.get("integrity")
    if isinstance(integrity, dict) and isinstance(integrity.get("files"), dict):
        for name, stats in file_stats.items():
            integrity["files"][name] = str(stats["sha256"])

    total_trits = 0
    zero_trits = 0.0
    for entry in patched.get("tensors") or []:
        name = str(entry.get("name") or "")
        n_trits = int(entry.get("n_trits") or entry.get("params") or 0)
        if name in zero_by_tensor:
            fraction = float(zero_by_tensor[name])
            entry["zero_fraction"] = round(fraction, 6)
        else:
            fraction = float(entry.get("zero_fraction") or 0.0)
        total_trits += n_trits
        zero_trits += fraction * n_trits
    for entry in (patched.get("non_ternary") or {}).get("tensors") or []:
        if str(entry.get("dtype")) == "trit":
            n_trits = int(entry.get("params") or 0)
            total_trits += n_trits
            zero_trits += float(entry.get("zero_fraction") or 0.0) * n_trits
    sparsity = patched.get("sparsity")
    if isinstance(sparsity, dict):
        fraction = zero_trits / total_trits if total_trits else 0.0
        sparsity["zero_fraction"] = round(fraction, 6)
        sparsity["collapsed"] = bool(
            int(sparsity.get("trits") or total_trits) and
            fraction > float(sparsity.get("warn_threshold") or 0.95)
        )
        sparsity["trits"] = int(total_trits)
    quantization = patched.setdefault("quantization", {})
    quantization["rounding"] = round_mode
    patched["smart"] = {
        "format": "eva32.export_smart",
        "format_version": TOOL_VERSION,
        "tool": "tools/export_smart.py",
        "round": round_mode,
        **info,
    }
    return patched


# ---------------------------------------------------------------------------
# Калибровка: Гессиан входа, статистика активаций, сэмплы
# ---------------------------------------------------------------------------


@dataclass
class TensorStats:
    """Что снято с входа одного Linear за калибровку."""

    tokens: int = 0
    abs_sum: Any = None      # (in,) float32 на устройстве модели
    hessian: Any = None      # (in, in) float32 (на устройстве до .cpu())
    samples: list[Any] = field(default_factory=list)
    sample_tokens: int = 0


@dataclass
class Calibration:
    """Статистика по всем целевым матрицам + параметры прогона."""

    stats: dict[str, TensorStats]
    info: dict[str, Any]


def _target_weight_names() -> dict[str, str]:
    """Суффикс qat.export -> путь в HF-именах модели."""
    paths = {}
    for suffix in cold_export.TARGET_SUFFIXES:
        if suffix in ("gate_proj", "up_proj", "down_proj"):
            paths[suffix] = f"mlp.{suffix}.weight"
        else:
            paths[suffix] = f"self_attn.{suffix}.weight"
    return paths


def calibration_texts(
    data: str | Path,
    *,
    lines: int,
    max_tokens: int,
    skip: int,
    tokenizer: Any,
) -> list[list[int]]:
    """Первые ``lines`` пар после ``skip`` в виде списков токенов.

    Читаются тем же ``qat.data.load_pairs``, что и обучение/PPL (берём
    ``question``/``answer`` без логитов). Текст — формат PPL
    (``Вопрос: …\\nОтвет: …``), чтобы активации были из того же
    распределения, что и замер.
    """
    from qat import data as qat_data  # noqa: PLC0415 - внутри: без torch не нужно

    if tokenizer is None:
        raise SmartExportError("калибровка: нет токенизатора")
    rows = qat_data.load_pairs(
        data, max_samples=int(lines) + int(skip), with_logits=False
    )
    if skip:
        rows = rows[int(skip):]
    if not rows:
        raise SmartExportError(
            f"калибровка: в {data} нет строк после пропуска {skip}"
        )
    sequences: list[list[int]] = []
    for row in rows:
        text = f"Вопрос: {row['question']}\nОтвет: {row['answer']}"
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        ids = [int(token) for token in ids[: int(max_tokens)]]
        if len(ids) >= 8:
            sequences.append(ids)
        if len(sequences) >= int(lines):
            break
    if not sequences:
        raise SmartExportError(
            "калибровка: после токенизации не осталось последовательностей"
        )
    return sequences


def collect_calibration(
    model: Any,
    tokenizer: Any,
    *,
    data: str | Path,
    lines: int,
    max_tokens: int,
    skip: int,
    sample_tokens: int,
    batch_size: int = 8,
    device: str = "cpu",
    echo: Any = print,
) -> Calibration:
    """Снять с входов целевых Linear Гессиан, |активации| и сэмплы.

    Один потоковый проход: батчи по ``batch_size`` с padding и маской (pad-токены
    не попадают в статистику). ``H = XᵀX`` копится на устройстве модели,
    в конце переносится на CPU. ``down_proj`` (вход 4864) — самый дорогой:
    его ``H`` это 94 МиБ fp32 на слой.
    """
    import torch  # noqa: PLC0415

    sequences = calibration_texts(
        data, lines=lines, max_tokens=max_tokens, skip=skip, tokenizer=tokenizer
    )
    names: dict[str, str] = {}
    modules: dict[str, Any] = {}
    # В state_dict путь кончается на ".weight", у модуля — без него:
    # "…self_attn.q_proj" / "…mlp.down_proj".
    wanted = {
        path[: -len(".weight")] for path in _target_weight_names().values()
    }
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(name == rel or name.endswith("." + rel) for rel in wanted):
            continue
        key = f"{name}.weight"
        names[key] = name
        modules[key] = module
    if not modules:
        raise SmartExportError(
            "калибровка: в модели не нашлось ни одного целевого Linear "
            f"({', '.join(sorted(wanted))})"
        )

    stats: dict[str, TensorStats] = {}
    state: dict[str, Any] = {"mask": None, "stats": stats, "sample_tokens": int(sample_tokens)}
    handles = []

    def make_hook(key: str):
        # torch 2.x: forward-hook с with_kwargs=True получает
        # (module, args, kwargs, output) — вход берём из args, output не нужен.
        def hook(module, args, kwargs, output):
            x = args[0] if args else (kwargs or {}).get("input")
            if x is None or not hasattr(x, "shape"):
                return
            x = x.detach().reshape(-1, int(x.shape[-1])).float()
            mask = state.get("mask")
            if mask is not None and int(mask.numel()) == int(x.shape[0]):
                x = x[mask.to(x.device)]
            if int(x.numel()) == 0:
                return
            item = stats[key]
            gram = x.transpose(0, 1) @ x
            if item.hessian is None:
                item.hessian = gram
            else:
                item.hessian.add_(gram)
            sums = x.abs().sum(dim=0)
            if item.abs_sum is None:
                item.abs_sum = sums
            else:
                item.abs_sum.add_(sums)
            item.tokens += int(x.shape[0])
            left = state["sample_tokens"] - item.sample_tokens
            if left > 0:
                take = min(left, int(x.shape[0]))
                item.samples.append(x[:take].detach().to("cpu").clone())
                item.sample_tokens += take

        return hook

    for key, module in modules.items():
        stats[key] = TensorStats()
        handles.append(module.register_forward_hook(make_hook(key), with_kwargs=True))
    started = time.time()
    total_tokens = 0
    try:
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                batch = sequences[start:start + batch_size]
                width = max(len(ids) for ids in batch)
                input_ids = torch.zeros((len(batch), width), dtype=torch.long)
                attention = torch.zeros((len(batch), width), dtype=torch.long)
                for row, ids in enumerate(batch):
                    input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                    attention[row, :len(ids)] = 1
                input_ids = input_ids.to(device)
                attention = attention.to(device)
                state["mask"] = attention.reshape(-1) > 0
                total_tokens += int(attention.sum().item())
                model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    use_cache=False,
                )
                if (start // batch_size + 1) % 8 == 0 or start + batch_size >= len(sequences):
                    echo(
                        f"[export_smart] калибровка: {min(start + batch_size, len(sequences))}"
                        f"/{len(sequences)} последовательностей, токенов {total_tokens}, "
                        f"{time.time() - started:.0f} с"
                    )
    finally:
        state["mask"] = None
        for handle in handles:
            handle.remove()

    for item in stats.values():
        if item.hessian is None:
            raise SmartExportError("калибровка: слой не получил ни одного токена")
        item.hessian = item.hessian.cpu()
        if item.abs_sum is not None:
            item.abs_sum = item.abs_sum.cpu().double()
        item.samples = (
            torch.cat(item.samples, dim=0) if item.samples else None
        )
    info = {
        "data": str(Path(data).resolve()),
        "lines": len(sequences),
        "skip": int(skip),
        "max_tokens": int(max_tokens),
        "tokens": int(total_tokens),
        "sample_tokens": int(sample_tokens),
        "batch_size": int(batch_size),
        "device": str(device),
        "seconds": round(time.time() - started, 1),
        "tensors": len(stats),
    }
    return Calibration(stats=stats, info=info)


# ---------------------------------------------------------------------------
# Квантизация: наивная (эталон) и GPTQ/OBQ
# ---------------------------------------------------------------------------


def block_absmax(weight: Any, block_size: int) -> Any:
    """absmax по блокам вдоль входной размерности: ``(out, in/block)``."""
    import torch  # noqa: PLC0415

    if weight.dim() != 2:
        raise SmartExportError(f"block_absmax: нужна 2-D матрица, а не {tuple(weight.shape)}")
    cols = int(weight.shape[1])
    if cols % block_size:
        raise SmartExportError(
            f"{cols} столбцов не кратны block_size={block_size}"
        )
    return weight.detach().float().reshape(int(weight.shape[0]), cols // block_size, block_size).abs().amax(dim=-1)


def quantize_naive(weight: Any, block_size: int = ternary.BLOCK_SIZE) -> dict[str, Any]:
    """Наивное правило QAT/cold_export: ``trit = clamp(round_half_away(w/absmax))``."""
    return ternary.quantize_torch(weight.detach().float().cpu(), block_size)


def dequantize(trits: Any, scales: Any, block_size: int = ternary.BLOCK_SIZE) -> Any:
    """``w = trit * scale`` по блокам (как читает рантайм)."""
    import torch  # noqa: PLC0415

    flat = trits.detach().reshape(int(trits.shape[0]), -1).float()
    cells = int(scales.shape[-1])
    expanded = scales.detach().reshape(int(trits.shape[0]), cells).float()
    expanded = expanded.repeat_interleave(block_size, dim=1)[:, : flat.shape[1]]
    return flat * expanded


def gptq_quantize(
    weight: Any,
    hessian: Any,
    *,
    block_size: int = ternary.BLOCK_SIZE,
    damp: float = GPTQ_DAMP,
) -> tuple[Any, Any, dict[str, Any]]:
    """GPTQ/OBQ: триты, минимизирующие ``‖XW − XW_t‖²``, а не ошибку веса.

    ``H = XᵀX`` (вход слоя), столбцы округляются по очереди; ошибка
    округлённого столбца делится на ``[H⁻¹]_jj`` и вычитается из остатка
    столбцов (линейная компенсация). Масштаб блока — ``absmax`` исходного
    веса и не пересчитывается: оптимизируется выбор трита, а не шаг.

    Возвращает ``(trits int8, scales float32, stats)``. Порядок столбцов
    последовательный (без act-order) — детерминированно и без перестановки,
    которая сломала бы границы блоков 128.
    """
    import torch  # noqa: PLC0415

    if weight.dim() != 2:
        raise SmartExportError(f"GPTQ: нужна 2-D матрица, а не {tuple(weight.shape)}")
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    if cols % block_size:
        raise SmartExportError(f"GPTQ: {cols} столбцов не кратны {block_size}")
    if tuple(hessian.shape) != (cols, cols):
        raise SmartExportError(
            f"GPTQ: Гессиан {tuple(hessian.shape)}, а вход {cols}"
        )
    work = weight.detach().float().cpu().clone()
    scales = block_absmax(work, block_size)
    H = hessian.detach().float().cpu().clone()
    diagonal = torch.diagonal(H).clone()
    dead = diagonal <= 0
    if bool(dead.any()):
        H[dead, dead] = 1.0
    alive = diagonal[~dead]
    mean_diag = float(alive.mean()) if alive.numel() else 1.0
    mean_diag = max(mean_diag, 1e-12)
    jitter = float(damp)
    inverse = None
    for attempt in range(3):
        candidate = H + torch.eye(cols, dtype=H.dtype) * mean_diag * jitter
        try:
            _chol, info = torch.linalg.cholesky_ex(candidate)
        except Exception:  # noqa: BLE001 - старый torch без cholesky_ex
            info = torch.zeros((), dtype=torch.int32)
        if int(info) == 0:
            # Обратная по Холецкому (potri) даёт симметричный H⁻¹; в 4864²
            # это вчетверо дороже linalg.inv, а разница в fp32 с
            # демпфированием 1% — в последних битах.
            inverse = torch.linalg.inv(candidate)
            break
        jitter *= 10.0
    if inverse is None:
        raise SmartExportError(
            "GPTQ: Гессиан не положительно определён даже с демпфированием "
            f"{jitter:g}; увеличьте --gptq-damp"
        )
    Hinv = inverse
    trits = torch.zeros((rows, cols), dtype=torch.int8)
    updated = 0
    with torch.no_grad():
        for column in range(cols):
            step = scales[:, column // block_size]
            values = work[:, column]
            zero = step <= 0
            safe = torch.where(zero, torch.ones_like(step), step)
            ratio = values / safe
            q = torch.sign(ratio) * torch.floor(ratio.abs() + 0.5)
            q = q.clamp(-1.0, 1.0)
            if bool(zero.any()):
                q = torch.where(zero, torch.zeros_like(q), q)
            restored = q * step
            trits[:, column] = q.to(torch.int8)
            curvature = float(Hinv[column, column])
            if curvature > 1e-12 and column + 1 < cols:
                error = (values - restored) / curvature
                # addr_ — внешнее произведение на месте: без временной
                # матрицы 896×4864 на каждый столбец.
                work[:, column + 1:].addr_(
                    error, Hinv[column, column + 1:], alpha=-1.0
                )
                updated += 1
            work[:, column] = restored
    stats = {
        "rows": rows,
        "cols": cols,
        "damp": float(damp),
        "jitter": float(jitter),
        "mean_diag": mean_diag,
        "compensated_columns": updated,
        "max_scale": float(scales.max()) if scales.numel() else 0.0,
    }
    return trits, scales, stats


# ---------------------------------------------------------------------------
# AWQ: выбор alpha, свёртка масштаба в соседние веса
# ---------------------------------------------------------------------------


def _alpha_error(
    samples: Any,
    weight: Any,
    scale_values: Sequence[float],
    block_size: int,
) -> float:
    """Ошибка выхода слоя после наивного округления отмасштабированного веса.

    Сравнивается ``X_s·(W/s)ᵀ`` с ``X_s·W_tᵀ``: это ровно ошибка, которую
    увидит читатель веса (``X_s`` — вход уже с учётом свёрнутого масштаба).
    """
    import torch  # noqa: PLC0415

    scale = torch.tensor(list(scale_values), dtype=torch.float32)
    folded = weight.detach().float().cpu() / scale[None, :]
    quant = ternary.quantize_torch(folded, block_size)
    restored = dequantize(quant["trits"], quant["scales"], block_size)
    delta = (folded - restored).transpose(0, 1)
    xs = samples.detach().float().cpu() * scale[None, :]
    diff = xs @ delta
    return float(diff.pow(2).sum())


def choose_alpha(
    alphas: Sequence[float],
    *,
    samples: Any | None,
    tensors: Sequence[Any],
    act_absmean: Sequence[float],
    weight_absmean: Sequence[float],
    block_size: int,
    group: Sequence[int] | None = None,
) -> tuple[float, list[float], dict[str, Any]]:
    """Выбрать ``alpha`` по ошибке выхода слоя на калибровочных активациях.

    ``group`` — карта «канал -> группа» для ``o_proj`` у GQA: масштаб внутри
    группы обязан быть один, поэтому кандидат сводится к группам до
    квантизации.
    """
    if not alphas:
        raise SmartExportError("choose_alpha: пустой список alpha")
    if samples is None or not tensors:
        raise SmartExportError("choose_alpha: нет калибровочных данных")
    best: tuple[float, float, list[float]] | None = None
    tried: list[dict[str, Any]] = []
    for alpha in alphas:
        scale = awq_scale(act_absmean, weight_absmean, alpha)
        if group is not None:
            _groups, scale = group_scales(scale, group)
        error = math.fsum(
            _alpha_error(samples, tensor, scale, block_size) for tensor in tensors
        )
        tried.append({"alpha": float(alpha), "error": error})
        if best is None or error < best[1]:
            best = (float(alpha), error, list(scale))
    assert best is not None
    return best[0], best[2], {"tried": tried, "error": best[1]}


def fold_awq_layer(
    weights: dict[str, Any],
    *,
    scales: dict[str, Any],
    biases: dict[str, Any],
    norms: dict[str, Any],
) -> None:
    """Свернуть масштабы AWQ в веса (in-place, точные тождества).

    ``weights``/``biases``/``norms`` — торч-тензоры **одного слоя**:
    ``q/k/v/o/gate/up/down``, их bias и веса норм. Каждое действие —
    умножение/деление строк или столбцов, поэтому выход слоя не меняется
    (проверяется тестом ``test_fold_keeps_outputs``).
    """
    import torch  # noqa: PLC0415

    with torch.no_grad():
        norm1 = norms["norm1"]
        norm1.mul_(scales["attn_in"])
        for key in ("q_proj", "k_proj", "v_proj"):
            weights[key].div_(scales["attn_in"][None, :])
        norm2 = norms["norm2"]
        norm2.mul_(scales["mlp_in"])
        for key in ("gate_proj", "up_proj"):
            weights[key].div_(scales["mlp_in"][None, :])
        # o_proj: вход — выход внимания (склеен по головам запроса).
        weights["o_proj"].div_(scales["o_in"][None, :])
        weights["v_proj"].mul_(scales["v_out"][:, None])
        bias_v = biases.get("v_proj")
        if bias_v is not None:
            bias_v.mul_(scales["v_out"])
        # down_proj: вход — silu(gate)·up; масштаб вносим в up (один множитель
        # произведения), обратный — в столбцы down.
        weights["down_proj"].div_(scales["down_in"][None, :])
        weights["up_proj"].mul_(scales["down_in"][:, None])


def _scale_tensor(values: Sequence[float]) -> Any:
    import torch  # noqa: PLC0415

    return torch.tensor([float(value) for value in values], dtype=torch.float32)


def _average_weight_stats(tensors: Sequence[Any]) -> list[float]:
    """Средняя |W| по каналам входа, усреднённая по группе матриц."""
    import torch  # noqa: PLC0415

    stats = [
        tensor.detach().float().abs().mean(dim=0).cpu().tolist() for tensor in tensors
    ]
    return average_stats(stats)


def _act_absmean(stats: TensorStats) -> list[float]:
    if stats.abs_sum is None or stats.tokens <= 0:
        raise SmartExportError("AWQ: нет статистики активаций")
    return (stats.abs_sum / float(stats.tokens)).tolist()


def plan_layer_scales(
    *,
    layer: int,
    weights: dict[str, Any],
    stats: dict[str, TensorStats],
    architecture: dict[str, Any],
    alphas: Sequence[float],
    block_size: int,
    echo: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Масштабы AWQ одного слоя (4 группы) + отчёт по alpha."""
    hidden = int(architecture["hidden"])
    q_dim = int(architecture["n_heads"]) * int(architecture["head_dim"])
    kv_dim = int(architecture["n_kv_heads"]) * int(architecture["head_dim"])
    prefix = f"model.layers.{layer}"
    qkv_names = [f"{prefix}.self_attn.{key}.weight" for key in ("q_proj", "k_proj", "v_proj")]
    gateup_names = [f"{prefix}.mlp.{key}.weight" for key in ("gate_proj", "up_proj")]
    names = {
        "q_proj": qkv_names[0], "k_proj": qkv_names[1], "v_proj": qkv_names[2],
        "gate_proj": gateup_names[0], "up_proj": gateup_names[1],
        "o_proj": f"{prefix}.self_attn.o_proj.weight",
        "down_proj": f"{prefix}.mlp.down_proj.weight",
    }
    report: dict[str, Any] = {"layer": int(layer)}
    scales: dict[str, Any] = {}

    # q, k, v: общий вход (выход input_layernorm) — общий масштаб.
    qkv_tensors = [weights[key] for key in ("q_proj", "k_proj", "v_proj")]
    alpha, values, detail = choose_alpha(
        alphas,
        samples=stats[names["q_proj"]].samples,
        tensors=qkv_tensors,
        act_absmean=_act_absmean(stats[names["q_proj"]]),
        weight_absmean=_average_weight_stats(qkv_tensors),
        block_size=block_size,
    )
    scales["attn_in"] = _scale_tensor(values)
    report["attn"] = {"alpha": alpha, **detail}

    # gate, up: общий вход (выход post_attention_layernorm).
    gateup_tensors = [weights[key] for key in ("gate_proj", "up_proj")]
    alpha, values, detail = choose_alpha(
        alphas,
        samples=stats[names["gate_proj"]].samples,
        tensors=gateup_tensors,
        act_absmean=_act_absmean(stats[names["gate_proj"]]),
        weight_absmean=_average_weight_stats(gateup_tensors),
        block_size=block_size,
    )
    scales["mlp_in"] = _scale_tensor(values)
    report["mlp"] = {"alpha": alpha, **detail}

    # o: вход — выход внимания; у GQA масштаб один на группу каналов v.
    mapping = kv_channel_map(
        q_dim,
        n_heads=int(architecture["n_heads"]),
        n_kv_heads=int(architecture["n_kv_heads"]),
        head_dim=int(architecture["head_dim"]),
    )
    o_stats = stats[names["o_proj"]]
    alpha, o_values, detail = choose_alpha(
        alphas,
        samples=o_stats.samples,
        tensors=[weights["o_proj"]],
        act_absmean=_act_absmean(o_stats),
        weight_absmean=_average_weight_stats([weights["o_proj"]]),
        block_size=block_size,
        group=mapping,
    )
    groups, per_channel = group_scales(o_values, mapping)
    if len(groups) != kv_dim:
        raise SmartExportError(
            f"слой {layer}: групп o_proj {len(groups)}, а каналов v {kv_dim}"
        )
    scales["o_in"] = _scale_tensor(per_channel)
    scales["v_out"] = _scale_tensor(groups)
    report["o"] = {"alpha": alpha, "groups": len(groups), **detail}

    # down: вход — silu(gate)·up, масштаб уходит в строки up.
    down_stats = stats[names["down_proj"]]
    alpha, values, detail = choose_alpha(
        alphas,
        samples=down_stats.samples,
        tensors=[weights["down_proj"]],
        act_absmean=_act_absmean(down_stats),
        weight_absmean=_average_weight_stats([weights["down_proj"]]),
        block_size=block_size,
    )
    scales["down_in"] = _scale_tensor(values)
    report["down"] = {"alpha": alpha, **detail}
    if echo is not None:
        echo(
            f"[export_smart] слой {layer}: alpha attn={report['attn']['alpha']:g} "
            f"mlp={report['mlp']['alpha']:g} o={report['o']['alpha']:g} "
            f"down={report['down']['alpha']:g}"
        )
    if int(scales["attn_in"].numel()) != hidden:
        raise SmartExportError(
            f"слой {layer}: attn_in {int(scales['attn_in'].numel())} != hidden {hidden}"
        )
    if int(scales["o_in"].numel()) != q_dim:
        raise SmartExportError(
            f"слой {layer}: o_in {int(scales['o_in'].numel())} != q_dim {q_dim}"
        )
    return scales, report


# ---------------------------------------------------------------------------
# План обновлений и запись артефактов
# ---------------------------------------------------------------------------


def _nonzero(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def plan_updates(
    model: Any,
    calibration: Calibration,
    *,
    round_mode: str,
    architecture: dict[str, Any],
    block_size: int,
    alphas: Sequence[float],
    damp: float,
    echo: Any = print,
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    """Посчитать триты/масштабы (и fp32-веса норм) для всех матриц модели.

    Возвращает ``(trit_updates, non_ternary_updates, info)``; имена — как в
    манифесте экспорта, поэтому запись идёт по готовым смещениям.
    """
    import torch  # noqa: PLC0415

    check_round_mode(round_mode)
    awq = round_mode in ("awq", "both")
    gptq = round_mode in ("gptq", "both")
    n_layers = int(architecture["n_layers"])
    paths = _target_weight_names()
    bias_paths = {
        "q_proj": "self_attn.q_proj.bias",
        "k_proj": "self_attn.k_proj.bias",
        "v_proj": "self_attn.v_proj.bias",
        "o_proj": "self_attn.o_proj.bias",
    }
    params = {
        name: parameter.detach()
        for name, parameter in model.named_parameters()
    }
    trit_updates: dict[str, dict[str, Any]] = {}
    non_updates: dict[str, bytes] = {}
    info: dict[str, Any] = {
        "round": round_mode,
        "block_size": int(block_size),
        "gptq": {"damp": float(damp)} if gptq else None,
        "awq": {"alphas": [float(value) for value in alphas]} if awq else None,
        "layers": [],
        "changed_trits": 0,
        "total_trits": 0,
    }
    for layer in range(n_layers):
        prefix = f"model.layers.{layer}"
        names = {key: f"{prefix}.{path}" for key, path in paths.items()}
        missing = [name for name in names.values() if name not in params]
        if missing:
            raise SmartExportError(
                f"слой {layer}: в модели нет {', '.join(missing[:3])}"
            )
        weights = {key: params[name] for key, name in names.items()}
        norm1_name = f"{prefix}.input_layernorm.weight"
        norm2_name = f"{prefix}.post_attention_layernorm.weight"
        v_bias_name = f"{prefix}.{bias_paths['v_proj']}"
        biases: dict[str, Any] = {}
        for key, path in bias_paths.items():
            candidate = f"{prefix}.{path}"
            if candidate in params:
                biases[key] = params[candidate]
        norms = {
            "norm1": params[norm1_name],
            "norm2": params[norm2_name],
        }
        layer_report: dict[str, Any] = {"layer": layer}
        input_scales: dict[str, Any] | None = None
        if awq:
            input_scales, layer_report["awq"] = plan_layer_scales(
                layer=layer,
                weights=weights,
                stats=calibration.stats,
                architecture=architecture,
                alphas=alphas,
                block_size=block_size,
                echo=echo,
            )
            fold_awq_layer(
                weights, scales=input_scales, biases=biases, norms=norms,
            )
            non_updates[norm1_name] = qat_export._to_numpy_bytes(
                norms["norm1"].detach().to("cpu").float(), "<f4"
            )
            non_updates[norm2_name] = qat_export._to_numpy_bytes(
                norms["norm2"].detach().to("cpu").float(), "<f4"
            )
            if v_bias_name in params:
                non_updates[v_bias_name] = qat_export._to_numpy_bytes(
                    params[v_bias_name].detach().to("cpu").float(), "<f4"
                )
        scale_for_input = {
            "q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in",
            "gate_proj": "mlp_in", "up_proj": "mlp_in",
            "o_proj": "o_in", "down_proj": "down_in",
        }
        for key in cold_export.TARGET_SUFFIXES:
            name = names[key]
            weight = weights[key]
            stats = calibration.stats.get(name)
            if stats is None:
                raise SmartExportError(f"калибровка не снимала {name}")
            if gptq:
                hessian = stats.hessian
                if input_scales is not None:
                    scale = input_scales[scale_for_input[key]]
                    hessian = hessian * scale[:, None] * scale[None, :]
                trits, scales, gptq_stats = gptq_quantize(
                    weight, hessian, block_size=block_size, damp=damp,
                )
                layer_report[key] = {"gptq": gptq_stats}
            else:
                quant = quantize_naive(weight, block_size)
                trits, scales = quant["trits"], quant["scales"]
            naive = quantize_naive(weight, block_size)
            changed = int((trits != naive["trits"]).sum().item())
            zero_count = int((trits == 0).sum().item())
            n_trits = int(trits.numel())
            trit_updates[name] = {
                "trits": trits,
                "scales": scales,
                "zero": zero_count,
                "changed": changed,
                "n_trits": n_trits,
            }
            info["changed_trits"] += changed
            info["total_trits"] += n_trits
            layer_report.setdefault(key, {})
            layer_report[key]["changed"] = changed
            layer_report[key]["zero_fraction"] = round(zero_count / n_trits, 6)
        info["layers"].append(layer_report)
    return trit_updates, non_updates, info


def _manifest_entries(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    trit = {
        str(entry.get("name")): entry
        for entry in manifest.get("tensors") or []
        if entry.get("trit_length")
    }
    non = {
        str(entry.get("name")): entry
        for entry in (manifest.get("non_ternary") or {}).get("tensors") or []
        if int(entry.get("length") or 0) > 0
    }
    return trit, non


def apply_updates(
    out_dir: str | Path,
    manifest: dict[str, Any],
    trit_updates: dict[str, dict[str, Any]],
    non_updates: dict[str, bytes],
    *,
    scale_dtype: str = "float32",
    block_size: int = ternary.BLOCK_SIZE,
    echo: Any = print,
) -> dict[str, Any]:
    """Записать триты/масштабы/нормы в уже существующие файлы экспорта.

    Файлы правятся на месте (``r+b``): длина сегментов та же, смещения из
    манифеста не меняются. Каждый записанный сегмент перечитывается и
    сверяется побайтово — иначе лучше упасть, чем отдать битый контейнер.
    """
    out_dir = Path(out_dir)
    trit_path = out_dir / cold_export._TRIT_FILE
    non_path = out_dir / cold_export._NON_TERNARY_FILE
    trit_entries, non_entries = _manifest_entries(manifest)
    unknown = sorted(set(trit_updates) - set(trit_entries))
    if unknown:
        raise SmartExportError(
            f"в манифесте нет сегментов для {', '.join(unknown[:3])}"
        )
    unknown_non = sorted(set(non_updates) - set(non_entries))
    if unknown_non:
        raise SmartExportError(
            f"в манифесте нет fp32-тензоров для {', '.join(unknown_non[:3])}"
        )
    zero_by_tensor: dict[str, float] = {}
    written = 0
    with trit_path.open("r+b") as handle:
        for name, update in trit_updates.items():
            entry = trit_entries[name]
            data, meta = encode_trit_segment_explicit(
                update["trits"], update["scales"],
                block_size=block_size, scale_dtype=scale_dtype,
            )
            expected = int(entry["trit_length"])
            if len(data) != expected:
                raise SmartExportError(
                    f"{name}: сегмент {len(data)} байт, а в манифесте {expected}: "
                    "раскладка не сходится, писать нельзя"
                )
            offset = int(entry["trit_offset"])
            handle.seek(offset)
            handle.write(data)
            handle.seek(offset)
            if handle.read(len(data)) != data:
                raise SmartExportError(f"{name}: сегмент не читается обратно")
            zero_by_tensor[name] = float(update["zero"]) / max(1, int(update["n_trits"]))
            written += 1
            if written % 24 == 0:
                echo(f"[export_smart] записано сегментов: {written}/{len(trit_updates)}")
    non_written = 0
    with non_path.open("r+b") as handle:
        for name, payload in non_updates.items():
            entry = non_entries[name]
            if len(payload) != int(entry["length"]):
                raise SmartExportError(
                    f"{name}: fp32-полезная нагрузка {len(payload)} байт, "
                    f"в манифесте {int(entry['length'])}"
                )
            offset = int(entry["offset"])
            handle.seek(offset)
            handle.write(payload)
            handle.seek(offset)
            if handle.read(len(payload)) != payload:
                raise SmartExportError(f"{name}: fp32-вес не читается обратно")
            non_written += 1
    files = {
        trit_path.name: {
            "sha256": _sha256_file(trit_path),
            "bytes": trit_path.stat().st_size,
        },
        non_path.name: {
            "sha256": _sha256_file(non_path),
            "bytes": non_path.stat().st_size,
        },
    }
    return {"files": files, "zero_by_tensor": zero_by_tensor, "non_written": non_written}


# ---------------------------------------------------------------------------
# Оркестрация: экспорт + умное округление
# ---------------------------------------------------------------------------


@dataclass
class SmartResult:
    out_dir: Path
    manifest_path: Path
    round_mode: str
    manifest: dict[str, Any]
    info: dict[str, Any]
    elapsed_s: float

    def summary(self) -> dict[str, Any]:
        trit = self.manifest.get("files", {}).get("ternary.trit", {})
        non = self.manifest.get("files", {}).get("non_ternary.bin", {})
        return {
            "out_dir": str(self.out_dir),
            "round": self.round_mode,
            "ternary_trit_bytes": int(trit.get("bytes") or 0),
            "non_ternary_bytes": int(non.get("bytes") or 0),
            "ternary_trit_sha256": trit.get("sha256"),
            "non_ternary_sha256": non.get("sha256"),
            "elapsed_s": round(self.elapsed_s, 1),
            **self.info,
        }


def _save_calibration(path: Path, calibration: Calibration) -> None:
    import torch  # noqa: PLC0415

    payload = {
        "format": "eva32.export_smart.calibration",
        "format_version": CALIB_CACHE_VERSION,
        "info": calibration.info,
        "stats": {
            name: {
                "tokens": int(item.tokens),
                "abs_sum": item.abs_sum,
                "hessian": item.hessian,
                "samples": item.samples,
                "sample_tokens": int(item.sample_tokens),
            }
            for name, item in calibration.stats.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_calibration(path: Path) -> Calibration:
    import torch  # noqa: PLC0415

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != CALIB_CACHE_VERSION:
        raise SmartExportError(
            f"{path}: версия кэша {payload.get('format_version')}, ждали "
            f"{CALIB_CACHE_VERSION}"
        )
    stats = {
        name: TensorStats(
            tokens=int(item["tokens"]),
            abs_sum=item["abs_sum"],
            hessian=item["hessian"],
            samples=item["samples"],
            sample_tokens=int(item["sample_tokens"]),
        )
        for name, item in payload["stats"].items()
    }
    return Calibration(stats=stats, info=dict(payload["info"]))


def export_smart(
    model_dir: str | Path,
    out_dir: str | Path,
    *,
    round_mode: str = "naive",
    embed_mode: str = "int8",
    embed_scale: str = "block",
    context_window: int = cold_export.DEFAULT_CONTEXT_WINDOW,
    calib_data: str | Path = DEFAULT_CALIB_DATA,
    calib_lines: int = DEFAULT_CALIB_LINES,
    calib_max_tokens: int = DEFAULT_CALIB_MAX_TOKENS,
    calib_skip: int = DEFAULT_CALIB_SKIP,
    sample_tokens: int = DEFAULT_SAMPLE_TOKENS,
    alphas: Sequence[float] | None = None,
    damp: float = GPTQ_DAMP,
    device: str = "auto",
    calib_cache: str | Path | None = None,
    recalibrate: bool = False,
    force: bool = True,
    echo: Any = print,
) -> SmartResult:
    """Экспортировать модель: наивно, затем (если просили) умно переокруглять.

    1. ``cold_export.export_model`` пишет готовый контейнер и манифест —
       формат, смещения, sha256, non_ternary.bin оттуда.
    2. Для ``awq|gptq|both`` калибровка (или кэш) даёт Гессианы/статистику,
       ``plan_updates`` считает новые триты и свёрнутые fp32-нормы,
       ``apply_updates`` пишет их на место и обновляет манифест.
    3. ``naive`` — это ровно шаг 1: побайтовое совпадение с cold_export
       проверяется тестом.
    """
    started = time.time()
    round_mode = check_round_mode(round_mode)
    check_calib_lines(calib_lines)
    alpha_values = parse_alphas(alphas)
    out_path = Path(out_dir).expanduser().resolve()
    model_path = Path(str(model_dir)).expanduser().resolve()
    if not model_path.is_dir():
        raise SmartExportError(
            f"{model_dir}: нужен локальный каталог HF-модели (сеть запрещена)"
        )
    source = cold_export.open_source(model_path)
    hf_config = cold_export.load_hf_config(model_path)
    cold_export.check_supported_config(hf_config)
    try:
        result = cold_export.export_model(
            source, hf_config, out_path,
            model_name=str(model_path),
            embed_mode=embed_mode,
            embed_scale=embed_scale,
            context_window=context_window,
            force=force,
            echo=echo,
        )
    finally:
        source.close()
    manifest = result.manifest
    info: dict[str, Any] = {
        "embed": embed_mode,
        "embed_scale": embed_scale,
        "context_window": int(context_window),
        "base_export_tool": "tools/cold_export.py",
    }
    if round_mode == "naive":
        info["calibration"] = None
    else:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        if device == "auto":
            device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        if device not in ("cpu", "mps", "cuda"):
            raise SmartExportError(f"--device={device}: cpu|mps|cuda|auto")
        cache_path = (
            Path(calib_cache).expanduser().resolve() if calib_cache else None
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=torch.float32,
            local_files_only=True,
            attn_implementation="eager",
        )
        model.eval()
        calibration: Calibration | None = None
        if cache_path is not None and cache_path.is_file() and not recalibrate:
            echo(f"[export_smart] калибровка из кэша {cache_path}")
            calibration = _load_calibration(cache_path)
            info["calibration_cached"] = True
        if calibration is None:
            echo(
                f"[export_smart] калибровка: {calib_lines} строк "
                f"(пропуск {calib_skip}) по {calib_max_tokens} токенов, {device}"
            )
            model.to(device)
            try:
                calibration = collect_calibration(
                    model, tokenizer,
                    data=calib_data,
                    lines=calib_lines,
                    max_tokens=calib_max_tokens,
                    skip=calib_skip,
                    sample_tokens=sample_tokens,
                    device=device,
                    echo=echo,
                )
            finally:
                model.to("cpu")
            if cache_path is not None:
                echo(f"[export_smart] калибровка -> {cache_path}")
                _save_calibration(cache_path, calibration)
            info["calibration_cached"] = False
        info["calibration"] = dict(calibration.info)
        echo(
            f"[export_smart] калибровка готова: {calibration.info['tokens']} токенов, "
            f"{calibration.info['seconds']} с"
        )
        architecture = dict(manifest["architecture"])
        trit_updates, non_updates, plan_info = plan_updates(
            model, calibration,
            round_mode=round_mode,
            architecture=architecture,
            block_size=int(manifest["quantization"]["block_size"]),
            alphas=alpha_values,
            damp=float(damp),
            echo=echo,
        )
        info["plan"] = plan_info
        scale_dtype = str(manifest["quantization"]["scale_dtype"])
        written = apply_updates(
            out_path, manifest, trit_updates, non_updates,
            scale_dtype=scale_dtype,
            block_size=int(manifest["quantization"]["block_size"]),
            echo=echo,
        )
        info["written"] = {
            "trit_segments": len(trit_updates),
            "non_ternary_tensors": written["non_written"],
            "changed_trit_fraction": round(
                plan_info["changed_trits"] / max(1, plan_info["total_trits"]), 6
            ),
        }
        from qat import export as _qat_export  # noqa: PLC0415

        manifest = build_smart_manifest(
            manifest,
            round_mode=round_mode,
            file_stats=written["files"],
            zero_by_tensor=written["zero_by_tensor"],
            zero_counts={},
            info=info,
        )
        _qat_export.write_json_atomic(result.manifest_path, manifest)
        del trit_updates, non_updates
    elapsed = time.time() - started
    info["elapsed_s"] = round(elapsed, 1)
    sidecar = out_path / "smart.json"
    sidecar.write_text(
        json.dumps(
            {
                "format": "eva32.export_smart.report",
                "format_version": TOOL_VERSION,
                "round": round_mode,
                "model": str(model_path),
                **info,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return SmartResult(
        out_dir=out_path,
        manifest_path=result.manifest_path,
        round_mode=round_mode,
        manifest=manifest,
        info=info,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/export_smart.py",
        description=(
            "Холодный экспорт HF-весов с умным округлением: AWQ-масштабирование "
            "и/или GPTQ/OBQ вместо наивного round(w/absmax). Формат .trit/"
            "non_ternary.bin/манифеста тот же, что у tools/cold_export.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  PYTHONPATH=.pylibs python3.13 tools/export_smart.py \\\n"
            "      --model .hfcache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/"
            "snapshots/<sha> \\\n"
            "      --out export/qwen05b-gptq --round gptq --embed int8\n"
            "  PYTHONPATH=.pylibs python3.13 tools/export_smart.py \\\n"
            "      --model … --out export/qwen05b-both --round both \\\n"
            "      --calib-cache data/smart/calib-0.5b.pt\n"
        ),
    )
    parser.add_argument("--model", required=True,
                        help="локальный каталог HF-модели (сеть не используется)")
    parser.add_argument("--out", required=True, help="каталог для trit/bin/manifest")
    parser.add_argument("--round", choices=ROUND_MODES, default="naive",
                        help="naive — эталон (как cold_export), awq — масштабы "
                             "каналов, gptq — округление по ошибке выхода, "
                             "both — оба")
    parser.add_argument("--embed", choices=cold_export.EMBED_MODES, default="int8",
                        help="таблица словаря: none|trit|int8 (по умолчанию int8 — "
                             "чтобы мерить урон Linear, а не таблицы)")
    parser.add_argument("--embed-scale", choices=cold_export.EMBED_SCALES,
                        default="block")
    parser.add_argument("--context-window", type=int,
                        default=cold_export.DEFAULT_CONTEXT_WINDOW)
    parser.add_argument("--calib-data", default=str(DEFAULT_CALIB_DATA),
                        help="JSONL с парами question/answer для калибровки")
    parser.add_argument("--calib-lines", type=int, default=DEFAULT_CALIB_LINES,
                        help=f"строк калибровки (1..{MAX_CALIB_LINES}, по умолчанию "
                             f"{DEFAULT_CALIB_LINES})")
    parser.add_argument("--calib-max-tokens", type=int,
                        default=DEFAULT_CALIB_MAX_TOKENS,
                        help="обрезать пару в токенах (как PPL, 128)")
    parser.add_argument("--calib-skip", type=int, default=DEFAULT_CALIB_SKIP,
                        help="пропустить первые N строк файла: там текст PPL")
    parser.add_argument("--calib-sample-tokens", type=int,
                        default=DEFAULT_SAMPLE_TOKENS,
                        help="сколько токенов входа хранить для выбора alpha")
    parser.add_argument("--alphas", default=None,
                        help="кандидаты alpha AWQ через запятую "
                             f"(по умолчанию {','.join(str(a) for a in AWQ_ALPHAS)})")
    parser.add_argument("--gptq-damp", type=float, default=GPTQ_DAMP,
                        help="демпфирование Гессиана в GPTQ (доля среднего "
                             "диагонального элемента)")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                        default="auto", help="устройство калибровочного forward")
    parser.add_argument("--calib-cache", default=None,
                        help="файл кэша калибровки: читается, если есть, иначе "
                             "пишется после прогона (экономит минуты между "
                             "вариантами округления)")
    parser.add_argument("--recalibrate", action="store_true",
                        help="пересчитать калибровку даже при готовом кэше")
    parser.add_argument("--force", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    echo = lambda message: print(message, file=sys.stderr, flush=True)  # noqa: E731
    try:
        result = export_smart(
            args.model, args.out,
            round_mode=args.round,
            embed_mode=args.embed,
            embed_scale=args.embed_scale,
            context_window=args.context_window,
            calib_data=args.calib_data,
            calib_lines=args.calib_lines,
            calib_max_tokens=args.calib_max_tokens,
            calib_skip=args.calib_skip,
            sample_tokens=args.calib_sample_tokens,
            alphas=args.alphas,
            damp=args.gptq_damp,
            device=args.device,
            calib_cache=args.calib_cache,
            recalibrate=args.recalibrate,
            echo=echo,
        )
    except (SmartExportError, cold_export.ColdExportError, ternary.TritError) as exc:
        print(f"[export_smart] {exc}")
        return 2
    summary = result.summary()
    print()
    print(f"[export_smart] готово за {result.elapsed_s:.0f} с -> {result.out_dir}")
    print(
        f"[export_smart] round={summary['round']} "
        f"ternary.trit {cold_export._human_bytes(summary['ternary_trit_bytes'])}, "
        f"non_ternary.bin {cold_export._human_bytes(summary['non_ternary_bytes'])}"
    )
    plan = result.info.get("plan") or {}
    if plan:
        print(
            f"[export_smart] изменено тритов против наивного: "
            f"{plan['changed_trits']:,} из {plan['total_trits']:,} "
            f"({100.0 * plan['changed_trits'] / max(1, plan['total_trits']):.1f}%)"
        )
        alphas: dict[str, list[float]] = {}
        for layer in plan.get("layers") or []:
            awq = layer.get("awq") or {}
            for group in ("attn", "mlp", "o", "down"):
                if group in awq:
                    alphas.setdefault(group, []).append(float(awq[group]["alpha"]))
        if alphas:
            for group, values in alphas.items():
                unique = sorted(set(values))
                print(
                    f"[export_smart] alpha {group}: "
                    + ", ".join(f"{value:g}×{values.count(value)}" for value in unique)
                )
    print(
        f"[export_smart] sha256: ternary.trit "
        f"{summary['ternary_trit_sha256'][:16]}…, non_ternary.bin "
        f"{summary['non_ternary_sha256'][:16]}…"
    )
    print(f"[export_smart] отчёт: {result.out_dir / 'smart.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
