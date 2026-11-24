#!/usr/bin/env python3
"""Холодный экспорт: HF-веса (safetensors) -> триты, без обучения.

Зачем
=====

``qat/export.py`` умеет только один источник — чекпоинт обученного прогона
(``model.pt`` с параметризованными весами). Чтобы померить урон от
третизации на готовой Qwen и проверить формат на настоящих весах, нужен
путь «HF safetensors -> ``ternary.trit`` + ``non_ternary.bin`` +
``manifest.json``» без QAT. Этот модуль его делает.

Формат не дублируется: сегменты TRT1 собирает
``qat.export.encode_trit_segment``/``encode_trit_table``, роли тензоров —
``qat.export.classify_state_key``, манифест — ``qat.export.build_manifest``.
``qat/export.py`` не правится.

Память
======

14B bf16 — это 29 ГБ шардов, целиком модель в RAM не влезает. Поэтому:

* формы тензоров читаются из заголовков safetensors (``get_slice``), без
  загрузки данных;
* веса читаются по одному тензору из нужного шарда (``safe_open``), после
  квантования ссылка отпускается — в памяти живёт один тензор, а не модель;
* большие тензоры (таблица словаря, матрицы 14B) квантуются **кусками по
  строкам**: ``_encode_trit_chunked`` повторяет раскладку сегмента, но не
  держит fp32-копию, де-квантованный дубль и проверку ``quantize_torch``
  одновременно на весь тензор. Тест сверяет побайтово, что кусковой путь
  даёт ровно тот же сегмент, что ``qat.export.encode_trit_segment``.

Замер памяти печатается в конце: пик RSS процесса, размер самого большого
шарда и их отношение (критерий задачи — не больше 2× шарда).

Квантование
===========

Только Linear: ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``,
``gate_proj``, ``up_proj``, ``down_proj``. Правило одно с QAT
(``qat/ternary.py``): ``scale = absmax`` на блок 128 вдоль входной
размерности, ``trit = clamp(round_half_away(w / scale), -1, +1)``, масштабы
fp32 (``--scale-dtype``). Нормы, bias и ``lm_head`` — как есть, fp32 в
``non_ternary.bin``. Таблица словаря — по флагу ``--embed none|trit|int8``
(``trit`` пишет её сегментом первой в ``ternary.trit``, ``int8`` — INT8-
сегментом kind 4 в ``non_ternary.bin``; в обоих случаях tied ``lm_head``
ссылается на неё), а гранулярность её масштаба — ``--embed-scale block|row``
(``row`` — один масштаб на строку, разложенный по блокам 128; раскладка
сегмента не меняется).

Манифест
========

Формат ``eva32.ternary.manifest`` v2 — ровно тот, что принимает
``tools/vtm_compile.py --manifest``: ``architecture`` (hidden, heads,
kv_heads, head_dim, intermediate, vocab, n_layers, rope_theta, eps, tie),
``quantization`` (block_size 128, scale_dtype), ``files`` со sha256,
``tensors[]`` со смещениями в ``ternary.trit``, ``non_ternary.tensors[]``
со смещениями в ``non_ternary.bin``. Сверху добавлены ключи, которых
компилятор не знает и игнорирует: ``context_window`` (длина KV-кэша в
контейнере, по умолчанию 1024 — 14B при 2048 не влезает в кэп
``MAX_KV_BYTES`` 512 МиБ), ``source`` (модель/шарды), ``hf_config`` и
``cold_export`` (опции и замер памяти).

Сверка (``--verify``)
=====================

Два forward'а на HF (2–3 промпта): полная точность и та же модель с
третизованными Linear в памяти. Печатается ``max|Δ|`` логитов последней
позиции, совпадение argmax и пересечение top-5 — это урон холодной
третизации, а не расхождение рантайма. Плюс спот-чек артефактов: сегменты
читаются из ``ternary.trit`` и побайтово сверяются с квантованием тех же
весов в памяти (должно быть точное равенство; иначе испорчен файл или
манифест).

Запуск
======

GPU-стек (torch/numpy/safetensors) лежит в ``.pylibs`` и собран под Python
3.13, поэтому экспорт и ``--verify`` запускаются так::

    PYTHONPATH=.pylibs python3.13 tools/cold_export.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct --allow-download \\
        --out export/qwen05b --verify

Дальше компиляция и запуск — без torch (чистый stdlib + C)::

    python3 tools/vtm_compile.py --manifest export/qwen05b/manifest.json \\
        --out export/qwen05b/model.vtm
    src/vtm/vtm run export/qwen05b/model.vtm --tokens 1,2,3 --max-new 1

Быстрая проверка на нескольких слоях: ``--limit-layers 4`` (манифест тогда
описывает частичную модель — она компилируется и исполняется, но это не
полная Qwen). ``--verify`` с ``--limit-layers`` несовместим.

Ошибки (неподдерживаемый конфиг, отсутствие зависимостей, битые шарды) —
код 2 и объяснение, а не молчаливый экспорт «чего получилось».
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import resource
import re
import struct
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _path in (str(_ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from qat import export as qat_export  # noqa: E402
from qat import ternary  # noqa: E402
from qat.config import DEFAULT_TARGET_SUFFIXES  # noqa: E402

try:  # импорт как пакета (tests/*, `python3 -m tools.cold_export`)
    from tools import vtm_ref  # noqa: E402
except ImportError:  # pragma: no cover - запуск как `python3 tools/cold_export.py`
    import vtm_ref  # type: ignore[no-redef]  # noqa: E402

PROJECT_ROOT = _ROOT
TOOL_VERSION = 1
#: Имя манифеста рядом с весами (как у qat.export).
MANIFEST_FILE = "manifest.json"
#: Длина KV-кэша в контейнере VTM по умолчанию.
#:
#: 14B (48 слоёв × kv_dim 1024 × K и V × fp32) при ctx=2048 — 805 МиБ, это
#: больше кэпа ``MAX_KV_BYTES`` (512 МиБ) в ``tools/vtm_ref.write_vtm`` и
#: ``src/vtm``. 1024 даёт 402 МиБ и влезает; для 0.5B/3B контекст можно
#: поднять флагом ``--context-window``.
DEFAULT_CONTEXT_WINDOW = 1024
#: Квантуются только эти Linear (порядок важен для детерминированного файла).
TARGET_SUFFIXES: tuple[str, ...] = tuple(DEFAULT_TARGET_SUFFIXES)
TARGET_NAMES = tuple(f"{name}.weight" for name in TARGET_SUFFIXES)
#: Куда идёт таблица словаря: ``none`` — fp32 в non_ternary.bin, ``trit`` —
#: сегмент TRT1 в ternary.trit, ``int8`` — сегмент INT8 (kind 4) в
#: non_ternary.bin (задача 0143; загрузчик и vtm_compile это уже умеют).
EMBED_MODES = ("none", "trit", "int8")
#: Гранулярность масштаба таблицы: ``block`` — absmax на блок 128 (как у
#: Linear и как писал QAT), ``row`` — один масштаб на строку таблицы,
#: разложенный по всем её блокам. Формат и рантайм от этого не меняются:
#: масштабов по-прежнему ceil(n/128), просто внутри строки они одинаковые.
EMBED_SCALES = ("block", "row")
SCALE_DTYPES = ("float32", "float16")
#: Линей, входная размерность которых обязана делиться на block_size.
#: Формат .trit хранит один масштаб на block_size подряд идущих весов и не
#: умеет row-padding, поэтому q/k/v/gate/up требуют block_size | hidden,
#: o_proj — block_size | q_dim, down_proj — block_size | intermediate.
#: Сколько элементов в куске при кусковом квантовании больших тензоров.
#: 4 Ми элементов fp32 — это 16 МиБ входа и ~64 МиБ рабочих копий, то есть
#: с запасом меньше самого маленького шарда 14B (~5 ГиБ).
CHUNK_ELEMENTS = 4 << 20
#: До этого размера тензор квантуется целиком через qat.export (одна копия,
#: ровно тот код, что в QAT-экспорте). Выше — кусками (см. _encode_trit_chunked).
CHUNK_MIN_ELEMENTS = 8 << 20
#: Тензоры, которые в HF-чекпоинтах встречаются, но к графу VTM отношения не
#: имеют: старые чекпоинты хранят предвычисленный inv_freq RoPE.
IGNORED_KEY_SUFFIXES = (
    "rotary_emb.inv_freq",
    "rotary_emb.cos_cached",
    "rotary_emb.sin_cached",
)
#: Нетернарные веса по умолчанию пустые: без этого кодек не собрать.
_EMBED_NAME = qat_export.EMBED_TENSOR_NAME
_NON_TERNARY_FILE = qat_export.NON_TERNARY_FILE
_TRIT_FILE = "ternary.trit"
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")


class ColdExportError(RuntimeError):
    """Неподдерживаемый конфиг/данные: код возврата 2, без трейсбека."""


# ---------------------------------------------------------------------------
# Окружение
# ---------------------------------------------------------------------------


def missing_modules() -> list[str]:
    """Чего не хватает для настоящего экспорта (не для тестов)."""
    missing: list[str] = []
    for module, package in (("torch", "torch"), ("safetensors", "safetensors")):
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(package)
    return missing


MISSING_HINT = (
    "холодному экспорту нужны torch и safetensors. В этом репозитории они\n"
    "лежат в .pylibs и собраны под Python 3.13:\n"
    "    PYTHONPATH=.pylibs python3.13 tools/cold_export.py --model ... --out ...\n"
    "Тесты (python3 -m unittest tests.test_cold_export) работают без torch."
)


def _human_bytes(size: float) -> str:
    value = float(size)
    for unit in ("Б", "КиБ", "МиБ", "ГиБ", "ТиБ"):
        if value < 1024.0 or unit == "ТиБ":
            return f"{value:.1f} {unit}" if unit != "Б" else f"{int(value)} Б"
        value /= 1024.0
    return f"{int(size)} Б"


def _peak_rss_bytes() -> int:
    """Пик RSS процесса: на macOS ru_maxrss в байтах, на Linux — в КиБ."""
    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage if sys.platform == "darwin" else usage * 1024


def _reset_peak_rss() -> None:
    """Сбросить пик RSS, чтобы замер относился к экспорту, а не к импорту torch."""
    try:  # pragma: no cover - зависит от платформы
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib")
        if sys.platform == "darwin" and hasattr(libc, "malloc_zone_pressure_relief"):
            # macOS не умеет сбрасывать ru_maxrss; пик всё равно останется
            # максимумом процесса, поэтому замер честно называется «пик процесса».
            return
        if sys.platform.startswith("linux"):
            # Linux: ru_maxrss из getrusage — максимум за жизнь процесса,
            # сброса нет; замер называем «пик процесса».
            return
    except Exception:  # noqa: BLE001 - замер не должен ломать экспорт
        return


# ---------------------------------------------------------------------------
# Источники весов: шарды safetensors и словарь (для тестов)
# ---------------------------------------------------------------------------


class WeightSource:
    """Минимум, который нужен экспорту от источника весов.

    Задача интерфейса — не дать экспорту «загрузить модель целиком»: тензоры
    выдаются по одному, формы читаются отдельно и дешево.
    """

    def keys(self) -> list[str]:  # pragma: no cover - интерфейс
        raise NotImplementedError

    def shapes(self) -> dict[str, tuple[int, ...]]:  # pragma: no cover
        raise NotImplementedError

    def tensor(self, key: str) -> Any:  # pragma: no cover
        raise NotImplementedError

    def release(self, key: str) -> None:
        """Отпустить ссылки на тензор после квантования (по умолчанию — gc)."""
        gc.collect()

    def describe(self) -> dict[str, Any]:
        return {"kind": type(self).__name__}

    def close(self) -> None:
        self.release("")


class DictSource(WeightSource):
    """Источник из обычного словаря тензоров — для тестов без torch и диска."""

    def __init__(self, state: dict[str, Any], label: str = "<dict>") -> None:
        self._state = dict(state)
        self._label = label

    def keys(self) -> list[str]:
        return sorted(self._state)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            key: tuple(int(x) for x in value.shape)
            for key, value in self._state.items()
            if hasattr(value, "shape")
        }

    def tensor(self, key: str) -> Any:
        if key not in self._state:
            raise ColdExportError(f"{self._label}: нет тензора {key}")
        return self._state[key]

    def release(self, key: str) -> None:
        # Словарь — собственность теста: не выкидываем, только предлагаем gc.
        gc.collect()

    def describe(self) -> dict[str, Any]:
        return {"kind": "dict", "label": self._label}

    def close(self) -> None:
        self._state.clear()


def _safetensors_index(model_dir: Path) -> dict[str, Path] | None:
    """Разобрать model.safetensors.index.json: тензор -> шард."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ColdExportError(f"{index_path}: не читается ({exc})") from exc
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ColdExportError(f"{index_path}: нет weight_map — это не safetensors-индекс")
    result: dict[str, Path] = {}
    for name, shard in weight_map.items():
        shard_path = model_dir / str(shard)
        if not shard_path.exists():
            raise ColdExportError(
                f"{index_path}: шард {shard} для {name} отсутствует в {model_dir}"
            )
        result[str(name)] = shard_path
    return result


class ShardSource(WeightSource):
    """Потоковое чтение safetensors по шардам.

    ``safe_open`` не грузит файл: тензор попадает в память только в
    ``tensor()``. Шард открывается один раз (mmap) и закрывается в
    ``close()``; одновременно открыто столько шардов, сколько нужно по ходу
    канонического порядка (обычно один-два — ключи слоёв лежат подряд).
    """

    def __init__(self, model_dir: Path | str) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.model_dir.is_dir():
            raise ColdExportError(f"{self.model_dir}: не каталог с HF-моделью")
        try:
            from safetensors import safe_open  # noqa: PLC0415 - опционален для тестов
        except ImportError as exc:
            raise ColdExportError(MISSING_HINT) from exc
        self._safe_open = safe_open
        index = _safetensors_index(self.model_dir)
        if index is None:
            files = sorted(self.model_dir.glob("*.safetensors"))
            if not files:
                raise ColdExportError(
                    f"{self.model_dir}: нет *.safetensors (ни индекса, ни шардов)"
                )
            # Без индекса раскладываем ключи по файлам сами.
            index = {}
            for path in files:
                with safe_open(str(path), framework="pt") as handle:
                    for key in handle.keys():
                        index[key] = path
        self.index: dict[str, Path] = index
        self._handles: dict[Path, Any] = {}
        self._stack = ExitStack()
        self._shapes: dict[str, tuple[int, ...]] | None = None

    # -- WeightSource -------------------------------------------------------
    def keys(self) -> list[str]:
        return sorted(self.index)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        if self._shapes is not None:
            return self._shapes
        shapes: dict[str, tuple[int, ...]] = {}
        for path in sorted(set(self.index.values())):
            handle = self._handle(path)
            for key in handle.keys():
                shapes[key] = tuple(int(x) for x in handle.get_slice(key).get_shape())
        self._shapes = shapes
        return shapes

    def tensor(self, key: str) -> Any:
        path = self.index.get(key)
        if path is None:
            raise ColdExportError(f"{self.model_dir}: нет тензора {key}")
        return self._handle(path).get_tensor(key)

    def describe(self) -> dict[str, Any]:
        shards = sorted(set(self.index.values()))
        return {
            "kind": "safetensors",
            "dir": str(self.model_dir),
            "shards": [
                {"name": path.name, "bytes": path.stat().st_size} for path in shards
            ],
            "largest_shard_bytes": max(
                (path.stat().st_size for path in shards), default=0
            ),
        }

    def close(self) -> None:
        self._stack.close()
        self._handles.clear()

    # -- внутреннее ---------------------------------------------------------
    def _handle(self, path: Path) -> Any:
        handle = self._handles.get(path)
        if handle is None:
            handle = self._stack.enter_context(
                self._safe_open(str(path), framework="pt")
            )
            self._handles[path] = handle
        return handle


def open_source(model: str | Path, *, allow_download: bool = False,
                cache_dir: str | Path | None = None) -> WeightSource:
    """Открыть источник: локальный каталог или id на HF (только по флагу)."""
    path = Path(str(model)).expanduser()
    if path.is_dir():
        return ShardSource(path)
    if path.exists():
        raise ColdExportError(f"{path}: это файл, а нужен каталог HF-модели")
    if not allow_download:
        raise ColdExportError(
            f"{model}: локального каталога нет. Укажите путь к скачанной модели\n"
            "или добавьте --allow-download (скачает snapshot в HF-кэш)."
        )
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:
        raise ColdExportError(
            f"для --allow-download нужен huggingface_hub ({exc})"
        ) from exc
    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    print(f"[cold_export] качаю {model} в HF-кэш (это может занять время)…",
          file=sys.stderr, flush=True)
    local = snapshot_download(
        str(model),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.model"],
        **kwargs,
    )
    print(f"[cold_export] snapshot: {local}", file=sys.stderr, flush=True)
    return ShardSource(local)


# ---------------------------------------------------------------------------
# HF-конфиг и проверка поддерживаемости
# ---------------------------------------------------------------------------


def load_hf_config(model_dir: Path | str) -> dict[str, Any]:
    """Прочитать config.json без transformers (нужны только числа)."""
    path = Path(model_dir) / "config.json"
    if not path.exists():
        raise ColdExportError(f"{path}: нет config.json — это не HF-модель")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ColdExportError(f"{path}: не читается ({exc})") from exc
    if not isinstance(config, dict):
        raise ColdExportError(f"{path}: конфиг должен быть JSON-объектом")
    return config


def check_supported_config(config: dict[str, Any],
                           architectures: Sequence[str] | None = None) -> None:
    """Отказаться от конфигов, которые VTM v0.1 не исполнит.

    Проверяется только то, что действительно меняет граф: активация (VTM
    умеет SiLU), rope_scaling/partial_rotary_factor (RoPE в v0.1 без
    масштабирования), sliding window короче контекста. Числа (hidden,
    heads, eps) дальше выводит ``qat.export.infer_architecture`` — там же
    отказ, если слои не бьются с формой весов.
    """
    model_type = str(config.get("model_type") or "")
    archs = [str(a) for a in (architectures or config.get("architectures") or [])]
    if model_type and model_type not in ("qwen2", "qwen2_5", "llama", "qwen3", ""):
        raise ColdExportError(
            f"model_type={model_type!r}: холодный экспорт поддерживает "
            "Qwen2/Qwen2.5/Llama-совместимые декодеры (VTM v0.1)"
        )
    if archs and not any(
        "CausalLM" in arch for arch in archs
    ):
        raise ColdExportError(
            f"architectures={archs}: нужна *ForCausalLM (декодер)"
        )
    act = str(config.get("hidden_act") or "silu").lower()
    if act not in ("silu", "swish"):
        raise ColdExportError(
            f"hidden_act={act!r}: VTM v0.1 исполняет только SiLU (SwiGLU)"
        )
    rope_scaling = config.get("rope_scaling")
    if rope_scaling:
        raise ColdExportError(
            f"rope_scaling={rope_scaling!r}: VTM v0.1 умеет только обычный RoPE "
            "без масштабирования; такая модель исполнится неверно, поэтому экспорт "
            "отказывается, а не пишет заведомо неверный контейнер"
        )
    factor = config.get("partial_rotary_factor")
    if factor is not None and float(factor) != 1.0:
        raise ColdExportError(
            f"partial_rotary_factor={factor}: VTM v0.1 вращает всю голову"
        )
    # Qwen2.5 держит sliding_window/max_window_layers в конфиге по умолчанию,
    # но включает окно только флагом use_sliding_window (у 0.5B/3B/7B/14B он
    # false). Отказываем только когда окно действительно включено.
    window = config.get("sliding_window")
    use_window = bool(config.get("use_sliding_window"))
    if window and use_window:
        raise ColdExportError(
            f"sliding_window={window} при use_sliding_window=true: VTM v0.1 не "
            "умеет скользящее окно и посчитал бы неверное внимание"
        )


def _blocked_dims(architecture: dict[str, Any]) -> list[tuple[str, int]]:
    """Входные размерности Linear: (что это, сколько). Должны делиться на блок.

    Блок идёт вдоль входной размерности (cols), поэтому важны только они:
    hidden у q/k/v/gate/up, q_dim у o_proj, intermediate у down_proj. Число
    строк (kv_dim, vocab, intermediate у gate/up) на блок не влияет — формат
    пишет масштаб на блок внутри строки. Таблица словаря тоже не требует
    кратности vocab.
    """
    hidden = int(architecture["hidden"])
    q_dim = int(architecture["n_heads"]) * int(architecture["head_dim"])
    inter = int(architecture["intermediate"])
    return [
        ("hidden (q/k/v/gate/up)", hidden),
        ("q_dim (o_proj)", q_dim),
        ("intermediate (down_proj)", inter),
    ]


def check_block_alignment(architecture: dict[str, Any], block_size: int,
                          embed_mode: str = "none",
                          embed_scale: str = "block") -> None:
    """Формат .trit не умеет row-padding: вход Linear обязан делиться на блок.

    Для таблицы словаря то же требование появляется в вариантах 0143:
    ``int8`` (масштаб по плоскому буферу, блоки не должны переезжать через
    строку) и ``row`` (масштаб строки раскладывается по её блокам). При
    ``embed none``/``trit``+``block`` раскладка прежняя и проверок нет.
    """
    bad = [
        (name, value) for name, value in _blocked_dims(architecture)
        if value % block_size
    ]
    if bad:
        details = ", ".join(f"{name}={value}" for name, value in bad)
        raise ColdExportError(
            f"размерность не делится на block_size={block_size}: {details}. "
            "Формат .trit хранит один масштаб на блок подряд идущих весов и не "
            "умеет padding по строкам; смените модель или --block-size"
        )
    hidden = int(architecture["hidden"])
    if embed_mode not in (None, "", "none") and (
        embed_mode == "int8" or embed_scale == "row"
    ) and hidden % block_size:
        raise ColdExportError(
            f"hidden={hidden} не делится на block_size={block_size}: "
            f"таблица словаря с --embed {embed_mode} --embed-scale {embed_scale} "
            "требует, чтобы блок 128 не переезжал через границу строки"
        )


def _architectures_from_shapes(shapes: dict[str, tuple[int, ...]]) -> list[dict[str, Any]]:
    """Какие тензоры есть в шардах: для отчёта о неподдержанных ключах."""
    return [{"name": key, "shape": list(shape)} for key, shape in sorted(shapes.items())]


# ---------------------------------------------------------------------------
# Архитектура
# ---------------------------------------------------------------------------


def infer_export_architecture(
    shapes: dict[str, tuple[int, ...]],
    hf_config: dict[str, Any],
    *,
    limit_layers: int | None = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    embed_mode: str = "none",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Вывести architecture манифеста через ``qat.export.infer_architecture``.

    ``limit_layers`` обрезает и формы, и конфиг: иначе проверка «число
    слоёв в конфиге против state_dict» справедливо упадёт.
    """
    if embed_mode not in EMBED_MODES:
        raise ColdExportError(
            f"--embed {embed_mode!r}: поддержано {', '.join(EMBED_MODES)}"
        )
    config = dict(hf_config or {})
    trimmed = dict(shapes)
    if limit_layers is not None:
        if limit_layers <= 0:
            raise ColdExportError(f"--limit-layers={limit_layers}: нужно > 0")
        layers: set[int] = set()
        for name in trimmed:
            match = _LAYER_RE.search(name)
            if match:
                layers.add(int(match.group(1)))
        if not layers:
            raise ColdExportError("в шардах нет номеров слоёв model.layers.N")
        available = max(layers) + 1
        if limit_layers > available:
            raise ColdExportError(
                f"--limit-layers={limit_layers}, а в шардах слоёв {available}"
            )
        trimmed = {
            name: shape for name, shape in trimmed.items()
            if not (m := _LAYER_RE.search(name)) or int(m.group(1)) < limit_layers
        }
        config["num_hidden_layers"] = limit_layers
        print(
            f"[cold_export] --limit-layers={limit_layers}: манифест описывает "
            f"{limit_layers} слоёв из {available}",
            file=sys.stderr,
            flush=True,
        )
    tie_hint = hf_config.get("tie_word_embeddings")
    tie_hint = bool(tie_hint) if tie_hint is not None else None
    merged_overrides = {"context_len": int(context_window)}
    merged_overrides.update(overrides or {})
    try:
        architecture = qat_export.infer_architecture(
            trimmed,
            config,
            tie_hint=tie_hint,
            overrides=merged_overrides,
            lm_head_trit=False,
            embed_mode=embed_mode,
        )
    except RuntimeError as exc:
        raise ColdExportError(f"архитектура не выводится: {exc}") from exc
    return architecture


# ---------------------------------------------------------------------------
# Квантование: один тензор -> сегмент TRT1
# ---------------------------------------------------------------------------


def _is_torch_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "numel") and hasattr(value, "shape")


def _chunk_rows(rows: int, cols: int) -> int:
    """Сколько строк квантовать за раз, чтобы границы слов не разъехались.

    Упаковка идёт по 20 тритов, а массив слов выравнивается по 160 тритов.
    Кусок из целых строк с числом элементов, кратным 160, склеивается со
    следующим без сдвига: конкатенация слов равна упаковке всего тензора.
    """
    # 160 | rows_per_chunk * cols  <=>  rows_per_chunk кратно 160/gcd(160, cols)
    unit = ternary.TRITS_PER_STORAGE_BLOCK // math.gcd(ternary.TRITS_PER_STORAGE_BLOCK, cols)
    unit = max(1, unit)
    # Добиваем до CHUNK_ELEMENTS, сохраняя кратность unit.
    wanted = max(1, CHUNK_ELEMENTS // max(1, cols))
    chunks = max(1, math.ceil(wanted / unit))
    return min(rows, max(unit, chunks * unit))


def _chunked_segment(tensor: Any, name: str, *, block_size: int,
                     scale_dtype: str) -> tuple[bytes, dict[str, Any]]:
    """Собрать сегмент TRT1 из torch-тензора кусками по строкам.

    Повторяет раскладку ``qat.export._encode_trit_segment_torch`` (заголовок
    ``TRT1``, масштабы fp32/fp16, слова по 20 тритов, добивка до 160), но не
    держит в памяти fp32-копию всего тензора. Тест
    ``test_chunked_matches_single_shot`` сверяет результат побайтово с
    ``qat.export.encode_trit_segment``.
    """
    import torch  # noqa: PLC0415 - вызывается только на настоящих весах

    shape = [int(x) for x in tensor.shape]
    if len(shape) != 2:
        raise ColdExportError(
            f"{name}: кусковое квантование только для 2-D матриц, форма {shape}"
        )
    rows, cols = shape
    if cols % block_size:
        raise ColdExportError(
            f"{name}: входная размерность {cols} не кратна block_size {block_size}"
        )
    per_row_blocks = cols // block_size
    step = _chunk_rows(rows, cols)

    scale_fmt = "<f4" if scale_dtype == "float32" else "<f2"
    torch_scale_dtype = torch.float32 if scale_dtype == "float32" else torch.float16
    scales_parts: list[bytes] = []
    words_parts: list[bytes] = []
    n_trits = 0
    zero_trits = 0
    for start in range(0, rows, step):
        chunk = tensor[start:start + step]
        n_rows = int(chunk.shape[0])
        quant = ternary.quantize_torch(chunk.detach().float(), block_size)
        trits = quant["trits"].cpu().contiguous()
        scales = quant["scales"].reshape(-1).cpu()
        expected = n_rows * per_row_blocks
        if int(scales.numel()) != expected:
            raise ColdExportError(
                f"{name}: масштабов {int(scales.numel())}, ожидалось {expected}"
            )
        scales_out = scales.to(torch_scale_dtype)
        if not bool(torch.isfinite(scales_out).all()):
            raise ColdExportError(f"{name}: масштаб не помещается в {scale_dtype}")
        words = ternary.pack20_torch(trits)
        zero_trits += int((trits == 0).sum().item())
        n_trits += int(trits.numel())
        scales_parts.append(qat_export._to_numpy_bytes(scales_out, scale_fmt))
        words_parts.append(qat_export._to_numpy_bytes(words, "<u4"))
        del chunk, quant, trits, scales, scales_out, words

    storage_blocks = (
        n_trits + ternary.TRITS_PER_STORAGE_BLOCK - 1
    ) // ternary.TRITS_PER_STORAGE_BLOCK
    total_scale_blocks = n_trits // block_size
    if total_scale_blocks * block_size < n_trits:
        raise ColdExportError(f"{name}: масштабов не хватает на все триты")
    words = b"".join(words_parts)
    expected_words = storage_blocks * ternary.WORDS_PER_STORAGE_BLOCK * 4
    if len(words) > expected_words:
        raise ColdExportError(
            f"{name}: слов {len(words)} байт, а блоков {storage_blocks} даёт "
            f"{expected_words}"
        )
    if len(words) < expected_words:
        words += b"\x00" * (expected_words - len(words))
    flags = ternary.FLAG_SCALE_F32 if scale_dtype == "float32" else 0
    header = struct.pack(
        "<4sBBQI", ternary.MAGIC, ternary.VERSION, flags, n_trits, total_scale_blocks
    )
    data = header + b"".join(scales_parts) + words
    info = ternary.segment_info(data, block_size)
    if info.n_scale_blocks != total_scale_blocks or info.n_words != storage_blocks * ternary.WORDS_PER_STORAGE_BLOCK:
        raise ColdExportError(f"{name}: заголовок сегмента не сошёлся с раскладкой")
    meta = {
        "params": n_trits,
        "n_trits": n_trits,
        "n_scale_blocks": total_scale_blocks,
        "scale_blocks_per_row": per_row_blocks,
        "n_words": (n_trits + ternary.TRITS_PER_WORD - 1) // ternary.TRITS_PER_WORD,
        "n_storage_blocks": storage_blocks,
        "n_storage_words": storage_blocks * ternary.WORDS_PER_STORAGE_BLOCK,
        "shape": shape,
        "dtype": "trit",
        "kind": "trit",
        "zero_fraction": round(zero_trits / n_trits, 6) if n_trits else 0.0,
        "chunk_rows": step,
    }
    return data, meta


def encode_trit_segment(
    weight: Any,
    name: str,
    *,
    block_size: int = ternary.BLOCK_SIZE,
    scale_dtype: str = "float32",
) -> tuple[bytes, dict[str, Any]]:
    """Сегмент TRT1 из веса: torch — с кусками, fake-тензор — чистый путь.

    Малые тензоры отдаются прямо в ``qat.export.encode_trit_segment`` (ровно
    код QAT-экспорта). Большие (таблица словаря, матрицы 14B) идут через
    ``_chunked_segment``: иначе fp32-копия + де-квантованный дубль +
    проверка ``quantize_torch`` занимают несколько ГБ на один тензор.
    """
    if not _is_torch_tensor(weight):
        # Чистый путь qat.export (fake-тензоры в тестах, без torch).
        data, meta, _payload = qat_export.encode_trit_segment(
            weight, name, block_size=block_size, scale_dtype=scale_dtype,
        )
        return data, meta
    numel = int(weight.numel())
    if numel <= CHUNK_MIN_ELEMENTS:
        data, meta, _payload = qat_export.encode_trit_segment(
            weight, name, block_size=block_size, scale_dtype=scale_dtype,
        )
        return data, meta
    return _chunked_segment(weight, name, block_size=block_size,
                            scale_dtype=scale_dtype)


# ---------------------------------------------------------------------------
# Таблица словаря: trit/int8 и посрочные масштабы (задача 0143)
# ---------------------------------------------------------------------------
#
# Формат контейнера не меняется ни в одном варианте:
#
# * ``trit`` — тот же TRT1, что у Linear (20 тритов в слово, 160 на блок
#   хранения), масштабов по-прежнему ceil(n/128);
# * ``int8`` — INT8-сегмент kind 4 (магия ``INT8``, 18-байтовый заголовок,
#   масштабы, знаковые q-байты), его уже читают ``vtm_compile``, C-загрузчик
#   и Python-эталон;
# * ``--embed-scale row`` — масштаб считается по строке таблицы и
#   раскладывается по её блокам повтором. Для рантайма это обычные
#   масштабы блока 128: меняется только их значение, а не раскладка. Поэтому
#   старые контейнеры читаются как раньше, а новый вариант виден только по
#   содержимому сегмента (и по ключу ``embed_scale`` в манифесте).
#
# Посрочный режим требует ``hidden % block_size == 0``: только тогда блок 128
# не пересекает границу строки и один масштаб строки можно разложить по её
# блокам, не меняя байтовой раскладки сегмента.

#: Нулевой блок/строка в int8: так делает QAT (``Int8Embedding``).
INT8_ZERO_SCALE = 1e-8 / 127.0


def _row_scale(rows: int, per_row_blocks: int) -> int:
    return rows * per_row_blocks


def _quantize_table_torch(
    chunk: Any, *, mode: str, scale_mode: str, block_size: int,
) -> tuple[Any, Any]:
    """Триты/int8 и плоские масштабы куска таблицы (torch-путь).

    ``mode="trit"``, ``scale_mode="block"`` — ровно правило QAT
    (``qat.ternary.quantize_torch``), поэтому байты такого сегмента совпадают
    с прежним экспортом. Остальные комбинации — новые варианты 0143.
    """
    import torch  # noqa: PLC0415 - вызывается только на настоящих весах

    w = chunk.detach().float()
    rows, cols = int(w.shape[0]), int(w.shape[1])
    per_row_blocks = cols // block_size
    if mode not in ("trit", "int8"):
        raise ColdExportError(f"таблица словаря: режим {mode!r}, ждали trit|int8")
    if scale_mode not in EMBED_SCALES:
        raise ColdExportError(f"масштаб таблицы: {scale_mode!r}, ждали {EMBED_SCALES}")
    if cols % block_size:
        raise ColdExportError(
            f"{cols} столбцов таблицы не кратны block_size={block_size}: "
            "раскладка масштабов по блокам не совпадёт со строками"
        )
    if scale_mode == "block" and mode == "trit":
        quant = ternary.quantize_torch(w, block_size)
        trits = quant["trits"].cpu().contiguous()
        # trits того же shape, что w; на всякий случай приводим к 2-D.
        trits = trits.reshape(rows, cols)
        scales = quant["scales"].reshape(-1).cpu()
        return trits, scales
    if scale_mode == "block":
        blocks = w.reshape(rows, per_row_blocks, block_size)
        absmax = blocks.abs().amax(dim=-1)
        scales = torch.where(
            absmax > 0, absmax / 127.0, torch.full_like(absmax, INT8_ZERO_SCALE)
        )
        q = torch.round(blocks / scales.unsqueeze(-1)).clamp(-127.0, 127.0)
        return q.reshape(rows, cols), scales.reshape(-1)
    # --- row: один масштаб на строку ---------------------------------------
    absmax = w.abs().amax(dim=1)
    if mode == "int8":
        scale_row = torch.where(
            absmax > 0, absmax / 127.0, torch.full_like(absmax, INT8_ZERO_SCALE)
        )
        q = torch.round(w / scale_row.unsqueeze(1)).clamp(-127.0, 127.0)
        q = torch.where((absmax > 0).unsqueeze(1), q, torch.zeros_like(q))
    else:
        safe = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
        ratio = w / safe.unsqueeze(1)
        q = torch.sign(ratio) * torch.floor(ratio.abs() + 0.5)
        q = q.clamp(-1.0, 1.0)
        q = torch.where((absmax > 0).unsqueeze(1), q, torch.zeros_like(q))
        scale_row = absmax  # нулевая строка -> нулевой масштаб (как у QAT)
    expanded = scale_row.unsqueeze(1).expand(rows, per_row_blocks)
    return q.reshape(rows, cols), expanded.reshape(-1).contiguous()


def _plain_table_quant(
    values: Sequence[float], rows: int, cols: int, *,
    mode: str, scale_mode: str, block_size: int,
) -> tuple[list[int], list[float]]:
    """То же правило без torch (fake-тензоры в тестах, чистый Python).

    Числа округляются до float32 на каждом шаге, как в torch-пути: тест
    сверяет, что оба пути дают одинаковые триты/q и масштабы.
    """
    if cols % block_size:
        raise ColdExportError(
            f"{cols} столбцов таблицы не кратны block_size={block_size}"
        )
    per_row_blocks = cols // block_size
    f32 = vtm_ref._f32
    vals = [f32(float(v)) for v in values]
    if len(vals) != rows * cols:  # pragma: no cover - защита от неверной shape
        raise ColdExportError(
            f"таблица: {len(vals)} значений на shape {(rows, cols)}"
        )
    quant: list[int] = [0] * len(vals)
    scales: list[float] = []
    for row in range(rows):
        base = row * cols
        if scale_mode == "block":
            for blk in range(per_row_blocks):
                start = base + blk * block_size
                block = vals[start:start + block_size]
                absmax = f32(max((abs(v) for v in block), default=0.0))
                if mode == "int8":
                    scale = f32(absmax / 127.0) if absmax > 0 else f32(INT8_ZERO_SCALE)
                else:
                    scale = absmax
                scales.append(scale)
                for offset, value in enumerate(block):
                    index = start + offset
                    if scale == 0.0:
                        quant[index] = 0
                    elif mode == "int8":
                        ratio = f32(value / scale)
                        quant[index] = max(-127, min(127, int(round(ratio))))
                    else:
                        ratio = f32(value / scale)
                        quant[index] = max(-1, min(1, int(_round_half_away(ratio))))
            continue
        row_values = vals[base:base + cols]
        absmax = f32(max((abs(v) for v in row_values), default=0.0))
        if mode == "int8":
            scale = f32(absmax / 127.0) if absmax > 0 else f32(INT8_ZERO_SCALE)
        else:
            scale = absmax
        scales.extend([scale] * per_row_blocks)
        for offset, value in enumerate(row_values):
            index = base + offset
            if scale == 0.0:
                quant[index] = 0
            elif mode == "int8":
                ratio = f32(value / scale)
                quant[index] = max(-127, min(127, int(round(ratio))))
            else:
                ratio = f32(value / scale)
                quant[index] = max(-1, min(1, int(_round_half_away(ratio))))
    return quant, scales


def _round_half_away(value: float) -> int:
    """Округление «половина от нуля», как ``torch.sign(x)*floor(|x|+0.5)``."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def _segment_parts(
    *, mode: str, n_values: int, scale_dtype: str, block_size: int,
) -> tuple[bytes, int]:
    """Заголовок сегмента и число масштабов для n_values."""
    flags = ternary.FLAG_SCALE_F32 if scale_dtype == "float32" else 0
    if mode == "trit":
        storage_blocks = (
            n_values + ternary.TRITS_PER_STORAGE_BLOCK - 1
        ) // ternary.TRITS_PER_STORAGE_BLOCK
        total_scale_blocks = n_values // block_size
        if total_scale_blocks * block_size < n_values:  # pragma: no cover
            raise ColdExportError(
                f"таблица: масштабов не хватает на {n_values} тритов"
            )
        header = struct.pack(
            "<4sBBQI", ternary.MAGIC, ternary.VERSION, flags, n_values,
            total_scale_blocks,
        )
        return header, total_scale_blocks
    n_scales = (n_values + vtm_ref.INT8_SCALE_BLOCK - 1) // vtm_ref.INT8_SCALE_BLOCK
    header = struct.pack(
        "<4sBBQI", vtm_ref.INT8_MAGIC, vtm_ref.INT8_VERSION, flags, n_values,
        n_scales,
    )
    return header, n_scales


def encode_embed_segment(
    weight: Any,
    name: str = _EMBED_NAME,
    *,
    mode: str = "trit",
    scale_mode: str = "block",
    block_size: int = ternary.BLOCK_SIZE,
    scale_dtype: str = "float32",
) -> tuple[bytes, dict[str, Any]]:
    """Сегмент таблицы словаря: TRT1 (``trit``) или INT8 (``int8``).

    Возвращает ``(payload, meta)``, как ``encode_trit_segment``: payload
    кладётся в ``ternary.trit`` (trit) или ``non_ternary.bin`` (int8), meta —
    в запись манифеста.

    Куски по строкам: таблица Qwen — это 136 Ми элементов (0.5B) и 311 Ми
    (3B), держать её fp32-копию, триты, упакованные слова и де-квантованный
    дубль одновременно нельзя.
    """
    if mode not in ("trit", "int8"):
        raise ColdExportError(f"--embed={mode!r}: таблица бывает trit|int8")
    if scale_mode not in EMBED_SCALES:
        raise ColdExportError(
            f"--embed-scale={scale_mode!r}: поддержано {', '.join(EMBED_SCALES)}"
        )
    shape = [int(x) for x in weight.shape]
    if len(shape) != 2:
        raise ColdExportError(f"{name}: таблица должна быть 2-D, а не {shape}")
    rows, cols = shape
    per_row_blocks = cols // block_size
    if cols % block_size:
        raise ColdExportError(
            f"{name}: {cols} столбцов не кратны block_size={block_size}; "
            "посрочный/блочный масштаб таблицы лёг бы поперёк строк"
        )
    torch_path = _is_torch_tensor(weight)
    step = _chunk_rows(rows, cols) if torch_path else rows
    scale_fmt = "<f4" if scale_dtype == "float32" else "<f2"
    torch_scale_dtype = None
    if torch_path:
        import torch  # noqa: PLC0415

        torch_scale_dtype = (
            torch.float32 if scale_dtype == "float32" else torch.float16
        )
    scales_parts: list[bytes] = []
    payload_parts: list[bytes] = []
    n_values = 0
    zero_count = 0
    for start in range(0, rows, step):
        if torch_path:
            chunk = weight[start:start + step]
            n_rows = int(chunk.shape[0])
            quant, scales = _quantize_table_torch(
                chunk, mode=mode, scale_mode=scale_mode, block_size=block_size,
            )
        else:
            # fake-тензор: одна порция целиком (тесты без torch, таблица мелкая)
            chunk = None
            n_rows = min(step, rows - start)
            quant, scales = _plain_table_quant(
                list(weight), n_rows, cols, mode=mode, scale_mode=scale_mode,
                block_size=block_size,
            )
        expected = n_rows * per_row_blocks
        if len(scales) != expected:
            raise ColdExportError(
                f"{name}: масштабов {len(scales)}, ожидалось {expected}"
            )
        if torch_path:
            scales_out = scales.to(torch_scale_dtype)
            if not bool(torch.isfinite(scales_out).all()):
                raise ColdExportError(f"{name}: масштаб не помещается в {scale_dtype}")
            scales_parts.append(qat_export._to_numpy_bytes(scales_out, scale_fmt))
            if mode == "trit":
                words = ternary.pack20_torch(quant.to(torch.int8).contiguous())
                payload_parts.append(qat_export._to_numpy_bytes(words, "<u4"))
                zero_count += int((quant == 0).sum().item())
                del words
            else:
                q_i8 = quant.to(torch.int8).cpu().contiguous()
                payload_parts.append(q_i8.view(torch.uint8).numpy().tobytes())
                del q_i8
            del scales_out
        else:
            scales_parts.append(struct.pack(f"<{len(scales)}{scale_fmt[1]}", *scales))
            if mode == "trit":
                payload_parts.append(_pack20_plain(quant))
                zero_count += sum(1 for v in quant if v == 0)
            else:
                payload_parts.append(bytes(value & 0xFF for value in quant))
        n_values += n_rows * cols
        del chunk, quant, scales

    header, n_scales = _segment_parts(
        mode=mode, n_values=n_values, scale_dtype=scale_dtype, block_size=block_size,
    )
    words_bytes = b"".join(payload_parts)
    if mode == "trit":
        storage_blocks = (
            n_values + ternary.TRITS_PER_STORAGE_BLOCK - 1
        ) // ternary.TRITS_PER_STORAGE_BLOCK
        expected_words = storage_blocks * ternary.WORDS_PER_STORAGE_BLOCK * 4
        if len(words_bytes) > expected_words:  # pragma: no cover - защита
            raise ColdExportError(
                f"{name}: слов {len(words_bytes)} байт, а блоков {storage_blocks} "
                f"даёт {expected_words}"
            )
        if len(words_bytes) < expected_words:
            words_bytes += b"\x00" * (expected_words - len(words_bytes))
    payload = header + b"".join(scales_parts) + words_bytes
    info: dict[str, Any] = {
        "params": n_values,
        "shape": shape,
        "dtype": mode,
        "kind": mode,
        "n_scale_blocks": n_scales,
        "scale_blocks_per_row": per_row_blocks,
        "scale_block": block_size,
        "scale_mode": scale_mode,
    }
    if mode == "trit":
        storage_blocks = (
            n_values + ternary.TRITS_PER_STORAGE_BLOCK - 1
        ) // ternary.TRITS_PER_STORAGE_BLOCK
        info.update({
            "n_trits": n_values,
            "n_words": (n_values + ternary.TRITS_PER_WORD - 1) // ternary.TRITS_PER_WORD,
            "n_storage_blocks": storage_blocks,
            "n_storage_words": storage_blocks * ternary.WORDS_PER_STORAGE_BLOCK,
            "zero_fraction": round(zero_count / n_values, 6) if n_values else 0.0,
        })
    return payload, info


def _pack20_plain(trits: Sequence[int]) -> bytes:
    """Упаковать триты в слова по 20 без torch (fake-тензоры в тестах)."""
    words_per_block = ternary.TRITS_PER_STORAGE_BLOCK // ternary.TRITS_PER_WORD
    n_words = (
        len(trits) + ternary.TRITS_PER_WORD - 1
    ) // ternary.TRITS_PER_WORD
    storage_blocks = (
        n_words + words_per_block - 1
    ) // words_per_block
    words = [0] * (storage_blocks * words_per_block)
    pow3 = [3 ** k for k in range(ternary.TRITS_PER_WORD)]
    # Хвост добивается нулевыми тритами (это цифра 1 в коде слова), а не
    # нулевыми словами: ровно так делает pack20_torch и qat.export, иначе
    # последнее слово сегмента разошлось бы с прежним экспортом.
    padded = list(trits) + [0] * ((-len(trits)) % ternary.TRITS_PER_WORD)
    for index, trit in enumerate(padded):
        word = index // ternary.TRITS_PER_WORD
        words[word] += (int(trit) + 1) * pow3[index % ternary.TRITS_PER_WORD]
    return struct.pack(f"<{len(words)}I", *words)


# ---------------------------------------------------------------------------
# Роли и порядок тензоров
# ---------------------------------------------------------------------------


def _role_of(key: str) -> qat_export.WeightRole:
    role = qat_export.classify_state_key(qat_export.normalize_state_key(key))
    return role


def _matrix_rank(key: str) -> int:
    try:
        return TARGET_SUFFIXES.index(key)
    except ValueError:  # pragma: no cover - ключи вне списка отсеиваются раньше
        return len(TARGET_SUFFIXES)


def canonical_order(roles: dict[str, qat_export.WeightRole]) -> list[str]:
    """Порядок обхода тензоров: детерминированный файл и предсказуемый прогресс.

    Эмбеддинг, финальная норма, слои 0..N (q/k/v/o/gate/up/down, norm1/norm2,
    bias), lm_head. В ``ternary.trit`` так первым оказывается эмбеддинг (если
    ``--embed trit``), как и в экспорте QAT.
    """
    def sort_key(item: tuple[str, qat_export.WeightRole]) -> tuple[int, int, int, str]:
        name, role = item
        if role.kind == "embed":
            return (0, -1, 0, name)
        if role.kind == "final_norm":
            return (2, -1, 0, name)
        if role.kind == "lm_head":
            return (3, -1, 0, name)
        layer = -1 if role.layer is None else int(role.layer)
        if role.kind == "matrix":
            rank = 0 + _matrix_rank(role.key)
        elif role.kind == "norm":
            rank = 10 + (0 if role.key == "norm1" else 1)
        elif role.kind == "bias":
            rank = 20 + _matrix_rank(role.key.replace("_bias", "_proj"))
        else:
            rank = 30
        return (1, layer, rank, name)

    return [name for name, _role in sorted(roles.items(), key=sort_key)]


# ---------------------------------------------------------------------------
# Экспорт
# ---------------------------------------------------------------------------


@dataclass
class ExportResult:
    """Что экспорт положил на диск и сколько это стоило по памяти."""

    out_dir: Path
    manifest_path: Path
    trit_path: Path
    non_ternary_path: Path
    manifest: dict[str, Any]
    architecture: dict[str, Any]
    elapsed_s: float
    peak_rss_bytes: int
    source_bytes: int
    tensor_count: int
    trit_params: int
    non_ternary_params: int

    @property
    def peak_to_source_ratio(self) -> float:
        return (self.peak_rss_bytes / self.source_bytes) if self.source_bytes else 0.0


def _write_fp32(handle: Any, value: Any) -> bytes:
    """fp32 little-endian байты тензора (bf16/fp16 -> fp32 без потерь)."""
    if not _is_torch_tensor(value):
        payload = qat_export._to_numpy_bytes(value, "<f4")
    else:
        payload = qat_export._to_numpy_bytes(value.detach().to("cpu").float(), "<f4")
    expected = int(value.numel()) * 4
    if len(payload) != expected:
        raise ColdExportError(
            f"fp32-выгрузка: {len(payload)} байт, ожидалось {expected}"
        )
    handle.write(payload)
    return payload


def export_model(
    source: WeightSource,
    hf_config: dict[str, Any],
    out_dir: Path | str,
    *,
    model_name: str | None = None,
    revision: str | None = None,
    block_size: int = ternary.BLOCK_SIZE,
    scale_dtype: str = "float32",
    embed_mode: str = "none",
    embed_scale: str = "block",
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    limit_layers: int | None = None,
    overrides: dict[str, Any] | None = None,
    force: bool = False,
    echo: Any = None,
) -> ExportResult:
    """Экспортировать источник весов в ``ternary.trit`` + ``non_ternary.bin``.

    Один проход по каноническому порядку тензоров: матрицы Linear уходят
    сегментами в ``ternary.trit``, эмбеддинг/нормы/bias/lm_head — fp32 в
    ``non_ternary.bin``. Тензор читается из шарда, квантуется, пишется и
    отпускается — модель целиком в памяти не живёт.
    """
    if block_size != ternary.BLOCK_SIZE:
        raise ColdExportError(
            f"--block-size={block_size}: формат VTM v0.1 хранит масштаб только "
            f"на {ternary.BLOCK_SIZE} тритов (MAX_VTM блок); поле в манифесте "
            "нужно ровно 128"
        )
    if scale_dtype not in SCALE_DTYPES:
        raise ColdExportError(f"--scale-dtype={scale_dtype}: {SCALE_DTYPES}")
    if embed_mode not in EMBED_MODES:
        raise ColdExportError(f"--embed={embed_mode}: {EMBED_MODES}")
    if embed_scale not in EMBED_SCALES:
        raise ColdExportError(f"--embed-scale={embed_scale}: {EMBED_SCALES}")
    if context_window <= 0:
        raise ColdExportError(f"--context-window={context_window}: нужно > 0")
    check_supported_config(hf_config)

    started = time.time()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    shapes = source.shapes()
    architecture = infer_export_architecture(
        shapes, hf_config,
        limit_layers=limit_layers,
        context_window=context_window,
        embed_mode=embed_mode,
        overrides=overrides,
    )
    check_block_alignment(architecture, block_size, embed_mode, embed_scale)

    # --- какие ключи берём --------------------------------------------------
    roles: dict[str, qat_export.WeightRole] = {}
    ignored: list[str] = []
    for key in source.keys():
        if key.endswith(IGNORED_KEY_SUFFIXES):
            ignored.append(key)
            continue
        role = _role_of(key)
        if role.kind == "unknown":
            ignored.append(key)
            continue
        if role.layer is not None and role.layer >= int(architecture["n_layers"]):
            continue  # обрезано --limit-layers
        roles[key] = role
    if ignored and echo is not None:
        echo(
            f"[cold_export] пропущено ключей вне графа VTM: {len(ignored)} "
            f"({', '.join(ignored[:3])}{'…' if len(ignored) > 3 else ''})"
        )
    order = canonical_order(roles)
    if not order:
        raise ColdExportError("в шардах не нашлось ни одного тензора Qwen/Llama-слоя")

    trit_path = out_path / _TRIT_FILE
    non_ternary_path = out_path / _NON_TERNARY_FILE
    manifest_path = out_path / MANIFEST_FILE
    if not force:
        existing = [p.name for p in (trit_path, non_ternary_path, manifest_path) if p.exists()]
        if existing:
            raise ColdExportError(
                f"в {out_path} уже есть экспорт ({', '.join(existing)}); "
                "добавьте --force, чтобы перезаписать"
            )

    tmp_trit = out_path / ("." + _TRIT_FILE + ".tmp")
    tmp_non = out_path / ("." + _NON_TERNARY_FILE + ".tmp")
    trit_entries: list[dict[str, Any]] = []
    non_entries: list[dict[str, Any]] = []
    trit_params = 0
    non_params = 0
    embed_entry: dict[str, Any] | None = None
    embed_int8_entry: dict[str, Any] | None = None
    seen_layers: dict[int, set[str]] = {}
    have_embed = False
    have_final_norm = False
    total = len(order)
    step_count = 0

    with tmp_trit.open("wb") as trit_handle, tmp_non.open("wb") as non_handle:
        trit_offset = 0
        non_offset = 0
        for index, key in enumerate(order, start=1):
            role = roles[key]
            value = source.tensor(key)
            shape = [int(x) for x in value.shape]
            if role.kind == "matrix":
                data, meta = encode_trit_segment(
                    value, key, block_size=block_size, scale_dtype=scale_dtype,
                )
                meta.update({
                    "name": key,
                    "role": "matrix",
                    "layer": role.layer,
                    "key": role.key,
                    "trit_offset": trit_offset,
                    "trit_length": len(data),
                })
                trit_handle.write(data)
                trit_offset += len(data)
                trit_entries.append(meta)
                trit_params += int(meta["params"])
                seen_layers.setdefault(int(role.layer), set()).add(role.key)
            elif role.kind == "embed" and embed_mode == "trit":
                data, meta = encode_embed_segment(
                    value, _EMBED_NAME, mode="trit", scale_mode=embed_scale,
                    block_size=block_size, scale_dtype=scale_dtype,
                )
                meta.update({
                    "name": _EMBED_NAME,
                    "role": "embed",
                    "layer": None,
                    "key": "embed",
                    "trit_offset": trit_offset,
                    "trit_length": len(data),
                })
                trit_handle.write(data)
                trit_offset += len(data)
                trit_entries.append(meta)
                trit_params += int(meta["params"])
                embed_entry = meta
                have_embed = True
            elif role.kind == "embed" and embed_mode == "int8":
                # INT8-сегмент (kind 4) лежит в non_ternary.bin: так его ищет
                # vtm_compile (`dtype: int8`) и так его читает C-загрузчик.
                data, meta = encode_embed_segment(
                    value, _EMBED_NAME, mode="int8", scale_mode=embed_scale,
                    block_size=block_size, scale_dtype=scale_dtype,
                )
                non_handle.write(data)
                entry = {
                    "name": _EMBED_NAME,
                    "role": "embed",
                    "layer": None,
                    "key": "embed",
                    "shape": shape,
                    "dtype": "int8",
                    "offset": non_offset,
                    "length": len(data),
                    "params": int(meta["params"]),
                    **{k: v for k, v in meta.items() if k != "shape"},
                }
                non_entries.append(entry)
                non_offset += len(data)
                non_params += int(meta["params"])
                embed_int8_entry = entry
                have_embed = True
            elif role.kind in ("embed", "final_norm", "norm", "bias", "lm_head"):
                if role.kind == "lm_head" and architecture["tie_word_embeddings"]:
                    # tied: байт не пишем, ссылку добавим ниже (как qat.export).
                    pass
                else:
                    payload = _write_fp32(non_handle, value)
                    entry = {
                        "name": key,
                        "role": role.kind,
                        "layer": role.layer,
                        "key": role.key,
                        "shape": shape,
                        "dtype": "float32",
                        "offset": non_offset,
                        "length": len(payload),
                        "params": int(value.numel()),
                    }
                    non_entries.append(entry)
                    non_offset += len(payload)
                    non_params += int(value.numel())
                    if role.kind == "embed":
                        have_embed = True
                    elif role.kind == "final_norm":
                        have_final_norm = True
            else:  # pragma: no cover - роли выше перечислены
                raise ColdExportError(f"{key}: неожиданная роль {role.kind!r}")
            step_count += 1
            if echo is not None and (index % 8 == 0 or index == total):
                echo(
                    f"[cold_export] {index}/{total} тензоров, "
                    f"тритов {trit_params:,}, нетернарных параметров "
                    f"{non_params:,}, RSS {_human_bytes(_peak_rss_bytes())}"
                )
            source.release(key)
            del value
    tmp_trit.replace(trit_path)
    tmp_non.replace(non_ternary_path)

    if architecture["tie_word_embeddings"]:
        if not have_embed and embed_entry is None:
            raise ColdExportError(
                "tie_word_embeddings=true, но model.embed_tokens.weight не найден: "
                "собирать lm_head не из чего"
            )
        if embed_mode == "trit":
            tied_shape = list(embed_entry["shape"]) if embed_entry else [
                int(architecture["vocab"]), int(architecture["hidden"])
            ]
            tied_params = int(embed_entry["params"]) if embed_entry else (
                int(architecture["vocab"]) * int(architecture["hidden"])
            )
        else:
            embed = next(
                (entry for entry in non_entries if entry["role"] == "embed"), None
            )
            if embed is None:
                raise ColdExportError(
                    "tie_word_embeddings=true, но fp32-эмбеддинга в экспорте нет"
                )
            tied_shape, tied_params = list(embed["shape"]), int(embed["params"])
        tied_dtype = {"trit": "trit", "int8": "int8"}.get(embed_mode, "float32")
        non_entries.append({
            "name": "lm_head.weight",
            "role": "lm_head",
            "layer": None,
            "key": "lm_head",
            "shape": tied_shape,
            "dtype": tied_dtype,
            "tied_to": _EMBED_NAME if embed_mode == "trit"
                       else "model.embed_tokens.weight",
            "params": tied_params,
        })

    # --- полнота графа ------------------------------------------------------
    if not have_embed:
        raise ColdExportError(
            "в экспорте нет model.embed_tokens.weight: без таблицы словаря "
            "компилятор VTM не соберёт граф"
        )
    if not have_final_norm:
        raise ColdExportError(
            "в экспорте нет model.norm.weight: без финальной нормы граф не собрать"
        )
    n_layers = int(architecture["n_layers"])
    for layer in range(n_layers):
        present = seen_layers.get(layer, set())
        missing = [key for key in TARGET_SUFFIXES if key not in present]
        if missing:
            raise ColdExportError(
                f"слой {layer}: нет матриц {', '.join(missing)} — шарды неполные "
                "или имена слоёв не Qwen/Llama"
            )

    files = qat_export.WeightFiles(
        non_ternary_path=non_ternary_path,
        non_ternary_entries=non_entries,
        non_ternary_params=non_params,
        trit_path=trit_path,
        tensor_entries=trit_entries,
        trit_params=trit_params,
        payload={},
        embed=embed_entry,
        embed_mode=embed_mode,
    )
    quant_cfg = {
        "block_size": block_size,
        "scale_dtype": scale_dtype,
        "target_suffixes": list(TARGET_SUFFIXES),
        "quantize_embeddings": embed_mode,
        "embed_scale": embed_scale,
        "rule": qat_export.QUANT_RULE,
    }
    manifest = qat_export.build_manifest(
        files=files,
        architecture=architecture,
        config={"source": "cold_export", "model": {"name_or_path": model_name}},
        quant_cfg=quant_cfg,
        block_size=block_size,
        scale_dtype=scale_dtype,
        run_dir=model_name or "<cold>",
        ckpt=model_name or "<cold>",
        step=0,
        samples_seen=0,
        model_cfg={"name_or_path": model_name, "revision": revision},
        file_stats={},
    )
    source_info = source.describe()
    peak_rss = _peak_rss_bytes()
    elapsed = time.time() - started
    manifest["training"] = {
        "run": None,
        "run_dir": None,
        "checkpoint": None,
        "step": None,
        "samples_seen": None,
        "trained": False,
        "kind": "cold_export",
        "note": "веса взяты из HF safetensors как есть; обучения не было",
        "config": {"source": "cold_export"},
    }
    manifest["non_ternary"]["source"] = "hf-safetensors"
    manifest["base_model"] = {
        "name_or_path": model_name,
        "revision": revision,
        "dir": source_info.get("dir"),
    }
    #: Длина KV-кэша в контейнере. Компилятор читает architecture.context_len;
    #: ключ продублирован наверху, чтобы сборщик/человек не искал его в arch.
    manifest["context_window"] = int(architecture["context_len"])
    manifest["source"] = {
        "kind": "cold_export",
        "tool": "tools/cold_export.py",
        "tool_version": TOOL_VERSION,
        **source_info,
    }
    manifest["hf_config"] = hf_config
    # qat.export считает секцию embeddings по files.embed (trit) или fp32;
    # int8 и посрочные масштабы он не знает, поэтому описание уточняется тут.
    # Компилятор смотрит на architecture.embed и на dtype записи, но человек
    # и отчёт читают именно этот блок.
    embed_fp32 = next(
        (entry for entry in non_entries
         if entry.get("role") == "embed" and entry.get("dtype") == "float32"),
        None,
    )
    if embed_entry is not None:
        embed_params = int(embed_entry["params"])
        embed_bytes = int(embed_entry["trit_length"])
    elif embed_int8_entry is not None:
        embed_params = int(embed_int8_entry["params"])
        embed_bytes = int(embed_int8_entry["length"])
    elif embed_fp32 is not None:
        embed_params = int(embed_fp32["params"])
        embed_bytes = int(embed_fp32["length"])
    else:  # pragma: no cover - have_embed уже проверен выше
        embed_params = int(architecture["vocab"]) * int(architecture["hidden"])
        embed_bytes = embed_params * 4
    manifest["embeddings"] = {
        "mode": embed_mode,
        "tensor": _EMBED_NAME,
        "storage": "ternary.trit" if embed_mode == "trit"
                   else _NON_TERNARY_FILE,
        "dtype": {"trit": "trit", "int8": "int8"}.get(embed_mode, "float32"),
        "params": embed_params,
        "bytes": embed_bytes,
        "block_size": block_size,
        "scale_mode": embed_scale,
        "scale_block": vtm_ref.INT8_SCALE_BLOCK if embed_mode == "int8" else block_size,
        "rule": qat_export.QUANT_RULE,
        "note": (
            "scale_mode=row: один масштаб на строку таблицы, разложенный по её "
            "блокам 128; раскладка сегмента та же, что при block"
            if embed_scale == "row" else
            "scale_mode=block: absmax на блок 128 вдоль строки таблицы"
        ),
    }
    manifest["cold_export"] = {
        "tool_version": TOOL_VERSION,
        "block_size": block_size,
        "scale_dtype": scale_dtype,
        "embed": embed_mode,
        "embed_scale": embed_scale,
        "context_window": int(architecture["context_len"]),
        "limit_layers": None if limit_layers is None else int(limit_layers),
        "n_layers_exported": n_layers,
        "elapsed_s": round(elapsed, 1),
        "memory": {
            "peak_rss_bytes": peak_rss,
            "peak_rss": _human_bytes(peak_rss),
            "shard_bytes": source_info.get("largest_shard_bytes", 0),
            "peak_to_shard": (
                round(peak_rss / source_info["largest_shard_bytes"], 3)
                if source_info.get("largest_shard_bytes") else None
            ),
            "limit": "≤ 2× размера шарда",
        },
        "ignored_keys": sorted(set(ignored)),
    }
    qat_export.write_json_atomic(manifest_path, manifest)
    return ExportResult(
        out_dir=out_path,
        manifest_path=manifest_path,
        trit_path=trit_path,
        non_ternary_path=non_ternary_path,
        manifest=manifest,
        architecture=architecture,
        elapsed_s=elapsed,
        peak_rss_bytes=peak_rss,
        source_bytes=int(source_info.get("largest_shard_bytes", 0)),
        tensor_count=len(trit_entries),
        trit_params=trit_params,
        non_ternary_params=non_params,
    )


# ---------------------------------------------------------------------------
# Сверка с HF
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = (
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Столица России —",
)


def _dequantized(weight: Any, block_size: int) -> Any:
    """Третизованный вес (trit × scale) по тому же правилу, что в файле."""
    quant = ternary.quantize_torch(weight.detach().float(), block_size)
    trits = quant["trits"]
    scales = quant["scales"]
    rows, cols = int(trits.shape[0]), int(trits.shape[1])
    expanded = scales.reshape(rows, -1).repeat_interleave(block_size, dim=1)[:, :cols]
    return (trits.float() * expanded).to(weight.dtype)


def _patch_ternary_modules(model: Any, block_size: int) -> list[str]:
    """Заменить веса Linear на их третизованную версию (in-place)."""
    import torch  # noqa: PLC0415

    patched: list[str] = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if name.split(".")[-1] not in TARGET_SUFFIXES:
                continue
            weight = module.weight
            deq = _dequantized(weight, block_size)
            weight.copy_(deq)
            patched.append(name)
            del deq
    return patched


def _patch_embedding(model: Any, block_size: int) -> bool:
    import torch  # noqa: PLC0415

    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Embedding):
                continue
            weight = module.weight
            deq = _dequantized(weight, block_size)
            weight.copy_(deq)
            del deq
            return True
    return False


def _logits_for(model: Any, tokenizer: Any, prompt: str) -> list[float]:
    import torch  # noqa: PLC0415

    encoded = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in encoded.items()}
    with torch.no_grad():
        output = model(**inputs)
    return [float(x) for x in output.logits[0, -1, :].float().cpu().tolist()]


def _topk(values: Sequence[float], k: int = 5) -> list[int]:
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:k]


def _compare(actual: Sequence[float], reference: Sequence[float]) -> dict[str, Any]:
    if len(actual) != len(reference):
        raise ColdExportError(
            f"логиты разной длины: {len(actual)} против {len(reference)}"
        )
    deltas = [abs(a - b) for a, b in zip(actual, reference)]
    argmax_a = max(range(len(actual)), key=lambda i: actual[i])
    argmax_b = max(range(len(reference)), key=lambda i: reference[i])
    top_a, top_b = _topk(actual), _topk(reference)
    return {
        "max_abs_delta": max(deltas),
        "mean_abs_delta": sum(deltas) / len(deltas),
        "argmax": argmax_a,
        "reference_argmax": argmax_b,
        "argmax_match": argmax_a == argmax_b,
        "top5_overlap": len(set(top_a) & set(top_b)),
        "reference_absmax": max(abs(x) for x in reference),
    }


def _spot_check_artifacts(result: ExportResult, model: Any, block_size: int) -> dict[str, Any]:
    """Прочитать сегменты из ternary.trit и сверить с квантованием в памяти.

    Это проверка «файл == то, что посчитано»: декодированные триты и масштабы
    обязаны совпасть с квантованием веса модели. Берём эмбеддинг (если он
    тритовый) и до двух матриц — чтобы не читать файл целиком.
    """
    import torch  # noqa: PLC0415

    manifest = result.manifest
    trit_bytes = result.trit_path.read_bytes()
    weights = dict(_named_parameter_map(model))
    checked: list[dict[str, Any]] = []
    for entry in manifest.get("tensors") or []:
        name = str(entry.get("name") or "")
        if name == _EMBED_NAME:
            source_name = "model.embed_tokens.weight"
        else:
            source_name = name
        source = weights.get(source_name)
        if source is None:
            continue
        if len(checked) >= 2 and name != _EMBED_NAME:
            break
        offset = int(entry.get("trit_offset", -1))
        length = int(entry.get("trit_length", -1))
        segment = trit_bytes[offset:offset + length]
        trits, scales = ternary.decode_segment(segment, block_size)
        expected = ternary.quantize_torch(source.detach().float(), block_size)
        expected_trits = expected["trits"].reshape(-1).tolist()
        expected_scales = expected["scales"].reshape(-1).tolist()
        trits_ok = len(trits) == len(expected_trits) and trits == expected_trits
        scale_delta = max(
            (abs(a - b) for a, b in zip(scales, expected_scales)), default=0.0
        )
        checked.append({
            "name": name,
            "trits": len(trits),
            "trits_match": bool(trits_ok),
            "scales_match": bool(scale_delta == 0.0),
            "scales_max_abs_delta": float(scale_delta),
        })
        del expected, expected_trits, expected_scales
    return {
        "checked": checked,
        "ok": bool(checked) and all(
            item["trits_match"] and item["scales_match"] for item in checked
        ),
    }


def _named_parameter_map(model: Any) -> Iterator[tuple[str, Any]]:
    for name, parameter in model.named_parameters():
        yield name, parameter


def verify_export(
    model_dir: Path | str,
    result: ExportResult,
    *,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    block_size: int = ternary.BLOCK_SIZE,
    tolerance: float = 1e-2,
    device: str = "auto",
    torch_dtype: str = "float32",
    echo: Any = print,
) -> dict[str, Any]:
    """Сверить третизацию с HF: два forward'а и проверка артефактов.

    1. HF полной точности — эталон;
    2. та же модель с третизованными Linear (и эмбеддингом, если он trit) —
       ровно те веса, что записаны в ``ternary.trit``;
    3. ``max|Δ|`` логитов последней позиции, argmax, top-5;
    4. спот-чек сегментов из файла против квантования в памяти.
    """
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:
        raise ColdExportError(f"--verify требует torch и transformers ({exc})") from exc

    if result.manifest.get("cold_export", {}).get("limit_layers"):
        raise ColdExportError(
            "--verify несовместим с --limit-layers: у частичной модели нет "
            "полного набора слоёв для forward'а"
        )
    dtype = getattr(torch, torch_dtype, torch.float32)
    if device == "auto":
        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    echo(f"[cold_export] verify: гружу HF {model_dir} ({torch_dtype}, {device})…")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=dtype, local_files_only=True,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

    full_logits = {prompt: _logits_for(model, tokenizer, prompt) for prompt in prompts}
    patched = _patch_ternary_modules(model, block_size)
    embed_patched = False
    if str(result.manifest.get("architecture", {}).get("embed")) == "trit":
        embed_patched = _patch_embedding(model, block_size)
    echo(
        f"[cold_export] verify: третизовано Linear {len(patched)}, "
        f"эмбеддинг {'да' if embed_patched else 'нет'}"
    )
    tern_logits = {prompt: _logits_for(model, tokenizer, prompt) for prompt in prompts}

    per_prompt: list[dict[str, Any]] = []
    for prompt in prompts:
        stats = _compare(tern_logits[prompt], full_logits[prompt])
        stats["prompt"] = prompt
        stats["tokens"] = len(tokenizer(prompt)["input_ids"])
        per_prompt.append(stats)
    artifacts = _spot_check_artifacts(result, model, block_size)

    max_delta = max(item["max_abs_delta"] for item in per_prompt)
    argmax_all = all(item["argmax_match"] for item in per_prompt)
    report = {
        "model": str(model_dir),
        "device": device,
        "torch_dtype": torch_dtype,
        "prompts": per_prompt,
        "max_abs_delta": max_delta,
        "argmax_all_match": argmax_all,
        "tolerance": tolerance,
        "within_tolerance": max_delta <= tolerance,
        "artifacts": artifacts,
        "linear_patched": len(patched),
        "embedding_patched": embed_patched,
    }
    echo("")
    echo("[cold_export] сверка логитов с HF (последняя позиция промпта):")
    for item in per_prompt:
        echo(
            f"  «{item['prompt'][:40]}»: max|Δ|={item['max_abs_delta']:.4e} "
            f"mean|Δ|={item['mean_abs_delta']:.2e} "
            f"argmax HF={item['reference_argmax']} трет={item['argmax']} "
            f"{'совпал' if item['argmax_match'] else 'НЕ совпал'}, "
            f"top5={item['top5_overlap']}/5 (|logit|∞={item['reference_absmax']:.2f})"
        )
    echo(
        f"  итог: max|Δ|={max_delta:.4e} (порог {tolerance:.0e}) "
        f"{'в пороге' if max_delta <= tolerance else 'выше порога'}; "
        f"argmax совпал на {sum(1 for i in per_prompt if i['argmax_match'])}"
        f"/{len(per_prompt)}"
    )
    echo(
        "  артефакты ternary.trit: "
        + ", ".join(
            f"{item['name']} триты {'=' if item['trits_match'] else '≠'}, "
            f"масштабы {'=' if item['scales_match'] else '≠'}"
            for item in artifacts["checked"]
        )
        if artifacts["checked"] else "  артефакты: нечего сверять"
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/cold_export.py",
        description=(
            "Холодный экспорт HF safetensors -> ternary.trit + non_ternary.bin "
            "+ manifest.json формата tools/vtm_compile.py (без обучения)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  PYTHONPATH=.pylibs python3.13 tools/cold_export.py \\\n"
            "      --model Qwen/Qwen2.5-0.5B-Instruct --allow-download \\\n"
            "      --out export/qwen05b --verify\n"
            "  PYTHONPATH=.pylibs python3.13 tools/cold_export.py \\\n"
            "      --model .hfcache/hub/models--Qwen--Qwen2.5-14B --limit-layers 2 \\\n"
            "      --out /tmp/q14-partial\n\n"
            "дальше без torch:\n"
            "  python3 tools/vtm_compile.py --manifest export/qwen05b/manifest.json "
            "--out export/qwen05b/model.vtm\n"
            "  src/vtm/vtm run export/qwen05b/model.vtm --tokens 1,2,3 --max-new 1\n"
        ),
    )
    parser.add_argument("--model", required=True,
                        help="каталог HF-модели или id (id — только с --allow-download)")
    parser.add_argument("--out", required=True, help="каталог для trit/bin/manifest")
    parser.add_argument("--allow-download", action="store_true",
                        help="разрешить скачать snapshot по id в HF-кэш")
    parser.add_argument("--cache-dir", default=None,
                        help="каталог HF-кэша (по умолчанию HF_HOME/hub)")
    parser.add_argument("--limit-layers", type=int, default=None,
                        help="экспортировать только первые N слоёв (быстрая проверка)")
    parser.add_argument("--context-window", "--context-len", type=int,
                        default=DEFAULT_CONTEXT_WINDOW, dest="context_window",
                        help=f"длина KV-кэша в контейнере (по умолчанию "
                             f"{DEFAULT_CONTEXT_WINDOW}: 14B при 2048 не влезает "
                             f"в кэп 512 МиБ)")
    parser.add_argument("--embed", choices=EMBED_MODES, default="none",
                        help="таблица словаря: none — fp32 в non_ternary.bin, "
                             "trit — сегментом TRT1 в ternary.trit, "
                             "int8 — INT8-сегментом (kind 4) в non_ternary.bin")
    parser.add_argument("--embed-scale", choices=EMBED_SCALES, default="block",
                        help="масштаб таблицы: block — absmax на блок 128 (как "
                             "у Linear), row — один масштаб на строку (только "
                             "по флагу, раскладка сегмента прежняя)")
    parser.add_argument("--block-size", type=int, default=ternary.BLOCK_SIZE,
                        help="блок absmax (VTM v0.1: только 128)")
    parser.add_argument("--scale-dtype", choices=SCALE_DTYPES, default="float32",
                        help="тип масштабов (по умолчанию float32, как в QAT)")
    parser.add_argument("--n-heads", type=int, default=None,
                        help="override числа голов (если config.json врёт/нет)")
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--rope-theta", type=float, default=None)
    parser.add_argument("--rms-norm-eps", type=float, default=None)
    parser.add_argument("--tie-embeddings", choices=("auto", "yes", "no"),
                        default="auto")
    parser.add_argument("--force", action="store_true", help="перезаписать экспорт")
    parser.add_argument("--verify", action="store_true",
                        help="после экспорта сверить логиты с HF (нужны torch+transformers)")
    parser.add_argument("--prompt", action="append", default=None,
                        help="промпт для --verify (можно несколько; по умолчанию 3)")
    parser.add_argument("--tolerance", type=float, default=1e-2,
                        help="порог max|Δ| логитов для --verify (1e-2)")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                        default="auto")
    parser.add_argument("--torch-dtype", choices=("float32", "bfloat16"),
                        default="float32", help="dtype HF-модели в --verify")
    parser.add_argument("--revision", default=None,
                        help="ревизия модели для манифеста (не для загрузки)")
    parser.add_argument("--json-out", default=None,
                        help="записать отчёт --verify в JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    missing = missing_modules()
    if missing:
        print(f"[cold_export] не хватает: {', '.join(missing)}")
        print()
        print(MISSING_HINT)
        return 2

    echo = lambda message: print(message, file=sys.stderr, flush=True)  # noqa: E731
    try:
        source = open_source(
            args.model, allow_download=args.allow_download, cache_dir=args.cache_dir,
        )
    except ColdExportError as exc:
        print(f"[cold_export] {exc}")
        return 2
    try:
        model_dir = getattr(source, "model_dir", None) or Path(args.model)
        try:
            hf_config = load_hf_config(model_dir)
        except ColdExportError:
            if args.allow_download:
                raise
            # источник без config.json рядом (например, тестовый словарь)
            hf_config = {}
        check_supported_config(hf_config)
        overrides: dict[str, Any] = {}
        for field_name in ("n_heads", "n_kv_heads", "head_dim", "rope_theta",
                           "rms_norm_eps"):
            value = getattr(args, field_name)
            if value is not None:
                overrides[field_name] = value
        if args.tie_embeddings != "auto":
            overrides["tie_word_embeddings"] = args.tie_embeddings == "yes"
        result = export_model(
            source, hf_config, args.out,
            model_name=str(args.model),
            revision=args.revision,
            block_size=args.block_size,
            scale_dtype=args.scale_dtype,
            embed_mode=args.embed,
            embed_scale=args.embed_scale,
            context_window=args.context_window,
            limit_layers=args.limit_layers,
            overrides=overrides,
            force=args.force,
            echo=echo,
        )
    except ColdExportError as exc:
        print(f"[cold_export] {exc}")
        return 2
    finally:
        source.close()

    manifest = result.manifest
    echo("")
    echo(f"[cold_export] готово за {result.elapsed_s:.0f} с -> {result.out_dir}")
    echo(
        f"[cold_export] архитектура: hidden={result.architecture['hidden']} "
        f"heads={result.architecture['n_heads']}/{result.architecture['n_kv_heads']} "
        f"head_dim={result.architecture['head_dim']} "
        f"intermediate={result.architecture['intermediate']} "
        f"vocab={result.architecture['vocab']} "
        f"layers={result.architecture['n_layers']} "
        f"rope_theta={result.architecture['rope_theta']:g} "
        f"eps={result.architecture['rms_norm_eps']:g} "
        f"tie={result.architecture['tie_word_embeddings']} "
        f"embed={result.architecture['embed']}"
        f"/{manifest.get('cold_export', {}).get('embed_scale', 'block')} "
        f"ctx={manifest['context_window']}"
    )
    echo(
        f"[cold_export] {result.trit_path.name}: "
        f"{_human_bytes(result.trit_path.stat().st_size)} "
        f"({result.tensor_count} сегментов, {result.trit_params:,} тритов)"
    )
    non_kind = "fp32"
    if manifest.get("embeddings", {}).get("dtype") == "int8":
        non_kind = "fp32, таблица словаря int8"
    echo(
        f"[cold_export] {result.non_ternary_path.name}: "
        f"{_human_bytes(result.non_ternary_path.stat().st_size)} "
        f"({result.non_ternary_params:,} параметров: {non_kind})"
    )
    memory = manifest["cold_export"]["memory"]
    echo(
        f"[cold_export] память: пик RSS {memory['peak_rss']}, "
        f"шард {_human_bytes(memory['shard_bytes'])}, "
        f"отношение {memory['peak_to_shard']} (критерий ≤ 2.0)"
    )
    echo(
        "[cold_export] sha256: "
        f"{result.trit_path.name} {manifest['files'][result.trit_path.name]['sha256'][:16]}…, "
        f"{result.non_ternary_path.name} "
        f"{manifest['files'][result.non_ternary_path.name]['sha256'][:16]}…"
    )

    if args.verify:
        try:
            report = verify_export(
                getattr(source, "model_dir", None) or args.model,
                result,
                prompts=tuple(args.prompt or DEFAULT_PROMPTS),
                block_size=args.block_size,
                tolerance=args.tolerance,
                device=args.device,
                torch_dtype=args.torch_dtype,
                echo=echo,
            )
        except ColdExportError as exc:
            print(f"[cold_export] {exc}")
            return 2
        if args.json_out:
            Path(args.json_out).expanduser().write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if not report["artifacts"]["ok"]:
            print("[cold_export] артефакты не совпали с квантованием весов — файл испорчен")
            return 1
        if not report["argmax_all_match"]:
            print("[cold_export] argmax разошёлся с HF на части промптов")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
