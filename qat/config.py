"""Конфиг QAT: dataclasses, чтение YAML и оценка памяти.

YAML импортируется лениво, чтобы `python -m qat.train --help`, `--dry-run`
и тесты работали на машине без GPU-стека. Если pyyaml нет, конфиги читаются
встроенным парсером простого подмножества YAML (см. `_parse_simple_yaml`).
Неизвестные ключи в YAML считаются ошибкой: опечатка в имени параметра не
должна молча теряться.

Модуль намеренно не импортирует torch: `estimate_memory` — чистая
арифметика над конфигом, её можно вызывать до установки GPU-стека.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
DEFAULT_EXCLUDE_SUFFIXES = ("lm_head", "embed_tokens", "router", "gate", "norm")

#: Архитектуры известных моделей — для оценки памяти без скачивания весов.
#: Числа параметров посчитаны по конфигу модели (эмбеддинг один: lm_head tied),
#: hidden/intermediate/layers/vocab нужны для оценки активаций.
MODEL_PRESETS: dict[str, dict[str, int]] = {
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "num_params": 1_543_712_768,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_hidden_layers": 28,
        "vocab_size": 151936,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "num_params": 3_085_936_640,
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_hidden_layers": 36,
        "vocab_size": 151936,
    },
}

ARCH_FIELDS = (
    "num_params",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "vocab_size",
)

#: master_dtype -> имя dtype для torch/transformers.
MASTER_TO_TORCH = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}
TORCH_TO_MASTER = {value: key for key, value in MASTER_TO_TORCH.items()}

#: Сколько байт на параметр занимают компоненты обучения.
DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}
AMP_TO_MASTER = {"bfloat16": "bf16", "float16": "fp16", "none": None}
OPTIMIZER_BYTES_PER_PARAM = {
    # bitsandbytes Adam8bit: два состояния по 1 байту.
    "adam8bit": 2,
    # AdamW: два fp32-состояния (m и v). Это консервативная оценка: если torch
    # создаст состояния в dtype параметра (bf16), фактически будет меньше, но
    # полагаться на это нельзя.
    "adamw": 8,
}
GRAD_DTYPE_BYTES = {"fp32": 4, "bf16": 2}

#: Почему 8, а не 1: без checkpointing на слой живут вход, QKV, выход attention,
#: gate/up, активации GELU и т.п. Коэффициент — эмпирическая верхняя оценка.
ACT_TENSORS_PER_LAYER_NO_CKPT = 8
#: Вход слоя (B×L×H) и рабочий набор пересчёта одного блока; добавляем к активациям
#: логиты (B×L×vocab) и запас на аллокатор/CUDA-контекст.
ACT_CKPT_TENSORS_PER_LAYER = 1
ACT_WORKSPACE_BYTES = 768 * 1024**2  # 0.75 ГиБ на контекст, workspace и фрагментацию
GPU_BUDGET_BYTES = 24 * 1024**3  # RTX 4090: 24 ГиБ, а не 24 ГБ



class ConfigError(RuntimeError):
    """Конфиг не читается или содержит недопустимые значения."""


@dataclass
class ModelConfig:
    """Базовая модель. Ревизию на GPU-машине надо зафиксировать коммитом."""

    name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str = "main"
    trust_remote_code: bool = False
    #: Мастер-веса: fp32 | bf16 | fp16. Пусто — взять из torch_dtype.
    #: bf16 экономит память, но обновления меньше половины шага bf16
    #: округляются в ноль: см. docs/qat.md, раздел «Память и OOM».
    #: Для QAT безопаснее fp32.
    master_dtype: str = ""
    #: Синоним master_dtype в терминах torch; синхронизируется при валидации.
    torch_dtype: str = "float32"
    attn_implementation: str = "sdpa"
    #: Размер и архитектура — только для estimate_memory. 0 — взять из
    #: MODEL_PRESETS по name_or_path; для локальной модели укажите явно.
    num_params: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    num_hidden_layers: int = 0
    vocab_size: int = 0


@dataclass
class QuantConfig:
    """Что и как квантуем."""

    block_size: int = 128
    scale_dtype: str = "float16"
    target_suffixes: tuple[str, ...] = field(default=DEFAULT_TARGET_SUFFIXES)
    exclude_suffixes: tuple[str, ...] = field(default=DEFAULT_EXCLUDE_SUFFIXES)
    # Эмбеддинги отдельным решением: "none" (fp16/fp32), "int8" или "trit".
    # Задача 0027: у 1.5B таблица словаря — половина модели, но при tied-весах это
    # ещё и выходной слой, поэтому триты могут ударить по логитам.
    quantize_embeddings: str = "none"


@dataclass
class DataConfig:
    """Пары «вопрос → ответ» и опциональный KL по логитам учителя."""

    path: str = "distill/out/pairs.jsonl"
    max_seq_len: int = 2048
    max_samples: int = 0
    min_answer_tokens: int = 1
    use_teacher_logits: bool = False
    kl_temperature: float = 1.0
    kl_weight: float = 0.5
    kl_chunk_size: int = 64


@dataclass
class TrainConfig:
    """Режим обучения, память и чекпоинты."""

    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    max_steps: int = 2000
    lr: float = 3.0e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    seed: int = 42
    amp_dtype: str = "bfloat16"
    #: Градиенты: fp32 (точнее, больше памяти) | bf16 (вдвое меньше).
    #: При master_dtype=bf16 градиенты и так bf16, апкаст до fp32 только добавит памяти.
    grad_dtype: str = "fp32"
    gradient_checkpointing: bool = True
    optimizer: str = "adam8bit"
    log_every: int = 10
    save_every: int = 100
    keep_last_checkpoints: int = 2
    save_optimizer: bool = True
    device: str = "auto"
    loss_chunk_size: int = 512


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


_SECTIONS = {
    "model": ModelConfig,
    "quant": QuantConfig,
    "data": DataConfig,
    "train": TrainConfig,
}


def _make_section(cls: type, name: str, raw: Any) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"секция {name!r} должна быть словарём, получено {type(raw).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(
            f"неизвестные ключи в {name}: {', '.join(unknown)}; "
            f"допустимые: {', '.join(sorted(known))}"
        )
    kwargs = dict(raw)
    # YAML-списки превращаем в tuple: в конфиге tuple-поля — betas и суффиксы.
    for key, value in list(kwargs.items()):
        if isinstance(value, list):
            kwargs[key] = tuple(value)
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"не удалось собрать секцию {name}: {exc}") from exc


def _strip_yaml_comment(line: str) -> str:
    """Убрать комментарий, не трогая # внутри кавычек."""
    in_single = in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            if index == 0 or line[index - 1].isspace():
                return line[:index]
    return line


def _split_flow_list(text: str) -> list[str]:
    """Разбить ``[a, b, "c, d"]`` по запятым верхнего уровня."""
    items: list[str] = []
    start = 0
    in_single = in_double = False
    for index, char in enumerate(text):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "," and not in_single and not in_double:
            items.append(text[start:index])
            start = index + 1
    items.append(text[start:])
    return [item for item in (part.strip() for part in items) if item]


def _parse_yaml_scalar(text: str) -> Any:
    """Скаляр или flow-список подмножества YAML, которое используют конфиги."""
    text = text.strip()
    if not text:
        raise ConfigError("пустое значение в конфиге")
    if text.startswith("[") and text.endswith("]"):
        return tuple(_parse_yaml_scalar(item) for item in _split_flow_list(text[1:-1]))
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "~"):
        return None
    if text[0] in ("'", '"') and text[-1] == text[0] and len(text) >= 2:
        return text[1:-1]
    if re.fullmatch(r"[+-]?\d[\d_]*", text):
        return int(text.replace("_", ""))
    if re.fullmatch(r"[+-]?(\d[\d_]*)?\.\d[\d_]*([eE][+-]?\d+)?", text) or \
            re.fullmatch(r"[+-]?\d[\d_]*[eE][+-]?\d+", text):
        return float(text.replace("_", ""))
    return text


def _parse_simple_yaml(text: str, path: str | Path = "<config>") -> dict[str, Any]:
    """Минимальный парсер YAML для конфигов QAT — когда pyyaml не установлен.

    Поддерживает ровно то, что есть в ``qat/configs/*.yaml``: секции верхнего
    уровня, ключи с отступом 2, скаляры, bool, inline-списки и комментарии.
    Всё остальное (якоря, многострочные строки, вложенность глубже одного
    уровня) — ошибка с номером строки, а не тихо неверный конфиг.
    """
    raw: dict[str, Any] = {}
    section: dict[str, Any] | None = None
    for lineno, original in enumerate(text.splitlines(), 1):
        line = _strip_yaml_comment(original).rstrip()
        if not line.strip():
            continue
        if "\t" in line:
            raise ConfigError(f"{path}:{lineno}: табы в YAML не поддерживаются")
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if ":" not in content:
            raise ConfigError(f"{path}:{lineno}: ожидается 'ключ: значение'")
        key, _, value = content.partition(":")
        key, value = key.strip(), value.strip()
        if not key:
            raise ConfigError(f"{path}:{lineno}: пустой ключ")
        if indent == 0:
            if value:
                raw[key] = _parse_yaml_scalar(value)
                section = None
            else:
                if key in raw:
                    raise ConfigError(f"{path}:{lineno}: повтор ключа {key!r}")
                raw[key] = {}
                section = raw[key]
        elif indent == 2 and section is not None:
            if not value:
                raise ConfigError(
                    f"{path}:{lineno}: вложенность глубже одного уровня не поддерживается"
                )
            if key in section:
                raise ConfigError(f"{path}:{lineno}: повтор ключа {key!r}")
            section[key] = _parse_yaml_scalar(value)
        else:
            raise ConfigError(
                f"{path}:{lineno}: неожиданный отступ {indent}; допустимо 0 или 2"
            )
    return raw


def load_config(path: str | Path) -> Config:
    """Прочитать YAML и проверить значения."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"конфиг не найден: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # noqa: PLC0415 - ленивый импорт: без GPU-стека не нужен
    except ImportError:
        # pyyaml нет — конфиги QAT простые, читаем встроенным парсером.
        raw = _parse_simple_yaml(text, path)
    else:
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: некорректный YAML: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("верхний уровень конфига должен быть словарём")
    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ConfigError(f"неизвестные секции: {', '.join(unknown)}")
    cfg = Config(**{name: _make_section(cls, name, raw.get(name, {}))
                    for name, cls in _SECTIONS.items()})
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    """Проверить инварианты, на которых держится обучение."""
    q, d, t = cfg.quant, cfg.data, cfg.train
    if q.block_size <= 0:
        raise ConfigError("quant.block_size должен быть > 0")
    if q.scale_dtype not in ("float16", "float32"):
        raise ConfigError("quant.scale_dtype: допустимо float16 или float32")
    if not q.target_suffixes:
        raise ConfigError("quant.target_suffixes не может быть пустым")
    overlap = set(q.target_suffixes) & set(q.exclude_suffixes)
    if overlap:
        raise ConfigError(f"суффиксы одновременно включены и исключены: {sorted(overlap)}")
    if d.max_seq_len <= 0:
        raise ConfigError("data.max_seq_len должен быть > 0")
    if d.min_answer_tokens < 1:
        raise ConfigError("data.min_answer_tokens должен быть >= 1")
    if d.use_teacher_logits and not (0.0 < d.kl_weight <= 1.0):
        raise ConfigError("data.kl_weight должен быть в (0, 1] при use_teacher_logits")
    if d.kl_temperature <= 0:
        raise ConfigError("data.kl_temperature должен быть > 0")
    if t.micro_batch_size <= 0 or t.grad_accum_steps <= 0:
        raise ConfigError("train.micro_batch_size и train.grad_accum_steps должны быть > 0")
    if t.max_steps <= 0:
        raise ConfigError("train.max_steps должен быть > 0")
    if t.lr <= 0:
        raise ConfigError("train.lr должен быть > 0")
    if not 0.0 <= t.min_lr_ratio <= 1.0:
        raise ConfigError("train.min_lr_ratio должен быть в [0, 1]")
    if t.warmup_steps < 0:
        raise ConfigError("train.warmup_steps не может быть отрицательным")
    if t.amp_dtype not in ("bfloat16", "float16", "none"):
        raise ConfigError("train.amp_dtype: допустимо bfloat16, float16 или none")
    if t.grad_dtype not in GRAD_DTYPE_BYTES:
        raise ConfigError("train.grad_dtype: допустимо fp32 или bf16")
    if t.optimizer not in ("adam8bit", "adamw"):
        raise ConfigError("train.optimizer: допустимо adam8bit или adamw")
    if t.log_every <= 0 or t.save_every <= 0:
        raise ConfigError("train.log_every и train.save_every должны быть > 0")
    if t.keep_last_checkpoints < 1:
        raise ConfigError("train.keep_last_checkpoints должен быть >= 1")
    if t.loss_chunk_size <= 0:
        raise ConfigError("train.loss_chunk_size должен быть > 0")
    if cfg.model.torch_dtype not in ("float32", "bfloat16", "float16"):
        raise ConfigError("model.torch_dtype: допустимо float32, bfloat16 или float16")
    # master_dtype — основной ключ (задача 0047), torch_dtype — совместимый.
    # Если заданы оба и расходятся, побеждает master_dtype: иначе нельзя отличить
    # явный torch_dtype от значения по умолчанию.
    master = resolve_master_dtype(cfg.model)
    cfg.model.master_dtype = master
    cfg.model.torch_dtype = MASTER_TO_TORCH[master]
    for name in ARCH_FIELDS:
        if int(getattr(cfg.model, name)) < 0:
            raise ConfigError(f"model.{name} не может быть отрицательным")
    if len(t.betas) != 2 or not all(0.0 <= b < 1.0 for b in t.betas):
        raise ConfigError("train.betas должен быть парой чисел в [0, 1)")


def resolve_master_dtype(model_cfg: ModelConfig) -> str:
    """master_dtype (fp32|bf16|fp16) с фолбэком на torch_dtype."""
    raw = (model_cfg.master_dtype or "").strip().lower()
    if raw:
        if raw not in MASTER_TO_TORCH:
            raise ConfigError(
                f"model.master_dtype: допустимо {', '.join(MASTER_TO_TORCH)}, получено {raw!r}"
            )
        return raw
    torch_name = (model_cfg.torch_dtype or "").strip().lower()
    if torch_name not in TORCH_TO_MASTER:
        raise ConfigError(
            f"model.torch_dtype: допустимо {', '.join(TORCH_TO_MASTER)}, получено {torch_name!r}"
        )
    return TORCH_TO_MASTER[torch_name]


def resolve_model_shape(model_cfg: ModelConfig) -> dict[str, int]:
    """Параметры и архитектура для оценки: из конфига или MODEL_PRESETS."""
    preset = MODEL_PRESETS.get(model_cfg.name_or_path.strip(), {})
    shape: dict[str, int] = {}
    missing: list[str] = []
    for name in ARCH_FIELDS:
        value = int(getattr(model_cfg, name) or 0)
        if value <= 0:
            value = int(preset.get(name, 0))
        if value <= 0:
            missing.append(name)
        shape[name] = value
    if missing:
        raise ConfigError(
            "для estimate_memory не хватает model." + ", model.".join(missing)
            + f" (модель {model_cfg.name_or_path!r} нет в MODEL_PRESETS): "
            "укажите числом в конфиге"
        )
    return shape


def _compute_dtype(cfg: Config, master: str) -> str:
    """dtype, в котором модель реально считает forward."""
    if cfg.train.amp_dtype == "none":
        return master
    return AMP_TO_MASTER[cfg.train.amp_dtype]  # type: ignore[return-value]


def _activation_bytes(cfg: Config, shape: dict[str, int], compute: str) -> int:
    """Оценка активаций и временных буферов, байт.

    ``B×L×H`` — один тензор активаций. С gradient checkpointing на слой
    хранится только вход блока, а внутри одного блока считается рабочий набор
    ``3H + 2I`` (QKV, gate/up). Без checkpointing на слой живут все
    промежуточные тензоры — коэффициент ``ACT_TENSORS_PER_LAYER_NO_CKPT``.
    Логиты считаем для всей длины: ``logits_to_keep`` уменьшает их, но
    dry-run не знает длину ответа, поэтому оценка сверху. Плюс запас на
    CUDA-контекст и фрагментацию аллокатора.
    """
    token_bytes = cfg.train.micro_batch_size * cfg.data.max_seq_len * DTYPE_BYTES[compute]
    hidden = shape["hidden_size"]
    layers = shape["num_hidden_layers"]
    tensors_per_layer = (
        ACT_CKPT_TENSORS_PER_LAYER
        if cfg.train.gradient_checkpointing
        else ACT_TENSORS_PER_LAYER_NO_CKPT
    )
    stored = token_bytes * hidden * layers * tensors_per_layer
    recompute = token_bytes * (3 * hidden + 2 * shape["intermediate_size"])
    logits = token_bytes * shape["vocab_size"]
    return stored + recompute + logits + ACT_WORKSPACE_BYTES


def estimate_memory(cfg: Config) -> dict[str, Any]:
    """Оценка памяти обучения по компонентам — без torch и без загрузки весов.

    Компоненты: мастер-веса, рабочая копия (если мастер не в compute dtype),
    состояния оптимизатора, градиенты, активации/workspace. Возвращает байты
    и ГиБ по каждому компоненту, сумму, запас до 24 ГиБ и признаки режима.
    """
    master = resolve_master_dtype(cfg.model)
    if cfg.train.optimizer not in OPTIMIZER_BYTES_PER_PARAM:
        raise ConfigError(f"train.optimizer: допустимо {', '.join(OPTIMIZER_BYTES_PER_PARAM)}")
    if cfg.train.grad_dtype not in GRAD_DTYPE_BYTES:
        raise ConfigError("train.grad_dtype: допустимо fp32 или bf16")
    shape = resolve_model_shape(cfg.model)
    params = shape["num_params"]
    compute = _compute_dtype(cfg, master)

    master_bytes = params * DTYPE_BYTES[master]
    # bf16-мастер уже в compute dtype — отдельной копии нет. Это и есть
    # экономия 6 ГиБ на 3B по сравнению с fp32-мастером + bf16 autocast.
    working_copy_bytes = 0 if compute == master else params * DTYPE_BYTES[compute]
    optimizer_bytes = params * OPTIMIZER_BYTES_PER_PARAM[cfg.train.optimizer]
    gradient_bytes = params * GRAD_DTYPE_BYTES[cfg.train.grad_dtype]
    activation_bytes = _activation_bytes(cfg, shape, compute)

    components = {
        "master": master_bytes,
        "working_copy": working_copy_bytes,
        "optimizer": optimizer_bytes,
        "gradients": gradient_bytes,
        "activations": activation_bytes,
    }
    total = sum(components.values())
    budget = GPU_BUDGET_BYTES
    gib = {name: round(value / 1024**3, 3) for name, value in components.items()}
    assumptions = [
        f"мастер {master}, вычисления {compute}, оптимизатор {cfg.train.optimizer}",
        f"микро-батч {cfg.train.micro_batch_size}, длина {cfg.data.max_seq_len}, "
        f"grad_dtype {cfg.train.grad_dtype}, "
        f"checkpointing {'вкл' if cfg.train.gradient_checkpointing else 'выкл'}",
        "логиты оценены на всю длину (logits_to_keep только уменьшит)",
        f"workspace/контекст: {ACT_WORKSPACE_BYTES / 1024**3:.2f} ГиБ",
    ]
    result: dict[str, Any] = {
        "model": cfg.model.name_or_path,
        "params": params,
        "master_dtype": master,
        "compute_dtype": compute,
        "amp_dtype": cfg.train.amp_dtype,
        "optimizer": cfg.train.optimizer,
        "grad_dtype": cfg.train.grad_dtype,
        "gradient_checkpointing": bool(cfg.train.gradient_checkpointing),
        "micro_batch_size": cfg.train.micro_batch_size,
        "max_seq_len": cfg.data.max_seq_len,
        "components": components,
        "components_gib": gib,
        "total_bytes": total,
        "total_gib": round(total / 1024**3, 3),
        "budget_gib": round(budget / 1024**3, 1),
        "fits_24gb": total <= budget,
        "margin_gib": round((budget - total) / 1024**3, 3),
        "assumptions": assumptions,
    }
    # Словарь + готовая таблица: вызывающему не нужно знать про форматтер.
    result["table"] = format_memory_table(result)
    return result


_MEMORY_COLUMNS = (
    ("master", "мастер"),
    ("working_copy", "копия"),
    ("optimizer", "опт.сост"),
    ("gradients", "градиенты"),
    ("activations", "активации"),
)


def format_memory_table(estimate: dict[str, Any]) -> str:
    """Человекочитаемая таблица компонентов для одного конфига."""
    lines = [
        f"модель: {estimate['model']} — {estimate['params'] / 1e9:.2f}B параметров",
        f"режим: мастер {estimate['master_dtype']}, считаем {estimate['compute_dtype']}, "
        f"{estimate['optimizer']}, grad {estimate['grad_dtype']}, "
        f"checkpointing {'вкл' if estimate['gradient_checkpointing'] else 'выкл'}",
        f"батч {estimate['micro_batch_size']} × длина {estimate['max_seq_len']}",
        "",
        f"{'компонент':<14}{'ГиБ':>8}{'байт':>18}",
    ]
    for key, label in _MEMORY_COLUMNS:
        lines.append(
            f"{label:<14}{estimate['components_gib'][key]:>8.2f}"
            f"{estimate['components'][key]:>18,}"
        )
    lines.append(f"{'-' * 40}")
    lines.append(f"{'итого':<14}{estimate['total_gib']:>8.2f}{estimate['total_bytes']:>18,}")
    verdict = "влезает" if estimate["fits_24gb"] else "НЕ влезает"
    margin = estimate["margin_gib"]
    lines.append(
        f"бюджет {estimate['budget_gib']:.0f} ГиБ: {verdict}; "
        f"запас {margin:+.2f} ГиБ"
    )
    return "\n".join(lines)


def format_memory_matrix(cfg: Config) -> str:
    """Таблица 1.5B/3B × fp32/bf16-мастер × adam8bit/adamw.

    Прочие параметры (батч, длина, grad_dtype, checkpointing) берутся из
    текущего конфига — таблица отвечает на вопрос «что будет, если сменить
    только мастер/оптимизатор».
    """
    rows = memory_matrix(cfg)
    header = (
        f"{'модель':<8}{'мастер':<8}{'оптимизатор':<12}"
        f"{'мастер':>8}{'копия':>8}{'опт.':>8}{'град.':>8}{'акт.':>8}"
        f"{'всего':>8}{'запас':>8}  {'24 ГиБ':<7}"
    )
    lines = [
        f"матрица памяти: micro_batch={cfg.train.micro_batch_size}, "
        f"seq_len={cfg.data.max_seq_len}, grad_dtype={cfg.train.grad_dtype}, "
        f"checkpointing={'вкл' if cfg.train.gradient_checkpointing else 'выкл'}; "
        "числа в ГиБ",
        "",
        header,
        "-" * len(header),
    ]
    for row in rows:
        label = row["label"]
        lines.append(
            f"{label:<8}{row['master_dtype']:<8}{row['optimizer']:<12}"
            f"{row['components_gib']['master']:>8.2f}"
            f"{row['components_gib']['working_copy']:>8.2f}"
            f"{row['components_gib']['optimizer']:>8.2f}"
            f"{row['components_gib']['gradients']:>8.2f}"
            f"{row['components_gib']['activations']:>8.2f}"
            f"{row['total_gib']:>8.2f}{row['margin_gib']:>+8.2f}"
            f"  {'да' if row['fits_24gb'] else 'НЕТ':<7}"
        )
    fits = [row for row in rows if row["fits_24gb"]]
    lines.append("")
    lines.append(
        f"укладываются в 24 ГиБ: {len(fits)} из {len(rows)}"
        + (" — " + ", ".join(row["label"] + "/" + row["master_dtype"] + "/"
                            + row["optimizer"] for row in fits) if fits else "")
    )
    return "\n".join(lines)


def memory_matrix(cfg: Config) -> list[dict[str, Any]]:
    """Оценки для всех комбинаций известных моделей, мастеров и оптимизаторов."""
    rows: list[dict[str, Any]] = []
    for name, preset in MODEL_PRESETS.items():
        for master in ("fp32", "bf16"):
            for optimizer in ("adam8bit", "adamw"):
                model = replace(
                    cfg.model,
                    name_or_path=name,
                    master_dtype=master,
                    num_params=preset["num_params"],
                    hidden_size=preset["hidden_size"],
                    intermediate_size=preset["intermediate_size"],
                    num_hidden_layers=preset["num_hidden_layers"],
                    vocab_size=preset["vocab_size"],
                )
                candidate = replace(
                    cfg,
                    model=model,
                    train=replace(cfg.train, optimizer=optimizer),
                )
                estimate = estimate_memory(candidate)
                estimate["label"] = _preset_label(preset["num_params"])
                rows.append(estimate)
    return rows


def _preset_label(params: int) -> str:
    known = {1_543_712_768: "1.5B", 3_085_936_640: "3B"}
    return known.get(params, f"{params / 1e9:.2f}B")


def apply_overrides(
    cfg: Config,
    *,
    data_path: str | None = None,
    max_steps: int | None = None,
) -> Config:
    """CLI-переопределения конфига (--data, --max-steps)."""
    new = replace(cfg)
    if data_path:
        new.data = replace(new.data, path=data_path)
    if max_steps is not None:
        if max_steps <= 0:
            raise ConfigError("--max-steps должен быть > 0")
        new.train = replace(new.train, max_steps=max_steps)
    return new


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Сериализуемый снимок конфига для run.json/manifest.json."""

    def _convert(value: Any) -> Any:
        if isinstance(value, tuple):
            return [_convert(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [_convert(v) for v in value]
        raise TypeError(f"неожиданный тип в конфиге: {type(value).__name__}")

    return {name: {f.name: _convert(getattr(getattr(cfg, name), f.name))
                   for f in fields(getattr(cfg, name))}
            for name in _SECTIONS}


def resolve_data_path(cfg: Config) -> Path:
    """Путь к данным: абсолютный как есть, относительный — от корня репозитория."""
    path = Path(cfg.data.path).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)
