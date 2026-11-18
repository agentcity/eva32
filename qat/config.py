"""Конфиг QAT: dataclasses + чтение YAML.

YAML импортируется лениво, чтобы `python -m qat.train --help` и тесты
работали на машине без GPU-стека. Неизвестные ключи в YAML считаются
ошибкой: опечатка в имени параметра не должна молча теряться.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
DEFAULT_EXCLUDE_SUFFIXES = ("lm_head", "embed_tokens", "router", "gate", "norm")


class ConfigError(RuntimeError):
    """Конфиг не читается или содержит недопустимые значения."""


@dataclass
class ModelConfig:
    """Базовая модель. Ревизию на GPU-машине надо зафиксировать коммитом."""

    name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str = "main"
    trust_remote_code: bool = False
    #: Мастер-веса. float32, а не bfloat16: при lr ~3e-5 обновления в bf16 тонут.
    torch_dtype: str = "float32"
    attn_implementation: str = "sdpa"


@dataclass
class QuantConfig:
    """Что и как квантуем."""

    block_size: int = 128
    scale_dtype: str = "float16"
    target_suffixes: tuple[str, ...] = field(default=DEFAULT_TARGET_SUFFIXES)
    exclude_suffixes: tuple[str, ...] = field(default=DEFAULT_EXCLUDE_SUFFIXES)


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


def load_config(path: str | Path) -> Config:
    """Прочитать YAML и проверить значения."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"конфиг не найден: {path}")
    try:
        import yaml  # noqa: PLC0415 - ленивый импорт: без GPU-стека не нужен
    except ImportError as exc:
        raise ConfigError(
            "для чтения конфига нужен pyyaml. На GPU-машине:\n"
            "    python3 -m pip install -r qat/requirements-qat.txt"
        ) from exc
    with path.open(encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
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
    if len(t.betas) != 2 or not all(0.0 <= b < 1.0 for b in t.betas):
        raise ConfigError("train.betas должен быть парой чисел в [0, 1)")


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
