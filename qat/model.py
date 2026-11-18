"""Модель Qwen2.5: загрузка, тернарные Linear, оптимизатор и LR.

Тернарными становятся только Linear в attention и FFN (q/k/v/o/gate/up/down).
Эмбеддинги, lm_head, layernorm, bias и возможные роутеры остаются в fp32
и обучаются как обычно.

Мастер-веса держим в float32, а считаем в bf16 autocast: обновления при
lr ~3e-5 в bf16 просто округляются до нуля, поэтому bf16-мастер для QAT
не годится (см. docs/distillation.md).
"""
from __future__ import annotations

import inspect
import math
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from .config import Config, QuantConfig, TrainConfig
from .ternary import BLOCK_SIZE, ternary_weight_torch
from .util import human_count


class TernaryWeight(nn.Module):
    """Параметризация ``Linear.weight``: forward — тернарный вес, backward — STE."""

    def __init__(self, block_size: int = BLOCK_SIZE) -> None:
        super().__init__()
        self.block_size = block_size

    def forward(self, weight: torch.Tensor) -> torch.Tensor:  # noqa: D401 - это параметризация
        with torch.no_grad():
            w_tern = ternary_weight_torch(weight, self.block_size).detach()
        # STE: forward = w_tern, backward = dL/dw (см. ternary.ternary_ste_torch).
        w_tern = w_tern.to(weight.dtype)
        return weight + (w_tern - weight).detach()

    def extra_repr(self) -> str:
        return f"block_size={self.block_size}"


class TernaryEmbedding(nn.Module):
    """Параметризация ``Embedding.weight``: таблица словаря тоже троичная.

    Масштаб считается по последней оси (hidden), поэтому блок режется вдоль него.
    Так же работает int8-вариант — через отдельный класс ниже.
    """

    def __init__(self, block_size: int = BLOCK_SIZE) -> None:
        super().__init__()
        self.block_size = block_size

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            w_tern = ternary_weight_torch(weight, self.block_size).detach()
        w_tern = w_tern.to(weight.dtype)
        return weight + (w_tern - weight).detach()

    def extra_repr(self) -> str:
        return f"block_size={self.block_size}"


class Int8Embedding(nn.Module):
    """Параметризация ``Embedding.weight``: 8-битная таблица с масштабом на блок."""

    def __init__(self, block_size: int = BLOCK_SIZE) -> None:
        super().__init__()
        self.block_size = block_size

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            blocks = weight.reshape(-1, self.block_size) if weight.numel() % self.block_size == 0 else None
            if blocks is None:
                scale = weight.detach().abs().max().clamp_min(1e-8) / 127.0
                w_q = (weight.detach() / scale).round().clamp(-127, 127) * scale
            else:
                scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
                w_q = ((blocks / scale).round().clamp(-127, 127) * scale).reshape(weight.shape)
        w_q = w_q.to(weight.dtype)
        return weight + (w_q - weight).detach()

    def extra_repr(self) -> str:
        return f"block_size={self.block_size}"


def apply_embedding_quant(model: nn.Module, quant: QuantConfig) -> dict[str, Any] | None:
    """Троичные или int8 эмбеддинги — по `quant.quantize_embeddings`."""
    mode = (quant.quantize_embeddings or "none").lower()
    if mode == "none":
        return None
    if mode not in {"trit", "int8"}:
        raise ValueError(f"quantize_embeddings: неизвестный режим {mode!r}")
    target = None
    for name, module in model.named_modules():
        if name.endswith("embed_tokens") and isinstance(module, nn.Embedding):
            target = (name, module)
    if target is None:
        raise ValueError("в модели не найден слой embed_tokens для квантования эмбеддингов")
    name, module = target
    wrapper = TernaryEmbedding(quant.block_size) if mode == "trit" else Int8Embedding(quant.block_size)
    register_parametrization(module, "weight", wrapper)
    return {"name": name, "mode": mode, "block_size": quant.block_size}


def resolve_device(name: str) -> torch.device:
    """auto -> cuda, иначе mps/cpu; явное имя проверяется на доступность."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("в конфиге device=cuda, но CUDA недоступна")
    if device.type == "mps" and not (getattr(torch.backends, "mps", None)
                                     and torch.backends.mps.is_available()):
        raise RuntimeError("в конфиге device=mps, но MPS недоступна")
    return device


def torch_dtype(name: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if name not in mapping:
        raise ValueError(f"неизвестный dtype: {name!r}")
    return mapping[name]


def load_tokenizer(cfg: Config) -> Any:
    """Токенизатор базовой модели; pad_token при необходимости = eos."""
    from transformers import AutoTokenizer  # noqa: PLC0415 - тяжёлый импорт

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        revision=cfg.model.revision,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("у токенизатора нет ни pad_token, ни eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(cfg: Config, device: torch.device) -> nn.Module:
    """Загрузить CausalLM в fp32 и поставить на устройство."""
    from transformers import AutoModelForCausalLM  # noqa: PLC0415 - тяжёлый импорт

    kwargs: dict[str, Any] = {
        "revision": cfg.model.revision,
        "trust_remote_code": cfg.model.trust_remote_code,
        "torch_dtype": torch_dtype(cfg.model.torch_dtype),
        "low_cpu_mem_usage": True,
    }
    if cfg.model.attn_implementation:
        kwargs["attn_implementation"] = cfg.model.attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **kwargs)
    except TypeError:
        # transformers 5.x переименовал torch_dtype -> dtype; пробуем новый вариант.
        if "torch_dtype" not in kwargs:
            raise
        kwargs["dtype"] = kwargs.pop("torch_dtype")
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **kwargs)
    model.config.use_cache = False
    if cfg.train.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:  # старые transformers не знают kwargs
            model.gradient_checkpointing_enable()
    model.to(device)
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    return model


def _is_excluded(name: str, quant: QuantConfig) -> bool:
    """Исключения: не трогаем выход, эмбеддинги, роутеры и т.п."""
    if name.endswith(tuple(quant.exclude_suffixes)):
        return True
    parts = set(name.split("."))
    return bool(parts & {"lm_head", "embed_tokens", "router"})


def select_linear_modules(model: nn.Module, quant: QuantConfig) -> list[tuple[str, nn.Linear]]:
    """Linear-слои под квантование: только целевые суффиксы и не исключённые."""
    targets: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.endswith(tuple(quant.target_suffixes)):
            continue
        if _is_excluded(name, quant):
            continue
        if module.weight.ndim != 2:
            continue
        targets.append((name, module))
    return targets


def apply_ternary(model: nn.Module, quant: QuantConfig) -> list[dict[str, Any]]:
    """Зарегистрировать параметризацию на целевых Linear и вернуть отчёт.

    Повторный вызов на уже обёрнутой модели ничего не дублирует.
    """
    report: list[dict[str, Any]] = []
    for name, module in select_linear_modules(model, quant):
        already = hasattr(module, "parametrizations") and "weight" in module.parametrizations
        if not already:
            register_parametrization(module, "weight", TernaryWeight(quant.block_size))
        original = module.parametrizations.weight.original
        report.append({
            "name": name,
            "shape": [int(x) for x in original.shape],
            "params": int(original.numel()),
            "block_size": quant.block_size,
        })
    if not report:
        raise RuntimeError(
            "не найдено ни одного Linear под квантование; проверьте "
            "quant.target_suffixes и имя модели"
        )
    return report


def parameter_report(model: nn.Module, ternary_report: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Сколько параметров всего и сколько из них в тернарных слоях."""
    total = sum(p.numel() for p in model.parameters())
    ternary = sum(int(item["params"]) for item in ternary_report)
    return {"total": int(total), "ternary": ternary, "fp": int(total) - ternary}


def print_model_report(
    model: nn.Module,
    ternary_report: Sequence[dict[str, Any]],
    device: torch.device,
) -> dict[str, int]:
    """Короткая сводка в stdout: модель, слои, параметры, устройство."""
    counts = parameter_report(model, ternary_report)
    share = 100.0 * counts["ternary"] / counts["total"] if counts["total"] else 0.0
    device_name = ""
    if device.type == "cuda":
        device_name = f" ({torch.cuda.get_device_name(device)})"
    print(f"[qat] устройство: {device}{device_name}")
    print(f"[qat] параметров всего: {human_count(counts['total'])}, "
          f"в тернарных слоях: {human_count(counts['ternary'])} ({share:.1f}%)")
    print(f"[qat] тернарных Linear: {len(ternary_report)}; "
          f"fp-параметров (эмбеддинги, lm_head, нормы, bias): {human_count(counts['fp'])}")
    return counts


def build_optimizer(
    model: nn.Module,
    cfg: Config,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, str]:
    """8-bit Adam из bitsandbytes; при недоступности — AdamW с предупреждением.

    weight_decay не применяется к bias и одномерным параметрам (нормы).
    """
    train = cfg.train
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": train.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    common = {"lr": train.lr, "betas": train.betas, "eps": train.eps}

    if train.optimizer == "adam8bit":
        if device.type != "cuda":
            print("[qat] ВНИМАНИЕ: 8-bit Adam требует CUDA; беру AdamW")
        else:
            try:
                import bitsandbytes as bnb  # noqa: PLC0415 - опциональная зависимость

                optimizer = bnb.optim.Adam8bit(groups, **common)
                print("[qat] оптимизатор: bitsandbytes Adam8bit (fp32-мастер, 8-бит состояния)")
                return optimizer, "adam8bit"
            except Exception as exc:  # noqa: BLE001 - bnb падает по-разному
                print(f"[qat] ВНИМАНИЕ: bitsandbytes Adam8bit недоступен ({exc}); беру AdamW")
    optimizer = torch.optim.AdamW(groups, **common)
    print("[qat] оптимизатор: torch.optim.AdamW (fp32-состояния, памяти нужно больше)")
    return optimizer, "adamw"


def lr_at(step: int, cfg: TrainConfig, max_steps: int | None = None) -> float:
    """LR на шаге (1-based): линейный warmup, затем cosine до min_lr_ratio * lr."""
    total = max_steps if max_steps is not None else cfg.max_steps
    base = cfg.lr
    min_lr = base * cfg.min_lr_ratio
    if step <= 0:
        return base if cfg.warmup_steps <= 0 else 0.0
    if cfg.warmup_steps > 0 and step <= cfg.warmup_steps:
        return base * step / cfg.warmup_steps
    denom = max(1, total - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / denom)
    return min_lr + (base - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def logits_keep_arg(model: nn.Module) -> str | None:
    """Имя аргумента «оставить только последние K логитов», если transformers умеет.

    На длинных промптах это экономит память: lm_head считается не для всех
    2048 позиций, а только для хвоста, где лежит ответ.
    """
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return None
    for candidate in ("logits_to_keep", "num_logits_to_keep"):
        if candidate in params:
            return candidate
    return None
