#!/usr/bin/env python3
"""Крошечная Qwen2: сквозная проверка цепочки QAT → .trit → .vtm → C.

Зачем
=====
Каждое звено проверено отдельно на синтетике, но **настоящие обученные веса
не проходили всю цепочку целиком**. Этот скрипт собирает Qwen2 на 4 009 984
параметра (без скачивания весов), обучает её QAT на ``data/train.jsonl`` и
проводит тот же чекпоинт через ``qat.export`` → ``tools/vtm_compile.py`` →
C-интерпретатор ``src/vtm/vtm``, после чего сверяет логиты HF и C.

Стадии (``--stage``):

* ``train``   — токенизация пар (question → answer, обрезка 512), тернарные
  Linear через ``qat.model.apply_ternary``, AdamW lr 3e-3, чекпоинт в
  ``runs/tiny/checkpoints/step_*``, ``run.json`` с ``architecture`` для
  экспорта (HF-конфиг крошечной модели на диск не сохраняется — экспорт
  читает архитектуру из ``run.json``, путь без transformers);
* ``export``  — ``qat.export``: ``ternary.trit`` + ``non_ternary.bin`` +
  ``manifest.json`` (+ ``ternary.pt``);
* ``compile`` — ``tools/vtm_compile.py --manifest ... --out runs/tiny/model.vtm``;
* ``run``     — ``./src/vtm/vtm run runs/tiny/model.vtm --tokens ... --max-new 8``;
* ``parity``  — логиты той же тернарной модели в HF против C-пробы на
  ``src/vtm/libvtm.a`` (CLI печатает только выбранный токен, не вектор).
* ``logits``  — «учитель» без API: логиты текущего tiny-чекпоинта для
  ``--logits-pairs`` пар в формате ``qat.kl``, HF-каталог учителя и два
  конфига ``qat.train`` (с KL и без) — задача 0107.
* ``topk``    — замер top-k против полного словаря: размер, память, KL и
  время шага для k=32/64/128 (задача 0111). В ``all`` не входит.
* ``all``     — всё подряд (по умолчанию), кроме ``logits`` и ``topk``.

Решения, которые стоит знать
============================
* **Токенизатор** — ``tests/vtm/fixtures/tokenizer.json`` (287 токенов,
  byte-level BPE). Его added_tokens идут с id 500–503 и в словарь модели
  (287) не входят, поэтому chat-шаблон не используется: промпт —
  обычный текст ``"Вопрос: …\\nОтвет:"``, а pad/eos заданы явно как id 0
  (он же настоящий токен таблицы — в фикстуре нет отдельного eos).
* **Масштабы тритов**: основной артефакт собран с ``--scale-dtype float32``,
  чтобы порог 1e-4 мерил расхождение рантайма, а не округление масштабов.
  Второй артефакт (``model_fp16.vtm``, ``export_fp16/``) — продовый
  ``scale_dtype: float16``: разница с ним показывает цену округления
  масштабов, а не ошибку C.
* **eps RMSNorm**: контейнер кодирует eps как ``imm/1000``, поэтому Qwen
  ``1e-6`` превращается в 0 (известное ограничение ``tools/vtm_compile.py``);
  вклад в логиты ~1e-6 относительных — учтено в объяснении max|Δ|.
* Скрипт ничего не скачивает, не ставит пакеты и не ходит в сеть. torch и
  transformers импортируются лениво: чистые функции (токенизация,
  архитектура, сверка логитов) работают и без них, их же проверяет
  ``tests/test_tiny_qat.py``.

Запуск::

    PYTHONPATH=.pylibs python3.13 scripts/tiny_qat.py all
    PYTHONPATH=.pylibs python3.13 scripts/tiny_qat.py train --steps 300
    python3 scripts/tiny_qat.py parity   # только сверка, если артефакты есть
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qat.util import human_bytes, utc_now, write_json_atomic  # noqa: E402

# --- константы крошечной модели (задача 0095) ------------------------------
MODEL_NAME = "tiny-qwen2-qat"
VOCAB_SIZE = 287
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 1024
NUM_HIDDEN_LAYERS = 4
NUM_ATTENTION_HEADS = 4
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS
EMBED_MODE = "trit"  # none | trit; ставится из --embed-mode

MAX_POSITIONS = 1024
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10000.0
EXPECTED_PARAMS = 4_009_984

MAX_SEQ_LEN = 512
MIN_ANSWER_TOKENS = 32
PAD_ID = 0
EOS_ID = 0

TOKENIZER_PATH = ROOT / "tests" / "vtm" / "fixtures" / "tokenizer.json"
TRAIN_PATH = ROOT / "data" / "train.jsonl"
VAL_PATH = ROOT / "data" / "val.jsonl"
DEFAULT_OUT = ROOT / "runs" / "tiny"
VTM_BIN = ROOT / "src" / "vtm" / "vtm"
VTM_LIB = ROOT / "src" / "vtm" / "libvtm.a"
COMPILER = ROOT / "tools" / "vtm_compile.py"

#: Формат промпта. Chat-шаблон недоступен: id служебных токенов (500+) вне
#: словаря модели (287), поэтому учим и спрашиваем обычным текстом.
PROMPT_TEMPLATE = "Вопрос: {question}\nОтвет:"


class TinyError(RuntimeError):
    """Стадия не может быть выполнена: нет данных, артефакта или инструмента."""


# ---------------------------------------------------------------------------
# Чистые функции (без torch): архитектура, токенизация, разбор и сверка
# ---------------------------------------------------------------------------


def architecture_dict() -> dict[str, Any]:
    """HF-подобный конфиг крошечной модели для ``run.json["architecture"]``.

    ``qat.export`` читает его как ``hf_config`` (``_read_run_architecture``),
    поэтому ключи должны быть в терминах HF: hidden_size, num_hidden_layers,
    num_attention_heads, tie_word_embeddings и т.д. Так экспорт работает без
    transformers и без сохранённого на диск конфига.
    """
    return {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": HIDDEN_SIZE,
        "intermediate_size": INTERMEDIATE_SIZE,
        "num_hidden_layers": NUM_HIDDEN_LAYERS,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB_SIZE,
        "max_position_embeddings": MAX_POSITIONS,
        "rms_norm_eps": RMS_NORM_EPS,
        "rope_theta": ROPE_THETA,
        "tie_word_embeddings": True,
        # В transformers 5.x Qwen2 держит bias у q/k/v (не у o_proj): он лежит
        # в non_ternary.bin, а компилятор применяет его как Q_BIAS/K_BIAS/V_BIAS.
        "attention_bias": True,
        "hidden_act": "silu",
        "torch_dtype": "float32",
        "use_cache": False,
    }


def prompt_text(question: str) -> str:
    """Текст промпта для вопроса (один и тот же в обучении и в инференсе)."""
    return PROMPT_TEMPLATE.format(question=str(question).strip())


def _ids(tokenizer: Any, text: str) -> list[int]:
    """Токенизировать без спецтокенов; поддержать оба API fast/slow."""
    out = tokenizer(text, add_special_tokens=False)
    ids = getattr(out, "input_ids", None)
    if ids is None:
        ids = out["input_ids"]
    return [int(token) for token in ids]


def encode_pair(
    tokenizer: Any,
    question: str,
    answer: str,
    *,
    max_seq_len: int = MAX_SEQ_LEN,
    min_answer_tokens: int = MIN_ANSWER_TOKENS,
    eos_id: int | None = EOS_ID,
) -> dict[str, Any] | None:
    """Пара → ``{input_ids, labels, answer_tokens}`` или None, если не влезла.

    Обрезка ровно до ``max_seq_len``: хвост промпта (если вопрос длинный) и
    хвост ответа. В отличие от ``qat.data.PairDataset`` длинные пары не
    выбрасываются — в наших данных короче 512 токенов почти нет (p50 2520),
    и выброс оставил бы от 3607 пар единицы. Метка eos добавляется последней
    и не обрезается; промпт получает label = -100.
    """
    if max_seq_len <= 0:
        raise TinyError("max_seq_len должен быть > 0")
    prompt_ids = _ids(tokenizer, prompt_text(question))
    answer_ids = _ids(tokenizer, str(answer).strip())
    if not answer_ids:
        return None
    # Хвост ответа не нужен, а eos (1 токен) обязан влезть.
    prompt_budget = max(0, max_seq_len - min_answer_tokens - (1 if eos_id is not None else 0))
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[len(prompt_ids) - prompt_budget:]
    room = max_seq_len - len(prompt_ids) - (1 if eos_id is not None else 0)
    answer_ids = answer_ids[:max(0, room)]
    if len(answer_ids) < min_answer_tokens:
        return None
    if eos_id is not None:
        answer_ids = answer_ids + [int(eos_id)]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "answer_tokens": len(answer_ids),
    }


def read_pairs(path: Path, limit: int = 0) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Прочитать JSONL (question/answer) без torch — как ``qat.data.load_pairs``."""
    path = Path(path)
    if not path.exists():
        raise TinyError(f"файл данных не найден: {path}")
    rows: list[dict[str, str]] = []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TinyError(f"{path}:{lineno}: битый JSON: {exc}") from exc
            question = row.get("question") if isinstance(row, dict) else None
            answer = row.get("answer") if isinstance(row, dict) else None
            if not isinstance(question, str) or not question.strip():
                skipped += 1
                continue
            if not isinstance(answer, str) or not answer.strip():
                skipped += 1
                continue
            rows.append({"question": question.strip(), "answer": answer.strip()})
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise TinyError(f"в {path} нет пригодных пар question/answer")
    return rows, {"pairs": len(rows), "skipped_invalid": skipped}


def tokenize_rows(
    tokenizer: Any,
    rows: Sequence[dict[str, str]],
    *,
    max_seq_len: int = MAX_SEQ_LEN,
    min_answer_tokens: int = MIN_ANSWER_TOKENS,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Все пары в тензорные примеры + статистика отброшенных."""
    samples: list[dict[str, Any]] = []
    stats = {"pairs": len(rows), "used": 0, "skipped": 0, "answer_tokens": 0}
    for row in rows:
        sample = encode_pair(
            tokenizer, row["question"], row["answer"],
            max_seq_len=max_seq_len, min_answer_tokens=min_answer_tokens,
        )
        if sample is None:
            stats["skipped"] += 1
            continue
        # Страховка: id вне словаря модели — это segfault/индекс в embedding.
        top = max(sample["input_ids"])
        if top >= VOCAB_SIZE or top < 0:
            raise TinyError(
                f"токенизатор выдал id {top} при словаре {VOCAB_SIZE}: "
                f"пары не годятся для крошечной модели"
            )
        samples.append(sample)
        stats["used"] += 1
        stats["answer_tokens"] += int(sample["answer_tokens"])
    if not samples:
        raise TinyError("после обрезки не осталось ни одной пары")
    stats["mean_answer_tokens"] = round(stats["answer_tokens"] / stats["used"], 1)
    return samples, stats


def compare_logits(hf_logits: Sequence[float], c_logits: Sequence[float]) -> dict[str, Any]:
    """Сверка векторов логитов: max|Δ|, средняя |Δ|, argmax и top-5."""
    if len(hf_logits) != len(c_logits):
        raise TinyError(
            f"длины логитов разные: HF {len(hf_logits)}, C {len(c_logits)}"
        )
    if not hf_logits:
        raise TinyError("пустые логиты")
    deltas = [abs(float(a) - float(b)) for a, b in zip(hf_logits, c_logits)]
    hf_top = sorted(range(len(hf_logits)), key=lambda i: hf_logits[i], reverse=True)
    c_top = sorted(range(len(c_logits)), key=lambda i: c_logits[i], reverse=True)
    scale = max(abs(float(v)) for v in hf_logits)
    return {
        "count": len(hf_logits),
        "max_abs_delta": max(deltas),
        "mean_abs_delta": sum(deltas) / len(deltas),
        "hf_abs_max": scale,
        "relative_max": (max(deltas) / scale) if scale else 0.0,
        "argmax_hf": hf_top[0],
        "argmax_c": c_top[0],
        "argmax_match": hf_top[0] == c_top[0],
        "top5_hf": hf_top[:5],
        "top5_c": c_top[:5],
        "top5_overlap": len(set(hf_top[:5]) & set(c_top[:5])),
    }


def parse_vtm_run_output(stdout: str) -> dict[str, Any]:
    """Разобрать вывод ``vtm run``: промпт, выбранные токены, счётчики."""
    result: dict[str, Any] = {"prompt": [], "generated": [], "tensors": None,
                              "instructions": None, "backend": None}
    for line in stdout.splitlines():
        line = line.strip()
        for prefix, key in (("prompt:", "prompt"), ("вывод:", "generated")):
            if line.startswith(prefix):
                tail = line[len(prefix):].strip()
                if tail and not tail.startswith("("):
                    result[key] = [int(part) for part in tail.split()]
        if line.startswith("тензоров:"):
            parts = line.replace(",", " ").split()
            for index, part in enumerate(parts):
                if part == "тензоров:" and index + 1 < len(parts):
                    result["tensors"] = int(parts[index + 1])
                elif part == "инструкций:" and index + 1 < len(parts):
                    result["instructions"] = int(parts[index + 1])
                elif part == "бэкенд:" and index + 1 < len(parts):
                    result["backend"] = parts[index + 1]
    return result


# ---------------------------------------------------------------------------
# Ленивые зависимости: torch/transformers только на стадиях обучения и сверки
# ---------------------------------------------------------------------------


def load_tokenizer(path: Path = TOKENIZER_PATH) -> Any:
    """``PreTrainedTokenizerFast`` из фикстуры; pad/eos задаются явно."""
    from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path))
    tokenizer.pad_token_id = PAD_ID
    tokenizer.eos_token_id = EOS_ID
    return tokenizer


def build_model() -> Any:
    """Крошечная Qwen2 из конфига — без скачивания весов."""
    from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: PLC0415

    arch = architecture_dict()
    config = Qwen2Config(
        vocab_size=arch["vocab_size"],
        hidden_size=arch["hidden_size"],
        intermediate_size=arch["intermediate_size"],
        num_hidden_layers=arch["num_hidden_layers"],
        num_attention_heads=arch["num_attention_heads"],
        num_key_value_heads=arch["num_key_value_heads"],
        max_position_embeddings=arch["max_position_embeddings"],
        tie_word_embeddings=True,
        rms_norm_eps=arch["rms_norm_eps"],
    )
    model = Qwen2ForCausalLM(config)
    model.config.use_cache = False
    return model


def quant_config(scale_dtype: str = "float32", embed_mode: str = "trit") -> Any:
    """``QuantConfig``: тернарные q/k/v/o/gate/up/down + эмбеддинг (none|trit)."""
    from qat.config import QuantConfig  # noqa: PLC0415

    return QuantConfig(
        block_size=128,
        scale_dtype=scale_dtype,
        quantize_embeddings=embed_mode,
    )


def run_config(scale_dtype: str) -> Any:
    """``qat.config.Config`` — ради ``config_to_dict`` (run.json, trainer_state)."""
    from qat.config import (  # noqa: PLC0415
        Config, DataConfig, ModelConfig, QuantConfig, TrainConfig,
    )

    return Config(
        model=ModelConfig(
            name_or_path=MODEL_NAME,
            revision="tiny-qat-cpu",
            trust_remote_code=False,
            master_dtype="fp32",
            torch_dtype="float32",
            attn_implementation="eager",
            num_params=EXPECTED_PARAMS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            vocab_size=VOCAB_SIZE,
        ),
        quant=QuantConfig(
            block_size=128,
            scale_dtype=scale_dtype,
            quantize_embeddings=EMBED_MODE,
        ),
        data=DataConfig(path="data/train.jsonl", max_seq_len=MAX_SEQ_LEN,
                        max_samples=0, min_answer_tokens=MIN_ANSWER_TOKENS),
        train=TrainConfig(
            micro_batch_size=8, grad_accum_steps=1, max_steps=500,
            lr=3e-3, min_lr_ratio=1.0, warmup_steps=0, weight_decay=0.0,
            grad_clip=1.0, amp_dtype="none", gradient_checkpointing=False,
            optimizer="adamw", log_every=100, save_every=500,
            keep_last_checkpoints=1, save_optimizer=False, device="cpu",
            seed=42,
        ),
    )


def collate(batch: Sequence[dict[str, Any]], pad_id: int, torch: Any) -> dict[str, Any]:
    """Склеить примеры в батч (padding по самой длинной), labels -100 у pad."""
    width = max(len(sample["input_ids"]) for sample in batch)
    rows = len(batch)
    input_ids = torch.full((rows, width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((rows, width), dtype=torch.long)
    labels = torch.full((rows, width), -100, dtype=torch.long)
    for index, sample in enumerate(batch):
        length = len(sample["input_ids"])
        input_ids[index, :length] = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask[index, :length] = 1
        labels[index, :length] = torch.tensor(sample["labels"], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "n_answer": int((labels != -100).sum().item()),
    }


def evaluate(model: Any, samples: Sequence[dict[str, Any]], *, batch_size: int,
             pad_id: int, torch: Any) -> dict[str, Any]:
    """Средний CE по токенам ответа на фиксированном наборе (eval, no_grad)."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    count = 0
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            tensor_batch = collate(chunk, pad_id, torch)
            n_answer = tensor_batch.pop("n_answer")
            if n_answer == 0:
                continue
            out = model(**tensor_batch)
            total_loss += float(out.loss.item()) * n_answer
            total_tokens += n_answer
            count += 1
    if was_training:
        model.train()
    if not total_tokens:
        raise TinyError("eval: в выборке нет токенов ответа")
    return {"loss": total_loss / total_tokens, "tokens": total_tokens, "batches": count}


# ---------------------------------------------------------------------------
# Стадия 1: QAT-обучение
# ---------------------------------------------------------------------------


def stage_train(args: argparse.Namespace) -> dict[str, Any]:
    """Построить крошечную Qwen2, навесить триты и обучить на парах."""
    import torch  # noqa: PLC0415

    from qat import checkpoint as checkpoint_mod  # noqa: PLC0415
    from qat.config import config_to_dict  # noqa: PLC0415
    from qat.model import apply_ternary, print_model_report  # noqa: PLC0415

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    started = time.time()
    print(f"[tiny] токенизатор: {TOKENIZER_PATH.relative_to(ROOT)}")
    tokenizer = load_tokenizer()
    train_rows, train_file = read_pairs(TRAIN_PATH, args.max_train_pairs)
    val_rows, val_file = read_pairs(VAL_PATH, args.eval_pairs)
    train_samples, train_stats = tokenize_rows(
        tokenizer, train_rows, max_seq_len=args.max_seq_len,
        min_answer_tokens=args.min_answer_tokens,
    )
    val_samples, val_stats = tokenize_rows(
        tokenizer, val_rows, max_seq_len=args.max_seq_len,
        min_answer_tokens=args.min_answer_tokens,
    )
    print(
        f"[tiny] данные: train {train_stats['used']}/{train_stats['pairs']} пар "
        f"(отброшено {train_stats['skipped']}, в среднем "
        f"{train_stats['mean_answer_tokens']} токенов ответа), "
        f"val {val_stats['used']} пар; обрезка {args.max_seq_len}"
    )

    model = build_model()
    model.train()
    quant = quant_config(args.scale_dtype, args.embed_mode)
    ternary_report = apply_ternary(model, quant)
    counts = print_model_report(model, ternary_report, torch.device("cpu"))
    if counts["total"] != EXPECTED_PARAMS:
        raise TinyError(
            f"параметров {counts['total']}, ожидалось {EXPECTED_PARAMS}: "
            f"конфигурация модели разошлась с задачей"
        )

    no_decay, decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.0},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )
    print(f"[tiny] оптимизатор: AdamW lr {args.lr:g}, батч {args.batch_size}, "
          f"шагов {args.steps}")

    before = evaluate(model, val_samples, batch_size=args.batch_size,
                      pad_id=PAD_ID, torch=torch)
    print(f"[tiny] val loss до обучения: {before['loss']:.4f} "
          f"({before['tokens']} токенов)")

    cfg = run_config(args.scale_dtype)
    cfg.train.max_steps = args.steps
    cfg.train.micro_batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.train.seed = args.seed
    cfg.train.log_every = args.log_every

    log_path = out_dir / "log.jsonl"
    log_path.write_text("", encoding="utf-8")
    history: list[dict[str, Any]] = [{"step": 0, "val_loss": before["loss"]}]
    train_losses: list[float] = []
    first_batch_loss: float | None = None
    step = 0
    epoch = 0
    rng = random.Random(args.seed)
    train_started = time.time()
    try:
        while step < args.steps:
            epoch += 1
            order = list(range(len(train_samples)))
            rng.shuffle(order)
            for start in range(0, len(order), args.batch_size):
                if step >= args.steps:
                    break
                chunk = [train_samples[index] for index in order[start:start + args.batch_size]]
                tensor_batch = collate(chunk, PAD_ID, torch)
                tensor_batch.pop("n_answer")
                out = model(**tensor_batch)
                loss = out.loss
                if first_batch_loss is None:
                    first_batch_loss = float(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                value = float(loss.item())
                train_losses.append(value)
                if step % args.log_every == 0 or step == 1:
                    elapsed = time.time() - train_started
                    record: dict[str, Any] = {
                        "step": step,
                        "loss": value,
                        "lr": args.lr,
                        "epoch": epoch,
                        "elapsed_s": round(elapsed, 2),
                    }
                    if step % args.eval_every == 0 or step == args.steps:
                        measured = evaluate(model, val_samples, batch_size=args.batch_size,
                                            pad_id=PAD_ID, torch=torch)
                        record["val_loss"] = measured["loss"]
                        history.append({"step": step, "val_loss": measured["loss"],
                                        "train_loss": value})
                    print(
                        f"[tiny] шаг {step}/{args.steps} loss={value:.4f}"
                        + (f" val={record['val_loss']:.4f}" if "val_loss" in record else "")
                        + f" ({elapsed:.0f} с)",
                        flush=True,
                    )
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        print(f"\n[tiny] прервано на шаге {step}: сохраняю чекпоинт")

    after = evaluate(model, val_samples, batch_size=args.batch_size,
                     pad_id=PAD_ID, torch=torch)
    print(f"[tiny] val loss после обучения: {after['loss']:.4f} "
          f"({after['tokens']} токенов)")

    trained_seconds = time.time() - train_started
    state = {
        "format_version": checkpoint_mod.STATE_FORMAT_VERSION,
        "step": step,
        "samples_seen": step * args.batch_size,
        "max_steps": args.steps,
        "config": config_to_dict(cfg),
        "data_fingerprint": {"path": str(TRAIN_PATH), "samples": train_stats["used"]},
        "created_at": utc_now(),
        "dry_run": False,
        "status": "done",
    }
    ckpt = checkpoint_mod.save_checkpoint(
        out_dir, step=step, model=model, optimizer=None, scaler=None,
        state=state, save_optimizer=False,
    )
    print(f"[tiny] чекпоинт: {ckpt.relative_to(ROOT)} "
          f"({human_bytes(checkpoint_mod.checkpoint_size(ckpt))})")

    metrics = {
        "stage": "train",
        "steps": step,
        "epochs": epoch,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "lr": args.lr,
        "seed": args.seed,
        "params": counts["total"],
        "ternary_params": counts["ternary"],
        "ternary_modules": len(ternary_report),
        "data": {
            "train": {**train_stats, "file": train_file},
            "val": {**val_stats, "file": val_file},
        },
        "val_loss_before": before["loss"],
        "val_loss_after": after["loss"],
        "val_loss_delta": after["loss"] - before["loss"],
        "batch_loss_first": first_batch_loss,
        "batch_loss_mean_first10": (
            sum(train_losses[:10]) / len(train_losses[:10]) if train_losses else None
        ),
        "batch_loss_mean_last10": (
            sum(train_losses[-10:]) / len(train_losses[-10:]) if train_losses else None
        ),
        "history": history,
        "checkpoint": str(ckpt),
        "trained_seconds": round(trained_seconds, 1),
        "total_seconds": round(time.time() - started, 1),
    }
    write_json_atomic(out_dir / "train_metrics.json", metrics)

    run_meta: dict[str, Any] = {
        "format": "eva32.qat.run",
        "format_version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "resumed": False,
        "dry_run": False,
        "config": config_to_dict(cfg),
        "config_path": str(Path(__file__).resolve()),
        "model": {
            "name_or_path": MODEL_NAME,
            "revision": "tiny-qat-cpu",
            "vocab_size": VOCAB_SIZE,
            "num_params": counts["total"],
            "ternary_params": counts["ternary"],
        },
        "quant": {
            "block_size": quant.block_size,
            "scale_dtype": quant.scale_dtype,
            "target_suffixes": list(quant.target_suffixes),
            "exclude_suffixes": list(quant.exclude_suffixes),
            "modules": ternary_report,
            "modules_count": len(ternary_report),
        },
        # Экспорт читает архитектуру отсюда (путь без transformers/HF-конфига).
        "architecture": architecture_dict(),
        "data": {"path": str(TRAIN_PATH), "samples": train_stats["used"]},
        "device": "cpu",
        "device_name": None,
        "optimizer": "adamw",
        "status": "done",
        "latest_step": step,
        "metrics": {
            "val_loss_before": before["loss"],
            "val_loss_after": after["loss"],
        },
    }
    write_json_atomic(out_dir / "run.json", run_meta)
    return metrics


# ---------------------------------------------------------------------------
# Стадия 1b: логиты учителя для KL-дистилляции (задача 0107)
# ---------------------------------------------------------------------------


def teacher_logits_rows(
    sequence_logits: Sequence[Any], prompt_len: int, answer_len: int,
) -> Sequence[Any]:
    """Строки логитов учителя для токенов ответа: позиции ``P-1 … P+A-2``.

    Логит на позиции ``i`` предсказывает метку ``i+1``, поэтому первый токен
    ответа (метка ``P``) предсказывает логит ``P-1``. Ровно эту же позицию
    берёт ученик (``student_logits[:, P-1]``), и ``qat.train._kl_sum``
    отсчитывает ``start_in_logits = first - 1``. Сдвиг на один — молча
    неверная дистилляция, поэтому сверка на старте: KL(ученик‖учитель) ≈ 0.

    Чистая функция: работает и со списками (тесты без torch), и с тензорами;
    тип среза сохраняется, поэтому вызывающий сам решает, нужен ли ``.tolist()``.
    """
    if prompt_len <= 0:
        raise TinyError(f"пустой промпт ({prompt_len} токенов): логиты учителя не выровнять")
    start = prompt_len - 1
    stop = start + answer_len
    if answer_len <= 0:
        raise TinyError(f"пустой ответ ({answer_len} токенов)")
    if stop > len(sequence_logits):
        raise TinyError(
            f"логитов {len(sequence_logits)}, а нужно {stop} "
            f"(промпт {prompt_len} + ответ {answer_len})"
        )
    return sequence_logits[start:stop]


def _kl_config(name_or_path: str, data_path: Path, *, kl: bool, steps: int,
               batch_size: int, lr: float, max_seq_len: int,
               min_answer_tokens: int, seed: int) -> dict[str, Any]:
    """Конфиг ``qat.train`` для tiny-модели (тот же путь, что на аренде)."""
    return {
        "model": {
            "name_or_path": name_or_path,
            "revision": "tiny-qat-cpu",
            "trust_remote_code": False,
            "master_dtype": "fp32",
            "attn_implementation": "eager",
            "num_params": EXPECTED_PARAMS,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_hidden_layers": NUM_HIDDEN_LAYERS,
            "vocab_size": VOCAB_SIZE,
        },
        "quant": {
            "block_size": 128,
            "scale_dtype": "float32",
            # Эмбеддинги не квантуем: tiny-чекпоинт обучался с fp32-таблицей,
            # иначе ученик стартует не с весов учителя и KL≈0 не проверить.
            "quantize_embeddings": "none",
        },
        "data": {
            "path": str(data_path),
            "max_seq_len": max_seq_len,
            "long_pairs": "truncate",
            "max_samples": 0,
            "min_answer_tokens": min_answer_tokens,
            "use_teacher_logits": bool(kl),
            "kl_temperature": 1.0,
            "kl_weight": 0.5,
            "kl_chunk_size": 64,
        },
        "train": {
            "micro_batch_size": batch_size,
            "grad_accum_steps": 1,
            "max_steps": steps,
            "lr": lr,
            "min_lr_ratio": 1.0,
            "warmup_steps": 0,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "amp_dtype": "none",
            "grad_dtype": "fp32",
            "gradient_checkpointing": False,
            "optimizer": "adamw",
            "log_every": 10,
            "save_every": 1000,
            "keep_last_checkpoints": 1,
            "save_optimizer": False,
            "device": "cpu",
            "seed": seed,
            "loss_chunk_size": 512,
        },
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Записать конфиг: yaml, а без pyyaml — JSON (YAML его читает как есть)."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError:  # pragma: no cover - pyyaml есть в .pylibs
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        return
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


def _export_teacher_model(model: Any, tokenizer: Any, out_dir: Path) -> int:
    """Каталог HF (fp32-мастер-веса + токенизатор) для ``qat.train``.

    tiny-чекпоинт — это state_dict **параметризованной** модели
    (``…parametrizations.weight.original``), а ``qat.train`` загружает обычный
    HF-каталог и навешивает триты сам. Поэтому параметризацию снимаем в
    мастер-веса (``leave_parametrized=False``): forward ученика на старте
    совпадает с учителем, и KL≈0 — это проверка выравнивания, а не совпадения.
    """
    import torch
    from torch.nn.utils.parametrize import remove_parametrizations

    for _name, module in model.named_modules():
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is not None and "weight" in parametrizations:
            remove_parametrizations(module, "weight", leave_parametrized=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=False)
    # eos=id 0 в фикстуре не помечен как special: без явного eos_token
    # qat.train.load_tokenizer не найдёт eos и упадёт.
    tokenizer.eos_token = tokenizer.convert_ids_to_tokens(EOS_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(out_dir))
    return sum(int(t.numel()) for t in model.state_dict().values())  # с tied-дублем


def _logits_setup(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """Общая подготовка стадий ``logits``/``topk``: чекпоинт, пары, учитель.

    Возвращает ``(context, dataset_parts)``: ``context`` — словарь с ``ckpt``,
    ``logits_out``, ``tokenizer``, ``teacher``, ``dataset``; ``dataset_parts`` —
    ``{"rows", "encoded"}`. Обе стадии снимают логиты с одного и того же
    учителя и на одних и тех же закодированных парах — иначе замер top-k и
    файл full-vocab были бы про разные данные.
    """
    import torch  # noqa: PLC0415

    from qat import checkpoint as checkpoint_mod  # noqa: PLC0415
    from qat.data import PairDataset, load_pairs  # noqa: PLC0415
    from qat.model import apply_ternary  # noqa: PLC0415

    ckpt = checkpoint_mod.latest_checkpoint(args.out)
    if ckpt is None:
        raise TinyError(
            f"нет чекпоинта в {Path(args.out) / 'checkpoints'}: сначала стадия train"
        )
    logits_out = Path(args.logits_out).expanduser()
    if not logits_out.is_absolute():
        logits_out = (ROOT / logits_out).resolve()
    logits_out.mkdir(parents=True, exist_ok=True)

    print(f"[tiny] учитель: {ckpt.relative_to(ROOT)}")
    tokenizer = load_tokenizer()
    rows = load_pairs(TRAIN_PATH, args.logits_pairs)
    dataset = PairDataset(
        rows, tokenizer,
        max_seq_len=args.max_seq_len,
        long_pairs="truncate",
        min_answer_tokens=args.min_answer_tokens,
        use_teacher_logits=False,
        vocab_size=VOCAB_SIZE,
    )
    encoded: list[dict[str, Any]] = []
    dropped = 0
    for index in range(len(rows)):
        sample = dataset.encode(index)
        if sample is None:
            dropped += 1
            continue
        encoded.append({"row": rows[index], "sample": sample})
    if not encoded:
        raise TinyError("ни одна пара не закодировалась: проверьте min_answer_tokens")
    if dropped:
        print(f"[tiny] ВНИМАНИЕ: {dropped} пар отброшено (не влезли/короткий ответ)")
    print(f"[tiny] пар: {len(encoded)}, обрезка {args.max_seq_len}, "
          f"токенов ответа в среднем "
          f"{sum(len(e['sample']['labels']) - e['sample']['labels'].count(-100) for e in encoded) / len(encoded):.0f}")

    device = torch.device("cpu")
    teacher = build_model()
    teacher.eval()
    apply_ternary(teacher, quant_config(args.scale_dtype, "none"))
    checkpoint_mod.restore_model(teacher, ckpt, device)
    context = {
        "ckpt": ckpt,
        "logits_out": logits_out,
        "tokenizer": tokenizer,
        "teacher": teacher,
        "dataset": dataset,
        "device": device,
    }
    return context, {"rows": rows, "encoded": encoded}


def _topk_path(logits_out: Path, k: int) -> Path:
    """Путь файла логитов: top-k пишется отдельно, чтобы не затирать full-vocab."""
    return logits_out / ("pairs_logits.jsonl" if k <= 0 else f"pairs_topk{k}.jsonl")


def stage_logits(args: argparse.Namespace) -> dict[str, Any]:
    """Логиты «учителя» (текущий tiny-чекпоинт) + конфиги прогонов qat.train.

    Без API и без сети: учитель — тот же tiny-чекпоинт, снятый после
    ``train``. Пишет в ``--logits-out`` (по умолчанию ``runs/tiny_kl``):

    * ``pairs_logits.jsonl`` (полный словарь) или ``pairs_topk{k}.jsonl``
      (``--top-k k``: только ``k`` индексов и логитов на токен ответа) —
      формат ``qat.kl``;
    * ``teacher_model/`` — HF-каталог для ``qat.train`` (fp32-мастер-веса);
    * ``config_kl.yaml`` / ``config_ce.yaml`` — два прогона, отличается только
      ``data.use_teacher_logits``; данные в них — записанный файл (top-k или
      полный), поэтому top-k проверяется тем же путём, что и full-vocab;
    * ``logits_metrics.json`` — размеры, CE и само-дистилляция: KL на старте
      (должна быть ~0, иначе разъехались позиции/промпт/токенизатор).
    """
    import torch  # noqa: PLC0415

    from qat.kl import (  # noqa: PLC0415
        TopKLogits, kl_div_sum_topk, kl_div_sum_torch, read_logits_file, read_topk_file,
        write_logits_file, write_topk_file,
    )
    from qat.model import apply_ternary  # noqa: PLC0415

    context, parts = _logits_setup(args)
    ckpt = context["ckpt"]
    logits_out = context["logits_out"]
    tokenizer = context["tokenizer"]
    teacher = context["teacher"]
    encoded = parts["encoded"]
    device = context["device"]
    top_k = int(getattr(args, "top_k", 0) or 0)
    if top_k < 0:
        raise TinyError(f"--top-k должен быть >= 0, а не {top_k}")

    out_rows: list[dict[str, Any]] = []
    answer_tokens = 0
    topk_mass = 0.0
    with torch.no_grad():
        for item in encoded:
            sample = item["sample"]
            input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
            prompt_len = next(i for i, value in enumerate(sample["labels"]) if value != -100)
            answer_len = len(sample["labels"]) - prompt_len
            logits = teacher(input_ids=input_ids, use_cache=False).logits[0]
            teacher_rows = teacher_logits_rows(logits, prompt_len, answer_len)
            row: dict[str, Any] = {
                "question": item["row"]["question"],
                "answer": item["row"]["answer"],
            }
            if top_k:
                # top-k строим сразу из тензора: полный словарь остаётся в
                # тензоре и не превращается в A×V Python-чисел ещё раз.
                if top_k > teacher_rows.size(-1):
                    raise TinyError(
                        f"--top-k {top_k} больше словаря {teacher_rows.size(-1)}")
                values, indices = torch.topk(teacher_rows.float(), top_k, dim=-1)
                row["topk"] = TopKLogits(indices.tolist(), values.tolist())
                probs = torch.softmax(teacher_rows.float(), dim=-1)
                topk_mass += float(
                    probs.gather(-1, indices).sum().item()) / max(1, answer_len)
            else:
                row["logits"] = teacher_rows.cpu().tolist()
            out_rows.append(row)
            answer_tokens += answer_len

    logits_path = _topk_path(logits_out, top_k)
    if top_k:
        written = write_topk_file(logits_path, out_rows, k=top_k)
    else:
        written = write_logits_file(logits_path, out_rows)
    print(f"[tiny] логиты: {written} пар -> {logits_path.relative_to(ROOT)} "
          f"({human_bytes(logits_path.stat().st_size)}"
          + (f", top-k={top_k}" if top_k else "") + ")")
    # Читаем записанный файл тем же парсером, что и тесты: артефакт для аренды
    # проверяется целиком по формату, а не «writer и reader когда-то сойдутся».
    if top_k:
        checked = read_topk_file(logits_path, vocab_size=VOCAB_SIZE, limit=3, max_k=top_k)
        print(f"[tiny] файл читается: {len(checked)} строк примеров, "
              f"{len(checked[0]['topk'])} x {checked[0]['topk'].k} top-k")
    else:
        checked = read_logits_file(logits_path, vocab_size=VOCAB_SIZE, limit=3)
        print(f"[tiny] файл читается: {len(checked)} строк примеров, "
              f"{len(checked[0]['logits'])} x {len(checked[0]['logits'][0])} логитов")

    teacher_dir = logits_out / "teacher_model"
    params = _export_teacher_model(teacher, tokenizer, teacher_dir)
    print(f"[tiny] учитель для qat.train: {teacher_dir.relative_to(ROOT)} "
          f"(элементов в state_dict: {params})")

    # --- сверка: ученик из HF-каталога на тех же весах vs файл логитов ------
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    student = AutoModelForCausalLM.from_pretrained(
        str(teacher_dir), dtype=torch.float32, attn_implementation="eager",
    )
    student.eval()
    apply_ternary(student, quant_config(args.scale_dtype, "none"))
    check_count = min(4, len(encoded))
    kl_total = 0.0
    tokens = 0
    max_delta = 0.0
    ce_total = 0.0
    with torch.no_grad():
        for index in range(check_count):
            sample = encoded[index]["sample"]
            prompt_len = next(i for i, value in enumerate(sample["labels"]) if value != -100)
            answer_len = len(sample["labels"]) - prompt_len
            input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
            labels = torch.tensor([sample["labels"]], dtype=torch.long, device=device)
            logits = student(input_ids=input_ids, use_cache=False).logits[0]
            student_rows = logits[prompt_len - 1:prompt_len - 1 + answer_len]
            if top_k:
                # Обе стороны усечены одинаково: у выровненных весов наборы
                # top-k совпадают, и KL по объединению — ровно 0. Сдвиг позиций
                # даёт заметную KL, то есть порог 1e-4 ловит именно его.
                teacher_topk = out_rows[index]["topk"]
                values, indices = torch.topk(student_rows.float(), top_k, dim=-1)
                student_topk = TopKLogits(indices.tolist(), values.tolist())
                kl_total += float(kl_div_sum_topk(teacher_topk, student_topk))
                teacher_indices = torch.tensor(
                    teacher_topk.indices, dtype=torch.long, device=device)
                teacher_values = torch.tensor(
                    teacher_topk.logits, dtype=torch.float32, device=device)
                picked = student_rows.float().gather(-1, teacher_indices)
                max_delta = max(max_delta, float(
                    (picked - teacher_values).abs().max().item()))
            else:
                teacher_rows = torch.tensor(out_rows[index]["logits"], dtype=torch.float32)
                kl_total += float(kl_div_sum_torch(torch, student_rows, teacher_rows).item())
                max_delta = max(max_delta, float(
                    (student_rows - teacher_rows).abs().max().item()))
            # sr[t] предсказывает метку P+t, поэтому метке P+1 отвечает sr[1].
            shift_logits = student_rows[1:]
            shift_labels = labels[0, prompt_len + 1:]
            ce_total += float(torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction="sum",
            ).item())
            tokens += int(shift_labels.numel())
    student_kl_per_token = kl_total / max(1, tokens)
    print(f"[tiny] сверка выравнивания: KL(учитель‖ученик) = "
          f"{student_kl_per_token:.3e} на токен, max|Δ| логитов = {max_delta:.3e}")
    if student_kl_per_token > 1e-4:
        raise TinyError(
            f"KL на старте {student_kl_per_token:.3e} на токен: ученик и учитель "
            f"не совпали — логиты учителя выровнены не на те позиции"
        )

    config_kl = logits_out / "config_kl.yaml"
    config_ce = logits_out / "config_ce.yaml"
    common = dict(
        steps=args.kl_steps, batch_size=args.batch_size, lr=args.kl_lr,
        max_seq_len=args.max_seq_len, min_answer_tokens=args.min_answer_tokens,
        seed=args.seed,
    )
    _write_yaml(config_kl, _kl_config(str(teacher_dir), logits_path, kl=True, **common))
    _write_yaml(config_ce, _kl_config(str(teacher_dir), logits_path, kl=False, **common))
    print(f"[tiny] конфиги: {config_kl.relative_to(ROOT)}, {config_ce.relative_to(ROOT)}")

    logits_bytes = logits_path.stat().st_size
    bytes_per_pair = logits_bytes / max(1, written)
    width = checked[0]["topk"].k if top_k else len(checked[0]["logits"][0])
    metrics = {
        "stage": "logits",
        "format": "topk" if top_k else "full",
        "top_k": top_k or None,
        "teacher_checkpoint": str(ckpt),
        "pairs": written,
        "answer_tokens": answer_tokens,
        "vocab_size": VOCAB_SIZE,
        "logits_bytes": logits_bytes,
        "logits_bytes_per_pair": bytes_per_pair,
        "logits_bytes_per_answer_token": logits_bytes / max(1, answer_tokens),
        "extrapolation_40k_pairs_bytes": bytes_per_pair * 40_000,
        "logits_path": str(logits_path),
        "logits_file_check": {
            "rows": len(checked),
            "width": width,
        },
        "teacher_model": str(teacher_dir),
        "configs": {"kl": str(config_kl), "ce": str(config_ce)},
        # Позиционная сверка: KL(ученик‖учитель) на одинаковых весах.
        "alignment": {
            "pairs_checked": check_count,
            "method": "union_topk" if top_k else "full_vocab",
            "kl_per_token": student_kl_per_token,
            "max_abs_logit_delta": max_delta,
            "teacher_ce": ce_total / max(1, tokens),
        },
    }
    if top_k:
        # Доля массы учителя, попавшая в top-k: именно остаток отброшен, и
        # именно он объясняет расхождение KL с полным словарём.
        metrics["alignment"]["topk_mass_mean"] = topk_mass / max(1, len(out_rows))
    write_json_atomic(logits_out / "logits_metrics.json", metrics)
    print(f"[tiny] метрики: {(logits_out / 'logits_metrics.json').relative_to(ROOT)}")
    return metrics


# ---------------------------------------------------------------------------
# Стадия 1c: замер top-k против полного словаря (задача 0111)
# ---------------------------------------------------------------------------


_MEASURE_WRAPPER = (
    "import json, resource, subprocess, sys\n"
    "code = subprocess.run(sys.argv[1:]).returncode\n"
    "usage = resource.getrusage(resource.RUSAGE_CHILDREN)\n"
    "print('[measure] ' + json.dumps(\n"
    "    {'returncode': code, 'maxrss': usage.ru_maxrss}))\n"
)


def _measure_peak_mib(callable_: Any) -> float:
    """Пик Python-аллокаций при вызове, МиБ (tracemalloc)."""
    import tracemalloc

    tracemalloc.start()
    try:
        callable_()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 2 ** 20


def _teacher_rows_pass(
    teacher: Any, encoded: Sequence[dict[str, Any]], device: Any,
) -> Any:
    """Генератор: (вопрос, ответ, логиты учителя [A, vocab]) по парам.

    Один проход forward на пару: так полный словарь нигде не накапливается —
    каждая строка сразу уходит в файл (или в top-k) и забывается.
    """
    import torch  # noqa: PLC0415

    with torch.no_grad():
        for item in encoded:
            sample = item["sample"]
            input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
            prompt_len = next(i for i, value in enumerate(sample["labels"]) if value != -100)
            answer_len = len(sample["labels"]) - prompt_len
            logits = teacher(input_ids=input_ids, use_cache=False).logits[0]
            yield (
                item["row"]["question"],
                item["row"]["answer"],
                teacher_logits_rows(logits, prompt_len, answer_len),
            )


def _topk_of_rows(rows: Any, k: int) -> Any:
    """``TopKLogits`` из тензора ``[A, V]``: индексы и логиты по убыванию."""
    import torch  # noqa: PLC0415

    from qat.kl import TopKLogits  # noqa: PLC0415

    values, indices = torch.topk(rows.float(), k, dim=-1)
    return TopKLogits(indices.tolist(), values.tolist())


def _student_rows(
    student: Any, sample: dict[str, Any], device: Any,
) -> Any:
    """Логиты ученика на позициях, которые предсказывают токены ответа."""
    import torch  # noqa: PLC0415

    with torch.no_grad():
        input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=device)
        prompt_len = next(i for i, value in enumerate(sample["labels"]) if value != -100)
        answer_len = len(sample["labels"]) - prompt_len
        logits = student(input_ids=input_ids, use_cache=False).logits[0]
        return logits[prompt_len - 1:prompt_len - 1 + answer_len]


def _fresh_run_dir(path: Path, *, root: Path) -> Path:
    """Пустой каталог прогона под ``root`` (qat.train отказывается писать в непустой)."""
    import shutil  # noqa: PLC0415

    resolved, resolved_root = path.resolve(), root.resolve()
    if resolved_root not in resolved.parents and resolved != resolved_root:
        raise TinyError(f"отказ чистить {resolved}: он вне {resolved_root}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _run_measured(argv: Sequence[str]) -> dict[str, Any]:
    """Запустить процесс и вернуть код возврата и пик RSS (МиБ) его самого.

    Обёртка в отдельном процессе нужна, чтобы ``ru_maxrss`` относился только к
    этому запуску, а не к максимуму по всем детям (иначе числа сравнивать нельзя).
    ``ru_maxrss`` в macOS — байты, в Linux — килобайты.
    """
    import subprocess  # noqa: PLC0415

    command = [sys.executable, "-c", _MEASURE_WRAPPER, *argv]
    # cwd=ROOT: дочерний `python -m qat.train` должен видеть пакет qat и данные
    # независимо от того, откуда запустили сам скрипт.
    process = subprocess.run(command, capture_output=True, text=True, cwd=str(ROOT))
    maxrss = 0
    for line in process.stdout.splitlines():
        if line.startswith("[measure] "):
            payload = json.loads(line[len("[measure] "):])
            maxrss = int(payload.get("maxrss") or 0)
    if process.returncode != 0:
        raise TinyError(
            f"qat.train завершился с кодом {process.returncode}:\n"
            + "\n".join(process.stdout.splitlines()[-20:])
            + "\n" + "\n".join(process.stderr.splitlines()[-20:])
        )
    scale = 1 if sys.platform == "darwin" else 1024
    return {"returncode": process.returncode,
            "maxrss_mib": maxrss * scale / 2 ** 20,
            "stdout_tail": process.stdout.splitlines()[-5:]}


def _step_timings(run_dir: Path) -> dict[str, Any]:
    """Шаги из log.jsonl прогона: step_sec, kl_sec, data_sec (последний шаг и среднее)."""
    records: list[dict[str, Any]] = []
    log_path = run_dir / "log.jsonl"
    if not log_path.exists():
        raise TinyError(f"нет {log_path}: qat.train не дошёл до логов")
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("step_sec") is not None:
                records.append(record)
    if not records:
        raise TinyError(f"в {log_path} нет шагов с step_sec")
    # Первый шаг включает ленивую инициализацию: для сравнения форматов берём
    # хвост (все шаги, кроме первого), а первый показываем отдельно.
    tail = records[1:] or records
    step_secs = sorted(r["step_sec"] for r in tail)
    return {
        "steps": len(records),
        "step_sec_last": records[-1]["step_sec"],
        "step_sec_mean": sum(step_secs) / len(step_secs),
        # Медиана устойчивее среднего: на ноутбуке один шаг легко ловит
        # чужую нагрузку и утраивает «время формата».
        "step_sec_median": step_secs[len(step_secs) // 2],
        "compute_sec_mean": sum(r.get("compute_sec", 0.0) for r in tail) / len(tail),
        "data_sec_mean": sum(r.get("data_sec", 0.0) for r in tail) / len(tail),
        "kl_sec_mean": (sum(r.get("kl_sec", 0.0) for r in tail) / len(tail)
                        if any("kl_sec" in r for r in tail) else None),
        "loss_last": records[-1].get("loss"),
        "kl_last": records[-1].get("kl"),
    }


def _write_benchmark_config(
    logits_out: Path, teacher_dir: Path, data_path: Path, name: str,
    args: argparse.Namespace, *, steps: int, batch_size: int, seed: int,
) -> Path:
    """Конфиг qat.train для замера шага; отличается только файлом логитов."""
    payload = _kl_config(
        str(teacher_dir), data_path, kl=True, steps=steps, batch_size=batch_size,
        lr=args.kl_lr, max_seq_len=args.max_seq_len,
        min_answer_tokens=args.min_answer_tokens, seed=seed,
    )
    # Каждый шаг в log.jsonl: иначе при 3 шагах замер есть только на первом,
    # а он включает прогрев.
    payload["train"]["log_every"] = 1
    path = logits_out / "topk" / f"config_{name}.yaml"
    _write_yaml(path, payload)
    return path


def _stream_peak_mib(
    path: Path, tokenizer: Any, k: int, *, seq_len: int = MAX_SEQ_LEN,
    limit: int = 0,
) -> float:
    """Пик памяти потокового чтения файла с логитами через PairStore + PairDataset."""
    from qat.data import PairDataset, PairStore  # noqa: PLC0415

    def read_all() -> None:
        store = PairStore(path, limit)
        dataset = PairDataset(
            store, tokenizer,
            max_seq_len=seq_len, long_pairs="truncate",
            min_answer_tokens=1, use_teacher_logits=True,
            vocab_size=VOCAB_SIZE, expected_top_k=(k or None),
        )
        for index in range(len(store)):
            dataset.encode(index)

    return _measure_peak_mib(read_all)


def _read_all_peak_mib(path: Path, k: int) -> float:
    """Пик памяти «прочитать всё в список» — то, что делает read_logits_file."""
    from qat.kl import read_logits_file, read_topk_file  # noqa: PLC0415

    def read_all() -> None:
        if k:
            read_topk_file(path, vocab_size=VOCAB_SIZE, max_k=k)
        else:
            read_logits_file(path, vocab_size=VOCAB_SIZE)

    return _measure_peak_mib(read_all)


def _time_stream_pass(
    path: Path, tokenizer: Any, k: int, *, seq_len: int, repeats: int = 3,
) -> float:
    """Лучшее время полного потокового прохода: PairStore + encode всех пар.

    Это «цена данных» без forward/backward: JSON-разбор строки и приведение
    логитов к тензору. ``kl_sec`` в логе шага меряет только KL, а ``data_sec``
    на 8 примерах шумит — здесь файл проходится целиком и берётся лучшее из
    повторов (холодный кеш файла не влияет на сравнение форматов).
    """
    from qat.data import PairDataset, PairStore  # noqa: PLC0415

    best = float("inf")
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        store = PairStore(path)
        dataset = PairDataset(
            store, tokenizer, max_seq_len=seq_len, long_pairs="truncate",
            min_answer_tokens=1, use_teacher_logits=True,
            vocab_size=VOCAB_SIZE, expected_top_k=(k or None),
        )
        for index in range(len(store)):
            dataset.encode(index)
        best = min(best, time.perf_counter() - started)
    return best


def stage_topk(args: argparse.Namespace) -> dict[str, Any]:
    """Замер top-k: k=32/64/128 против полного словаря на tiny (задача 0111).

    Что меряется и зачем:

    * **размер**: байт на пару и экстраполяция на 40 тыс. пар — сверка с
      оценкой «4,9 ТБ full-vocab против 4,1 ГБ top-k=64». В JSON выходит
      больше оценки: она считает ~4 байта на кандидата, а JSONL — ~7; зато
      формат читается любым парсером и не требует своего загрузчика;
    * **качество**: KL с top-k учителя против полной KL при **одном и том же**
      ученике — расхождение на токен и доля массы учителя, попавшая в top-k;
      иначе сравнивались бы разные модели, а не разные форматы;
    * **память**: пик потокового чтения (``PairStore``) против «прочитать всё
      в список», плюс тот же файл, повторённый N раз: потоковая память не
      должна расти с числом пар;
    * **время шага**: настоящий ``qat.train`` на нескольких шагах для
      full-vocab и каждого k (step_sec, kl_sec, data_sec, пик RSS процесса).
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    from qat import checkpoint as checkpoint_mod  # noqa: PLC0415
    from qat.kl import (  # noqa: PLC0415
        TopKLogits, iter_logits_file, kl_divergence, kl_divergence_topk_full,
        kl_div_sum, kl_div_sum_topk, log_softmax, topk_from_logits,
        write_logits_file, write_topk_file,
    )
    from qat.model import apply_ternary  # noqa: PLC0415

    values = sorted({int(part) for part in str(args.topk_values).split(",") if part.strip()})
    if not values or values[0] <= 0:
        raise TinyError(f"--topk-values: ожидались k > 0, а не {args.topk_values!r}")
    if values[-1] > VOCAB_SIZE:
        raise TinyError(f"--topk-values: k={values[-1]} больше словаря {VOCAB_SIZE}")
    if args.topk_steps <= 0:
        raise TinyError(f"--topk-steps должен быть > 0, а не {args.topk_steps}")
    if args.topk_mem_scale < 1:
        raise TinyError(f"--topk-mem-scale должен быть >= 1, а не {args.topk_mem_scale}")

    context, parts = _logits_setup(args)
    encoded = parts["encoded"]
    logits_out = context["logits_out"]
    tokenizer = context["tokenizer"]
    teacher = context["teacher"]
    device = context["device"]
    teacher_dir = logits_out / "teacher_model"
    if not teacher_dir.exists():
        raise TinyError(
            f"нет {teacher_dir}: сначала стадия logits (она пишет HF-каталог учителя)"
        )
    topk_dir = logits_out / "topk"
    topk_dir.mkdir(parents=True, exist_ok=True)

    # --- файлы: полный словарь и каждый k (один проход на файл) -------------
    print("[tiny] пишу логиты: full-vocab и " + ", ".join(f"k={k}" for k in values))
    files: dict[int, Path] = {}
    files[0] = _topk_path(logits_out, 0)
    written = write_logits_file(
        files[0],
        ({"question": question, "answer": answer, "logits": rows.tolist()}
         for question, answer, rows in _teacher_rows_pass(teacher, encoded, device)),
    )
    for k in values:
        path = _topk_path(logits_out, k)
        count = write_topk_file(
            path,
            ({"question": question, "answer": answer,
              "topk": _topk_of_rows(rows, k)}
             for question, answer, rows in _teacher_rows_pass(teacher, encoded, device)),
            k=k,
        )
        if count != written:
            raise TinyError(f"k={k}: строк {count}, а в full-vocab {written}")
        files[k] = path

    # --- размеры и экстраполяция -------------------------------------------
    answer_tokens = sum(
        len(item["sample"]["labels"]) - item["sample"]["labels"].count(-100)
        for item in encoded
    )
    variants: dict[str, dict[str, Any]] = {}
    for k, path in files.items():
        size = path.stat().st_size
        variants["full" if k == 0 else f"k{k}"] = {
            "k": k or None,
            "path": str(path),
            "bytes": size,
            "bytes_per_pair": size / written,
            "bytes_per_answer_token": size / max(1, answer_tokens),
            "extrapolation_40k_pairs_bytes": size / written * 40_000,
        }
    full_bytes_per_token = variants["full"]["bytes_per_answer_token"]
    for name, variant in variants.items():
        variant["size_vs_full"] = variant["bytes_per_answer_token"] / full_bytes_per_token

    # --- ученик для замера качества ----------------------------------------
    student = AutoModelForCausalLM.from_pretrained(
        str(teacher_dir), dtype=torch.float32, attn_implementation="eager",
    )
    student.eval()
    apply_ternary(student, quant_config(args.scale_dtype, "none"))
    student_ckpt: Path | None = None
    if args.topk_student:
        student_ckpt = Path(args.topk_student).expanduser()
        if not student_ckpt.exists():
            raise TinyError(f"--topk-student: нет {student_ckpt}")
    else:
        # Ученик из 0107 (обучен с KL на этих же парах): его веса отличаются от
        # учителя, поэтому расхождение top-k — про усечение, а не про KL(0).
        student_ckpt = checkpoint_mod.latest_checkpoint(logits_out / "kl")
        if student_ckpt is None:
            print("[tiny] ВНИМАНИЕ: ученик = учитель (чекпоинта kl нет): полная KL "
                  "равна 0, расхождение top-k — только сдвиг от усечения")
    if student_ckpt is not None:
        checkpoint_mod.restore_model(student, student_ckpt, device)
        print(f"[tiny] ученик для качества: {student_ckpt}")
    else:
        print("[tiny] ученик для качества: веса учителя (тождественный)")

    # --- качество: KL top-k против полной KL, по токенам --------------------
    quality = {k: {"kl_sum": 0.0, "delta_sum": 0.0, "delta_abs_max": 0.0,
                   "coverage_sum": 0.0} for k in values}
    kl_full_total = 0.0
    tokens = 0
    for index, row in enumerate(iter_logits_file(files[0], vocab_size=VOCAB_SIZE)):
        teacher_rows = row["logits"]
        student_rows = _student_rows(student, encoded[index]["sample"], device).tolist()
        if len(teacher_rows) != len(student_rows):
            raise TinyError(
                f"{row['question'][:40]!r}: строк учителя {len(teacher_rows)}, "
                f"ученика {len(student_rows)} — позиции разъехались"
            )
        for teacher_row, student_row in zip(teacher_rows, student_rows):
            probs = log_softmax(teacher_row)
            full_row_kl = kl_divergence(teacher_row, student_row)
            kl_full_total += full_row_kl
            tokens += 1
            for k in values:
                indices, logits_k = topk_from_logits(teacher_row, k)
                row_kl = kl_divergence_topk_full(indices, logits_k, student_row)
                delta = row_kl - full_row_kl
                quality[k]["kl_sum"] += row_kl
                quality[k]["delta_sum"] += delta
                quality[k]["delta_abs_max"] = max(
                    quality[k]["delta_abs_max"], abs(delta))
                quality[k]["coverage_sum"] += sum(math.exp(probs[i]) for i in indices)
    if tokens == 0:
        raise TinyError("ни одного токена в логитах: проверьте пары и обрезку")
    for k in values:
        variant = variants[f"k{k}"]
        variant.update({
            "kl_full_per_token": kl_full_total / tokens,
            "kl_topk_per_token": quality[k]["kl_sum"] / tokens,
            "kl_delta_mean_per_token": quality[k]["delta_sum"] / tokens,
            "kl_delta_abs_max_per_token": quality[k]["delta_abs_max"],
            "kl_relative_delta": (quality[k]["delta_sum"] / kl_full_total
                                  if kl_full_total else None),
            "coverage_mean": quality[k]["coverage_sum"] / tokens,
        })
    variants["full"]["kl_full_per_token"] = kl_full_total / tokens

    # --- память: потоковое чтение против «прочитать всё» --------------------
    print("[tiny] память: потоковое чтение (PairStore) против чтения всего файла")
    memory: dict[str, Any] = {}
    for k, path in files.items():
        name = "full" if k == 0 else f"k{k}"
        stream_peak = _stream_peak_mib(path, tokenizer, k, seq_len=args.max_seq_len)
        read_all_peak = _read_all_peak_mib(path, k)
        variants[name]["memory"] = {
            "stream_peak_mib": stream_peak,
            "read_all_peak_mib": read_all_peak,
            "pairs": written,
        }
        memory[name] = variants[name]["memory"]
        print(f"        {name:>4}: stream {stream_peak:7.1f} МиБ, "
              f"read-all {read_all_peak:7.1f} МиБ")
    # --- цена данных без forward/backward: чтение всех пар файла ----------
    print("[tiny] чтение данных: весь файл через PairStore + encode")
    for k, path in files.items():
        name = "full" if k == 0 else f"k{k}"
        seconds = _time_stream_pass(path, tokenizer, k, seq_len=args.max_seq_len)
        variants[name]["load"] = {
            "seconds": seconds,
            "pairs": written,
            "pairs_per_second": written / seconds if seconds else None,
            "mib_per_second": (path.stat().st_size / 2 ** 20 / seconds
                               if seconds else None),
        }
        print(f"        {name:>4}: {seconds:.2f} с на {written} пар "
              f"({written / seconds:.0f} пар/с)")

    # Тот же файл в N раз длиннее: потоковая память не должна вырасти, а
    # read-all — вырастет линейно. Это и есть критерий «не растёт с числом пар».
    scale_k = 64 if 64 in values else values[-1]
    scale_path = topk_dir / f"pairs_topk{scale_k}_x{args.topk_mem_scale}.jsonl"
    print(f"[tiny] масштаб: {scale_path.name} (k={scale_k} × {args.topk_mem_scale})")
    try:
        with scale_path.open("w", encoding="utf-8") as handle:
            for _ in range(args.topk_mem_scale):
                for row in iter_logits_file(files[scale_k], vocab_size=VOCAB_SIZE):
                    packed = {"question": row["question"], "answer": row["answer"],
                              "topk": row["topk"].to_json()}
                    handle.write(json.dumps(packed, ensure_ascii=False) + "\n")
        scaling = {
            "k": scale_k,
            "multiplier": args.topk_mem_scale,
            "pairs": written * args.topk_mem_scale,
            "bytes": scale_path.stat().st_size,
            "stream_peak_mib": _stream_peak_mib(
                scale_path, tokenizer, scale_k, seq_len=args.max_seq_len),
            "read_all_peak_mib": _read_all_peak_mib(scale_path, scale_k),
        }
        print(f"        пар {scaling['pairs']}: stream "
              f"{scaling['stream_peak_mib']:.1f} МиБ, "
              f"read-all {scaling['read_all_peak_mib']:.1f} МиБ")
    finally:
        scale_path.unlink(missing_ok=True)

    # --- время шага: настоящий qat.train на full и на каждом k ---------------
    print(f"[tiny] время шага: qat.train, по {args.topk_steps} шагов на формат")
    timing: dict[str, Any] = {}
    for k, path in files.items():
        name = "full" if k == 0 else f"k{k}"
        config_path = _write_benchmark_config(
            logits_out, teacher_dir, path, name, args,
            steps=args.topk_steps, batch_size=args.batch_size, seed=args.seed,
        )
        run_dir = _fresh_run_dir(topk_dir / "runs" / name, root=logits_out)
        measured = _run_measured([
            sys.executable, "-m", "qat.train",
            "--config", str(config_path),
            "--out", str(run_dir),
            "--max-steps", str(args.topk_steps),
            "--no-save",
        ])
        timings = _step_timings(run_dir)
        timing[name] = {
            "k": k or None,
            "data_path": str(path),
            "config": str(config_path),
            "run_dir": str(run_dir),
            "maxrss_mib": measured["maxrss_mib"],
            **timings,
        }
        variants[name]["timing"] = timing[name]
        print(f"        {name:>4}: step {timings['step_sec_mean']:.2f} с "
              f"(kl {timings['kl_sec_mean']:.3f} с, data {timings['data_sec_mean']:.2f} с), "
              f"RSS {measured['maxrss_mib']:.0f} МиБ")

    metrics = {
        "stage": "topk",
        "teacher_checkpoint": str(context["ckpt"]),
        "student_checkpoint": str(student_ckpt) if student_ckpt is not None else None,
        "student_is_teacher": student_ckpt is None,
        "vocab_size": VOCAB_SIZE,
        "pairs": written,
        "answer_tokens": answer_tokens,
        "topk_values": values,
        "full": variants["full"],
        "variants": {f"k{k}": variants[f"k{k}"] for k in values},
        "memory": memory,
        "scaling": scaling,
        "timing": timing,
        "notes": {
            "kl": "KL считается по top-k учителя и полному словарю ученика; "
                  "хвост учителя за top-k отброшен — это оценка, а не полная KL",
            "memory": "tracemalloc: пик Python-аллокаций на один проход; "
                      "read_all — весь файл списками (как read_logits_file)",
            "timing": "qat.train --no-save, последние шаги без первого (прогрев); "
                      "maxrss — пик RSS процесса обучения",
        },
    }
    write_json_atomic(topk_dir / "topk_metrics.json", metrics)
    print(f"[tiny] метрики: {(topk_dir / 'topk_metrics.json').relative_to(ROOT)}")
    for k in values:
        variant = variants[f"k{k}"]
        print(f"[tiny] k={k:>3}: {variant['bytes_per_pair'] / 1024:8.1f} КиБ/пара, "
              f"KL Δ {variant['kl_delta_mean_per_token']:+.3e}/токен "
              f"(полная {variant['kl_full_per_token']:.3e}), "
              f"масса в top-k {variant['coverage_mean']:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Стадия 2: экспорт qat.export
# ---------------------------------------------------------------------------


def stage_export(args: argparse.Namespace) -> dict[str, Any]:
    """Запустить ``qat.export`` на чекпоинте прогона (архитектура из run.json)."""
    from qat import export as qat_export  # noqa: PLC0415

    out_dir = Path(args.out)
    run_path = out_dir / "run.json"
    if not run_path.exists():
        raise TinyError(f"нет {run_path}: сначала стадия train")
    export_dir = out_dir / "export"
    argv = [
        "--run", str(out_dir),
        "--out", str(export_dir),
        "--force",
        "--scale-dtype", args.scale_dtype,
    ]
    print(f"[tiny] экспорт: qat.export {' '.join(argv)}")
    code = qat_export.main(argv)
    if code != 0:
        raise TinyError(f"qat.export завершился с кодом {code}")

    manifest_path = export_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pt_path = export_dir / "ternary.pt"
    files = {
        name: {
            "bytes": (export_dir / name).stat().st_size,
            "sha256": record.get("sha256"),
        }
        for name, record in (manifest.get("files") or {}).items()
    }
    metrics = {
        "stage": "export",
        "scale_dtype": args.scale_dtype,
        "export_dir": str(export_dir),
        "manifest": str(manifest_path),
        "trit_bytes": (export_dir / "ternary.trit").stat().st_size,
        "non_ternary_bytes": (export_dir / "non_ternary.bin").stat().st_size,
        "pt_bytes": pt_path.stat().st_size if pt_path.exists() else None,
        "files": files,
        "architecture": manifest.get("architecture"),
        "tensors": len(manifest.get("tensors") or []),
        "trit_params": (manifest.get("totals") or {}).get("trit_params"),
        "non_ternary_params": (manifest.get("totals") or {}).get("non_ternary_params"),
        "quantize_embeddings": manifest.get("embeddings", {}).get("mode"),
    }
    print(
        f"[tiny] экспорт готов: ternary.trit {human_bytes(metrics['trit_bytes'])}, "
        f"non_ternary.bin {human_bytes(metrics['non_ternary_bytes'])}, "
        f"тензоров {metrics['tensors']}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Стадия 3: компиляция .vtm
# ---------------------------------------------------------------------------


def stage_compile(args: argparse.Namespace) -> dict[str, Any]:
    """``tools/vtm_compile.py --manifest export/manifest.json --out model.vtm``."""
    out_dir = Path(args.out)
    manifest = out_dir / "export" / "manifest.json"
    if not manifest.exists():
        raise TinyError(f"нет {manifest}: сначала стадия export")
    vtm_path = out_dir / "model.vtm"
    cmd = [sys.executable, str(COMPILER), "--manifest", str(manifest),
           "--out", str(vtm_path)]
    print(f"[tiny] компиляция: {' '.join(cmd)}")
    proc = subprocess.run([str(part) for part in cmd], cwd=str(ROOT),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=600)
    if proc.returncode != 0:
        raise TinyError(
            f"компилятор вернул {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
        )
    print(proc.stdout.rstrip())
    metrics = {
        "stage": "compile",
        "vtm": str(vtm_path),
        "vtm_bytes": vtm_path.stat().st_size,
        "compiler_stdout": proc.stdout,
    }
    print(f"[tiny] контейнер: {vtm_path.relative_to(ROOT)} "
          f"({human_bytes(metrics['vtm_bytes'])})")
    return metrics


# ---------------------------------------------------------------------------
# Стадия 4: C-интерпретатор
# ---------------------------------------------------------------------------


def ensure_c_runtime() -> None:
    """Проверить собранный рантайм; собрать, только если его нет вовсе."""
    missing = [path for path in (VTM_BIN, VTM_LIB) if not path.exists()]
    if not missing:
        return
    makefile = ROOT / "src" / "vtm" / "Makefile"
    if not makefile.exists():
        raise TinyError(f"нет {VTM_BIN} и нет Makefile: C-рантайм не собрать")
    print(f"[tiny] C-рантайм не собран ({', '.join(str(p) for p in missing)}): "
          f"make -C src/vtm")
    proc = subprocess.run(["make", "-C", str(ROOT / "src" / "vtm")],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=900)
    if proc.returncode != 0:
        raise TinyError(f"make -C src/vtm упал ({proc.returncode}):\n{proc.stderr}")
    for path in (VTM_BIN, VTM_LIB):
        if not path.exists():
            raise TinyError(f"после make нет {path}")


def _prompt_ids(args: argparse.Namespace) -> tuple[list[int], str]:
    """Id промпта для C и HF: из ``--prompt`` или первый вопрос val."""
    tokenizer = load_tokenizer()
    if args.prompt:
        text = args.prompt
    else:
        rows, _ = read_pairs(VAL_PATH, 1)
        text = prompt_text(rows[0]["question"])
    ids = _ids(tokenizer, text)
    if args.prompt_tokens and len(ids) > args.prompt_tokens:
        ids = ids[len(ids) - args.prompt_tokens:]
    if not ids:
        raise TinyError("пустой промпт после токенизации")
    if max(ids) >= VOCAB_SIZE:
        raise TinyError(f"id {max(ids)} вне словаря {VOCAB_SIZE}")
    return ids, text


def stage_run(args: argparse.Namespace) -> dict[str, Any]:
    """Прогнать ``vtm run ... --tokens <ids> --max-new 8`` (штатный CLI)."""
    ensure_c_runtime()
    out_dir = Path(args.out)
    vtm_path = out_dir / "model.vtm"
    if not vtm_path.exists():
        raise TinyError(f"нет {vtm_path}: сначала стадия compile")
    token_ids, text = _prompt_ids(args)
    cmd = [
        str(VTM_BIN), "run", str(vtm_path),
        "--tokens", ",".join(str(token) for token in token_ids),
        "--max-new", str(args.max_new),
    ]
    print(f"[tiny] C-прогон: {' '.join(cmd[:3])} ... --max-new {args.max_new}")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=1800)
    (out_dir / "c_run.txt").write_text(
        f"$ {' '.join(cmd)}\n\nSTDOUT\n{proc.stdout}\nSTDERR\n{proc.stderr}",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise TinyError(
            f"vtm run вернул {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
        )
    print(proc.stdout.rstrip())
    parsed = parse_vtm_run_output(proc.stdout)
    metrics = {
        "stage": "run",
        "prompt_text": text,
        "token_ids": token_ids,
        "max_new": args.max_new,
        "generated": parsed["generated"],
        "tensors": parsed["tensors"],
        "instructions": parsed["instructions"],
        "backend": parsed["backend"],
        "stdout": proc.stdout,
    }
    print(f"[tiny] C выбрал токены: {parsed['generated']}")
    return metrics


# ---------------------------------------------------------------------------
# Стадия 5: сверка логитов HF ↔ C
# ---------------------------------------------------------------------------


def hf_logits(args: argparse.Namespace, token_ids: Sequence[int]) -> dict[str, Any]:
    """Логиты последней позиции в HF на тех же тернарных весах."""
    import torch  # noqa: PLC0415

    from qat import checkpoint as checkpoint_mod  # noqa: PLC0415
    from qat.model import apply_embedding_quant, apply_ternary  # noqa: PLC0415

    out_dir = Path(args.out)
    ckpt = checkpoint_mod.latest_checkpoint(out_dir)
    if ckpt is None:
        raise TinyError(f"в {out_dir}/checkpoints нет чекпоинтов: сначала train")
    model = build_model()
    quant = quant_config(args.scale_dtype, args.embed_mode)
    apply_ternary(model, quant)
    state = checkpoint_mod.load_model_state(ckpt)
    model.load_state_dict(state, strict=True)
    # Эмбеддинг квантуется ПОСЛЕ загрузки мастер-весов (0043): обёртка меняет
    # имена ключей, поэтому до load_state_dict она несовместима с чекпоинтом.
    # Без неё эталон остаётся fp32, а C читает тритовую таблицу — Δ = 3.07.
    apply_embedding_quant(model, quant)
    # При tied-весах lm_head — та же таблица, но параметризация висит на
    # embed_tokens, и lm_head.weight остаётся мастер-весами. В контейнере C
    # считает выход тритовой таблицей, поэтому эталон обязан тоже: иначе
    # сравниваются разные модели (Δ = 3.7 при правильном рантайме).
    if (quant.quantize_embeddings or "none") != "none" and getattr(
        model.config, "tie_word_embeddings", False
    ):
        from qat import ternary  # noqa: PLC0415
        w = model.model.embed_tokens.weight.detach()
        q = ternary.quantize_torch(w, quant.block_size)
        deq = q["trits"].float() * q["scales"].repeat_interleave(
            quant.block_size
        ).reshape(w.shape)
        model.model.embed_tokens.weight.data = deq.clone()
        if getattr(model, "lm_head", None) is not None:
            model.lm_head.weight.data = deq.clone()
    del state
    model.to(torch.device("cpu"))
    model.eval()
    ids = torch.tensor([list(token_ids)], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits[0, -1].to(torch.float32).tolist()
    return {
        "checkpoint": str(ckpt),
        "logits": logits,
        "argmax": max(range(len(logits)), key=lambda i: logits[i]),
        "abs_max": max(abs(value) for value in logits),
    }


def c_logits(out_dir: Path, vtm_path: Path, token_ids: Sequence[int]) -> dict[str, Any]:
    """Вектор логитов из C-пробы на ``src/vtm/libvtm.a``.

    ``vtm run`` печатает только выбранный токен, поэтому для max|Δ| нужен
    прямой вызов ``vtm_run`` + чтение тензора ``logits``. Готовая проба и её
    сборка живут в ``tools/parity_real.py`` (тот же libvtm.a, что у CLI).
    """
    ensure_c_runtime()
    try:
        from tools import parity_real  # noqa: PLC0415 - импорт внутри стадии
    except ImportError as exc:  # pragma: no cover - репозиторий без tools/
        raise TinyError(f"не импортируется tools/parity_real.py: {exc}") from exc
    cc = parity_real.find_c_compiler()
    if not cc:
        raise TinyError("нет C-компилятора (cc/clang/gcc) для пробы логитов")
    probe = parity_real.run_c_probe(out_dir, vtm_path, token_ids, cc)
    logits = [float(value) for value in probe.logits]
    return {
        "logits": logits,
        "argmax": max(range(len(logits)), key=lambda i: logits[i]),
        "abs_max": max(abs(value) for value in logits),
        "probe": {"compiler": cc, "source": str(out_dir / "parity_probe.c")},
    }


def stage_parity(args: argparse.Namespace) -> dict[str, Any]:
    """Сверить логиты HF и C на одном промпте (основной и fp16-артефакт)."""
    out_dir = Path(args.out)
    vtm_path = out_dir / "model.vtm"
    if not vtm_path.exists():
        raise TinyError(f"нет {vtm_path}: сначала стадия compile")
    token_ids, text = _prompt_ids(args)
    started = time.time()
    print(f"[tiny] сверка: промпт {len(token_ids)} токенов, "
          f"чекпоинт {out_dir.name}")
    hf = hf_logits(args, token_ids)
    print(f"[tiny] HF (тернарные веса): argmax {hf['argmax']}, "
          f"max|logit| {hf['abs_max']:.4f}")

    c = c_logits(out_dir, vtm_path, token_ids)
    comparison = compare_logits(hf["logits"], c["logits"])
    comparison["threshold"] = args.threshold
    comparison["passed"] = bool(
        comparison["max_abs_delta"] <= args.threshold and comparison["argmax_match"]
    )
    print(
        f"[tiny] C: argmax {c['argmax']}, max|Δ| {comparison['max_abs_delta']:.3e} "
        f"(порог {args.threshold:g}, средняя |Δ| {comparison['mean_abs_delta']:.3e}, "
        f"argmax {'совпал' if comparison['argmax_match'] else 'НЕ совпал'}, "
        f"top-5 {comparison['top5_overlap']}/5)"
    )

    fp16: dict[str, Any] | None = None
    fp16_vtm = out_dir / "model_fp16.vtm"
    if not args.skip_fp16 and fp16_vtm.exists():
        try:
            c_fp16 = c_logits(out_dir, fp16_vtm, token_ids)
            fp16 = compare_logits(hf["logits"], c_fp16["logits"])
            fp16["threshold"] = args.threshold
            fp16["passed"] = bool(
                fp16["max_abs_delta"] <= args.threshold and fp16["argmax_match"]
            )
            fp16["artifact"] = str(fp16_vtm)
            print(
                f"[tiny] C (fp16-масштабы): max|Δ| {fp16['max_abs_delta']:.3e}, "
                f"argmax {'совпал' if fp16['argmax_match'] else 'НЕ совпал'}"
            )
        except TinyError as exc:
            print(f"[tiny] ВНИМАНИЕ: fp16-артефакт не сверен: {exc}")
            fp16 = {"error": str(exc), "artifact": str(fp16_vtm)}

    result = {
        "stage": "parity",
        "prompt_text": text,
        "token_ids": list(token_ids),
        "checkpoint": hf["checkpoint"],
        "scale_dtype": args.scale_dtype,
        "hf_logits": hf["logits"],
        "c_logits": c["logits"],
        "hf_argmax": hf["argmax"],
        "c_argmax": c["argmax"],
        "max_abs_delta": comparison["max_abs_delta"],
        "mean_abs_delta": comparison["mean_abs_delta"],
        "hf_abs_max": comparison["hf_abs_max"],
        "relative_max": comparison["relative_max"],
        "argmax_match": comparison["argmax_match"],
        "top5_hf": comparison["top5_hf"],
        "top5_c": comparison["top5_c"],
        "top5_overlap": comparison["top5_overlap"],
        "threshold": args.threshold,
        "passed": comparison["passed"],
        "comparison": comparison,
        "fp16": fp16,
        "vtm": str(vtm_path),
        "seconds": round(time.time() - started, 1),
        "notes": [
            "эталон — та же тернарная модель (apply_ternary) на мастер-весах "
            "чекпоинта; расхождение только от порядка суммирования, RoPE/ATTN "
            "и eps RMSNorm (в контейнере imm/1000 = 0 вместо 1e-6)",
        ],
    }
    write_json_atomic(out_dir / "parity.json", result)
    if not comparison["passed"]:
        print(
            f"[tiny] ВНИМАНИЕ: порог {args.threshold:g} не пройден: "
            f"max|Δ| {comparison['max_abs_delta']:.3e}"
        )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


STAGES = ("all", "train", "logits", "topk", "export", "compile", "run", "parity")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 scripts/tiny_qat.py",
        description="QAT крошечной Qwen2 → .trit → .vtm → C: сквозная проверка цепочки.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  PYTHONPATH=.pylibs python3.13 scripts/tiny_qat.py all\n"
            "  PYTHONPATH=.pylibs python3.13 scripts/tiny_qat.py train --steps 300\n"
            "  python3 scripts/tiny_qat.py parity --skip-fp16\n"
        ),
    )
    parser.add_argument("--stage", choices=STAGES, default=None,
                        help="что делать (по умолчанию all)")
    parser.add_argument("command", nargs="?", default=None, choices=STAGES,
                        help="то же, что --stage (удобно: scripts/tiny_qat.py train)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"каталог артефактов (по умолчанию {DEFAULT_OUT})")
    parser.add_argument("--steps", type=int, default=500,
                        help="шагов обучения (по умолчанию 500; хватает, чтобы loss упал)")
    parser.add_argument("--batch-size", type=int, default=8, help="микро-батч (8)")
    parser.add_argument("--lr", type=float, default=3e-3, help="lr AdamW (3e-3)")
    parser.add_argument("--seed", type=int, default=42, help="seed (42)")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                        help=f"обрезка пары, токенов ({MAX_SEQ_LEN})")
    parser.add_argument("--min-answer-tokens", type=int, default=MIN_ANSWER_TOKENS,
                        help=f"минимум токенов ответа после обрезки ({MIN_ANSWER_TOKENS})")
    parser.add_argument("--max-train-pairs", type=int, default=0,
                        help="ограничить train пар (0 — все)")
    parser.add_argument("--eval-pairs", type=int, default=64,
                        help="пар val для замера loss (64)")
    parser.add_argument("--log-every", type=int, default=100,
                        help="логировать loss каждые N шагов (100)")
    parser.add_argument("--eval-every", type=int, default=100,
                        help="мерить val loss каждые N шагов (100)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="clip градиентов (1.0)")
    parser.add_argument("--embed-mode", choices=("none", "trit"), default="trit",
                        help="эмбеддинги: trit (по умолчанию, 0043) или none (fp32)")
    parser.add_argument("--scale-dtype", choices=("float32", "float16"), default="float32",
                        help="масштабы тритов в .trit (float32 — чтобы порог 1e-4 "
                             "мерил рантайм, float16 — продовый вариант)")
    parser.add_argument("--max-new", type=int, default=8,
                        help="токенов в C-прогоне (8)")
    parser.add_argument("--prompt", default=None,
                        help="текст промпта для C/HF (по умолчанию первый вопрос val)")
    parser.add_argument("--prompt-tokens", type=int, default=0,
                        help="обрезать промпт до N последних токенов (0 — весь)")
    parser.add_argument("--threshold", type=float, default=1e-4,
                        help="порог max|Δ| C↔HF (1e-4)")
    parser.add_argument("--skip-fp16", action="store_true",
                        help="не собирать/сверять второй артефакт с fp16-масштабами")
    parser.add_argument("--logits-out", default=str(ROOT / "runs" / "tiny_kl"),
                        help="каталог пары «логиты + конфиги qat.train» (runs/tiny_kl)")
    parser.add_argument("--logits-pairs", type=int, default=40,
                        help="пар с логитами учителя (40)")
    parser.add_argument("--kl-steps", type=int, default=150,
                        help="шагов в сгенерированных конфигах KL/CE (150)")
    parser.add_argument("--kl-lr", type=float, default=1e-3,
                        help="lr в сгенерированных конфигах KL/CE (1e-3)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="стадия logits: писать top-k из k кандидатов вместо "
                             "полного словаря (0 — полный словарь)")
    parser.add_argument("--topk-values", default="32,64,128",
                        help="стадия topk: ширины k через запятую (32,64,128)")
    parser.add_argument("--topk-steps", type=int, default=3,
                        help="стадия topk: шагов qat.train на формат (3)")
    parser.add_argument("--topk-student", default=None,
                        help="стадия topk: чекпоинт ученика для замера качества "
                             "(по умолчанию последний из <logits-out>/kl)")
    parser.add_argument("--topk-mem-scale", type=int, default=10,
                        help="стадия topk: во сколько раз удлинить файл для "
                             "проверки, что память не растёт (10)")
    return parser


def _update_report(out_dir: Path, stage: str, metrics: dict[str, Any]) -> None:
    """Сложить результаты стадии в runs/tiny/report.json (не теряя прошлых)."""
    path = out_dir / "report.json"
    report: dict[str, Any] = {}
    if path.exists():
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = {}
    report.setdefault("format", "eva32.tiny_qat.report")
    report["updated_at"] = utc_now()
    report[stage] = metrics
    summary = report.setdefault("summary", {})
    if stage == "train":
        summary["val_loss_before"] = metrics.get("val_loss_before")
        summary["val_loss_after"] = metrics.get("val_loss_after")
        summary["steps"] = metrics.get("steps")
    elif stage == "logits":
        alignment = metrics.get("alignment") or {}
        summary["kl_alignment"] = alignment.get("kl_per_token")
        summary["kl_pairs"] = metrics.get("pairs")
    elif stage == "topk":
        summary["topk_pairs"] = metrics.get("pairs")
        variants = metrics.get("variants") or {}
        summary["topk_bytes_per_pair"] = {
            name: variant.get("bytes_per_pair") for name, variant in variants.items()
        }
        summary["topk_kl_delta_per_token"] = {
            name: variant.get("kl_delta_mean_per_token")
            for name, variant in variants.items()
        }
    elif stage == "compile":
        summary["vtm_bytes"] = metrics.get("vtm_bytes")
    elif stage == "parity":
        summary["max_abs_delta"] = metrics.get("max_abs_delta")
        summary["argmax_match"] = metrics.get("argmax_match")
        summary["parity_passed"] = metrics.get("passed")
        if metrics.get("fp16"):
            summary["fp16_max_abs_delta"] = (metrics["fp16"] or {}).get("max_abs_delta")
    write_json_atomic(path, report)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.stage = args.command or args.stage or "all"
    out_dir = Path(args.out).expanduser()
    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()
    args.out = out_dir
    stages = ["train", "export", "compile", "run", "parity"] if args.stage == "all" \
        else [args.stage]
    started = time.time()
    try:
        for stage in stages:
            print(f"\n=== tiny_qat: {stage} ===")
            if stage == "train":
                metrics = stage_train(args)
            elif stage == "logits":
                metrics = stage_logits(args)
            elif stage == "topk":
                metrics = stage_topk(args)
            elif stage == "export":
                metrics = stage_export(args)
                if not args.skip_fp16 and args.scale_dtype != "float16":
                    print("\n=== tiny_qat: export (fp16-масштабы, для сравнения) ===")
                    fp16_args = argparse.Namespace(**vars(args))
                    fp16_args.scale_dtype = "float16"
                    fp16_args.out = out_dir
                    # Второй экспорт пишет в export_fp16/, компилятор — в
                    # model_fp16.vtm: основной маршрут остаётся float32.
                    _stage_export_variant(fp16_args, variant="fp16")
            elif stage == "compile":
                metrics = stage_compile(args)
                fp16_dir = out_dir / "export_fp16" / "manifest.json"
                if not args.skip_fp16 and fp16_dir.exists():
                    _stage_compile_variant(args, "fp16")
            elif stage == "run":
                metrics = stage_run(args)
            else:
                metrics = stage_parity(args)
            _update_report(out_dir, stage, metrics)
    except TinyError as exc:
        print(f"[tiny] ОШИБКА: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(
            f"[tiny] ОШИБКА: нет зависимости ({exc}). Токенизация нужна всем "
            f"стадиям (transformers), train/parity — ещё и torch:\n"
            f"    PYTHONPATH=.pylibs python3.13 scripts/tiny_qat.py {args.stage}",
            file=sys.stderr,
        )
        return 2
    summary = _summary_text(out_dir)
    if summary:
        print(f"\n=== итог ===")
        print(summary)
    print(f"\n[tiny] всего {time.time() - started:.0f} с; "
          f"артефакты: {out_dir}")
    return 0


def _stage_export_variant(args: argparse.Namespace, *, variant: str) -> dict[str, Any]:
    """Экспорт в ``export_<variant>/`` (fp16) — только для сравнения порогов."""
    from qat import export as qat_export  # noqa: PLC0415

    export_dir = Path(args.out) / f"export_{variant}"
    argv = [
        "--run", str(Path(args.out)),
        "--out", str(export_dir),
        "--force",
        "--scale-dtype", args.scale_dtype,
    ]
    print(f"[tiny] вариант {variant}: qat.export {' '.join(argv)}")
    code = qat_export.main(argv)
    if code != 0:
        print(f"[tiny] ВНИМАНИЕ: экспорт {variant} вернул {code}", file=sys.stderr)
    return {"export_dir": str(export_dir), "code": code}


def _stage_compile_variant(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    """Собрать ``model_<variant>.vtm`` из ``export_<variant>/manifest.json``."""
    out_dir = Path(args.out)
    manifest = out_dir / f"export_{variant}" / "manifest.json"
    vtm_path = out_dir / f"model_{variant}.vtm"
    cmd = [sys.executable, str(COMPILER), "--manifest", str(manifest),
           "--out", str(vtm_path)]
    print(f"[tiny] вариант {variant}: {' '.join(cmd)}")
    proc = subprocess.run([str(part) for part in cmd], cwd=str(ROOT),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=600)
    if proc.returncode != 0:
        print(f"[tiny] ВНИМАНИЕ: компилятор {variant} вернул {proc.returncode}: "
              f"{proc.stderr}", file=sys.stderr)
        return {"code": proc.returncode}
    print(f"[tiny] вариант {variant}: {vtm_path.relative_to(ROOT)} "
          f"({human_bytes(vtm_path.stat().st_size)})")
    return {"vtm": str(vtm_path), "vtm_bytes": vtm_path.stat().st_size}


def _summary_text(out_dir: Path) -> str:
    """Короткая сводка из report.json для конца прогона."""
    path = out_dir / "report.json"
    if not path.exists():
        return ""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    summary = report.get("summary") or {}
    lines: list[str] = []
    if "val_loss_before" in summary:
        lines.append(
            f"val loss: {summary['val_loss_before']:.4f} -> "
            f"{summary['val_loss_after']:.4f} за {summary.get('steps')} шагов"
        )
    if "kl_alignment" in summary:
        lines.append(
            f"логиты учителя: {summary.get('kl_pairs')} пар, "
            f"KL на старте {summary['kl_alignment']:.3e} на токен"
        )
    if "vtm_bytes" in summary:
        lines.append(f".vtm: {human_bytes(summary['vtm_bytes'])}")
    if "max_abs_delta" in summary:
        lines.append(
            f"max|Δ| C↔HF: {summary['max_abs_delta']:.3e} "
            f"(argmax {'совпал' if summary.get('argmax_match') else 'НЕ совпал'})"
        )
    if summary.get("fp16_max_abs_delta") is not None:
        lines.append(
            f"max|Δ| с fp16-масштабами: {summary['fp16_max_abs_delta']:.3e}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
