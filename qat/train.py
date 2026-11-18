#!/usr/bin/env python3
"""QAT-заготовка EVA32: тернарное дообучение Qwen2.5 на парах дистилляции.

Запуск на GPU-машине::

    python3 -m pip install -r qat/requirements-qat.txt
    python3 -m qat.train --dry-run
    python3 -m qat.train --config qat/configs/qwen1.5b.yaml \\
        --data distill/out/pairs.jsonl --out runs/pilot1

Продолжить после обрыва::

    python3 -m qat.train --config qat/configs/qwen1.5b.yaml \\
        --out runs/pilot1 --resume

Модуль импортируется без torch: `--help` и проверка зависимостей работают на
машине разработчика, где GPU-стека нет. Реальные импорты — внутри main().
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from .config import (
    PROJECT_ROOT,
    ConfigError,
    apply_overrides,
    config_to_dict,
    load_config,
    resolve_data_path,
)
from .util import human_bytes, utc_now, write_json_atomic, write_text_atomic

DEFAULT_CONFIG = PROJECT_ROOT / "qat" / "configs" / "qwen1.5b.yaml"
REQUIRED_MODULES = (("torch", "torch"), ("transformers", "transformers"), ("yaml", "pyyaml"))


# ---------------------------------------------------------------------------
# CLI и проверка окружения
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m qat.train",
        description="QAT: тернарное дообучение Qwen2.5 на парах «вопрос → ответ».",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  python3 -m qat.train --dry-run\n"
            "      3 шага на случайных данных: проверить, что всё стартует\n\n"
            "  python3 -m qat.train --config qat/configs/qwen1.5b.yaml \\\n"
            "      --data distill/out/pairs.jsonl --out runs/pilot1\n"
            "      пилот целиком\n\n"
            "  python3 -m qat.train --config qat/configs/qwen1.5b.yaml \\\n"
            "      --out runs/pilot1 --resume\n"
            "      продолжить с последнего чекпоинта\n\n"
            "  python3 -m qat.train --config qat/configs/qwen1.5b.yaml \\\n"
            "      --out runs/pilot1 --resume --max-steps 3000\n"
            "      продолжить и увеличить число шагов\n\n"
            "зависимости ставятся только на GPU-машине:\n"
            "  python3 -m pip install -r qat/requirements-qat.txt\n\n"
            "проверка кода без GPU:\n"
            "  python3 -m unittest discover -s tests -v\n"
        ),
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG),
        help="YAML-конфиг (по умолчанию qat/configs/qwen1.5b.yaml)",
    )
    parser.add_argument(
        "--data", default=None,
        help="JSONL с парами question/answer; переопределяет data.path из конфига",
    )
    parser.add_argument(
        "--out", default=None,
        help="каталог прогона (по умолчанию runs/pilot1, для --dry-run runs/dryrun)",
    )
    parser.add_argument(
        "--resume", nargs="?", const="latest", default=None, metavar="CHECKPOINT",
        help="продолжить с чекпоинта; без значения — с последнего в runs/<name>/checkpoints",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="переопределить train.max_steps (например, дотянуть прогон)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="3 шага на случайных данных без датасета; проверить запуск и память",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="не писать чекпоинты (для быстрой проверки; при обрыве прогресс потеряется)",
    )
    parser.add_argument(
        "--allow-data-change", action="store_true",
        help="разрешить --resume, даже если файл данных изменился (по умолчанию это ошибка)",
    )
    return parser


def missing_modules() -> list[str]:
    """Каких зависимостей GPU-стека не хватает (без импорта тяжёлых модулей)."""
    missing: list[str] = []
    for module, package in REQUIRED_MODULES:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(package)
    return missing


def print_missing_dependencies(missing: Sequence[str]) -> None:
    print("[qat] не хватает зависимостей: " + ", ".join(missing))
    print()
    print("QAT запускается только на GPU-машине. Установите зависимости:")
    print("    python3 -m pip install -r qat/requirements-qat.txt")
    print()
    print("На машине разработчика проверить код можно без GPU-стека:")
    print("    python3 -m unittest discover -s tests -v")
    print("    python3 -m qat.train --help")
    print()
    print("Если torch уже стоит, но не тот: проверьте, что Python 3.10–3.12 "
          "и что установлен CUDA-вариант torch.")


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------


def _resolve_out(args: argparse.Namespace) -> Path:
    if args.out:
        path = Path(args.out).expanduser()
        return (path if path.is_absolute() else (Path.cwd() / path)).resolve()
    return (PROJECT_ROOT / "runs" / ("dryrun" if args.dry_run else "pilot1")).resolve()


def _resolve_resume(args: argparse.Namespace, out_dir: Path, checkpoint_mod: Any) -> Path | None:
    if args.resume is None:
        return None
    if not out_dir.exists():
        raise RuntimeError(f"--resume: каталога прогона нет: {out_dir}")
    if args.resume == "latest":
        found = checkpoint_mod.latest_checkpoint(out_dir)
        if found is None:
            raise RuntimeError(f"--resume: в {out_dir}/checkpoints нет чекпоинтов")
        return found
    path = Path(args.resume).expanduser()
    path = (path if path.is_absolute() else (Path.cwd() / path)).resolve()
    if not path.exists():
        raise RuntimeError(f"--resume: путь не существует: {path}")
    if (path / "model.pt").exists():
        return path
    found = checkpoint_mod.latest_checkpoint(path)
    if found is None:
        raise RuntimeError(f"--resume: в {path} нет чекпоинтов")
    return found


def _rng_state(torch: Any, device: Any) -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _make_scaler(torch: Any, device: Any, amp_dtype: str) -> Any:
    enabled = amp_dtype == "float16" and device.type == "cuda"
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):  # torch < 2.3 или другой набор аргументов
        return torch.cuda.amp.GradScaler(enabled=True)


def _autocast(torch: Any, device: Any, amp_dtype: str) -> Any:
    if amp_dtype == "none" or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _gpu_mem_gb(torch: Any, device: Any) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    return (
        torch.cuda.memory_allocated() / 2**30,
        torch.cuda.max_memory_allocated() / 2**30,
    )


def _first_supervised(labels: Any) -> int | None:
    positions = (labels != -100).nonzero(as_tuple=False)
    if positions.numel() == 0:
        return None
    return int(positions[:, 1].min().item())


def _chunked_ce(
    torch: Any,
    logits: Any,
    labels: Any,
    chunk_size: int,
    logits_start: int = 0,
) -> tuple[Any, Any]:
    """CE по ответу, сумма по токенам; логиты режем по длине, чтобы не пухло.

    ``logits_start`` — позиция первого логита в исходной последовательности:
    при ``logits_to_keep`` модель возвращает только хвост, и метки надо
    сдвинуть на столько же.
    """
    import torch.nn.functional as F  # noqa: PLC0415 - ленивый импорт

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, logits_start + 1:]
    if shift_logits.size(1) != shift_labels.size(1):
        raise RuntimeError("не выровнены логиты и метки: проверьте logits_to_keep")
    total = (shift_labels != -100).sum()
    loss_sum = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, shift_labels.size(1), chunk_size):
        part_logits = shift_logits[:, start:start + chunk_size, :]
        part_labels = shift_labels[:, start:start + chunk_size]
        loss_sum = loss_sum + F.cross_entropy(
            part_logits.reshape(-1, part_logits.size(-1)),
            part_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
    return loss_sum, total


def _kl_sum(
    torch: Any,
    logits: Any,
    labels: Any,
    teacher_logits: Sequence[Any],
    logits_start: int,
    temperature: float,
    chunk_size: int,
) -> Any:
    """KL(учитель || ученик) по токенам ответа, с температурой T².

    logits — срез модели начиная с позиции ``logits_start``; teacher_logits[b]
    выровнены на токены ответа (первый supervised-токен). Считаем по кускам,
    чтобы [длина ответа, словарь] не раздувал память.
    """
    import torch.nn.functional as F  # noqa: PLC0415

    batch_size = logits.size(0)
    total = logits.new_zeros((), dtype=torch.float32)
    for b in range(batch_size):
        supervised = (labels[b] != -100).nonzero(as_tuple=False).reshape(-1)
        teacher = teacher_logits[b]
        if supervised.numel() == 0 or teacher is None:
            continue
        first = int(supervised[0].item())
        start_in_logits = first - logits_start - 1
        if start_in_logits < 0:
            raise RuntimeError("не выровнены логиты ученика и учителя")
        count = min(int(teacher.size(0)), int(supervised.numel()),
                    logits.size(1) - start_in_logits)
        if count <= 0:
            continue
        if int(teacher.size(1)) != int(logits.size(-1)):
            raise RuntimeError(
                f"словарь учителя {teacher.size(1)} != словаря ученика {logits.size(-1)}"
            )
        for start in range(0, count, chunk_size):
            stop = min(count, start + chunk_size)
            student = logits[b, start_in_logits + start:start_in_logits + stop].float()
            target = teacher[start:stop].to(device=logits.device, dtype=torch.float32)
            total = total + F.kl_div(
                F.log_softmax(student / temperature, dim=-1),
                F.log_softmax(target / temperature, dim=-1),
                log_target=True,
                reduction="sum",
            ) * (temperature ** 2)
    return total



# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    missing = missing_modules()
    if missing:
        print_missing_dependencies(missing)
        return 2

    try:
        cfg = load_config(args.config)
        cfg = apply_overrides(
            cfg,
            data_path=args.data,
            max_steps=(3 if args.dry_run else args.max_steps),
        )
    except ConfigError as exc:
        print(f"[qat] ошибка конфига: {exc}")
        return 2
    if args.dry_run and args.max_steps is not None and args.max_steps != 3:
        print("[qat] --dry-run всегда делает 3 шага; --max-steps проигнорирован")

    out_dir = _resolve_out(args)
    if not out_dir.exists() and args.resume is not None:
        print(f"[qat] --resume: каталога прогона нет: {out_dir}")
        return 2
    if out_dir.exists() and args.resume is None and not args.dry_run:
        if not out_dir.is_dir() or any(out_dir.iterdir()):
            print(f"[qat] каталог прогона уже существует и не пуст: {out_dir}")
            print("       продолжить: --resume; или укажите другой --out")
            return 2

    # --- тяжёлые импорты только после проверки зависимостей -----------------
    import torch  # noqa: PLC0415
    from . import checkpoint as checkpoint_mod  # noqa: PLC0415
    from .data import (  # noqa: PLC0415
        DataError, PairDataset, PairStream, RandomTokenStream, data_fingerprint, load_pairs,
    )
    from .model import (  # noqa: PLC0415
        apply_ternary, build_optimizer, load_model, load_tokenizer, logits_keep_arg, lr_at,
        print_model_report, resolve_device,
    )

    try:
        resume_ckpt = _resolve_resume(args, out_dir, checkpoint_mod)
    except RuntimeError as exc:
        print(f"[qat] {exc}")
        return 2

    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)

    # --- данные (до модели: битый JSONL должен падать за секунды) -----------
    data_path = resolve_data_path(cfg)
    fingerprint: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    if not args.dry_run:
        try:
            rows = load_pairs(data_path, cfg.data.max_samples)
            fingerprint = data_fingerprint(data_path, len(rows))
        except DataError as exc:
            print(f"[qat] ошибка данных: {exc}")
            return 2

    # --- проверка совместимости с предыдущим прогоном ----------------------
    run_json_path = out_dir / "run.json"
    old_run: dict[str, Any] | None = None
    if run_json_path.exists():
        try:
            old_run = json.loads(run_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            old_run = None
    # --dry-run не должен затирать настоящий прогон: разрешаем непустой каталог,
    # только если он сам помечен как dry-run (или это первый запуск).
    if args.dry_run and args.resume is None and out_dir.is_dir():
        if old_run is not None and not old_run.get("dry_run"):
            print(f"[qat] {out_dir} — настоящий прогон; --dry-run в него писать нельзя")
            print("       укажите другой --out, например runs/dryrun-2")
            return 2
        if old_run is None and any(out_dir.iterdir()):
            print(f"[qat] {out_dir} не пуст и не помечен как dry-run; "
                  f"укажите другой --out")
            return 2
    if resume_ckpt is not None and old_run is not None:
        old_data = old_run.get("data") or {}
        new_sha = (fingerprint or {}).get("sha256")
        if old_data.get("sha256") and new_sha and old_data["sha256"] != new_sha:
            if not args.allow_data_change:
                print("[qat] файл данных изменился с начала прогона; --resume остановлен.")
                print("       это защита от продолжения на других данных. Если так и надо:")
                print("       добавьте --allow-data-change (или начните новый --out)")
                return 2
            print("[qat] ВНИМАНИЕ: данные изменились, продолжаю из-за --allow-data-change")
        old_quant = old_run.get("quant") or {}
        if old_quant.get("block_size") not in (None, cfg.quant.block_size):
            print(f"[qat] block_size в прогоне {old_quant.get('block_size')}, "
                  f"в конфиге {cfg.quant.block_size}: так нельзя")
            return 2

    if args.resume is not None and resume_ckpt is None:
        print("[qat] --resume: чекпоинт не найден")
        return 2

    device = resolve_device(cfg.train.device)
    print(f"[qat] конфиг: {Path(args.config).resolve()}")
    print(f"[qat] режим: {'dry-run (3 шага, случайные данные)' if args.dry_run else 'обучение'}")
    if fingerprint:
        print(f"[qat] данные: {data_path} — {len(rows)} пар, "
              f"{human_bytes(fingerprint['bytes'])}, sha256 {fingerprint['sha256'][:12]}…")
    print(f"[qat] батч: микро {cfg.train.micro_batch_size} × накопление "
          f"{cfg.train.grad_accum_steps} = {cfg.train.micro_batch_size * cfg.train.grad_accum_steps} "
          f"последовательностей, длина {cfg.data.max_seq_len}")
    print(f"[qat] grad checkpointing: {'вкл' if cfg.train.gradient_checkpointing else 'выкл'}, "
          f"amp: {cfg.train.amp_dtype}, мастер-веса: {cfg.model.torch_dtype}")
    if cfg.model.revision == "main":
        print("[qat] ВНИМАНИЕ: revision=main; на GPU-машине зафиксируйте commit в конфиге")

    # --- модель -------------------------------------------------------------
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg, device)
    ternary_report = apply_ternary(model, cfg.quant)
    counts = print_model_report(model, ternary_report, device)
    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))

    optimizer, optimizer_name = build_optimizer(model, cfg, device)
    scaler = _make_scaler(torch, device, cfg.train.amp_dtype)
    if cfg.train.amp_dtype == "bfloat16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            print("[qat] ВНИМАНИЕ: GPU не умеет bf16; поставьте amp_dtype: float16 или none")
            return 2
    keep_arg = logits_keep_arg(model)

    # --- чекпоинт и возобновление ------------------------------------------
    start_step = 0
    start_samples = 0
    last_saved_step = -1
    if resume_ckpt is not None:
        state = checkpoint_mod.load_trainer_state(resume_ckpt)
        if state.get("format_version") != checkpoint_mod.STATE_FORMAT_VERSION:
            print(f"[qat] чекпоинт версии {state.get('format_version')}, "
                  f"код ждёт {checkpoint_mod.STATE_FORMAT_VERSION}")
            return 2
        saved_cfg = state.get("config") or {}
        saved_model = (saved_cfg.get("model") or {}).get("name_or_path")
        if saved_model and saved_model != cfg.model.name_or_path:
            print(f"[qat] чекпоинт от модели {saved_model}, а в конфиге {cfg.model.name_or_path}")
            return 2
        saved_quant = (saved_cfg.get("quant") or {}).get("block_size")
        if saved_quant and int(saved_quant) != cfg.quant.block_size:
            print(f"[qat] чекпоинт с block_size={saved_quant}, в конфиге {cfg.quant.block_size}")
            return 2
        saved_fp = state.get("data_fingerprint") or {}
        if (fingerprint and saved_fp.get("sha256")
                and saved_fp["sha256"] != fingerprint["sha256"]
                and not args.allow_data_change):
            print("[qat] данные в чекпоинте и сейчас различаются; "
                  "используйте --allow-data-change, если это осознанно")
            return 2
        print(f"[qat] возобновление: {resume_ckpt} "
              f"({checkpoint_mod.describe_checkpoint(resume_ckpt)})")
        checkpoint_mod.restore_model(model, resume_ckpt, device)
        loaded_opt = checkpoint_mod.restore_optimizer(optimizer, resume_ckpt, device)
        checkpoint_mod.restore_scaler(scaler, state)
        checkpoint_mod.restore_rng(state, device)
        start_step = int(state["step"])
        start_samples = int(state.get("samples_seen", 0))
        last_saved_step = start_step
        print(f"[qat] шаг {start_step}, обработано последовательностей {start_samples}, "
              f"оптимизатор: {'загружен' if loaded_opt else 'сброшен'}")
    max_steps = cfg.train.max_steps
    if start_step >= max_steps:
        print(f"[qat] прогон уже завершён: шаг {start_step} >= max_steps {max_steps}")
        print(f"[qat] экспорт: python3 -m qat.export --run {out_dir} --out export/")
        return 0

    # --- поток данных -------------------------------------------------------
    try:
        if args.dry_run:
            stream: Any = RandomTokenStream(
                vocab_size=vocab_size,
                seq_len=cfg.data.max_seq_len,
                micro_batch_size=cfg.train.micro_batch_size,
                grad_accum_steps=cfg.train.grad_accum_steps,
                seed=cfg.train.seed,
            )
        else:
            dataset = PairDataset(
                rows,
                tokenizer,
                max_seq_len=cfg.data.max_seq_len,
                min_answer_tokens=cfg.data.min_answer_tokens,
                use_teacher_logits=cfg.data.use_teacher_logits,
                vocab_size=vocab_size,
            )
            stream = PairStream(
                dataset,
                micro_batch_size=cfg.train.micro_batch_size,
                grad_accum_steps=cfg.train.grad_accum_steps,
                seed=cfg.train.seed,
                pad_token_id=int(tokenizer.pad_token_id),
                start_sample=start_samples,
            )
    except DataError as exc:
        print(f"[qat] ошибка данных: {exc}")
        return 2

    # --- run.json, лог -------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    run_meta: dict[str, Any] = {
        "format": "eva32.qat.run",
        "format_version": 1,
        "created_at": (old_run or {}).get("created_at", utc_now()),
        "updated_at": utc_now(),
        "resumed": resume_ckpt is not None,
        "dry_run": bool(args.dry_run),
        "config": config_to_dict(cfg),
        "config_path": str(Path(args.config).resolve()),
        "model": {
            "name_or_path": cfg.model.name_or_path,
            "revision": cfg.model.revision,
            "vocab_size": vocab_size,
            "num_params": counts["total"],
            "ternary_params": counts["ternary"],
        },
        "quant": {
            "block_size": cfg.quant.block_size,
            "scale_dtype": cfg.quant.scale_dtype,
            "target_suffixes": list(cfg.quant.target_suffixes),
            "modules": ternary_report,
            "modules_count": len(ternary_report),
        },
        "data": fingerprint or {"path": str(data_path), "samples": 0, "dry_run": True},
        "device": str(device),
        "device_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        "optimizer": optimizer_name,
        "status": "running",
    }
    write_json_atomic(run_json_path, run_meta)
    log_path = out_dir / "log.jsonl"
    log_fh = log_path.open("a" if resume_ckpt is not None else "w", encoding="utf-8")

    def log_event(**fields: Any) -> None:
        record = {"ts": utc_now(), "step": fields.get("step"), **fields}
        log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_fh.flush()

    if args.no_save:
        print("[qat] ВНИМАНИЕ: --no-save — при обрыве прогресс будет потерян")

    # --- сигналы: оборвать аренду, не потеряв чекпоинт ----------------------
    stop = {"requested": False, "signal": None}

    def _handler(signum: int, _frame: Any) -> None:
        if stop["requested"]:
            raise KeyboardInterrupt
        stop["requested"] = True
        stop["signal"] = signum
        print(f"\n[qat] получен сигнал {signum}: завершаю после текущего шага "
              f"(чекпоинт будет сохранён)")

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass

    def save_now(step: int, *, status: str = "running") -> Path | None:
        if args.no_save:
            return None
        state_to_save = {
            "format_version": checkpoint_mod.STATE_FORMAT_VERSION,
            "step": step,
            "samples_seen": stream.samples_seen,
            "max_steps": max_steps,
            "config": config_to_dict(cfg),
            "data_fingerprint": fingerprint,
            "created_at": utc_now(),
            "dry_run": bool(args.dry_run),
            "status": status,
            "rng": _rng_state(torch, device),
        }
        started = time.perf_counter()
        path = checkpoint_mod.save_checkpoint(
            out_dir,
            step=step,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            state=state_to_save,
            save_optimizer=cfg.train.save_optimizer,
        )
        removed = checkpoint_mod.cleanup_checkpoints(out_dir, cfg.train.keep_last_checkpoints)
        took = time.perf_counter() - started
        print(f"[qat] чекпоинт: {path} "
              f"({human_bytes(checkpoint_mod.checkpoint_size(path))}, {took:.0f} с)"
              + (f"; удалено старых: {len(removed)}" if removed else ""))
        run_meta["latest_step"] = step
        run_meta["status"] = status
        run_meta["updated_at"] = utc_now()
        write_json_atomic(run_json_path, run_meta)
        return path

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"[qat] старт: шаг {start_step} -> {max_steps}, каталог {out_dir}")
    print(f"[qat] чекпоинты каждые {cfg.train.save_every} шагов, "
          f"храним последних {cfg.train.keep_last_checkpoints}")

    step = start_step
    diverged = False
    interrupted = False
    run_started = time.time()

    try:
        while step < max_steps:
            if stop["requested"]:
                interrupted = True
                break
            step_t0 = time.perf_counter()
            try:
                micro_batches = stream.next_step()
            except DataError as exc:
                print(f"[qat] ошибка данных: {exc}")
                diverged = True
                break
            data_sec = time.perf_counter() - step_t0
            # Готовим шаг заранее: нужен точный знаменатель (число предсказываемых
            # supervised-токенов) до backward. Позиция 0 не предсказывается никогда,
            # поэтому в знаменатель не входит.
            prepared: list[tuple[dict[str, Any], int, int, int]] = []
            step_tokens = 0
            for batch in micro_batches:
                labels_cpu = batch["labels"]
                keep = labels_cpu.size(1)
                logits_start = 0
                if keep_arg:
                    first = _first_supervised(labels_cpu)
                    if first is not None:
                        logits_start = max(0, first - 1)
                        keep = labels_cpu.size(1) - logits_start
                usable = int((labels_cpu[:, logits_start + 1:] != -100).sum().item())
                step_tokens += usable
                prepared.append((batch, keep, logits_start, usable))
            if step_tokens <= 0:
                print("[qat] в шаге нет ни одного supervised-токена; проверьте данные")
                diverged = True
                break

            lr = lr_at(step + 1, cfg.train, max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)

            ce_total = torch.zeros((), device=device)
            kl_total = torch.zeros((), device=device)
            loss_total = torch.zeros((), device=device)
            supervised_total = torch.zeros((), device=device, dtype=torch.long)
            compute_t0 = time.perf_counter()
            for batch, keep, logits_start, usable in prepared:
                labels = batch["labels"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                forward_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "use_cache": False,
                }
                if keep_arg and keep < input_ids.size(1):
                    forward_kwargs[keep_arg] = keep

                with _autocast(torch, device, cfg.train.amp_dtype):
                    outputs = model(**forward_kwargs)
                    logits = outputs.logits
                    if logits.size(1) != keep:
                        raise RuntimeError(
                            f"модель вернула {logits.size(1)} логитов вместо {keep}: "
                            f"аргумент {keep_arg} не поддержан"
                        )
                    ce_sum, n_sup = _chunked_ce(
                        torch, logits, labels, cfg.train.loss_chunk_size, logits_start
                    )
                    supervised_total = supervised_total + n_sup
                    if int(n_sup.item()) != usable:
                        raise RuntimeError(
                            f"предсказываемых токенов {int(n_sup.item())}, ожидалось {usable}"
                        )
                    kl_sum = None
                    if cfg.data.use_teacher_logits:
                        if not batch.get("teacher_logits"):
                            raise RuntimeError("use_teacher_logits: в батче нет logits учителя")
                        kl_sum = _kl_sum(
                            torch, logits, labels, batch["teacher_logits"],
                            logits_start, cfg.data.kl_temperature, cfg.data.kl_chunk_size,
                        )
                    loss = ce_sum / step_tokens
                    if kl_sum is not None:
                        loss = loss + cfg.data.kl_weight * kl_sum / step_tokens

                if not bool(torch.isfinite(loss).all()):
                    print(f"[qat] шаг {step + 1}: loss не конечен (loss={loss.item()}); "
                          f"обучение остановлено, сохраняю чекпоинт")
                    diverged = True
                    break
                ce_total = ce_total + ce_sum.detach()
                if kl_sum is not None:
                    kl_total = kl_total + kl_sum.detach()
                loss_total = loss_total + loss.detach()

                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if diverged:
                break
            if int(supervised_total.item()) != step_tokens:
                raise RuntimeError(
                    f"токенов в loss {int(supervised_total.item())}, а в шаге {step_tokens}"
                )

            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            if cfg.train.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.grad_clip
                )
                grad_norm_value = float(grad_norm.detach().item())
            else:
                grad_norm_value = float("nan")
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            step += 1
            compute_sec = time.perf_counter() - compute_t0
            total_sec = time.perf_counter() - step_t0
            loss_value = float(loss_total.item())
            ce_value = float((ce_total / step_tokens).item())
            kl_value = (float((kl_total / step_tokens).item())
                        if cfg.data.use_teacher_logits else None)
            tok_s = step_tokens / total_sec if total_sec > 0 else 0.0
            gpu_alloc, gpu_peak = _gpu_mem_gb(torch, device)

            if step % cfg.train.log_every == 0 or step == start_step + 1:
                fields = {
                    "step": step,
                    "loss": round(loss_value, 6),
                    "ce": round(ce_value, 6),
                    "lr": f"{lr:.3e}",
                    "tok_s": round(tok_s, 1),
                    "tokens": step_tokens,
                    "samples_seen": stream.samples_seen,
                    "epoch": stream.epoch,
                    "grad_norm": round(grad_norm_value, 4),
                    "gpu_gb": round(gpu_alloc, 2) if gpu_alloc is not None else None,
                    "gpu_peak_gb": round(gpu_peak, 2) if gpu_peak is not None else None,
                    "step_sec": round(total_sec, 2),
                    "compute_sec": round(compute_sec, 2),
                    "data_sec": round(data_sec, 3),
                }
                if kl_value is not None:
                    fields["kl"] = round(kl_value, 6)
                line = (f"[qat] шаг {step}/{max_steps}  loss={loss_value:.4f}  "
                        f"ce={ce_value:.4f}  lr={lr:.2e}  {tok_s:.0f} ток/с")
                if gpu_alloc is not None:
                    line += f"  GPU {gpu_alloc:.1f}/{gpu_peak:.1f} ГиБ"
                print(line)
                log_event(**fields)

            if step % cfg.train.save_every == 0:
                saved = save_now(step)
                if saved is not None:
                    last_saved_step = step

        if step > 0 and step != last_saved_step and not args.no_save:
            save_now(step, status=("interrupted" if interrupted else
                                   "diverged" if diverged else "finished"))
            last_saved_step = step
    finally:
        log_fh.close()

    elapsed = time.time() - run_started
    if diverged:
        print(f"[qat] остановлено на шаге {step}: проблема с данными или loss. "
              f"Чекпоинт шага {last_saved_step if last_saved_step >= 0 else 'нет'}.")
        return 2
    if interrupted:
        print(f"[qat] прервано сигналом на шаге {step}. Продолжить:")
        print(f"      python3 -m qat.train --config {args.config} --out {out_dir} --resume")
        return 0

    if not args.no_save:
        write_text_atomic(out_dir / "DONE", f"step={step}\n{utc_now()}\n")
    run_meta["status"] = "finished"
    run_meta["updated_at"] = utc_now()
    write_json_atomic(run_json_path, run_meta)
    print(f"[qat] готово: шагов {step}, обработано последовательностей {stream.samples_seen}, "
          f"время {elapsed / 60:.1f} мин")
    if not args.dry_run and not args.no_save:
        print("[qat] экспорт троичных весов:")
        print(f"      python3 -m qat.export --run {out_dir} --out export/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
