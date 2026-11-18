#!/usr/bin/env python3
"""Экспорт QAT-чекпоинта в тернарный артефакт.

    python3 -m qat.export --run runs/pilot1 --out export/

Пишет три файла:

* ``ternary.pt`` — int8-триты и масштабы по тензорам (основной артефакт);
* ``ternary.trit`` — те же веса, упакованные по 20 тритов в 32-битное слово
  (формат docs/trit-format.md), сегментами подряд;
* ``manifest.json`` — ревизия базовой модели, конфиг, шаг, контрольные суммы
  и таблица смещений тензоров в ``.trit``.

Экспорт не требует transformers: он читает state_dict чекпоинта и квантует
мастер-веса заново тем же кодом, что и обучение (qat.ternary).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from . import ternary
from .config import PROJECT_ROOT
from .util import human_bytes, human_count, sha256_file, utc_now, write_json_atomic

PARAM_SUFFIX = ".parametrizations.weight.original"
REQUIRED_MODULES = (("torch", "torch"), ("numpy", "numpy"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m qat.export",
        description="Экспорт тернарных весов из QAT-чекпоинта в .pt/.trit + manifest.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  python3 -m qat.export --run runs/pilot1 --out export/\n"
            "  python3 -m qat.export --run runs/pilot1 --step 200 --out export/\n"
            "  python3 -m qat.export --run runs/pilot1/checkpoints/step_000200 --out export/\n\n"
            "тернарный формат описан в docs/trit-format.md; контрольные векторы\n"
            "проверяются тестами: python3 -m unittest discover -s tests -v\n"
        ),
    )
    parser.add_argument("--run", required=True,
                        help="каталог прогона runs/<name> или конкретный checkpoints/step_*")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "export"),
                        help="каталог для ternary.pt, ternary.trit и manifest.json")
    parser.add_argument("--step", type=int, default=None,
                        help="номер шага; по умолчанию последний чекпоинт")
    parser.add_argument("--scale-dtype", choices=("float16", "float32"), default=None,
                        help="тип масштабов (по умолчанию из конфига прогона)")
    parser.add_argument("--force", action="store_true",
                        help="перезаписать существующий экспорт")
    return parser


def missing_modules() -> list[str]:
    missing: list[str] = []
    for module, package in REQUIRED_MODULES:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(package)
    return missing


def print_missing(missing: Sequence[str]) -> None:
    print("[qat.export] не хватает зависимостей: " + ", ".join(missing))
    print()
    print("Экспорт запускается на GPU-машине (нужен только torch/numpy):")
    print("    python3 -m pip install -r qat/requirements-qat.txt")
    print()
    print("Без GPU-стека проверяется только чистый кодек:")
    print("    python3 -m unittest discover -s tests -v")


def _resolve_source(
    source: Path,
    step: int | None,
    checkpoint_mod: Any,
) -> tuple[Path, Path]:
    """Вернуть (каталог прогона, каталог чекпоинта)."""
    source = source.expanduser().resolve()
    if (source / "model.pt").exists():
        ckpt = source
        run_dir = source.parent.parent
        if step is not None and checkpoint_mod.checkpoint_step(source) != step:
            raise RuntimeError(
                f"--step {step}, а указан чекпоинт шага "
                f"{checkpoint_mod.checkpoint_step(source)}"
            )
        return run_dir, ckpt
    if (source / "checkpoints").exists():
        if step is not None:
            ckpt = source / "checkpoints" / f"step_{step:06d}"
            if not ckpt.exists():
                raise RuntimeError(f"чекпоинта шага {step} нет: {ckpt}")
            return source, ckpt
        ckpt = checkpoint_mod.latest_checkpoint(source)
        if ckpt is None:
            raise RuntimeError(f"в {source}/checkpoints нет чекпоинтов")
        return source, ckpt
    raise RuntimeError(
        f"{source} не похож на прогон QAT: нет model.pt или checkpoints/"
    )


def _read_run_meta(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _to_numpy_bytes(tensor: Any, dtype: str) -> bytes:
    """Тензор на CPU -> little-endian байты без зависимости от порядка байт хоста."""
    array = tensor.detach().cpu().numpy()
    return array.astype(dtype, copy=False).tobytes(order="C")


def _check_packing(trits: Any, words: Any, count: int = 160) -> None:
    """Сверить первые слова с чистым Python-кодеком (он же проверен тестами)."""
    flat = trits.reshape(-1)
    take = min(count, int(flat.numel()))
    if take == 0:
        return
    n_words = (take + ternary.TRITS_PER_WORD - 1) // ternary.TRITS_PER_WORD
    blob = _to_numpy_bytes(words[:n_words], "<u4")
    decoded = ternary.unpack20(blob, take)
    expected = [int(x) for x in flat[:take].tolist()]
    if decoded != expected:
        raise RuntimeError("упаковка 20 тритов в слово не сошлась с кодеком")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    missing = missing_modules()
    if missing:
        print_missing(missing)
        return 2

    import numpy  # noqa: F401,PLC0415 - нужен для tensor.numpy(), проверен выше
    import torch  # noqa: PLC0415

    from . import checkpoint as checkpoint_mod  # noqa: PLC0415

    try:
        run_dir, ckpt = _resolve_source(Path(args.run), args.step, checkpoint_mod)
    except RuntimeError as exc:
        print(f"[qat.export] {exc}")
        return 2

    state = checkpoint_mod.load_trainer_state(ckpt)
    run_meta = _read_run_meta(run_dir)
    config = state.get("config") or run_meta.get("config") or {}
    quant_cfg = config.get("quant") or {}
    model_cfg = config.get("model") or {}
    block_size = int(quant_cfg.get("block_size", ternary.BLOCK_SIZE))
    scale_dtype = args.scale_dtype or quant_cfg.get("scale_dtype", "float16")
    if scale_dtype not in ("float16", "float32"):
        print(f"[qat.export] неизвестный --scale-dtype: {scale_dtype}")
        return 2
    step = int(state.get("step", checkpoint_mod.checkpoint_step(ckpt)))
    samples_seen = int(state.get("samples_seen", 0))

    out_dir = Path(args.out).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pt_path = out_dir / "ternary.pt"
    trit_path = out_dir / "ternary.trit"
    manifest_path = out_dir / "manifest.json"
    if not args.force and (pt_path.exists() or trit_path.exists() or manifest_path.exists()):
        print(f"[qat.export] в {out_dir} уже есть экспорт; добавьте --force, чтобы перезаписать")
        return 2

    print(f"[qat.export] чекпоинт: {ckpt}")
    print(f"[qat.export] шаг {step}, обработано последовательностей {samples_seen}")
    print(f"[qat.export] квантование: блок {block_size}, масштабы {scale_dtype}")
    if model_cfg.get("revision") == "main":
        print("[qat.export] ВНИМАНИЕ: в конфиге revision=main; в manifest попадёт "
              "незакреплённая ревизия")

    started = time.time()
    state_dict = checkpoint_mod.load_model_state(ckpt)
    target_keys = sorted(k for k in state_dict if k.endswith(PARAM_SUFFIX))
    if not target_keys:
        print("[qat.export] в чекпоинте нет параметризованных весов — это не QAT-модель?")
        return 2

    expected_modules = {
        item.get("name") for item in (run_meta.get("quant") or {}).get("modules", [])
        if item.get("name")
    }
    found_modules = {key[:-len(PARAM_SUFFIX)] for key in target_keys}
    if expected_modules and expected_modules != found_modules:
        missing_mods = sorted(expected_modules - found_modules)
        extra_mods = sorted(found_modules - expected_modules)
        print("[qat.export] ВНИМАНИЕ: набор тернарных слоёв в чекпоинте и run.json различается")
        if missing_mods:
            print(f"               нет в чекпоинте: {missing_mods[:5]}")
        if extra_mods:
            print(f"               лишние в чекпоинте: {extra_mods[:5]}")

    # Считаем нетернарные параметры для manifest: они остаются в чекпоинте.
    # data_ptr дедуплицирует tied-веса (lm_head и embed_tokens — один тензор).
    non_ternary_params = 0
    non_ternary_names: list[str] = []
    seen_ptrs: set[int] = set()
    for key, value in state_dict.items():
        if key.endswith(PARAM_SUFFIX) or ".parametrizations." in key:
            continue
        if hasattr(value, "numel"):
            ptr = int(value.data_ptr())
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            non_ternary_params += int(value.numel())
            non_ternary_names.append(key)

    tensors: dict[str, Any] = {}
    manifest_tensors: list[dict[str, Any]] = []
    total_params = 0
    flags = ternary.FLAG_SCALE_F32 if scale_dtype == "float32" else 0
    scale_np_dtype = "<f4" if scale_dtype == "float32" else "<f2"
    torch_scale_dtype = torch.float32 if scale_dtype == "float32" else torch.float16

    tmp_trit = out_dir / ".ternary.trit.tmp"
    with tmp_trit.open("wb") as fh:
        for index, key in enumerate(target_keys, 1):
            name = key[:-len(PARAM_SUFFIX)]
            weight = state_dict.pop(key)
            if weight.ndim < 1:
                raise RuntimeError(f"{name}: неожиданная форма {tuple(weight.shape)}")
            quant = ternary.quantize_torch(weight, block_size)
            trits = quant["trits"].cpu().contiguous()
            scales = quant["scales"].reshape(-1).cpu()
            n_trits = int(trits.numel())
            in_features = int(weight.shape[-1])
            if in_features % block_size != 0:
                raise RuntimeError(
                    f"{name}: входная размерность {in_features} не кратна block_size "
                    f"{block_size}; формат .trit хранит один масштаб на 128 подряд идущих "
                    f"весов и не умеет row-padding. Смените block_size или модель."
                )
            rows = n_trits // in_features
            scale_blocks_per_row = int(quant["n_scale_blocks"])
            total_scale_blocks = int(scales.numel())
            if total_scale_blocks != rows * scale_blocks_per_row:
                raise RuntimeError(
                    f"{name}: масштабов {total_scale_blocks}, ожидалось "
                    f"{rows * scale_blocks_per_row} (строк {rows} × блоков {scale_blocks_per_row})"
                )
            if total_scale_blocks * block_size < n_trits:
                raise RuntimeError(f"{name}: масштабов не хватает на все триты")

            scales_out = scales.to(torch_scale_dtype)
            if not bool(torch.isfinite(scales_out).all()):
                raise RuntimeError(f"{name}: масштаб не помещается в {scale_dtype}")
            scales_bytes = _to_numpy_bytes(scales_out, scale_np_dtype)

            words = ternary.pack20_torch(trits)
            storage_blocks = (n_trits + ternary.TRITS_PER_STORAGE_BLOCK - 1) // ternary.TRITS_PER_STORAGE_BLOCK
            storage_words = storage_blocks * ternary.WORDS_PER_STORAGE_BLOCK
            if int(words.numel()) < storage_words:
                words = torch.cat([
                    words,
                    torch.zeros(storage_words - int(words.numel()), dtype=torch.int64),
                ])
            if int(words.numel()) != storage_words:
                raise RuntimeError(f"{name}: слов {int(words.numel())}, ожидалось {storage_words}")
            if int(words.max().item()) > ternary.MAX_WORD:
                raise RuntimeError(f"{name}: слово вне диапазона 3^20")
            _check_packing(trits, words)
            words_bytes = _to_numpy_bytes(words, "<u4")

            header = struct.pack(
                "<4sBBQI", ternary.MAGIC, ternary.VERSION, flags, n_trits, total_scale_blocks,
            )
            info = ternary.segment_info(header, block_size)
            if info.n_scale_blocks != total_scale_blocks or info.n_words != storage_words:
                raise RuntimeError(f"{name}: заголовок сегмента не сошёлся с раскладкой")
            offset = fh.tell()
            fh.write(header)
            fh.write(scales_bytes)
            fh.write(words_bytes)
            length = fh.tell() - offset
            expected_length = len(header) + len(scales_bytes) + len(words_bytes)
            if length != expected_length:
                raise RuntimeError(f"{name}: длина сегмента {length}, ожидалась {expected_length}")

            shape = [int(x) for x in weight.shape]
            tensors[name] = {
                "trits": trits,
                "scales": scales_out,
                "shape": shape,
                "n_scale_blocks": total_scale_blocks,
                "scale_blocks_per_row": scale_blocks_per_row,
            }
            total_params += int(trits.numel())
            manifest_tensors.append({
                "name": name,
                "shape": shape,
                "params": n_trits,
                "n_trits": n_trits,
                "n_scale_blocks": total_scale_blocks,
                "scale_blocks_per_row": scale_blocks_per_row,
                "n_words": (n_trits + ternary.TRITS_PER_WORD - 1) // ternary.TRITS_PER_WORD,
                "n_storage_blocks": storage_blocks,
                "n_storage_words": storage_words,
                "trit_offset": int(offset),
                "trit_length": int(length),
            })
            if index % 25 == 0 or index == len(target_keys):
                print(f"[qat.export] тензоров: {index}/{len(target_keys)}, "
                      f"параметров: {human_count(total_params)}")

    tmp_trit.replace(trit_path)
    del state_dict  # освобождаем нетернарные веса до torch.save

    payload = {
        "format": "eva32.ternary.pt",
        "format_version": 1,
        "created_at": utc_now(),
        "block_size": block_size,
        "scale_dtype": scale_dtype,
        "packing": {
            "scheme": "trit20",
            "trits_per_word": ternary.TRITS_PER_WORD,
            "words_per_storage_block": ternary.WORDS_PER_STORAGE_BLOCK,
            "storage_block_trits": ternary.TRITS_PER_STORAGE_BLOCK,
            "scale_block": block_size,
            "endianness": "little",
            "reference": "docs/trit-format.md",
        },
        "source": {
            "run": str(run_dir),
            "checkpoint": str(ckpt),
            "step": step,
            "samples_seen": samples_seen,
            "model": model_cfg.get("name_or_path"),
            "revision": model_cfg.get("revision"),
        },
        "tensors": tensors,
    }
    tmp_pt = out_dir / ".ternary.pt.tmp"
    torch.save(payload, tmp_pt)
    tmp_pt.replace(pt_path)
    del payload, tensors

    pt_hash = sha256_file(pt_path)
    trit_hash = sha256_file(trit_path)
    manifest = {
        "format": "eva32.ternary.manifest",
        "format_version": 1,
        "created_at": utc_now(),
        "training": {
            "run": run_dir.name,
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt),
            "step": step,
            "samples_seen": samples_seen,
            "max_steps": (config.get("train") or {}).get("max_steps"),
            "config": config,
        },
        "base_model": {
            "name_or_path": model_cfg.get("name_or_path"),
            "revision": model_cfg.get("revision"),
        },
        "quantization": {
            "block_size": block_size,
            "scale_dtype": scale_dtype,
            "rule": "scale=absmax(block); trit=clamp(round_half_away(w/scale),-1,1); w=trit*scale",
            "ste": "w + (w_tern - w).detach()",
            "target_suffixes": quant_cfg.get("target_suffixes"),
        },
        "packing": {
            "magic": ternary.MAGIC.decode("ascii"),
            "version": ternary.VERSION,
            "trits_per_word": ternary.TRITS_PER_WORD,
            "words_per_storage_block": ternary.WORDS_PER_STORAGE_BLOCK,
            "storage_block_trits": ternary.TRITS_PER_STORAGE_BLOCK,
            "scale_block": block_size,
            "n_blocks_header": "число групп масштаба (scale_block), а не 160-тритовых блоков",
            "words_padding": "хвост тензора добит нулевыми тритами до 8 слов",
            "endianness": "little",
            "note": (
                "docs/trit-format.md §8 оставлял размер группы масштаба открытым; "
                "здесь он равен block_size из конфига QAT (128). Слова упакованы "
                "по 20 тритов, массив слов выровнен на 32-байтовые блоки по 8 слов. "
                "Сегменты .trit идут подряд в порядке tensors[]."
            ),
        },
        "non_ternary": {
            "stored_in_this_export": False,
            "count": len(non_ternary_names),
            "params": non_ternary_params,
            "source": str(ckpt / "model.pt"),
            "note": (
                "эмбеддинги, lm_head, layernorm, bias и прочее в ternary.pt не входят: "
                "их берут из этого же чекпоинта (или из базовой модели, если слой не обучался)"
            ),
        },
        "files": {
            "ternary.pt": {"bytes": pt_path.stat().st_size, "sha256": pt_hash},
            "ternary.trit": {"bytes": trit_path.stat().st_size, "sha256": trit_hash},
        },
        "totals": {
            "tensors": len(manifest_tensors),
            "params": total_params,
            "bytes_per_param_bits": 32.0 / ternary.TRITS_PER_WORD,
        },
        "tensors": manifest_tensors,
    }
    write_json_atomic(manifest_path, manifest)

    elapsed = time.time() - started
    print(f"[qat.export] тензоров: {len(manifest_tensors)}, "
          f"тернарных параметров: {human_count(total_params)}")
    print(f"[qat.export] {pt_path.name}: {human_bytes(pt_path.stat().st_size)}  "
          f"sha256 {pt_hash[:16]}…")
    print(f"[qat.export] {trit_path.name}: {human_bytes(trit_path.stat().st_size)}  "
          f"sha256 {trit_hash[:16]}…")
    print(f"[qat.export] {manifest_path.name}: {len(manifest_tensors)} записей")
    print(f"[qat.export] готово за {elapsed:.0f} с -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
