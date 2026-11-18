"""Чекпоинты QAT: сохранение, загрузка, поиск последнего, ротация.

Каждый чекпоинт — каталог ``checkpoints/step_000100``:

* ``model.pt`` — state_dict параметризованной модели (мастер-веса fp32);
* ``optimizer.pt`` — состояние оптимизатора (опционально);
* ``trainer_state.pt`` — шаг, samples_seen, конфиг, RNG, отпечаток данных.

Запись идёт в ``.tmp`` и затем переименовывается: оборванный на середине
чекпоинт не должен выглядеть как валидный. Указатель ``latest.txt``
перезаписывается атомарно последним.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch

from .util import human_bytes, write_text_atomic

STATE_FORMAT_VERSION = 1


def _load(path: Path, *, map_location: Any = "cpu", mmap: bool = False) -> Any:
    """torch.load с совместимостью между версиями и без weights_only-сюрпризов.

    Чекпоинты пишем мы сами, поэтому weights_only=False здесь допустим;
    для входящих файлов извне так делать нельзя.
    """
    kwargs: dict[str, Any] = {"map_location": map_location, "weights_only": False}
    if mmap:
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)
    except RuntimeError:
        if "mmap" not in kwargs:
            raise
        kwargs.pop("mmap")
        return torch.load(path, **kwargs)


def checkpoint_dirs(run_dir: str | Path) -> list[Path]:
    """Все целые чекпоинты run-каталога, по возрастанию шага."""
    root = Path(run_dir) / "checkpoints"
    if not root.exists():
        return []
    out: list[tuple[int, Path]] = []
    for path in root.glob("step_*"):
        if not path.is_dir() or not path.name[5:].isdigit():
            continue
        out.append((int(path.name[5:]), path))
    return [path for _, path in sorted(out)]


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    dirs = checkpoint_dirs(run_dir)
    return dirs[-1] if dirs else None


def checkpoint_step(path: str | Path) -> int:
    name = Path(path).name
    if not name.startswith("step_") or not name[5:].isdigit():
        raise ValueError(f"не похоже на каталог чекпоинта: {path}")
    return int(name[5:])


def _fsync_tree(directory: Path) -> None:
    """fsync всех файлов каталога и самого каталога (ADR 0008).

    rename без fsync переносит только имя: содержимое может остаться в кэше ОС,
    и после потери питания чекпоинт окажется пустым.
    """
    from .util import fsync_dir  # noqa: PLC0415

    for entry in sorted(directory.rglob("*")):
        if not entry.is_file():
            continue
        try:
            fd = os.open(str(entry), os.O_RDONLY)
        except OSError:
            continue
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)
    fsync_dir(directory)


def save_checkpoint(
    run_dir: str | Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    state: dict[str, Any],
    save_optimizer: bool = True,
) -> Path:
    """Сохранить чекпоинт атомарно и обновить latest.txt. Вернуть каталог."""
    run_dir = Path(run_dir)
    root = run_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step_{step:06d}"
    tmp = root / f".step_{step:06d}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    try:
        torch.save(model.state_dict(), tmp / "model.pt")
        if save_optimizer and optimizer is not None:
            torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            state = dict(state)
            state["grad_scaler"] = scaler.state_dict()
        torch.save(state, tmp / "trainer_state.pt")
        _fsync_tree(tmp)
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    write_text_atomic(run_dir / "latest.txt", final.name + "\n")
    return final


def load_trainer_state(ckpt_dir: str | Path) -> dict[str, Any]:
    """Шаг, samples_seen, конфиг, RNG — всё, кроме весов."""
    path = Path(ckpt_dir) / "trainer_state.pt"
    if not path.exists():
        raise FileNotFoundError(f"в {ckpt_dir} нет trainer_state.pt")
    state = _load(path)
    if not isinstance(state, dict) or "step" not in state:
        raise RuntimeError(f"{path}: повреждён trainer_state.pt")
    return state


def load_model_state(ckpt_dir: str | Path) -> dict[str, Any]:
    """Прочитать model.pt (state_dict) с CPU-маппингом и mmap, если можно."""
    path = Path(ckpt_dir) / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"в {ckpt_dir} нет model.pt")
    state_dict = _load(path, map_location="cpu", mmap=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"{path}: ожидался state_dict")
    return state_dict


def restore_model(model: torch.nn.Module, ckpt_dir: str | Path, device: torch.device) -> None:
    """Загрузить мастер-веса. mkdir/mmап не нужны: сначала CPU, потом на устройство."""
    state_dict = load_model_state(ckpt_dir)
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.to(device)


def restore_optimizer(
    optimizer: torch.optim.Optimizer,
    ckpt_dir: str | Path,
    device: torch.device,
) -> bool:
    """Загрузить состояние оптимизатора; False — если оно не совместимо.

    У 8-bit Adam из bitsandbytes состояние хранится в собственном формате.
    Если версия библиотеки не умеет его читать, обучение продолжается с
    заново накопленными моментами — веса и шаг при этом не теряются.
    """
    path = Path(ckpt_dir) / "optimizer.pt"
    if not path.exists():
        return False
    try:
        state = _load(path, map_location=device)
        optimizer.load_state_dict(state)
        return True
    except Exception as exc:  # noqa: BLE001 - разные версии bnb/torch падают по-разному
        print(f"[qat] ВНИМАНИЕ: состояние оптимизатора не загружено ({exc}); "
              f"моменты Adam начнутся заново, веса и шаг сохранены")
        return False


def restore_scaler(scaler: Any, state: dict[str, Any]) -> None:
    if scaler is not None and state.get("grad_scaler") is not None:
        try:
            scaler.load_state_dict(state["grad_scaler"])
        except (KeyError, RuntimeError) as exc:
            print(f"[qat] ВНИМАНИЕ: состояние GradScaler не загружено ({exc})")


def restore_rng(state: dict[str, Any], device: torch.device) -> None:
    """Восстановить RNG: без этого после resume меняются dropout/порядок, если появятся."""
    import random

    rng = state.get("rng") or {}
    if "python" in rng:
        random.setstate(rng["python"])
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if device.type == "cuda" and "cuda" in rng:
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except (RuntimeError, TypeError) as exc:
            print(f"[qat] ВНИМАНИЕ: CUDA RNG не восстановлен ({exc})")


def cleanup_checkpoints(run_dir: str | Path, keep_last: int) -> list[Path]:
    """Оставить последние N чекпоинтов, остальные удалить. Вернуть удалённые."""
    if keep_last < 1:
        return []
    dirs = checkpoint_dirs(run_dir)
    removed: list[Path] = []
    for path in dirs[:-keep_last]:
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed


def checkpoint_size(ckpt_dir: str | Path) -> int:
    return sum(p.stat().st_size for p in Path(ckpt_dir).glob("*") if p.is_file())


def describe_checkpoint(ckpt_dir: str | Path) -> str:
    return f"{Path(ckpt_dir).name} ({human_bytes(checkpoint_size(ckpt_dir))})"
