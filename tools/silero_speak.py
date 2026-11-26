#!/usr/bin/env python3.13
"""Один вызов Silero TTS: текст из stdin → WAV (48 кГц) в файл из argv.

Зачем отдельный процесс. torch тянет ~1,5–2 с на импорт и держит в памяти
~300 МБ; чат-сервер не должен платить эту цену постоянно. Поэтому
``eva/tts.py::SileroTTS`` запускает этот скрипт отдельным процессом на каждую
фразу (как Piper), а torch живёт только внутри него.

Запуск (обычно из ``eva/tts.py`` или ``tools/install_silero.sh``)::

    TORCH_HOME=vendor/silero PYTHONPATH=vendor/torch-py \\
    SILERO_SPEAKER=baya python3.13 tools/silero_speak.py out.wav <<'TXT'
    Привет, это проверка голоса.
    TXT

Модель: ``torch.hub.load('snakers4/silero-models', 'silero_tts',
language='ru', speaker='v4_ru', trust_repo=True)``. Первый запуск качает
репозиторий и веса в ``$TORCH_HOME/hub`` (по умолчанию ``vendor/silero``);
дальше работает офлайн. Сеть в самом синтезе не нужна.

На stdout печатается одна JSON-строка с замерами (load_ms/synth_ms/
duration_s/bytes) — установщик показывает её пользователю, а ``eva/tts.py``
игнорирует. Диагностика torch идёт в stderr.

Голоса v4_ru: ``aidar`` (мужской), ``baya`` (женский, по умолчанию),
``kseniya`` (женский), ``xenia`` (женский), ``eugene`` (мужской),
``random``. Пресеты Piper (``irina``/``soft``/``young``) здесь не
применяются: ``eva/tts.py`` их отбрасывает.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import NoReturn

#: Корень репозитория: tools/silero_speak.py -> tools -> корень.
ROOT = Path(__file__).resolve().parents[1]

#: Частота дискретизации Silero v4_ru. 48 кГц — лучший вариант модели и
#: его играют и Web Audio, и AVPlayer без пересэмплинга.
DEFAULT_SAMPLE_RATE = 48000

#: Голос по умолчанию: EVA — женский ассистент. Меняется
#: ``SILERO_SPEAKER`` или ``--speaker`` (это голос в ``apply_tts``, а не
#: вариант модели).
DEFAULT_SPEAKER = "baya"

#: Вариант модели в ``torch.hub.load``: ``speaker='v4_ru'`` — именно id
#: модели, а не имя голоса. Голоса (baya/kseniya/…) передаются позже в
#: ``model.apply_tts``.
MODEL_ID = "v4_ru"

#: Веса и репозиторий torch.hub кладутся сюда (``$TORCH_HOME/hub``), чтобы
#: не занимать домашний кэш и оставаться внутри vendor/.
DEFAULT_TORCH_HOME = ROOT / "vendor" / "silero"


def _fail(message: str, code: int = 2) -> NoReturn:
    print(f"silero_speak: {message}", file=sys.stderr)
    raise SystemExit(code)


def _check_torch() -> "object":
    """Импортировать torch из PYTHONPATH и вернуть модуль.

    Никаких подстановок путей внутри скрипта: если ``vendor/torch-py`` нет в
    ``PYTHONPATH``, лучше понятная ошибка, чем случайный чужой torch из
    ``.pylibs`` (там лежит сломанный source-checkout).
    """
    try:
        import torch  # noqa: PLC0415 — импорт дорогой, только внутри процесса
    except Exception as exc:  # noqa: BLE001 — сообщение важнее типа
        _fail(
            "не удалось импортировать torch "
            f"({type(exc).__name__}: {exc}). Проверьте PYTHONPATH="
            f"{ROOT / 'vendor' / 'torch-py'} и запустите "
            "sh tools/install_silero.sh"
        )
    return torch


def _load_model(torch):
    """Загрузить модель Silero v4_ru через torch.hub (кэш в TORCH_HOME)."""
    try:
        model, _ = torch.hub.load(
            "snakers4/silero-models",
            "silero_tts",
            language="ru",
            speaker=MODEL_ID,
            trust_repo=True,
        )
    except Exception as exc:  # noqa: BLE001 — сеть/кэш/версия модели
        _fail(
            f"не удалось загрузить модель Silero v4_ru ({type(exc).__name__}: "
            f"{exc}). Нужна сеть при первой установке; кэш — "
            f"{os.environ.get('TORCH_HOME', DEFAULT_TORCH_HOME)}/hub"
        )
    return model


def _write_wav(path: Path, audio, sample_rate: int) -> int:
    """Записать тензор float [-1, 1] как PCM16 mono WAV; вернуть число байт."""
    torch = sys.modules["torch"]
    samples = (
        audio.detach()
        .to("cpu")
        .to(torch.float32)
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .round()
        .to(torch.int16)
    )
    pcm = samples.numpy().tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return len(pcm)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Silero TTS: текст из stdin → WAV 48 кГц в файл."
    )
    parser.add_argument("out_path", help="куда писать WAV")
    parser.add_argument(
        "--speaker",
        default=os.environ.get("SILERO_SPEAKER", DEFAULT_SPEAKER),
        help=f"голос v4_ru (по умолчанию {DEFAULT_SPEAKER})",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"частота, Гц (по умолчанию {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="текст (по умолчанию читается из stdin)",
    )
    args = parser.parse_args(argv)

    text = (args.text if args.text is not None else sys.stdin.read()).strip()
    if not text:
        _fail("пустой текст для синтеза")

    out_path = Path(args.out_path).expanduser()
    speaker = (args.speaker or DEFAULT_SPEAKER).strip() or DEFAULT_SPEAKER

    # Модель и torch.hub живут под TORCH_HOME; по умолчанию — vendor/silero.
    os.environ.setdefault("TORCH_HOME", str(DEFAULT_TORCH_HOME))
    threads = os.environ.get("SILERO_THREADS", "4").strip() or "4"
    try:
        thread_count = max(1, int(threads))
    except ValueError:
        thread_count = 4

    torch = _check_torch()
    try:
        torch.set_num_threads(thread_count)
    except Exception:  # noqa: BLE001 — не критично для синтеза
        pass

    started = time.perf_counter()
    model = _load_model(torch)
    load_ms = (time.perf_counter() - started) * 1000.0
    try:
        model.to(torch.device("cpu"))
    except Exception as exc:  # noqa: BLE001
        _fail(f"модель не переехала на CPU: {exc}")

    synth_started = time.perf_counter()
    try:
        audio = model.apply_tts(
            text=text,
            speaker=speaker,
            sample_rate=int(args.sample_rate),
        )
    except Exception as exc:  # noqa: BLE001 — неизвестный голос/пустой текст
        known = getattr(model, "speakers", None)
        hint = f" Доступные голоса: {', '.join(known)}." if known else ""
        _fail(
            f"синтез не удался ({type(exc).__name__}: {exc}).{hint}"
        )
    synth_ms = (time.perf_counter() - synth_started) * 1000.0
    total_ms = (time.perf_counter() - started) * 1000.0

    pcm_bytes = _write_wav(out_path, audio, int(args.sample_rate))
    duration_s = float(audio.numel()) / float(args.sample_rate)
    summary = {
        "engine": "silero",
        "speaker": speaker,
        "sample_rate": int(args.sample_rate),
        "text": text,
        "duration_s": round(duration_s, 3),
        "pcm_bytes": pcm_bytes,
        "load_ms": round(load_ms, 1),
        "synth_ms": round(synth_ms, 1),
        "total_ms": round(total_ms, 1),
        "torch": getattr(torch, "__version__", "?"),
        "out": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
