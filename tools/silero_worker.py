#!/usr/bin/env python3.13
"""Долгоживущий Silero TTS: модель грузится один раз, фразы — по протоколу.

Задача [`0195`](../ai/tasks/0195-silero-worker.md). ``tools/silero_speak.py``
(задача 0193) синтезирует одну фразу и завершается, поэтому каждая новая
фраза платит холодный старт: импорт torch + загрузка модели ≈1,5–2,5 с. Здесь
модель загружается **один раз**, после чего процесс читает запросы со stdin
до EOF/``SIGTERM``. ``eva/tts.py::SileroTTS`` держит ровно один такой процесс
на процесс чата (ленивый старт при первой фразе, keep-alive, ``atexit``).

Протокол — JSON-строка на строку, UTF-8::

    → {"text": "Привет", "speaker": "baya", "out": "/tmp/reply.wav"}
    ← {"ok": true, "ms": 421.3, "duration_s": 1.11, "pcm_bytes": 106560, ...}
    → {"text": "Привет"}                 # без "out" — WAV в base64
    ← {"ok": true, "ms": 421.3, "wav_b64": "UklGR..."}
    → {"cmd": "ping"}
    ← {"ok": true, "pong": true, "pid": 4242, "uptime_ms": 812.4}
    → {"cmd": "shutdown"}
    ← {"ok": true, "bye": true}

Ошибка синтеза не убивает worker: ответ ``{"ok": false, "error": "..."}``, а
следующая строка обрабатывается как обычно (важно: битый голос/пустой текст —
это ошибка запроса, а не повод перезапускать процесс). При старте, когда
модель загружена, печатается ``{"ok": true, "ready": true, ...}`` — клиент по
ней понимает, что worker готов, и меряет честный cold start.

Почему stdin/stdout, а не UNIX-сокет в ``/tmp``: пайп живёт ровно столько,
сколько жив родитель, не оставляет мусорных файлов, не требует прав и не
конфликтует между несколькими чат-серверами. Цена — один процесс-родитель,
который сам управляет жизненным циклом (``eva/tts.py``), зато никакой
уборки сокетов и никакого риска подключиться «не к тому» worker'у.

Запуск (обычно из ``eva/tts.py``; руками — для отладки)::

    TORCH_HOME=vendor/silero PYTHONPATH=vendor/torch-py \\
        python3.13 tools/silero_worker.py --speaker baya --sample-rate 48000
    {"text": "Проверка голоса", "out": "/tmp/check.wav"}
    {"cmd": "shutdown"}

Окружение: ``PYTHONPATH=vendor/torch-py`` (чужой torch из site-packages и
``.pylibs`` не подмешивается), ``TORCH_HOME=vendor/silero`` (кэш модели),
``SILERO_THREADS`` (потоки torch, по умолчанию 4; ``OMP_NUM_THREADS`` тоже).
Сеть нужна только установщику; сам синтез офлайн.

Код загрузки модели и записи WAV берётся из :mod:`silero_speak`, чтобы
одноразовый и долгоживущий варианты не разъезжались (модель, частота, формат
WAV, голоса — одни и те же).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, NoReturn, TextIO

#: Корень репозитория: tools/silero_worker.py -> tools -> корень.
ROOT = Path(__file__).resolve().parents[1]

try:  # прямой запуск: sys.path[0] == tools/, import silero_speak
    from silero_speak import (  # noqa: PLC0415 — модуль рядом, без побочек
        DEFAULT_SAMPLE_RATE,
        DEFAULT_SPEAKER,
        DEFAULT_TORCH_HOME,
        _check_torch,
        _load_model,
        _write_wav,
    )
except ImportError:  # импорт как пакета: from tools import silero_worker (тесты)
    from tools.silero_speak import (  # type: ignore[no-redef]
        DEFAULT_SAMPLE_RATE,
        DEFAULT_SPEAKER,
        DEFAULT_TORCH_HOME,
        _check_torch,
        _load_model,
        _write_wav,
    )

#: Версия протокола. Клиент проверяет её только для диагностики: несовпадение
#: не ломает работу, пока живы поля ``ok``/``ready``.
PROTOCOL_VERSION = 1


# --------------------------------------------------------------------------- #
# ввод-вывод протокола
# --------------------------------------------------------------------------- #
def _emit(stream: TextIO, payload: dict[str, Any]) -> None:
    """Отправить один JSON-объект одной строкой и сразу сбросить буфер.

    Без ``flush`` клиент ждал бы ответа до заполнения буфера пайпа: для
    диалога это эквивалентно зависанию.
    """
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()


def _safe_emit(stream: TextIO, payload: dict[str, Any]) -> bool:
    """``_emit`` без падения на закрытом пайпе (родитель умер). ``False`` — писать некуда."""
    try:
        _emit(stream, payload)
    except (BrokenPipeError, OSError, ValueError):
        return False
    return True


def _error(stream: TextIO, message: str, **extra: Any) -> bool:
    return _safe_emit(stream, {"ok": False, "error": message, **extra})


def _proc_stats() -> dict[str, float]:
    """RSS (пик, МБ) и CPU-время процесса (с) — для ``ping``/``ready``.

    Внешние ``ps``/``top`` не нужны: цифры для отчёта и диагностики worker
    отдаёт сам. ``ru_maxrss`` — пик RSS: на macOS в байтах, на Linux в
    килобайтах (документированное различие ``resource``).
    """
    rss_mb = 0.0
    try:
        import resource

        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rss_mb = peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0
    except (ImportError, OSError, ValueError):  # pragma: no cover — экзотика
        pass
    return {
        "rss_mb": round(rss_mb, 1),
        "cpu_s": round(time.process_time(), 3),
    }


def _on_signal(signum: int, _frame: Any) -> NoReturn:
    """SIGTERM/SIGINT — выйти штатно (код 0), как по EOF.

    ``KeyboardInterrupt`` здесь не нужен: клиент отличает «убили» от «упал»
    по коду возврата, и 0 читается как честное завершение.
    """
    print(f"silero_worker: сигнал {signum}, завершаюсь", file=sys.stderr)
    raise SystemExit(0)


# --------------------------------------------------------------------------- #
# модель
# --------------------------------------------------------------------------- #
class SileroBackend:
    """Модель Silero, загруженная один раз: текст → WAV 48 кГц.

    Интерфейс минимальный (``synthesize`` + ``speakers`` + ``torch_version``),
    чтобы тесты подставляли заглушку без torch: протокол проверяется отдельно
    от нейросети.
    """

    def __init__(
        self,
        torch,
        model,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ) -> None:
        self.torch = torch
        self.model = model
        self.sample_rate = int(sample_rate)
        speakers = getattr(model, "speakers", None)
        self.speakers: tuple[str, ...] = tuple(speakers) if speakers else ()
        self.jit_profiling = "?"
        #: Замеры старта (мс): импорт torch и загрузка модели по отдельности.
        self.import_ms = 0.0
        self.model_ms = 0.0

    @property
    def torch_version(self) -> str:
        return str(getattr(self.torch, "__version__", "?"))

    def synthesize(
        self,
        text: str,
        speaker: str,
        out_path: Path | None,
        *,
        sample_rate: int | None = None,
    ) -> dict[str, Any]:
        """Синтезировать фразу; вернуть замеры и признак готового WAV.

        Исключения не глотаются: ``serve`` превращает их в ``ok=false`` и
        продолжает слушать (одна кривая фраза не должна стоить перезапуска
        процесса с моделью).
        """
        rate = int(sample_rate or self.sample_rate)
        started = time.perf_counter()
        try:
            audio = self.model.apply_tts(text=text, speaker=speaker, sample_rate=rate)
        except Exception as exc:  # noqa: BLE001 — текст ошибки важнее типа
            hint = (
                f" Доступные голоса: {', '.join(self.speakers)}."
                if self.speakers
                else ""
            )
            raise RuntimeError(
                f"синтез не удался ({type(exc).__name__}: {exc}).{hint}"
            ) from None
        synth_ms = (time.perf_counter() - started) * 1000.0
        info: dict[str, Any] = {
            "ms": round(synth_ms, 1),
            "duration_s": round(float(audio.numel()) / float(rate), 3),
            "sample_rate": rate,
            "speaker": speaker,
        }
        if out_path is None:
            # Без "out": WAV едет в base64 (клиенту не нужен временный файл).
            with tempfile.TemporaryDirectory(prefix="silero-worker-") as tmp:
                path = Path(tmp) / "reply.wav"
                info["pcm_bytes"] = _write_wav(path, audio, rate)
                info["wav_b64"] = base64.b64encode(path.read_bytes()).decode("ascii")
        else:
            info["pcm_bytes"] = _write_wav(out_path, audio, rate)
            info["out"] = str(out_path)
        return info


def _tune_torch(torch) -> str:
    """Настроить TorchScript на предсказуемую задержку, вернуть режим.

    Profiling-executor (по умолчанию) первые два вызова ``apply_tts`` тратит
    ~0,9 с на профилирование графа и только потом включает оптимизированный
    план: вторая фраза диалога внезапно стоит ~0,85 с. Если отключить
    профилирование, первый вызов дороже на ~0,1 с, а второй и все следующие —
    0,02–0,25 с. Звук при этом **бит-в-бит тот же** (проверено сравнением
    PCM для одной и той же фразы), поэтому режим включён по умолчанию;
    ``SILERO_JIT_PROFILING=1`` возвращает поведение torch.

    Возвращает строку для ``ready``-ответа: ``"off"`` / ``"on"``.
    """
    raw = os.environ.get("SILERO_JIT_PROFILING", "").strip().lower()
    if raw in ("1", "true", "yes", "on", "да"):
        return "on"
    for name, value in (
        ("_jit_set_profiling_executor", False),
        ("_jit_set_profiling_mode", False),
    ):
        setter = getattr(getattr(torch, "_C", None), name, None)
        if setter is None:  # pragma: no cover — другая сборка torch
            return "on"
        try:
            setter(value)
        except Exception:  # noqa: BLE001 — приватный API, не критичен
            return "on"
    return "off"


def load_backend(*, sample_rate: int = DEFAULT_SAMPLE_RATE) -> SileroBackend:
    """Импортировать torch из ``PYTHONPATH``, загрузить модель v4_ru, отдать backend.

    Дорогая часть: ``import torch`` + ``torch.hub.load`` (кэш в ``TORCH_HOME``).
    Вызывается ровно один раз за жизнь процесса. ``_check_torch``/``_load_model``
    из ``silero_speak`` при проблеме печатают причину в stderr и выходят
    (``SystemExit``) — ``serve`` перехватывает это и отдаёт ``ready=false``.
    """
    os.environ.setdefault("TORCH_HOME", str(DEFAULT_TORCH_HOME))
    threads_raw = os.environ.get("SILERO_THREADS", "4").strip() or "4"
    try:
        threads = max(1, int(threads_raw))
    except ValueError:
        threads = 4

    import_started = time.perf_counter()
    torch = _check_torch()
    import_ms = (time.perf_counter() - import_started) * 1000.0
    try:
        torch.set_num_threads(threads)
    except Exception:  # noqa: BLE001 — не критично для синтеза
        pass
    model_started = time.perf_counter()
    model = _load_model(torch)
    model_ms = (time.perf_counter() - model_started) * 1000.0
    try:
        model.to(torch.device("cpu"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"модель не переехала на CPU: {exc}") from None
    backend = SileroBackend(torch, model, sample_rate=sample_rate)
    backend.import_ms = import_ms
    backend.model_ms = model_ms
    backend.jit_profiling = _tune_torch(torch)
    return backend


# --------------------------------------------------------------------------- #
# сервер
# --------------------------------------------------------------------------- #
def _handle(
    request: Any,
    *,
    backend: Any,
    stdout: TextIO,
    default_speaker: str,
    sample_rate: int,
    started: float,
    served: int,
) -> tuple[int, bool]:
    """Обработать один запрос. Вернуть ``(число синтезов, продолжать ли)``."""
    if not isinstance(request, dict):
        _error(stdout, "запрос должен быть JSON-объектом")
        return served, True

    command = str(request.get("cmd") or "synth").strip().lower()
    if command == "ping":
        _safe_emit(
            stdout,
            {
                "ok": True,
                "pong": True,
                "pid": os.getpid(),
                "uptime_ms": round((time.perf_counter() - started) * 1000.0, 1),
                "requests": served,
                **_proc_stats(),
            },
        )
        return served, True
    if command in ("shutdown", "exit", "stop"):
        _safe_emit(stdout, {"ok": True, "bye": True, "requests": served})
        return served, False
    if command != "synth":
        _error(stdout, f"неизвестная команда: {command!r}")
        return served, True

    text = str(request.get("text") or "").strip()
    if not text:
        _error(stdout, "пустой текст для синтеза")
        return served, True
    speaker = str(request.get("speaker") or default_speaker).strip().lower()
    if not speaker:
        speaker = default_speaker
    known = getattr(backend, "speakers", ()) or ()
    if known and speaker not in known:
        _error(
            stdout,
            f"неизвестный голос {speaker!r}; допустимо: {', '.join(known)}",
        )
        return served, True

    raw_out = request.get("out")
    out_path = Path(str(raw_out)).expanduser() if raw_out else None
    raw_rate = request.get("sample_rate")
    try:
        rate = int(raw_rate) if raw_rate else sample_rate
    except (TypeError, ValueError):
        rate = sample_rate

    try:
        info = backend.synthesize(text, speaker, out_path, sample_rate=rate)
    except Exception as exc:  # noqa: BLE001 — фраза не удалась, worker живёт
        _error(stdout, str(exc) or type(exc).__name__)
        return served, True

    served += 1
    _safe_emit(stdout, {"ok": True, **info, "requests": served})
    return served, True


def serve(
    *,
    backend: Any | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    speaker: str = DEFAULT_SPEAKER,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> int:
    """Загрузить модель (если не передана) и обслуживать запросы до EOF.

    ``backend``/``stdin``/``stdout`` подставляются в тестах: протокол
    проверяется без torch и без реальной модели. Возвращает код выхода:
    0 — EOF, ``shutdown`` или сигнал; 1 — модель не загрузилась.
    """
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    started = time.perf_counter()

    if backend is None:
        try:
            backend = load_backend(sample_rate=sample_rate)
        except SystemExit:
            # Причина уже напечатана в stderr (silero_speak._fail).
            _error(
                stdout,
                "не удалось загрузить torch/модель v4_ru (подробности в stderr)",
                ready=False,
            )
            return 1
        except Exception as exc:  # noqa: BLE001
            _error(
                stdout,
                f"не удалось загрузить модель: {type(exc).__name__}: {exc}",
                ready=False,
            )
            return 1

    load_ms = (time.perf_counter() - started) * 1000.0
    if not _safe_emit(
        stdout,
        {
            "ok": True,
            "ready": True,
            "pid": os.getpid(),
            "load_ms": round(load_ms, 1),
            "speaker": speaker,
            "sample_rate": int(sample_rate),
            "torch": str(getattr(backend, "torch_version", "?")),
            "protocol": PROTOCOL_VERSION,
            "jit_profiling": str(getattr(backend, "jit_profiling", "?")),
            "import_ms": round(float(getattr(backend, "import_ms", 0.0)), 1),
            "model_ms": round(float(getattr(backend, "model_ms", 0.0)), 1),
            **_proc_stats(),
        },
    ):
        return 0

    served = 0
    # Чтение построчное и блокирующее: idle CPU ≈ 0% (никакого опроса).
    try:
        for line in stdin:
            text = line.strip()
            if not text:
                continue
            try:
                request = json.loads(text)
            except json.JSONDecodeError as exc:
                _error(stdout, f"битый JSON: {exc.msg}")
                continue
            served, keep_going = _handle(
                request,
                backend=backend,
                stdout=stdout,
                default_speaker=speaker,
                sample_rate=int(sample_rate),
                started=started,
                served=served,
            )
            if not keep_going:
                return 0
    except (BrokenPipeError, OSError, ValueError):
        # Родитель закрыл пайп: писать некуда, выходим без шума.
        return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Silero TTS worker: модель один раз, фразы — JSON-строками."
    )
    parser.add_argument(
        "--speaker",
        default=os.environ.get("SILERO_SPEAKER", DEFAULT_SPEAKER),
        help=f"голос v4_ru по умолчанию (по умолчанию {DEFAULT_SPEAKER})",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"частота синтеза, Гц (по умолчанию {DEFAULT_SAMPLE_RATE})",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("TORCH_HOME", str(DEFAULT_TORCH_HOME))
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    return serve(speaker=args.speaker, sample_rate=int(args.sample_rate))


if __name__ == "__main__":
    raise SystemExit(main())
