"""Локальный бэкенд ВТМ для гибрида: C-ABI ``src/vtm/api.h`` через ``ctypes``.

Зачем: до этой задачи «маленькая модель» в ``eva`` умела только HTTP
(``eva/clients/openai_compat.py``). Своя ВТМ уже имеет C-ABI с колбэком
стриминга и отменой из другого потока, но из Python её никто не звал. Здесь
тонкая обвязка без демона и без сети: ``libvtm`` грузится в процесс, генерация
идёт на месте.

Как это работает:

* ``vtm_api_open()`` — долгоживущий контекст (одна загруженная машина);
* ``vtm_api_generate()`` — промпт ``int32`` id, каждый токен приходит в
  Python-колбэк; колбэк можно превратить в генератор (``stream_ids``);
* ``vtm_api_cancel()`` — атомарный флаг, безопасен из любого потока;
* ``vtm_api_last_error()`` — текст последней ошибки контекста.

Библиотека не входит в репозиторий как ``.dylib``/``.so``: в ``src/vtm``
лежит статический ``libvtm.a``, а Makefile чужой (задачу 0089 он не трогает).
Поэтому порядок поиска такой:

1. ``EVA_VTM_LIB`` — готовый ``.dylib``/``.so`` (путь к ``.a`` отвергается:
   ctypes не умеет статические архивы);
2. ``src/vtm/libvtm.dylib`` / ``src/vtm/libvtm.so``, если кто-то собрал рядом;
3. автоматическая сборка ``.dylib``/``.so`` из ``src/vtm/libvtm.a`` в
   кэш-каталог временной директории (``cc -dynamiclib`` / ``cc -shared``).

Конфиг через окружение (все переменные необязательны)::

    EVA_VTM_MODEL       путь к .vtm (по умолчанию demo/model.vtm, если есть)
    EVA_VTM_LIB         путь к libvtm.dylib / libvtm.so
    EVA_VTM_TOKENIZER   путь к tokenizer.json (по умолчанию рядом с моделью)
    EVA_VTM_MAX_TOKENS  потолок новых токенов за генерацию (по умолчанию 64, 0 — без лимита)
    EVA_VTM_BACKEND     auto|simd|scalar|lut|direct (auto — как VTM_BACKEND/vtm_load)
    EVA_VTM_THREADS     потоков в SIMD-ядрах; транслируется в VTM_THREADS (C-рантайм)
    EVA_VTM_CACHE       кэш префикса: 1 включает, 0/no/false выключает
    EVA_VTM_PREFILL_BATCH
                        токенов на пачку prefill (по умолчанию 128, 0 — по токену)
    EVA_VTM_CC          компилятор для автосборки (по умолчанию cc)

Пример::

    from eva.backends import VtmBackend

    with VtmBackend() as backend:                 # demo/model.vtm, если есть
        for chunk in backend.stream_text("привет"):
            print(chunk, end="", flush=True)

    # Без токенизатора — на уровне id:
    with VtmBackend(model="demo/model.vtm", max_tokens=4) as backend:
        ids = backend.generate([1, 2, 3])
        print(backend.last_error)

Потоки и владение:

* один контекст — одна генерация; параллельный вызов получает
  :class:`VtmBusy` (C-ABI в этом случае вернул бы ``VTM_E_ARG``);
* ``cancel()`` можно звать из любого потока, в том числе во время
  ``generate()``; отмена проверяется между токенами;
* ``close()`` нельзя звать из потока генерации (например, из ``on_token``) —
  для этого есть ``cancel()``;
* ``.chat``/``.stream_chat`` совместимы по форме с клиентами ``eva.clients``
  (``chat(messages, max_tokens=...)`` -> ``Mapping`` с ``text``/``usage``,
  ``stream_chat`` -> итератор строк), поэтому бэкенд можно подставить в
  ``eva.router`` как «маленькую модель».

Чат-шаблон (задача 0200):

* если рядом с токенизатором есть ``tokenizer_config.json`` с ChatML-шаблоном
  (Qwen), промпт рендерится как ``<|im_start|>role\\n…<|im_end|>\\n`` +
  ``<|im_start|>assistant\\n`` (:mod:`eva.chat_template`); иначе — прежний
  ``role: content``. Ручной ``prompt_builder`` по-прежнему важнее;
* рассуждения выключены по умолчанию (``think=False``): в промпте сразу
  закрытый пустой ``<think>``, поэтому ``max_tokens`` целиком уходит на ответ;
* ``think=True`` включает рассуждение с бюджетом ``reasoning_budget``: при
  исчерпании бюджет рассуждение принудительно закрывается и ответ
  **досказывается** вторым прогоном. ``max_tokens`` тогда — лимит ответа,
  полный лимит новых токенов = ``reasoning_budget + max_tokens``.

Только стандартная библиотека: ``ctypes``, ``subprocess`` (только для
одноразовой автосборки библиотеки), ``threading``, ``queue``.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .tokenizer import BPETokenizer, StreamDecoder, TokenizerError
from ..chat_template import (
    DEFAULT_REASONING_BUDGET,
    THINK_CLOSE_SUFFIX,
    ChatTemplate,
    detect_chat_template,
    reasoning_steps,
)

__all__ = [
    "VTMConfig",
    "VtmBackend",
    "VtmBusy",
    "VtmCancelled",
    "VtmError",
    "VtmGeneration",
    "VtmUnavailable",
    "available",
    "build_shared_library",
    "default_model_path",
    "ensure_library",
    "format_chat_prompt",
    "load_library",
]

# --- C-ABI (src/vtm/api.h) -------------------------------------------------

VTM_API_ABI_VERSION = 1

VTM_API_OK = 0
VTM_API_STOPPED = 1
VTM_API_CANCELLED = 2
VTM_API_EOS = 3

_BACKENDS = ("auto", "simd", "scalar", "lut", "direct")

_ENV_MODEL = "EVA_VTM_MODEL"
_ENV_LIB = "EVA_VTM_LIB"
_ENV_TOKENIZER = "EVA_VTM_TOKENIZER"
_ENV_MAX_TOKENS = "EVA_VTM_MAX_TOKENS"
_ENV_BACKEND = "EVA_VTM_BACKEND"
_ENV_CACHE = "EVA_VTM_CACHE"
_ENV_PREFILL_BATCH = "EVA_VTM_PREFILL_BATCH"
_ENV_THREADS = "EVA_VTM_THREADS"
_ENV_CC = "EVA_VTM_CC"

DEFAULT_MAX_TOKENS = 64
# Батчевый prefill (0114, 0180): по умолчанию 128 токенов на пачку, как и
# умолчание C-ABI (VTM_API_PREFILL_BATCH_DEFAULT). Короткий промпт считается
# по длине; 0 — выключатель (прежний путь по токену).
# 256 — потолок пачки ВТМ после 0181 (слот активаций упакован по интервалам
# жизни тензоров: 30B 3,41 → 0,60 МБ на токен). Плотные 3B всё равно
# режутся гейтом «пачка выгодна» до 12, регресса нет.
DEFAULT_PREFILL_BATCH = 256


class VtmError(RuntimeError):
    """Ошибка обвязки ВТМ: библиотека, контекст, генерация или токенизатор."""


class VtmUnavailable(VtmError):
    """``libvtm`` не найдена/не собрана — можно деградировать (тесты: skip)."""


class VtmCancelled(VtmError):
    """Генерация отменена через :meth:`VtmBackend.cancel`."""


class VtmBusy(VtmError):
    """Контекст занят другой генерацией (C-ABI: один поток на контекст)."""


# --- структуры C-ABI --------------------------------------------------------

class _VtmApiOptions(ctypes.Structure):
    """Раскладка ``vtm_api_options`` (api.h, ABI version 1)."""

    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("backend_name", ctypes.c_char_p),
        ("backend", ctypes.c_void_p),
        ("max_new_tokens", ctypes.c_uint32),
        ("eos_token_id", ctypes.c_int32),
        # Первый элемент прежнего reserved[6] занят cache_prefix (0087), поэтому
        # sizeof остался 56 — раскладка ABI v1 не менялась.
        ("cache_prefix", ctypes.c_uint32),
        # Следующий элемент reserved занят батчевым prefill (0114); sizeof = 56 не менялся.
        ("prefill_batch", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


_TOKEN_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_void_p
)


# --- пути и конфиг ----------------------------------------------------------

def _repo_root() -> Path:
    # eva/backends/vtm_backend.py -> eva/backends -> eva -> <repo>
    return Path(__file__).resolve().parents[2]


def _dylib_filename() -> str:
    return "libvtm.dylib" if sys.platform == "darwin" else "libvtm.so"


def default_model_path(environ: Mapping[str, str] | None = None) -> Path | None:
    """Путь к модели по умолчанию: ``EVA_VTM_MODEL`` или ``demo/model.vtm``."""
    env = os.environ if environ is None else environ
    raw = (env.get(_ENV_MODEL) or "").strip()
    if raw:
        return Path(raw).expanduser()
    demo = _repo_root() / "demo" / "model.vtm"
    return demo if demo.is_file() else None


def _parse_int(name: str, value: Any, *, minimum: int) -> int:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: ожидалось целое число, получено {value!r}") from exc
    if number < minimum:
        raise ValueError(f"{name}: ожидалось >= {minimum}, получено {number}")
    return number


@dataclass(frozen=True)
class VTMConfig:
    """Конфиг бэкенда (обычно ``VTMConfig.from_env()``)."""

    model: str | None = None
    library: str | None = None
    tokenizer: str | None = None
    backend: str = "auto"
    max_tokens: int = DEFAULT_MAX_TOKENS
    threads: int | None = None
    # Кэш префикса (0087): следующий запрос с тем же началом не пересчитывается.
    # Включён по умолчанию (2026-11-26): в чате промпт = системный + история,
    # и каждый вопрос заново пересчитывал 200–500 токенов (6–13 с TTFT).
    # С кэшем пересчитывается только новый хвост. Откат — EVA_VTM_CACHE_PREFIX=0
    # или cache_prefix=False в конфиге.
    cache_prefix: bool = True
    # Батчевый prefill (0114, 0180): промпт (а с кэшем — некэшированный хвост)
    # считается пачками по N токенов; 0 — выключатель, по токену.
    # По умолчанию 128, как в C-ABI: короткий промпт — по длине.
    prefill_batch: int = DEFAULT_PREFILL_BATCH

    def __post_init__(self) -> None:
        if self.max_tokens < 0:
            raise ValueError(f"max_tokens: ожидалось >= 0, получено {self.max_tokens}")
        if self.threads is not None and self.threads < 1:
            raise ValueError(f"threads: ожидалось >= 1, получено {self.threads}")
        if self.prefill_batch < 0:
            raise ValueError(
                f"prefill_batch: ожидалось >= 0, получено {self.prefill_batch}"
            )
        if self.backend not in _BACKENDS:
            raise ValueError(
                f"backend={self.backend!r}: ожидалось одно из {', '.join(_BACKENDS)}"
            )

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        model: str | None = None,
        library: str | None = None,
        tokenizer: str | None = None,
        backend: str | None = None,
        max_tokens: int | None = None,
        threads: int | None = None,
        cache_prefix: bool | None = None,
        prefill_batch: int | None = None,
    ) -> "VTMConfig":
        """Собрать конфиг: аргументы важнее окружения, окружение — дефолтов."""
        env = os.environ if environ is None else environ

        def pick(argument: Any, variable: str) -> str | None:
            if argument is not None:
                return str(argument)
            raw = env.get(variable)
            return str(raw) if raw not in (None, "") else None

        resolved_model = pick(model, _ENV_MODEL)
        if resolved_model is None:
            demo = _repo_root() / "demo" / "model.vtm"
            resolved_model = str(demo) if demo.is_file() else None

        resolved_tokenizer = pick(tokenizer, _ENV_TOKENIZER)
        if resolved_tokenizer is None and resolved_model:
            sibling = Path(resolved_model).expanduser().with_name("tokenizer.json")
            if sibling.is_file():
                resolved_tokenizer = str(sibling)

        raw_backend = pick(backend, _ENV_BACKEND) or "auto"
        resolved_backend = raw_backend.strip().lower()
        if resolved_backend not in _BACKENDS:
            raise ValueError(
                f"EVA_VTM_BACKEND={raw_backend!r}: ожидалось одно из "
                f"{', '.join(_BACKENDS)}"
            )

        raw_max = pick(max_tokens, _ENV_MAX_TOKENS)
        resolved_max = (
            DEFAULT_MAX_TOKENS if raw_max is None
            else _parse_int(_ENV_MAX_TOKENS, raw_max, minimum=0)
        )
        raw_threads = pick(threads, _ENV_THREADS)
        resolved_threads = (
            None if raw_threads is None
            else _parse_int(_ENV_THREADS, raw_threads, minimum=1)
        )
        raw_batch = pick(prefill_batch, _ENV_PREFILL_BATCH)
        resolved_batch = (
            DEFAULT_PREFILL_BATCH if raw_batch is None
            else _parse_int(_ENV_PREFILL_BATCH, raw_batch, minimum=0)
        )

        return cls(
            model=resolved_model,
            library=pick(library, _ENV_LIB),
            tokenizer=resolved_tokenizer,
            backend=resolved_backend,
            max_tokens=resolved_max,
            threads=resolved_threads,
            cache_prefix=(
                cache_prefix
                if cache_prefix is not None
                else env.get(_ENV_CACHE, "") not in ("", "0", "no", "false")
            ),
            prefill_batch=resolved_batch,
        )


# --- поиск и сборка библиотеки ---------------------------------------------

def _as_library_path(path: str | os.PathLike[str]) -> Path:
    return Path(path).expanduser()


def _reject_static(path: Path) -> None:
    if path.suffix in (".a", ".lib"):
        raise VtmUnavailable(
            f"{path}: это статический архив, ctypes умеет только .dylib/.so; "
            "соберите разделяемую библиотеку или задайте EVA_VTM_LIB на неё"
        )


def build_shared_library(
    static_lib: str | os.PathLike[str],
    *,
    out_dir: str | os.PathLike[str] | None = None,
    cc: str | None = None,
    timeout: float = 180.0,
) -> Path:
    """Собрать ``.dylib``/``.so`` из ``libvtm.a`` во временный кэш.

    Makefile в ``src/vtm`` чужой (зона задачи 0089), поэтому сборка делается
    здесь: один вызов компилятора, результат кэшируется по mtime/size архива
    и платформе. Возвращает путь к готовой библиотеке.
    """
    static = _as_library_path(static_lib)
    if not static.is_file():
        raise VtmUnavailable(f"нет статической библиотеки: {static}")
    compiler = cc or os.environ.get(_ENV_CC, "cc")
    if shutil.which(compiler) is None:
        raise VtmUnavailable(
            f"компилятор {compiler!r} не найден; задайте EVA_VTM_CC или "
            "постройте libvtm.dylib/libvtm.so сами и укажите EVA_VTM_LIB"
        )
    stat = static.stat()
    digest = hashlib.sha256(
        "|".join(
            (
                str(static.resolve()),
                str(stat.st_mtime_ns),
                str(stat.st_size),
                sys.platform,
                os.uname().machine if hasattr(os, "uname") else "",
                compiler,
            )
        ).encode("utf-8")
    ).hexdigest()[:16]

    if out_dir is None:
        uid = getattr(os, "getuid", lambda: 0)()
        cache = Path(tempfile.gettempdir()) / f"eva-vtm-lib-{uid}"
    else:
        cache = _as_library_path(out_dir)
    try:
        cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        raise VtmUnavailable(f"автосборка: каталог кэша {cache}: {exc}") from exc
    target = cache / f"libvtm-{digest}{Path(_dylib_filename()).suffix}"
    if target.is_file():
        return target

    macos = sys.platform == "darwin"
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    command = [compiler, "-dynamiclib" if macos else "-shared", "-o", str(temporary)]
    if macos:
        command += ["-Wl,-all_load", str(static)]
        # libvtm.a на Darwin содержит backend_metal.o (задача 0192): без этих
        # фреймворков линковка падает на _objc_msgSend/MTLCreateSystemDefaultDevice.
        command += [
            "-framework", "Metal",
            "-framework", "Foundation",
            "-lobjc",
        ]
    else:
        command += ["-Wl,--whole-archive", str(static), "-Wl,--no-whole-archive"]
    command += ["-lm", "-lpthread"]

    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VtmUnavailable(f"автосборка {static.name}: {exc}") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise VtmUnavailable(
            "не удалось собрать разделяемую библиотеку ВТМ из "
            f"{static}: код {proc.returncode}; "
            f"stderr: {stderr[-600:] or '<пусто>'}"
        )
    try:
        os.replace(temporary, target)
    except OSError as exc:
        raise VtmUnavailable(
            f"автосборка: не удалось переименовать {temporary}: {exc}"
        ) from exc
    return target


def ensure_library(
    library: str | os.PathLike[str] | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    build: bool = True,
) -> Path:
    """Найти (или собрать) загружаемую ``.dylib``/``.so`` ВТМ.

    ``library``/``EVA_VTM_LIB`` важнее поиска; ``root`` — корень репозитория.
    Если готовой разделяемой библиотеки нет, а ``src/vtm/libvtm.a`` есть,
    она собирается в кэш. Иначе — :class:`VtmUnavailable` с инструкцией.
    """
    env = os.environ if environ is None else environ
    root_path = _as_library_path(root) if root is not None else _repo_root()
    explicit = library if library is not None else env.get(_ENV_LIB)
    if explicit:
        path = _as_library_path(str(explicit))
        _reject_static(path)
        if not path.is_file():
            raise VtmUnavailable(
                f"библиотека {str(explicit)!r}: файла нет; укажите путь к "
                "libvtm.dylib/libvtm.so"
            )
        return path

    for name in (_dylib_filename(), "libvtm.dylib", "libvtm.so"):
        candidate = root_path / "src" / "vtm" / name
        if candidate.is_file():
            return candidate

    static = root_path / "src" / "vtm" / "libvtm.a"
    if static.is_file():
        if not build:
            raise VtmUnavailable(
                f"нет разделяемой библиотеки ВТМ; соберите её из {static} "
                "или задайте EVA_VTM_LIB"
            )
        return build_shared_library(static, cc=env.get(_ENV_CC))

    raise VtmUnavailable(
        "не найдена библиотека ВТМ: нет EVA_VTM_LIB, "
        f"{root_path / 'src' / 'vtm' / _dylib_filename()} и {static}. "
        "Соберите `make -C src/vtm libvtm.a`, затем задайте EVA_VTM_LIB на "
        ".dylib/.so (или оставьте автосборку)"
    )


def available(
    library: str | os.PathLike[str] | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """``True``, если библиотеку можно найти/собрать и загрузить."""
    try:
        load_library(library, root=root, environ=environ)
    except VtmUnavailable:
        return False
    return True


class _Library:
    """Загруженная ``libvtm`` с проверенной сигнатурой C-ABI."""

    def __init__(self, path: Path) -> None:
        self.path = str(path)
        try:
            self.cdll = ctypes.CDLL(self.path)
        except OSError as exc:
            raise VtmUnavailable(f"не удалось загрузить {self.path}: {exc}") from exc
        try:
            self.cdll.vtm_api_open.restype = ctypes.c_void_p
            self.cdll.vtm_api_open.argtypes = [
                ctypes.c_char_p,
                ctypes.POINTER(_VtmApiOptions),
            ]
            self.cdll.vtm_api_generate.restype = ctypes.c_int
            self.cdll.vtm_api_generate.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_size_t,
                _TOKEN_CALLBACK,
                ctypes.c_void_p,
            ]
            self.cdll.vtm_api_cancel.restype = None
            self.cdll.vtm_api_cancel.argtypes = [ctypes.c_void_p]
            # Добавлено в 0185; у старой собранной библиотеки символа может не
            # быть — тогда флаг снимается только следующим generate().
            clear = getattr(self.cdll, "vtm_api_cancel_clear", None)
            if clear is not None:
                clear.restype = None
                clear.argtypes = [ctypes.c_void_p]
            self.clear_cancel = clear
            self.cdll.vtm_api_close.restype = None
            self.cdll.vtm_api_close.argtypes = [ctypes.c_void_p]
            self.cdll.vtm_api_last_error.restype = ctypes.c_char_p
            self.cdll.vtm_api_last_error.argtypes = [ctypes.c_void_p]
        except AttributeError as exc:
            raise VtmUnavailable(
                f"{self.path}: нет символа C-ABI ВТМ ({exc}); это точно libvtm?"
            ) from exc


_LIBRARIES: dict[str, _Library] = {}
_LIBRARIES_LOCK = threading.Lock()


def load_library(
    library: str | os.PathLike[str] | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> _Library:
    """Найти, при необходимости собрать, загрузить и закэшировать библиотеку."""
    path = ensure_library(library, root=root, environ=environ)
    key = str(path.resolve())
    with _LIBRARIES_LOCK:
        cached = _LIBRARIES.get(key)
        if cached is not None:
            return cached
        loaded = _Library(path)
        _LIBRARIES[key] = loaded
        return loaded


def _decode_error(raw: bytes | None) -> str:
    if not raw:
        return ""
    return raw.decode("utf-8", "replace").strip()


# --- контекст C-ABI ---------------------------------------------------------

@dataclass(frozen=True)
class VtmGeneration:
    """Итог одного ``vtm_api_generate()``."""

    ids: tuple[int, ...]
    finish_reason: str  # "length" | "stop" | "eos"

    @property
    def count(self) -> int:
        return len(self.ids)


class _VtmContext:
    """Владелец ``vtm_api_t*``; не потокобезопасен, синхронизует бэкенд."""

    def __init__(
        self,
        library: _Library,
        model: str | os.PathLike[str],
        *,
        backend: str = "auto",
        max_new_tokens: int = 0,
        eos_token_id: int = -1,
        cache_prefix: bool | None = None,
        prefill_batch: int = DEFAULT_PREFILL_BATCH,
    ) -> None:
        self._lib = library
        self._handle: int | None = None
        # Замок «идёт ли C-генерация»: нужен, чтобы «поздняя» отмена не
        # оставила флаг в C и не отменила следующий запуск (0185).
        self._state = threading.Lock()
        self._running = False
        self._stop_requested = False
        self.model = str(model)
        self._max_new = int(max_new_tokens) if max_new_tokens > 0 else 0
        options = _VtmApiOptions()
        options.struct_size = ctypes.sizeof(_VtmApiOptions)
        options.flags = 0
        # "auto" оставляем NULL: тогда vtm_load() уважает VTM_BACKEND из env.
        options.backend_name = None if backend in ("", "auto") else backend.encode()
        options.max_new_tokens = int(max_new_tokens) if max_new_tokens > 0 else 0
        options.eos_token_id = int(eos_token_id)
        # Кэш префикса (0087): по умолчанию выключен, чтобы не менять поведение.
        if cache_prefix is None:
            cache_prefix = os.environ.get(_ENV_CACHE, "") not in ("", "0", "no", "false")
        options.cache_prefix = 1 if cache_prefix else 0
        options.prefill_batch = int(prefill_batch or 0)
        try:
            handle = library.cdll.vtm_api_open(self.model.encode(), ctypes.byref(options))
        except Exception as exc:  # noqa: BLE001 — ctypes зовёт C, падать нельзя молча
            raise VtmError(f"vtm_api_open: {self.model}: {exc}") from exc
        if not handle:
            detail = _decode_error(library.cdll.vtm_api_last_error(None))
            raise VtmError(
                f"vtm_api_open: {self.model}: "
                f"{detail or 'не удалось открыть (нет текста ошибки)'}"
            )
        self._handle = int(handle)

    @property
    def closed(self) -> bool:
        return self._handle is None

    @property
    def last_error(self) -> str:
        if self._handle is None:
            return ""
        return _decode_error(self._lib.cdll.vtm_api_last_error(self._handle))

    def cancel(self) -> None:
        """Атомарный флаг отмены: идущую генерацию или ближайшую."""
        if self._handle is not None:
            self._lib.cdll.vtm_api_cancel(self._handle)

    def cancel_active(self) -> bool:
        """Отменить только идущую генерацию; ``False`` — нечего отменять.

        В отличие от :meth:`cancel`, не ставит флаг, если C-вызов уже
        вернулся: иначе «поздний» стоп отменил бы следующий запрос. Замок
        закрывает окно между возвратом ``vtm_api_generate()`` и снятием
        ``_running``: отмена либо попадает внутрь вызова, либо не ставится.
        """
        with self._state:
            if self._handle is None or not self._running:
                return False
            self._stop_requested = True
            self._lib.cdll.vtm_api_cancel(self._handle)
            return True

    def generate(
        self,
        prompt_ids: Sequence[int],
        *,
        on_token: Callable[[int], None] | None = None,
        max_new_tokens: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> VtmGeneration:
        """Один прогон. Может бросить :class:`VtmCancelled`/`VtmError`."""
        if self._handle is None:
            raise VtmError("vtm: контекст закрыт")
        ids = [int(token) for token in prompt_ids]
        if not ids:
            raise VtmError("vtm_api_generate: пустой промпт (n=0)")
        for index, token in enumerate(ids):
            if token < 0:
                raise VtmError(f"vtm_api_generate: prompt_ids[{index}]={token} отрицателен")
        limit = None if max_new_tokens is None else int(max_new_tokens)
        if limit is not None and limit < 0:
            raise ValueError(f"max_new_tokens: ожидалось >= 0, получено {limit}")
        if limit == 0:
            limit = None

        collected: list[int] = []
        failure: list[BaseException] = []

        def _callback(raw_ids, count, _user):  # noqa: ANN001 — тип ctypes
            try:
                for index in range(count):
                    token = int(raw_ids[index])
                    if limit is not None and len(collected) >= limit:
                        return 1
                    collected.append(token)
                    if on_token is not None:
                        on_token(token)
                    if should_stop is not None and should_stop():
                        return 1
                    if limit is not None and len(collected) >= limit:
                        return 1
            except BaseException as exc:  # noqa: BLE001 — исключение из callback
                failure.append(exc)
                return 1
            return 0

        callback = _TOKEN_CALLBACK(_callback)
        prompt = (ctypes.c_int32 * len(ids))(*ids)
        with self._state:
            self._running = True
            self._stop_requested = False
        try:
            status = int(
                self._lib.cdll.vtm_api_generate(
                    self._handle, prompt, ctypes.c_size_t(len(ids)), callback, None
                )
            )
        finally:
            # Отмена, пришедшая после возврата C, не должна ждать следующий
            # запрос: снимаем оставшийся флаг (0185).
            with self._state:
                self._running = False
                late = self._stop_requested
                self._stop_requested = False
            if late and self._lib.clear_cancel is not None and self._handle is not None:
                self._lib.clear_cancel(self._handle)
        if failure:
            # Исключение из колбэка нельзя бросить через C: оно уже остановило
            # генерацию (STOPPED), здесь поднимаем исходное.
            raise failure[0]
        if status < 0:
            detail = self.last_error
            raise VtmError(
                f"vtm_api_generate: код {status}: "
                f"{detail or 'нет текста ошибки'}"
            )
        if status == VTM_API_CANCELLED:
            raise VtmCancelled("генерация отменена")
        if status == VTM_API_EOS:
            reason = "eos"
        elif status == VTM_API_STOPPED:
            reason = "length" if limit is not None and len(collected) >= limit else "stop"
        elif status == VTM_API_OK:
            # VTM_API_OK — лимит max_new_tokens контекста или HALT программы.
            ours = limit is not None and len(collected) >= limit
            native = self._max_new > 0 and len(collected) >= self._max_new
            reason = "length" if (ours or native) else "stop"
        else:
            raise VtmError(f"vtm_api_generate: неизвестный код {status}")
        return VtmGeneration(ids=tuple(collected), finish_reason=reason)

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            self._lib.cdll.vtm_api_close(handle)


# --- формат сообщений -------------------------------------------------------

def _role_content(message: Any) -> tuple[str, str]:
    if isinstance(message, Mapping):
        role = message.get("role") or "user"
        content = message.get("content")
    else:
        role = getattr(message, "role", None) or "user"
        content = getattr(message, "content", None)
    return str(role), "" if content is None else str(content)


def format_chat_prompt(messages: Any) -> str:
    """Переносимый дефолт для моделей без ChatML: ``role: content`` + ``assistant:``.

    Для Qwen автоматически берётся настоящий шаблон из
    :mod:`eva.chat_template` (``<|im_start|>role\\n…``). Сюда бэкенд
    приходит, только если рядом с токенизатором нет ``tokenizer_config.json``
    с ChatML-шаблоном (или явно задан ``chat_template="plain"``).
    """
    if isinstance(messages, str):
        return messages
    lines: list[str] = []
    for message in messages or []:
        role, content = _role_content(message)
        lines.append(f"{role}: {content}")
    if not lines:
        return ""
    lines.append("assistant:")
    return "\n".join(lines)


# --- бэкенд -----------------------------------------------------------------

#: Минимальный потолок C-контекста: длинный ответ не должен обрываться на
#: дефолтном лимите конфига. Реальный лимит прогона задаёт вызывающий код.
_NATIVE_TOKEN_FLOOR = 4096


def _native_token_limit(
    max_tokens: int, *, thinking: bool, reasoning_budget: int
) -> int:
    """Потолок ``max_new_tokens`` C-контекста с учётом бюджета рассуждения.

    C-рантайм обрезает любую генерацию по этому потолку. При рассуждении
    прогонов два (бюджет + ответ), поэтому потолок — их сумма; иначе первый
    прогон обрывался бы по ``max_tokens`` и «бюджет» 64 молча становился 32
    (задача 0200).
    """
    limit = int(max_tokens)
    if thinking and limit > 0:
        limit += max(0, int(reasoning_budget))
    return limit


class VtmBackend:
    """Локальная ВТМ как «маленькая модель»: генерация, стриминг, отмена.

    Создаётся лениво: библиотека и контекст открываются при первом обращении
    (``open()``, ``generate()``, ``chat()``), поэтому импорт модуля ничего не
    грузит и не падает, если ВТМ не собрана. ``VtmUnavailable`` означает
    «локальной модели нет» — вызывающий может откатиться на HTTP-клиент.
    """

    def __init__(
        self,
        config: VTMConfig | None = None,
        *,
        model: str | None = None,
        library: str | None = None,
        tokenizer: str | None = None,
        backend: str | None = None,
        max_tokens: int | None = None,
        threads: int | None = None,
        cache_prefix: bool | None = None,
        prefill_batch: int | None = None,
        environ: Mapping[str, str] | None = None,
        prompt_builder: Callable[[Any], str] | None = None,
        think: bool = False,
        reasoning_budget: int = DEFAULT_REASONING_BUDGET,
        chat_template: str = "auto",
        no_think_suffix: bool = False,
    ) -> None:
        resolved = config if config is not None else VTMConfig.from_env(environ)
        overrides = {
            "model": model,
            "library": library,
            "tokenizer": tokenizer,
            "backend": backend,
            "max_tokens": max_tokens,
            "threads": threads,
        }
        if cache_prefix is not None:
            overrides["cache_prefix"] = cache_prefix
        if prefill_batch is not None:
            overrides["prefill_batch"] = prefill_batch
        overrides = {key: value for key, value in overrides.items() if value is not None}
        if overrides:
            resolved = replace(resolved, **overrides)
        self.config = resolved
        if isinstance(reasoning_budget, bool) or int(reasoning_budget) < 0:
            raise ValueError(
                f"reasoning_budget: ожидалось >= 0, получено {reasoning_budget!r}"
            )
        self.think = bool(think)
        self.reasoning_budget = int(reasoning_budget)
        self.no_think_suffix = bool(no_think_suffix)
        mode = str(chat_template or "auto").strip().lower()
        if mode not in ("auto", "qwen", "plain"):
            raise ValueError(
                f"chat_template={chat_template!r}: ожидалось auto|qwen|plain"
            )
        self.chat_template = mode
        # Шаблон ищем по соседнему tokenizer_config.json: он есть у Qwen, и в
        # нём лежит настоящий ChatML (задача 0200). Нет шаблона — прежний
        # переносимый ``role: content``.
        self.template: ChatTemplate | None = (
            None if mode == "plain" else detect_chat_template(self.config.tokenizer)
        )
        self._thinking = bool(
            self.think
            and self.template is not None
            and self.template.supports_thinking
        )
        self.prompt_builder = prompt_builder or self._default_prompt_builder()

        self._library: _Library | None = None
        self._context: _VtmContext | None = None
        self._tokenizer: BPETokenizer | None = None
        self._state = threading.Condition()
        self._busy = False
        self._closed = False
        self._active_abort: threading.Event | None = None
        self._generation_thread: threading.Thread | None = None

    def _default_prompt_builder(self) -> Callable[[Any], str]:
        """``prompt_builder``: ChatML для Qwen, иначе прежний ``role: content``.

        Для ``think=True`` рендер идёт с ``enable_thinking=True``: промпт
        кончается ``<|im_start|>assistant\\n``, и модель сама открывает
        ``<think>``. Для ``think=False`` после ``assistant`` стоит пустой
        закрытый ``<think>\\n\\n</think>\\n\\n`` — модель не рассуждает.
        """
        if self.template is None:
            return format_chat_prompt
        return self.template.builder(
            enable_thinking=self._thinking,
            no_think_suffix=self.no_think_suffix,
        )

    # -- свойства ----------------------------------------------------------

    @property
    def model_path(self) -> str | None:
        return self.config.model

    @property
    def library_path(self) -> str | None:
        return self._library.path if self._library is not None else None

    @property
    def opened(self) -> bool:
        return self._context is not None and not self._context.closed

    @property
    def tokenizer(self) -> BPETokenizer:
        """Токенизатор из ``EVA_VTM_TOKENIZER``/tokenizer.json (лениво)."""
        if self._tokenizer is not None:
            return self._tokenizer
        path = self.config.tokenizer
        if not path:
            raise VtmError(
                "токенизатор не задан: EVA_VTM_TOKENIZER или tokenizer.json "
                "рядом с моделью; для работы по id он не нужен"
            )
        try:
            self._tokenizer = BPETokenizer.from_file(path)
        except TokenizerError as exc:
            raise VtmError(str(exc)) from exc
        return self._tokenizer

    @property
    def last_error(self) -> str:
        """Текст последней ошибки C-контекста ("" — ошибок не было)."""
        context = self._context
        return context.last_error if context is not None else ""

    @property
    def busy(self) -> bool:
        with self._state:
            return self._busy

    @property
    def supports_thinking(self) -> bool:
        """Умеет ли модель рассуждение (Qwen3 + ``think=True``)."""
        return self._thinking

    def render_chat(self, messages: Any, *, enable_thinking: bool | None = None) -> str:
        """Промпт по шаблону модели (ChatML у Qwen) без генерации.

        ``enable_thinking=None`` — как сконфигурировано (``think``); явный
        флаг позволяет отрендерить оба режима одним бэкендом. Для модели без
        ChatML-шаблона работает прежний ``role: content``.
        """
        if isinstance(messages, str):
            return messages
        if self.template is None:
            return format_chat_prompt(messages)
        thinking = self._thinking if enable_thinking is None else bool(enable_thinking)
        return self.template.render(
            messages,
            enable_thinking=thinking,
            no_think_suffix=self.no_think_suffix,
        )

    # -- открытие/закрытие --------------------------------------------------

    def open(self) -> "VtmBackend":
        """Открыть библиотеку и контекст (idempotent)."""
        with self._state:
            self._ensure_open_locked()
        return self

    def _ensure_open_locked(self) -> _VtmContext:
        if self._closed:
            raise VtmError("VtmBackend: бэкенд закрыт")
        if self._context is not None and not self._context.closed:
            return self._context
        model = self.config.model
        if not model:
            raise VtmUnavailable(
                "модель ВТМ не задана: EVA_VTM_MODEL или demo/model.vtm; "
                "соберите демо: python3 scripts/demo_ternary_model.py --out demo"
            )
        model_path = Path(model).expanduser()
        if not model_path.is_file():
            raise VtmError(f"vtm: файл модели не найден: {model_path}")
        if self._library is None:
            self._library = load_library(self.config.library)
        self._apply_threads()
        eos_id = -1
        if self.config.tokenizer:
            eos_id = self.tokenizer.eos_id
        # Потолок C-контекста должен вмещать оба прогона: при рассуждении
        # первый ограничен бюджетом, второй — max_tokens (см. ниже).
        # Потолок C-контекста — НЕ лимит ответа. C останавливает генерацию
        # по этому числу и рапортует «stop», поэтому он должен быть не меньше
        # любого запрошенного ответа: иначе длинный ответ обрывался на
        # config.max_tokens (256) при запросе 400 (жалоба «ответ обрезается»).
        # Реальный лимит каждого прогона задаёт _answer_limit() и проверяет
        # колбэк в Python; рассуждение (0200) по-прежнему считает бюджет
        # отдельно — здесь берём максимум, чтобы ничего не срезать.
        native_limit = max(
            _native_token_limit(
                self.config.max_tokens,
                thinking=self._thinking,
                reasoning_budget=self.reasoning_budget,
            ),
            _NATIVE_TOKEN_FLOOR,
        )
        self._context = _VtmContext(
            self._library,
            model_path,
            backend=self.config.backend,
            max_new_tokens=native_limit,
            eos_token_id=eos_id,
            cache_prefix=self.config.cache_prefix,
            prefill_batch=self.config.prefill_batch,
        )
        return self._context

    def _apply_threads(self) -> None:
        # C-рантайм читает VTM_THREADS при инициализации пула; другого входа
        # в C-ABI нет. Не перетираем переменную, если её уже задали снаружи.
        if self.config.threads is not None and not os.environ.get("VTM_THREADS"):
            os.environ["VTM_THREADS"] = str(self.config.threads)

    def close(self, timeout: float | None = None) -> None:
        """Дождаться генерации, отменить её и освободить контекст.

        Нельзя звать из потока генерации (``on_token``/колбэка) — там нужен
        :meth:`cancel`, иначе получится самоблокировка.
        """
        with self._state:
            self._closed = True
            context = self._context
            if context is not None:
                context.cancel()
            if self._busy:
                if self._generation_thread is threading.current_thread():
                    raise VtmError(
                        "close() нельзя вызывать из потока генерации; "
                        "используйте cancel()"
                    )
                if not self._state.wait_for(lambda: not self._busy, timeout):
                    raise VtmError("close(): генерация не завершилась за отведённое время")
            self._context = None
        if context is not None:
            context.close()

    def __enter__(self) -> "VtmBackend":
        return self.open()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover — защита от утечки контекста
        try:
            context = getattr(self, "_context", None)
            if context is not None:
                context.close()
                self._context = None
        except Exception:  # noqa: BLE001 — в __del__ нельзя шуметь
            pass

    def __repr__(self) -> str:
        state = "open" if self.opened else "closed" if self._closed else "lazy"
        return (
            f"VtmBackend(model={self.config.model!r}, library={self.library_path!r}, "
            f"state={state}, max_tokens={self.config.max_tokens})"
        )

    # -- генерация ----------------------------------------------------------

    def generate(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> list[int]:
        """Сгенерировать id по промпту ``prompt_ids`` (синхронно, до конца)."""
        return list(
            self.generate_result(
                prompt_ids, max_new_tokens=max_new_tokens, on_token=on_token
            ).ids
        )

    def generate_result(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> VtmGeneration:
        """Как :meth:`generate`, но с ``finish_reason``."""
        with self._state:
            if self._closed:
                raise VtmError("VtmBackend: бэкенд закрыт")
            if self._busy:
                raise VtmBusy(
                    "контекст ВТМ занят: дождитесь завершения текущей генерации"
                )
            context = self._ensure_open_locked()
            abort = threading.Event()
            self._busy = True
            self._active_abort = abort
            self._generation_thread = threading.current_thread()
        try:
            return context.generate(
                prompt_ids,
                on_token=on_token,
                max_new_tokens=max_new_tokens,
                should_stop=abort.is_set,
            )
        finally:
            with self._state:
                self._busy = False
                self._active_abort = None
                self._generation_thread = None
                self._state.notify_all()

    def stream_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> Iterator[int]:
        """Генератор id: каждый токен приходит отдельным элементом.

        Генерация идёт в отдельном потоке, поэтому вызывающий может отменить
        её :meth:`cancel` из любого потока. Если потребитель перестал читать
        (``close()`` генератора), генерация останавливается и отменяется.
        """
        items: queue.Queue[Any] = queue.Queue()
        sentinel = object()
        abort = threading.Event()
        failure: list[BaseException] = []

        with self._state:
            if self._closed:
                raise VtmError("VtmBackend: бэкенд закрыт")
            if self._busy:
                raise VtmBusy(
                    "контекст ВТМ занят: дождитесь завершения текущей генерации"
                )
            context = self._ensure_open_locked()
            self._busy = True
            self._active_abort = abort
            self._generation_thread = None  # заполнится в worker

        def worker() -> None:
            with self._state:
                self._generation_thread = threading.current_thread()

            def emit(token: int) -> None:
                items.put(token)
                if on_token is not None:
                    on_token(token)

            try:
                context.generate(
                    prompt_ids,
                    on_token=emit,
                    max_new_tokens=max_new_tokens,
                    should_stop=abort.is_set,
                )
            except BaseException as exc:  # noqa: BLE001 — переносим в вызывающий
                failure.append(exc)
            finally:
                items.put(sentinel)
                with self._state:
                    self._busy = False
                    self._active_abort = None
                    self._generation_thread = None
                    self._state.notify_all()

        thread = threading.Thread(target=worker, name="eva-vtm-generate", daemon=True)
        thread.start()
        try:
            while True:
                item = items.get()
                if item is sentinel:
                    break
                yield int(item)
        finally:
            if thread.is_alive():
                # Потребитель ушёл (break/close): не оставляем C-машину работать.
                # cancel_active, а не cancel: если генерация уже закончилась,
                # флаг не должен «протечь» в следующий запрос (0185).
                abort.set()
                self.cancel_active()
                thread.join(timeout=5.0)
        if failure:
            raise failure[0]

    def stream_text(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int | None = None,
        on_token: Callable[[int], None] | None = None,
    ) -> Iterator[str]:
        """Генератор текста: по чанку на токен, склейка == ``decode(ids)``."""
        tokenizer = self.tokenizer
        decoder: StreamDecoder = tokenizer.stream_decoder()
        for token in self.stream_ids(
            prompt_ids, max_new_tokens=max_new_tokens, on_token=on_token
        ):
            yield decoder.push(token)
        tail = decoder.finish()
        if tail:
            yield tail

    def generate_text(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        """Синхронная генерация + декодирование в строку."""
        generation = self.generate_result(prompt_ids, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(generation.ids)

    def encode(self, text: str, *, allow_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(text, allow_special_tokens=allow_special_tokens)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids)

    # -- отмена -------------------------------------------------------------

    def cancel(self) -> None:
        """Попросить текущую/следующую генерацию остановиться.

        Безопасно из любого потока и в любой момент, в том числе во время
        ``generate()``. Если вызвать до генерации — отменится ближайшая.
        Отмена приходит через C-флаг, поэтому текущая генерация завершается
        :class:`VtmCancelled`; HTTP-подобный «мягкий» останов изнутри
        колбэка — это возврат ненулевого значения из ``on_token`` (см.
        ``_VtmContext.generate``).
        """
        with self._state:
            context = self._context
        if context is not None and not context.closed:
            context.cancel()

    def cancel_active(self) -> bool:
        """Отменить только идущую генерацию; ``False`` — отменять нечего.

        Отличие от :meth:`cancel`: флаг в C-ABI живёт до ближайшего
        ``generate()``, поэтому «поздняя» отмена (сервер прислал её через
        миллисекунду после конца ответа) отменила бы следующий запрос. Здесь
        флаг ставится только пока бэкенд занят (``_busy``), а если генерации
        нет — вызов ничего не делает. Именно этот метод зовёт сервер
        (``tools/chat_server.py``) при обрыве SSE и «Стопе».

        Потокобезопасен: ``_busy`` читается под тем же замком, под которым его
        выставляют :meth:`generate_result`/:meth:`stream_ids`.
        """
        with self._state:
            busy = self._busy
            context = self._context
        if not busy or context is None or context.closed:
            return False
        # Точная проверка «C ещё внутри generate()» — в контексте: окно между
        # возвратом vtm_api_generate() и снятием _busy иначе оставило бы флаг
        # следующему запросу.
        cancel_active = getattr(context, "cancel_active", None)
        if callable(cancel_active):
            return bool(cancel_active())
        context.cancel()
        return True

    # -- клиентский интерфейс (для eva.router) ------------------------------

    def _answer_limit(self, max_tokens: int | None) -> int | None:
        """Лимит ответа: ``max_tokens`` или ``VTMConfig.max_tokens``.

        ``None`` — без лимита (так было и раньше: ``max_new_tokens=None``).
        В режиме рассуждения это лимит только ответа; бюджет размышления
        прибавляется отдельно.
        """
        if max_tokens is not None:
            limit = int(max_tokens)
            if limit < 1:
                raise ValueError(f"max_tokens: ожидалось >= 1, получено {limit}")
            return limit
        configured = int(self.config.max_tokens)
        return configured if configured > 0 else None

    def _reasoning_steps(
        self, prompt_ids: Sequence[int], *, answer_tokens: int
    ) -> Iterator[Any]:
        """Шаги ответа с бюджетом рассуждения (``think=False`` — один прогон)."""
        tokenizer = self.tokenizer
        return reasoning_steps(
            prompt_ids,
            stream_ids=lambda ids, max_new_tokens: self.stream_ids(
                ids, max_new_tokens=max_new_tokens
            ),
            encode=tokenizer.encode,
            decode=tokenizer.decode,
            answer_tokens=answer_tokens,
            reasoning_budget=self.reasoning_budget,
            think=self._thinking,
            eos_id=tokenizer.eos_id,
        )

    def chat(self, messages: Any, max_tokens: int | None = None) -> dict[str, Any]:
        """``messages`` -> ответ одним куском (форма ``OpenAIClient.chat``).

        В режиме ``think=True`` текст содержит рассуждение (``<think>…``);
        ``usage.reasoning_tokens``/``usage.answer_tokens`` показывают, сколько
        ушло на каждую часть. Лимит новых токенов — ``reasoning_budget +
        max_tokens``.
        """
        prompt = self._prompt(messages)
        tokenizer = self.tokenizer
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            raise VtmError("chat: пустой промпт после токенизации")
        limit = self._answer_limit(max_tokens)
        finish = "stop"
        reasoning_tokens = 0
        answer_tokens = 0
        second_pass = False
        if limit is None:
            # Явный «без лимита»: прежний путь одним прогоном.
            generation = self.generate_result(prompt_ids, max_new_tokens=None)
            ids = [
                token
                for token in generation.ids
                if not (tokenizer.eos_id >= 0 and token == tokenizer.eos_id)
            ]
            finish = "length" if generation.finish_reason == "length" else "stop"
        else:
            ids = []
            for step in self._reasoning_steps(prompt_ids, answer_tokens=limit):
                if step.phase == "close":
                    second_pass = True
                    ids.extend(tokenizer.encode(THINK_CLOSE_SUFFIX))
                    continue
                if step.phase == "end":
                    finish = "length" if step.reason == "length" else "stop"
                    continue
                if step.phase == "answer":
                    second_pass = True
                    answer_tokens += 1
                else:
                    reasoning_tokens += 1
                ids.append(int(step.token))
        text = tokenizer.decode(ids)
        completion = reasoning_tokens + answer_tokens if limit is not None else len(ids)
        usage: dict[str, Any] = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": completion,
            "total_tokens": len(prompt_ids) + completion,
        }
        if reasoning_tokens:
            usage["reasoning_tokens"] = reasoning_tokens
            usage["answer_tokens"] = answer_tokens
            if second_pass:
                # Второй прогон пересчитывает промпт вместе с рассуждением:
                # это настоящий prefill, а не повторное использование кэша.
                usage["second_pass_prompt_tokens"] = (
                    len(prompt_ids)
                    + reasoning_tokens
                    + len(tokenizer.encode(THINK_CLOSE_SUFFIX))
                )
        return {
            "text": text,
            "usage": usage,
            "finish_reason": finish,
            "model": self.config.model or "",
        }

    def stream_chat(self, messages: Any, max_tokens: int | None = None) -> Iterator[str]:
        """``messages`` -> поток текстовых чанков (для ``eva.router.stream``)."""
        prompt = self._prompt(messages)
        tokenizer = self.tokenizer
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            raise VtmError("stream_chat: пустой промпт после токенизации")
        decoder = tokenizer.stream_decoder()
        limit = self._answer_limit(max_tokens)
        if limit is None:
            tokens: Iterator[tuple[str, int | None]] = (
                ("answer", token)
                for token in self.stream_ids(prompt_ids, max_new_tokens=None)
            )
        else:
            tokens = (
                (step.phase, step.token)
                for step in self._reasoning_steps(prompt_ids, answer_tokens=limit)
            )
        for phase, token in tokens:
            if phase == "close":
                yield THINK_CLOSE_SUFFIX
                continue
            if phase == "end" or token is None:
                continue
            if tokenizer.eos_id >= 0 and int(token) == tokenizer.eos_id:
                break
            chunk = decoder.push(int(token))
            if chunk:
                yield chunk
        tail = decoder.finish()
        if tail:
            yield tail

    def _prompt(self, messages: Any) -> str:
        if isinstance(messages, str):
            return messages
        return self.prompt_builder(messages)
