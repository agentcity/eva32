"""Спекулятивный декод ВТМ: черновик 0.5B проверяет цель 3B одной пачкой.

Зачем
=====

Декод 3B упирается в память: на каждый токен читается 1,98 ГБ весов (задача
0201 — 20,4 ток/с живьём). Один из способов отдать больше токенов за то же
чтение — спекулятивный декод: маленькая модель (Qwen2.5-0.5B) предлагает K
токенов, целевая считает их **одним** вызовом ``vtm_run_batch()`` и даёт
argmax на каждой позиции; совпавший префикс принимается, к нему добавляется
токен цели, при расхождении — откат на принятый префикс. На greedy выход
обязан совпасть с обычным декодом токен в токен.

Как это устроено здесь
======================

* **C-помощник** :file:`tools/vtm_spec_probe.c` — тонкая обвязка над
  ``vtm_load()``/``vtm_run()``/``vtm_run_batch()``. Публичный C-ABI
  (``src/vtm/api.h``) не отдаёт argmax на каждой позиции пачки (у него
  ``emit`` только для генерации, а в машине после пачки остаются логиты
  последнего токена), поэтому помощник читает историю SAMP машины
  (``samp_hist`` в ``vtm_internal.h``): для жадного SAMP это ровно argmax
  каждой позиции. ``src/vtm`` при этом не меняется.
* **Откат** — это ``vtm_set_position()`` назад: KV-кэш адресуется глобальной
  позицией, ATTN читает только позиции до текущей, а KVW перезаписывает
  строку при следующем токене. Штраф за повторы принудительно выключен
  (``force_greedy``: ``repetition_window = 0``), поэтому лишние записи в
  ``samp_hist`` на выход не влияют; помощник умеет сохранять/восстанавливать
  историю (``vtm_spec_hist_*``), но декодер этим не пользуется.
* Сэмплинг принудительно жадный: мета Qwen2.5-3B и Qwen3-30B — ``greedy``,
  и паритет проверяется именно на нём. Температура — отвергнута (см.
  ``docs/vtm.md``): rejection sampling требует вероятностей, а C-помощник
  отдаёт только argmax.

Формула раунда (K = ``k``, целевая позиция p — последний принятый токен):

* черновик уже дал ``D`` (argmax на p+1), цель — ``B``; если ``B != D``,
  принимается только ``B``, обе машины кормят его и раунд кончается;
* иначе ``B`` принимается, черновик генерирует ещё ``K-1`` кандидатов
  ``D1..D_{K-1}`` (K-1 форвардов), цель считает пачку
  ``[B, D1, ..., D_{K-1}]`` (K токенов на позициях p+1..p+K) и отдаёт
  ``y[0..K-1]``: ``y[i]`` проверяет ``D_{i+1}``, ``y[K-1]`` — бонусный токен
  цели. Принимается префикс ``D1..Dj``, к нему добавляется бонус ``y[j]``.

Замеры (M1 Pro, 16 ГиБ, машина под нагрузкой ~11) — в ``docs/vtm.md``;
главный вывод честный: на этой машине спекуляция **проигрывает** обычному
декоду (0,5–0,8×), потому что (1) 0,5B-черновик здесь всего вдвое дешевле
3B на токен, (2) батчевые ядра `kind 6/7` дают ~80 Г MAC/с — столько же,
сколько поштучный декод и так снимает с памяти. Выигрыш есть только там,
где пачка реально дешевле: 30B MoE (эксперт читается один раз на пачку).

Публичный API
=============

    from eva.speculative import SpeculativeDecoder

    dec = SpeculativeDecoder("runs/q4/0.5b-q4.vtm", "runs/q4/3b-q4k.vtm", k=4)
    ids = dec.generate(prompt_ids, max_new_tokens=32)
    print(dec.stats.acceptance_rate)
    dec.close()

CLI для замеров::

    python3 -m eva.speculative --draft runs/q4/0.5b-q4.vtm \\
        --target runs/q4/3b-q4k.vtm --prompt "Столица Франции — это" \\
        --max-new 32 --k 4 --backend simd
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

__all__ = [
    "SpecError",
    "SpeculativeDecoder",
    "SpecStats",
    "SpecUnavailable",
    "build_probe",
    "probe_path",
]

ROOT = Path(__file__).resolve().parent.parent
PROBE_SOURCE = ROOT / "tools" / "vtm_spec_probe.c"
STATIC_LIBRARY = ROOT / "src" / "vtm" / "libvtm.a"

_ENV_PROBE = "EVA_VTM_SPEC_PROBE"
_ENV_CC = "EVA_VTM_CC"
_ENV_BACKEND = "EVA_VTM_BACKEND"
_ENV_SPEC_BACKEND = "EVA_VTM_SPEC_BACKEND"


class SpecUnavailable(RuntimeError):
    """Нет ``cc``/``libvtm.a``/исходника помощника — спекуляция невозможна."""


class SpecError(RuntimeError):
    """Ошибка машины/помощника (текст — из ``vtm_spec_error``)."""


# --------------------------------------------------------------------------- #
# сборка C-помощника
# --------------------------------------------------------------------------- #

def _dylib_suffix() -> str:
    return ".dylib" if sys.platform == "darwin" else ".so"


def _probe_digest(cc: str) -> str:
    parts = [cc, sys.platform]
    for path in (PROBE_SOURCE, STATIC_LIBRARY):
        stat = path.stat()
        parts.append(f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}")
    if hasattr(os, "uname"):
        parts.append(os.uname().machine)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def probe_path(*, cc: str | None = None) -> Path:
    """Путь к собранному помощнику (без сборки)."""
    compiler = cc or os.environ.get(_ENV_CC, "cc")
    cache = Path(tempfile.gettempdir()) / f"eva-vtm-spec-{getattr(os, 'getuid', lambda: 0)()}"
    return cache / f"libvtmspec-{_probe_digest(compiler)}{_dylib_suffix()}"


def build_probe(*, cc: str | None = None, force: bool = False) -> Path:
    """Собрать ``tools/vtm_spec_probe.c`` в разделяемую библиотеку.

    Кэш — во временном каталоге, ключ — mtime/size исходника, ``libvtm.a``,
    компилятора и платформы (как ``build_shared_library`` в
    ``eva/backends/vtm_backend.py``, но с нашим .c).
    """
    configured = os.environ.get(_ENV_PROBE)
    if configured and not force:
        path = Path(configured)
        if not path.is_file():
            raise SpecUnavailable(f"{_ENV_PROBE}: нет файла {path}")
        return path
    compiler = cc or os.environ.get(_ENV_CC, "cc")
    if shutil.which(compiler) is None:
        raise SpecUnavailable(
            f"компилятор {compiler!r} не найден; задайте {_ENV_CC} или "
            f"{_ENV_PROBE} (готовую библиотеку)"
        )
    if not PROBE_SOURCE.is_file():
        raise SpecUnavailable(f"нет исходника помощника: {PROBE_SOURCE}")
    if not STATIC_LIBRARY.is_file():
        raise SpecUnavailable(
            f"нет статической библиотеки ВТМ: {STATIC_LIBRARY} "
            "(соберите `make -C src/vtm libvtm.a`)"
        )
    target = probe_path(cc=compiler)
    if target.is_file() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    obj = target.with_suffix(".o")
    base = [compiler, "-std=c99", "-O2", "-I", str(ROOT / "src" / "vtm")]
    compile_cmd = base + ["-c", str(PROBE_SOURCE), "-o", str(obj)]
    macos = sys.platform == "darwin"
    link_cmd = [compiler, "-dynamiclib" if macos else "-shared", "-o", str(target), str(obj)]
    if macos:
        link_cmd += ["-Wl,-all_load", str(STATIC_LIBRARY),
                     "-framework", "Metal", "-framework", "Foundation", "-lobjc"]
    else:
        link_cmd += ["-Wl,--whole-archive", str(STATIC_LIBRARY),
                     "-Wl,--no-whole-archive"]
    link_cmd += ["-lm", "-lpthread"]
    for command, what in ((compile_cmd, "компиляция"), (link_cmd, "линковка")):
        try:
            proc = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=300.0, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SpecUnavailable(f"{what} помощника: {exc}") from exc
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", "replace").strip()
            raise SpecUnavailable(
                f"{what} помощника: код {proc.returncode}; {stderr[-600:] or '<пусто>'}"
            )
    try:
        obj.unlink(missing_ok=True)
    except OSError:
        pass
    return target


# --------------------------------------------------------------------------- #
# ctypes-обвязка
# --------------------------------------------------------------------------- #

class _Library:
    """Загруженный помощник + сигнатуры (одна на процесс)."""

    def __init__(self, path: Path) -> None:
        self.path = str(path)
        # Спекуляции нужен samp_hist по КАЖДОМУ токену пачки: ускорение
        # «lm_head только для последнего» (0215) его не заполняет. Флаг
        # ставим до загрузки библиотеки — C читает его при разметке пачки.
        os.environ.setdefault("VTM_SAMP_HIST_ALL", "1")
        try:
            self.cdll = ctypes.CDLL(self.path)
        except OSError as exc:
            raise SpecUnavailable(f"не загрузить {path}: {exc}") from exc
        c = self.cdll
        c.vtm_spec_open.restype = ctypes.c_void_p
        c.vtm_spec_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p,
                                    ctypes.c_char_p, ctypes.c_int]
        c.vtm_spec_close.argtypes = [ctypes.c_void_p]
        c.vtm_spec_error.restype = ctypes.c_char_p
        c.vtm_spec_error.argtypes = [ctypes.c_void_p]
        c.vtm_spec_clear_error.argtypes = [ctypes.c_void_p]
        c.vtm_spec_backend.restype = ctypes.c_char_p
        c.vtm_spec_backend.argtypes = [ctypes.c_void_p]
        c.vtm_spec_force_greedy.restype = ctypes.c_int
        c.vtm_spec_force_greedy.argtypes = [ctypes.c_void_p]
        c.vtm_spec_prefill.restype = ctypes.c_int
        c.vtm_spec_prefill.argtypes = [ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.c_int32),
                                       ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_int32)]
        c.vtm_spec_feed.restype = ctypes.c_int
        c.vtm_spec_feed.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                    ctypes.POINTER(ctypes.c_int32)]
        c.vtm_spec_batch.restype = ctypes.c_int
        c.vtm_spec_batch.argtypes = [ctypes.c_void_p,
                                     ctypes.POINTER(ctypes.c_int32),
                                     ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_int32)]
        c.vtm_spec_rewind.restype = ctypes.c_int
        c.vtm_spec_rewind.argtypes = [ctypes.c_void_p, ctypes.c_int]
        c.vtm_spec_reset.restype = ctypes.c_int
        c.vtm_spec_reset.argtypes = [ctypes.c_void_p]
        c.vtm_spec_position.restype = ctypes.c_int
        c.vtm_spec_position.argtypes = [ctypes.c_void_p]
        c.vtm_spec_generated.restype = ctypes.c_int
        c.vtm_spec_generated.argtypes = [ctypes.c_void_p]
        c.vtm_spec_hist_len.restype = ctypes.c_int
        c.vtm_spec_hist_len.argtypes = [ctypes.c_void_p]
        c.vtm_spec_sampling.restype = ctypes.c_int
        c.vtm_spec_sampling.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 5
        c.vtm_spec_instr_count.restype = ctypes.c_int
        c.vtm_spec_instr_count.argtypes = [ctypes.c_void_p]
        c.vtm_spec_context_len.restype = ctypes.c_int
        c.vtm_spec_context_len.argtypes = [ctypes.c_void_p]


_LIBRARIES: dict[str, _Library] = {}


def _library() -> _Library:
    path = build_probe()
    key = str(path)
    lib = _LIBRARIES.get(key)
    if lib is None:
        lib = _Library(path)
        _LIBRARIES[key] = lib
    return lib


def _as_ids(ids: Sequence[int]) -> Any:
    return (ctypes.c_int32 * len(ids))(*[int(x) for x in ids])


class SpecModel:
    """Одна машина ВТМ, открытая помощником (draft или target)."""

    def __init__(self, path: str | os.PathLike[str], *, backend: str = "auto",
                 library: _Library | None = None) -> None:
        self.path = str(path)
        self.backend_name = backend
        self._lib = library or _library()
        err = ctypes.create_string_buffer(512)
        handle = self._lib.cdll.vtm_spec_open(
            self.path.encode("utf-8"), backend.encode("utf-8") or None, err, 512
        )
        if not handle:
            raise SpecError(f"{path}: {err.value.decode('utf-8', 'replace')}")
        self._handle = ctypes.c_void_p(handle)

    # -- жизненный цикл ----------------------------------------------------
    def close(self) -> None:
        if self._handle is not None:
            self._lib.cdll.vtm_spec_close(self._handle)
            self._handle = None

    def __enter__(self) -> "SpecModel":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover — защита от утечки
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- свойства ----------------------------------------------------------
    @property
    def backend(self) -> str:
        return self._lib.cdll.vtm_spec_backend(self._handle).decode("utf-8", "replace")

    @property
    def position(self) -> int:
        return int(self._lib.cdll.vtm_spec_position(self._handle))

    @property
    def instr_count(self) -> int:
        return int(self._lib.cdll.vtm_spec_instr_count(self._handle))

    @property
    def context_len(self) -> int:
        return int(self._lib.cdll.vtm_spec_context_len(self._handle))

    def sampling(self) -> dict[str, Any]:
        temp = ctypes.c_double()
        top_k = ctypes.c_int()
        top_p = ctypes.c_double()
        rep = ctypes.c_double()
        window = ctypes.c_int()
        ok = self._lib.cdll.vtm_spec_sampling(
            self._handle, ctypes.byref(temp), ctypes.byref(top_k),
            ctypes.byref(top_p), ctypes.byref(rep), ctypes.byref(window),
        )
        if not ok:
            return {}
        return {
            "temperature": temp.value,
            "top_k": top_k.value,
            "top_p": top_p.value,
            "repetition_penalty": rep.value,
            "repetition_window": window.value,
        }

    # -- ошибки ------------------------------------------------------------
    def _error(self) -> str:
        raw = self._lib.cdll.vtm_spec_error(self._handle)
        return raw.decode("utf-8", "replace") if raw else "неизвестная ошибка"

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise SpecError(f"{what}: {self._error()}")
        self._lib.cdll.vtm_spec_clear_error(self._handle)

    # -- операции ----------------------------------------------------------
    def force_greedy(self) -> None:
        self._check(self._lib.cdll.vtm_spec_force_greedy(self._handle),
                    "force_greedy")

    def reset(self) -> None:
        self._check(self._lib.cdll.vtm_spec_reset(self._handle), "reset")

    def prefill(self, prompt_ids: Sequence[int]) -> int:
        if not prompt_ids:
            raise SpecError("prefill: пустой промпт")
        ids = _as_ids(prompt_ids)
        out = ctypes.c_int32()
        self._check(
            self._lib.cdll.vtm_spec_prefill(self._handle, ids, len(prompt_ids),
                                            ctypes.byref(out)),
            "prefill",
        )
        return int(out.value)

    def feed(self, token: int) -> int:
        out = ctypes.c_int32()
        self._check(self._lib.cdll.vtm_spec_feed(self._handle, int(token),
                                                 ctypes.byref(out)), "feed")
        return int(out.value)

    def batch(self, ids: Sequence[int]) -> list[int]:
        if not ids:
            raise SpecError("batch: пустой список")
        arr = _as_ids(ids)
        out = (ctypes.c_int32 * len(ids))()
        self._check(
            self._lib.cdll.vtm_spec_batch(self._handle, arr, len(ids), out),
            "batch",
        )
        return [int(x) for x in out]

    def rewind(self, position: int) -> None:
        self._check(self._lib.cdll.vtm_spec_rewind(self._handle, int(position)),
                    "rewind")


# --------------------------------------------------------------------------- #
# статистика и декодер
# --------------------------------------------------------------------------- #

@dataclass
class SpecStats:
    """Счётчики одного прогона (``generate`` сбрасывает их в начале)."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    rounds: int = 0            # раундов, где пошли в пачку
    rejected_first: int = 0    # отказов на первом же токене (B != D)
    batch_calls: int = 0
    batch_tokens: int = 0      # сколько токенов проверено пачками
    drafts_proposed: int = 0
    drafts_accepted: int = 0
    accepted_prefix_sum: int = 0
    target_batch_s: float = 0.0
    draft_s: float = 0.0
    target_single_s: float = 0.0
    elapsed_s: float = 0.0
    cpu_s: float = 0.0         # process_time() за прогон (CPU, не wall)

    @property
    def acceptance_rate(self) -> float:
        if not self.drafts_proposed:
            return 0.0
        return self.drafts_accepted / self.drafts_proposed

    @property
    def mean_accepted_prefix(self) -> float:
        return (self.accepted_prefix_sum / self.rounds) if self.rounds else 0.0

    @property
    def tokens_per_s(self) -> float:
        return (self.generated_tokens / self.elapsed_s) if self.elapsed_s > 0 else 0.0

    @property
    def tokens_per_batch(self) -> float:
        return (self.generated_tokens / self.batch_calls) if self.batch_calls else 0.0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            acceptance_rate=self.acceptance_rate,
            mean_accepted_prefix=self.mean_accepted_prefix,
            tokens_per_s=self.tokens_per_s,
            tokens_per_batch=self.tokens_per_batch,
        )
        return data


class SpeculativeDecoder:
    """Две машины ВТМ в одном процессе: жадный спекулятивный декод.

    ``k`` — сколько токенов цель проверяет одной пачкой (включая последний
    принятый токен; полезных кандидатов из пачки ``k-1``). ``k >= 2``.
    """

    def __init__(
        self,
        draft: str | os.PathLike[str],
        target: str | os.PathLike[str],
        *,
        k: int = 4,
        backend: str | None = None,
        eos_token_id: int = -1,
        library: _Library | None = None,
    ) -> None:
        if isinstance(k, bool) or int(k) < 2:
            raise ValueError(f"k: ожидалось целое >= 2, получено {k!r}")
        self.k = int(k)
        self.eos_token_id = int(eos_token_id)
        self.backend = (
            backend
            or os.environ.get(_ENV_SPEC_BACKEND)
            or os.environ.get(_ENV_BACKEND)
            or "simd"   # паритет с поштучным декодом: батч Q4_K побитов
        )
        self.stats = SpecStats()
        self._stop = False
        lib = library or _library()
        self.draft = SpecModel(draft, backend=self.backend, library=lib)
        try:
            self.target = SpecModel(target, backend=self.backend, library=lib)
        except Exception:
            self.draft.close()
            raise
        for model in (self.draft, self.target):
            model.force_greedy()

    def close(self) -> None:
        for model in (getattr(self, "draft", None), getattr(self, "target", None)):
            if model is not None:
                model.close()

    def __enter__(self) -> "SpeculativeDecoder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def cancel(self) -> None:
        """Мягкая отмена: проверится между раундами (и в конце генерации)."""
        self._stop = True

    # -- генерация ---------------------------------------------------------
    def iter_tokens(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 64,
    ) -> Iterator[int]:
        """Генерация по одному токену (стриминг); ``generate`` — её список."""
        prompt = [int(x) for x in prompt_ids]
        if not prompt:
            raise ValueError("generate: пустой промпт")
        if max_new_tokens <= 0:
            return
        self._stop = False
        stats = SpecStats(prompt_tokens=len(prompt))
        self.stats = stats
        started = time.perf_counter()
        cpu_started = time.process_time()
        produced = 0

        def alive() -> bool:
            return not self._stop and produced < max_new_tokens

        try:
            t0 = time.perf_counter()
            pending_target = self.target.prefill(prompt)
            stats.target_single_s += time.perf_counter() - t0
            t0 = time.perf_counter()
            pending_draft = self.draft.prefill(prompt)
            stats.draft_s += time.perf_counter() - t0

            while alive():
                token = pending_target
                produced += 1
                yield token
                if self.eos_token_id >= 0 and token == self.eos_token_id:
                    return
                if not alive():
                    return
                if token != pending_draft:
                    # Черновик разошёлся сразу. Цель отдала свой токен; кормим
                    # его обеим машинам — следующий токен цели станет pending.
                    stats.rejected_first += 1
                    t0 = time.perf_counter()
                    pending_target = self.target.feed(token)
                    stats.target_single_s += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    pending_draft = self.draft.feed(token)
                    stats.draft_s += time.perf_counter() - t0
                    continue

                # Согласие: принимаем pending_target и проверяем ещё K-1
                # кандидатов черновика одной пачкой цели.
                cands: list[int] = []
                t0 = time.perf_counter()
                cur = self.draft.feed(token)
                cands.append(cur)
                for _ in range(1, self.k - 1):
                    cur = self.draft.feed(cur)
                    cands.append(cur)
                stats.draft_s += time.perf_counter() - t0
                stats.rounds += 1
                stats.drafts_proposed += len(cands)

                t0 = time.perf_counter()
                verified = self.target.batch([token] + cands)
                stats.target_batch_s += time.perf_counter() - t0
                stats.batch_calls += 1
                stats.batch_tokens += 1 + len(cands)

                # verified[i] — argmax цели после B,D1..D_i; кандидат D_{i+1}
                # (cands[i]) стоит на следующей позиции, поэтому его проверяет
                # verified[i]. Последний verified — бонусный токен цели.
                accepted = 0
                for i, cand in enumerate(cands):
                    if verified[i] != cand:
                        break
                    accepted += 1
                stats.drafts_accepted += accepted
                stats.accepted_prefix_sum += accepted
                bonus = verified[accepted]

                # Цель: последний принятый токен (B или D_accepted) на позиции
                # p+1+accepted; после пачки позиция p+K. Отвергнутые строки KV
                # перезапишет следующая пачка.
                self.target.rewind(self.target.position - (self.k - 1 - accepted))
                if accepted == len(cands):
                    # Принят и последний кандидат D_{K-1}: черновик должен его
                    # скормить, тогда его pending — D_K, а у цели на той же
                    # позиции лежит бонус (сравнятся в начале итерации).
                    pending_draft = self.draft.feed(cands[-1])
                else:
                    # Откат на принятый префикс; следующий кандидат черновика
                    # при этом состоянии уже вычислен — это cands[accepted].
                    self.draft.rewind(
                        self.draft.position - (len(cands) - 1 - accepted)
                    )
                    pending_draft = cands[accepted]
                pending_target = bonus

                for cand in cands[:accepted]:
                    produced += 1
                    yield cand
                    if self.eos_token_id >= 0 and cand == self.eos_token_id:
                        return
                    if not alive():
                        return
                # bonus станет pending_target и будет выдан в начале следующей
                # итерации — ровно один раз (он же первый токен следующей пачки).
        finally:
            stats.elapsed_s = time.perf_counter() - started
            stats.cpu_s = time.process_time() - cpu_started
            stats.generated_tokens = produced

    def generate(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 64,
        on_token: Callable[[int], None] | None = None,
    ) -> list[int]:
        out: list[int] = []
        for token in self.iter_tokens(prompt_ids, max_new_tokens=max_new_tokens):
            out.append(token)
            if on_token is not None:
                on_token(token)
        return out

    def stream_ids(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int | None = None,
        on_token: Callable[[int], None] | None = None,
        **_ignored: Any,
    ) -> Iterator[int]:
        """Совместим с ``VtmBackend.stream_ids``: по одному токену, сразу."""
        for token in self.iter_tokens(
            prompt_ids,
            max_new_tokens=64 if max_new_tokens is None else int(max_new_tokens),
        ):
            if on_token is not None:
                on_token(token)
            yield token

    def target_only(
        self,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 64,
        on_token: Callable[[int], None] | None = None,
    ) -> list[int]:
        """Обычный жадный декод той же цели (эталон для паритета и ток/с)."""
        prompt = [int(x) for x in prompt_ids]
        out: list[int] = []
        token = self.target.prefill(prompt)
        while len(out) < max_new_tokens:
            out.append(token)
            if on_token is not None:
                on_token(token)
            if self.eos_token_id >= 0 and token == self.eos_token_id:
                break
            if len(out) >= max_new_tokens:
                break
            token = self.target.feed(token)
        return out


# --------------------------------------------------------------------------- #
# CLI замеров
# --------------------------------------------------------------------------- #

def _format_stats(stats: SpecStats, *, k: int) -> str:
    return (
        f"спекуляция K={k}: {stats.generated_tokens} ток за {stats.elapsed_s:.2f} с "
        f"({stats.tokens_per_s:.1f} ток/с), принято {stats.drafts_accepted}/"
        f"{stats.drafts_proposed} ({100.0 * stats.acceptance_rate:.0f}%), "
        f"средний префикс {stats.mean_accepted_prefix:.2f}, "
        f"пачка {stats.batch_calls}×{stats.batch_tokens / max(1, stats.batch_calls):.1f} ток, "
        f"цель {stats.target_batch_s:.2f}+{stats.target_single_s:.2f} с, "
        f"черновик {stats.draft_s:.2f} с, CPU {stats.cpu_s:.2f} с "
        f"({stats.cpu_s / max(1, stats.generated_tokens) * 1000:.0f} мс/ток)"
    )


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m eva.speculative",
        description="Замер спекулятивного декода ВТМ: target-only против spec.",
    )
    parser.add_argument("--draft", required=True, help=".vtm черновика (0.5B)")
    parser.add_argument("--target", required=True, help=".vtm цели (3B/30B)")
    parser.add_argument("--prompt", default="Столица Франции — это",
                        help="текст промпта (кодируется --tokenizer)")
    parser.add_argument("--tokenizer", default=str(ROOT / "runs" / "q4" / "tokenizer.json"))
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--backend", default=None,
                        help="auto|simd|scalar|hybrid|metal (по умолчанию "
                             "EVA_VTM_SPEC_BACKEND, затем EVA_VTM_BACKEND, затем simd)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    from eva.backends.tokenizer import BPETokenizer

    tokenizer = BPETokenizer.from_file(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt)
    decoder = SpeculativeDecoder(args.draft, args.target, k=args.k,
                                 backend=args.backend,
                                 eos_token_id=int(getattr(tokenizer, "eos_id", -1)))
    result: dict[str, Any] = {"k": args.k, "prompt_tokens": len(prompt_ids)}
    try:
        # target-only в том же процессе и с тем же жадным сэмплером
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        base = decoder.target_only(prompt_ids, max_new_tokens=args.max_new)
        base_cpu = time.process_time() - cpu0
        base_s = time.perf_counter() - t0
        result["target_only"] = {
            "tokens": len(base),
            "seconds": base_s,
            "cpu_seconds": base_cpu,
            "tokens_per_s": len(base) / base_s if base_s else 0.0,
        }
        spec = decoder.generate(prompt_ids, max_new_tokens=args.max_new)
        result["speculative"] = decoder.stats.as_dict()
        result["speculative"]["seconds"] = decoder.stats.elapsed_s
        result["parity"] = spec == base
        result["text"] = tokenizer.decode(spec)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(f"промпт: {len(prompt_ids)} ток · K={args.k} · backend={decoder.target.backend}")
            print(f"target-only: {len(base)} ток за {base_s:.2f} с "
                  f"({len(base) / base_s:.2f} ток/с), CPU {base_cpu:.2f} с")
            print(_format_stats(decoder.stats, k=args.k))
            speed = decoder.stats.tokens_per_s / max(
                1e-9, result["target_only"]["tokens_per_s"]
            )
            print(f"к target-only: {speed:.2f}× "
                  f"(>1 — быстрее, <1 — медленнее)")
            print(f"паритет (greedy): {'да' if result['parity'] else 'НЕТ'}")
            print(f"текст: {result['text']!r}")
    finally:
        decoder.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
