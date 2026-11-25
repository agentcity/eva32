#!/usr/bin/env python3
"""Тесты больших форм и KV-бюджета компилятора (задача 0135).

Проверяется то, что до этого не было валидировано:

* KV-кэш 7B/14B/32B (224/768/1024 МиБ при ctx 2048) против кэпа загрузчика
  512 МиБ: компилятор сам урезает `context_len` 2048 → 1024 → 512 и печатает
  предупреждение с числами; если не влезает даже 512 — `ValueError` с
  командой пересборки `-DVTM_MAX_KV_BYTES=...`;
* синтетика трёх больших форм собирается (`make_synthetic_shape`), в мете и
  таблице тензоров правильные размерности, KV-кэши урезаны, а программа и
  таблица не зависят от размера: счётчики в заголовке — u32, u16 только у
  индексов тензоров в инструкции (запас 32B: 1415 из 65535);
* 32B-раскладка (~6,9 ГиБ тритов) не влезает в контейнер v0.2 (смещения u32,
  предел 4 ГиБ): компилятор считает граф и KV, но писатель отказывает с
  числами и вариантами (7B/14B влезают);
* потоковый писатель `vtm_ref.write_vtm_stream` (нужен, чтобы 32B ≈ 7 ГиБ
  не поднимать в памяти) даёт валидный контейнер: Python-эталон и C-рантайм
  его читают и считают токен.

Тяжёлый прогон «собрать 7B/14B целиком, загрузить C и прогнать токен; 32B —
граф + внятный отказ по 4 ГиБ» (≈4,5 ГиБ на диске, ~минута) — под
`VTM_COMPILE_LARGE_HEAVY=1`:

    VTM_COMPILE_LARGE_HEAVY=1 python3 -m unittest tests.test_compile_large -v

Запуск:
    python3 -m unittest tests.test_compile_large
"""
from __future__ import annotations

import contextlib
import functools
import io
import os
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import vtm_compile, vtm_ref  # noqa: E402

TOOLS = os.path.join(ROOT, "tools")
VTM_BIN = os.path.join(ROOT, "src", "vtm", "vtm")
MIB = 1 << 20

#: Формы задачи 0135 и их KV-кэш при ctx 2048 (2 × слоёв × ctx × kv_dim × 4).
EXPECTED_KV_2048 = {
    "1.5b": 112 * MIB,
    "7b": 224 * MIB,
    "14b": 768 * MIB,
    "32b": 1024 * MIB,
}
#: KV-кэш после урезания под кэп 512 МиБ (ctx 1024 для 14B/32B).
EXPECTED_CONTEXT = {"1.5b": 2048, "7b": 2048, "14b": 1024, "32b": 1024}
EXPECTED_KV_FITTED = {"1.5b": 112 * MIB, "7b": 224 * MIB,
                      "14b": 384 * MIB, "32b": 512 * MIB}


def shape_cfg(name: str) -> vtm_compile.ModelConfig:
    return vtm_compile.ModelConfig(**vtm_compile.SYNTHETIC_SHAPES[name])


def quiet_stderr() -> contextlib.AbstractContextManager:
    return contextlib.redirect_stderr(io.StringIO())


@functools.lru_cache(maxsize=None)
def build_shape(name: str) -> tuple[vtm_ref.VTM, str]:
    """Собрать синтетику формы один раз на процесс; вернуть (vtm, stderr)."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        vtm = vtm_compile.make_synthetic_shape(name)
    return vtm, err.getvalue()


class TestKvBudget(unittest.TestCase):
    """Чистая арифметика KV-кэша и решение по `context_len`."""

    def test_kv_bytes_of_known_forms(self):
        # Числа задачи: 7B — 224, 14B — 768, 32B — 1024 МиБ при ctx 2048.
        for name, expected in EXPECTED_KV_2048.items():
            cfg = shape_cfg(name)
            self.assertEqual(vtm_compile.kv_cache_bytes(cfg), expected, name)
            self.assertEqual(vtm_compile.kv_cache_bytes(cfg, 2048), expected, name)

    def test_fit_does_not_touch_small_forms(self):
        for name in ("1.5b", "7b"):
            cfg, info = vtm_compile.fit_context_window(shape_cfg(name))
            self.assertEqual(cfg.context_len, 2048, name)
            self.assertFalse(info["reduced"], name)
            self.assertEqual(info["chosen"], 2048, name)
            self.assertEqual(info["kv_bytes"], EXPECTED_KV_2048[name], name)

    def test_fit_shrinks_14b_and_32b_to_1024(self):
        for name in ("14b", "32b"):
            cfg, info = vtm_compile.fit_context_window(shape_cfg(name))
            self.assertEqual(cfg.context_len, EXPECTED_CONTEXT[name], name)
            self.assertTrue(info["reduced"], name)
            self.assertEqual(info["chosen"], 1024, name)
            self.assertEqual(info["kv_bytes"], EXPECTED_KV_FITTED[name], name)
            self.assertEqual(info["requested_kv_bytes"], EXPECTED_KV_2048[name], name)
            # Кэп инклюзивный: 32B при ctx 1024 — ровно 512 МиБ и влезает.
            self.assertLessEqual(info["kv_bytes"], info["max_bytes"], name)

    def test_cap_is_inclusive_on_the_boundary(self):
        # 32B: ctx 1024 — ровно 512 МиБ; на байт меньше кэп уже не пускает
        # и компилятор обязан уйти на 512 (256 МиБ).
        cfg, info = vtm_compile.fit_context_window(
            shape_cfg("32b"), max_bytes=512 * MIB
        )
        self.assertEqual(cfg.context_len, 1024)
        cfg, info = vtm_compile.fit_context_window(
            shape_cfg("32b"), max_bytes=512 * MIB - 1
        )
        self.assertEqual(cfg.context_len, 512)
        self.assertEqual(info["kv_bytes"], 256 * MIB)

    def test_fit_never_increases_context(self):
        # 256 МиБ: ctx 1024 (384 МиБ) не влезает, 512 (192 МиБ) влезает.
        cfg, info = vtm_compile.fit_context_window(
            shape_cfg("14b"), max_bytes=256 * MIB
        )
        self.assertEqual(cfg.context_len, 512)  # меньше 512 не предлагаем
        self.assertEqual(info["chosen"], 512)
        # Запросили 8192 — шаги 2048/1024; 2048 не влезает, 1024 влезает.
        cfg, _ = vtm_compile.fit_context_window(
            vtm_compile.ModelConfig(
                **{**vtm_compile.SYNTHETIC_SHAPES["14b"], "context_len": 8192}
            )
        )
        self.assertEqual(cfg.context_len, 1024)
        # Запросили 512 — урезать некуда и не нужно.
        cfg, _ = vtm_compile.fit_context_window(
            vtm_compile.ModelConfig(
                **{**vtm_compile.SYNTHETIC_SHAPES["14b"], "context_len": 512}
            )
        )
        self.assertEqual(cfg.context_len, 512)

    def test_warning_has_the_numbers_and_rebuild_command(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            cfg = shape_cfg("14b")
            vtm = vtm_compile.compile_model(cfg)
        text = err.getvalue()
        self.assertIn("768.0 МиБ", text)       # было при ctx 2048
        self.assertIn("512.0 МиБ", text)       # кэп VTM_MAX_KV_BYTES
        self.assertIn("384.0 МиБ", text)       # стало при ctx 1024
        self.assertIn("context_len уменьшен до 1024", text)
        self.assertIn("-DVTM_MAX_KV_BYTES=805306368", text)
        # Решение видно и без stderr: в мете контейнера.
        self.assertEqual(vtm.meta["context_len"], 1024)
        self.assertEqual(vtm.meta["context_window"], 1024)
        self.assertTrue(vtm.meta["kv_cache"]["reduced"])
        self.assertEqual(vtm.meta["kv_cache"]["bytes"], 384 * MIB)

    def test_error_when_even_512_does_not_fit(self):
        with self.assertRaises(ValueError) as caught:
            vtm_compile.compile_model(shape_cfg("32b"), max_kv_bytes=64 * MIB)
        text = str(caught.exception)
        self.assertIn("64.0 МиБ", text)                     # кэп
        self.assertIn("1.0 ГиБ", text)                      # запрошенный ctx 2048
        self.assertIn("256.0 МиБ", text)                    # даже ctx 512
        self.assertIn("-DVTM_MAX_KV_BYTES=1073741824", text)  # хватит на 2048
        self.assertIn("make clean", text)

    def test_meta_kv_cache_matches_tensors(self):
        for name, expected_ctx in EXPECTED_CONTEXT.items():
            vtm, _ = build_shape(name)
            caches = [t for t in vtm.tensors if t.is_kv_cache]
            cfg = shape_cfg(name)
            self.assertEqual(len(caches), 2 * cfg.n_layers, name)
            expected = (expected_ctx, cfg.kv_dim)
            for tensor in caches:
                self.assertEqual(tuple(tensor.dims), expected, tensor.name)
            kv = vtm.meta["kv_cache"]
            self.assertEqual(kv["bytes"], EXPECTED_KV_FITTED[name], name)
            self.assertEqual(kv["context_len"], expected_ctx, name)
            self.assertEqual(
                vtm.meta["model"]["context_len"], expected_ctx, name
            )
            self.assertEqual(
                ("context_window" in vtm.meta), expected_ctx != 2048, name
            )


class TestShapeGraphs(unittest.TestCase):
    """Граф большой формы: размеры, счётчики, независимость от размера."""

    def test_three_shapes_compile(self):
        for name, layers in (("7b", 28), ("14b", 48), ("32b", 64)):
            vtm, warning = build_shape(name)
            # 22 тензора на слой + emb/x/final_norm/hfinal/logits/x.col/token;
            # 17 инструкций на слой + EMB/RMSN/CPY/MM/SAMP/HALT.
            self.assertEqual(len(vtm.tensors), 22 * layers + 7, name)
            self.assertEqual(len(vtm.program), 17 * layers + 6, name)
            if name == "7b":
                self.assertEqual(warning, "", "7B (224 МиБ) влезает в кэп")
            else:
                self.assertIn(
                    "-DVTM_MAX_KV_BYTES=", warning,
                    f"{name}: урезание ctx должно печататься с командой пересборки",
                )

    def test_tensor_table_and_program_do_not_depend_on_size(self):
        # Заголовок: n_instr/n_tensors/n_program — u32 (формат "<4sHHIII").
        # Это и есть ответ на «нет u16-счётчиков»: u16 — только индексы
        # тензоров в инструкции (dst/a/b/c), и они ограничены 65535.
        packed = vtm_ref._HEADER_STRUCT.pack(
            vtm_ref.VTM_MAGIC, vtm_ref.VTM_VERSION, 0, 70_000, 65_535, 70_000 * 16
        )
        header = vtm_ref._HEADER_STRUCT.unpack(packed)
        self.assertEqual((header[3], header[4], header[5]), (70_000, 65_535, 1_120_000))
        self.assertEqual(vtm_ref.MAX_TENSORS, 0xFFFF)
        with self.assertRaises(ValueError):
            vtm_ref.Instr(vtm_ref.Op.HALT, dst=0x10000)

        for name in ("7b", "14b", "32b"):
            vtm, _ = build_shape(name)
            self.assertLess(len(vtm.tensors), vtm_ref.MAX_TENSORS, name)
            self.assertLess(len(vtm.program), 2 ** 32, name)
            max_operand = max(
                max(instr.dst, instr.a, instr.b, instr.c) for instr in vtm.program
            )
            self.assertLess(max_operand, 0x10000, name)
            # Запас: 32B — 1415 тензоров из 65535.
            self.assertGreater(
                vtm_ref.MAX_TENSORS - len(vtm.tensors), 40 * len(vtm.tensors), name
            )

    def test_program_is_identical_for_different_context(self):
        # ctx влияет только на dims KV-кэшей и мету; байткод тот же.
        small, _ = build_shape("32b")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            big = vtm_compile.make_synthetic_shape("32b", context_len=512)
        self.assertEqual(
            [instr.pack() for instr in small.program],
            [instr.pack() for instr in big.program],
        )
        self.assertEqual(
            [t.name for t in small.tensors], [t.name for t in big.tensors]
        )
        small_cache = next(t for t in small.tensors if t.name == "l0.kcache")
        big_cache = next(t for t in big.tensors if t.name == "l0.kcache")
        self.assertEqual(tuple(small_cache.dims), (1024, 1024))
        self.assertEqual(tuple(big_cache.dims), (512, 1024))

    def test_estimated_sizes_grow_with_shape(self):
        sizes = {}
        for name in ("1.5b", "7b", "14b", "32b"):
            vtm, _ = build_shape(name)
            sizes[name] = vtm.estimated_file_size()
        self.assertLess(sizes["1.5b"], sizes["7b"])
        self.assertLess(sizes["7b"], sizes["14b"])
        self.assertLess(sizes["14b"], sizes["32b"])
        self.assertGreater(sizes["32b"], 6 * 1024 ** 3)
        self.assertGreater(sizes["14b"], 3 * 1024 ** 3)
        self.assertGreater(sizes["7b"], 1 * 1024 ** 3)

    def test_32b_auto_chooses_v03_and_v02_still_refuses(self):
        # 32B-раскладка (~6,9 ГиБ тритов в packed20) не влезает в v0.2
        # (u32-смещения, 4 ГиБ). Задача 0173: auto выбирает v0.3 (u64), а
        # принудительная v0.2 обязана отказать с числами — до записи файла.
        vtm, _ = build_shape("32b")
        size_v02 = vtm.estimated_file_size(2)
        self.assertGreater(size_v02, 2 ** 32)
        self.assertEqual(vtm.estimated_file_size("auto"),
                         vtm.estimated_file_size(3))
        self.assertLess(build_shape("14b")[0].estimated_file_size(2), 2 ** 32)
        self.assertLess(build_shape("7b")[0].estimated_file_size(2), 2 ** 32)
        self.assertEqual(
            vtm_ref.choose_container_version(size_v02, "auto"), 3
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "32b.vtm")
            with self.assertRaises(ValueError) as caught:
                vtm_ref.write_vtm_stream(
                    path, vtm.tensors, vtm.program, vtm.meta,
                    container_version=2,
                )
            text = str(caught.exception)
            self.assertIn("32-битными", text)
            self.assertIn(f"{size_v02} байт", text)
            self.assertIn("предел 4 ГиБ", text)
            self.assertIn("u64", text)
            # Отказ — до открытия файла: мусора на диске не осталось.
            self.assertFalse(os.path.exists(path))

    def test_unknown_shape_lists_known_forms(self):
        with self.assertRaises(ValueError) as caught:
            vtm_compile.make_synthetic_shape("70b")
        for name in vtm_compile.SYNTHETIC_SHAPES:
            self.assertIn(name, str(caught.exception))

    def test_known_arch_recognises_large_forms(self):
        for name in ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct",
                     "Qwen/Qwen2.5-32B-Instruct"):
            self.assertIn(name, vtm_compile._KNOWN_ARCH)


class TestStreamWriter(unittest.TestCase):
    """`write_vtm_stream`: контейнер без весов читается Python и C."""

    def _model(self, context_len: int = 8):
        cfg = vtm_compile.ModelConfig(
            hidden=32, n_heads=2, n_kv_heads=1, head_dim=16,
            intermediate=64, vocab=32, n_layers=2, context_len=context_len,
            eps=1e-3, tie_word_embeddings=True, lm_head="mm",
        )
        embed = vtm_ref.Tensor.placeholder(
            "emb.w", (cfg.vocab, cfg.hidden), vtm_ref.KIND_TRITS
        )
        with quiet_stderr():
            vtm = vtm_compile.compile_model(cfg, {"embed": embed})
        for tensor in vtm.tensors:
            if tensor.data is not None:
                tensor.data = None
        return vtm

    def test_roundtrip_layout_and_zero_logits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stream.vtm")
            vtm = self._model()
            written = vtm_ref.write_vtm_stream(
                path, vtm.tensors, vtm.program, vtm.meta, integrity=True
            )
            self.assertEqual(os.path.getsize(path), written.estimated_file_size())
            loaded = vtm_ref.read_vtm(path)
            self.assertEqual(len(loaded.tensors), len(vtm.tensors))
            self.assertEqual(
                [t.name for t in loaded.tensors], [t.name for t in vtm.tensors]
            )
            self.assertEqual(
                [instr.pack() for instr in loaded.program],
                [instr.pack() for instr in vtm.program],
            )
            self.assertEqual(loaded.meta["integrity"]["algo"], "sha256")
            for tensor in loaded.tensors:
                if tensor.is_kv_cache:
                    self.assertEqual(tuple(tensor.dims), (8, 16))
                elif tensor.kind == vtm_ref.KIND_TRITS:
                    # Слова pack20 из нулей дают триты -1, но масштабы 0,
                    # поэтому все веса равны нулю.
                    self.assertEqual(set(tensor.scales), {0.0})
                elif tensor.kind == vtm_ref.KIND_F32:
                    self.assertEqual(set(tensor.data), {0.0})
            result = vtm_ref.execute(loaded, [1], 0)
            self.assertEqual(result.logits, [0.0] * 32)
            self.assertEqual(vtm_ref._argmax(result.logits), 0)

    def test_c_runtime_loads_streamed_container(self):
        if not os.access(VTM_BIN, os.X_OK):
            self.skipTest("C-рантайм не собран: src/vtm/vtm отсутствует")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stream.vtm")
            vtm = self._model()
            vtm_ref.write_vtm_stream(
                path, vtm.tensors, vtm.program, vtm.meta, integrity=True
            )
            proc = subprocess.run(
                [VTM_BIN, "run", path, "--tokens", "1", "--max-new", "1"],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0 and "версия контейнера" in proc.stderr:
                self.skipTest("C-рантайм устарел: " + proc.stderr.strip())
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn(f"тензоров: {len(vtm.tensors)}", proc.stdout)
            self.assertIn("вывод:", proc.stdout)
            line = next(
                (line for line in proc.stdout.splitlines()
                 if line.startswith("вывод:")), ""
            )
            self.assertEqual(line.split(":", 1)[1].split(), ["0"])

    def test_c_loads_more_than_255_tensors_and_instructions(self):
        # 15 слоёв дают 337 тензоров и 261 инструкцию: если бы счётчики были
        # u8, контейнер бы не прочитался. Это проверка «нет счётчиков u8» на
        # живом C-загрузчике (предел таблицы — u16-индексы, 65535).
        if not os.access(VTM_BIN, os.X_OK):
            self.skipTest("C-рантайм не собран: src/vtm/vtm отсутствует")
        cfg = vtm_compile.ModelConfig(
            hidden=16, n_heads=2, n_kv_heads=1, head_dim=8,
            intermediate=32, vocab=16, n_layers=15, context_len=8,
            eps=1e-3, tie_word_embeddings=True, lm_head="mm",
        )
        embed = vtm_ref.Tensor.placeholder(
            "emb.w", (cfg.vocab, cfg.hidden), vtm_ref.KIND_TRITS
        )
        with quiet_stderr():
            vtm = vtm_compile.compile_model(cfg, {"embed": embed})
        for tensor in vtm.tensors:
            if tensor.data is not None:
                tensor.data = None
        self.assertGreater(len(vtm.tensors), 255)
        self.assertGreater(len(vtm.program), 255)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "many.vtm")
            vtm_ref.write_vtm_stream(
                path, vtm.tensors, vtm.program, vtm.meta, integrity=True
            )
            proc = subprocess.run(
                [VTM_BIN, "run", path, "--tokens", "1", "--max-new", "1"],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0 and "версия контейнера" in proc.stderr:
                self.skipTest("C-рантайм устарел: " + proc.stderr.strip())
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn(f"тензоров: {len(vtm.tensors)}", proc.stdout)
            self.assertIn(f"инструкций: {len(vtm.program)}", proc.stdout)
            self.assertIn("вывод:", proc.stdout)

    def test_stream_writer_refuses_to_drop_real_weights(self):
        # Лучше ошибка, чем молча записать нули вместо весов.
        tensors = [vtm_ref.Tensor.f32("w", (4,), [1.0, 2.0, 3.0, 4.0])]
        program = [vtm_ref.Instr(vtm_ref.Op.HALT)]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError) as caught:
                vtm_ref.write_vtm_stream(
                    os.path.join(tmp, "w.vtm"), tensors, program, {"format": "vtm"}
                )
            self.assertIn("пишет нули", str(caught.exception))
            self.assertIn("write_vtm", str(caught.exception))

    def test_kv_cap_is_enforced_by_both_writers(self):
        # 140000 × 1024 × 4 = 572 МиБ > 512 МиБ: оба писателя обязаны
        # отказать и позвать пересборку с -DVTM_MAX_KV_BYTES.
        tensors = [vtm_ref.Tensor.kv_cache("kc", (140_000, 1024))]
        program = [vtm_ref.Instr(vtm_ref.Op.HALT)]
        for writer in (vtm_ref.write_vtm, vtm_ref.write_vtm_stream):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "cap.vtm")
                with self.assertRaises(ValueError) as caught:
                    writer(path, tensors, program, {"format": "vtm"})
                self.assertIn("-DVTM_MAX_KV_BYTES=", str(caught.exception))
                self.assertIn(
                    str(vtm_ref.MAX_KV_BYTES), str(caught.exception)
                )


class TestCliShape(unittest.TestCase):
    """CLI: --shape без гигантских файлов (ошибки и план)."""

    def _run(self, *args, timeout: int = 300):
        return subprocess.run(
            [sys.executable, os.path.join(TOOLS, "vtm_compile.py"), *args],
            cwd=ROOT, capture_output=True, text=True, timeout=timeout,
        )

    def test_shape_requires_synthetic(self):
        proc = self._run("--shape", "7b", "--qwen-config", "x.yaml")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--shape работает только с --synthetic", proc.stderr)

    def test_ctx_requires_shape(self):
        proc = self._run("--synthetic", "--ctx", "512", "--out", "/tmp/never.vtm")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--ctx работает только с --synthetic --shape", proc.stderr)

    def test_14b_plan_prints_reduction(self):
        proc = self._run("--synthetic", "--shape", "14b")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("VTM синтетика формы 14b", proc.stdout)
        self.assertIn("KV-кэш: 384.0 МиБ при context_len=1024", proc.stdout)
        self.assertIn("урезано с context_len=2048", proc.stdout)
        # Граф построен, но файл не писан.
        self.assertIn("файл не записан", proc.stdout)
        self.assertIn("768.0 МиБ", proc.stderr)  # предупреждение

    def test_32b_plan_warns_it_does_not_fit(self):
        proc = self._run("--synthetic", "--shape", "32b")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("VTM синтетика формы 32b", proc.stdout)
        self.assertIn("KV-кэш: 512.0 МиБ при context_len=1024", proc.stdout)
        self.assertIn("оценка размера: 6.9 ГиБ (контейнер v0.3)", proc.stdout)
        self.assertIn("v0.2 не влезает", proc.stdout)
        self.assertIn("предел 4 ГиБ", proc.stdout)
        self.assertIn("u64", proc.stdout)

    def test_budget_too_small_is_a_clear_error(self):
        proc = self._run(
            "--synthetic", "--shape", "32b", "--max-kv-bytes", str(64 * MIB)
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("KV-кэш не влезает", proc.stderr)
        self.assertIn("-DVTM_MAX_KV_BYTES=1073741824", proc.stderr)


@unittest.skipUnless(
    os.environ.get("VTM_COMPILE_LARGE_HEAVY") == "1",
    "тяжёлый прогон: VTM_COMPILE_LARGE_HEAVY=1 (≈4,5 ГиБ и минута)",
)
class TestLargeEndToEnd(unittest.TestCase):
    """7B/14B/32B целиком: сборка, размер, время, запуск 1 токена в C."""

    def test_shapes_build_load_and_run_one_token(self):
        if not os.access(VTM_BIN, os.X_OK):
            self.skipTest("C-рантайм не собран: src/vtm/vtm отсутствует")
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("7b", "14b"):
                path = os.path.join(tmp, f"synth_{name}.vtm")
                started = time.perf_counter()
                proc = subprocess.run(
                    [sys.executable, os.path.join(TOOLS, "vtm_compile.py"),
                     "--synthetic", "--shape", name, "--out", path],
                    cwd=ROOT, capture_output=True, text=True, timeout=3600,
                )
                build_s = time.perf_counter() - started
                self.assertEqual(proc.returncode, 0, proc.stderr)
                size = os.path.getsize(path)
                self.assertGreater(size, 1 << 30, name)
                self.assertIn(f"VTM синтетика формы {name}", proc.stdout)

                run = subprocess.run(
                    [VTM_BIN, "run", path, "--tokens", "1", "--max-new", "1"],
                    cwd=ROOT, capture_output=True, text=True, timeout=3600,
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("вывод:", run.stdout)
                results.append((name, size, build_s))
                os.unlink(path)

            # 32B: auto выбрал бы v0.3 (~6,9 ГиБ), но проверяем отказ
            # принудительной v0.2 — CLI обязан объяснить числами, а не
            # падать стектрейсом; файла на диске не остаётся.
            started = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, os.path.join(TOOLS, "vtm_compile.py"),
                 "--synthetic", "--shape", "32b", "--container-version", "2",
                 "--out", os.path.join(tmp, "synth_32b.vtm")],
                cwd=ROOT, capture_output=True, text=True, timeout=3600,
            )
            graph_s = time.perf_counter() - started
            self.assertEqual(proc.returncode, 2, proc.stdout)
            self.assertIn("32-битными", proc.stderr)
            self.assertIn("предел 4 ГиБ", proc.stderr)
            self.assertEqual(
                [item for item in os.listdir(tmp) if item.endswith(".vtm")], []
            )
        for name, size, build_s in results:
            print(f"{name}: файл {size / 2**30:.2f} ГиБ, сборка {build_s:.1f} с")
        print(f"32b: граф собран, v0.2 отвергнута (4 ГиБ), попытка {graph_s:.1f} с")


if __name__ == "__main__":
    unittest.main()
