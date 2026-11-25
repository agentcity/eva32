#!/usr/bin/env python3
"""Контейнер v0.3: u64-смещения (ADR 0019, задача 0173).

Проверяется:

* v0.3 пишется и читается Python-эталоном: запись таблицы 60 байт,
  `offset`/`scale_offset` — u64, `meta_offset` — u64 в «резерве» заголовка,
  `program_offset` не хранится;
* `auto` пишет v0.2, пока файл < 4 ГиБ, и выбирает v0.3, когда иначе нельзя;
  явная v0.2 на большом файле отказывает с числами до создания файла;
* C-загрузчик читает v0.3 и считает токен;
* тяжёлый прогон (`VTM_U64_HEAVY=1`): контейнер 5 ГиБ, offset'ы за 4 ГиБ,
  C грузит и генерирует токен; печатаются размер, время сборки и пик RSS.

Запуск:
    python3 -m unittest tests.test_u64_container -v
    VTM_U64_HEAVY=1 python3 -m unittest tests.test_u64_container -v
"""
from __future__ import annotations

import os
import resource
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import vtm_compile, vtm_ref  # noqa: E402

VTM_BIN = os.path.join(ROOT, "src", "vtm", "vtm")
TOOLS = os.path.join(ROOT, "tools")
GIB = 1 << 30
MIB = 1 << 20


def tiny_model():
    """Крошечная модель с SAMP: C-рантайм может сгенерировать токен."""
    cfg = vtm_compile.ModelConfig(
        hidden=32, n_heads=2, n_kv_heads=1, head_dim=16,
        intermediate=64, vocab=32, n_layers=1, context_len=8,
        eps=1e-3, tie_word_embeddings=True, lm_head="mm",
    )
    embed = vtm_ref.Tensor.placeholder(
        "emb.w", (cfg.vocab, cfg.hidden), vtm_ref.KIND_TRITS
    )
    vtm = vtm_compile.compile_model(cfg, {"embed": embed})
    for tensor in vtm.tensors:
        if tensor.data is not None:
            tensor.data = None
    return vtm


def prepend_f32(vtm, dims):
    """Добавить неиспользуемые fp32-тензоры в начало таблицы.

    Программа сдвигается на число добавленных тензоров (кроме ROPE `b` —
    там число голов, а не индекс). Так маленькие тензоры и мета оказываются
    за 4 ГиБ, а считать C будет их, а не гигантские пустышки.
    """
    huge = [
        vtm_ref.Tensor.placeholder(f"huge{i}.f32", (n,), vtm_ref.KIND_F32)
        for i, n in enumerate(dims)
    ]
    shift = len(huge)

    def sh(index: int) -> int:
        return index + shift

    program = []
    for ins in vtm.program:
        b = ins.b if ins.op == vtm_ref.Op.ROPE else sh(ins.b)
        program.append(vtm_ref.Instr(
            ins.op, ins.flags, sh(ins.dst), sh(ins.a), b, sh(ins.c), ins.imm
        ))
    return huge + list(vtm.tensors), program


class TestVersionChoice(unittest.TestCase):
    def test_auto_is_v02_for_small_files(self):
        vtm = tiny_model()
        size = vtm.estimated_file_size()
        self.assertLess(size, 4 * GIB)
        self.assertEqual(vtm.estimated_file_size("auto"), size)
        self.assertEqual(vtm_ref.choose_container_version(size, "auto"), 2)
        self.assertGreater(vtm.estimated_file_size(3), size)  # запись 60 байт

    def test_auto_is_v03_when_v02_does_not_fit(self):
        vtm = vtm_ref.VTM(
            tensors=[vtm_ref.Tensor.placeholder("big.f32", (1 << 30,),
                                                vtm_ref.KIND_F32)],
            program=[vtm_ref.Instr(vtm_ref.Op.HALT)],
            meta={},
        )
        size_v02 = vtm.estimated_file_size(2)
        self.assertGreaterEqual(size_v02, 4 * GIB)
        self.assertEqual(vtm_ref.choose_container_version(size_v02, "auto"), 3)
        self.assertEqual(vtm.estimated_file_size("auto"),
                         vtm.estimated_file_size(3))
        self.assertEqual(vtm_ref.parse_container_version("v0.2"), 2)
        self.assertEqual(vtm_ref.parse_container_version("v0.3"), 3)
        with self.assertRaises(ValueError):
            vtm_ref.parse_container_version(4)
        with self.assertRaises(ValueError):
            vtm_ref.parse_container_version(True)

    def test_explicit_v02_refuses_big_file_before_writing(self):
        vtm = vtm_ref.VTM(
            tensors=[vtm_ref.Tensor.placeholder("big.f32", (1 << 30,),
                                                vtm_ref.KIND_F32)],
            program=[vtm_ref.Instr(vtm_ref.Op.HALT)],
            meta={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "big.vtm")
            with self.assertRaises(ValueError) as caught:
                vtm_ref.write_vtm_stream(path, vtm.tensors, vtm.program, {},
                                         container_version=2)
            text = str(caught.exception)
            self.assertIn("32-битными", text)
            self.assertIn("предел 4 ГиБ", text)
            self.assertIn("u64", text)
            self.assertFalse(os.path.exists(path))


class TestV03RoundTrip(unittest.TestCase):
    def test_write_read_and_layout(self):
        # Настоящие (маленькие) веса: проверяем write_vtm целиком, не только
        # заголовок потокового писателя.
        vtm = vtm_compile.make_synthetic(seed=1, size=32, mode="mlp")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v03.vtm")
            written = vtm_ref.write_vtm(
                path, vtm.tensors, vtm.program, vtm.meta,
                container_version=3, integrity=True,
            )
            self.assertEqual(written.version, 3)
            with open(path, "rb") as handle:
                head = handle.read(32)
            magic, version, flags, n_instr, n_tensors, n_program = (
                struct.unpack_from("<4sHHIII", head, 0)
            )
            self.assertEqual(magic, b"VTM1")
            self.assertEqual(version, 3)
            meta_offset, meta_len = struct.unpack_from("<QI", head, 20)
            self.assertGreater(meta_offset, 0)
            self.assertGreater(meta_len, 0)
            self.assertEqual(
                os.path.getsize(path),
                meta_offset + vtm_ref.align_up(meta_len) + n_instr * 16,
            )
            entry = 32
            with open(path, "rb") as handle:
                raw = struct.unpack_from("<24sBB4IQQH",
                                         handle.read(32 + 60 + 16), entry)
            self.assertEqual(raw[1], vtm.tensors[0].kind)
            self.assertEqual(raw[7], 0)  # первый offset — начало данных

            loaded = vtm_ref.read_vtm(path)
            self.assertEqual(loaded.version, 3)
            self.assertEqual([t.name for t in loaded.tensors],
                             [t.name for t in vtm.tensors])
            self.assertEqual(loaded.meta["integrity"]["algo"], "sha256")
            before = vtm_ref.execute(written, [], 0)
            after = vtm_ref.execute(loaded, [], 0)
            self.assertEqual(before.logits, after.logits)

    def test_stream_v03_c_runtime_counts_token(self):
        if not os.access(VTM_BIN, os.X_OK):
            self.skipTest("C-рантайм не собран: src/vtm/vtm отсутствует")
        vtm = tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stream_v03.vtm")
            written = vtm_ref.write_vtm_stream(
                path, vtm.tensors, vtm.program, vtm.meta,
                container_version=3, integrity=True,
            )
            self.assertEqual(written.version, 3)
            with open(path, "rb") as handle:
                self.assertEqual(struct.unpack_from("<H", handle.read(6), 4)[0], 3)
            proc = subprocess.run(
                [VTM_BIN, "run", path, "--tokens", "1", "--max-new", "1"],
                cwd=ROOT, capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("вывод:", proc.stdout)
            inspect = subprocess.run(
                [VTM_BIN, "inspect", path],
                cwd=ROOT, capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(inspect.returncode, 0, inspect.stderr)
            self.assertIn("тензоров:", inspect.stdout)

    def test_v03_header_via_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cli_v03.vtm")
            proc = subprocess.run(
                [sys.executable, os.path.join(TOOLS, "vtm_compile.py"),
                 "--synthetic", "--container-version", "3", "--out", path],
                cwd=ROOT, capture_output=True, text=True, timeout=300,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(path, "rb") as handle:
                self.assertEqual(struct.unpack_from("<H", handle.read(6), 4)[0], 3)


@unittest.skipUnless(
    os.environ.get("VTM_U64_HEAVY") == "1",
    "тяжёлый прогон: VTM_U64_HEAVY=1 (5 ГиБ на диске)",
)
class TestFiveGiBContainer(unittest.TestCase):
    def test_five_gib_c_loads_and_counts_token(self):
        if not os.access(VTM_BIN, os.X_OK):
            self.skipTest("C-рантайм не собран: src/vtm/vtm отсутствует")
        needed = 7 * GIB
        if shutil.disk_usage(tempfile.gettempdir()).free < needed:
            self.skipTest(f"мало места: нужно ~{needed / GIB:.0f} ГиБ")

        vtm = tiny_model()
        tensors, program = prepend_f32(vtm, (1 << 30, 1 << 28))  # 4 ГиБ + 1 ГиБ
        model = vtm_ref.VTM(tensors=tensors, program=program, meta=vtm.meta)
        self.assertGreater(model.estimated_file_size(2), 4 * GIB)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "u64_5gib.vtm")
            started = time.perf_counter()
            written = vtm_ref.write_vtm_stream(
                path, tensors, program, vtm.meta,
                container_version="auto", integrity=False, chunk=8 * MIB,
            )
            build_s = time.perf_counter() - started
            size = os.path.getsize(path)
            self.assertEqual(written.version, 3)
            self.assertGreater(size, 4 * GIB)
            self.assertGreater(size, 5 * GIB)
            # offset маленького тензора (после двух гигантов) — за 4 ГиБ.
            small = next(t for t in written.tensors if t.name == "emb.w")
            self.assertGreater(small.offset, 4 * GIB)
            with open(path, "rb") as handle:
                handle.seek(32)
                raw = struct.unpack("<24sBB4IQQH", handle.read(60))
                self.assertEqual(raw[7], 0)  # huge0.f32
            before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            run = subprocess.run(
                [VTM_BIN, "run", path, "--tokens", "1", "--max-new", "1"],
                cwd=ROOT, capture_output=True, text=True, timeout=600,
            )
            after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("вывод:", run.stdout)
            # ru_maxrss на macOS — байты, на Linux — КиБ.
            peak = after if sys.platform == "darwin" else after * 1024
            print(
                f"\n5 ГиБ v0.3: файл {size / GIB:.2f} ГиБ, "
                f"сборка {build_s:.2f} с, пик RSS C-рантайма "
                f"{peak / MIB:.1f} МиБ, вывод: {run.stdout.strip().splitlines()[-2]}",
            )
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
