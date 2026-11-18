"""Целостность контейнера `.vtm` (задача 0066): sha256 данных и меты.

Проверяем контракт целиком:

* писатель `vtm_ref.write_vtm(..., integrity=True)` кладёт в мету
  `meta["integrity"]` c `data_sha256` (хеш секции данных) и `meta_sha256`
  (хеш JSON меты, где значение хеша заменено 64 символами '0');
* Python-эталон и C-загрузчик отвергают контейнер с подменённым байтом в
  данных и в мете и грузят целый;
* `VTM_SKIP_HASH=1` отключает сверку с предупреждением (отладка);
* старый контейнер без блока `integrity` читается как раньше (совместимость).

C-часть компилирует крошечный харнесс из исходников `src/vtm/` во временном
каталоге, поэтому не зависит от собранного `src/vtm/vtm` и `main.c`.

Запуск:
    python3 -m unittest tests.test_container_hash -v
    python3 -m unittest discover -s tests
"""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import vtm_ref  # noqa: E402

VTM_DIR = os.path.join(ROOT, "src", "vtm")
CC = os.environ.get("CC", "cc")


def _model():
    tensors = [
        vtm_ref.Tensor.ternary("w", (2, 8), [1, -1, 0, 1] * 4, [1.0]),
        vtm_ref.Tensor.f32("x", (8,), [float(i) for i in range(8)]),
        vtm_ref.Tensor.f32("y", (2,), [0.0, 0.0]),
    ]
    program = [
        vtm_ref.Instr(vtm_ref.Op.TMV, dst=2, a=0, b=1),
        vtm_ref.Instr(vtm_ref.Op.HALT),
    ]
    meta = {"note": "integrity", "rope_theta": 10000.0}
    return tensors, program, meta


def _layout(data):
    """(data_start, meta_offset, meta_len) из заголовка контейнера."""
    n_tensors = struct.unpack_from("<I", data, 12)[0]
    meta_offset, meta_len = struct.unpack_from("<II", data, 24)
    data_start = vtm_ref.align_up(vtm_ref.HEADER_SIZE +
                                  vtm_ref.TENSOR_ENTRY_SIZE * n_tensors)
    return data_start, meta_offset, meta_len


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = self._tmp.name

    def path(self, name):
        return os.path.join(self.tmpdir, name)

    def good_file(self, name="good.vtm", integrity=True):
        tensors, program, meta = _model()
        path = self.path(name)
        vtm_ref.write_vtm(path, tensors, program, meta, integrity=integrity)
        with open(path, "rb") as handle:
            return path, handle.read()

    def patched(self, mutate, name):
        _path, data = self.good_file()
        mutated = bytearray(data)
        mutate(mutated)
        path = self.path(name)
        with open(path, "wb") as handle:
            handle.write(bytes(mutated))
        return path


class TestHashInMeta(_TmpCase):
    def test_writer_puts_hashes_in_meta_not_header(self):
        path, data = self.good_file()
        self.assertEqual(struct.unpack_from("<H", data, 4)[0], 2)  # версия та же
        data_start, meta_offset, meta_len = _layout(data)
        meta = json.loads(data[meta_offset:meta_offset + meta_len])
        block = meta[vtm_ref.INTEGRITY_KEY]
        self.assertEqual(block["algo"], "sha256")
        self.assertEqual(
            block["data_sha256"],
            hashlib.sha256(data[data_start:meta_offset]).hexdigest(),
        )
        # meta_sha256 считается по мете с "000…0" вместо значения.
        meta_bytes = bytearray(data[meta_offset:meta_offset + meta_len])
        pos = vtm_ref._json_string_value_offset(meta_bytes,
                                                vtm_ref.INTEGRITY_META_FIELD)
        self.assertIsNotNone(pos)
        meta_bytes[pos:pos + 64] = b"0" * 64
        self.assertEqual(block["meta_sha256"],
                         hashlib.sha256(meta_bytes).hexdigest())
        n_instr = struct.unpack_from("<I", data, 8)[0]
        self.assertEqual(os.path.getsize(path),
                         vtm_ref.align_up(meta_offset + meta_len)
                         + n_instr * vtm_ref.INSTR_SIZE)

    def test_intact_container_loads(self):
        path, _data = self.good_file()
        loaded = vtm_ref.read_vtm(path)
        self.assertEqual(loaded.meta["note"], "integrity")
        self.assertIn(vtm_ref.INTEGRITY_KEY, loaded.meta)
        # Данные не изменились: значения читаются как раньше.
        self.assertEqual(loaded.tensors[1].dense(),
                         [float(i) for i in range(8)])

    def test_data_byte_flip_rejected(self):
        def mutate(data):
            data_start, _meta_offset, _meta_len = _layout(data)
            # offset второго тензора (x) в таблице: запись №1, поле offset — 42.
            offset = struct.unpack_from("<I", data, vtm_ref.HEADER_SIZE +
                                        vtm_ref.TENSOR_ENTRY_SIZE + 42)[0]
            data[data_start + offset + 3] ^= 0xFF

        path = self.patched(mutate, "bad_data.vtm")
        with self.assertRaisesRegex(ValueError, "sha256 секции данных"):
            vtm_ref.read_vtm(path)

    def test_meta_byte_flip_rejected(self):
        def mutate(data):
            _data_start, meta_offset, _meta_len = _layout(data)
            # Меняем значение в мете, не трогая блок integrity и длину JSON.
            pos = data.find(b"10000.0", meta_offset)
            self.assertGreater(pos, 0)
            data[pos] = ord("2")

        path = self.patched(mutate, "bad_meta.vtm")
        with self.assertRaisesRegex(ValueError, "sha256 метаданных"):
            vtm_ref.read_vtm(path)

    def test_meta_hash_field_tamper_rejected(self):
        def mutate(data):
            _data_start, meta_offset, _meta_len = _layout(data)
            pos = data.find(b'"meta_sha256"', meta_offset)
            self.assertGreater(pos, 0)
            start = data.find(b'"', data.find(b":", pos) + 1) + 1
            data[start] = ord("0") if data[start] != ord("0") else ord("1")

        path = self.patched(mutate, "bad_hash.vtm")
        with self.assertRaisesRegex(ValueError, "sha256 метаданных"):
            vtm_ref.read_vtm(path)

    def test_damaged_integrity_block_rejected(self):
        def mutate(data):
            # Не hex (верхний регистр): поля есть, но хеши не распознаются.
            for field in (b'"data_sha256"', b'"meta_sha256"'):
                pos = data.find(field)
                self.assertGreater(pos, 0)
                start = data.find(b'"', data.find(b":", pos) + 1) + 1
                data[start] = ord("Z")

        path = self.patched(mutate, "damaged_block.vtm")
        with self.assertRaisesRegex(ValueError, "integrity"):
            vtm_ref.read_vtm(path)

    def test_skip_env_disables_verification(self):
        def mutate(data):
            data_start, _meta_offset, _meta_len = _layout(data)
            offset = struct.unpack_from("<I", data, vtm_ref.HEADER_SIZE +
                                        vtm_ref.TENSOR_ENTRY_SIZE + 42)[0]
            data[data_start + offset + 3] ^= 0xFF

        path = self.patched(mutate, "skip.vtm")
        old = os.environ.get(vtm_ref.INTEGRITY_SKIP_ENV)
        os.environ[vtm_ref.INTEGRITY_SKIP_ENV] = "1"
        try:
            loaded = vtm_ref.read_vtm(path)
        finally:
            if old is None:
                os.environ.pop(vtm_ref.INTEGRITY_SKIP_ENV, None)
            else:
                os.environ[vtm_ref.INTEGRITY_SKIP_ENV] = old
        self.assertIn(vtm_ref.INTEGRITY_KEY, loaded.meta)

    def test_old_container_without_integrity_still_loads(self):
        tensors, program, _meta = _model()
        path = self.path("old.vtm")
        # Контейнер до задачи 0066: меты нет вовсе.
        vtm_ref.write_vtm(path, tensors, program, {})
        with open(path, "rb") as handle:
            data = handle.read()
        _data_start, _meta_offset, meta_len = _layout(data)
        self.assertEqual(meta_len, 0)
        loaded = vtm_ref.read_vtm(path)
        self.assertEqual(loaded.meta, {})

    def test_env_enables_writing_without_flag(self):
        tensors, program, meta = _model()
        old = os.environ.get(vtm_ref.INTEGRITY_WRITE_ENV)
        os.environ[vtm_ref.INTEGRITY_WRITE_ENV] = "1"
        try:
            path = self.path("env.vtm")
            vtm_ref.write_vtm(path, tensors, program, meta)
        finally:
            if old is None:
                os.environ.pop(vtm_ref.INTEGRITY_WRITE_ENV, None)
            else:
                os.environ[vtm_ref.INTEGRITY_WRITE_ENV] = old
        self.assertIn(vtm_ref.INTEGRITY_KEY, vtm_ref.read_vtm(path).meta)

    def test_explicit_false_beats_env(self):
        tensors, program, meta = _model()
        old = os.environ.get(vtm_ref.INTEGRITY_WRITE_ENV)
        os.environ[vtm_ref.INTEGRITY_WRITE_ENV] = "1"
        try:
            path = self.path("off.vtm")
            vtm_ref.write_vtm(path, tensors, program, meta, integrity=False)
        finally:
            if old is None:
                os.environ.pop(vtm_ref.INTEGRITY_WRITE_ENV, None)
            else:
                os.environ[vtm_ref.INTEGRITY_WRITE_ENV] = old
        self.assertNotIn(vtm_ref.INTEGRITY_KEY, vtm_ref.read_vtm(path).meta)


_HARNESS = r"""
#include "vtm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    FILE *f;
    long n;
    unsigned char *buf;
    vtm_status st = VTM_OK;
    char err[256];
    vtm_t *m;

    if (argc < 2) return 2;
    f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }
    if (fseek(f, 0, SEEK_END) != 0) return 2;
    n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) return 2;
    buf = (unsigned char *)malloc((size_t)n ? (size_t)n : 1u);
    if (!buf) return 2;
    if (n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n) return 2;
    fclose(f);
    m = vtm_load_mem(buf, (size_t)n, &st, err, sizeof err);
    if (!m) {
        printf("status=%d\n", (int)st);
        fprintf(stderr, "%s\n", err);
        free(buf);
        return 1;
    }
    printf("ok tensors=%u\n", vtm_tensor_count(m));
    vtm_free(m);
    free(buf);
    return 0;
}
"""


class TestCLoader(_TmpCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which(CC):
            raise unittest.SkipTest(f"нет C-компилятора {CC!r}")
        cls._ctmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._ctmp.cleanup)
        src = os.path.join(cls._ctmp.name, "harness.c")
        with open(src, "w", encoding="utf-8") as handle:
            handle.write(_HARNESS)
        cls.exe = os.path.join(cls._ctmp.name, "harness")
        cmd = [
            CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
            "-D_POSIX_C_SOURCE=200809L", "-DVTM_NO_AVX2",
            "-I", VTM_DIR, src,
            os.path.join(VTM_DIR, "loader.c"),
            os.path.join(VTM_DIR, "interp.c"),
            os.path.join(VTM_DIR, "backend_scalar.c"),
            os.path.join(VTM_DIR, "backend_neon.c"),
            os.path.join(VTM_DIR, "trit.c"),
            "-lm", "-o", cls.exe,
        ]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(
                "харнесс C-загрузчика не собрался:\n"
                + proc.stdout + proc.stderr
            )

    def _run_c(self, path, skip_hash=False):
        env = dict(os.environ)
        env.pop("VTM_SKIP_HASH", None)
        if skip_hash:
            env["VTM_SKIP_HASH"] = "1"
        return subprocess.run([self.exe, path], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=120)

    def test_c_loads_intact_and_rejects_data_tamper(self):
        path, _data = self.good_file()
        proc = self._run_c(path)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok tensors=3", proc.stdout)

        def mutate(data):
            data_start, _meta_offset, _meta_len = _layout(data)
            offset = struct.unpack_from("<I", data, vtm_ref.HEADER_SIZE +
                                        vtm_ref.TENSOR_ENTRY_SIZE + 42)[0]
            data[data_start + offset + 3] ^= 0xFF

        bad = self.patched(mutate, "c_bad_data.vtm")
        proc = self._run_c(bad)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("status=-15", proc.stdout)  # VTM_E_INTEGRITY
        self.assertIn("sha256 секции данных", proc.stderr)
        self.assertIn("повреждён", proc.stderr)

    def test_c_rejects_meta_tamper(self):
        def mutate(data):
            _data_start, meta_offset, _meta_len = _layout(data)
            pos = data.find(b"10000.0", meta_offset)
            self.assertGreater(pos, 0)
            data[pos] = ord("2")

        bad = self.patched(mutate, "c_bad_meta.vtm")
        proc = self._run_c(bad)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        self.assertIn("status=-15", proc.stdout)
        self.assertIn("sha256 метаданных", proc.stderr)

    def test_c_skip_env_warns_and_loads(self):
        def mutate(data):
            data_start, _meta_offset, _meta_len = _layout(data)
            offset = struct.unpack_from("<I", data, vtm_ref.HEADER_SIZE +
                                        vtm_ref.TENSOR_ENTRY_SIZE + 42)[0]
            data[data_start + offset + 3] ^= 0xFF

        bad = self.patched(mutate, "c_skip.vtm")
        proc = self._run_c(bad, skip_hash=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("VTM_SKIP_HASH", proc.stderr)

    def test_c_loads_old_container_without_hash(self):
        _path, data = self.good_file("c_old.vtm", integrity=False)
        path = self.path("c_old.vtm")
        proc = self._run_c(path)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
