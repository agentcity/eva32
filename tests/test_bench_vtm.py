#!/usr/bin/env python3
"""Тесты ``tools/bench_vtm.py`` — без C, без сети и без больших моделей.

Проверяется всё, что можно проверить на Python-уровне:

* разбор вывода ``vtm run`` (включая «пусто», запятую как разделитель и
  отсутствие строки времени);
* сборка команд ``run``/``inspect``/``--synthetic`` (порядок аргументов,
  ``--backend``, ``--trace``);
* медианы/таблица/JSON-отчёт;
* подготовка контейнера: расширение KV-кэша под длинный контекст и вставка
  ``SAMP`` перед ``HALT`` — на настоящем ``.vtm``, собранном ``vtm_ref``;
* разбивка по операциям: дублирование инструкций, разбор trace;
* mock-запуск через подмену ``subprocess.run`` и через инжектированный runner —
  ровно так бенчмарк проверяется в CI без собранного рантайма.

Запуск:
    python3 -m unittest tests.test_bench_vtm -v
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import bench_vtm, vtm_ref  # noqa: E402

VOCAB = 16
HIDDEN = 8


# ---------------------------------------------------------------------------
# Крошечный контейнер: TMV + RMSN (+SAMP) на настоящем vtm_ref
# ---------------------------------------------------------------------------


def write_tiny_model(
    path: str,
    *,
    context: int = 4,
    with_samp: bool = False,
    with_cache: bool = True,
    integrity: bool | None = None,
) -> str:
    """Собрать валидный v0.2-контейнер без C-рантайма.

    Граф: ``RMSN → TMV → [SAMP] → HALT``. Тритовый ``w`` — ``[vocab, hidden]``,
    ``logits`` — вектор словаря, так что формы правдоподобны, хотя C в тестах
    не запускается.
    """
    tensors = [
        vtm_ref.Tensor.f32(
            "emb.w", (VOCAB, HIDDEN), [0.01 * index for index in range(VOCAB * HIDDEN)]
        ),
        vtm_ref.Tensor.f32("x", (HIDDEN,), [0.1] * HIDDEN),
        vtm_ref.Tensor.f32("norm.w", (HIDDEN,), [1.0] * HIDDEN),
        vtm_ref.Tensor.ternary(
            "w",
            (VOCAB, HIDDEN),
            [((index % 3) - 1) for index in range(VOCAB * HIDDEN)],
        ),
        vtm_ref.Tensor.f32("logits", (VOCAB,), [0.0] * VOCAB),
        vtm_ref.Tensor.f32("token", (1,), [0.0]),
    ]
    if with_cache:
        tensors.append(
            vtm_ref.Tensor.f32(
                "l0.kcache", (context, HIDDEN), [0.0] * (context * HIDDEN)
            )
        )
        tensors.append(
            vtm_ref.Tensor.f32(
                "l0.vcache", (context, HIDDEN), [0.0] * (context * HIDDEN)
            )
        )
    index = {tensor.name: position for position, tensor in enumerate(tensors)}
    program = [
        vtm_ref.Instr(op=int(vtm_ref.Op.RMSN), dst=index["x"], a=index["x"],
                      b=index["norm.w"], imm=1),
        vtm_ref.Instr(op=int(vtm_ref.Op.TMV), dst=index["logits"], a=index["w"],
                      b=index["x"]),
    ]
    if with_samp:
        program.append(
            vtm_ref.Instr(op=int(vtm_ref.Op.SAMP), dst=index["token"],
                          a=index["logits"], imm=0)
        )
    program.append(vtm_ref.Instr(op=int(vtm_ref.Op.HALT)))
    meta = {"model": {"context_len": context}}
    params = inspect.signature(vtm_ref.write_vtm).parameters
    if integrity is not None and "integrity" in params:
        vtm_ref.write_vtm(path, tensors, program, meta, integrity=integrity)
    else:
        vtm_ref.write_vtm(path, tensors, program, meta)
    return path


def _proc(command, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(list(command), returncode, stdout, stderr)


def fake_run_stdout(command, *, time_ms=0.9, backend="scalar-lut243"):
    """Правдоподобный stdout ``vtm run`` для mock-раннера."""
    tokens: list[int] = []
    max_new = 0
    if "--tokens" in command:
        tokens = [int(part) for part in command[command.index("--tokens") + 1].split(",")]
    if "--max-new" in command:
        max_new = int(command[command.index("--max-new") + 1])
    generated = [((index * 3) + 1) % VOCAB for index in range(max_new)]
    token_part = ""
    if max_new:
        rate = max_new * 1000.0 / time_ms if time_ms else 0.0
        token_part = f", {max_new} токен(ов), {rate:.1f} ток/с"
    return "\n".join(
        [
            f"модель: {command[2]}",
            f"тензоров: 6, инструкций: 3, бэкенд: {backend}",
            "prompt: " + " ".join(str(token) for token in tokens),
            "вывод:   " + (" ".join(str(token) for token in generated) if generated else "(пусто)"),
            f"время: {time_ms:.3f} мс{token_part}",
        ]
    ) + "\n"


class FakeRunner:
    """Подмена процесса: help/inspect/run по виду команды, без C."""

    def __init__(
        self,
        *,
        time_ms: float = 0.9,
        trace_supported: bool = False,
        trace_lines: str | None = None,
        fail: bool = False,
        inspect_wall: bool = True,
    ):
        self.time_ms = time_ms
        self.trace_supported = trace_supported
        self.trace_lines = trace_lines
        self.fail = fail
        self.inspect_wall = inspect_wall
        self.calls: list[list[str]] = []

    def __call__(self, command, timeout):
        command = list(command)
        self.calls.append(command)
        if "--help" in command:
            text = "VTM\n  vtm run model.vtm --tokens 1,2,3 --max-new 8\n"
            if self.trace_supported:
                text += "  --trace  печатать время по операциям\n"
            return _proc(command, 0, text, "")
        if len(command) > 1 and command[1] == "inspect":
            return _proc(
                command,
                0,
                "файл: x\nтензоров: 6, инструкций: 3, бэкенд: scalar-lut243\n",
                "",
            )
        if self.fail:
            return _proc(command, 1, "", "vtm run: что-то сломалось\n")
        text = fake_run_stdout(command, time_ms=self.time_ms)
        if "--trace" in command and self.trace_lines:
            text += self.trace_lines
        return _proc(command, 0, text, "")

    def run_calls(self) -> list[list[str]]:
        return [
            call
            for call in self.calls
            if len(call) > 1 and call[1] == "run"
        ]


# ---------------------------------------------------------------------------
# Разбор вывода
# ---------------------------------------------------------------------------


class TestParseRunOutput(unittest.TestCase):
    def test_full_output(self):
        stdout = (
            "модель: demo/model.vtm\n"
            "тензоров: 29, инструкций: 22, бэкенд: simd-neon\n"
            "prompt: 1 2 3\n"
            "вывод:   4 113 60\n"
            "время: 1.234 мс, 3 токен(ов), 2431.6 ток/с\n"
        )
        parsed = bench_vtm.parse_run_output(stdout)
        self.assertAlmostEqual(parsed.time_ms, 1.234)
        self.assertEqual(parsed.prompt, [1, 2, 3])
        self.assertEqual(parsed.generated, [4, 113, 60])
        self.assertEqual(parsed.backend, "simd-neon")
        self.assertEqual(parsed.tensor_count, 29)
        self.assertEqual(parsed.instr_count, 22)
        self.assertAlmostEqual(parsed.tokens_per_s, 2431.6)

    def test_without_generation(self):
        stdout = (
            "модель: m.vtm\n"
            "тензоров: 2, инструкций: 1, бэкенд: scalar-lut243\n"
            "prompt: 1\n"
            "вывод:  (пусто)\n"
            "время: 0.500 мс\n"
        )
        parsed = bench_vtm.parse_run_output(stdout)
        self.assertAlmostEqual(parsed.time_ms, 0.5)
        self.assertEqual(parsed.generated, [])
        self.assertIsNone(parsed.tokens_per_s)

    def test_comma_decimal_and_no_counters(self):
        parsed = bench_vtm.parse_run_output("время: 1,5 мс, 2 токен(ов), 1333,3 ток/с\n")
        self.assertAlmostEqual(parsed.time_ms, 1.5)
        self.assertAlmostEqual(parsed.tokens_per_s, 1333.3)
        self.assertIsNone(parsed.backend)

    def test_missing_time_is_error(self):
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.parse_run_output("модель: m.vtm\nвывод:  1\n")

    def test_bad_token_list_is_error(self):
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.parse_run_output(
                "prompt: 1 два 3\nвывод:  1\nвремя: 1.0 мс\n"
            )

    def test_not_text_is_error(self):
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.parse_run_output(b"bytes")  # type: ignore[arg-type]

    def test_stream_output_has_no_time_in_stdout(self):
        # В --stream служебные строки идут в stderr; stdout — только текст
        # ответа, поэтому parse_run_output обязан ругаться, а не выдумывать.
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.parse_run_output("привет")


# ---------------------------------------------------------------------------
# Команды
# ---------------------------------------------------------------------------


class TestCommands(unittest.TestCase):
    def test_run_command(self):
        command = bench_vtm.build_run_command("/x/vtm", "m.vtm", [1, 2, 3], 4)
        self.assertEqual(command[:3], ["/x/vtm", "run", "m.vtm"])
        self.assertEqual(command[command.index("--tokens") + 1], "1,2,3")
        self.assertEqual(command[command.index("--max-new") + 1], "4")
        self.assertNotIn("--backend", command)
        self.assertNotIn("--trace", command)

    def test_run_command_backend_and_trace(self):
        command = bench_vtm.build_run_command(
            "vtm", "m.vtm", [7], 1, backend="simd", trace=True
        )
        self.assertEqual(command[command.index("--backend") + 1], "simd")
        self.assertIn("--trace", command)

    def test_run_command_rejects_bad_args(self):
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.build_run_command("vtm", "m.vtm", [], 1)
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.build_run_command("vtm", "m.vtm", [1], -1)
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.build_run_command("vtm", "m.vtm", [1], 1, backend="cuda")

    def test_inspect_command(self):
        self.assertEqual(
            bench_vtm.build_inspect_command("/x/vtm", "m.vtm"),
            ["/x/vtm", "inspect", "m.vtm"],
        )

    def test_synthetic_command_flags(self):
        command = bench_vtm.build_synthetic_command(
            "python3", "/x/vtm_compile.py", "/w/s.vtm", size=256, seed=11, mode="attn"
        )
        self.assertEqual(command[1], "/x/vtm_compile.py")
        self.assertIn("--synthetic", command)
        self.assertEqual(command[command.index("--size") + 1], "256")
        self.assertEqual(command[command.index("--seed") + 1], "11")
        self.assertEqual(command[command.index("--mode") + 1], "attn")
        self.assertEqual(command[command.index("--out") + 1], "/w/s.vtm")


class TestIntegrityContainers(unittest.TestCase):
    """Производные контейнеры не должны терять/ломать sha256 (задача 0066)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workdir = os.path.join(self.tmp.name, "work")
        os.makedirs(self.workdir)

    @unittest.skipUnless(
        "integrity" in inspect.signature(vtm_ref.write_vtm).parameters,
        "vtm_ref без поддержки integrity (до 0066)",
    )
    def test_prepare_recomputes_hash(self):
        path = write_tiny_model(
            os.path.join(self.tmp.name, "hash.vtm"), context=4, integrity=True
        )
        source = vtm_ref.read_vtm(path)  # проверяет sha256
        self.assertIn("integrity", source.meta)
        prepared = bench_vtm.prepare_container(path, 64, self.workdir)
        loaded = vtm_ref.read_vtm(prepared)  # старый хеш здесь бы не сошёлся
        self.assertIn("integrity", loaded.meta)
        self.assertGreaterEqual(bench_vtm.container_context(loaded), 64)
        self.assertEqual(loaded.op_counts().get("SAMP"), 1)

    @unittest.skipUnless(
        "integrity" in inspect.signature(vtm_ref.write_vtm).parameters,
        "vtm_ref без поддержки integrity (до 0066)",
    )
    def test_variant_write_keeps_container_readable(self):
        path = write_tiny_model(
            os.path.join(self.tmp.name, "hash2.vtm"), context=8, with_samp=True,
            integrity=True,
        )
        vtm = vtm_ref.read_vtm(path)
        variant = bench_vtm.duplicate_instructions(vtm.program, ("TMV",), 2)
        target = os.path.join(self.workdir, "variant.vtm")
        bench_vtm.write_container(target, vtm, program=variant)
        loaded = vtm_ref.read_vtm(target)
        self.assertEqual(len(loaded.program), len(vtm.program) + 2)
        self.assertIn("integrity", loaded.meta)

    @unittest.skipUnless(
        "integrity" in inspect.signature(vtm_ref.write_vtm).parameters,
        "vtm_ref без поддержки integrity (до 0066)",
    )
    def test_write_container_can_drop_integrity(self):
        path = write_tiny_model(
            os.path.join(self.tmp.name, "hash3.vtm"), context=8, with_samp=True,
            integrity=True,
        )
        vtm = vtm_ref.read_vtm(path)
        target = os.path.join(self.workdir, "nohash.vtm")
        bench_vtm.write_container(target, vtm, integrity=False)
        loaded = vtm_ref.read_vtm(target)
        self.assertNotIn("integrity", loaded.meta)

    def test_write_container_without_integrity_support(self):
        """Старый vtm_ref: блок integrity вырезается, контейнер остаётся валидным."""
        path = write_tiny_model(os.path.join(self.tmp.name, "legacy.vtm"), context=8)
        vtm = vtm_ref.read_vtm(path)
        vtm.meta["integrity"] = {"algo": "sha256", "data_sha256": "0" * 64,
                                 "meta_sha256": "0" * 64}
        target = os.path.join(self.workdir, "legacy2.vtm")
        fake_signature = mock.Mock()
        fake_signature.parameters = {"path": None, "tensors": None, "program": None}
        with mock.patch.object(bench_vtm.inspect, "signature",
                               return_value=fake_signature):
            bench_vtm.write_container(target, vtm)
        loaded = vtm_ref.read_vtm(target)
        self.assertNotIn("integrity", loaded.meta)
        self.assertEqual(len(loaded.program), len(vtm.program))


# ---------------------------------------------------------------------------
# Статистика и таблицы
# ---------------------------------------------------------------------------


class TestStatsAndTables(unittest.TestCase):
    def test_summarize(self):
        stats = bench_vtm.summarize([3.0, 1.0, 2.0])
        self.assertEqual(stats["median"], 2.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 3.0)
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["mean"], 2.0)

    def test_summarize_even_count_median(self):
        self.assertEqual(bench_vtm.summarize([1.0, 2.0, 3.0, 4.0])["median"], 2.5)

    def test_summarize_empty(self):
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.summarize([])

    def test_format_table_widths_and_mismatch(self):
        text = bench_vtm.format_table(
            ["a", "bb"], [["1", "2"], ["333", "4"]], aligns=("<", ">")
        )
        lines = text.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[1].startswith("---"))
        self.assertIn("333", lines[3])
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.format_table(["a", "b"], [["1"]])

    def _scenario(self, **overrides):
        row = {
            "context": 128,
            "gen": 1,
            "backend": "simd-neon",
            "backend_requested": "simd",
            "forwards": 128,
            "time_ms": {"median": 6.4},
            "ms_per_token": 6.4,
            "tokens_per_s": 156.25,
            "ms_per_step": 0.05,
            "wall_ms": {"median": 10.0},
            "overhead_ms": 3.6,
            "budget_ok": None,
        }
        row.update(overrides)
        return row

    def test_scenario_table_without_budget(self):
        text = bench_vtm.scenario_table([self._scenario()], None)
        self.assertIn("контекст", text)
        self.assertIn("simd-neon", text)
        self.assertIn("—", text)

    def test_scenario_table_budget_cells(self):
        rows = [
            self._scenario(budget_ok=True),
            self._scenario(backend="scalar-lut243", budget_ok=False),
        ]
        text = bench_vtm.scenario_table(rows, 33.3)
        self.assertIn("ok", text)
        self.assertIn("ПРЕВЫШЕН", text)

    def test_breakdown_table_noise_and_residual(self):
        rows = [
            {
                "cluster": "TMV",
                "instructions": 8,
                "total_ms": 3.5,
                "ms_per_step": 0.027,
                "share_pct": 56.0,
            },
            {
                "cluster": "KVW",
                "instructions": 1,
                "total_ms": -0.01,
                "ms_per_step": -0.0001,
                "share_pct": -0.1,
                "within_noise": True,
            },
        ]
        text = bench_vtm.breakdown_table(rows, 0.3)
        self.assertIn("TMV", text)
        self.assertIn("≈0", text)
        self.assertIn("остаток", text)

    def test_format_report_contains_verdict_and_budget(self):
        report = {
            "model": {
                "path": "m.vtm",
                "file_bytes": 1024,
                "tensors": 6,
                "instructions": 3,
                "context": 8,
                "vocab": 16,
                "ops": {"TMV": 1, "HALT": 1},
                "source": "m.vtm",
            },
            "cli": {"vtm_bin": "/x/vtm", "trace": False},
            "host": {
                "machine": "arm64",
                "platform": "test",
                "python": "3.14",
                "cpu_count": 2,
            },
            "config": {"repeat": 1, "gen": [1], "contexts": [128]},
            "load": {"inspect_wall_ms": 3.0},
            "scenarios": [self._scenario()],
            "breakdown": [
                {
                    "method": "duplication",
                    "note": bench_vtm.BREAKDOWN_NOTE,
                    "scenario": {"context": 128, "gen": 1, "backend": "simd-neon"},
                    "base_time_ms": 6.4,
                    "covered_ms": 6.0,
                    "residual_share_pct": 6.2,
                    "residual_ms": 0.4,
                    "rows": [
                        {
                            "cluster": "TMV",
                            "instructions": 8,
                            "total_ms": 3.5,
                            "ms_per_step": 0.027,
                            "share_pct": 56.0,
                        }
                    ],
                }
            ],
            "crosscheck": {
                "context": 128,
                "backend_diff_ms_per_step": 0.2,
                "breakdown_delta_ms_per_step": 0.19,
                "relative_diff_pct": 5.0,
            },
            "budget_ms_token": 33.3,
            "budget_ok": True,
            "verdict": {
                "cluster": "TMV",
                "share_pct": 56.0,
                "ms_per_step": 0.027,
                "context": 128,
                "backend": "simd-neon",
            },
            "verdicts": [
                {
                    "cluster": "TMV",
                    "share_pct": 56.0,
                    "ms_per_step": 0.027,
                    "context": 128,
                    "backend": "simd-neon",
                }
            ],
            "notes": ["тестовая заметка"],
        }
        text = bench_vtm.format_report(report)
        self.assertIn("оптимизировать первым: TMV", text)
        self.assertIn("бюджет: 33.300 мс/шаг — ok", text)
        self.assertIn("сверка: разница scalar-simd", text)
        self.assertIn("тестовая заметка", text)


# ---------------------------------------------------------------------------
# Контейнеры: контекст, SAMP, словарь
# ---------------------------------------------------------------------------


class TestContainerPrep(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workdir = os.path.join(self.tmp.name, "work")
        os.makedirs(self.workdir)

    def test_info(self):
        path = write_tiny_model(os.path.join(self.tmp.name, "m.vtm"), context=4)
        info = bench_vtm.read_container_info(path)
        self.assertEqual(info.context, 4)
        self.assertEqual(info.vocab, VOCAB)
        self.assertEqual(info.tensors, 8)
        self.assertEqual(info.instructions, 3)
        self.assertEqual(info.ops["TMV"], 1)

    def test_expand_context_and_add_samp(self):
        path = write_tiny_model(os.path.join(self.tmp.name, "m.vtm"), context=4)
        prepared = bench_vtm.prepare_container(path, 128, self.workdir)
        self.assertNotEqual(prepared, path)
        self.assertTrue(prepared.startswith(self.workdir))
        info = bench_vtm.read_container_info(prepared)
        self.assertGreaterEqual(info.context, 128)
        self.assertEqual(info.ops.get("SAMP"), 1)
        self.assertEqual(info.vocab, VOCAB)
        # Исходный контейнер не изменился.
        self.assertEqual(bench_vtm.read_container_info(path).context, 4)

    def test_prepare_returns_source_when_it_fits(self):
        path = write_tiny_model(
            os.path.join(self.tmp.name, "m.vtm"), context=64, with_samp=True
        )
        prepared = bench_vtm.prepare_container(path, 32, self.workdir)
        self.assertEqual(prepared, path)

    def test_prepare_without_cache_fails(self):
        path = write_tiny_model(
            os.path.join(self.tmp.name, "m.vtm"), with_cache=False, with_samp=True
        )
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.prepare_container(path, 64, self.workdir)

    def test_add_samp_inserts_before_halt(self):
        path = write_tiny_model(os.path.join(self.tmp.name, "m.vtm"), with_samp=False)
        loaded = vtm_ref.read_vtm(path)
        self.assertTrue(bench_vtm.add_samp_if_missing(loaded))
        self.assertEqual(int(loaded.program[-1].op), int(vtm_ref.Op.HALT))
        self.assertEqual(int(loaded.program[-2].op), int(vtm_ref.Op.SAMP))
        # Второй вызов ничего не добавляет.
        self.assertFalse(bench_vtm.add_samp_if_missing(loaded))

    def test_add_samp_requires_halt(self):
        path = write_tiny_model(os.path.join(self.tmp.name, "m.vtm"))
        loaded = vtm_ref.read_vtm(path)
        loaded.program[-1] = vtm_ref.Instr(
            op=int(vtm_ref.Op.TMV), dst=4, a=3, b=1
        )
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.add_samp_if_missing(loaded)

    def test_container_context_from_meta(self):
        tensors = [vtm_ref.Tensor.f32("x", (4,), [0.0] * 4)]
        program = [vtm_ref.Instr(op=int(vtm_ref.Op.HALT))]
        path = os.path.join(self.tmp.name, "meta.vtm")
        vtm_ref.write_vtm(
            path, tensors, program,
            {"vocab_size": 16, "model": {"context_len": 512}},
        )
        self.assertEqual(bench_vtm.read_container_info(path).context, 512)

    def test_container_vocab_falls_back_to_emb(self):
        tensors = [vtm_ref.Tensor.f32("emb.w", (24, 4), [0.0] * 96)]
        program = [vtm_ref.Instr(op=int(vtm_ref.Op.HALT))]
        path = os.path.join(self.tmp.name, "emb.vtm")
        vtm_ref.write_vtm(path, tensors, program, {})
        self.assertEqual(bench_vtm.read_container_info(path).vocab, 24)

    def test_broken_container_is_bench_error(self):
        path = os.path.join(self.tmp.name, "bad.vtm")
        with open(path, "wb") as handle:
            handle.write(b"not a vtm file")
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.read_container_info(path)

    def test_token_ids_deterministic_and_in_range(self):
        first = bench_vtm.token_ids(64, VOCAB)
        second = bench_vtm.token_ids(64, VOCAB)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertTrue(all(0 <= token < VOCAB for token in first))
        self.assertGreater(len(set(first)), 1)
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.token_ids(0, VOCAB)
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.token_ids(8, 0)

    def test_scenarios_product(self):
        rows = bench_vtm.iter_scenarios([128, 512], [1, 32], ["scalar", "simd"])
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [row.label() for row in rows[:4]],
            [
                "ctx=128 gen=1 scalar",
                "ctx=128 gen=1 simd",
                "ctx=128 gen=32 scalar",
                "ctx=128 gen=32 simd",
            ],
        )
        self.assertEqual(rows[0].forwards, 128)
        self.assertEqual(bench_vtm.Scenario(8, 2, "scalar").forwards, 9)


# ---------------------------------------------------------------------------
# Разбивка по операциям
# ---------------------------------------------------------------------------


class TestBreakdownHelpers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = write_tiny_model(os.path.join(self.tmp.name, "m.vtm"))
        self.vtm = vtm_ref.read_vtm(self.path)

    def test_duplicate_instructions(self):
        doubled = bench_vtm.duplicate_instructions(self.vtm.program, ("TMV",), 3)
        self.assertEqual(len(doubled), len(self.vtm.program) + 3)
        names = [vtm_ref.OP_NAMES[int(instr.op)] for instr in doubled]
        self.assertEqual(names.count("TMV"), 4)
        self.assertEqual(names.count("HALT"), 1)
        self.assertEqual(names[-1], "HALT")
        # Копии идут сразу после оригинала и повторяют его поля.
        self.assertEqual(doubled[2], doubled[1])
        self.assertEqual(bench_vtm.duplicate_instructions(self.vtm.program, ("TMV",), 0),
                         list(self.vtm.program))
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.duplicate_instructions(self.vtm.program, ("TMV",), -1)

    def test_duplicate_nothing_for_absent_op(self):
        same = bench_vtm.duplicate_instructions(self.vtm.program, ("ATNN",), 4)
        self.assertEqual(len(same), len(self.vtm.program))

    def test_parse_trace_lines(self):
        text = (
            "trace: время по операциям\n"
            "TMV   8 x 3.500 ms\n"
            "ATTN: 1.250 мс\n"
            "RMSN 250 µs\n"
            "SAMP 0.5 ms\n"
            "время: 6.000 мс\n"
        )
        parsed = bench_vtm.parse_trace_output(text)
        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(parsed["TMV"], 3.5)
        self.assertAlmostEqual(parsed["ATTN"], 1.25)
        self.assertAlmostEqual(parsed["RMSN"], 0.25)
        self.assertAlmostEqual(parsed["SAMP"], 0.5)

    def test_parse_trace_json_forms(self):
        self.assertEqual(
            bench_vtm.parse_trace_output('{"TMV": 2.0, "ATTN": 1.0}')["TMV"], 2.0
        )
        nested = bench_vtm.parse_trace_output('{"ops": {"TMV": 4.0}}')
        self.assertEqual(nested["TMV"], 4.0)
        listed = bench_vtm.parse_trace_output(
            '[{"op": "TMV", "ms": 1.5}, {"op": "KVW", "us": 500}]'
        )
        self.assertAlmostEqual(listed["TMV"], 1.5)
        self.assertAlmostEqual(listed["KVW"], 0.5)

    def test_parse_trace_garbage(self):
        self.assertIsNone(bench_vtm.parse_trace_output(""))
        self.assertIsNone(bench_vtm.parse_trace_output("ничего полезного\n"))
        self.assertIsNone(bench_vtm.parse_trace_output('{"TMV": "быстро"}'))

    def test_trace_rows_and_shares(self):
        rows = bench_vtm.breakdown_from_trace(
            {"TMV": 6.0, "ATTN": 3.0, "KVW": 1.0},
            context=100,
            gen=1,
            backend="simd",
            forwards=100,
        )
        self.assertEqual(rows[0]["cluster"], "TMV")
        self.assertAlmostEqual(rows[0]["total_ms"], 6.0)
        self.assertAlmostEqual(rows[0]["ms_per_step"], 0.06)
        self.assertAlmostEqual(rows[0]["share_pct"], 60.0)


# ---------------------------------------------------------------------------
# Mock-запуск: командная строка и measure_scenario
# ---------------------------------------------------------------------------


class TestMockRuns(unittest.TestCase):
    def test_run_and_parse_with_fake_runner(self):
        runner = FakeRunner(time_ms=2.5)
        command = bench_vtm.build_run_command("vtm", "m.vtm", [1, 2], 3, backend="simd")
        output, wall_ms = bench_vtm.run_and_parse(command, runner=runner)
        self.assertAlmostEqual(output.time_ms, 2.5)
        self.assertEqual(len(output.generated), 3)
        self.assertGreaterEqual(wall_ms, 0.0)
        self.assertEqual(runner.run_calls()[0], command)

    def test_run_cli_failure(self):
        runner = FakeRunner(fail=True)
        with self.assertRaises(bench_vtm.BenchError) as ctx:
            bench_vtm.run_cli(["vtm", "run"], runner=runner)
        self.assertIn("что-то сломалось", str(ctx.exception))

    def test_run_cli_missing_binary(self):
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.run_cli(["/nonexistent/vtm", "run"])

    def test_measure_scenario_median(self):
        calls = {"index": 0}
        times = [3.0, 1.0, 2.0]

        def runner(command, timeout):
            calls["index"] += 1
            return _proc(command, 0, fake_run_stdout(command, time_ms=times[calls["index"] - 1]))

        # measure_scenario не читает контейнер: весь ввод-вывод — через runner.
        scenario = bench_vtm.Scenario(context=10, gen=2, backend="scalar")
        measured = bench_vtm.measure_scenario(
            "m.vtm", scenario, vtm_bin="vtm", vocab=VOCAB, repeat=3, runner=runner
        )
        self.assertAlmostEqual(measured["time_ms"]["median"], 2.0)
        self.assertEqual(measured["forwards"], 11)
        self.assertAlmostEqual(measured["ms_per_token"], 1.0)
        self.assertAlmostEqual(measured["tokens_per_s"], 1000.0)
        self.assertAlmostEqual(measured["ms_per_step"], 2.0 / 11)

    def test_measure_scenario_wrong_generated_count(self):
        class ShortRunner:
            def __call__(self, command, timeout):
                return _proc(command, 0, fake_run_stdout(command, time_ms=1.0).replace(
                    "вывод:   1 4", "вывод:   1"
                ))

        scenario = bench_vtm.Scenario(context=4, gen=2, backend="scalar")
        with self.assertRaises(bench_vtm.BenchError):
            bench_vtm.measure_scenario(
                "m.vtm", scenario, vtm_bin="vtm", vocab=VOCAB, repeat=1,
                runner=ShortRunner(),
            )

    def test_patch_subprocess_run(self):
        """Ровно то, что просит задача: mock через подмену subprocess."""
        captured = {}

        def fake_run(command, capture_output, text, timeout):
            captured["command"] = list(command)
            return _proc(command, 0, fake_run_stdout(command, time_ms=4.0))

        with mock.patch("tools.bench_vtm.subprocess.run", side_effect=fake_run):
            output, _ = bench_vtm.run_and_parse(
                bench_vtm.build_run_command("vtm", "m.vtm", [5], 1),
                timeout=1.0,
            )
        self.assertAlmostEqual(output.time_ms, 4.0)
        self.assertEqual(captured["command"][:3], ["vtm", "run", "m.vtm"])

    def test_default_runner_timeout_is_bench_error(self):
        def raising(command, capture_output, text, timeout):
            raise subprocess.TimeoutExpired(command, timeout)

        with mock.patch("tools.bench_vtm.subprocess.run", side_effect=raising):
            with self.assertRaises(bench_vtm.BenchError):
                bench_vtm.run_cli(["vtm", "run"], timeout=0.01)

    def test_detect_trace(self):
        # FakeRunner не читает файл, но detect_trace сначала проверяет
        # существование бинарника — берём настоящий путь.
        existing = sys.executable
        self.assertFalse(bench_vtm.detect_trace(existing, runner=FakeRunner()))
        self.assertTrue(
            bench_vtm.detect_trace(existing, runner=FakeRunner(trace_supported=True))
        )
        self.assertFalse(bench_vtm.detect_trace("/nonexistent/vtm"))


# ---------------------------------------------------------------------------
# main: JSON-отчёт и коды возврата (C не нужен)
# ---------------------------------------------------------------------------


class TestMainJson(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model = write_tiny_model(
            os.path.join(self.tmp.name, "m.vtm"), context=8, with_samp=True
        )
        self.workdir = os.path.join(self.tmp.name, "work")
        # Класс не запускает настоящий vtm (runner подменяется), но main()
        # проверяет существование бинарника до вызова раннера. В чистом клоне
        # src/vtm/vtm не собран, поэтому даём заглушку-скрипт: тесты не должны
        # зависеть от собранного C.
        self.vtm_stub = os.path.join(self.tmp.name, "vtm-stub")
        with open(self.vtm_stub, "w", encoding="utf-8") as fh:
            fh.write("#!/bin/sh\nexit 0\n")
        os.chmod(self.vtm_stub, 0o755)
        self.base_args = [
            "--vtm", self.vtm_stub,
            "--model", self.model,
            "--contexts", "8",
            "--gen", "2",
            "--backends", "scalar",
            "--repeat", "1",
            "--no-breakdown",
            "--workdir", self.workdir,
        ]

    def _main(self, args, runner):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = bench_vtm.main(args, runner=runner)
        return code, out.getvalue()

    def test_json_is_valid_and_complete(self):
        runner = FakeRunner(time_ms=1.5)
        code, output = self._main(self.base_args + ["--json"], runner)
        self.assertEqual(code, 0)
        report = json.loads(output)
        self.assertEqual(report["tool"], "bench_vtm")
        self.assertEqual(report["version"], bench_vtm.REPORT_VERSION)
        self.assertIn("scenarios", report)
        self.assertEqual(len(report["scenarios"]), 1)
        scenario = report["scenarios"][0]
        self.assertEqual(scenario["context"], 8)
        self.assertEqual(scenario["gen"], 2)
        self.assertEqual(scenario["backend"], "scalar-lut243")
        self.assertAlmostEqual(scenario["time_ms"]["median"], 1.5)
        self.assertEqual(scenario["forwards"], 9)
        self.assertAlmostEqual(scenario["ms_per_token"], 0.75)
        self.assertEqual(report["model"]["vocab"], VOCAB)
        self.assertEqual(report["model"]["source"], self.model)
        self.assertEqual(report["breakdown"], [])
        self.assertFalse(report["cli"]["trace"])
        self.assertTrue(any("разбивка отключена" in note for note in report["notes"]))

    def test_json_out_file(self):
        path = os.path.join(self.tmp.name, "report.json")
        runner = FakeRunner(time_ms=1.5)
        code, output = self._main(
            self.base_args + ["--json", "--json-out", path], runner
        )
        self.assertEqual(code, 0)
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertEqual(report["scenarios"][0]["context"], 8)
        self.assertIn('"tool": "bench_vtm"', output)

    def test_table_output(self):
        runner = FakeRunner(time_ms=1.5)
        code, output = self._main(self.base_args, runner)
        self.assertEqual(code, 0)
        self.assertIn("Сценарии", output)
        self.assertIn("scalar-lut243", output)
        self.assertIn("мс/шаг", output)

    def test_budget_exceeded_returns_1(self):
        runner = FakeRunner(time_ms=500.0)
        code, output = self._main(
            self.base_args + ["--json", "--budget-ms-token", "1"],
            runner,
        )
        self.assertEqual(code, 1)
        report = json.loads(output)
        self.assertFalse(report["budget_ok"])
        self.assertFalse(report["scenarios"][0]["budget_ok"])

    def test_budget_ok_returns_0(self):
        runner = FakeRunner(time_ms=1.0)
        code, _ = self._main(
            self.base_args + ["--budget-ms-token", "100"], runner
        )
        self.assertEqual(code, 0)

    def test_breakdown_from_mock(self):
        args = [
            "--model", self.model,
            "--contexts", "8",
            "--gen", "1",
            "--backends", "scalar",
            "--repeat", "1",
            "--breakdown-contexts", "8",
            "--breakdown-backends", "scalar",
            "--breakdown-repeat", "1",
            "--dup-copies", "2",
            "--workdir", self.workdir,
            "--json",
        ]
        runner = FakeRunner(time_ms=2.0)
        code, output = self._main(args, runner)
        self.assertEqual(code, 0)
        report = json.loads(output)
        self.assertEqual(len(report["breakdown"]), 1)
        item = report["breakdown"][0]
        self.assertEqual(item["method"], "duplication")
        self.assertEqual(item["scenario"]["context"], 8)
        clusters = {row["cluster"] for row in item["rows"]}
        self.assertIn("TMV", clusters)
        self.assertIn("RMSN", clusters)
        # ATTENTION в крошечном контейнере нет — он в списке отсутствующих.
        self.assertIn("ATTN", item["absent_clusters"])
        self.assertIn("verdict", report)
        self.assertTrue(any("--trace" in note for note in report["notes"]))

    def test_trace_used_when_supported(self):
        args = [
            "--model", self.model,
            "--contexts", "8",
            "--gen", "1",
            "--backends", "scalar",
            "--repeat", "1",
            "--breakdown-contexts", "8",
            "--breakdown-backends", "scalar",
            "--breakdown-repeat", "1",
            "--dup-copies", "1",
            "--workdir", self.workdir,
            "--json",
        ]
        trace = "TMV 0.60 мс\nRMSN 0.10 мс\nSAMP 0.05 мс\n"
        runner = FakeRunner(time_ms=2.0, trace_supported=True, trace_lines=trace)
        code, output = self._main(args, runner)
        self.assertEqual(code, 0)
        report = json.loads(output)
        item = report["breakdown"][0]
        self.assertEqual(item["method"], "trace")
        self.assertTrue(item["trace"]["parsed"])
        self.assertIn("TMV", {row["cluster"] for row in item["rows"]})

    def test_trace_present_but_unparsed_falls_back(self):
        args = [
            "--model", self.model,
            "--contexts", "8",
            "--gen", "1",
            "--backends", "scalar",
            "--repeat", "1",
            "--breakdown-contexts", "8",
            "--breakdown-backends", "scalar",
            "--breakdown-repeat", "1",
            "--dup-copies", "1",
            "--workdir", self.workdir,
            "--json",
        ]
        runner = FakeRunner(time_ms=2.0, trace_supported=True, trace_lines="мусор\n")
        code, output = self._main(args, runner)
        self.assertEqual(code, 0)
        item = json.loads(output)["breakdown"][0]
        self.assertEqual(item["method"], "duplication")
        self.assertFalse(item["trace"]["parsed"])

    def test_trace_on_without_support_returns_2(self):
        args = self.base_args + ["--trace", "on", "--json"]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = bench_vtm.main(args, runner=FakeRunner(trace_supported=False))
        self.assertEqual(code, 2)
        self.assertIn("--trace", err.getvalue())

    def test_missing_model_returns_2(self):
        args = [
            "--model", os.path.join(self.tmp.name, "nope.vtm"),
            "--contexts", "8", "--gen", "1", "--repeat", "1",
            "--no-breakdown", "--workdir", self.workdir, "--json",
        ]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = bench_vtm.main(args, runner=FakeRunner())
        self.assertEqual(code, 2)
        self.assertIn("не найден", err.getvalue())

    def test_missing_binary_returns_2(self):
        args = [
            "--vtm", os.path.join(self.tmp.name, "no-vtm"),
            "--model", self.model,
            "--contexts", "8", "--gen", "1", "--repeat", "1",
            "--no-breakdown", "--workdir", self.workdir, "--json",
        ]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = bench_vtm.main(args, runner=FakeRunner())
        self.assertEqual(code, 2)
        self.assertIn("нет бинарника", err.getvalue())

    def test_bad_backend_argument_returns_2(self):
        args = self.base_args + ["--backends", "cuda", "--json"]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = bench_vtm.main(args, runner=FakeRunner())
        self.assertEqual(code, 2)

    def test_synthetic_command_is_used(self):
        out_dir = os.path.join(self.tmp.name, "synth")
        runner = FakeRunner(time_ms=1.0)
        # Компилятор в тесте не запускается по-настоящему: подменяем ensure_synthetic.
        fake_path = write_tiny_model(os.path.join(self.tmp.name, "synth.vtm"), context=8,
                                     with_samp=True)
        args = [
            "--synthetic-size", "64",
            "--contexts", "8", "--gen", "1", "--backends", "scalar",
            "--repeat", "1", "--no-breakdown", "--workdir", out_dir, "--json",
        ]
        with mock.patch.object(bench_vtm, "ensure_synthetic", return_value=fake_path):
            code, output = self._main(args, runner)
        self.assertEqual(code, 0)
        report = json.loads(output)
        self.assertEqual(report["config"]["synthetic"]["size"], 64)
        self.assertTrue(any("синтетика" in note for note in report["notes"]))


# ---------------------------------------------------------------------------
# Дополнительно: разбивка напрямую, crosscheck, аргументы
# ---------------------------------------------------------------------------


class TestMeasureBreakdown(unittest.TestCase):
    """measure_breakdown целиком, но на mock-раннере: без C."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.workdir = os.path.join(self.tmp.name, "work")
        os.makedirs(self.workdir)
        self.path = write_tiny_model(
            os.path.join(self.tmp.name, "m.vtm"), context=8, with_samp=True
        )
        self.info = bench_vtm.read_container_info(self.path)

    def test_rows_and_residual(self):
        runner = FakeRunner(time_ms=2.0)
        item = bench_vtm.measure_breakdown(
            self.path,
            context=8,
            gen=1,
            backend="scalar",
            vtm_bin="vtm",
            info=self.info,
            copies=2,
            repeat=1,
            workdir=self.workdir,
            runner=runner,
        )
        self.assertEqual(item["method"], "duplication")
        self.assertEqual(item["scenario"]["forwards"], 8)
        clusters = [row["cluster"] for row in item["rows"]]
        self.assertIn("TMV", clusters)
        self.assertIn("RMSN", clusters)
        self.assertIn("ATTN", item["absent_clusters"])
        for row in item["rows"]:
            self.assertEqual(row["instructions"], 1)
            self.assertIn("share_pct", row)
        self.assertAlmostEqual(
            item["covered_ms"], sum(row["total_ms"] for row in item["rows"])
        )
        self.assertAlmostEqual(
            item["residual_ms"],
            item["base_time_ms"] - item["covered_ms"],
        )
        # Дублирующие контейнеры лежат в workdir и валидны.
        variants = [name for name in os.listdir(self.workdir) if name.startswith("dup-")]
        self.assertEqual(len(variants), len(item["rows"]))
        for name in variants:
            vtm_ref.read_vtm(os.path.join(self.workdir, name))

    def test_noise_flag_for_zero_cost(self):
        # Время не растёт при копировании (константа в mock): класс — шум.
        item = bench_vtm.measure_breakdown(
            self.path,
            context=8,
            gen=1,
            backend="scalar",
            vtm_bin="vtm",
            info=self.info,
            copies=2,
            repeat=1,
            workdir=self.workdir,
            runner=FakeRunner(time_ms=2.0),
        )
        self.assertTrue(all(row["within_noise"] for row in item["rows"]))

    def test_breakdown_context_without_scenarios(self):
        """Разбивка на контексте, которого нет в сценарной матрице."""
        runner = FakeRunner(time_ms=2.0)
        args = [
            "--model", self.path,
            "--contexts", "8",
            "--gen", "1",
            "--backends", "scalar",
            "--repeat", "1",
            "--breakdown-contexts", "16",
            "--breakdown-backends", "scalar",
            "--breakdown-repeat", "1",
            "--dup-copies", "1",
            "--workdir", self.workdir,
            "--json",
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = bench_vtm.main(args, runner=runner)
        self.assertEqual(code, 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report["breakdown"][0]["scenario"]["context"], 16)
        # Сценарий остался только один (ctx=8).
        self.assertEqual(len(report["scenarios"]), 1)


class TestCrosscheck(unittest.TestCase):
    @staticmethod
    def _item(backend, base_ms, tmv_step, rmsn_step, forwards):
        return {
            "method": "duplication",
            "scenario": {"context": 128, "backend": backend, "forwards": forwards},
            "base_time_ms": base_ms,
            "rows": [
                {"cluster": "TMV", "ms_per_step": tmv_step, "total_ms": 0.0},
                {"cluster": "RMSN", "ms_per_step": rmsn_step, "total_ms": 0.0},
            ],
        }

    def test_crosscheck_matches(self):
        breakdown = [
            self._item("scalar-lut243", 40.0, 0.25, 0.005, 128),
            self._item("simd-neon", 6.0, 0.03, 0.004, 128),
        ]
        check = bench_vtm._crosscheck_breakdown(breakdown)
        self.assertIsNotNone(check)
        self.assertEqual(check["context"], 128)
        self.assertAlmostEqual(check["backend_diff_ms_per_step"], 0.265625)
        self.assertAlmostEqual(check["breakdown_delta_ms_per_step"], 0.221)
        self.assertGreater(check["relative_diff_pct"], 0.0)

    def test_crosscheck_needs_two_backends(self):
        breakdown = [self._item("scalar-lut243", 40.0, 0.25, 0.005, 128)]
        self.assertIsNone(bench_vtm._crosscheck_breakdown(breakdown))
        self.assertIsNone(bench_vtm._crosscheck_breakdown([]))


class TestArguments(unittest.TestCase):
    def test_int_list_arg(self):
        self.assertEqual(bench_vtm._int_list_arg("128, 512,2048"), [128, 512, 2048])
        with self.assertRaises(Exception):
            bench_vtm._int_list_arg("128,x")
        with self.assertRaises(Exception):
            bench_vtm._int_list_arg("0")
        with self.assertRaises(Exception):
            bench_vtm._int_list_arg(",")

    def test_backend_list_arg(self):
        self.assertEqual(bench_vtm._backend_list_arg("simd, scalar"), ["simd", "scalar"])
        with self.assertRaises(Exception):
            bench_vtm._backend_list_arg("cuda")
        with self.assertRaises(Exception):
            bench_vtm._backend_list_arg("")

    def test_parser_defaults(self):
        args = bench_vtm.build_parser().parse_args([])
        self.assertEqual(args.contexts, list(bench_vtm.DEFAULT_CONTEXTS))
        self.assertEqual(args.gen, list(bench_vtm.DEFAULT_GEN))
        self.assertEqual(args.backends, ",".join(bench_vtm.DEFAULT_BACKENDS))
        self.assertEqual(args.repeat, bench_vtm.DEFAULT_REPEAT)
        self.assertIsNone(args.budget_ms_token)
        self.assertFalse(args.json)

    def test_parser_rejects_zero_context(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                bench_vtm.build_parser().parse_args(["--contexts", "0"])
        self.assertEqual(ctx.exception.code, 2)

    def test_bad_breakdown_backend_returns_2(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model = write_tiny_model(os.path.join(tmp.name, "m.vtm"), context=8, with_samp=True)
        args = [
            "--model", model,
            "--contexts", "8", "--gen", "1", "--backends", "scalar",
            "--repeat", "1",
            "--breakdown-contexts", "8",
            "--breakdown-backends", "cuda",
            "--workdir", os.path.join(tmp.name, "work"),
            "--json",
        ]
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = bench_vtm.main(args, runner=FakeRunner())
        self.assertEqual(code, 2)
        self.assertIn("cuda", err.getvalue())


class TestScenarioProductInMain(unittest.TestCase):
    def test_matrix_size_and_order(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model = write_tiny_model(os.path.join(tmp.name, "m.vtm"), context=8, with_samp=True)
        args = [
            "--model", model,
            "--contexts", "8,9",
            "--gen", "1,2",
            "--backends", "scalar,simd",
            "--repeat", "1",
            "--no-breakdown",
            "--workdir", os.path.join(tmp.name, "work"),
            "--json",
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = bench_vtm.main(args, runner=FakeRunner())
        self.assertEqual(code, 0)
        report = json.loads(out.getvalue())
        self.assertEqual(len(report["scenarios"]), 8)
        self.assertEqual(
            [(row["context"], row["gen"], row["backend_requested"]) for row in report["scenarios"][:2]],
            [(8, 1, "scalar"), (8, 1, "simd")],
        )
        self.assertEqual(report["scenarios"][-1]["forwards"], 9 + 2 - 1)


if __name__ == "__main__":
    unittest.main()
