#!/usr/bin/env python3
"""Тесты ``tools/bench_shape.py`` — без C и без 1.5B-контейнера.

Проверяется всё, что можно проверить на Python-уровне:

* расчёт размеров по формуле: тритовый payload, параметры формы, полный размер
  файла — сверка с настоящим контейнером, записанным потоковым писателем;
* валидность случайных слов pack20 (каждое < 3²⁰) и детерминизм по seed;
* раскладка KV-кэша: в файле его нет (``TFLAG_KV_CACHE``), offset'ы нулевые;
* разбор вывода ``vtm run`` (``stats:``, шапка с бэкендом) и ``/usr/bin/time``
  в обеих ветках — macOS ``-l`` и GNU ``-v``;
* форматирование таблицы/отчёта и JSON-сериализуемость;
* mock-запуск с инжектированным runner — так матрица «потоки × mmap»
  проверяется в CI без собранного рантайма;
* тяжёлый прогон формы 1.5B (сборка + один замер) — под
  ``VTM_BENCH_SHAPE_HEAVY=1``.

Запуск:
    python3 -m unittest tests.test_bench_shape -v
    VTM_BENCH_SHAPE_HEAVY=1 python3 -m unittest tests.test_bench_shape -v
"""
from __future__ import annotations

import contextlib
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

from tools import bench_shape, vtm_ref  # noqa: E402

#: Крошечная форма той же архитектуры: сборка — доли секунды.
TINY = bench_shape.Shape(
    hidden=32,
    layers=2,
    heads=4,
    kv_heads=2,
    head_dim=8,
    intermediate=64,
    vocab=128,
    ctx=16,
)


def write_tiny(path: str, shape: bench_shape.Shape = TINY) -> tuple[bench_shape.ContainerPlan, int]:
    plan = bench_shape.build_plan(shape)
    size = bench_shape.write_container(path, plan)
    return plan, size


def fake_runner_factory(calls: list, *, inspect_rss: int = 100 * 1024 * 1024,
                        first_rss: int = 200 * 1024 * 1024,
                        many_rss: int = 500 * 1024 * 1024,
                        first_s: float = 0.5, many_s: float = 2.0,
                        backend: str = "simd-neon", fail: bool = False):
    """Runner, который отвечает как ``vtm`` и ``/usr/bin/time``, без C."""

    def runner(command, timeout, env):
        command = [str(part) for part in command]
        calls.append((command, dict(env or {})))
        if fail:
            return subprocess.CompletedProcess(command, 3, "", "vtm: сломалось")
        if "/usr/bin/true" in command or command[-1] == "true":
            return subprocess.CompletedProcess(
                command, 0, "", "        0.00 real         0.00 user         0.00 sys\n"
            )
        if "inspect" in command:
            stderr = (
                "        0.001 real         0.001 user         0.000 sys\n"
                f"{inspect_rss}  maximum resident set size\n"
            )
            stdout = (
                f"тензоров: 623, инструкций: 482, бэкенд: {backend}\n"
            )
            return subprocess.CompletedProcess(command, 0, stdout, stderr)
        if "run" in command:
            index = command.index("--max-new")
            max_new = int(command[index + 1])
            seconds, rss = (first_s, first_rss) if max_new == 1 else (many_s, many_rss)
            stderr = (
                f"stats: токенов={max_new}, время={seconds:.6f} с, "
                f"токенов/с={max_new / seconds:.2f}\n"
                "        0.001 real         0.001 user         0.000 sys\n"
                f"{rss}  maximum resident set size\n"
            )
            stdout = f"тензоров: 623, инструкций: 482, бэкенд: {backend}\n"
            return subprocess.CompletedProcess(command, 0, stdout, stderr)
        raise AssertionError(f"неожиданная команда: {command}")

    return runner


# ---------------------------------------------------------------------------
# Форма
# ---------------------------------------------------------------------------


class TestShape(unittest.TestCase):
    def test_default_is_qwen_1_5b(self) -> None:
        shape = bench_shape.DEFAULT_SHAPE
        self.assertEqual(shape.hidden, 1536)
        self.assertEqual(shape.layers, 28)
        self.assertEqual(shape.heads, 12)
        self.assertEqual(shape.kv_heads, 2)
        self.assertEqual(shape.head_dim, 128)
        self.assertEqual(shape.intermediate, 8960)
        self.assertEqual(shape.vocab, 151936)
        self.assertEqual(shape.ctx, 2048)
        self.assertEqual(shape.q_dim, 1536)
        self.assertEqual(shape.kv_dim, 256)

    def test_rejects_bad_forms(self) -> None:
        with self.assertRaises(bench_shape.BenchShapeError):
            bench_shape.Shape(heads=3, kv_heads=2)
        with self.assertRaises(bench_shape.BenchShapeError):
            bench_shape.Shape(head_dim=7)
        with self.assertRaises(bench_shape.BenchShapeError):
            bench_shape.Shape(hidden=0)
        with self.assertRaises(bench_shape.BenchShapeError):
            bench_shape.Shape(trit_scale=0.0)

    def test_reduced_is_smaller_and_valid(self) -> None:
        small = bench_shape.DEFAULT_SHAPE.reduced()
        self.assertLess(small.hidden, bench_shape.DEFAULT_SHAPE.hidden)
        self.assertLess(small.layers, bench_shape.DEFAULT_SHAPE.layers)
        self.assertLess(small.vocab, bench_shape.DEFAULT_SHAPE.vocab)
        self.assertLess(small.ctx, bench_shape.DEFAULT_SHAPE.ctx)
        self.assertEqual(small.heads % small.kv_heads, 0)
        self.assertEqual(small.head_dim % 2, 0)
        counts = bench_shape.shape_param_counts(small)
        self.assertLess(counts["trit_params"],
                        bench_shape.shape_param_counts(bench_shape.DEFAULT_SHAPE)["trit_params"])

    def test_slug(self) -> None:
        self.assertEqual(TINY.slug(), "h32_l2_v128_c16")

    def test_prompt_tokens_in_vocab_and_deterministic(self) -> None:
        first = bench_shape.prompt_token_ids(TINY, 5)
        second = bench_shape.prompt_token_ids(TINY, 5)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertTrue(all(0 <= token < TINY.vocab for token in first))
        self.assertEqual(len(bench_shape.prompt_token_ids(TINY, 0)), 1)


# ---------------------------------------------------------------------------
# Размеры по формуле
# ---------------------------------------------------------------------------


class TestSizeFormula(unittest.TestCase):
    def test_trit_payload_size_matches_vtm_ref(self) -> None:
        for n in (1, 20, 128, 160, 300, 1280, 1536 * 1536):
            tensor = vtm_ref.Tensor.placeholder("w", (n,), vtm_ref.KIND_TRITS)
            self.assertEqual(
                bench_shape.trit_payload_size(n),
                vtm_ref.tensor_payload_size(tensor),
                n,
            )

    def test_trit_payload_size_breakdown(self) -> None:
        # 1536×1536: 2 359 296 тритов; слова — блоками по 160, масштабы по 128.
        n = 1536 * 1536
        words = -(-n // 160) * 8
        scales = -(-n // 128)
        self.assertEqual(
            bench_shape.trit_payload_size(n),
            18 + scales * 4 + words * 4,
        )

    def test_shape_param_counts_match_real_graph(self) -> None:
        plan = bench_shape.build_plan(TINY)
        counts = bench_shape.shape_param_counts(TINY)
        trits = sum(t.n_elements for t in plan.tensors if t.kind == vtm_ref.KIND_TRITS)
        self.assertEqual(counts["trit_params"], trits)
        self.assertEqual(counts["trit_weights"] + counts["trit_embed"], trits)
        # Таблица словаря — ровно vocab×hidden, tied lm_head не дублируется.
        self.assertEqual(counts["trit_embed"], TINY.vocab * TINY.hidden)
        self.assertEqual(len(plan.tensors), 1 + 22 * TINY.layers + 6)

    def test_file_size_formula(self) -> None:
        """Полный размер = заголовок + таблица + payload'ы + мета + программа."""
        plan = bench_shape.build_plan(TINY)
        table_end = bench_shape.HEADER_SIZE + bench_shape.TENSOR_ENTRY_SIZE * len(plan.tensors)
        data_start = vtm_ref.align_up(table_end)
        data = sum(
            vtm_ref.align_up(vtm_ref.tensor_payload_size(t)) for t in plan.tensors
        )
        meta = bench_shape.encode_meta(plan.meta)
        expected = (
            data_start
            + data
            + vtm_ref.align_up(len(meta))
            + len(plan.program) * bench_shape.INSTR_SIZE
        )
        self.assertEqual(plan.file_size, expected)
        self.assertEqual(plan.file_size, plan.vtm.estimated_file_size())

    def test_kv_cache_takes_memory_but_not_file(self) -> None:
        plan = bench_shape.build_plan(TINY)
        caches = [t for t in plan.tensors if bench_shape.tensor_is_kv_cache(t)]
        self.assertEqual(len(caches), 2 * TINY.layers)
        self.assertGreater(plan.kv_cache_bytes, 0)
        self.assertEqual(plan.kv_cache_bytes, sum(t.n_elements * 4 for t in caches))
        # В файле кэша нет: его payload в размер не входит.
        self.assertEqual(sum(vtm_ref.tensor_payload_size(t) for t in caches), 0)
        self.assertEqual(plan.data_bytes, sum(
            vtm_ref.tensor_payload_size(t) for t in plan.tensors
        ))


# ---------------------------------------------------------------------------
# Запись контейнера
# ---------------------------------------------------------------------------


class TestWriteContainer(unittest.TestCase):
    def test_size_and_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.vtm")
            plan, size = write_tiny(path)
            self.assertEqual(size, plan.file_size)
            self.assertEqual(os.path.getsize(path), plan.file_size)

            loaded = vtm_ref.read_vtm(path)
            self.assertEqual(len(loaded.tensors), len(plan.tensors))
            self.assertEqual(len(loaded.program), len(plan.program))
            self.assertEqual(loaded.tensors[0].name, "emb.w")
            self.assertEqual(loaded.flags, bench_shape.HEADER_FLAGS)
            self.assertEqual(loaded.meta, plan.meta)

    def test_kv_cache_layout_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.vtm")
            plan, _size = write_tiny(path)
            loaded = vtm_ref.read_vtm(path)
            caches = [t for t in loaded.tensors if t.is_kv_cache]
            self.assertTrue(caches)
            for tensor in caches:
                self.assertEqual(tensor.offset, 0)
                self.assertEqual(tensor.scale_offset, 0)
                self.assertIsNone(tensor.data)
            self.assertEqual(
                sorted(t.name for t in caches),
                sorted(f"l{layer}.{kind}cache" for layer in range(TINY.layers)
                       for kind in ("k", "v")),
            )

    def test_random_words_are_valid_pack20(self) -> None:
        """Каждое слово < 3²⁰ (иначе C/``unpack20`` отвергнут контейнер)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.vtm")
            plan, _size = write_tiny(path)
            with open(path, "rb") as handle:
                raw = handle.read()
            data_start = vtm_ref.align_up(
                bench_shape.HEADER_SIZE
                + bench_shape.TENSOR_ENTRY_SIZE * len(plan.tensors)
            )
            offset = plan.tensors[0].offset
            n_scales = vtm_ref.n_scale_blocks(plan.tensors[0].n_elements)
            words = data_start + offset + 18 + n_scales * 4
            n_words = vtm_ref.storage_words(plan.tensors[0].n_elements)
            values = [
                int.from_bytes(raw[words + 4 * i:words + 4 * i + 4], "little")
                for i in range(n_words)
            ]
            self.assertTrue(all(value <= 3 ** 20 - 1 for value in values))
            self.assertGreater(len(set(values)), 10)
            # Триты читаются обратно и лежат в {-1, 0, +1}.
            trits = plan.tensors[0].n_elements
            decoded = vtm_ref.unpack20(raw[words:words + n_words * 4], trits)
            self.assertEqual(len(decoded), trits)
            self.assertTrue(set(decoded) <= {-1, 0, 1})

    def test_norm_weights_are_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.vtm")
            write_tiny(path)
            loaded = vtm_ref.read_vtm(path)
            for name in ("l0.norm1.w", "l0.norm2.w", "final_norm.w"):
                self.assertTrue(all(value == 1.0 for value in loaded.values(name)), name)
            self.assertTrue(all(value == 0.0 for value in loaded.values("l0.h1")))

    def test_writer_is_deterministic(self) -> None:
        import hashlib

        def digest(path: str) -> str:
            with open(path, "rb") as handle:
                return hashlib.sha256(handle.read()).hexdigest()

        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "a.vtm")
            second = os.path.join(tmp, "b.vtm")
            third = os.path.join(tmp, "c.vtm")
            bench_shape.write_container(first, bench_shape.build_plan(TINY))
            bench_shape.write_container(second, bench_shape.build_plan(TINY))
            self.assertEqual(digest(first), digest(second))
            bench_shape.write_container(
                third, bench_shape.build_plan(TINY), seed=TINY.seed + 1
            )
            self.assertNotEqual(digest(first), digest(third))

    def test_container_executes_in_python_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiny.vtm")
            write_tiny(path)
            loaded = vtm_ref.read_vtm(path)
            result = vtm_ref.execute(loaded, [1, 2], 1)
            self.assertIsNotNone(result.logits)
            self.assertEqual(len(result.logits), TINY.vocab)
            self.assertTrue(all(value == value for value in result.logits))  # не NaN

    def test_meta_encoder_matches_vtm_ref(self) -> None:
        encoder = getattr(vtm_ref, "_encode_meta", None)
        if encoder is None:  # pragma: no cover - приватного хелпера может не быть
            self.skipTest("в vtm_ref нет _encode_meta")
        meta = {"b": 1, "a": "кириллица", "c": [1, 2]}
        self.assertEqual(bench_shape.encode_meta(meta), encoder(meta))


# ---------------------------------------------------------------------------
# Разбор вывода
# ---------------------------------------------------------------------------


MAC_TIME = """\
        1.23 real         0.50 user         0.30 sys
          10485760  maximum resident set size
                 0  average shared memory size
"""

LINUX_TIME = """\
\tCommand being timed: "vtm inspect m.vtm"
\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:01.23
\tMaximum resident set size (kbytes): 10240
"""


class TestParsers(unittest.TestCase):
    def test_parse_time_macos(self) -> None:
        parsed = bench_shape.parse_time_output(MAC_TIME)
        self.assertEqual(parsed["max_rss_bytes"], 10485760)
        self.assertAlmostEqual(parsed["wall_s"], 1.23)

    def test_parse_time_gnu(self) -> None:
        parsed = bench_shape.parse_time_output(LINUX_TIME)
        self.assertEqual(parsed["max_rss_bytes"], 10240 * 1024)
        self.assertAlmostEqual(parsed["wall_s"], 1.23)

    def test_parse_time_clock_formats(self) -> None:
        self.assertAlmostEqual(bench_shape._seconds_from_clock("1:02:03.5"), 3723.5)
        self.assertAlmostEqual(bench_shape._seconds_from_clock("0:01.50"), 1.5)
        self.assertAlmostEqual(bench_shape._seconds_from_clock("2.25"), 2.25)
        self.assertIsNone(bench_shape._seconds_from_clock("мусор"))

    def test_parse_time_without_lines(self) -> None:
        self.assertEqual(bench_shape.parse_time_output(""), {})
        self.assertEqual(bench_shape.parse_time_output("какой-то вывод\n"), {})

    def test_parse_time_requires_text(self) -> None:
        with self.assertRaises(bench_shape.BenchShapeError):
            bench_shape.parse_time_output(b"\x00")  # type: ignore[arg-type]

    def test_parse_stats(self) -> None:
        stats = bench_shape.parse_stats_output(
            "мусор\nstats: токенов=4, время=0.123456 с, токенов/с=32.40\n"
        )
        self.assertIsNotNone(stats)
        self.assertEqual(stats.tokens, 4)
        self.assertAlmostEqual(stats.seconds, 0.123456)
        self.assertAlmostEqual(stats.tok_per_s, 32.40)
        self.assertIsNone(bench_shape.parse_stats_output("нет статистики"))
        self.assertEqual(stats.as_dict()["tokens"], 4)

    def test_parse_run_header(self) -> None:
        header = bench_shape.parse_run_header(
            "модель: x\nтензоров: 623, инструкций: 482, бэкенд: simd-neon\n"
        )
        self.assertEqual(header["tensors"], 623)
        self.assertEqual(header["instructions"], 482)
        self.assertEqual(header["backend"], "simd-neon")
        self.assertEqual(bench_shape.parse_run_header("ничего"), {})

    def test_build_run_command(self) -> None:
        command = bench_shape.build_run_command(
            "/bin/vtm", "m.vtm", [1, 2, 3], 4, backend="scalar"
        )
        self.assertEqual(command[:3], ["/bin/vtm", "run", "m.vtm"])
        self.assertIn("1,2,3", command)
        self.assertEqual(command[command.index("--max-new") + 1], "4")
        self.assertEqual(command[command.index("--backend") + 1], "scalar")
        self.assertIn("--stats", command)
        without_backend = bench_shape.build_run_command("/bin/vtm", "m", [1], 1)
        self.assertNotIn("--backend", without_backend)

    def test_build_run_command_rejects_bad_args(self) -> None:
        with self.assertRaises(bench_shape.BenchShapeError):
            bench_shape.build_run_command("/bin/vtm", "m", [], 1)
        with self.assertRaises(bench_shape.BenchShapeError):
            bench_shape.build_run_command("/bin/vtm", "m", [1], -1)

    def test_build_inspect_command(self) -> None:
        self.assertEqual(
            bench_shape.build_inspect_command("/bin/vtm", "m.vtm"),
            ["/bin/vtm", "inspect", "m.vtm"],
        )

    def test_wrap_time(self) -> None:
        self.assertEqual(
            bench_shape.wrap_time(["vtm", "run"], ["/usr/bin/time", "-l"]),
            ["/usr/bin/time", "-l", "vtm", "run"],
        )
        self.assertEqual(bench_shape.wrap_time(["vtm"], None), ["vtm"])


# ---------------------------------------------------------------------------
# Вердикт и форматирование
# ---------------------------------------------------------------------------


class TestVerdict(unittest.TestCase):
    def test_confirm_and_refute(self) -> None:
        ok = bench_shape.verdict(30.0, 30.0, unit="ток/с")
        self.assertEqual(ok["verdict"], "подтверждена")
        self.assertEqual(ok["ratio"], 1.0)
        low = bench_shape.verdict(30.0, 3.0)
        self.assertIn("ниже", low["verdict"])
        self.assertIn("10.0×", low["verdict"])
        high = bench_shape.verdict(30.0, 60.0)
        self.assertIn("выше", high["verdict"])

    def test_missing_values(self) -> None:
        self.assertIsNone(bench_shape.verdict(None, 10.0))
        self.assertIsNone(bench_shape.verdict(10.0, None))
        self.assertIsNone(bench_shape.verdict(0.0, 10.0))

    def test_pick_best(self) -> None:
        rows = [
            {"label": "a", "decode_tok_s": None},
            {"label": "b", "decode_tok_s": 2.0},
            {"label": "c", "decode_tok_s": 5.0},
        ]
        self.assertEqual(bench_shape.pick_best(rows)["label"], "c")
        self.assertIsNone(bench_shape.pick_best([{"decode_tok_s": None}]))


class TestFormatting(unittest.TestCase):
    def _report(self) -> dict:
        rows = [
            {
                "backend": "simd-neon", "threads": 1, "mmap": True,
                "label": "auto/thr1/mmap", "load_s": 0.5, "load_rss_mib": 300.0,
                "ttft_s": 0.2, "decode_s_per_token": 0.1, "decode_tok_s": 10.0,
                "peak_rss_mib": 350.0, "backend_requested": None,
            },
            {
                "backend": "scalar-lut243", "threads": 4, "mmap": False,
                "label": "scalar/thr4/no-mmap", "load_s": 1.0, "load_rss_mib": 420.0,
                "ttft_s": 1.0, "decode_s_per_token": 1.0, "decode_tok_s": 1.0,
                "peak_rss_mib": 460.0, "backend_requested": "scalar",
            },
        ]
        plan = bench_shape.build_plan(TINY)
        self.assertLess(plan.trit_params, bench_shape.VERDICT_MIN_TRITS)
        # Вердикт считается только для «настоящей» формы; здесь снижаем порог,
        # чтобы проверить его форматирование на крошечном плане.
        with mock.patch.object(bench_shape, "VERDICT_MIN_TRITS", 0):
            return bench_shape.build_report(
                plan, rows,
                container={**plan.summary(), "file_bytes": plan.file_size,
                           "file_mib": plan.file_size / 2 ** 20, "build_s": 0.1,
                           "build_rss_mib": 30.0, "write_s": 0.2},
                notes=["тестовое примечание"],
            )

    def test_table_columns(self) -> None:
        table = bench_shape.format_measurements(self._report()["measurements"])
        for column in ("бэкенд", "mmap", "пот.", "загрузка,с", "RSS,МиБ",
                       "TTFT,с", "декод,ток/с", "пик RSS,МиБ"):
            self.assertIn(column, table)
        self.assertIn("simd-neon", table)
        self.assertIn("scalar-lut243", table)
        self.assertIn("10.00", table)
        self.assertIn("1.00", table)

    def test_table_handles_missing_values(self) -> None:
        table = bench_shape.format_measurements([
            {"backend": None, "mmap": False, "threads": 8, "load_s": None,
             "load_rss_mib": None, "ttft_s": None, "decode_tok_s": None,
             "peak_rss_mib": None},
        ])
        self.assertIn("—", table)
        self.assertIn("нет", table)

    def test_report_text(self) -> None:
        text = bench_shape.format_report(self._report())
        self.assertIn("форма: hidden=32", text)
        self.assertIn("контейнер:", text)
        self.assertIn("KV-кэш", text)
        self.assertIn("оценка 30 ток/с", text)
        self.assertIn("опровергнута", text)
        self.assertIn("скалярный бэкенд", text)
        self.assertIn("тестовое примечание", text)

    def test_report_json_serializable(self) -> None:
        report = self._report()
        encoded = json.dumps(report, ensure_ascii=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["shape"]["hidden"], 32)
        self.assertEqual(len(decoded["measurements"]), 2)
        self.assertEqual(decoded["verdict"]["best_label"], "auto/thr1/mmap")

    def test_format_plan(self) -> None:
        plan = bench_shape.build_plan(TINY)
        text = bench_shape.format_plan(plan, build_s=0.1, build_rss_mib=30.0,
                                       write_s=0.2)
        self.assertIn("контейнер:", text)
        self.assertIn("сборка графа", text)
        self.assertIn("запись на диск", text)

    def test_mib(self) -> None:
        self.assertEqual(bench_shape._mib(1048576), 1.0)
        self.assertEqual(bench_shape._mib(0), 0.0)
        self.assertIsNone(bench_shape._mib(None))


# ---------------------------------------------------------------------------
# Mock-запуск: матрица без рантайма
# ---------------------------------------------------------------------------


class TestMeasureMock(unittest.TestCase):
    def test_derives_ttft_and_decode(self) -> None:
        calls: list = []
        row = bench_shape.measure_config(
            "/bin/vtm", "m.vtm",
            bench_shape.MeasureConfig(threads=4, mmap=False, backend="scalar",
                                      prompt_tokens=1, max_new=4),
            TINY,
            runner=fake_runner_factory(calls),
            time_cmd=["/usr/bin/time", "-l"],
            overhead_s=0.0,
        )
        self.assertEqual(row["threads"], 4)
        self.assertFalse(row["mmap"])
        self.assertEqual(row["backend"], "simd-neon")
        self.assertEqual(row["backend_requested"], "scalar")
        self.assertAlmostEqual(row["ttft_s"], 0.5)
        self.assertAlmostEqual(row["decode_s_per_token"], 0.5)
        self.assertAlmostEqual(row["decode_tok_s"], 2.0)
        self.assertEqual(row["peak_rss_mib"], 500.0)
        self.assertEqual(row["load_rss_mib"], 100.0)
        self.assertEqual(row["tensors"], 623)
        self.assertEqual(row["instructions"], 482)

        env = calls[0][1]
        self.assertEqual(env["VTM_THREADS"], "4")
        self.assertEqual(env["VTM_NO_MMAP"], "1")
        self.assertEqual(env["VTM_BACKEND"], "scalar")

    def test_env_for_mmap_and_auto_backend(self) -> None:
        calls: list = []
        bench_shape.measure_config(
            "/bin/vtm", "m.vtm",
            bench_shape.MeasureConfig(threads=8, mmap=True, backend=None,
                                      prompt_tokens=2, max_new=2),
            TINY,
            runner=fake_runner_factory(calls),
            time_cmd=["/usr/bin/time", "-l"],
        )
        env = calls[0][1]
        self.assertEqual(env["VTM_THREADS"], "8")
        self.assertNotIn("VTM_NO_MMAP", env)
        self.assertNotIn("VTM_BACKEND", env)

    def test_single_token_run_has_no_decode(self) -> None:
        row = bench_shape.measure_config(
            "/bin/vtm", "m.vtm",
            bench_shape.MeasureConfig(threads=1, mmap=True, max_new=1),
            TINY,
            runner=fake_runner_factory([]),
        )
        self.assertAlmostEqual(row["ttft_s"], 0.5)
        self.assertIsNone(row["decode_tok_s"])
        self.assertIsNone(row["decode_s_per_token"])

    def test_run_failure_is_reported(self) -> None:
        with self.assertRaises(bench_shape.BenchShapeError) as ctx:
            bench_shape.measure_config(
                "/bin/vtm", "m.vtm",
                bench_shape.MeasureConfig(threads=1, mmap=True, max_new=2),
                TINY,
                runner=fake_runner_factory([], fail=True),
            )
        self.assertIn("код 3", str(ctx.exception))

    def test_measure_list_matrix_shape(self) -> None:
        calls: list = []
        rows = bench_shape.measure_config_list(
            "/bin/vtm", "m.vtm", TINY,
            threads=[1, 4], mmap_modes=[True, False], backends=[None, "scalar"],
            prompt_tokens=1, max_new=2,
            runner=fake_runner_factory(calls),
            time_cmd=["/usr/bin/time", "-l"],
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [(row["threads"], row["mmap"], row["backend_requested"]) for row in rows][:2],
            [(1, True, None), (4, True, None)],
        )
        self.assertEqual(rows[0]["label"], "auto/thr1/mmap")
        self.assertEqual(rows[-1]["label"], "scalar/thr4/no-mmap")

    def test_calibrate_time_overhead(self) -> None:
        calls: list = []
        overhead = bench_shape.calibrate_time_overhead(
            runner=fake_runner_factory(calls),
            time_cmd=["/usr/bin/time", "-l"],
        )
        self.assertGreaterEqual(overhead, 0.0)
        self.assertTrue(all("/usr/bin/true" in call[0] for call in calls))

    def test_time_overhead_without_wrapper(self) -> None:
        self.assertEqual(
            bench_shape.calibrate_time_overhead(runner=None, time_cmd=None), 0.0
        )


class TestMainMock(unittest.TestCase):
    def _run(self, argv: list[str], runner=None) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = bench_shape.main(argv, runner=runner)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_dry_run_json(self) -> None:
        code, stdout, _stderr = self._run([
            "--hidden", "32", "--layers", "2", "--heads", "4", "--kv-heads", "2",
            "--head-dim", "8", "--intermediate", "64", "--vocab", "128",
            "--ctx", "16", "--dry-run", "--json",
        ])
        self.assertEqual(code, 0)
        payload = json.loads(stdout)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["container"]["tensors"], 1 + 22 * 2 + 6)
        self.assertEqual(payload["shape"]["hidden"], 32)

    def test_dry_run_text(self) -> None:
        code, stdout, _stderr = self._run([
            "--hidden", "32", "--layers", "2", "--heads", "4", "--kv-heads", "2",
            "--head-dim", "8", "--intermediate", "64", "--vocab", "128",
            "--ctx", "16", "--dry-run",
        ])
        self.assertEqual(code, 0)
        self.assertIn("контейнер:", stdout)

    def test_build_only_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "built.vtm")
            code, stdout, stderr = self._run([
                "--hidden", "32", "--layers", "2", "--heads", "4", "--kv-heads", "2",
                "--head-dim", "8", "--intermediate", "64", "--vocab", "128",
                "--ctx", "16", "--build-only", "--json", "--out", path,
            ])
            self.assertEqual(code, 0)
            payload = json.loads(stdout)
            self.assertEqual(payload["container"]["file_bytes"], os.path.getsize(path))
            self.assertIn("записан", stderr)

    def test_missing_binary_is_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "built.vtm")
            code, stdout, stderr = self._run([
                "--hidden", "32", "--layers", "2", "--heads", "4", "--kv-heads", "2",
                "--head-dim", "8", "--intermediate", "64", "--vocab", "128",
                "--ctx", "16", "--out", path,
                "--vtm-bin", os.path.join(tmp, "нет-такого"),
            ])
            self.assertEqual(code, 1)
            self.assertEqual(stdout, "")
            self.assertIn("make -C src/vtm vtm", stderr)

    def test_max_new_one_is_rejected(self) -> None:
        code, _stdout, stderr = self._run([
            "--hidden", "32", "--layers", "2", "--heads", "4", "--kv-heads", "2",
            "--head-dim", "8", "--intermediate", "64", "--vocab", "128",
            "--ctx", "16", "--max-new", "1", "--dry-run", "--json",
        ])
        # сухой прогон не меряет декод, поэтому max-new не мешает
        self.assertEqual(code, 0)
        code, _stdout, stderr = self._run([
            "--hidden", "32", "--layers", "2", "--heads", "4", "--kv-heads", "2",
            "--head-dim", "8", "--intermediate", "64", "--vocab", "128",
            "--ctx", "16", "--max-new", "1",
        ])
        self.assertEqual(code, 2)
        self.assertIn("--max-new >= 2", stderr)

    def test_bad_shape_exit_code(self) -> None:
        code, _stdout, stderr = self._run([
            "--hidden", "32", "--heads", "3", "--kv-heads", "2", "--dry-run",
        ])
        self.assertEqual(code, 2)
        self.assertIn("kv_heads", stderr)

    def test_small_shape_has_no_verdict(self) -> None:
        plan = bench_shape.build_plan(TINY)
        report = bench_shape.build_report(
            plan,
            [{"backend": "simd", "threads": 1, "mmap": True, "label": "x",
              "decode_tok_s": 100.0, "peak_rss_mib": 2.0,
              "backend_requested": None}],
            container={**plan.summary(), "file_bytes": plan.file_size,
                       "file_mib": 0.0},
        )
        self.assertIsNone(report["verdict"]["tok_s"])
        self.assertIsNone(report["verdict"]["rss"])
        self.assertTrue(any("не применяются" in note for note in report["notes"]))

    def test_full_measure_mock(self) -> None:
        calls: list = []
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.vtm")
            binary = os.path.join(tmp, "vtm")
            with open(binary, "w", encoding="utf-8") as handle:
                handle.write("#!/bin/sh\n")
            os.chmod(binary, 0o755)
            with mock.patch.object(bench_shape, "VERDICT_MIN_TRITS", 0):
                code, stdout, _stderr = self._run([
                    "--hidden", "32", "--layers", "2", "--heads", "4",
                    "--kv-heads", "2", "--head-dim", "8", "--intermediate", "64",
                    "--vocab", "128", "--ctx", "16", "--out", path,
                    "--vtm-bin", binary, "--json",
                    "--threads", "1,2", "--mmap", "both",
                    "--backends", "auto,scalar", "--max-new", "4",
                ], runner=fake_runner_factory(calls))
            self.assertEqual(code, 0)
            payload = json.loads(stdout)
            self.assertEqual(len(payload["measurements"]), 8)
            self.assertIn("опровергнута", payload["verdict"]["tok_s"]["verdict"])
            self.assertEqual(payload["verdict"]["best_label"], "auto/thr1/mmap")
            self.assertEqual(len({row["label"] for row in payload["measurements"]}), 8)


# ---------------------------------------------------------------------------
# Тяжёлый прогон: настоящая 1.5B-форма
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    os.environ.get("VTM_BENCH_SHAPE_HEAVY") == "1",
    "тяжёлый прогон: VTM_BENCH_SHAPE_HEAVY=1 (345 МиБ и секунды на токен)",
)
class TestHeavyDefaultShape(unittest.TestCase):
    def test_build_and_measure(self) -> None:
        vtm_bin = os.environ.get("VTM_BIN", os.path.join(ROOT, "src", "vtm", "vtm"))
        if not os.access(vtm_bin, os.X_OK):
            self.skipTest(f"нет собранного рантайма: {vtm_bin} (make -C src/vtm vtm)")

        shape = bench_shape.DEFAULT_SHAPE
        plan = bench_shape.build_plan(shape)
        self.assertGreater(plan.file_size, 300 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shape.vtm")
            size = bench_shape.write_container(path, plan)
            self.assertEqual(size, os.path.getsize(path))
            self.assertEqual(size, plan.file_size)

            # Заголовок и таблицу читаем без распаковки тритов: vtm_ref.read_vtm
            # развернул бы 1,5 млрд тритов в списки Python (десятки гигабайт).
            with open(path, "rb") as handle:
                head = handle.read(bench_shape.HEADER_SIZE)
            magic, version, flags, n_instr, n_tensors, _n_prog = (
                bench_shape._HEADER.unpack_from(head, 0)
            )
            self.assertEqual(magic, vtm_ref.VTM_MAGIC)
            self.assertEqual(version, vtm_ref.VTM_VERSION)
            self.assertEqual(n_tensors, len(plan.tensors))
            self.assertEqual(n_instr, len(plan.program))
            self.assertEqual(flags, bench_shape.HEADER_FLAGS)

            time_cmd = bench_shape.time_utility_command()
            row = bench_shape.measure_config(
                vtm_bin, path,
                bench_shape.MeasureConfig(threads=1, mmap=True, max_new=2),
                shape,
                time_cmd=time_cmd,
            )
            self.assertGreater(row["load_s"], 0.0)
            self.assertGreater(row["ttft_s"], 0.0)
            self.assertGreater(row["decode_tok_s"], 0.0)
            if time_cmd:
                self.assertIsNotNone(row["peak_rss_mib"])
                self.assertGreater(row["peak_rss_mib"], 200.0)


if __name__ == "__main__":
    unittest.main()
