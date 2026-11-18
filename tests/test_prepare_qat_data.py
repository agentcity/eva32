"""Тесты подготовки данных QAT (задача 0092).

Проверяется то, что нельзя проверить глазами: детерминизм разбиения и его
независимость от порядка строк, отсутствие пересечения train/val, точный
подсчёт пар длиннее контекста (включая границу 2048/2049), арифметика шагов,
дедупликация по нормализованному вопросу и паритет Python-BPE с ``vtm-tok``.

Запуск: python3 -m unittest discover -s tests -v
"""
import contextlib
import importlib.util
import io
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "prepare_qat_data.py"
FIXTURE_TOKENIZER = ROOT / "tests" / "vtm" / "fixtures" / "tokenizer.json"
VTM_TOK = ROOT / "src" / "vtm" / "vtm-tok"

_spec = importlib.util.spec_from_file_location("prepare_qat_data", TOOL)
qat_data = importlib.util.module_from_spec(_spec)
sys.modules.setdefault(_spec.name, qat_data)
_spec.loader.exec_module(qat_data)


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------


def make_pairs(n, prefix="вопрос"):
    rows = []
    for i in range(n):
        question = f"{prefix} номер {i}?"
        rows.append({
            "question": question,
            "answer": f"ответ {i}",
            "_key": qat_data.normalize_question(question),
        })
    return rows


def keys(rows):
    return {row["_key"] for row in rows}


class LengthTokenizer:
    """Токен = символ: длина ответа равна длине текста до eos."""

    eos_id = 7
    prompt_len = 10

    def encode(self, text):
        if text.startswith("<|im_start|>system"):
            return [0] * self.prompt_len
        return [0] * len(text)


class TinyPairTokenizer(LengthTokenizer):
    """Для count_pair: фиксированные длины prompt/answer."""

    def __init__(self, prompt_len, answer_ids):
        self.prompt_len = prompt_len
        self.answer_ids = list(answer_ids)

    def encode(self, text):
        if text.startswith("<|im_start|>system"):
            return [0] * self.prompt_len
        return list(self.answer_ids)


class TokenizerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = qat_data.BPETokenizer.load(FIXTURE_TOKENIZER)


# ---------------------------------------------------------------------------
# Нормализация и дедупликация
# ---------------------------------------------------------------------------


class TestNormalizeQuestion(unittest.TestCase):
    def test_case_and_punctuation(self):
        self.assertEqual(
            qat_data.normalize_question("  Как   УЛУЧШИТЬ код?! "),
            "как улучшить код",
        )

    def test_punctuation_does_not_glue_words(self):
        self.assertEqual(qat_data.normalize_question("тест,тест"), "тест тест")

    def test_yo_casefold(self):
        self.assertEqual(qat_data.normalize_question("ЁЖИК"), "ёжик")

    def test_typographic_quotes(self):
        self.assertEqual(
            qat_data.normalize_question("«Как» — это «что»?"),
            "как это что",
        )


class TestLoadPairs(unittest.TestCase):
    def _write(self, directory, lines):
        path = Path(directory) / "pairs.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_cleaning_and_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [
                json.dumps({"question": "Как тест?", "answer": "Ответ 1", "meta": {"m": 1}}),
                json.dumps({"question": "как  ТЕСТ", "answer": "Ответ 2"}),
                json.dumps({"question": "Другой вопрос", "answer": "Ответ 3"}),
                json.dumps({"question": "   ", "answer": "Ответ 4"}),
                "{не json}",
                json.dumps({"question": "Без ответа"}),
            ])
            pairs, stats = qat_data.load_pairs(path)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(stats["rows"], 6)
        self.assertEqual(stats["duplicates_removed"], 1)
        self.assertEqual(stats["skipped_empty"], 2)
        self.assertEqual(stats["skipped_invalid"], 1)
        self.assertEqual(pairs[0]["answer"], "Ответ 1")  # остаётся первый дубль
        self.assertEqual(pairs[0]["meta"], {"m": 1})
        self.assertIn("_key", pairs[0])  # внутренний ключ, в файлы не пишется

    def test_missing_file(self):
        with self.assertRaises(qat_data.DataError):
            qat_data.load_pairs("/nonexistent/pairs.jsonl")

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            path.write_text("\n\n", encoding="utf-8")
            with self.assertRaises(qat_data.DataError):
                qat_data.load_pairs(path)


# ---------------------------------------------------------------------------
# Разбиение
# ---------------------------------------------------------------------------


class TestSplit(unittest.TestCase):
    def test_ratio_90_10(self):
        train, val, stats = qat_data.split_pairs(make_pairs(1000))
        self.assertEqual(len(train), 900)
        self.assertEqual(len(val), 100)
        self.assertEqual(stats["pairs"], 1000)
        self.assertEqual(stats["overlap_normalized"], 0)
        self.assertEqual(stats["overlap_raw"], 0)

    def test_small_sets(self):
        train, val, _ = qat_data.split_pairs(make_pairs(3))
        self.assertEqual((len(train), len(val)), (3 - 1, 1))
        with self.assertRaises(qat_data.DataError):
            qat_data.split_pairs(make_pairs(1))

    def test_deterministic_same_input(self):
        rows = make_pairs(500)
        first = qat_data.split_pairs(rows)
        second = qat_data.split_pairs(rows)
        self.assertEqual(keys(first[0]), keys(second[0]))
        self.assertEqual(keys(first[1]), keys(second[1]))

    def test_independent_of_input_order(self):
        rows = make_pairs(500)
        shuffled = list(rows)
        random.Random(123).shuffle(shuffled)
        a_train, a_val, _ = qat_data.split_pairs(rows)
        b_train, b_val, _ = qat_data.split_pairs(shuffled)
        self.assertEqual(keys(a_train), keys(b_train))
        self.assertEqual(keys(a_val), keys(b_val))

    def test_seed_changes_split(self):
        rows = make_pairs(500)
        _, val_a, _ = qat_data.split_pairs(rows, seed=42)
        _, val_b, _ = qat_data.split_pairs(rows, seed=43)
        self.assertNotEqual(keys(val_a), keys(val_b))

    def test_leakage_is_error(self):
        rows = make_pairs(2)
        rows[1]["_key"] = rows[0]["_key"]
        rows[1]["question"] = rows[0]["question"]
        with self.assertRaises(qat_data.DataError):
            qat_data.split_pairs(rows)

    def test_invalid_fraction(self):
        with self.assertRaises(qat_data.DataError):
            qat_data.split_pairs(make_pairs(10), val_fraction=0.0)
        with self.assertRaises(qat_data.DataError):
            qat_data.split_pairs(make_pairs(10), val_fraction=1.0)


# ---------------------------------------------------------------------------
# Подсчёт токенов и обрезка
# ---------------------------------------------------------------------------


class TestCountPair(unittest.TestCase):
    def test_boundary_2048_kept_2049_dropped(self):
        tokenizer = TinyPairTokenizer(prompt_len=2000, answer_ids=(1,) * 47)
        at_limit = qat_data.count_pair(tokenizer, "q", "a", max_seq_len=2048)
        self.assertEqual(at_limit["full"], 2048)  # 2000 + 47 + eos
        self.assertFalse(at_limit["over_context"])

        tokenizer = TinyPairTokenizer(prompt_len=2000, answer_ids=(1,) * 48)
        over = qat_data.count_pair(tokenizer, "q", "a", max_seq_len=2048)
        self.assertEqual(over["full"], 2049)
        self.assertTrue(over["over_context"])

    def test_eos_not_duplicated(self):
        tokenizer = TinyPairTokenizer(prompt_len=5, answer_ids=(1, 2, LengthTokenizer.eos_id))
        counts = qat_data.count_pair(tokenizer, "q", "a")
        self.assertEqual(counts["answer"], 3)

    def test_eos_appended_once(self):
        tokenizer = TinyPairTokenizer(prompt_len=5, answer_ids=(1, 2))
        counts = qat_data.count_pair(tokenizer, "q", "a")
        self.assertEqual(counts["answer"], 3)
        self.assertEqual(counts["full"], 8)

    def test_plain_style_has_no_chat_markers(self):
        tokenizer = LengthTokenizer()
        counts = qat_data.count_pair(tokenizer, "сколько?", "да", prompt_style="plain")
        self.assertEqual(counts["prompt"], len("сколько?"))
        self.assertEqual(counts["full"], len("сколько?") + len("да") + 1)
        self.assertEqual(qat_data.render_prompt("сколько?", "plain"), "сколько?")

    def test_question_braces_survive_template(self):
        prompt = qat_data.render_prompt("что делает `f{}`?", "qwen")
        self.assertIn("что делает `f{}`?", prompt)


class TestTokenizeSplit(unittest.TestCase):
    def test_over_context_and_distribution(self):
        tokenizer = LengthTokenizer()
        rows = [
            {"question": "q1", "answer": "a" * 2037},  # 10 + 2037 + 1 = 2048
            {"question": "q2", "answer": "a" * 2038},  # 2049
            {"question": "q3", "answer": "a" * 3000},
        ]
        result = qat_data.tokenize_split(rows, tokenizer, max_seq_len=2048)
        self.assertEqual(result["pairs"], 3)
        self.assertEqual(result["over_context"]["count"], 2)
        self.assertEqual(result["over_context"]["answer_only_count"], 1)
        self.assertAlmostEqual(result["over_context"]["fraction"], 2 / 3, places=6)
        self.assertEqual(result["length"]["full"]["max"], 3011)
        self.assertEqual(result["length"]["full"]["p50"], 2049)  # nearest-rank
        self.assertEqual(result["length"]["full"]["p95"], 3011)
        self.assertEqual(result["tokens"]["answer"], (2038 + 2039 + 3001))
        self.assertEqual(result["tokens"]["prompt"], 30)
        self.assertEqual(result["tokens"]["full"], 2038 + 2039 + 3001 + 30)

    def test_empty_split(self):
        result = qat_data.tokenize_split([], LengthTokenizer())
        self.assertEqual(result["pairs"], 0)
        self.assertEqual(result["tokens"]["full"], 0)
        self.assertEqual(result["length"]["full"]["p95"], 0)


class TestPercentileAndHistogram(unittest.TestCase):
    def test_percentiles(self):
        values = list(range(1, 101))
        self.assertEqual(qat_data.percentile(values, 50), 50)
        self.assertEqual(qat_data.percentile(values, 95), 95)
        self.assertEqual(qat_data.percentile(values, 100), 100)
        self.assertEqual(qat_data.percentile([], 95), 0)
        self.assertEqual(qat_data.percentile([7], 50), 7)

    def test_histogram_buckets(self):
        buckets = qat_data.histogram([0, 511, 512, 4095, 4096, 9000], 512, 4096)
        self.assertEqual(
            buckets,
            {"0-511": 2, "512-1023": 1, "3584-4095": 1, "4096+": 2},
        )


# ---------------------------------------------------------------------------
# Шаги
# ---------------------------------------------------------------------------


class TestSteps(unittest.TestCase):
    def test_effective_and_micro_batches(self):
        steps = qat_data.steps_for_pairs(100, 10, epochs=(1, 2, 3))
        profile = steps["profiles"]["qwen1.5b"]
        self.assertEqual(profile["micro_batch"], 4)
        self.assertEqual(profile["grad_accum"], 8)
        self.assertEqual(profile["effective_batch"], 32)
        self.assertEqual(profile["val_batches_per_epoch"], 1)  # ceil(10/32)
        self.assertEqual(profile["epochs"]["1"]["steps"], 4)   # ceil(100/32)
        self.assertEqual(profile["epochs"]["3"]["steps"], 10)  # ceil(300/32)
        self.assertEqual(profile["epochs"]["1"]["micro_batches"], 25)

        profile3b = steps["profiles"]["qwen3b"]
        self.assertEqual(profile3b["micro_batch"], 2)
        self.assertEqual(profile3b["grad_accum"], 16)
        self.assertEqual(profile3b["epochs"]["1"]["micro_batches"], 50)
        self.assertEqual(profile3b["epochs"]["2"]["steps"], 7)

    def test_real_pair_counts(self):
        steps = qat_data.steps_for_pairs(3607, 401)
        for name in ("qwen1.5b", "qwen3b"):
            profile = steps["profiles"][name]
            self.assertEqual(profile["epochs"]["1"]["steps"], 113)
            self.assertEqual(profile["epochs"]["2"]["steps"], 226)
            self.assertEqual(profile["epochs"]["3"]["steps"], 339)
            self.assertEqual(profile["val_batches_per_epoch"], 13)

    def test_formula_is_named(self):
        steps = qat_data.steps_for_pairs(10, 1)
        self.assertIn("ceil", steps["formula"])


# ---------------------------------------------------------------------------
# BPE
# ---------------------------------------------------------------------------


class TestBPETokenizer(TokenizerCase):
    def test_fixture_metadata(self):
        info = self.tokenizer.info()
        self.assertEqual(info["vocab_size"], 287)
        self.assertEqual(info["merges"], 31)
        self.assertEqual(info["added_tokens"], 4)
        self.assertEqual(info["special_tokens"], 4)
        self.assertTrue(info["byte_level"])
        self.assertEqual(info["eos_id"], 501)

    def test_known_ids_cyrillic(self):
        # Значения из отчёта 0012: Привет, мир! -> 275,44,281,33.
        self.assertEqual(self.tokenizer.encode("Привет, мир!"), [275, 44, 281, 33])

    def test_known_ids_ascii(self):
        self.assertEqual(self.tokenizer.encode("hello world"), [259, 264])

    def test_special_token_is_one_id(self):
        self.assertEqual(self.tokenizer.encode("<|im_start|>"), [500])
        self.assertEqual(self.tokenizer.encode("<|im_end|>"), [501])

    def test_empty_text(self):
        self.assertEqual(self.tokenizer.encode(""), [])

    def test_pre_tokenize_pieces(self):
        pieces = self.tokenizer.pre_tokenize("hello  world")
        self.assertEqual(pieces, ["hello", " ", " world"])

    def test_old_tokenizer_json_without_model_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            path.write_text(json.dumps({
                "model": {"vocab": {"a": 0, "b": 1}, "merges": []},
            }), encoding="utf-8")
            tokenizer = qat_data.BPETokenizer.load(path)
            self.assertEqual(tokenizer.encode("ab"), [0, 1])

    def test_unsupported_pre_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            path.write_text(json.dumps({
                "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []},
                "pre_tokenizer": {"type": "Split", "pattern": {}},
            }), encoding="utf-8")
            with self.assertRaises(qat_data.TokenizerError):
                qat_data.BPETokenizer.load(path)

    def test_broken_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            path.write_bytes(b"{not json")
            with self.assertRaises(qat_data.TokenizerError):
                qat_data.BPETokenizer.load(path)


@unittest.skipUnless(VTM_TOK.is_file(), "vtm-tok не собран (src/vtm/vtm-tok)")
class TestParityWithVtmTok(TokenizerCase):
    SAMPLES = [
        "Привет, мир!",
        "hello world",
        "Привет 👋 мир 🌍",
        "<|im_start|>user\nКак дела?<|im_end|>\n<|im_start|>assistant\n",
        "def f(x):\n    return x ** 2\n",
        "  несколько   пробелов  ",
        "don't stop",
        "Съешь ещё этих мягких французских булок, да выпей чаю.",
    ]

    def _vtm_ids(self, text):
        result = subprocess.run(
            [str(VTM_TOK), "-t", str(FIXTURE_TOKENIZER), "encode", text],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return [int(part) for part in result.stdout.strip().split(",") if part]

    def test_ids_match(self):
        for text in self.SAMPLES:
            with self.subTest(text=text):
                self.assertEqual(self.tokenizer.encode(text), self._vtm_ids(text))


# ---------------------------------------------------------------------------
# Полный проход
# ---------------------------------------------------------------------------


class TestPrepareEndToEnd(unittest.TestCase):
    def _write_pairs(self, directory, count=60):
        path = Path(directory) / "pairs.jsonl"
        rows = []
        for i in range(count):
            answer = "строка " * (i * 3 + 1)
            rows.append({"question": f"Как сделать шаг {i}?", "answer": answer})
        rows.append({"question": "как  СДЕЛАТЬ шаг 0", "answer": "дубль"})
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    def test_outputs_and_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs_path = self._write_pairs(tmp)
            out = Path(tmp) / "out1"
            stats = qat_data.prepare(
                pairs_path=pairs_path,
                out_dir=out,
                tokenizer_path=FIXTURE_TOKENIZER,
                seed=7,
                max_seq_len=64,
                quiet=True,
            )

            self.assertTrue((out / "train.jsonl").is_file())
            self.assertTrue((out / "val.jsonl").is_file())
            stats_on_disk = json.loads((out / "stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats, stats_on_disk)

            self.assertEqual(stats["source"]["pairs"], 60)
            self.assertEqual(stats["source"]["duplicates_removed"], 1)
            self.assertEqual(stats["split"]["train_pairs"] + stats["split"]["val_pairs"], 60)
            self.assertEqual(stats["split"]["overlap_normalized"], 0)
            self.assertEqual(stats["tokenizer"]["primary"]["vocab_size"], 287)
            self.assertTrue(stats["tokenizer"]["primary"]["is_repo_fixture"])

            train = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            val = [json.loads(line) for line in (out / "val.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(train), stats["split"]["train_pairs"])
            self.assertEqual(len(val), stats["split"]["val_pairs"])
            for row in train + val:
                self.assertNotIn("_key", row)
                self.assertTrue(row["question"])
                self.assertTrue(row["answer"])

            train_keys = {qat_data.normalize_question(row["question"]) for row in train}
            val_keys = {qat_data.normalize_question(row["question"]) for row in val}
            self.assertEqual(train_keys & val_keys, set())

            tokenized = qat_data.tokenize_split(
                train, qat_data.BPETokenizer.load(FIXTURE_TOKENIZER),
                max_seq_len=64,
            )
            self.assertEqual(
                stats["tokenization"]["splits"]["train"]["over_context"]["count"],
                tokenized["over_context"]["count"],
            )
            self.assertEqual(stats["tokenization"]["splits"]["train"]["tokens"], tokenized["tokens"])

            steps = stats["steps"]["raw"]["profiles"]
            self.assertEqual(steps["qwen1.5b"]["micro_batch"], 4)
            self.assertEqual(steps["qwen3b"]["micro_batch"], 2)
            self.assertIn("1", steps["qwen1.5b"]["epochs"])
            self.assertIn("3", steps["qwen1.5b"]["epochs"])

    def test_byte_identical_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs_path = self._write_pairs(tmp)
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            qat_data.prepare(pairs_path=pairs_path, out_dir=first,
                             tokenizer_path=FIXTURE_TOKENIZER, max_seq_len=64, quiet=True)
            qat_data.prepare(pairs_path=pairs_path, out_dir=second,
                             tokenizer_path=FIXTURE_TOKENIZER, max_seq_len=64, quiet=True)
            for name in ("train.jsonl", "val.jsonl", "stats.json"):
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                    f"{name} должен быть побайтно детерминирован",
                )

    def test_input_order_does_not_change_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs_path = self._write_pairs(tmp)
            rows = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines()]
            random.Random(99).shuffle(rows)
            shuffled_path = Path(tmp) / "shuffled.jsonl"
            shuffled_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            first = qat_data.prepare(
                pairs_path=pairs_path, out_dir=Path(tmp) / "a",
                tokenizer_path=FIXTURE_TOKENIZER, quiet=True,
            )
            shuffled = qat_data.prepare(
                pairs_path=shuffled_path, out_dir=Path(tmp) / "b",
                tokenizer_path=FIXTURE_TOKENIZER, quiet=True,
            )
            # Состав половин задаётся хешем вопроса, а не позицией строки.
            self.assertEqual(first["split"], shuffled["split"])

            def split_questions(path):
                return {
                    qat_data.normalize_question(json.loads(line)["question"])
                    for line in (Path(tmp) / path).read_text(encoding="utf-8").splitlines()
                }

            self.assertEqual(split_questions("a/train.jsonl"), split_questions("b/train.jsonl"))
            self.assertEqual(split_questions("a/val.jsonl"), split_questions("b/val.jsonl"))

    def test_cli_help(self):
        result = subprocess.run(
            [sys.executable, str(TOOL), "--help"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--tokenizer", result.stdout)

    def test_main_reports_missing_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs_path = self._write_pairs(tmp, count=4)
            with contextlib.redirect_stderr(io.StringIO()):
                code = qat_data.main([
                    "--pairs", str(pairs_path),
                    "--out-dir", str(Path(tmp) / "out"),
                    "--tokenizer", str(Path(tmp) / "nope.json"),
                    "--quiet",
                ])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
