"""CLI гибрида: парсер, ответ через скилл, человеческая ошибка без модели."""
import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eva import cli  # noqa: E402


class TestParser(unittest.TestCase):
    def test_ask_defaults(self):
        args = cli.build_parser().parse_args(["ask", "привет"])
        self.assertEqual(args.text, "привет")
        self.assertFalse(args.explain)

    def test_serve_defaults(self):
        args = cli.build_parser().parse_args(["serve"])
        self.assertEqual(args.port, 8010)
        self.assertEqual(args.host, "127.0.0.1")

    def test_requires_command(self):
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args([])


class TestAsk(unittest.TestCase):
    def test_skill_answers_without_model(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(["ask", "Сколько стоит тариф Старт?", "--explain"])
        self.assertEqual(code, 0)
        self.assertIn("990", out.getvalue())
        self.assertIn("маршрут:  skill", err.getvalue())
        self.assertIn("0.0000", err.getvalue())

    def test_model_missing_is_human_error(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(["ask", "Почему небо голубое? Объясни подробно, что там с рассеянием"])
        self.assertEqual(code, 2)
        self.assertIn("не могу ответить", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())


if __name__ == "__main__":
    unittest.main()
