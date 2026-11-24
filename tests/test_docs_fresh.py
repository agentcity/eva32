"""Свежесть документации: парсер «Актуально на», возраст, ссылки, CLI.

Задача 0164. Тесты не зависят от «сегодня»: дата передаётся в функцию или в
`--today` явно, поэтому набор не покраснеет через месяц сам по себе.

Отдельно проверяется главное обещание механизма: любой документ из `docs/`
(кроме `docs/adr/`) и корневой `README.md` несут разбираемую дату. Возраст в
этом тесте не проверяется — иначе он бы ломался при каждом обновлении docs.
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import check_docs_dates as checker  # noqa: E402

TODAY = date(2026, 11, 24)


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class ParseTests(unittest.TestCase):
    def test_plain(self):
        text = "# Док\n\n> Актуально на 2026-11-24.\n"
        self.assertEqual(checker.parse_actual_dates(text), [TODAY])

    def test_bold(self):
        text = "**Актуально на 2026-11-01**\n"
        self.assertEqual(
            checker.parse_actual_dates(text), [date(2026, 11, 1)]
        )

    def test_quote_and_extra_words(self):
        text = "> Актуально на 2026-11-24, проверено скриптом.\n"
        self.assertEqual(checker.parse_actual_dates(text), [TODAY])

    def test_no_date(self):
        with self.assertRaises(checker.Problem):
            checker.actual_date("# Док\n\nБез даты.\n")

    def test_invalid_calendar_date(self):
        with self.assertRaises(ValueError):
            checker.parse_actual_dates("Актуально на 2026-02-30\n")

    def test_two_different_dates(self):
        text = "Актуально на 2026-11-24\nАктуально на 2026-11-01\n"
        with self.assertRaises(checker.Problem) as ctx:
            checker.actual_date(text)
        self.assertIn("несколько", str(ctx.exception))

    def test_two_same_dates_ok(self):
        text = "Актуально на 2026-11-24\nАктуально на 2026-11-24\n"
        self.assertEqual(checker.actual_date(text), TODAY)


class AgeTests(unittest.TestCase):
    def check(self, found: date, max_age: int = 30) -> list[str]:
        text = f"Актуально на {found.isoformat()}\n"
        _, problems = checker.check_date(Path("x.md"), text, TODAY, max_age)
        return problems

    def test_fresh(self):
        self.assertEqual(self.check(TODAY), [])

    def test_exactly_max_age_ok(self):
        self.assertEqual(self.check(date(2026, 10, 25)), [])

    def test_older_than_max_age(self):
        problems = self.check(date(2026, 10, 24))
        self.assertEqual(len(problems), 1)
        self.assertIn("устарел", problems[0])

    def test_future(self):
        problems = self.check(date(2026, 11, 25))
        self.assertEqual(len(problems), 1)
        self.assertIn("будущем", problems[0])

    def test_custom_max_age(self):
        self.assertEqual(self.check(date(2026, 8, 1), max_age=180), [])


class LinkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write(self.root, "docs/target.md", "Актуально на 2026-11-24\n")
        write(self.root, "tools/tool.py", "")

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self, text: str) -> list[str]:
        doc = self.root / "docs" / "doc.md"
        doc.write_text(text, encoding="utf-8")
        _, problems = checker.check_links(doc, text, self.root)
        return problems

    def test_relative_link_ok(self):
        self.assertEqual(self.run_check("[цель](target.md)\n"), [])

    def test_broken_link(self):
        problems = self.run_check("[цель](missing.md)\n")
        self.assertEqual(len(problems), 1)
        self.assertIn("битая ссылка", problems[0])

    def test_external_and_anchor_skipped(self):
        text = "[сеть](https://example.com)\n[якорь](#раздел)\n"
        self.assertEqual(self.run_check(text), [])

    def test_link_with_anchor_checks_file(self):
        self.assertEqual(self.run_check("[цель](target.md#L10)\n"), [])
        problems = self.run_check("[цель](missing.md#L10)\n")
        self.assertEqual(len(problems), 1)

    def test_absolute_link_rejected(self):
        problems = self.run_check("[цель](/docs/target.md)\n")
        self.assertEqual(len(problems), 1)
        self.assertIn("абсолютный", problems[0])

    def test_code_path_ok(self):
        self.assertEqual(self.run_check("см. `tools/tool.py`\n"), [])

    def test_code_path_broken(self):
        problems = self.run_check("см. `tools/missing.py`\n")
        self.assertEqual(len(problems), 1)
        self.assertIn("битая ссылка", problems[0])

    def test_code_path_ignored_for_unknown_root(self):
        self.assertEqual(self.run_check("см. `runs/q4/3b.vtm`\n"), [])

    def test_code_path_with_symbol_checks_file(self):
        # `path::symbol` — ссылка на файл; если файла нет, это битая ссылка.
        write(self.root, "tests/test_x.py", "")
        self.assertEqual(
            self.run_check("см. `tests/test_x.py::TestY`\n"), []
        )
        problems = self.run_check("см. `tests/missing.py::TestY`\n")
        self.assertEqual(len(problems), 1)

    def test_links_inside_code_fence_ignored(self):
        text = "```\n[пример](missing.md)\n```\n"
        self.assertEqual(self.run_check(text), [])


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = checker.main(
                ["--root", str(self.root), "--today", "2026-11-24", *argv]
            )
        return code, out.getvalue(), err.getvalue()

    def test_fresh_tree_ok(self):
        write(self.root, "README.md", "# Корень\n\nАктуально на 2026-11-24\n")
        write(self.root, "docs/a.md", "# A\n\nАктуально на 2026-11-24\n")
        code, out, err = self.run_cli()
        self.assertEqual(code, 0, err)
        self.assertIn("2 документов", out)
        self.assertIn("OK", out)

    def test_stale_doc_fails(self):
        write(self.root, "docs/a.md", "# A\n\nАктуально на 2026-10-01\n")
        code, _, err = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("устарел", err)

    def test_future_doc_fails(self):
        write(self.root, "docs/a.md", "# A\n\nАктуально на 2027-01-01\n")
        code, _, err = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("будущем", err)

    def test_missing_date_fails(self):
        write(self.root, "docs/a.md", "# A\n")
        code, _, err = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("нет строки", err)

    def test_adr_excluded(self):
        write(self.root, "docs/adr/0001-x.md", "# ADR\n\n- Дата: 2020-01-01\n")
        write(self.root, "docs/a.md", "Актуально на 2026-11-24\n")
        code, out, err = self.run_cli()
        self.assertEqual(code, 0, err)
        self.assertIn("1 документов", out)

    def test_broken_link_fails_and_no_links_skips(self):
        write(self.root, "docs/a.md", "Актуально на 2026-11-24\n\n[x](nope.md)\n")
        code, _, err = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("битая ссылка", err)
        code, _, _ = self.run_cli("--no-links")
        self.assertEqual(code, 0)

    def test_checked_links_counted(self):
        write(self.root, "docs/a.md", "Актуально на 2026-11-24\n")
        write(self.root, "docs/b.md", "[a](a.md)\n\nАктуально на 2026-11-24\n")
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("проверено 1", out)

    def test_empty_docs_fails(self):
        self.root.mkdir(exist_ok=True)
        code, _, err = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("не найдено ни одного документа", err)


class RealDocsTests(unittest.TestCase):
    """Документы репозитория: дата есть и разбирается (возраст не проверяем)."""

    def test_repo_layout(self):
        docs = ROOT / "docs"
        self.assertTrue(docs.is_dir())
        self.assertTrue((docs / "README.md").is_file(), "нет индекса docs/README.md")

    def test_every_doc_has_parseable_date(self):
        missing = []
        for path in checker.docs_files(ROOT):
            text = path.read_text(encoding="utf-8")
            try:
                checker.actual_date(text)
            except (checker.Problem, ValueError) as exc:
                missing.append(f"{path.relative_to(ROOT)}: {exc}")
        self.assertEqual(missing, [], "документы без даты:\n" + "\n".join(missing))

    def test_adr_are_not_required(self):
        adr = ROOT / "docs" / "adr"
        if not adr.is_dir():
            self.skipTest("нет docs/adr")
        files = checker.docs_files(ROOT)
        for path in files:
            self.assertNotIn("adr", path.relative_to(ROOT).parts)


if __name__ == "__main__":
    unittest.main()
