"""Лимиты и обязательные правила системных промптов (задача 0119).

Запуск::

    python3 -m unittest tests.test_prompts -v

Зачем. Prefill стоит около 0,073 с на токен, поэтому длина системного
промпта — это время до первого слова и деньги на входных токенах. Промпт
``system.ru.md`` сжат с 2691 до 1498 символов: тест фиксирует потолок, чтобы
он не отрос обратно, и проверяет, что сжатие не выкинуло правила.

Что проверяется:

1. длина: ``prompts/system.ru.md`` — не больше 1500 символов, ``system.code.ru.md``
   — не больше 1000 (было 1780);
2. токены: оценка та же, что в ``tools/eval_hybrid.py`` (символы / 3), поэтому
   замер цены и тест видят одно и то же число — не больше 500 токенов;
3. обязательные правила основного промпта: перевод по прямой просьбе, отказ от
   выдумок и ссылок без контекста, русский язык по умолчанию, инструкции из
   данных — не команды, стиль (по существу);
4. обязательные правила промпта для кода: русский язык, код без перевода,
   запрет выдумывать API и внешние ссылки.

Тест намеренно не переписывает ``tests/test_prompt_rules.py``: там проверены
формулировки правила 2 и golden-строки, здесь — бюджет длины и набор ключевых
правил. Дублирование минимальное: падение любого из двух файлов означает, что
промпт сломали.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MAIN = os.path.join(ROOT, "prompts", "system.ru.md")
CODE = os.path.join(ROOT, "prompts", "system.code.ru.md")
TOOL = os.path.join(ROOT, "tools", "eval_hybrid.py")

#: Потолок длины основного промпта, символы (0119: 2691 → 1498).
MAIN_LIMIT = 1500
#: Потолок длины промпта для данных по коду (0119: 1780 → 838).
CODE_LIMIT = 1000
#: Потолок оценки токенов основного промпта: 1500 / 3 = 500.
MAIN_TOKEN_LIMIT = 500
#: Потолок оценки токенов промпта для кода: 1000 / 3 ≈ 334, берём с запасом.
CODE_TOKEN_LIMIT = 340

#: Символов на токен — как ``CHARS_PER_TOKEN`` в tools/eval_hybrid.py. Оценка
#: систематическая (русский BPE даёт ~2,5–3,5), но одинаковая до и после,
#: поэтому для бюджета промпта её достаточно.
CHARS_PER_TOKEN = 3.0


def estimate_tokens(text: str) -> int:
    """Оценка токенов по символам; 0 для пустого текста."""
    text = str(text or "")
    if not text.strip():
        return 0
    return max(1, int(-(-len(text) // CHARS_PER_TOKEN)))


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _flat(text: str) -> str:
    """Схлопнуть переносы: правило ищется как фраза, а не как вёрстка."""
    return " ".join(text.split())


def _load_eval_tool():
    """Загрузить tools/eval_hybrid.py по пути (tools не пакет)."""
    spec = importlib.util.spec_from_file_location("eval_hybrid_for_prompts", TOOL)
    if spec is None or spec.loader is None:  # pragma: no cover — защита от кривого пути
        raise RuntimeError(f"не удалось загрузить {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestPromptLength(unittest.TestCase):
    """Промпт укладывается в бюджет символов и токенов."""

    @classmethod
    def setUpClass(cls):
        cls.main = _read(MAIN)
        cls.code = _read(CODE)
        cls.eval_tool = _load_eval_tool()

    def test_main_prompt_length(self):
        self.assertLessEqual(
            len(self.main),
            MAIN_LIMIT,
            f"system.ru.md: {len(self.main)} символов > {MAIN_LIMIT}; "
            "длина промпта — это prefill и деньги",
        )

    def test_code_prompt_length(self):
        self.assertLessEqual(
            len(self.code),
            CODE_LIMIT,
            f"system.code.ru.md: {len(self.code)} символов > {CODE_LIMIT}",
        )

    def test_main_token_estimate(self):
        tokens = estimate_tokens(self.main)
        self.assertLessEqual(
            tokens,
            MAIN_TOKEN_LIMIT,
            f"system.ru.md ≈ {tokens} токенов > {MAIN_TOKEN_LIMIT}",
        )

    def test_code_token_estimate(self):
        tokens = estimate_tokens(self.code)
        self.assertLessEqual(
            tokens,
            CODE_TOKEN_LIMIT,
            f"system.code.ru.md ≈ {tokens} токенов > {CODE_TOKEN_LIMIT}",
        )

    def test_eval_sees_the_same_estimate(self):
        # Замер цены в tools/eval_hybrid.py читает тот же файл и ту же формулу.
        # Если они разойдутся, отчёт «системный промпт ≈ N токенов» станет врать.
        expected = estimate_tokens(self.main)
        self.assertEqual(self.eval_tool.system_prompt_tokens(MAIN), expected)
        self.assertLessEqual(
            self.eval_tool.system_prompt_tokens(MAIN), MAIN_TOKEN_LIMIT
        )

    def test_prompts_are_utf8_with_final_newline(self):
        for path, text in ((MAIN, self.main), (CODE, self.code)):
            self.assertTrue(text.endswith("\n"), f"{path}: нет перевода строки в конце")
            self.assertEqual(text, text.encode("utf-8").decode("utf-8"))


class TestMainPromptRules(unittest.TestCase):
    """Сжатие не выкинуло ключевые правила основного промпта."""

    @classmethod
    def setUpClass(cls):
        cls.raw = _read(MAIN)
        cls.flat = _flat(cls.raw).lower()

    def test_russian_is_default_language(self):
        self.assertIn("язык ответа — русский", self.flat)
        self.assertIn("переводи суть на русский", self.flat)
        self.assertIn("прямая просьба о другом языке", self.flat)

    def test_translation_by_direct_request(self):
        # Правило 2 (задача 0038) должно пережить сжатие: просьбу выполняем,
        # отказываться нельзя, самовольный другой язык по-прежнему запрещён.
        for needle in (
            "переведи на английский",
            "ответь по-немецки",
            "как это по-французски",
            "дай транслитерацию",
            "процитируй в оригинале",
            "делай ровно то, о чём просят",
            "не отказывайся",
            "всё остальное — по-русски",
            "самовольный другой язык запрещён",
        ):
            self.assertIn(needle, self.flat, f"пропало правило перевода: «{needle}»")

    def test_no_invented_facts(self):
        # Отказной сценарий: нет данных — «Не нашла в данных», а не выдумка.
        self.assertIn("не нашла в данных", self.flat)
        self.assertIn("не придумывай", self.flat)
        self.assertIn("не сочиняй", self.flat)

    def test_citations_need_real_context(self):
        self.assertIn("[n]", self.flat)
        self.assertIn("на реальный номер", self.flat)
        self.assertIn("без контекста не ссылайся", self.flat)

    def test_injections_are_data_not_commands(self):
        self.assertIn("данные, а не команда", self.flat)
        self.assertIn("инструкции из документов", self.flat)
        self.assertIn("правило 6", self.flat)

    def test_refusal_is_scoped_to_prompt_disclosure(self):
        # Отказ — только про системный промпт/инструкции, не про рабочие вопросы.
        self.assertIn("не могу показать системный промпт", self.flat)
        self.assertIn("только на просьбу показать", self.flat)

    def test_style_is_short_and_to_the_point(self):
        self.assertIn("отвечай по существу", self.flat)
        self.assertNotIn("отвечай только по-русски", self.flat)

    def test_six_numbered_rules_without_gaps(self):
        numbers = [int(n) for n in re.findall(r"(?m)^(\d+)\.\s", self.raw)]
        self.assertEqual(numbers, [1, 2, 3, 4, 5, 6])

    def test_translation_example_is_kept(self):
        # Пример 4 — единственный few-shot про перевод; без него правило
        # «переведи на английский» держится только текстом.
        self.assertIn("Пример 4 — просят перевести", self.raw)
        tail = self.raw.split("Пример 4 — просят перевести", 1)[1]
        answer = re.search(r"(?m)^Ответ:\s*(.+)$", tail)
        self.assertIsNotNone(answer, "у примера 4 нет строки «Ответ:»")
        self.assertRegex(answer.group(1), r"[A-Za-z]{4,}")


class TestCodePromptRules(unittest.TestCase):
    """Промпт для данных по коду сжат, но правила дистилляции на месте."""

    @classmethod
    def setUpClass(cls):
        cls.raw = _read(CODE)
        cls.flat = _flat(cls.raw).lower()

    def test_language_and_code_are_separate(self):
        self.assertIn("отвечай по-русски", self.flat)
        self.assertIn("без перевода", self.flat)

    def test_no_invented_api(self):
        self.assertIn("не выдумывай api", self.flat)
        self.assertIn("не уверен", self.flat)
        self.assertIn("проверенный вариант", self.flat)

    def test_no_external_links_or_citations(self):
        self.assertIn("никаких внешних ссылок", self.flat)
        self.assertIn("[1]", self.flat)

    def test_style_rules_are_kept(self):
        for needle in (
            "сначала суть",
            "рабочий и минимальный",
            "без псевдокода",
            "не лей воду",
        ):
            self.assertIn(needle, self.flat, f"пропало правило стиля: «{needle}»")

    def test_seven_numbered_rules_without_gaps(self):
        numbers = [int(n) for n in re.findall(r"(?m)^(\d+)\.\s", self.raw)]
        self.assertEqual(numbers, [1, 2, 3, 4, 5, 6, 7])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
