"""Нормализация чисел для TTS: ``eva/ru_numbers.py`` и подключение в ``eva/tts.py``.

Запуск::

    python3 -m unittest tests.test_tts_numbers -v
    python3 -m unittest discover -s tests

Что проверяется (задача 0205):

* целые 0…10¹² словами, включая формы «тысяча/тысячи/тысяч» и
  «миллион/миллиона/миллионов»;
* дробные «1,5» / «1.5», знаки (%, ₽, «руб.», №), диапазоны и даты;
* числа внутри латиницы/кода/URL не трогаются, пунктуация и пробелы вокруг
  числа сохраняются;
* интеграция: в движок уходит текст со словами, а не с цифрами — как через
  ``tts.synthesize`` (нормализация до выбора движка), так и через прямой
  ``adapter.synthesize`` (нормализация в ``_clean_text``).

Тесты не запускают Silero/ElevenLabs/say: движок подменяется записывающей
заглушкой, сеть и звук не нужны.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva import tts  # noqa: E402
from eva.ru_numbers import cardinal_words, numbers_to_words  # noqa: E402

#: Правдоподобный WAV: сигнатура RIFF и достаточно байт.
FAKE_WAV = b"RIFF" + b"\x00" * 9000


class CardinalWordsTests(unittest.TestCase):
    """Целые числа: базовые разряды и словарные формы."""

    CASES = {
        0: "ноль",
        1: "один",
        2: "два",
        5: "пять",
        11: "одиннадцать",
        21: "двадцать один",
        100: "сто",
        101: "сто один",
        1000: "тысяча",
        1500: "тысяча пятьсот",
        2000: "две тысячи",
        5000: "пять тысяч",
        21000: "двадцать одна тысяча",
        22000: "двадцать две тысячи",
        25000: "двадцать пять тысяч",
        1000000: "миллион",
        2000000: "два миллиона",
        5000000: "пять миллионов",
        10 ** 9: "миллиард",
        10 ** 12: "триллион",
    }

    def test_cardinal_words(self):
        for number, expected in self.CASES.items():
            with self.subTest(number=number):
                self.assertEqual(cardinal_words(number), expected)

    def test_thousand_forms(self):
        self.assertEqual(numbers_to_words("1000"), "тысяча")
        self.assertEqual(numbers_to_words("2000"), "две тысячи")
        self.assertEqual(numbers_to_words("5000"), "пять тысяч")
        self.assertEqual(numbers_to_words("21000"), "двадцать одна тысяча")

    def test_million_forms(self):
        self.assertEqual(numbers_to_words("1000000"), "миллион")
        self.assertEqual(numbers_to_words("2000000"), "два миллиона")
        self.assertEqual(numbers_to_words("5000000"), "пять миллионов")

    def test_too_large_number_is_left_as_is(self):
        # Выше 10¹² разряды не озвучиваем, чтобы не соврать: лучше цифры.
        too_big = str(10 ** 12 + 1)
        self.assertEqual(numbers_to_words(too_big), too_big)

    def test_genitive_for_ranges(self):
        self.assertEqual(cardinal_words(10, case="gen"), "десяти")
        self.assertEqual(cardinal_words(2000, case="gen"), "двух тысяч")
        self.assertEqual(cardinal_words(1500, case="gen"), "тысячи пятисот")


class NumberNormalizationTests(unittest.TestCase):
    """Текст: числа, дробные, знаки, диапазоны и даты."""

    def test_number_inside_sentence(self):
        self.assertEqual(
            numbers_to_words("Цена 1500 рублей"),
            "Цена тысяча пятьсот рублей",
        )
        self.assertEqual(
            numbers_to_words("Скидка 20 процентов"),
            "Скидка двадцать процентов",
        )

    def test_punctuation_and_spaces_are_kept(self):
        self.assertEqual(
            numbers_to_words("Цена 1500 рублей."),
            "Цена тысяча пятьсот рублей.",
        )
        self.assertEqual(
            numbers_to_words("Скидка 20%,"),
            "Скидка двадцать процентов,",
        )
        self.assertEqual(numbers_to_words(" 1500 "), " тысяча пятьсот ")

    def test_fractions(self):
        self.assertEqual(numbers_to_words("1,5"), "одна целая пять десятых")
        self.assertEqual(numbers_to_words("1.5"), "одна целая пять десятых")
        self.assertEqual(numbers_to_words("0,5"), "ноль целых пять десятых")
        self.assertEqual(numbers_to_words("1,1"), "одна целая одна десятая")
        self.assertEqual(numbers_to_words("2,25"), "две целых двадцать пять сотых")

    def test_percent_agreement(self):
        self.assertEqual(numbers_to_words("1%"), "один процент")
        self.assertEqual(numbers_to_words("2%"), "два процента")
        self.assertEqual(numbers_to_words("5%"), "пять процентов")
        self.assertEqual(numbers_to_words("21%"), "двадцать один процент")
        self.assertEqual(numbers_to_words("Скидка 20%"), "Скидка двадцать процентов")

    def test_currency_and_number_sign(self):
        self.assertEqual(
            numbers_to_words("Цена 1500 ₽"), "Цена тысяча пятьсот рублей"
        )
        self.assertEqual(numbers_to_words("1 ₽"), "один рубль")
        self.assertEqual(numbers_to_words("2 руб."), "два рубля")
        self.assertEqual(numbers_to_words("5 руб."), "пять рублей")
        self.assertEqual(numbers_to_words("руб."), "рублей")
        self.assertEqual(numbers_to_words("№ 5"), "номер пять")
        self.assertEqual(numbers_to_words("№5"), "номер пять")

    def test_grouped_thousands(self):
        self.assertEqual(numbers_to_words("1 500 рублей"), "тысяча пятьсот рублей")
        self.assertEqual(numbers_to_words("2 000 000"), "два миллиона")
        self.assertEqual(numbers_to_words("1\u00a0500 ₽"), "тысяча пятьсот рублей")

    def test_ruble_short_form(self):
        self.assertEqual(numbers_to_words("1500 р."), "тысяча пятьсот рублей")

    def test_phone_plus(self):
        self.assertEqual(numbers_to_words("+7"), "плюс семь")
        self.assertIn("плюс семь", numbers_to_words("тел. +7 доб. 5"))

    def test_ranges(self):
        self.assertEqual(numbers_to_words("10-20"), "от десяти до двадцати")
        self.assertEqual(numbers_to_words("от 5 до 10"), "от пяти до десяти")
        self.assertEqual(
            numbers_to_words("от 5 до 10%"), "от пяти до десяти процентов"
        )
        self.assertEqual(numbers_to_words("10 - 20"), "от десяти до двадцати")
        self.assertEqual(
            numbers_to_words("1500-2000"), "от тысячи пятисот до двух тысяч"
        )
        self.assertEqual(
            numbers_to_words("10-20%"), "от десяти до двадцати процентов"
        )
        self.assertEqual(
            numbers_to_words("1000-2000 ₽"),
            "от тысячи до двух тысяч рублей",
        )

    def test_hyphen_inside_word_is_not_a_range(self):
        self.assertEqual(numbers_to_words("какой-то"), "какой-то")
        self.assertEqual(
            numbers_to_words("какой-то 5-10 тест"),
            "какой-то от пяти до десяти тест",
        )
        # Телефон — не диапазон: три группы через дефис.
        phone = numbers_to_words("8-800-555-35-35")
        self.assertNotIn("от восьми", phone)
        self.assertNotIn(" до ", phone)

    def test_dates(self):
        self.assertEqual(
            numbers_to_words("01.02.2026"),
            "первое февраля две тысячи двадцать шесть",
        )
        self.assertEqual(
            numbers_to_words("31.12.2025."),
            "тридцать первое декабря две тысячи двадцать пять.",
        )
        self.assertEqual(
            numbers_to_words("2026-02-01"),
            "первое февраля две тысячи двадцать шесть",
        )
        self.assertEqual(
            numbers_to_words("2026-02"), "февраль две тысячи двадцать шесть"
        )

    def test_code_latin_and_urls_are_untouched(self):
        text = (
            "v2.0 GPT-4 report_2026.pdf `код 1500` "
            "https://example.com/1500 и 1.2.3"
        )
        self.assertEqual(numbers_to_words(text), text)
        self.assertEqual(numbers_to_words("EVA_TTS=0"), "EVA_TTS=0")

    def test_text_without_numbers_is_unchanged(self):
        for text in ("", "без чисел", "Привет, мир!"):
            with self.subTest(text=text):
                self.assertEqual(numbers_to_words(text), text)

    def test_none_is_empty(self):
        self.assertEqual(numbers_to_words(None), "")

    def test_idempotent(self):
        source = (
            "01.02.2026: цена 1500 ₽ (скидка 10-20%), рост 12,5% за 3 дня, "
            "v2.0 и `код 42`"
        )
        once = numbers_to_words(source)
        self.assertNotIn("1500", once)
        self.assertEqual(numbers_to_words(once), once)


class RecordingTTS(tts.TTSAdapter):
    """Движок-заглушка: запоминает текст, который реально уходит в синтез."""

    name = "recording"

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.cleaned: list[str] = []

    def available(self) -> bool:
        return True

    def build_command(
        self,
        text: str,
        out_path: str | Path,
        *,
        rate: int | None = None,
        voice: str | None = None,
    ) -> tuple[list[str], str | None]:
        return [], None

    def synthesize(
        self,
        text: str,
        out_path: str | Path | None = None,
        *,
        rate: int | None = None,
        voice: str | None = None,
        timeout: float = tts.DEFAULT_TIMEOUT,
    ) -> bytes | Path:
        self.spoken.append(text)
        self.cleaned.append(tts._clean_text(text))
        if out_path is not None:
            target = Path(out_path).expanduser()
            target.write_bytes(FAKE_WAV)
            return target
        return FAKE_WAV


class EngineIntegrationTests(unittest.TestCase):
    """В синтез уходит текст со словами, а не с цифрами."""

    def test_synthesize_normalizes_before_engine(self):
        engine = RecordingTTS()
        data = tts.synthesize("Цена 1500 рублей, скидка 20%.", engine=engine)

        self.assertEqual(data, FAKE_WAV)
        self.assertEqual(len(engine.spoken), 1)
        spoken = engine.spoken[0]
        self.assertNotIn("1500", spoken)
        self.assertNotIn("20", spoken)
        self.assertIn("тысяча пятьсот", spoken)
        self.assertIn("двадцать процентов", spoken)
        # Адаптер нормализует повторно, но текст уже словами — не портится.
        self.assertEqual(engine.cleaned[0], spoken)

    def test_synthesize_audio_normalizes_too(self):
        engine = RecordingTTS()
        audio, content_type = tts.synthesize_audio(
            "Абонемент 25000 ₽ в месяц", engine=engine
        )

        self.assertEqual(audio, FAKE_WAV)
        self.assertEqual(content_type, "audio/wav")
        self.assertNotIn("25000", engine.spoken[0])
        self.assertIn("двадцать пять тысяч рублей", engine.spoken[0])

    def test_direct_adapter_synthesize_normalizes_in_clean_text(self):
        # Прямой вызов адаптера (так делает scripts/voice_loop.py) минует
        # _synthesize_with_fallback: нормализация обязана быть в _clean_text.
        engine = RecordingTTS()
        engine.synthesize("Скидка 20% и 1,5 рубля")

        self.assertIn("20", engine.spoken[0])  # сырой текст до чистки
        cleaned = engine.cleaned[0]
        self.assertNotIn("20", cleaned)
        self.assertIn("двадцать процентов", cleaned)
        self.assertIn("одна целая пять десятых", cleaned)

    def test_clean_text_raises_on_blank_text(self):
        with self.assertRaises(tts.TTSError):
            tts._clean_text("   ")


if __name__ == "__main__":
    unittest.main()
