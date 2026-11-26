"""Символы, сокращения, телефоны и email в озвучке (задача 0206).

Запуск::

    python3 -m unittest tests.test_tts_symbols -v
    python3 -m unittest discover -s tests

Что проверяется:

* каждый знак из списка задачи даёт непустую речь, а сам знак из текста
  исчезает: «×» → «умножить на», «≤» → «не больше», «€» → «евро»,
  «½» → «одна вторая», тире/многоточие → запятая, кавычки/©®™ — убираются;
* сокращения («т.д.», «ул.», «г.», «шт.», «тыс.», «мск») читаются словами,
  а не по буквам;
* телефон из 7+ цифр (с +7/8, пробелами, скобками, дефисами) читается
  **по цифрам**, а не как одно большое число;
* email проговаривается: «info собака example точка com»;
* единицы: «км²», «м²», «км/ч», «₽/шт», «°C»;
* нормализация идемпотентна и не трогает код в бэктиках, URL, латинские
  токены/версии/IP (`v2.0`, `GPT-4`, `--prefill`, `EVA_TTS=0`);
* интеграция: в TTS-движок уходит текст словами, а не со знаками;
* живые замеры WAV через ``POST /api/tts`` (только с ``EVA_TTS_LIVE=1``):
  фраза со знаком звучит длиннее, чем та же фраза без него.

Тесты по умолчанию не зовут Silero/ElevenLabs/say: движок подменяется
записывающей заглушкой, сеть и звук не нужны. Живые замеры пропускаются,
пока не выставлен ``EVA_TTS_LIVE=1`` и не поднят chat_server (по умолчанию
``http://127.0.0.1:8765/api/tts``, другой адрес — ``EVA_TTS_URL``).
"""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva import tts  # noqa: E402
from eva.ru_speech import speech_to_words  # noqa: E402

#: Правдоподобный WAV: сигнатура RIFF и достаточно байт.
FAKE_WAV = b"RIFF" + b"\x00" * 9000

#: Замеры WAV — только по явному запросу: нужен живой сервер и голос.
LIVE = os.environ.get("EVA_TTS_LIVE") == "1"
TTS_URL = os.environ.get("EVA_TTS_URL", "http://127.0.0.1:8765/api/tts")


class SymbolWordsTests(unittest.TestCase):
    """Знак из списка задачи → ожидаемая речь (и знака больше нет)."""

    #: (знак, фраза, ожидаемый нормализованный текст). «°C» проверяется как
    #: диграф: одиночный «°» мог бы сработать и без него.
    CASES = (
        ("+", "5 + 3", "пять плюс три"),
        ("-", "-5", "минус пять"),
        ("−", "5 − 3", "пять минус три"),
        ("×", "2 × 3", "два умножить на три"),
        ("*", "2 * 3", "два умножить на три"),
        ("÷", "6 ÷ 2", "шесть делить на два"),
        ("/", "6 / 2", "шесть делить на два"),
        ("=", "2 = 2", "два равно два"),
        ("<", "5 < 7", "пять меньше семь"),
        (">", "7 > 5", "семь больше пять"),
        ("≤", "5 ≤ 7", "пять не больше семь"),
        ("≥", "7 ≥ 5", "семь не меньше пять"),
        ("±", "5 ± 1", "пять плюс-минус один"),
        ("≈", "≈ 5", "примерно пять"),
        ("°", "20°", "двадцать градусов"),
        ("°C", "20°C", "двадцать градусов Цельсия"),
        ("²", "5²", "пять в квадрате"),
        ("³", "2³", "два в кубе"),
        ("€", "50 €", "пятьдесят евро"),
        ("$", "50 $", "пятьдесят долларов"),
        ("£", "50 £", "пятьдесят фунтов"),
        ("₽", "50 ₽", "пятьдесят рублей"),
        ("%", "50%", "пятьдесят процентов"),
        ("‰", "5 ‰", "пять промилле"),
        ("№", "№ 5", "номер пять"),
        ("§", "§ 5", "параграф пять"),
        ("&", "Иванов & Ко", "Иванов и Ко"),
        ("~", "~5", "примерно пять"),
        ("^", "2^3", "два в степени три"),
        ("→", "Шаг 1 → Шаг 2", "Шаг один стрелка вправо Шаг два"),
        ("←", "назад ←", "назад стрелка влево"),
        ("↑", "вверх ↑", "вверх стрелка вверх"),
        ("↓", "вниз ↓", "вниз стрелка вниз"),
        ("½", "½", "одна вторая"),
        ("¼", "¼", "одна четвёртая"),
        ("¾", "¾", "три четвёртых"),
        ("©", "© 2026 Компания", "две тысячи двадцать шесть Компания"),
        ("®", "Бренд ®", "Бренд"),
        ("™", "Текст ™ тут", "Текст тут"),
        ("«»", "«Привет», мир!", "Привет, мир!"),
        ('""', 'Он сказал "да"', "Он сказал да"),
        ("—", "Москва — столица", "Москва, столица"),
        ("–", "Москва – столица", "Москва, столица"),
        ("‒", "Москва ‒ столица", "Москва, столица"),
        ("…", "Подождите…", "Подождите,"),
    )

    def test_symbol_becomes_words(self):
        for symbol, phrase, expected in self.CASES:
            with self.subTest(symbol=symbol):
                self.assertEqual(speech_to_words(phrase), expected)

    def test_no_symbol_survives(self):
        # Критерий приёмки: каждый знак из списка даёт речь, а не молчание.
        for symbol, phrase, _ in self.CASES:
            with self.subTest(symbol=symbol):
                spoken = speech_to_words(phrase)
                for char in symbol:
                    self.assertNotIn(
                        char, spoken, f"знак {char!r} остался в {spoken!r}"
                    )
                self.assertTrue(
                    any(char.isalpha() for char in spoken),
                    f"после нормализации {phrase!r} нет слов: {spoken!r}",
                )

    def test_hyphen_inside_word_is_kept(self):
        # Дефис в слове — не минус и не тире: «по-русски» не ломается.
        self.assertEqual(speech_to_words("по-русски"), "по-русски")
        self.assertEqual(speech_to_words("какой-то"), "какой-то")


class AbbreviationTests(unittest.TestCase):
    """Сокращения читаются словами, а не по буквам."""

    def test_common_abbreviations(self):
        self.assertEqual(
            speech_to_words("И т.д. и т.п., ул. Ленина, г. Москва"),
            "И так далее и тому подобное, улица Ленина, город Москва",
        )
        self.assertEqual(speech_to_words("т.е. это то же"), "то есть это то же")
        self.assertEqual(speech_to_words("2026 г."), "две тысячи двадцать шесть года")
        self.assertEqual(
            speech_to_words("15:00 мск"),
            "пятнадцать ноль по московскому времени",
        )

    def test_units_agree_with_number(self):
        self.assertEqual(speech_to_words("1 шт."), "одна штука")
        self.assertEqual(speech_to_words("2 шт."), "две штуки")
        self.assertEqual(speech_to_words("5 шт."), "пять штук")
        self.assertEqual(speech_to_words("21 шт."), "двадцать одна штука")
        self.assertEqual(speech_to_words("2 ч."), "два часа")
        self.assertEqual(speech_to_words("5 ч."), "пять часов")
        self.assertEqual(speech_to_words("30 мин."), "тридцать минут")
        self.assertEqual(speech_to_words("2 руб."), "два рубля")
        self.assertEqual(speech_to_words("5 коп."), "пять копеек")
        self.assertEqual(speech_to_words("15 тыс."), "пятнадцать тысяч")
        self.assertEqual(speech_to_words("2 млн"), "два миллиона")
        self.assertEqual(speech_to_words("3 млрд"), "три миллиарда")

    def test_bare_abbreviations(self):
        # Сокращение без числа: множественная форма (так пишут в шапках цен).
        self.assertEqual(speech_to_words("коп."), "копеек")
        self.assertEqual(speech_to_words("мин."), "минут")
        self.assertEqual(speech_to_words("млн"), "миллионов")
        self.assertEqual(speech_to_words("млрд"), "миллиардов")
        self.assertEqual(speech_to_words("мск"), "по московскому времени")

    def test_grouped_thousands_with_unit(self):
        # «1 500 шт.» — разряды, а не «1500» и не телефон.
        self.assertEqual(speech_to_words("1 500 шт."), "тысяча пятьсот штук")
        self.assertEqual(
            speech_to_words("1 500 000 руб."), "миллион пятьсот тысяч рублей"
        )


class PhoneTests(unittest.TestCase):
    """Телефон — по цифрам, а не одним числом."""

    ELEVEN = "восемь девять девять девять один два три четыре пять шесть семь"

    def test_eleven_digits_by_digits(self):
        self.assertEqual(speech_to_words("8 999 123 45 67"), self.ELEVEN)
        # Никакого «восемь миллиардов…»: цифры читаются по одной.
        self.assertNotIn("миллиард", speech_to_words("8 999 123 45 67"))
        self.assertIn("восемь девять", speech_to_words("8 999 123 45 67"))

    def test_country_code_and_separators(self):
        plus_seven = "плюс семь " + self.ELEVEN[len("восемь ") :]
        self.assertEqual(speech_to_words("+7 (999) 123-45-67"), plus_seven)
        self.assertEqual(speech_to_words("+79991234567"), plus_seven)
        self.assertEqual(
            speech_to_words("8-800-555-35-35"),
            "восемь восемь ноль ноль пять пять пять три пять три пять",
        )
        self.assertEqual(
            speech_to_words("8 (800) 555-35-35"),
            "восемь восемь ноль ноль пять пять пять три пять три пять",
        )

    def test_not_a_phone(self):
        # Разряды, дата и диапазон телефоном не считаются.
        self.assertEqual(speech_to_words("1 500 000"), "миллион пятьсот тысяч")
        self.assertEqual(
            speech_to_words("2026-02-01"), "первое февраля две тысячи двадцать шесть"
        )
        self.assertEqual(speech_to_words("10-20"), "от десяти до двадцати")
        self.assertEqual(speech_to_words("1500-2000"), "от тысячи пятисот до двух тысяч")


class EmailTests(unittest.TestCase):
    """Email проговаривается, а не выпадает целиком."""

    def test_email_reads_aloud(self):
        self.assertEqual(
            speech_to_words("Пишите на info@example.com"),
            "Пишите на info собака example точка com",
        )
        self.assertEqual(
            speech_to_words("user.name+t@sub.example.ru"),
            "user.name+t собака sub точка example точка ru",
        )

    def test_email_in_code_is_untouched(self):
        text = "`info@example.com` (код)"
        self.assertEqual(speech_to_words(text), text)


class UnitTests(unittest.TestCase):
    """Единицы измерения — словами."""

    def test_area_and_speed(self):
        self.assertEqual(speech_to_words("5 км²"), "пять километров в квадрате")
        self.assertEqual(speech_to_words("10 м²"), "десять метров в квадрате")
        self.assertEqual(speech_to_words("5 км³"), "пять километров в кубе")
        self.assertEqual(speech_to_words("60 км/ч"), "шестьдесят километров в час")
        self.assertEqual(speech_to_words("10 м/с"), "десять метров в секунду")

    def test_price_per_unit(self):
        self.assertEqual(speech_to_words("₽/шт"), "рублей за штуку")
        self.assertEqual(speech_to_words("₽/кг"), "рублей за килограмм")
        self.assertEqual(speech_to_words("50₽/шт"), "пятьдесят рублей за штуку")

    def test_celsius(self):
        self.assertEqual(speech_to_words("1 °C"), "один градус Цельсия")
        self.assertEqual(speech_to_words("5 °C"), "пять градусов Цельсия")
        self.assertEqual(speech_to_words("20 °C"), "двадцать градусов Цельсия")


class SafetyTests(unittest.TestCase):
    """Код, URL и латинские токены не нормализуются."""

    #: Код в бэктиках, URL, версия/IP, латинские токены и флаги — как есть.
    UNTOUCHED = (
        "v2.0 GPT-4 report_2026.pdf `код 1500 × 2` "
        "https://example.com/5?a=1 1.2.3 --prefill EVA_TTS=0"
    )

    def test_code_url_and_tech_tokens(self):
        self.assertEqual(speech_to_words(self.UNTOUCHED), self.UNTOUCHED)
        self.assertEqual(speech_to_words("C# и C++"), "C# и C++")
        self.assertEqual(speech_to_words("don't touch"), "don't touch")

    def test_url_is_not_rewritten(self):
        url = "https://example.com/2×3?a=5#x"
        self.assertEqual(speech_to_words(url), url)

    def test_empty_and_none(self):
        self.assertEqual(speech_to_words(""), "")
        self.assertEqual(speech_to_words(None), "")

    def test_idempotent(self):
        source = (
            "01.02.2026: 2 × 3 = 6, скидка 10-20% (≈ 50 €), 5 км², 60 км/ч, "
            "₽/шт, info@example.com, тел. 8 999 123 45 67, т.д., "
            "Москва — столица…, v2.0 и `код 1500`"
        )
        once = speech_to_words(source)
        self.assertEqual(speech_to_words(once), once)
        for marker in (
            "умножить на",
            "равно",
            "процентов",
            "евро",
            "километров в квадрате",
            "километров в час",
            "рублей за штуку",
            "собака",
        ):
            self.assertIn(marker, once)
        self.assertNotIn("×", once)
        self.assertNotIn("€", once)
        self.assertNotIn("₽", once)
        self.assertNotIn("@example", once)


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
    """В синтез уходит текст словами, а не со знаками."""

    def test_synthesize_normalizes_before_engine(self):
        engine = RecordingTTS()
        data = tts.synthesize("Цена 50 €, 5 км² и info@example.com", engine=engine)

        self.assertEqual(data, FAKE_WAV)
        spoken = engine.spoken[0]
        self.assertIn("пятьдесят евро", spoken)
        self.assertIn("километров в квадрате", spoken)
        self.assertIn("info собака example точка com", spoken)
        for symbol in ("€", "²", "@"):
            self.assertNotIn(symbol, spoken)
        # Адаптер нормализует повторно, но текст уже словами — не портится.
        self.assertEqual(engine.cleaned[0], spoken)

    def test_direct_adapter_synthesize_normalizes_in_clean_text(self):
        # Прямой вызов адаптера (scripts/voice_loop.py) минует
        # _synthesize_with_fallback: нормализация обязана быть в _clean_text.
        engine = RecordingTTS()
        engine.synthesize("2 × 3, тел. 8 999 123 45 67, т.д.")

        cleaned = engine.cleaned[0]
        self.assertIn("два умножить на три", cleaned)
        self.assertIn("восемь девять девять девять", cleaned)
        self.assertIn("так далее", cleaned)
        self.assertNotIn("×", cleaned)

    def test_normalize_tts_text_matches_pipeline(self):
        text = "№ 5 в очереди: 50 € за 5 км² — быстро…"
        self.assertEqual(tts.normalize_tts_text(text), speech_to_words(text))


def _wav_seconds(data: bytes) -> float:
    """Длительность WAV в секундах (0.0 — не WAV)."""
    try:
        with wave.open(io.BytesIO(data)) as wav:
            return wav.getnframes() / float(wav.getframerate() or 1)
    except (wave.Error, EOFError):
        return 0.0


@unittest.skipUnless(
    LIVE,
    "живые замеры: EVA_TTS_LIVE=1 и запущенный tools/chat_server.py",
)
class LiveWavTests(unittest.TestCase):
    """Живые замеры через ``POST /api/tts``: знак добавляет речи.

    «До» — та же фраза без знака (именно её слышали раньше: знак выпадал),
    «после» — фраза со знаком. Для «#тег» замер невозможен: серверная чистка
    ``clean_tts_text`` заменяет ``#`` пробелом до нормализации (см. отчёт),
    поэтому решётка проверяется тестом нормализованного текста выше.
    """

    PAIRS = (
        ("2 × 3", "2 3"),
        ("5 км²", "5 км"),
        ("Цена 50 €", "Цена 50"),
        ("50 $", "50"),
        ("И т.д.", "И т"),
        ("№ 5 в очереди", "5 в очереди"),
    )

    PHRASES = (
        "2 × 3",
        "5 км²",
        "Цена 50 €",
        "8 999 123 45 67",
        "info@example.com",
        "т.д.",
    )

    def _tts(self, text: str) -> bytes:
        payload = json.dumps({"text": text, "voice": "baya"}).encode("utf-8")
        request = urllib.request.Request(
            TTS_URL, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                self.assertEqual(response.status, 200)
                return response.read()
        except (urllib.error.URLError, OSError) as exc:
            self.skipTest(f"/api/tts недоступен ({TTS_URL}): {exc}")

    def test_symbol_makes_phrase_longer(self):
        for voiced_text, dropped_text in self.PAIRS:
            with self.subTest(text=voiced_text):
                voiced = self._tts(voiced_text)
                dropped = self._tts(dropped_text)
                self.assertGreater(
                    _wav_seconds(voiced),
                    _wav_seconds(dropped),
                    f"«{voiced_text}» не длиннее, чем «{dropped_text}»",
                )

    def test_normalized_text_sounds_the_same(self):
        for text in self.PHRASES:
            with self.subTest(text=text):
                normalized = tts.normalize_tts_text(text)
                raw = _wav_seconds(self._tts(text))
                norm = _wav_seconds(self._tts(normalized))
                self.assertGreater(raw, 0.0)
                # Один и тот же текст — один и тот же звук (допуск на округление
                # кадров WAV); сервер нормализует фразу сам.
                self.assertAlmostEqual(raw, norm, delta=max(0.05, raw * 0.02))


if __name__ == "__main__":
    unittest.main()
