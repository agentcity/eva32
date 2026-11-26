"""Ударения для Silero: ``eva/ru_stress.py`` и подключение в ``eva/tts.py``.

Запуск::

    python3 -m unittest tests.test_ru_stress -v
    python3 -m unittest discover -s tests

Что проверяется (задача 0222):

* словарь 40+ слов и словоформ: Колобок/Колобка/Колобку, Теремок/Теремка,
  Журавль/Журавля, Алёнушка, Иванушка, Царевич/Царевна, Баба-Яга,
  Сивка-Бурка, Хаврошечка, Морозко, Снегурочка, Ряба, Курочка, Мышка,
  Лягушка-квакушка, Вершки, Корешки, Медведь, Афанасьев, Билибин и другие;
* знак ``+`` ставится ровно перед ударной гласной, регистр сохраняется;
* идемпотентность: уже размеченное слово не получает второй ``+``;
* слова короче двух букв, латиница, числа, код в бэктиках, URL и email не
  трогаются;
* ``EVA_STRESS=0`` отключает разметку;
* внешний ``data/stress.txt`` (``слово<TAB>разметка``) читается с кэшем,
  перекрывает встроенный словарь, а битая строка даёт предупреждение и
  пропускается, не роняя синтез;
* интеграция: в Silero уходит текст с ``+``, а текст ответа/нормализации —
  без плюсов; прямой вызов адаптера тоже размечает.

Тесты не запускают Silero/ElevenLabs/say: адаптер подменяется записывающей
заглушкой, сеть и звук не нужны.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva import ru_stress, tts  # noqa: E402

#: Правдоподобный WAV: сигнатура RIFF и достаточно байт.
FAKE_WAV = b"RIFF" + b"\x00" * 9000

#: 90+ слов и словоформ из словаря задачи: что должно звучать правильно.
CASES = {
    # -- сказки ---------------------------------------------------------- #
    "Колобок": "Колоб+ок",
    "Колобка": "Колобк+а",
    "Колобку": "Колобк+у",
    "Колобком": "Колобк+ом",
    "Колобке": "Колобк+е",
    "Колобки": "Колобк+и",
    "Теремок": "Терем+ок",
    "Теремка": "Теремк+а",
    "Теремку": "Теремк+у",
    "Теремки": "Теремк+и",
    "Терем": "Т+ерем",
    "Журавль": "Жур+авль",
    "Журавля": "Журавл+я",
    "Журавлю": "Журавл+ю",
    "Журавли": "Журавл+и",
    "Алёнушка": "Ал+ёнушка",
    "Алёнушки": "Ал+ёнушки",
    "Аленушка": "Ал+енушка",
    "Иванушка": "Ив+анушка",
    "Иванушку": "Ив+анушку",
    "Иван": "Ив+ан",
    "Иван-царевич": "Ив+ан-цар+евич",
    "Царевич": "Цар+евич",
    "Царевича": "Цар+евича",
    "Царевна": "Цар+евна",
    "Царевну": "Цар+евну",
    "Царевна-лягушка": "Цар+евна-ляг+ушка",
    "Баба-Яга": "Б+аба-Яг+а",
    "Бабы-Яги": "Б+абы-Яг+и",
    "Бабу-Ягу": "Б+абу-Яг+у",
    "Сивка-Бурка": "С+ивка-Б+урка",
    "Сивку-Бурку": "С+ивку-Б+урку",
    "Хаврошечка": "Хавр+ошечка",
    "Хаврошечки": "Хавр+ошечки",
    "Морозко": "Мор+озко",
    "Снегурочка": "Снег+урочка",
    "Снегурочку": "Снег+урочку",
    "Ряба": "Р+яба",
    "Курочка": "К+урочка",
    "Курочку": "К+урочку",
    "Мышка": "М+ышка",
    "Мышку": "М+ышку",
    "Лягушка": "Ляг+ушка",
    "Квакушка": "Квак+ушка",
    "Лягушка-квакушка": "Ляг+ушка-квак+ушка",
    "Вершки": "Вершк+и",
    "Вершков": "Вершк+ов",
    "Корешки": "Корешк+и",
    "Корешков": "Корешк+ов",
    "Корешок": "Кореш+ок",
    "Медведь": "Медв+едь",
    "Медведя": "Медв+едя",
    "Афанасьев": "Афан+асьев",
    "Афанасьева": "Афан+асьева",
    "Билибин": "Бил+ибин",
    "Билибина": "Бил+ибина",
    "Василиса": "Васил+иса",
    "Премудрая": "Прем+удрая",
    "Емеля": "Ем+еля",
    "Дурак": "Дур+ак",
    "Кощей": "Кощ+ей",
    "Бессмертный": "Бессм+ертный",
    "Леший": "Л+еший",
    "Кикимора": "Кик+имора",
    "Водяной": "Водян+ой",
    "Домовой": "Домов+ой",
    "Русалка": "Рус+алка",
    "Змей-Горыныч": "Зм+ей-Гор+ыныч",
    "Горыныч": "Гор+ыныч",
    "Финист": "Фин+ист",
    "Сокол": "С+окол",
    "Марья": "М+арья",
    "Моревна": "Мор+евна",
    "Никита": "Ник+ита",
    "Кожемяка": "Кожем+яка",
    "Петушок": "Петуш+ок",
    "Гребешок": "Гребеш+ок",
    "Лисичка": "Лис+ичка",
    "Сестричка": "Сестр+ичка",
    "Зайка": "З+айка",
    "Козлята": "Козл+ята",
    "Козлёнок": "Козл+ёнок",
    "Репка": "Р+епка",
    "Дедка": "Д+едка",
    "Бабка": "Б+абка",
    "Внучка": "Вн+учка",
    "Жучка": "Ж+учка",
    "Кошка": "К+ошка",
    "Сказка": "Ск+азка",
    "Сказочник": "Ск+азочник",
    # -- частотная лексика ---------------------------------------------- #
    "звонит": "звон+ит",
    "позвонит": "позвон+ит",
    "красивее": "крас+ивее",
    "каталог": "катал+ог",
    "договор": "догов+ор",
    "дефис": "деф+ис",
    "алкоголь": "алког+оль",
    "жалюзи": "жалюз+и",
    "творог": "твор+ог",
    "щавель": "щав+ель",
    "торты": "т+орты",
    "банты": "б+анты",
    "шарфы": "ш+арфы",
    "средства": "ср+едства",
    "документ": "докум+ент",
    "километр": "килом+етр",
    "магазин": "магаз+ин",
    "процент": "проц+ент",
    "эксперт": "эксп+ерт",
    "цемент": "цем+ент",
    "столяр": "стол+яр",
    "недуг": "нед+уг",
    "кухонный": "к+ухонный",
    "украинский": "укра+инский",
    "обеспечение": "обесп+ечение",
    "ходатайство": "ход+атайство",
    "квартал": "кварт+ал",
    "бухгалтер": "бухг+алтер",
    "диспансер": "диспанс+ер",
    "свекла": "св+екла",
    "петля": "петл+я",
}


class ApplyStressTests(unittest.TestCase):
    """Слова, словоформы, регистр и правила разметки."""

    def test_dictionary_has_at_least_forty_words(self):
        dictionary = ru_stress.builtin_stress_dictionary()
        self.assertGreaterEqual(len(dictionary), 40)
        for word in CASES:
            with self.subTest(word=word):
                self.assertIn(word.lower(), dictionary)

    def test_words_and_forms(self):
        for word, expected in CASES.items():
            with self.subTest(word=word):
                self.assertEqual(ru_stress.apply_stress(word), expected)

    def test_sentence(self):
        self.assertEqual(
            ru_stress.apply_stress("Колобок и Теремок"),
            "Колоб+ок и Терем+ок",
        )
        self.assertEqual(
            ru_stress.apply_stress("Колобка позвал Журавль."),
            "Колобк+а позвал Жур+авль.",
        )

    def test_case_is_preserved(self):
        self.assertEqual(ru_stress.apply_stress("колобок"), "колоб+ок")
        self.assertEqual(ru_stress.apply_stress("Колобок"), "Колоб+ок")
        self.assertEqual(ru_stress.apply_stress("КОЛОБОК"), "КОЛОБ+ОК")
        self.assertEqual(ru_stress.apply_stress("Колобка"), "Колобк+а")
        self.assertEqual(ru_stress.apply_stress("БАБА-ЯГА"), "Б+АБА-ЯГ+А")

    def test_idempotent(self):
        source = (
            "Колобок, Баба-Яга и Теремок: звонит километр, "
            "документ и Снегурочка."
        )
        once = ru_stress.apply_stress(source)
        self.assertIn("Колоб+ок", once)
        self.assertIn("Б+аба-Яг+а", once)
        self.assertEqual(ru_stress.apply_stress(once), once)
        self.assertEqual(ru_stress.apply_stress(once + once), once + once)

    def test_already_marked_text_is_untouched(self):
        for text in (
            "Колоб+ок",
            "Б+аба-Яг+а",
            "Ив+ан-цар+евич",
            "Колоб+ок и Терем+ок",
        ):
            with self.subTest(text=text):
                self.assertEqual(ru_stress.apply_stress(text), text)

    def test_short_latin_code_url_and_numbers_are_untouched(self):
        text = (
            "`Колобок` https://example.com/Колобок Kolobok GPT-4 v2.0 "
            "EVA_TTS=0 1500"
        )
        self.assertEqual(ru_stress.apply_stress(text), text)
        # Слово рядом с цифрой/латиницей — часть технического токена.
        self.assertEqual(ru_stress.apply_stress("Колобок2"), "Колобок2")
        self.assertEqual(ru_stress.apply_stress("2Колобок"), "2Колобок")
        self.assertEqual(ru_stress.apply_stress("KolobokКолобок"), "KolobokКолобок")
        # Короткое слово (одна буква) не трогается, даже если бы было в словаре.
        self.assertEqual(ru_stress.apply_stress("Я и он"), "Я и он")

    def test_email_is_untouched(self):
        self.assertEqual(
            ru_stress.apply_stress("Пишите на kolobok@example.com"),
            "Пишите на kolobok@example.com",
        )
        self.assertEqual(
            ru_stress.apply_stress("Почта: колобок@сказка.ру"),
            "Почта: колобок@сказка.ру",
        )

    def test_empty_and_none(self):
        self.assertEqual(ru_stress.apply_stress(""), "")
        self.assertEqual(ru_stress.apply_stress(None), "")

    def test_text_without_dictionary_words_is_unchanged(self):
        for text in ("Привет, мир!", "Как дела?", "Это обычный текст."):
            with self.subTest(text=text):
                self.assertEqual(ru_stress.apply_stress(text), text)

    def test_zero_flag_disables(self):
        for value in ("0", "false", "no", "off", "нет", "выкл"):
            with self.subTest(value=value):
                self.assertEqual(
                    ru_stress.apply_stress("Колобок", env={"EVA_STRESS": value}),
                    "Колобок",
                )
                with mock.patch.dict(os.environ, {"EVA_STRESS": value}):
                    self.assertEqual(ru_stress.apply_stress("Колобок"), "Колобок")

    def test_enabled_by_default_and_by_truthy_flag(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(ru_stress.stress_enabled())
            self.assertEqual(ru_stress.apply_stress("Колобок"), "Колоб+ок")
        for value in ("1", "true", "yes", "on"):
            with self.subTest(value=value):
                self.assertEqual(
                    ru_stress.apply_stress("Колобок", env={"EVA_STRESS": value}),
                    "Колоб+ок",
                )
        self.assertFalse(ru_stress.stress_enabled({"EVA_STRESS": "0"}))
        self.assertTrue(ru_stress.stress_enabled({"EVA_STRESS": "1"}))


class DictionaryFileTests(unittest.TestCase):
    """``data/stress.txt``: чтение, кэш, перекрытие и битые строки."""

    def write(self, content: str) -> Path:
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", delete=False
        )
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        handle.write(content)
        handle.close()
        return Path(handle.name)

    def test_repository_file_is_valid(self):
        entries = ru_stress.load_stress_file(ru_stress.DEFAULT_STRESS_FILE)
        self.assertGreaterEqual(len(entries), 10)
        self.assertEqual(entries["настенька"], "Н+астенька")
        self.assertEqual(entries["муромец"], "М+уромец")

    def test_file_extends_dictionary(self):
        path = self.write("Муромец\tМ+уромец\nСалтан\tСалт+ан\n")
        env = {"EVA_STRESS_FILE": str(path)}
        self.assertEqual(ru_stress.apply_stress("Муромец", env=env), "М+уромец")
        self.assertEqual(ru_stress.apply_stress("Салтан", env=env), "Салт+ан")

    def test_file_overrides_builtin(self):
        path = self.write("Колобок\tКол+обок\n")
        env = {"EVA_STRESS_FILE": str(path)}
        self.assertEqual(ru_stress.apply_stress("Колобок", env=env), "Кол+обок")

    def test_broken_line_logs_and_is_skipped(self):
        path = self.write(
            "# комментарий\n"
            "\n"
            "Мышка\tМ+ышка\n"
            "без таба\n"
            "Колобок\tКолобок\n"
            "Теремок\tТерем+ок\n"
            "Баба-Яга\tБ+аба-Яг+а\n"
        )
        with self.assertLogs(ru_stress.logger, level="WARNING") as logs:
            entries = ru_stress.load_stress_file(path)
        self.assertEqual(entries["мышка"], "М+ышка")
        self.assertEqual(entries["теремок"], "Терем+ок")
        self.assertEqual(entries["баба-яга"], "Б+аба-Яг+а")
        self.assertNotIn("колобок", entries)
        output = "\n".join(logs.output)
        self.assertIn("строк", output)
        self.assertIn("TAB", output)

    def test_plus_not_before_vowel_is_broken_line(self):
        path = self.write("Колобок\tКоло+бок\n")
        with self.assertLogs(ru_stress.logger, level="WARNING") as logs:
            entries = ru_stress.load_stress_file(path)
        self.assertEqual(entries, {})
        self.assertIn("гласной", "\n".join(logs.output))

    def test_missing_file_is_not_an_error(self):
        with self.assertLogs(ru_stress.logger, level="INFO"):
            entries = ru_stress.load_stress_file("/nonexistent/stress.txt")
        self.assertEqual(entries, {})
        self.assertEqual(ru_stress.apply_stress("Колобок"), "Колоб+ок")

    def test_dictionary_is_cached_until_file_changes(self):
        path = self.write("Муромец\tМ+уромец\n")
        env = {"EVA_STRESS_FILE": str(path)}
        first = ru_stress.stress_dictionary(env=env)
        second = ru_stress.stress_dictionary(env=env)
        self.assertIs(first, second)
        self.assertIn("муромец", first)
        # Файл изменился — кэш обязан перечитаться (mtime/размер).
        path.write_text("Муромец\tМ+уромец\nДобрыня\tДобр+ыня\n", encoding="utf-8")
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        refreshed = ru_stress.stress_dictionary(env=env)
        self.assertIsNot(first, refreshed)
        self.assertIn("добрыня", refreshed)


class RecordingSilero(tts.SileroTTS):
    """Silero-заглушка: запоминает текст, который ушёл бы в модель."""

    def __init__(self, *, env=None) -> None:
        super().__init__(env=env)
        self.spoken: list[str] = []

    def resolve(self):  # noqa: ANN201 - подпись не важна для заглушки
        return None

    def synthesize_timed(
        self,
        text: str,
        out_path: str | Path,
        *,
        rate: int | None = None,
        voice: str | None = None,
        timeout: float = tts.DEFAULT_TIMEOUT,
    ) -> tts.SynthesisResult:
        # Так же, как настоящий SileroTTS: разметка — в synthesize_timed.
        self.spoken.append(self._prepare_text(text))
        target = Path(out_path).expanduser()
        target.write_bytes(FAKE_WAV)
        return tts.SynthesisResult(
            path=target, synth_ms=1.0, first_fragment_ms=1.0, engine="silero"
        )


class EngineIntegrationTests(unittest.TestCase):
    """``+`` доходит до синтеза, но не до текста ответа."""

    def test_prepare_text_adds_stress(self):
        engine = tts.SileroTTS(env={})
        self.assertEqual(
            engine._prepare_text("Колобок и Теремок"), "Колоб+ок и Терем+ок"
        )

    def test_silero_gets_marked_text(self):
        engine = RecordingSilero(env={})
        data = tts.synthesize("Колобок и Теремок", engine=engine)
        self.assertEqual(data, FAKE_WAV)
        self.assertEqual(len(engine.spoken), 1)
        self.assertEqual(engine.spoken[0], "Колоб+ок и Терем+ок")
        self.assertIn("+", engine.spoken[0])

    def test_chat_answer_has_no_plus(self):
        answer = "Колобок и Теремок и Баба-Яга"
        self.assertNotIn("+", answer)
        # Ответ пользователю не проходит через apply_stress: только чистка.
        self.assertEqual(tts.normalize_tts_text(answer), answer)
        self.assertEqual(tts._clean_text(answer), answer)
        self.assertNotIn("+", tts.normalize_tts_text(answer))
        engine = RecordingSilero(env={})
        tts.synthesize(answer, engine=engine)
        self.assertIn("+", engine.spoken[0])

    def test_direct_adapter_synthesize_marks_text(self):
        engine = RecordingSilero(env={})
        engine.synthesize("Колобок")
        self.assertEqual(engine.spoken[-1], "Колоб+ок")

    def test_zero_flag_reaches_engine(self):
        engine = RecordingSilero(env={"EVA_STRESS": "0"})
        tts.synthesize("Колобок и Теремок", engine=engine)
        self.assertEqual(engine.spoken[0], "Колобок и Теремок")
        self.assertNotIn("+", engine.spoken[0])

    def test_blank_text_still_raises(self):
        with self.assertRaises(tts.TTSError):
            tts.SileroTTS(env={})._prepare_text("   ")


if __name__ == "__main__":
    unittest.main()
