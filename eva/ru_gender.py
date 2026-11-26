"""Женский род Евы в первом лице (задача 0207).

Ева — девушка, но 3B-модель не держит род из промпта и пишет о себе «я понял»,
«готов», «сделал», «уверен». Промпт учит правилу (``prompts/system*.md``), а
этот модуль — страховка: он заменяет мужские формы на женские **только в
первом лице**, то есть когда слово стоит после «я» (в том числе через одну-две
частицы: «я уже», «я всё», «я не», «я бы», «я же», «я так»). Всё остальное —
«он понял», «ты готов», «вы готовы», «она сказала» — остаётся как есть.

Что не трогается:

* третье и второе лицо (там род не Евы) — правило работает от «я», а не от
  списка слов;
* цитаты в кавычках «…», „…“, "…" и код в бэктиках (строчный `` `…` `` и
  ограждённые блоки): чужие слова и код не переписываются, даже если внутри
  есть «я понял»;
* уже женские формы: :func:`feminize` идемпотентна, второй прогон ничего не
  меняет.

Примеры::

    >>> feminize("Я понял, сейчас проверю.")
    'Я поняла, сейчас проверю.'
    >>> feminize("Я не уверен, что готов.")
    'Я не уверена, что готова.'
    >>> feminize("Он сказал: «я понял».")
    'Он сказал: «я понял».'
    >>> feminize("`я понял` — это код")
    '`я понял` — это код'

Стриминг: :class:`GenderStreamer` принимает куски ответа и отдаёт готовый
текст. Решение о слове зависит от предыдущего слова, поэтому поток держит
только **последнее недописанное слово** (и хвост бэктиков — чтобы не
разрезать «```» на два «`»). Так дельты не рвутся: слово целиком уходит
одним куском, а его граница — пробел или пунктуация.

Формы: прошедшее время (-л → -ла), краткие прилагательные (готов → готова,
уверен → уверена, рад → рада, должен → должна, обязан → обязана) и
возвратные глаголы (-лся → -лась). Регистр сохраняется: «Понял» → «Поняла».
"""

from __future__ import annotations

import re

__all__ = ["GenderStreamer", "feminize"]

#: Мужская форма → женская. Только слова, которые Ева говорит о себе;
#: пары намеренно перечислены явно, а не выводятся из окончаний: «-л → -ла»
#: дало бы «стол → стола», а «был» в первом лице не встречается.
MASCULINE_TO_FEMININE: dict[str, str] = {
    "понял": "поняла",
    "сделал": "сделала",
    "смог": "смогла",
    "успел": "успела",
    "готов": "готова",
    "уверен": "уверена",
    "рад": "рада",
    "должен": "должна",
    "обязан": "обязана",
    "нашёл": "нашла",
    "нашел": "нашла",
    "посмотрел": "посмотрела",
    "проверил": "проверила",
    "подумал": "подумала",
    "ответил": "ответила",
    "сказал": "сказала",
    "хотел": "хотела",
    "решил": "решила",
    "начал": "начала",
    "закончил": "закончила",
    "забыл": "забыла",
    "помнил": "помнила",
    "знал": "знала",
    "видел": "видела",
    "услышал": "услышала",
    "пришёл": "пришла",
    "пришел": "пришла",
    "ушёл": "ушла",
    "ушел": "ушла",
    "взял": "взяла",
    "дал": "дала",
    "ждал": "ждала",
    "исправил": "исправила",
    "добавил": "добавила",
    "убрал": "убрала",
    # Возвратные глаголы, которые не сводятся к -лся (основа на «с»).
    "ошибся": "ошиблась",
    "забылся": "забылась",
}

#: Частицы/наречия между «я» и формой: «я уже понял», «я всё проверил»,
#: «я не уверен», «я бы сказал», «я же говорил», «я так решил». «все» — тот же
#: «всё» без ё: модели часто пишут без неё.
_PARTICLES = frozenset({"уже", "всё", "все", "не", "бы", "же", "так"})

#: Сколько частиц допустимо между «я» и формой. В живой речи их не больше
#: двух («я всё же проверил»); дальше связь с «я» уже не гарантирована.
_MAX_PARTICLES = 2

#: Слово — только кириллица: «понял», «нашёл». Пунктуация и пробелы — границы.
_WORD_RE = re.compile(r"[А-Яа-яЁё]+")

#: Хвост из бэктиков в конце куска: «`» может оказаться началом «```».
_TRAILING_BACKTICKS_RE = re.compile(r"`+$")

#: Недописанное слово в конце куска: его нельзя решать, пока не видно границы.
_TRAILING_WORD_RE = re.compile(r"[А-Яа-яЁё]+$")

#: Возвратные глаголы: «обрадовался» → «обрадовалась». Основа любая, поэтому
#: правило общее; срабатывает только в первом лице (см. _GenderState).
_REFLEXIVE_END = "лся"
_REFLEXIVE_FEMININE_END = "лась"


def _match_case(source: str, replacement: str) -> str:
    """Сохранить регистр исходного слова: «Понял» → «Поняла», «ПОНЯЛ» → «ПОНЯЛА»."""
    if not source:
        return replacement
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


class GenderStreamer:
    """Потоковая замена мужского рода на женский в первом лице.

    ``feed(chunk)`` принимает очередной кусок ответа и возвращает текст,
    который уже можно показывать. Кусок может обрываться посреди слова —
    тогда последнее слово ждёт границы (пробел, пунктуация) или
    :meth:`flush`. Состояние (кавычки, код, «я» и частицы) живёт между
    вызовами, поэтому поток любой нарезки даёт тот же результат, что
    :func:`feminize` по целому тексту.

    Защищённые области — код в бэктиках (`` `…` ``, ``` ```…``` ```) и
    цитаты («…», „…“, "…"). Внутри них текст не меняется; слово из такой
    области не считается «предыдущим» для следующего — иначе «я» из цитаты
    перекрасило бы слово снаружи.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._chain = -1
        self._in_code = False
        self._in_fence = False
        self._quote_depth = 0

    # -- состояние ----------------------------------------------------------

    @property
    def protected(self) -> bool:
        """Сейчас внутри кода или цитаты: слово трогать нельзя."""
        return self._in_fence or self._in_code or self._quote_depth > 0

    def reset(self) -> None:
        """Начать заново: новый ответ, старые кавычки/код не тянутся."""
        self._pending = ""
        self._chain = -1
        self._in_code = False
        self._in_fence = False
        self._quote_depth = 0

    # -- публичный поток ----------------------------------------------------

    def feed(self, chunk: object) -> str:
        """Принять кусок текста и вернуть его безопасную (готовую) часть."""
        value = str(chunk or "")
        if not value:
            return ""
        buffer = self._pending + value
        self._pending = ""
        # Недописанное слово — в запас: «понял» может оказаться «поняла».
        word = _TRAILING_WORD_RE.search(buffer)
        if word is not None:
            self._pending = buffer[word.start():]
            buffer = buffer[:word.start()]
        # Хвост «`»/«``» тоже придерживаем: иначе «```» распадётся на
        # отдельные переключатели кода и защита «съедет».
        ticks = _TRAILING_BACKTICKS_RE.search(buffer)
        if ticks is not None and len(ticks.group(0)) < 3:
            self._pending = buffer[ticks.start():] + self._pending
            buffer = buffer[:ticks.start()]
        return self._transform(buffer)

    def flush(self) -> str:
        """Отдать то, что ждало границы; вызывать в конце ответа."""
        if not self._pending:
            return ""
        pending, self._pending = self._pending, ""
        return self._transform(pending)

    # -- внутреннее ---------------------------------------------------------

    def _transform(self, text: str) -> str:
        if not text:
            return ""
        parts: list[str] = []
        position = 0
        for match in _WORD_RE.finditer(text):
            gap = text[position:match.start()]
            if gap:
                self._scan_gap(gap)
                parts.append(gap)
            parts.append(self._word(match.group(0)))
            position = match.end()
        tail = text[position:]
        if tail:
            self._scan_gap(tail)
            parts.append(tail)
        return "".join(parts)

    def _scan_gap(self, gap: str) -> None:
        """Обновить состояние защиты по пунктуации/пробелам между словами."""
        index = 0
        length = len(gap)
        while index < length:
            if gap.startswith("```", index):
                self._in_fence = not self._in_fence
                index += 3
                continue
            char = gap[index]
            if char == "`" and not self._in_fence:
                self._in_code = not self._in_code
            elif char == "«":
                self._quote_depth += 1
            elif char in "»“”":
                if self._quote_depth > 0:
                    self._quote_depth -= 1
                elif char in "“”":
                    self._quote_depth += 1
            elif char == "„":
                self._quote_depth += 1
            elif char == "\"":
                self._quote_depth = 0 if self._quote_depth else 1
            index += 1

    def _word(self, word: str) -> str:
        lower = word.lower()
        if self.protected:
            # Чужое слово (цитата/код) не может быть «предыдущим я».
            self._chain = -1
            return word
        if lower == "я":
            self._chain = 0
            return word
        feminine = MASCULINE_TO_FEMININE.get(lower)
        if (
            feminine is None
            and len(lower) > len(_REFLEXIVE_END)
            and lower.endswith(_REFLEXIVE_END)
        ):
            feminine = lower[: -len(_REFLEXIVE_END)] + _REFLEXIVE_FEMININE_END
        if feminine is not None:
            in_first_person = 0 <= self._chain <= _MAX_PARTICLES
            self._chain = -1
            if in_first_person:
                return _match_case(word, feminine)
            return word
        if (
            self._chain >= 0
            and lower in _PARTICLES
            and self._chain < _MAX_PARTICLES
        ):
            self._chain += 1
            return word
        self._chain = -1
        return word


def feminize(text: object) -> str:
    """Заменить мужские формы первого лица на женские.

    Не потоковый вариант :class:`GenderStreamer` для готового текста (история,
    озвучка, тесты). Идемпотентна: уже женские формы не меняются, второй
    прогон возвращает тот же текст.
    """
    value = str(text or "")
    if not value:
        return ""
    streamer = GenderStreamer()
    return streamer.feed(value) + streamer.flush()
