"""Чат-шаблон Qwen (ChatML) и бюджет рассуждений для локальной ВТМ.

Зачем: ``eva/backends/vtm_backend.py`` до этой задачи собирал промпт как
``role: content`` + ``assistant:``. Это переносимый дефолт для демо, но **не**
шаблон Qwen: модель не видит структуру реплик и не может отключить
рассуждения. Qwen3 ждёт::

    <|im_start|>system
    …<|im_end|>
    <|im_start|>user
    …<|im_end|>
    <|im_start|>assistant
    <think>

а при ``enable_thinking=False`` — сразу закрытый пустой блок::

    <|im_start|>assistant
    <think>

    </think>


Именно так рендерит официальный ``chat_template`` из
``tokenizer_config.json`` Qwen3; здесь тот же рендер без Jinja.

Два режима:

* **без рассуждений** (по умолчанию) — ``enable_thinking=False``: пустой
  закрытый ``<think>`` (плюс, если попросят, суффикс ``/no_think`` в последнем
  сообщении пользователя). Модель отвечает сразу, ``max_tokens`` не тратится
  на размышления;
* **с рассуждением** — ``enable_thinking=True``: модель пишет ``<think>…``,
  но не дольше ``reasoning_budget`` токенов. Бюджет исчерпан — рассуждение
  принудительно закрывается, и **вторым прогоном** досказывается ответ
  (:func:`reasoning_steps`). Ответ не теряется ни при каком бюджете.

Определение модели: если рядом с ``tokenizer.json`` есть
``tokenizer_config.json`` (для ``qwen3-tokenizer.json`` — ``qwen3-tokenizer_config.json``)
с непустым ``chat_template``, содержащим ChatML-маркеры ``<|im_start|>``,
модель считается Qwen-подобной — применяется этот рендер. Иначе остаётся
прежний ``format_chat_prompt``.

Только стандартная библиотека: ``json``, ``dataclasses``, ``pathlib``. Ни
сети, ни ``jinja2``, ни модели — модуль можно тестировать без ВТМ.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

__all__ = [
    "DEFAULT_REASONING_BUDGET",
    "IM_END",
    "IM_START",
    "THINK_CLOSE",
    "THINK_CLOSE_SUFFIX",
    "THINK_OPEN",
    "ChatTemplate",
    "ReasoningStep",
    "detect_chat_template",
    "format_messages",
    "reasoning_steps",
    "strip_think",
]

#: Маркеры ChatML (Qwen).
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

#: Теги рассуждения Qwen3. В словаре ``qwen3-tokenizer.json`` это отдельные
#: токены 151667/151668, поэтому ``encode`` даёт ровно один id на тег.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
#: Альтернативное имя конца рассуждения у части Qwen-подобных шаблонов.
THINK_CLOSE_ALT = "<｜end▁of▁thinking｜>"
#: Чем закрываем рассуждение, исчерпав бюджет (перевод строки — чтобы ответ
#: начинался с новой строки, как после настоящего ``</think>``).
THINK_CLOSE_SUFFIX = THINK_CLOSE + "\n"

#: Сколько токенов размышления разрешено по умолчанию (``--reasoning-budget``).
DEFAULT_REASONING_BUDGET = 256

_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?\b[^>]*>.*?</think(?:ing)?\s*>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_OPEN_RE = re.compile(
    r"<think(?:ing)?\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL
)


# --------------------------------------------------------------------------- #
# Рендер ChatML
# --------------------------------------------------------------------------- #

def _role_content(message: Any) -> tuple[str, str]:
    """``(role, content)`` из mapping/объекта; как в ``vtm_backend``."""
    if isinstance(message, Mapping):
        role = message.get("role") or "user"
        content = message.get("content")
    else:
        role = getattr(message, "role", None) or "user"
        content = getattr(message, "content", None)
    return str(role), "" if content is None else str(content)


def _with_no_think_suffix(items: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Дописать ``/no_think`` последнему сообщению пользователя.

    Мягкий выключатель Qwen3: работает и там, где шаблон не поддерживает
    ``enable_thinking``. Пустой закрытый ``<think>`` надёжнее, поэтому суффикс
    включается только явно (``no_think_suffix=True``).
    """
    result = list(items)
    for index in range(len(result) - 1, -1, -1):
        role, content = result[index]
        if role != "user":
            continue
        if "/no_think" not in content:
            content = content.rstrip() + "\n/no_think"
        result[index] = (role, content)
        break
    return result


def format_messages(
    messages: Any,
    *,
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
    supports_thinking: bool = True,
    no_think_suffix: bool = False,
) -> str:
    """Рендер ChatML: ``<|im_start|>role\\ncontent<|im_end|>\\n`` + ассистент.

    ``messages`` — строка (возвращается как есть) или последовательность
    mapping/объектов с ``role``/``content``. Если модели не умеет
    рассуждения (``supports_thinking=False``, Qwen2.5), пустой ``<think>``
    не добавляется: для неё это обычный незнакомый текст.

    ``enable_thinking=False`` и ``supports_thinking=True`` дают закрытый
    пустой блок ``<think>\\n\\n</think>\\n\\n`` после ``assistant`` — ровно то,
    что рендерит официальный шаблон Qwen3 при ``enable_thinking=false``.
    """
    if isinstance(messages, str):
        return messages
    items = [_role_content(message) for message in messages or []]
    if not items:
        return ""
    if no_think_suffix and supports_thinking and not enable_thinking:
        items = _with_no_think_suffix(items)
    parts: list[str] = []
    for role, content in items:
        parts.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
        if supports_thinking and not enable_thinking:
            parts.append(f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Определение шаблона по tokenizer_config.json
# --------------------------------------------------------------------------- #

def _tokenizer_config_path(tokenizer: str | Path) -> Path:
    """``…/tokenizer.json`` -> ``…/tokenizer_config.json`` (та же приставка).

    У моделей в ``runs/q4`` токенизатор называется ``qwen3-tokenizer.json``,
    и конфиг рядом — ``qwen3-tokenizer_config.json``; привязка по суффиксу
    ``tokenizer.json`` находит его без угадывания «любого конфига в каталоге».
    """
    path = Path(tokenizer).expanduser()
    name = path.name
    if name.endswith("tokenizer.json"):
        return path.with_name(name[: -len("tokenizer.json")] + "tokenizer_config.json")
    return path.parent / "tokenizer_config.json"


@dataclass(frozen=True)
class ChatTemplate:
    """Найденный ChatML-шаблон модели.

    ``supports_thinking`` — в конфиге есть ``enable_thinking``/``<think>``
    (Qwen3 и новее). ``source`` — путь к ``tokenizer_config.json`` (для
    отчёта/диагностики), он же попадает в ``repr``.
    """

    name: str = "qwen"
    supports_thinking: bool = False
    source: str = ""

    def render(
        self,
        messages: Any,
        *,
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
        no_think_suffix: bool = False,
    ) -> str:
        """Промпт для ``messages``; рассуждение — по флагу и возможностям."""
        return format_messages(
            messages,
            enable_thinking=bool(enable_thinking) and self.supports_thinking,
            add_generation_prompt=add_generation_prompt,
            supports_thinking=self.supports_thinking,
            no_think_suffix=no_think_suffix,
        )

    def builder(
        self,
        *,
        enable_thinking: bool = False,
        no_think_suffix: bool = False,
    ) -> Callable[[Any], str]:
        """``prompt_builder`` для ``VtmBackend``: ``messages -> str``."""
        return lambda messages: self.render(
            messages,
            enable_thinking=enable_thinking,
            no_think_suffix=no_think_suffix,
        )


def _config_from_mapping(config: Mapping[str, Any], *, source: str) -> ChatTemplate | None:
    template = config.get("chat_template")
    if isinstance(template, list):
        # Новый формат HF: [{"name": "default", "template": "..."}, …].
        chosen = ""
        for entry in template:
            if isinstance(entry, Mapping) and isinstance(entry.get("template"), str):
                if str(entry.get("name") or "default") == "default" or not chosen:
                    chosen = str(entry["template"])
        template = chosen
    if not isinstance(template, str) or not template.strip():
        return None
    if IM_START not in template:
        # Есть chat_template, но он не ChatML (Llama/SOLAR/…): свой рендер
        # применять нельзя — пусть работает прежний ``role: content``.
        return None
    supports_thinking = ("enable_thinking" in template) or (
        THINK_OPEN in template or THINK_CLOSE in template
    )
    return ChatTemplate(
        name="qwen3" if supports_thinking else "qwen",
        supports_thinking=supports_thinking,
        source=source,
    )


def detect_chat_template(
    tokenizer: str | Path | Mapping[str, Any] | None,
) -> ChatTemplate | None:
    """Найти ChatML-шаблон по ``tokenizer_config.json`` (или ``None``).

    ``tokenizer`` — путь к ``tokenizer.json`` (тогда конфиг ищется рядом по
    имени), уже разобранный mapping (тесты) или ``None``. Ошибки чтения и
    битый JSON не бросаются: без шаблона бэкенд откатывается на прежний
    ``format_chat_prompt``, а не падает.
    """
    if tokenizer is None:
        return None
    if isinstance(tokenizer, Mapping):
        return _config_from_mapping(tokenizer, source="<mapping>")
    try:
        path = _tokenizer_config_path(tokenizer)
    except (TypeError, ValueError):
        return None
    if not path.is_file():
        return None
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, Mapping):
        return None
    return _config_from_mapping(config, source=str(path))


# --------------------------------------------------------------------------- #
# Рассуждение: бюджет и досказывание ответа
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ReasoningStep:
    """Шаг генерации: токен фазы, принудительное закрытие или итог.

    * ``phase="reasoning"`` / ``"answer"`` — очередной токен (``token``);
    * ``phase="close"`` — бюджет рассуждения исчерпан, рассуждение нужно
      закрыть текстом ``THINK_CLOSE_SUFFIX`` (токены уже добавлены в промпт
      второго прогона);
    * ``phase="end"`` — генерация закончена, ``reason="eos"|"length"|"stop"``.
    """

    phase: str
    token: int | None = None
    reason: str = ""


def reasoning_steps(
    prompt_ids: Sequence[int],
    *,
    stream_ids: Callable[..., Iterator[int]],
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
    answer_tokens: int,
    reasoning_budget: int = DEFAULT_REASONING_BUDGET,
    think: bool = True,
    eos_id: int = -1,
    close_marker: str = THINK_CLOSE_SUFFIX,
) -> Iterator[ReasoningStep]:
    """Токены одного ответа с бюджетом рассуждения (генератор).

    ``stream_ids(prompt_ids, max_new_tokens=N)`` — генератор id (у
    ``VtmBackend`` это одноимённый метод). ``reasoning_budget`` — сколько
    токенов разрешено на ``<think>``; ``answer_tokens`` — сколько на сам
    ответ. Итоговый лимит новых токенов — сумма, а не один общий лимит.

    Порядок при ``think=True``:

    1. первый прогон с лимитом ``reasoning_budget``;
    2. если модель сама закрыла ``</think>`` — второй прогон продолжает с
       этого места (ответ мог начаться в первом прогоне);
    3. если бюджет кончился до закрытия — ``phase="close"`` (вызывающий
       отдаёт ``THINK_CLOSE_SUFFIX``) и второй прогон с промптом
       ``prompt + рассуждение + "</think>\\n"`` и лимитом ``answer_tokens``.

    Если первый прогон закончился EOS, рассуждение (и ответ вместе с ним)
    уже целиком сгенерировано — второго прогона нет.
    """
    answer_tokens = int(answer_tokens)
    if answer_tokens < 1:
        raise ValueError(f"answer_tokens: ожидалось >= 1, получено {answer_tokens}")
    reasoning_budget = int(reasoning_budget)
    if reasoning_budget < 0:
        raise ValueError(
            f"reasoning_budget: ожидалось >= 0, получено {reasoning_budget}"
        )
    prompt = list(int(token) for token in prompt_ids)
    eos = int(eos_id)
    state = {"eos": False}

    def _tokens(pass_prompt: Sequence[int], limit: int) -> Iterator[int]:
        state["eos"] = False
        for raw in stream_ids(list(pass_prompt), max_new_tokens=int(limit)):
            token = int(raw)
            if eos >= 0 and token == eos:
                state["eos"] = True
                return
            yield token

    if not think or reasoning_budget == 0:
        # Без рассуждения: один прогон, весь лимит — ответу.
        generated = 0
        for token in _tokens(prompt, answer_tokens):
            generated += 1
            yield ReasoningStep("answer", token)
        if state["eos"]:
            reason = "eos"
        else:
            reason = "length" if generated >= answer_tokens else "stop"
        yield ReasoningStep("end", reason=reason)
        return

    reasoning: list[int] = []
    for token in _tokens(prompt, reasoning_budget):
        reasoning.append(token)
        yield ReasoningStep("reasoning", token)
    if state["eos"]:
        # EOS внутри бюджета: модель закончила весь ответ, досказывать нечего.
        yield ReasoningStep("end", reason="eos")
        return
    # EOS не было — либо бюджет кончился, либо генерация остановилась по
    # внешнему лимиту. В обоих случаях ответ досказываем отдельным прогоном.
    text = decode(reasoning)
    closed = (THINK_CLOSE in text) or (THINK_CLOSE_ALT in text)
    extra: list[int] = []
    if not closed:
        extra = [int(token) for token in encode(close_marker)]
        yield ReasoningStep("close")
    second = prompt + reasoning + extra
    generated = 0
    for token in _tokens(second, answer_tokens):
        generated += 1
        yield ReasoningStep("answer", token)
    if state["eos"]:
        reason = "eos"
    else:
        reason = "length" if generated >= answer_tokens else "stop"
    yield ReasoningStep("end", reason=reason)


# --------------------------------------------------------------------------- #
# Текст ответа: убрать рассуждение
# --------------------------------------------------------------------------- #

def strip_think(text: str) -> str:
    """Убрать ``<think>…</think>`` (включая незакрытый) из текста ответа.

    Рассуждение не должно попадать в историю диалога: оно съедает контекст и
    в следующем промпте бесполезно. Незакрытый блок (оборванный бюджетом)
    удаляется до конца строки — иначе половина «мысли» выглядела бы ответом.
    """
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", str(text))
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    return cleaned.strip()
