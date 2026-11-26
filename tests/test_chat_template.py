"""Тесты Qwen-шаблона и бюджета рассуждений (задача 0200).

Что проверяем:

* рендер ChatML: ``<|im_start|>role\\n…<|im_end|>\\n`` + ``assistant``; при
  ``enable_thinking=False`` — пустой закрытый ``<think>``, при ``True`` — без
  него; ``/no_think`` — только по явному запросу;
* определение шаблона по ``tokenizer_config.json`` (Qwen3/Qwen2.5/не-Qwen);
* ``VtmBackend`` сам выбирает Qwen-рендер, а без шаблона остаётся прежний
  ``role: content`` (libvtm и модель не нужны: промпт строится лениво);
* бюджет: исчерпан — рассуждение закрывается, ответ **досказывается** вторым
  прогоном; модель закрыла ``</think>`` сама — второго «закрытия» нет;
  EOS внутри бюджета — второго прогона нет;
* ``VtmChatBackend``: по умолчанию ``<think>`` не появляется и промпт
  содержит пустой закрытый блок; с ``think=True`` ответ есть даже при
  исчерпанном бюджете; ``--think`` без Qwen-шаблона — ошибка конфига;
* ``tools/chat_server.py`` пробрасывает ``--think``/``--reasoning-budget``
  в фабрику бэкенда.

Сети, GPU и ``ctypes`` нет: генерация — заглушки, модели — игрушечный
токенизатор. Настоящие ``tokenizer_config.json`` из ``runs/q4`` используются
только для чтения (если файлов нет — подтест скипается).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva.backends import VtmBackend, format_chat_prompt  # noqa: E402
from eva.backends.vtm_backend import _native_token_limit  # noqa: E402
from eva.chat_template import (  # noqa: E402
    DEFAULT_REASONING_BUDGET,
    THINK_CLOSE_SUFFIX,
    ChatTemplate,
    detect_chat_template,
    format_messages,
    reasoning_steps,
    strip_think,
)
from tools import chat as chat_cli  # noqa: E402
from tools import chat_server as server_mod  # noqa: E402

QWEN3_CONFIG = ROOT / "runs" / "q4" / "qwen3-tokenizer_config.json"
QWEN25_CONFIG = ROOT / "runs" / "q4" / "tokenizer_config.json"
QWEN3_TOKENIZER = ROOT / "runs" / "q4" / "qwen3-tokenizer.json"
QWEN25_TOKENIZER = ROOT / "runs" / "q4" / "tokenizer.json"
FIXTURE_TOKENIZER = ROOT / "tests" / "vtm" / "fixtures" / "tokenizer.json"


# --------------------------------------------------------------------------- #
# игрушечный токенизатор и заглушка ВТМ
# --------------------------------------------------------------------------- #

class ToyTokenizer:
    """Словарь из нескольких спецтокенов + посимвольный запас.

    ``decode(encode(text)) == text`` для любого текста: этого хватает, чтобы
    проверить рендер, склейку стрима и закрытие ``</think>`` без 11 МБ
    настоящего ``tokenizer.json``.
    """

    SPECIAL = {
        "<|im_start|>": 90,
        "<|im_end|>": 99,
        "<think>": 91,
        "</think>": 92,
        "\n": 93,
    }

    def __init__(self, eos_id: int = 99) -> None:
        self.eos_id = eos_id
        self._id_to_text = {value: text for text, value in self.SPECIAL.items()}
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._next = 200

    def encode(self, text: str, *, allow_special_tokens: bool = True) -> list[int]:
        ids: list[int] = []
        index = 0
        ordered = sorted(self.SPECIAL.items(), key=lambda item: -len(item[0]))
        while index < len(text):
            for token, token_id in ordered:
                if text.startswith(token, index):
                    ids.append(token_id)
                    index += len(token)
                    break
            else:
                char = text[index]
                if char not in self._char_to_id:
                    self._char_to_id[char] = self._next
                    self._id_to_char[self._next] = char
                    self._next += 1
                ids.append(self._char_to_id[char])
                index += 1
        return ids

    def decode(self, ids) -> str:
        parts: list[str] = []
        for raw in ids:
            token = int(raw)
            if token in self._id_to_text:
                parts.append(self._id_to_text[token])
            elif token in self._id_to_char:
                parts.append(self._id_to_char[token])
        return "".join(parts)

    def stream_decoder(self) -> "ToyDecoder":
        return ToyDecoder(self)


class ToyDecoder:
    """Один токен = готовый текст (игрушечному токенизатору байты не нужны)."""

    def __init__(self, tokenizer: ToyTokenizer) -> None:
        self._tokenizer = tokenizer

    def push(self, token: int) -> str:
        return self._tokenizer.decode([token])

    def finish(self) -> str:
        return ""


class FakeQwenInner:
    """Заглушка ``VtmBackend`` с Qwen-шаблоном и заданными прогонами."""

    def __init__(
        self,
        *,
        rounds: list[list[int] | str] | tuple = (),
        template: ChatTemplate | None = None,
        thinking: bool = False,
        supports_thinking: bool = True,
        eos_id: int | None = None,
    ) -> None:
        self.tokenizer = ToyTokenizer(
            eos_id=self.tokenizer_eos if eos_id is None else eos_id
        )
        self.template = template or ChatTemplate(
            name="qwen3", supports_thinking=True, source="<test>"
        )
        self.supports_thinking = supports_thinking and self.template.supports_thinking
        self.thinking = thinking
        # Строки кодируются токенизатором заглушки: иначе id из другого
        # игрушечного словаря не декодируются.
        self.rounds = [
            self.tokenizer.encode(round_) if isinstance(round_, str) else list(round_)
            for round_ in rounds
        ]
        self.calls: list[tuple[list[int], int | None]] = []
        self.prompt_builder = self.template.builder(enable_thinking=thinking)

    tokenizer_eos = 99

    def encode(self, text: str, *, allow_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)

    def stream_ids(self, prompt_ids, *, max_new_tokens=None):
        self.calls.append((list(prompt_ids), max_new_tokens))
        index = len(self.calls) - 1
        ids = self.rounds[index] if index < len(self.rounds) else []
        yield from ids


def _ids(text: str, tokenizer: ToyTokenizer | None = None) -> list[int]:
    return (tokenizer or ToyTokenizer()).encode(text)

# --------------------------------------------------------------------------- #
# 1. Рендер шаблона
# --------------------------------------------------------------------------- #

class TestRender(unittest.TestCase):
    def test_chatml_layout(self) -> None:
        text = format_messages(
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "u2"},
            ]
        )
        self.assertEqual(
            text,
            "<|im_start|>system\ns<|im_end|>\n"
            "<|im_start|>user\nu<|im_end|>\n"
            "<|im_start|>assistant\na<|im_end|>\n"
            "<|im_start|>user\nu2<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )

    def test_no_thinking_closes_think_block(self) -> None:
        text = format_messages([{"role": "user", "content": "вопрос"}])
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        # Пустой блок уже закрыт: модели нечего «додумывать».
        self.assertNotIn("<|im_start|>assistant\n<think>\n\n\n", text)

    def test_thinking_enabled_opens_block(self) -> None:
        text = format_messages(
            [{"role": "user", "content": "вопрос"}], enable_thinking=True
        )
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", text)

    def test_model_without_thinking_gets_plain_assistant(self) -> None:
        text = format_messages(
            [{"role": "user", "content": "u"}], supports_thinking=False
        )
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", text)
        # enable_thinking=True у Qwen2.5 ничего не меняет.
        self.assertEqual(
            text,
            format_messages(
                [{"role": "user", "content": "u"}],
                enable_thinking=True,
                supports_thinking=False,
            ),
        )

    def test_no_think_suffix_only_on_request(self) -> None:
        plain = format_messages([{"role": "user", "content": "u"}])
        self.assertNotIn("/no_think", plain)
        with_suffix = format_messages(
            [{"role": "user", "content": "u"}], no_think_suffix=True
        )
        self.assertIn("u\n/no_think<|im_end|>", with_suffix)
        # Включать рассуждение и суффикс одновременно нельзя.
        self.assertNotIn(
            "/no_think",
            format_messages(
                [{"role": "user", "content": "u"}],
                enable_thinking=True,
                no_think_suffix=True,
            ),
        )

    def test_string_passthrough_and_empty(self) -> None:
        self.assertEqual(format_messages("raw"), "raw")
        self.assertEqual(format_messages([]), "")

    def test_add_generation_prompt_off(self) -> None:
        text = format_messages(
            [{"role": "user", "content": "u"}], add_generation_prompt=False
        )
        self.assertEqual(text, "<|im_start|>user\nu<|im_end|>\n")

    def test_strip_think(self) -> None:
        self.assertEqual(strip_think("<think>a</think>\nответ"), "ответ")
        self.assertEqual(strip_think("до <think>a</think> после"), "до  после")
        # Оборванное бюджетом рассуждение — не ответ.
        self.assertEqual(strip_think("<think>недодумал"), "")
        self.assertEqual(strip_think(""), "")


class TestDetect(unittest.TestCase):
    def test_qwen3_config(self) -> None:
        if not QWEN3_CONFIG.is_file():
            self.skipTest(f"нет {QWEN3_CONFIG}")
        template = detect_chat_template(QWEN3_TOKENIZER)
        self.assertIsNotNone(template)
        assert template is not None
        self.assertEqual(template.name, "qwen3")
        self.assertTrue(template.supports_thinking)
        self.assertIn("qwen3-tokenizer_config.json", template.source)

    def test_qwen25_config_is_chatml_without_thinking(self) -> None:
        if not QWEN25_CONFIG.is_file():
            self.skipTest(f"нет {QWEN25_CONFIG}")
        template = detect_chat_template(QWEN25_TOKENIZER)
        self.assertIsNotNone(template)
        assert template is not None
        self.assertFalse(template.supports_thinking)
        self.assertNotIn("<think>", template.render([{"role": "user", "content": "u"}]))

    def test_no_config_is_none(self) -> None:
        self.assertIsNone(detect_chat_template(FIXTURE_TOKENIZER))
        self.assertIsNone(detect_chat_template(None))

    def test_non_chatml_template_is_ignored(self) -> None:
        config = {"chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}"}
        self.assertIsNone(detect_chat_template(config))

    def test_mapping_with_enable_thinking(self) -> None:
        template = detect_chat_template(
            {"chat_template": "<|im_start|>{{ role }}{{ enable_thinking }}</|im_end|>"}
        )
        self.assertIsNotNone(template)
        assert template is not None
        self.assertTrue(template.supports_thinking)


# --------------------------------------------------------------------------- #
# 2. VtmBackend выбирает шаблон сам (без libvtm и без модели)
# --------------------------------------------------------------------------- #

class TestVtmBackendTemplate(unittest.TestCase):
    def _backend(self, tokenizer: Path, **kwargs) -> VtmBackend:
        return VtmBackend(
            model=str(ROOT / "нет-такой-модели.vtm"),
            tokenizer=str(tokenizer),
            **kwargs,
        )

    def test_qwen3_template_selected_automatically(self) -> None:
        if not QWEN3_TOKENIZER.is_file():
            self.skipTest("нет qwen3-tokenizer.json")
        backend = self._backend(QWEN3_TOKENIZER)
        self.assertIsNotNone(backend.template)
        self.assertFalse(backend.supports_thinking)  # think по умолчанию выключен
        prompt = backend.render_chat([{"role": "user", "content": "привет"}])
        self.assertIn("<|im_start|>user\nпривет<|im_end|>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        # think=True — промпт без пустого блока, модель рассуждает сама.
        thinking = self._backend(QWEN3_TOKENIZER, think=True)
        self.assertTrue(thinking.supports_thinking)
        self.assertTrue(
            thinking.render_chat([{"role": "user", "content": "u"}]).endswith(
                "<|im_start|>assistant\n"
            )
        )
        # Явный флаг перекрывает конфигурацию.
        self.assertTrue(
            backend.render_chat([{"role": "user", "content": "u"}], enable_thinking=True)
            .endswith("<|im_start|>assistant\n")
        )

    def test_non_qwen_keeps_old_format(self) -> None:
        backend = self._backend(FIXTURE_TOKENIZER)
        self.assertIsNone(backend.template)
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]
        self.assertEqual(backend.render_chat(messages), format_chat_prompt(messages))
        self.assertEqual(backend.prompt_builder(messages), format_chat_prompt(messages))

    def test_plain_mode_ignores_template(self) -> None:
        if not QWEN3_TOKENIZER.is_file():
            self.skipTest("нет qwen3-tokenizer.json")
        backend = self._backend(QWEN3_TOKENIZER, chat_template="plain")
        self.assertIsNone(backend.template)
        self.assertEqual(
            backend.render_chat([{"role": "user", "content": "u"}]),
            "user: u\nassistant:",
        )

    def test_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            self._backend(FIXTURE_TOKENIZER, reasoning_budget=-1)
        with self.assertRaises(ValueError):
            self._backend(FIXTURE_TOKENIZER, chat_template="llama")


# --------------------------------------------------------------------------- #
# 3. Бюджет рассуждения (мок generator)
# --------------------------------------------------------------------------- #

class TestReasoningSteps(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = ToyTokenizer()
        self.prompt = [90, 1, 2, 99]
        self.calls: list[tuple[list[int], int]] = []

    def _run(self, rounds, **kwargs):
        def stream_ids(prompt_ids, *, max_new_tokens=None):
            self.calls.append((list(prompt_ids), int(max_new_tokens)))
            index = len(self.calls) - 1
            return iter(rounds[index] if index < len(rounds) else [])

        return list(
            reasoning_steps(
                self.prompt,
                stream_ids=stream_ids,
                encode=self.tokenizer.encode,
                decode=self.tokenizer.decode,
                eos_id=self.tokenizer.eos_id,
                answer_tokens=kwargs.pop("answer_tokens", 8),
                **kwargs,
            )
        )

    def test_no_think_is_one_pass(self) -> None:
        answer = self.tokenizer.encode("ответ!")
        steps = self._run([answer], think=False, answer_tokens=len(answer))
        self.assertEqual(
            [(s.phase, s.token) for s in steps],
            [("answer", token) for token in answer] + [("end", None)],
        )
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1], len(answer))

    def test_budget_exhausted_closes_and_answers(self) -> None:
        reasoning = self.tokenizer.encode("<think>долго думаю")
        answer = self.tokenizer.encode("ответ!")
        steps = self._run(
            [reasoning, answer],
            think=True,
            reasoning_budget=3,
            answer_tokens=10,
        )
        phases = [step.phase for step in steps]
        self.assertEqual(phases.count("reasoning"), len(reasoning))
        self.assertEqual(phases.count("close"), 1)
        self.assertEqual(phases.count("answer"), len(answer))
        self.assertEqual(steps[-1].reason, "stop")
        # Второй прогон: тот же промпт + рассуждение + закрытый </think>\n.
        self.assertEqual(len(self.calls), 2)
        expected_prompt = self.prompt + reasoning + self.tokenizer.encode(
            THINK_CLOSE_SUFFIX
        )
        self.assertEqual(self.calls[1][0], expected_prompt)
        self.assertEqual(self.calls[1][1], 10)
        self.assertEqual(self.calls[0][1], 3)

    def test_natural_close_does_not_add_marker(self) -> None:
        reasoning = self.tokenizer.encode("<think>хватит</think>\n")
        answer = self.tokenizer.encode("ответ")
        steps = self._run([reasoning, answer], think=True, reasoning_budget=10)
        self.assertNotIn("close", [step.phase for step in steps])
        self.assertEqual(self.calls[1][0], self.prompt + reasoning)
        self.assertEqual([s.token for s in steps if s.phase == "answer"], answer)

    def test_eos_inside_budget_stops_without_second_pass(self) -> None:
        reasoning = self.tokenizer.encode("готово")
        steps = self._run(
            [reasoning + [self.tokenizer.eos_id], []], think=True, reasoning_budget=10
        )
        self.assertEqual([step.phase for step in steps][-1], "end")
        self.assertEqual(steps[-1].reason, "eos")
        self.assertEqual(len(self.calls), 1)

    def test_answer_never_lost_when_budget_zero(self) -> None:
        answer = self.tokenizer.encode("ответ")
        steps = self._run([answer], think=True, reasoning_budget=0)
        self.assertEqual([s.token for s in steps if s.phase == "answer"], answer)

    def test_answer_length_budget_reports_length(self) -> None:
        answer = self.tokenizer.encode("ответ")
        steps = self._run([answer], think=False, answer_tokens=len(answer))
        self.assertEqual(steps[-1].reason, "length")

    def test_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            self._run([], answer_tokens=0)
        with self.assertRaises(ValueError):
            self._run([], think=True, reasoning_budget=-1)


# --------------------------------------------------------------------------- #
# 4. VtmChatBackend: по умолчанию без <think>, с бюджетом ответ есть
# --------------------------------------------------------------------------- #

class TestVtmChatBackendReasoning(unittest.TestCase):
    def test_default_has_no_think_block(self) -> None:
        inner = FakeQwenInner(rounds=["ответ!"])
        backend = chat_cli.VtmChatBackend(model="fake.vtm", inner=inner)
        events = list(
            backend.stream([{"role": "user", "content": "вопрос"}], max_tokens=8)
        )
        text = "".join(event.text for event in events)
        self.assertEqual(text, "ответ!")
        self.assertNotIn("<think>", text)
        self.assertNotIn("</think>", text)
        # Промпт — настоящий ChatML с пустым закрытым <think>: модель не
        # получает повода рассуждать, а max_tokens целиком идёт ответу.
        prompt = inner.prompt_builder([{"role": "user", "content": "вопрос"}])
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        done = events[-1]
        self.assertEqual(done.usage["completion_tokens"], 6)
        self.assertNotIn("reasoning_tokens", done.usage)
        self.assertEqual(len(inner.calls), 1)
        self.assertEqual(inner.calls[0][1], 8)

    def test_budget_exhausted_still_answers(self) -> None:
        inner = FakeQwenInner(rounds=["<think>долго думаю", "ответ!"], thinking=True)
        reasoning = inner.tokenizer.encode("<think>долго думаю")
        answer = inner.tokenizer.encode("ответ!")
        backend = chat_cli.VtmChatBackend(
            model="fake.vtm", inner=inner, think=True, reasoning_budget=2
        )
        events = list(
            backend.stream([{"role": "user", "content": "q"}], max_tokens=8)
        )
        text = "".join(event.text for event in events)
        self.assertIn("<think>", text)
        self.assertIn(THINK_CLOSE_SUFFIX.strip(), text)
        self.assertIn("ответ!", text)
        # Рассуждение закрыто, и ответ идёт после закрывающего тега.
        self.assertLess(text.index(THINK_CLOSE_SUFFIX.strip()), text.index("ответ!"))
        done = events[-1]
        self.assertEqual(done.finish_reason, "stop")
        self.assertEqual(done.usage["reasoning_tokens"], len(reasoning))
        self.assertEqual(done.usage["answer_tokens"], len(answer))
        self.assertEqual(
            done.usage["completion_tokens"], len(reasoning) + len(answer)
        )
        self.assertEqual(len(inner.calls), 2)
        self.assertEqual(inner.calls[0][1], 2)  # бюджет размышления
        self.assertEqual(inner.calls[1][1], 8)  # отдельный лимит ответа
        expected_prompt = (
            inner.encode(inner.prompt_builder([{"role": "user", "content": "q"}]))
            + reasoning
            + inner.encode(THINK_CLOSE_SUFFIX)
        )
        self.assertEqual(inner.calls[1][0], expected_prompt)

    def test_think_without_qwen_template_is_config_error(self) -> None:
        inner = FakeQwenInner(supports_thinking=False)
        with self.assertRaises(chat_cli.ChatConfigError) as ctx:
            chat_cli.VtmChatBackend(
                model="fake.vtm", inner=inner, think=True, reasoning_budget=8
            )
        self.assertIn("Qwen", str(ctx.exception))

    def test_non_qwen_uses_old_prompt(self) -> None:
        inner = FakeQwenInner(
            rounds=["ответ"],
            template=ChatTemplate(name="plain", supports_thinking=False),
        )
        backend = chat_cli.VtmChatBackend(model="fake.vtm", inner=inner)
        events = list(backend.stream([{"role": "user", "content": "q"}], max_tokens=4))
        prompt = inner.prompt_builder([{"role": "user", "content": "q"}])
        self.assertNotIn("<think>", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertEqual("".join(e.text for e in events), "ответ")


class TestNativeLimit(unittest.TestCase):
    """Потолок C-контекста: бюджет + ответ, иначе бюджет молча режется."""

    def test_thinking_adds_budget(self) -> None:
        self.assertEqual(
            _native_token_limit(32, thinking=True, reasoning_budget=64), 96
        )

    def test_without_thinking_is_answer_only(self) -> None:
        self.assertEqual(
            _native_token_limit(32, thinking=False, reasoning_budget=64), 32
        )

    def test_zero_means_unlimited(self) -> None:
        self.assertEqual(
            _native_token_limit(0, thinking=True, reasoning_budget=64), 0
        )


class TestSessionBudgetReserve(unittest.TestCase):
    def test_trim_reserves_reasoning_budget(self) -> None:
        counter = chat_cli.TokenCounter(exact=True, counter=len)
        session = chat_cli.ChatSession(
            ctx=100, max_tokens=10, reasoning_budget=20, counter=counter
        )
        session.add_user("привет")
        _, trim = session.build_messages()
        # ctx - (ответ + бюджет) - overhead = 100 - 30 - 8
        self.assertEqual(trim.budget_tokens, 62)

    def test_negative_budget_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chat_cli.ChatSession(ctx=100, max_tokens=10, reasoning_budget=-1)


# --------------------------------------------------------------------------- #
# 5. chat_server пробрасывает флаги
# --------------------------------------------------------------------------- #

class TestServerFlags(unittest.TestCase):
    def test_factory_passes_think_to_chat_backend(self) -> None:
        model = server_mod.ModelSpec(
            id="m", label="m", target="runs/m.vtm", tokenizer="runs/tokenizer.json"
        )
        factory = server_mod._chat_cli_factory(
            "vtm", model, 5.0, think=True, reasoning_budget=64
        )
        with mock.patch.object(chat_cli, "build_backend") as build:
            factory()
        build.assert_called_once_with(
            backend="vtm",
            model="runs/m.vtm",
            tokenizer="runs/tokenizer.json",
            timeout=5.0,
            think=True,
            reasoning_budget=64,
        )

    def test_create_server_defaults_think_off(self) -> None:
        registry = (
            server_mod.BackendSpec(
                id="vtm",
                label="ВТМ",
                kind="vtm",
                models=(server_mod.ModelSpec(id="m", label="m", target="m.vtm"),),
            ),
        )
        with mock.patch.object(server_mod, "default_adapters", return_value={}) as build:
            server = server_mod.create_server(
                port=0,
                registry=registry,
                system_prompt="",
                think=True,
                reasoning_budget=64,
            )
        try:
            self.assertTrue(server.think)
            self.assertEqual(server.reasoning_budget, 64)
            build.assert_called_once()
            self.assertTrue(build.call_args.kwargs["think"])
            self.assertEqual(build.call_args.kwargs["reasoning_budget"], 64)
        finally:
            server.server_close()

    def test_create_server_rejects_negative_budget(self) -> None:
        with self.assertRaises(ValueError):
            server_mod.create_server(port=0, reasoning_budget=-1)

    def test_default_budget_matches_constant(self) -> None:
        self.assertEqual(server_mod.DEFAULT_REASONING_BUDGET, DEFAULT_REASONING_BUDGET)
        self.assertEqual(chat_cli.DEFAULT_REASONING_BUDGET, DEFAULT_REASONING_BUDGET)


if __name__ == "__main__":
    unittest.main()
