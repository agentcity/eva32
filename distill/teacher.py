"""Учитель: OpenAI-совместимый эндпоинт с мок-режимом.

Мок нужен, чтобы конвейер проверялся без сети и без ключа: он отдаёт несколько
заранее испорченных ответов, чтобы фильтры было видно в деле.
"""
from __future__ import annotations

import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path: str | None = None) -> None:
    """Подхватить .env из корня репозитория, не перетирая уже заданные переменные.

    Ключ не должен ходить через переписку: положили в .env (он в .gitignore) — и всё.
    """
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"\''))

MOCK_ANSWERS = [
    "Тариф «Старт» стоит 990 ₽ в месяц и подходит для небольших команд до трёх человек.",
    "Тариф стоит 990 ₽ в месяц — это дешевле, чем у конкурентов, поэтому берите.",
    "Стоимость 990 ₽/мес. 这是最好的方案，建议购买。",
    "Не нашла в данных такую услугу. Уточните, пожалуйста, что именно вас интересует.",
    "Конечно! Вот детали: подключение бесплатно, дальнейшее обслуживание по тарифу.",
]


class Teacher:
    def __init__(self, base: str | None = None, key: str | None = None,
                 model: str | None = None, mock: bool = False):
        load_env()
        self.base = (base or os.environ.get("TEACHER_BASE", "")).rstrip("/")
        self.key = key or os.environ.get("TEACHER_KEY", "")
        self.model = model or os.environ.get("TEACHER_MODEL", "deepseek-v4.1-flash")
        self.mock = mock
        self._i = 0
        self.last_usage: dict = {}
        self.last_reasoning: str = ""
        # reasoning стоит 30-45% цены, а в обучение не идёт: по умолчанию выключен
        self.reasoning = os.environ.get("TEACHER_REASONING", "off").lower() == "on"
        self.max_tokens = 900

    @staticmethod
    def build_message(question: str, context: str | None = None) -> str:
        """Точно тот же формат, что в контракте: «Контекст: [n] … Вопрос: …».

        Без контекста учитель отвечает общими словами («уточните, о каком сервисе
        речь») — такому ученицу учить нельзя.
        """
        if not context:
            return question
        return f"Контекст:\n{context}\n\nВопрос: {question}"

    def ask(self, question: str, context: str | None = None,
            system: str | None = None, max_tokens: int | None = None) -> str:
        if self.mock:
            answer = MOCK_ANSWERS[self._i % len(MOCK_ANSWERS)]
            self._i += 1
            return answer
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": self.build_message(question, context)})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if not self.reasoning:
            payload["reasoning"] = {"exclude": True}
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.load(resp)
        msg = data["choices"][0]["message"]
        self.last_usage = data.get("usage") or {}
        # reasoning-часть не идёт в обучение, но стоит токенов — считаем отдельно
        self.last_reasoning = msg.get("reasoning_content") or ""
        return msg["content"]
