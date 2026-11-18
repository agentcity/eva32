"""Учитель: OpenAI-совместимый эндпоинт с мок-режимом.

Мок нужен, чтобы конвейер проверялся без сети и без ключа: он отдаёт несколько
заранее испорченных ответов, чтобы фильтры было видно в деле.
"""
from __future__ import annotations

import json
import os
import urllib.request

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
        self.base = (base or os.environ.get("TEACHER_BASE", "")).rstrip("/")
        self.key = key or os.environ.get("TEACHER_KEY", "")
        self.model = model or os.environ.get("TEACHER_MODEL", "deepseek-v4.1-flash")
        self.mock = mock
        self._i = 0

    def ask(self, question: str) -> str:
        if self.mock:
            answer = MOCK_ANSWERS[self._i % len(MOCK_ANSWERS)]
            self._i += 1
            return answer
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0.7,
            "max_tokens": 800,
        }
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]
