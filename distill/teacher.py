"""Учитель: OpenAI-совместимый эндпоинт с мок-режимом.

Мок нужен, чтобы конвейер проверялся без сети и без ключа: он отдаёт несколько
заранее испорченных ответов, чтобы фильтры было видно в деле.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
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


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    """Сколько секунд просит подождать сервер (Retry-After), если указал."""
    value = exc.headers.get("Retry-After") if exc.headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


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
        self.max_tokens = 1800
        self.retries = int(os.environ.get("TEACHER_RETRIES", "4"))
        self.timeout_s = float(os.environ.get("TEACHER_TIMEOUT", "60"))

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
            # exclude только ПРЯЧЕТ рассуждения, но модель всё равно думает и платит
            # за токены. enabled=false отключает их совсем: замер на кодовых парах
            # 0,104 -> 0,032 ₽ за пару при том же качестве ответа.
            payload["reasoning"] = {"enabled": False}
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"},
        )
        # Один сетевой сбой не должен убивать прогон на тысячи запросов: повторяем
        # с паузой (2, 4, 8 с), а на 429/5xx уважаем Retry-After. Ошибки 4xx кроме
        # 429 не повторяем — это наша ошибка в запросе, а не сеть.
        data = None
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.load(resp)
                break
            except urllib.error.HTTPError as exc:  # noqa: PERF203
                last_error = exc
                if exc.code != 429 and 400 <= exc.code < 500:
                    raise
                wait = _retry_after(exc) or min(2 ** attempt, 30)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                wait = min(2 ** attempt, 30)
            if attempt < self.retries:
                time.sleep(wait)
        if data is None:
            raise last_error if last_error else RuntimeError("учитель: нет ответа")
        msg = data["choices"][0]["message"]
        self.last_usage = data.get("usage") or {}
        # reasoning-часть не идёт в обучение, но стоит токенов — считаем отдельно
        self.last_reasoning = msg.get("reasoning_content") or ""
        return msg["content"]
