"""Конфигурация гибридного ассистента: модели, цены, лимиты, пороги.

Только стандартная библиотека. Значения читаются из переменных окружения
(``EVA_*``), чтобы провайдер и бюджет не были захардкожены (правило 3 из
``docs/hybrid.md``).

Основные переменные:

============================ ===============================================
``EVA_SMALL_BASE``            base URL малой (локальной) модели
``EVA_SMALL_KEY``             ключ малой модели (локально обычно пустой)
``EVA_SMALL_MODEL``           имя модели
``EVA_SMALL_TIMEOUT``         таймаут одного вызова, сек
``EVA_SMALL_RETRIES``         число повторов внутри клиента
``EVA_SMALL_PRICE_PROMPT``    ₽ за 1000 prompt-токенов
``EVA_SMALL_PRICE_COMPLETION``₽ за 1000 completion-токенов
``EVA_BIG_*``                 то же для большой модели
``EVA_MAX_COST_RUB``          бюджет на запрос, ₽ (``none``/отрицательное — без лимита)
``EVA_MAX_TOKENS``            максимум completion-токенов на ответ
``EVA_AUTH_TOKEN``            Bearer-токен HTTP-сервера (пусто — без проверки)
``EVA_RATE_LIMIT_PER_MIN``    запросов в минуту на ключ (``0`` — без лимита)
``EVA_CLASSIFIER_LONG_CHARS`` порог «длинного» запроса для классификатора
``EVA_CLASSIFIER_LATIN_RATIO``порог доли латиницы
``EVA_CLASSIFIER_MARKERS``    маркеры рассуждения через запятую
============================ ===============================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = [
    "Config",
    "DEFAULT_RATE_LIMIT_PER_MIN",
    "DEFAULT_REASON_MARKERS",
    "auth_settings",
    "load",
    "from_env",
]

#: Лимит запросов HTTP-сервера по умолчанию: в минуту на ключ.
DEFAULT_RATE_LIMIT_PER_MIN = 60

#: Маркеры, после которых малой модели обычно не хватает знаний/дисциплины.
DEFAULT_REASON_MARKERS = (
    "почему",
    "зачем",
    "объясни",
    "объясните",
    "сравни",
    "сравните",
    "напиши код",
    "напишите код",
    "переведи",
    "переведите",
    "посчитай",
    "посчитайте",
    "докажи",
    "проанализируй",
    "как работает",
    "что такое",
)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _env_str(env: Mapping[str, str], name: str, default: str) -> str:
    value = _clean(env.get(name))
    return default if value is None else value


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    value = _clean(env.get(name))
    if value is None:
        return default
    try:
        return float(value.replace(",", ".").replace("₽", "").strip())
    except ValueError as exc:
        raise ValueError(f"{name}: ожидалось число, получено {value!r}") from exc


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = _clean(env.get(name))
    if value is None:
        return default
    try:
        return int(float(value.replace(",", ".")))
    except ValueError as exc:
        raise ValueError(f"{name}: ожидалось целое число, получено {value!r}") from exc


def auth_settings(env: Mapping[str, str] | None = None) -> tuple[str, int]:
    """Токен и лимит запросов из окружения — только эти две настройки.

    Серверу нужны ровно они, поэтому собирать весь :class:`Config` (с базами и
    ценами моделей) на старте HTTP-процесса незачем. Некорректный
    ``EVA_RATE_LIMIT_PER_MIN`` — ``ValueError``: лучше не подняться вовсе, чем
    молча остаться без лимита. Отрицательный лимит читается как ``0``
    (лимит выключен).
    """
    source = os.environ if env is None else env
    token = _env_str(source, "EVA_AUTH_TOKEN", "")
    limit = _env_int(source, "EVA_RATE_LIMIT_PER_MIN", DEFAULT_RATE_LIMIT_PER_MIN)
    return token, max(0, limit)


@dataclass(frozen=True)
class Config:
    """Настройки маршрутизатора, клиентов, бюджета и классификатора."""

    # --- малая (локальная) модель ---
    small_base: str = "http://127.0.0.1:8001/v1"
    small_key: str = ""
    small_model: str = "eva-small"
    small_timeout: float = 30.0
    small_retries: int = 1
    small_price_prompt_rub_per_1k: float = 0.0
    small_price_completion_rub_per_1k: float = 0.0

    # --- большая (облачная) модель ---
    big_base: str = ""
    big_key: str = ""
    big_model: str = ""
    big_timeout: float = 60.0
    big_retries: int = 2
    big_price_prompt_rub_per_1k: float = 0.4
    big_price_completion_rub_per_1k: float = 1.2

    # --- бюджет и лимиты ---
    max_cost_rub: float | None = 5.0
    max_tokens: int = 512

    # --- доступ к HTTP-серверу (eva/server.py) ---
    auth_token: str = ""
    rate_limit_per_min: int = DEFAULT_RATE_LIMIT_PER_MIN

    # --- пороги классификатора (не хардкодятся в classify) ---
    classifier_long_chars: int = 120
    classifier_latin_ratio: float = 0.35
    classifier_reason_markers: tuple[str, ...] = DEFAULT_REASON_MARKERS

    # ------------------------------------------------------------------ #
    # производные свойства
    # ------------------------------------------------------------------ #
    @property
    def big_available(self) -> bool:
        """Настроена ли большая модель (base + model)."""
        return bool(self.big_base and self.big_model)

    @property
    def small_available(self) -> bool:
        return bool(self.small_base and self.small_model)

    @property
    def reason_markers(self) -> tuple[str, ...]:
        """Алиас для классификатора."""
        return self.classifier_reason_markers

    @property
    def long_text_chars(self) -> int:
        """Алиас для классификатора."""
        return self.classifier_long_chars

    @property
    def latin_ratio_threshold(self) -> float:
        """Алиас для классификатора."""
        return self.classifier_latin_ratio

    @property
    def classifier(self) -> dict[str, Any]:
        """Пороги одним словарём — чтобы классификатор не знал имён полей."""
        return {
            "long_chars": self.classifier_long_chars,
            "latin_ratio": self.classifier_latin_ratio,
            "reason_markers": self.classifier_reason_markers,
        }

    def prices(self, kind: str) -> tuple[float, float]:
        """(₽/1k prompt, ₽/1k completion) для маршрута."""
        if kind == "big":
            return (self.big_price_prompt_rub_per_1k, self.big_price_completion_rub_per_1k)
        if kind == "small":
            return (self.small_price_prompt_rub_per_1k, self.small_price_completion_rub_per_1k)
        return (0.0, 0.0)

    def estimate_cost(self, kind: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Оценка цены по тарифу модели, ₽ (без сетевого вызова)."""
        prompt_price, completion_price = self.prices(kind)
        cost = (max(0, int(prompt_tokens)) / 1000.0) * prompt_price
        cost += (max(0, int(completion_tokens)) / 1000.0) * completion_price
        return round(cost, 6)

    def max_tokens_for(self, kind: str) -> int:
        return max(1, int(self.max_tokens))

    # ------------------------------------------------------------------ #
    # загрузка
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Config":
        """Собрать конфиг из окружения (или из переданного словаря)."""
        source = os.environ if env is None else env

        max_cost_raw = _clean(source.get("EVA_MAX_COST_RUB"))
        if max_cost_raw is None:
            max_cost: float | None = 5.0
        elif max_cost_raw.lower() in {"none", "off", "unlimited", "unlim", "inf"}:
            max_cost = None
        else:
            value = _env_float(source, "EVA_MAX_COST_RUB", 5.0)
            max_cost = None if value < 0 else value

        markers_raw = _clean(source.get("EVA_CLASSIFIER_MARKERS"))
        if markers_raw is None:
            markers = DEFAULT_REASON_MARKERS
        else:
            markers = tuple(m.strip().lower() for m in markers_raw.split(",") if m.strip())
            if not markers:
                markers = DEFAULT_REASON_MARKERS

        auth_token, rate_limit_per_min = auth_settings(source)

        return cls(
            small_base=_env_str(source, "EVA_SMALL_BASE", "http://127.0.0.1:8001/v1"),
            small_key=_env_str(source, "EVA_SMALL_KEY", ""),
            small_model=_env_str(source, "EVA_SMALL_MODEL", "eva-small"),
            small_timeout=_env_float(source, "EVA_SMALL_TIMEOUT", 30.0),
            small_retries=_env_int(source, "EVA_SMALL_RETRIES", 1),
            small_price_prompt_rub_per_1k=_env_float(source, "EVA_SMALL_PRICE_PROMPT", 0.0),
            small_price_completion_rub_per_1k=_env_float(source, "EVA_SMALL_PRICE_COMPLETION", 0.0),
            big_base=_env_str(source, "EVA_BIG_BASE", ""),
            big_key=_env_str(source, "EVA_BIG_KEY", ""),
            big_model=_env_str(source, "EVA_BIG_MODEL", ""),
            big_timeout=_env_float(source, "EVA_BIG_TIMEOUT", 60.0),
            big_retries=_env_int(source, "EVA_BIG_RETRIES", 2),
            big_price_prompt_rub_per_1k=_env_float(source, "EVA_BIG_PRICE_PROMPT", 0.4),
            big_price_completion_rub_per_1k=_env_float(source, "EVA_BIG_PRICE_COMPLETION", 1.2),
            max_cost_rub=max_cost,
            max_tokens=_env_int(source, "EVA_MAX_TOKENS", 512),
            auth_token=auth_token,
            rate_limit_per_min=rate_limit_per_min,
            classifier_long_chars=_env_int(source, "EVA_CLASSIFIER_LONG_CHARS", 120),
            classifier_latin_ratio=_env_float(source, "EVA_CLASSIFIER_LATIN_RATIO", 0.35),
            classifier_reason_markers=markers,
        )


def load(env: Mapping[str, str] | None = None) -> Config:
    """Удобный алиас :meth:`Config.from_env`."""
    return Config.from_env(env)


def from_env(env: Mapping[str, str] | None = None) -> Config:
    """Удобный алиас :meth:`Config.from_env`."""
    return Config.from_env(env)
