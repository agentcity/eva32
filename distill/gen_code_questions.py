#!/usr/bin/env python3
"""Генератор русских вопросов по программированию для дистилляции.

Вопросы собираются из тем, предметов и формулировок — без сети и без учителя.
Результат: JSONL с полями question/meta, по одному на строку.

    python3 distill/gen_code_questions.py --count 10000 --out distill/code_questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import random

TOPICS: dict[str, list[str]] = {
    "python": [
        "списковые включения", "генераторы", "декораторы", "контекстные менеджеры",
        "GIL", "asyncio", "типизация", "dataclasses", "f-строки", "исключения",
        "множества", "словари", "срезы", "замыкания", "итераторы", "модули",
        "пакеты и импорты", "многопроцессность", "потоки", "виртуальные окружения",
        "сортировка", "копирование объектов", "изменяемые аргументы по умолчанию",
        "работа с файлами", "json", "регулярные выражения", "модуль collections",
        "модуль itertools", "модуль pathlib", "logging",
    ],
    "javascript": [
        "замыкания", "this", "прототипы", "промисы", "async/await", "event loop",
        "деструктуризация", "spread", "модули ESM", "стрелочные функции",
        "массивы и их методы", "объекты", "Map и Set", "JSON", "fetch",
        "обработка ошибок", "таймеры", "DOM-события", "localStorage",
        "иммутабельность", "опциональная цепочка", "нулевое слияние",
    ],
    "sql": [
        "JOIN", "LEFT JOIN", "GROUP BY", "HAVING", "оконные функции",
        "индексы", "транзакции", "уровни изоляции", "подзапросы", "CTE",
        "UNION", "NULL", "агрегаты", "EXPLAIN", "нормализация", "внешние ключи",
        "UPSERT", "пагинация", "дедлоки", "VACUUM",
    ],
    "algorithms": [
        "сложность O(n)", "бинарный поиск", "сортировка слиянием", "быстрая сортировка",
        "хеш-таблицы", "связный список", "стек и очередь", "дерево поиска",
        "обход графа", "динамическое программирование", "жадные алгоритмы",
        "рекурсия", "мемоизация", "BFS", "DFS", "куча", "битовые операции",
    ],
    "git": [
        "rebase", "merge", "cherry-pick", "revert", "reset", "stash",
        "конфликты слияния", "ветки", "теги", "reflog", "squash", "submodules",
        "hooks", "bisect", "git blame",
    ],
    "linux": [
        "права доступа", "chmod", "chown", "процессы", "systemd", "cron",
        "ssh", "bash-скрипты", "grep и find", "перенаправления", "переменные окружения",
        "диски и монтирование", "логи", "сеть и порты", "tmux",
    ],
    "oop": [
        "наследование и композиция", "полиморфизм", "инкапсуляция",
        "интерфейсы и абстракции", "SOLID", "паттерн фабрика", "паттерн стратегия",
        "паттерн наблюдатель", "паттерн синглтон", "dependency injection",
    ],
    "async": [
        "асинхронность против потоков", "блокирующие вызовы", "гонки данных",
        "мьютексы и семафоры", "пулы воркеров", "обратное давление",
        "отмена задач", "таймауты",
    ],
    "testing": [
        "юнит-тесты", "интеграционные тесты", "моки и стабы", "фикстуры",
        "параметризация", "покрытие тестами", "flake8 и pylint", "CI",
    ],
    "security": [
        "SQL-инъекции", "XSS", "CSRF", "хранение паролей", "хеширование",
        "JWT", "валидация входных данных", "права и роли", "секреты в коде",
    ],
}

FORMS = [
    ("explain", "Объясни простыми словами: {s} в {t}."),
    ("explain", "Как работает {s} в {t}?"),
    ("compare", "Чем отличается {s1} от {s2} в {t}?"),
    ("why", "Почему в {t} важно понимать {s}?"),
    ("how", "Как правильно использовать {s} в {t}? Приведи пример."),
    ("code", "Напиши на {lang} пример с {s} и объясни коротко."),
    ("debug", "Найди типичную ошибку при работе с {s} в {t} и покажи исправление."),
    ("pitfall", "Какие подводные камни у {s} в {t}?"),
    ("practice", "Что спрашивают на собеседовании про {s} в {t} и как отвечать?"),
    ("refactor", "Как улучшить код, где используется {s} в {t}?"),
]

LANG = {"python": "Python", "javascript": "JavaScript", "sql": "SQL", "linux": "bash"}


def build(count: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    seen: set[str] = set()
    topics = list(TOPICS)
    guard = 0
    while len(out) < count and guard < count * 400:
        guard += 1
        t = rng.choice(topics)
        subjects = TOPICS[t]
        kind, template = rng.choice(FORMS)
        s1 = rng.choice(subjects)
        s2 = rng.choice([x for x in subjects if x != s1])
        question = template.format(
            s=s1, s1=s1, s2=s2, t=_topic_ru(t), lang=LANG.get(t, "Python")
        )
        if question in seen:
            continue
        seen.add(question)
        out.append({
            "question": question,
            "meta": {"source": "gen_code_questions", "topic": t,
                     "kind": kind, "level": rng.choice(["junior", "middle", "senior"])},
        })
    return out


def _topic_ru(topic: str) -> str:
    return {
        "python": "Python", "javascript": "JavaScript", "sql": "SQL",
        "algorithms": "алгоритмах", "git": "Git", "linux": "Linux",
        "oop": "ООП", "async": "асинхронном программировании",
        "testing": "тестировании", "security": "безопасности",
    }[topic]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--out", default="distill/code_questions.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = build(args.count, args.seed)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"вопросов: {len(rows)} → {args.out}")
    if rows:
        print("примеры:")
        for row in rows[:3]:
            print(" -", row["question"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
