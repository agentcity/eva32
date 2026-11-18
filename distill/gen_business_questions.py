#!/usr/bin/env python3
"""Генератор русских бизнес-вопросов для дистилляции (домен orioncode).

Программирование — не единственное, что должен знать ассистент. Реальному бизнесу
нужны ответы про тарифы, запись, оплату, возвраты, претензии, интеграции. Публичных
русских наборов таких диалогов нет, поэтому вопросы собираются из сценариев ×
формулировок — без сети и без учителя.

    python3 distill/gen_business_questions.py --count 25000 --out distill/business_questions.jsonl
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog.json"

#: Темы, для которых в каталоге есть цифры: вопрос получит контекст [n].
GROUNDED = {"тариф", "услуги", "оплата"}


def load_catalog() -> list[dict]:
    if not CATALOG.exists():
        return []
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def _catalog_line(item: dict, n: int) -> str:
    """Одна строка контекста в формате контракта: [n] — факт."""
    kind = item.get("kind", "")
    head = f"{'Тариф' if kind == 'tariff' else 'Услуга'} «{item.get('name','')}»"
    price = item.get("price_rub")
    period = item.get("period")
    tail = []
    if price:
        tail.append(f"{price} ₽" + ("/мес" if period == "month" else ""))
    tail.extend(item.get("limits") or [])
    tail.extend(item.get("notes") or [])
    return f"[{n}] {head} — " + ", ".join(str(x) for x in tail if x) + "."


def build_context(rng: random.Random, catalog: list[dict], topic: str) -> str | None:
    """Собрать 1–2 факта из каталога под тему вопроса."""
    if topic not in GROUNDED or not catalog:
        return None
    if topic == "тариф":
        pool = [c for c in catalog if c.get("kind") == "tariff"]
    elif topic == "услуги":
        pool = [c for c in catalog if c.get("kind") == "service"]
    else:  # оплата — тарифы, потому что цены там
        pool = list(catalog)
    if not pool:
        return None
    rng.shuffle(pool)
    chosen = pool[: rng.choice([1, 1, 2])]
    return " ".join(_catalog_line(item, i + 1) for i, item in enumerate(chosen))

#: Темы домена. Ключ — тема, значения — предметы разговора.
TOPICS: dict[str, list[str]] = {
    "тариф": [
        "тариф «Старт»", "тариф «Бизнес»", "тариф «Про»", "тариф «Корпоративный»",
        "годовая оплата", "помесячная оплата", "смена тарифа", "пробный период",
        "бесплатный тариф", "тариф для ИП", "тариф для самозанятых",
    ],
    "оплата": [
        "оплата картой", "оплата по счёту", "счёт для юрлица", "закрывающие документы",
        "акт и счёт-фактура", "оплата через СБП", "рассрочка", "возврат оплаты",
        "автоплатёж", "привязка карты", "чек после оплаты", "НДС в стоимости",
    ],
    "запись": [
        "запись на услугу", "перенос записи", "отмена записи", "напоминание о записи",
        "свободные слоты", "запись через мессенджер", "подтверждение записи",
        "лист ожидания", "повторная запись", "запись к конкретному специалисту",
    ],
    "клиенты": [
        "карточка клиента", "история переписки", "импорт клиентов", "дубли клиентов",
        "сегменты клиентов", "теги клиентов", "поиск клиента", "удаление клиента",
        "согласие на обработку данных", "персональные данные клиентов",
    ],
    "переписка": [
        "несколько каналов в одном окне", "WhatsApp и Telegram вместе",
        "автоответ вне рабочего времени", "шаблоны ответов", "пересылка диалога коллеге",
        "оценка качества ответа", "часы работы в автоответе", "уведомления о новых сообщениях",
    ],
    "услуги": [
        "каталог услуг", "категории услуг", "стоимость услуги", "длительность услуги",
        "описание услуги", "фотографии услуг", "скидки на услуги", "акции",
        "пакет услуг", "абонемент", "сертификат", "подарочный сертификат",
    ],
    "сотрудники": [
        "добавление сотрудника", "роли сотрудников", "права доступа", "расписание сотрудника",
        "загрузка сотрудника", "график смен", "несколько филиалов", "личный кабинет сотрудника",
    ],
    "претензии": [
        "жалоба на услугу", "возврат денег", "претензия по качеству", "компенсация",
        "сроки ответа на претензию", "сбой в работе сервиса", "потеря данных",
        "дубль списания", "не пришло уведомление",
    ],
    "интеграции": [
        "интеграция с сайтом", "форма заявки с сайта", "интеграция с 1С",
        "выгрузка в Excel", "API для интеграции", "вебхуки", "экспорт данных",
        "перенос с другого сервиса", "уведомления в Telegram",
    ],
    "безопасность": [
        "двухфакторная аутентификация", "резервные копии", "где хранятся данные",
        "соответствие 152-ФЗ", "кто видит переписку", "доступ к записям разговоров",
        "удаление данных организации", "восстановление пароля",
    ],
    "старт": [
        "начало работы", "настройка за 10 минут", "обучение сотрудников", "перенос данных",
        "пробный период", "демонстрация", "консультация", "партнёрская программа",
    ],
    "поддержка": [
        "сроки ответа поддержки", "поддержка в выходные", "телефон поддержки",
        "чат поддержки", "статус заявки", "обновления сервиса", "плановые работы",
    ],
}

#: Формулировки: (тип, шаблон). {s} — предмет, {t} — тема.
FORMS: list[tuple[str, str]] = [
    ("how", "Как настроить: {s}?"),
    ("how", "Как работает: {s}?"),
    ("how", "Где найти: {s}?"),
    ("how", "Как изменить: {s}?"),
    ("how", "Как отключить: {s}?"),
    ("price", "Сколько стоит: {s}?"),
    ("price", "Какая цена: {s}?"),
    ("price", "Скидка на {s} есть?"),
    ("compare", "Чем отличается {s1} от такого же варианта — {s2}?"),
    ("compare", "Что лучше: {s1} или {s2}?"),
    ("problem", "Не работает: {s}. Что делать?"),
    ("problem", "Помогите, не получается настроить: {s}."),
    ("problem", "Ошибка при работе с «{s}», как исправить?"),
    ("policy", "Можно ли вернуть деньги за {s}?"),
    ("policy", "Что будет, если не оплатить {s}?"),
    ("policy", "Нужно ли платить за {s} отдельно?"),
    ("support", "Куда обратиться по теме «{t}»?"),
    ("start", "С чего начать, если нужно: {s}?"),
    ("terms", "Какие условия: {s}?"),
    ("terms", "На сколько человек рассчитан {s}?"),
    ("client", "Клиент спрашивает про {s} — что ответить?"),
    ("client", "Клиент жалуется на {s} — как ответить?"),
    ("short", "Что такое {s}?"),
    ("short", "Расскажите коротко про {s}."),
]

#: Вежливые зачины: реальный клиент редко пишет телеграфно.
PREFIXES: list[str] = [
    "", "", "",
    "Здравствуйте! ",
    "Добрый день! ",
    "Подскажите, пожалуйста: ",
    "Скажите, пожалуйста, ",
    "Уточните, пожалуйста: ",
    "Вы не подскажете, ",
    "Хочу узнать: ",
    "Помогите разобраться: ",
    "У меня вопрос: ",
    "Мне нужно понять: ",
    "Пожалуйста, объясните: ",
]

#: Контекст клиента — вопрос звучит как живой, а не как пункт меню.
CONTEXTS: list[str] = [
    "",
    "",
    "",
    " (у нас небольшая команда)",
    " (мы только начинаем)",
    " (у нас несколько филиалов)",
    " (мы работаем с юрлицами)",
    " (у нас много клиентов)",
    " (мы переходим с другого сервиса)",
    " (у нас ИП)",
]

TOPIC_RU = {k: v for k, v in {
    "тариф": "тарифы", "оплата": "оплату", "запись": "запись", "клиенты": "клиентов",
    "переписка": "переписку", "услуги": "услуги", "сотрудники": "сотрудников",
    "претензии": "претензии", "интеграции": "интеграции", "безопасность": "безопасность",
    "старт": "начало работы", "поддержка": "поддержку",
}.items()}


#: Вопросы, которые можно задать про конкретную позицию каталога.
CATALOG_FORMS: list[tuple[str, str]] = [
    ("price", "Сколько стоит {what}?"),
    ("price", "Какая цена у {what}?"),
    ("price", "Сколько нужно платить за {what} в месяц?"),
    ("price", "Есть ли скидка на {what}?"),
    ("terms", "Что входит в {what}?"),
    ("terms", "Какие условия у {what}?"),
    ("terms", "На сколько сотрудников рассчитан {noun} {what}?"),
    ("terms", "Как часто нужно оплачивать {what}?"),
    ("client", "Клиент спрашивает, сколько стоит {what}. Что ответить?"),
    ("client", "Клиент говорит, что {what} дорого. Что ответить?"),
    ("compare", "Чем отличается {what} от {what2}?"),
    ("compare", "Что выгоднее: {what} или {what2}?"),
]


def build_from_catalog(count: int, seed: int = 42) -> list[dict]:
    """Вопросы строго про позиции каталога: контекст совпадает с темой вопроса."""
    rng = random.Random(seed)
    catalog = load_catalog()
    if not catalog:
        return []
    out: list[dict] = []
    seen: set[str] = set()
    guard = 0
    while len(out) < count and guard < count * 100:
        guard += 1
        item = rng.choice(catalog)
        other = rng.choice([c for c in catalog if c["id"] != item["id"]])
        forms = [f for f in CATALOG_FORMS if not (f[1].count("{noun}") and item.get("kind") != "tariff")]
        kind, template = rng.choice(forms)
        prefix = rng.choice(PREFIXES)
        noun = "тариф" if item.get("kind") == "tariff" else "услуга"
        noun2 = "тариф" if other.get("kind") == "tariff" else "услуга"
        base = f"«{item['name']}»"
        base2 = f"«{other['name']}»"
        question = template.format(what=base, what2=base2, noun=noun, noun2=noun2)
        question = f"{prefix}{question}"
        if question in seen:
            continue
        seen.add(question)
        context = _catalog_line(item, 1)
        if kind == "compare":
            context += " " + _catalog_line(other, 2)
        out.append({
            "question": question,
            "context": context,
            "meta": {
                "source": "gen_business_questions",
                "topic": item.get("kind", "service"),
                "kind": kind,
                "item": item["id"],
                "grounded": True,
            },
        })
    return out


def build(count: int, seed: int = 42, with_context: bool = False) -> list[dict]:
    rng = random.Random(seed)
    catalog = load_catalog() if with_context else []
    out: list[dict] = []
    seen: set[str] = set()
    topics = list(TOPICS)
    guard = 0
    while len(out) < count and guard < count * 200:
        guard += 1
        topic = rng.choice(topics)
        subjects = TOPICS[topic]
        kind, template = rng.choice(FORMS)
        s1 = rng.choice(subjects)
        s2 = rng.choice([x for x in subjects if x != s1])
        try:
            question = template.format(s=s1, s1=s1, s2=s2, t=TOPIC_RU[topic])
        except (KeyError, IndexError):
            continue
        # Двойные ёлочки от вложенных кавычек в названиях тарифов: «тариф «Старт»».
        question = question.replace("««", "«").replace("»»", "»")
        question = f"{rng.choice(PREFIXES)}{question}{rng.choice(CONTEXTS)}"
        if question in seen:
            continue
        seen.add(question)
        row: dict = {"question": question}
        context = build_context(rng, catalog, topic) if with_context else None
        if context:
            row["context"] = context
        row["meta"] = {
                "source": "gen_business_questions",
                "topic": topic,
                "kind": kind,
                "level": "client",
                "grounded": bool(context),
            }
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=25000)
    ap.add_argument("--out", default="distill/business_questions.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from-catalog", action="store_true",
                    help="вопросы строго про позиции каталога (контекст совпадает)")
    ap.add_argument("--with-context", action="store_true",
                    help="подмешать факты каталога (иначе учитель ответит «не нашла в данных»)")
    args = ap.parse_args()
    if args.from_catalog:
        rows = build_from_catalog(args.count, args.seed)
    else:
        rows = build(args.count, args.seed, with_context=args.with_context)
    with open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"вопросов: {len(rows)} → {args.out}")
    for row in rows[:3]:
        print(" -", row["question"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
