"""Делексикализация: заменить реальные факты каталога вымышленными.

Зачем. «Тариф «Профи» — 2490 ₽/мес» — это факт, который меняется. Если учить его в
веса, модель запомнит цифру и через месяц будет уверенно врать. Правильный путь:
цифры отдаёт RAG/скилл, а модель учится **форме ответа** — сослаться на [1], не
выдумывать, взять число из контекста.

Поэтому при обучении подменяем реальные значения вымышленными и согласованными внутри
пары: модель видит тот же синтаксис и те же конструкции, но не запоминает наш каталог.
Реальные пары остаются для проверки RAG-пути.
"""
from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog.json"

#: Вымышленные названия: тарифы — короткие, услуги — по смыслу реальной услуги,
#: чтобы текст оставался осмысленным («доставка» не превращалась в «проверку текстов»).
FAKE_TARIFFS = ["«Луч»", "«Опора»", "«Вектор»", "«Гранит»", "«Кедр»", "«Компас»"]
#: Существительное + род, чтобы прилагательное согласовывалось: «Срочный аудит»,
#: а не «Срочная аудит».
_SERVICE_NOUNS: dict[str, tuple[str, str]] = {
    "доставк": ("Доставка", "f"),
    "поддержк": ("Поддержка", "f"),
    "аудит": ("Аудит", "m"),
    "настройк": ("Настройка", "f"),
    "разработк": ("Разработка", "f"),
    "реклам": ("Реклама", "f"),
    "ведени": ("Ведение", "n"),
}
_SERVICE_ADJ = {
    "f": ["Срочная", "Полная", "Базовая", "Гибкая", "Надёжная", "Лёгкая", "Глубокая"],
    "m": ["Срочный", "Полный", "Базовый", "Гибкий", "Надёжный", "Лёгкий", "Глубокий"],
    "n": ["Срочное", "Полное", "Базовое", "Гибкое", "Надёжное", "Лёгкое", "Глубокое"],
}


def _fake_price(real: int | str, index: int) -> str:
    """Правдоподобная вымышленная цена того же порядка.

    Сохраняем масштаб (и «/мес» в тексте остаётся осмысленным): 6000 → ~6300,
    48000 → ~51000. Округляем до десятков или сотен по величине.
    """
    try:
        value = int(real)
    except (TypeError, ValueError):
        return str(real)
    factor = 0.85 + 0.09 * (index % 6)
    fake = value * factor
    if fake >= 10000:
        return str(int(round(fake / 1000) * 1000))
    if fake >= 1000:
        return str(int(round(fake / 100) * 100))
    return str(int(round(fake / 10) * 10))


def _fake_service_name(name: str, index: int) -> str:
    """Вымышленное имя услуги, сохраняющее её род занятий и род."""
    low = name.lower()
    noun, gender = next(
        ((v, g) for k, (v, g) in _SERVICE_NOUNS.items() if k in low), ("Услуга", "f")
    )
    adjs = _SERVICE_ADJ[gender]
    return f"«{adjs[index % len(adjs)]} {noun.lower()}»"


def load_catalog() -> list[dict]:
    if not CATALOG.exists():
        return []
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def build_mapping(catalog: list[dict]) -> dict[str, str]:
    """Согласованная таблица замен: реальное значение → вымышленное.

    Числа подменяются детерминированно (по позиции в каталоге), чтобы один и тот же
    тариф всегда имел одну и ту же вымышленную цену внутри файла.
    """
    mapping: dict[str, str] = {}
    tariffs = [c for c in catalog if c.get("kind") == "tariff"]
    services = [c for c in catalog if c.get("kind") != "tariff"]
    for i, item in enumerate(tariffs):
        mapping[f"«{item['name']}»"] = FAKE_TARIFFS[i % len(FAKE_TARIFFS)]
        if item.get("price_rub"):
            mapping[str(item["price_rub"])] = _fake_price(item["price_rub"], i)
    for i, item in enumerate(services):
        mapping[f"«{item['name']}»"] = _fake_service_name(item["name"], i)
        if item.get("price_rub"):
            mapping[str(item["price_rub"])] = _fake_price(item["price_rub"], i + 3)
    return mapping


def apply(text: str, mapping: dict[str, str]) -> str:
    """Заменить значения по таблице: длинные ключи раньше коротких."""
    # «12 000 ₽» с пробелом-разделителем разрядов: приводим к слитному виду,
    # иначе замена числа проскочит мимо.
    out = re.sub(r"(?<=\d) (?=\d{3}\b)", "", text)
    for key in sorted(mapping, key=len, reverse=True):
        # Цены заменяем как отдельные числа, чтобы «990» не билось внутри «1990».
        if key.isdigit():
            out = re.sub(rf"(?<!\\d){re.escape(key)}(?!\\d)", mapping[key], out)
        else:
            out = out.replace(key, mapping[key])
    return out


def delex_pair(question: str, answer: str, mapping: dict[str, str]) -> tuple[str, str]:
    return apply(question, mapping), apply(answer, mapping)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    mapping = build_mapping(load_catalog())
    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = changed = 0
    with src.open(encoding="utf-8") as fh, out.open("w", encoding="utf-8") as sink:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            before = (row.get("question", ""), row.get("answer", ""))
            q, a = delex_pair(*before, mapping)
            if "context" in row:
                row["context"] = apply(row["context"], mapping)
            if (q, a) != before:
                changed += 1
            row["question"], row["answer"] = q, a
            row.setdefault("meta", {})["delexicalized"] = True
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
    print(json.dumps({"строк": rows, "заменено": changed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
