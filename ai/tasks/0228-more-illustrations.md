---
task_id: "0228"
type: feature
title: Иллюстрации на каждый абзац — первые 9 сказок
created_at: 2026-11-26T23:50:00Z
author: yuri
assignee: agent:more-images
status: in_progress
scope: {docs: true, code: true, data: true}
---

# Мало картинок: хочется по одной на абзац

## Контекст
Сейчас у 9 сказок по 2–4 иллюстрации на сказку. Ребёнку нужно чаще: на каждый
абзац (в среднем 6–10 на сказку).

## Источники (только свободные)
1. **Wikimedia Commons** — не только поиск по названию, но и категории:
   `Category:Illustrations by Ivan Bilibin`, `Category:Russian fairy tales`,
   `Category:Alexander Kurkin`, `Category:Boris Zvorykin`, `Category:Georgy Narbut`,
   `Category:Illustrations of fairy tales`, `Category:Skazki`.
   API: `action=query&generator=categorymembers&gcmtitle=…&prop=imageinfo&iiprop=url|extmetadata&iiurlwidth=1024`.
2. **Русская Викитека/ Викисклад**: сканы дореволюционных изданий
   (Афанасьев, «Сказки» с иллюстрациями) — целые страницы тоже годятся.
3. **Internet Archive** (`archive.org/advancedsearch.php`): книги
   `licenseurl` = PD/CC, сканы иллюстраций; брать только явно свободные.
4. **openverse.org** API (агрегатор CC/PD) — как дополнительный поиск.

Все изображения — **public domain / CC0 / CC-BY** (CC-BY требует подписи в
credits). «All rights reserved» не брать.

## Требования
1. Для 9 сказок (kolobok, repka, teremok, lisa-i-zhuravl, tri-medvedya,
   gusi-lebedi, tsarevna-lyagushka, baba-yaga, sivka-burka) — **6–10 картинок
   на сказку** (итого 60–90), ширина 1024, JPEG ≤300 КБ, суммарно ≤25 МБ.
2. Привязка к абзацам: `after_chars` на границах абзацев текста (по одной
   картинке на абзац; если картинок больше — по две на длинный абзац).
3. `credits.md` по каждой сказке: файл Commons/источник, автор, лицензия.
4. Не ломать: если картинок меньше абзацев — допустимо, но не меньше 6.
5. Тесты: количество ≥6 на сказку, у всех credits, `after_chars` по возрастанию
   и внутри длины текста, без дублей файлов (одинаковая картинка дважды — нет).
6. Живой прогон: «расскажи сказку про колобка» → 8+ image-событий.

## Критерии приёмки
- [ ] 9 сказок по 6–10 картинок, все PD/CC с credits
- [ ] after_chars на абзацах, без дублей
- [ ] суммарно ≤25 МБ, каждая ≤300 КБ
- [ ] тесты и полный набор зелёные

## Отчёт исполнителя

_Заполняется по завершении._
