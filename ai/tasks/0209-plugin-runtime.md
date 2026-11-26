---
task_id: "0209"
type: feature
title: Рантайм плагинов — Ева растёт расширениями
created_at: 2026-11-26T15:45:00Z
author: yuri
assignee: agent:plugins
status: done
scope: {docs: true, code: true, data: false}
---

# Плагины как у DSH: поставить, включить, пользоваться

## Контекст
ADR 0022. Сейчас время/курсы/котировки, нормализация чисел и род живут в ядре.
Нужно: каталог `plugins/<id>/` + манифест + discovery + включение/выключение, а
ядро знает только «вызови скиллы, подложи факты, прогони фильтры».

## Требования
1. **Контракт** `eva/plugins/`:
   - `manifest.py`: чтение `plugin.json` (`id`, `name`, `version`, `api`=1,
     `entry`, `capabilities`, `permissions`, `description`, `owner`),
     валидация, отказ при чужой версии API;
   - `registry.py`: discovery из `plugins/` (и `EVA_PLUGINS_DIR`), загрузка
     включённых по `config/plugins.json` (`{"enabled": ["time","currency"]}`),
     регистрация способностей; список для UI/API;
   - `context.py`: объект `PluginContext` (настройки плагина, лог, безопасный
     fetch с allowlist из `permissions.network`).
2. **Точки расширения**: `skill.fetch(ctx, question) -> Fact | None`,
   `filter.run(text) -> text` (с `priority`), `tts.engine`, `command`.
3. **Сборка фактов** в чате: перед генерацией ядро спрашивает все включённые
   skill-плагины, собирает `[Факт: …]` в системный блок (как в 0208), не больше
   N фактов.
4. **Первый плагин-набор**: `plugins/time/`, `plugins/currency/`,
   `plugins/stocks/` (перенос из 0208, если он ещё не влит — сделай свои тонкие
   версии), `plugins/ru-numbers/` и `plugins/ru-gender/` (фильтры из 0205/0207,
   ядро оставляет их вызов через плагин, но при выключенных плагинах чат работает).
5. **Безопасность (best-effort)**: сеть только к хостам из `permissions.network`
   (обёртка над urllib), пути — только из `permissions.filesystem`; отказ —
   ошибка с именем плагина.
6. **API/UI**: `GET /api/plugins` (список, включён, версия, возможности),
   `POST /api/plugins {id, enabled}` — включение/выключение (без перезапуска,
   если плагин не держит состояние).
7. **Документация** `docs/plugins.md`: как написать плагин (пример «погода»),
   контракт, permissions, версия API, как включить.

## Критерии приёмки
- [ ] `plugins/time` работает: «который час» → реальное время
- [ ] `plugins/currency` и `plugins/stocks` дают реальные цифры
- [ ] выключение плагина убирает возможность, чат продолжает работать
- [ ] `/api/plugins` показывает список, включение/выключение действует
- [ ] плагин с `api: 2` отклоняется с понятным текстом
- [ ] пример нового плагина в docs (погода) можно скопировать и он заработает
- [ ] тесты: манифест, discovery, включение/выключение, permissions-отказ
- [ ] полный набор зелёный

## Отчёт исполнителя

**Статус: сделано.** Рантайм `eva/plugins/` (манифест, discovery, контекст,
registry), первые пять плагинов в `plugins/`, сборка фактов и фильтры в
`tools/chat_server.py` / `tools/chat.py`, `GET|POST /api/plugins`,
`docs/plugins.md`.

### Живые цифры (сервер на 127.0.0.1, 2026-11-26 ~15:28 MSK)

| Вопрос | Факт в системном блоке |
|---|---|
| который час | `[Факт: 26 ноября 2026, 15:28 MSK (источник: часы сервера)]` |
| курс доллара | `[Факт: 1 USD = 82,3130 ₽ (источник: ЦБ РФ на 26.11.2026)]` |
| цена акции Сбер | `[Факт: SBER (Сбербанк) = 284,97 ₽, за день -2,45 ₽ (-0,85 %), закрытие вчера 287,42 ₽ (источник: MOEX ISS, 26.11.2026 15:28 MSK)]` |

Выключение `currency` через `POST /api/plugins {"id":"currency","enabled":false}`
убрало факт из следующего ответа; чат продолжил работать. Включение вернуло
курс без перезапуска. Сервер отдаёт `GET /api/plugins` со списком, версиями,
способностями, permissions, `enabled`/`loaded` и `last_error`.

### Что где

* `eva/plugins/manifest.py` — `plugin.json`, `api=1`, capabilities,
  permissions (hosts/paths), `owner`, `priority`, `scope`, `stateful`;
  чужая версия API — понятный отказ до исполнения кода.
* `eva/plugins/registry.py` — discovery `plugins/` + `EVA_PLUGINS_DIR`,
  `config/plugins.json`, загрузка entry, `skill.fetch` / `filter.run` /
  `tts.engine` / `command`, лимит фактов, включение/выключение без рестарта.
* `eva/plugins/context.py` — настройки, лог, `fetch` с allowlist хостов
  (включая редиректы) и путями `permissions.filesystem`; отказ называет
  плагин.
* `plugins/time|currency|stocks|ru-numbers|ru-gender` — обёртки над
  `eva/skills/fresh.py`, `eva/skills/intent.py`, `eva/ru_numbers.py`,
  `eva/ru_gender.py`; логика не дублируется. `currency`/`stocks` ходят в сеть
  только через `PluginContext`.
* `tools/chat_server.py` — факты skill-плагинов в системный блок (≤3),
  фильтры в потоке, `GET|POST /api/plugins`, `--facts-limit`, TTS через
  фильтры; ядро работает и без плагинов.
* `tools/chat.py` — то же для REPL/`--once`.
* `tests/test_plugins.py` — 43 теста: манифест (валидный/битый/api=2),
  discovery, включение/выключение, permissions-отказ, лимит и порядок фактов,
  фильтры (в т.ч. потоковые), факт в системном блоке, «выключили — чат жив».
* `docs/plugins.md` — контракт и полный пример «погода» на wttr.in.

### Проверки

```
python3 -m unittest discover -s tests   # 2777 тестов, OK (73 skip)
python3 tools/check_docs_dates.py       # 31 документ, 940 ссылок — OK
```

Замечание: в этом окружении Python не доверяет TLS-сертификату MOEX
(корпоративный перехват). `plugins/stocks` пробует https, а при ошибке
проверки сертификата повторяет тот же публичный запрос по http к тому же
разрешённому хосту; ключей и приватных данных в запросе нет.
