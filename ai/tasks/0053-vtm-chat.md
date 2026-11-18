---
task_id: "0053"
type: feature
title: Чат-шаблон Qwen в текстовом пути
created_at: 2026-11-18T07:20:00Z
author: yuri
assignee: agent:vtm-chat
status: in_progress
scope:
  docs: false
  code: true
  data: false
---

# Чат-шаблон Qwen в текстовом пути

## Проблема

`vtm run --text` шлёт сырой текст. Qwen ожидает `<|im_start|>system...<|im_end|>` и
спецтокены; без шаблона модель на обученных данных будет отвечать мусором.

## Требования

- `--chat` включает шаблон Qwen2.5: system/user/assistant, `<|im_end|>` в конце.
- Спецтокены берутся из tokenizer.json (added_tokens), а не хардкодом.
- `--system "..."` для системного сообщения.
- Обратная совместимость: без `--chat` поведение прежнее.
- Тест: собранный промпт совпадает с эталонной строкой (фикстура).

## Критерии приёмки

[ ] `make test` зелёный
[ ] `--chat --system "..."` даёт точный шаблон (сверка с эталоном в тесте)
[ ] без `--chat` вывод не меняется
[ ] спецтокены не ломают обычный текст

## Контекст и ограничения

- Зона: src/vtm/main.c, src/vtm/tokenizer.hpp (только API), docs/vtm.md (раздел CLI)
- Другие файлы не трогать: параллельно работают другие агенты.
- Сборка: `cd src/vtm && make test`; флаги `-Wall -Wextra -Werror`.
- Не ломать существующие тесты и поведение без флагов.
- Не коммитить: сдавать правки и отчёт.

## Отчёт исполнителя

Сделано (без коммитов):

- `src/vtm/main.c` — флаги `--chat` и `--system "..."`; сборка промпта
  Qwen2.5, печать шаблона в выводе CLI; без `--chat` путь прежний.
- `src/vtm/tokenizer.hpp` — публичный `AddedTokenInfo` и inline
  `Tokenizer::added_tokens()` (только API, `tokenizer.cpp` не менялся).
- `tests/vtm/test_vtm.c` — интеграционный `test_cli_chat`: точное совпадение
  собранного промпта с эталоном из фикстуры, id 500/501/259 в `prompt:`,
  неизменность вывода без `--chat`, код 2 на `--system` без `--chat`.
- `docs/vtm.md` — раздел «CLI» с шаблоном и правилами флагов.

Как строится шаблон: строки `<|im_start|>`/`<|im_end|>` не захардкожены, а
берутся из `added_tokens` (`special == true`) загруженного `tokenizer.json`
(поиск по `content`); роль `system` появляется только при `--system`, как в
официальном шаблоне Qwen. Промпт собирается как строка и кодируется тем же
BPE (`allow_special_tokens`), поэтому разделители становятся одним токеном
с id из файла.

Проверки:

```
$ cd src/vtm && make test
./test_vtm
...
все тесты VTM: ok

$ make test-all
./test_tokenizer "../../tests/vtm/fixtures/tokenizer.json"
...
OK: 131 проверок, 0 провалов

$ python3 -m unittest discover -s tests
Ran 799 tests in 14.443s
OK (skipped=2)
```

Ручная сверка (`demo`-модель, фикстура):

```
$ ./vtm run /tmp/vtm_chat_demo.vtm --text hello --chat \
    --system "Ты ассистент" --tokenizer ../../tests/vtm/fixtures/tokenizer.json
шаблон:
<|im_start|>system
Ты ассистент<|im_end|>
<|im_start|>user
hello<|im_end|>
<|im_start|>assistant
prompt: 500 115 ... 501 10 500 117 115 101 114 10 259 501 10 500 97 115 ... 116 10
```

Подмена id в копии фикстуры (500/501 → 900/901) дала `prompt: 900 ... 901`,
т. е. id действительно из JSON; словарь без `im_start`/`im_end` — код 2 и
`vtm run: чат-шаблон: в tokenizer.json нет special added_token ...`.

Риски:

- `src/vtm/README.md` (раздел «Текст вместо токенов») теперь устарел: там
  написано, что чат-шаблонов нет. Файл вне зоны задачи — не правил.
- Спецтокен, набранный пользователем в `--text`, остаётся спецтокеном и в
  `--chat` (так же ведёт себя HF); защиты от инъекции ролей нет.
- `--chat` без `--system` не подставляет дефолтное системное сообщение —
  выбран вариант «как в официальном шаблоне Qwen, если system нет».
- Сборка общая с задачами 0050/0051: `make` мог пересобрать `vtm` из
  параллельно правящихся `interp.c`/`backend_scalar.c`; тесты на момент
  прогона зелёные.

