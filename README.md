# EVA32

Бандл модели для чата orioncode: веса (ссылка + ревизия), промпт, конфиг генерации, сервинг, eval.

Чего здесь нет и не должно быть: кода чата и Symfony (orioncode), RAG-индекса и документов (pgvector),
секретов, весов в git.

- `MODEL_CARD.md` — что за модель, лицензия, ограничения.
- `prompts/system.ru.md` — персона и правила.
- `config/generation.yaml` — параметры генерации.
- `serving/compose.yaml` — vLLM на GPU-сервере.
- `eval/` — золотой набор и прогон.
- `dev/` — локальные эксперименты (Ollama), не прод.
- `Makefile` — команды: стенд, eval, статус, vLLM.

## Команды

```bash
make                 # список команд
make status          # что запущено: процессы, порты, API, что в памяти
make bundle          # хеш бандла (промпт + конфиг) — версия для eval/CHANGELOG
make dev             # поднять стенд и прогнать золотой набор
make eval-dev        # только прогон на стенде (Ollama, :11434)
make eval-dev REPEAT=3   # каждая строка трижды — видно, что нестабильно
make dev-stop        # выгрузить модель и погасить сервер
make eval            # прогон против vLLM (:8000)
make compose-up      # поднять vLLM на GPU-сервере (нужен .env)
```

Результаты прогонов — `eval/results/eval-*.json`: вердикты по строкам, ответы модели и `bundle`.

