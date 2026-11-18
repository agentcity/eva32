# EVA32

Бандл модели для чата orioncode: веса (ссылка + ревизия), промпт, конфиг генерации, сервинг, eval.

Чего здесь нет и не должно быть: кода чата и Symfony (orioncode), RAG-индекса и документов (pgvector),
секретов, весов в git.

- `MODEL_CARD.md` — что за модель, лицензия, ограничения.
- `docs/contract.md` — контракт с orioncode: промпт, контекст, стриминг, ошибки.
- `docs/roadmap.md` — что нужно для полноценной Евы: фазы, критерии, зависимости.
- `docs/work-plan.md` — очерёдность работ: спринты 1–5, гейты, критический путь.
- `docs/plan-russian.md` — план по русскому языку: что измерять и в каком порядке чинить.
- `docs/quantization.md` — квантование: Q4 сейчас, тернарная гипотеза (4 трита) как исследование.
- `docs/trit-format.md` — спецификация формата тритов: 20 тритов в 32-битном слове.
- `docs/trit-emulation.md` — как получить троичные веса и как их исполнять на бинарном CPU.
- `docs/distillation.md` — дистилляция в троичную модель: шаги, бюджет, риски.
- `bench/ternary_bench.c` — замер int8 / int4 / тритов (память и время счёта).
- `tools/trit_codec.py` — референсный кодек и контрольные векторы.
- `docs/gpu-checklist.md` — подъём на GPU: ревизия, стоп-токен, thinking, healthcheck.
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

python3 eval/run_eval.py --dry-run   # тела запросов без отправки — сверка с docs/contract.md
```

Результаты прогонов — `eval/results/eval-*.json`: вердикты по строкам, ответы модели и `bundle`.

