# EVA32 — команды бандла.
# `make help` — список. Переменные переопределяются на лету:
#   make eval-dev DEV_MODEL=qwen2.5:7b   # тяжёлая проверка, ~4.7 ГБ RAM

SHELL      := /bin/bash
PY         ?= python3
DEV_MODEL  ?= qwen2.5:3b
REPEAT     ?= 1
DEV_PORT   ?= 11434
PROD_PORT  ?= 8000
COMPOSE    := docker compose -f serving/compose.yaml

.PHONY: help status bundle eval eval-dev dev-serve dev-stop dev pull-dev test distill distill-mock teacher-models compose-up compose-down compose-logs clean ci

help:  ## список команд
	@echo "EVA32"
	@echo
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

status:  ## что запущено: процессы, порты, API, загруженная модель
	@echo "процессы:"
	@pgrep -fl 'ollama|llama-server' || echo "  нет"
	@echo "порт $(DEV_PORT):"
	@lsof -nP -iTCP:$(DEV_PORT) -sTCP:LISTEN 2>/dev/null || echo "  свободен"
	@echo "порт $(PROD_PORT):"
	@lsof -nP -iTCP:$(PROD_PORT) -sTCP:LISTEN 2>/dev/null || echo "  свободен"
	@echo "ollama/api:"
	@curl -sS -m 2 http://127.0.0.1:$(DEV_PORT)/api/version 2>/dev/null && echo || echo "  не отвечает"
	@echo "vllm/api:"
	@curl -sS -m 2 http://127.0.0.1:$(PROD_PORT)/v1/models 2>/dev/null && echo || echo "  не отвечает"
	@echo "загружено в память:"
	@ollama ps 2>/dev/null | tail -n +2 | grep . || echo "  сервер не запущен"

bundle:  ## хеш бандла (промпт + конфиг)
	@$(PY) eval/run_eval.py --bundle

dev-serve:  ## поднять локальный Ollama для стенда
	@if curl -sS -m 1 http://127.0.0.1:$(DEV_PORT)/api/version >/dev/null 2>&1; then \
	  echo "ollama уже слушает $(DEV_PORT)"; \
	else \
	  nohup ollama serve >/tmp/ollama.log 2>&1 & \
	  for i in $$(seq 1 40); do \
	    if curl -sS -m 1 http://127.0.0.1:$(DEV_PORT)/api/version >/dev/null 2>&1; then \
	      echo "подняли ollama на $(DEV_PORT)"; exit 0; \
	    fi; \
	    sleep 0.5; \
	  done; \
	  echo "не поднялось за 20с, смотри /tmp/ollama.log" >&2; exit 1; \
	fi

dev-stop:  ## выгрузить модель и погасить сервер
	@ollama stop $(DEV_MODEL) 2>/dev/null || true
	@pkill -f 'ollama serve' 2>/dev/null || true
	@sleep 1; pgrep -fl 'ollama|llama-server' || echo "остановлено, процессов нет"

eval-dev:  ## прогон золотого набора на стенде
	@EVA_BASE=http://127.0.0.1:$(DEV_PORT) $(PY) eval/run_eval.py --model $(DEV_MODEL) --repeat $(REPEAT)

eval:  ## прогон против прод-эндпоинта (vLLM)
	@set -a; [ -f .env ] && . ./.env; set +a; \
	  EVA_BASE=http://127.0.0.1:$(PROD_PORT) $(PY) eval/run_eval.py \
	    --model $${EVA_MODEL:-eva32} --repeat $(REPEAT)

dev: dev-serve eval-dev  ## стенд + прогон одной командой

pull-dev:  ## дотянуть модели из dev/models.lock (нужен запущенный демон)
	@curl -sS -m 2 http://127.0.0.1:$(DEV_PORT)/api/version >/dev/null 2>&1 \
	  || { echo "демон не запущен: make dev-serve"; exit 1; }
	@grep -v '^#' dev/models.lock | awk -F'\t' 'NF{print $$1}' | while read -r m; do \
	  if ollama list 2>/dev/null | awk 'NR>1{print $$1}' | grep -qx "$$m"; then echo "есть  $$m"; \
	  else echo "тяну  $$m"; ollama pull "$$m" || exit 1; fi; \
	done

compose-up:  ## поднять vLLM на GPU (нужен .env)
	$(COMPOSE) --env-file .env up -d

compose-down:  ## погасить vLLM
	$(COMPOSE) --env-file .env down

compose-logs:  ## логи vLLM
	$(COMPOSE) --env-file .env logs -f --tail=200

test:  ## юнит-тесты (фильтры дистилляции, кодек)
	@$(PY) -m unittest discover -s tests -q

teacher-models:  ## какие модели доступны учителю (нужен TEACHER_KEY в .env)
	@$(PY) distill/list_models.py

distill-mock:  ## конвейер дистилляции на заглушке (без сети и ключей)
	@$(PY) distill/run.py --mock --limit 10

distill:  ## сбор пар у настоящего учителя (нужны TEACHER_BASE/TEACHER_KEY)
	@$(PY) distill/run.py --questions distill/questions.txt

clean:  ## удалить локальные результаты eval и пары дистилляции
	rm -rf eval/results distill/out

# То же, что .github/workflows/ci.yml, но локально и без Docker.
ci:  ## проверка как в CI: C (сборка, тесты, UBSan), Python, реестр, документация
	@$(PY) tools/ci.py
