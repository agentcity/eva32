# EVA32 — чек-лист первого подъёма на GPU

> Актуально на 2026-11-26.

## Зачем

Бандл проверялся только на ноутбуке со стендом qwen2.5 через Ollama. Всё, что зависит
от весов и от vLLM — ревизия, стоп-токен, режим размышлений, стриминг, 503 на старте,
сетевая экспозиция, — не проверялось ни разу. `dev/README.md` честно перечисляет, чего
стенд не проверяет: thinking у qwen отсутствует, скорость и память не переносятся.

Порядок шагов неслучаен: сначала фиксируется, **что именно** крутится (ревизия и `.env`),
потом — **как** модель завершает ход и не тратит ли токены на reasoning, только потом
контракт, старт, eval и сеть. Прогон eval на незакреплённой ревизии или в дефолтном
режиме размышлений меряет не то, что будет в проде.

Все команды — на GPU-сервере, из корня репозитория (там, где `Makefile`).

**Пометка «⚠️ требует сверки»** — точное имя параметра, поля или строки лога в этом
репозитории не подтверждено. Такой шаг не угадывает: он даёт команду, которая показывает
фактическое поведение, и критерий приёмки. Решение записывается в `MODEL_CARD.md` /
`docs/contract.md` отдельной правкой, а не «по памяти».

---

## 1. Ревизия весов закреплена и совпадает с `MODEL_CARD.md`

`MODEL_REVISION` в `.env` — единственное место, откуда vLLM берёт коммит. В `1.0.1` он
закреплён (`60d8d70…`) и совпадает с `MODEL_CARD.md`. Важно: `:?` в `serving/compose.yaml`
ловит **пустое** значение, но не заглушку — если кто-то вернёт `replace-with-commit`,
compose пропустит его и падение случится на скачивании, через минуты и с невнятной ошибкой.

### 1.1 `.env` существует и без заглушек

```bash
test -f .env || { echo "нет .env: cp .env.example .env"; exit 1; }
chmod 600 .env
grep -nE '^(MODEL_REVISION|VLLM_API_KEY|HF_HOME)=' .env
grep -q 'replace-with-commit' .env && echo "ЗАГЛУШКА в MODEL_REVISION"
grep -q 'change-me'           .env && echo "ЗАГЛУШКА в VLLM_API_KEY"

# ревизия в .env должна совпадать с MODEL_CARD.md, иначе прогон будет не за ту модель
set -a; . ./.env; set +a
CARD=$(grep -oE '[0-9a-f]{40}' MODEL_CARD.md | head -1)
[ "$CARD" = "$MODEL_REVISION" ] && echo "ревизия совпадает с MODEL_CARD" \
  || echo "РАСХОЖДЕНИЕ: в MODEL_CARD $CARD, в .env $MODEL_REVISION"
```

**ожидаемо:** три переменные видны; обе проверки на заглушку молчат; «ревизия совпадает
с MODEL_CARD»; `ls -l .env` — `-rw-------`.

### 1.2 Ревизия — полный 40-символьный commit, а не тег и не `main`

```bash
set -a; . ./.env; set +a
echo "$MODEL_REVISION"
[[ "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  && echo "похоже на commit" \
  || echo "НЕ commit: тег/ветка/обрезанный sha — для прода не годится"
```

**ожидаемо:** 40 hex-символов, `похоже на commit`. Короткий sha или `main` — не фиксация:
завтра по нему приедет другое.

### 1.3 Скачанный снапшот в `HF_HOME` — именно эта ревизия

```bash
set -a; . ./.env; set +a
STORE="$HF_HOME/hub/models--deepseek-ai--DeepSeek-V4-Flash"
SNAP="$STORE/snapshots/$MODEL_REVISION"

ls -d "$SNAP" && echo "снапшот на месте"
cat "$STORE/refs/main"; echo          # что HF считает текущим
ls "$SNAP" | head -20                 # *.json + веса
du -shL "$SNAP"                       # размер с разыменованием симлинков
```

**ожидаемо:** каталог `snapshots/<MODEL_REVISION>` существует и непуст; `refs/main`
печатает **тот же** commit; в листинге `config.json`, `tokenizer_config.json`,
`generation_config.json` и шарды весов; `du -shL` — порядка 160 ГБ (159,6 ГБ в `MODEL_CARD.md`).
Если `refs/main` другой — веса качались по другой ревизии, vLLM подхватит не то.

Скачивать заранее не обязательно: `--revision` скачает сам при старте. Если хотите
предзагрузку — только с явной ревизией, например тем же тегом образа, что в compose
(`vllm/vllm-openai`, версию закрепить по рецепту):

```bash
docker run --rm -v "$HF_HOME:/models/hf" -e HF_HOME=/models/hf \
  --entrypoint python3 vllm/vllm-openai:"<закреплённый-тег>" \
  -c "from huggingface_hub import snapshot_download as d; print(d('deepseek-ai/DeepSeek-V4-Flash', revision='$MODEL_REVISION'))"
```

**ожидаемо:** печатается путь внутри `snapshots/$MODEL_REVISION`. ⚠️ требует сверки:
наличие `huggingface_hub` и `python3` в образе — проверить `--entrypoint python3 -c "import huggingface_hub"`;
если нет, предзагрузку делать нечем, и остаётся старт с `--revision`.

### 1.4 Что реально передано контейнеру

```bash
docker compose -f serving/compose.yaml --env-file .env config > /tmp/eva-compose.yml
grep -nE 'revision|api-key|max-model-len|tensor-parallel|127\.0\.0\.1' /tmp/eva-compose.yml
```

**ожидаемо:** `--revision=<ваш commit>`; `--max-model-len=65536`; `--tensor-parallel-size=8`;
порт `127.0.0.1:8000`. `--api-key=...` тоже будет виден — **`/tmp/eva-compose.yml` содержит
ключ**: не прикладывать к тикетам/чатам, после проверки `rm -f /tmp/eva-compose.yml`.

Заодно проверьте том с весами — путь должен быть абсолютным:

```bash
grep -n -A4 'volumes:' /tmp/eva-compose.yml
```

**ожидаемо:** источник тома — абсолютный путь (например `/mnt/models/hf`). Если видите
буквальный `~/...` — `HF_HOME` не задан в `.env`, и compose подставил fallback
`${HF_HOME:-~/.cache/huggingface}`; тильду он не раскрывает, получится каталог `~` рядом
с репозиторием.

### 1.5 `MODEL_CARD.md` обновлён

```bash
grep -n 'Ревизия' MODEL_CARD.md
grep -q '<закрепить commit>' MODEL_CARD.md && echo "ЗАГЛУШКА в MODEL_CARD"
grep -c "$MODEL_REVISION" MODEL_CARD.md
```

**ожидаемо:** в строке «Ревизия» стоит тот же commit, что в `.env`; заглушки нет;
`grep -c` вернул ≥ 1. `docs/contract.md` §7 требует именно совпадения `.env` ⇄ `MODEL_CARD.md`.
Проверка «вручную глазами» здесь допустима и обязательна: формат строки в карточке
не машиночитаемый контракт.

**Почему нельзя проверить ревизию через `/v1/models`:** этот ответ отдаёт идентификатор
модели (`eva32`, из `--served-model-name`), и только. Ревизии там нет — не полагайтесь
на него. ⚠️ требует сверки: набор полей `/v1/models` зависит от версии vLLM, поэтому
ревизию подтверждают логами и снапшотом, а не этим эндпоинтом.

### 1.6 Ревизия в логах старта

```bash
docker compose -f serving/compose.yaml --env-file .env logs --tail=300 vllm \
  | grep -iE 'revision|snapshot|commit|model=' 
```

**ожидаемо:** в строке о загрузке модели виден путь/идентификатор с вашим commit.
⚠️ требует сверки: точный текст строки зависит от версии vLLM — ищите **подстроку
с commit**, а не конкретную фразу; если таких строк нет вовсе, сверьте формат логов
по рецепту vLLM и не считайте шаг пройденным.

---

## 2. Стоп-токен: что фактически завершает ход

В `config/generation.yaml` стоит `stop: ["<|im_end|>"]`, унаследованный от легаси-персоны
на qwen2.5 (тот же маркер прошит в `dev/Modelfile`). Для DeepSeek V4-Flash это **не
подтверждено**: `docs/contract.md` §8 прямо держит это в открытых вопросах. Проверяем не
«на глаз», а тремя независимыми фактами: что написано в токенизаторе, что уходит в
запросе, и что делает живая модель.

### 2.1 Что считается концом хода по файлам модели (без запуска)

```bash
set -a; . ./.env; set +a
SNAP="$HF_HOME/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/$MODEL_REVISION"

python3 - "$SNAP" <<'PY'
import json, pathlib, sys
snap = pathlib.Path(sys.argv[1])
tc = json.loads((snap / "tokenizer_config.json").read_text(encoding="utf-8"))
print("eos_token    :", tc.get("eos_token"), "| eos_token_id:", tc.get("eos_token_id"))
print("bos/eos ids  :", tc.get("bos_token_id"), tc.get("eos_token_id"))
gg_path = snap / "generation_config.json"
if gg_path.exists():
    gg = json.loads(gg_path.read_text(encoding="utf-8"))
    print("gen eos ids  :", gg.get("eos_token_id"), "| pad:", gg.get("pad_token_id"))
tpl = tc.get("chat_template") or ""
jinja = snap / "chat_template.jinja"
if not tpl and jinja.exists():
    tpl = jinja.read_text(encoding="utf-8")
print("chat_template:", "есть" if tpl else "НЕТ", f"({len(tpl)} символов)")
print("есть '<|im_end|>' в шаблоне:", "<|im_end|>" in tpl)
for i, line in enumerate(tpl.splitlines(), 1):
    low = line.lower()
    if "im_end" in line or "end of turn" in low or "eot" in low or "eos" in low:
        print(f"  tpl:{i}: {line.strip()[:160]}")
PY
```

**ожидаемо:** видно **настоящее** имя eos/маркера конца хода и то, встречается ли
`<|im_end|>` в шаблоне вообще. Возможные исходы и что они значат:

- в шаблоне есть `<|im_end|>` — конфиг совпадает с шаблоном, шаг 2.3 всё равно прогнать;
- в шаблоне другой маркер (у DeepSeek он свой, конкретную строку берите из вывода, не из
  этого документа) — вероятно, `stop` в конфиге инертен; это гипотеза, факт даёт 2.3,
  решение — 2.5;
- `chat_template` нет ни в `tokenizer_config.json`, ни рядом — ⚠️ требует сверки: шаблон
  может лежать в репозитории модели отдельным файлом (`chat_template.json`, `*.jinja`);
  искать `ls "$SNAP" | grep -i -E 'template|tokenizer|generation'` и по карточке модели.

### 2.2 Что сейчас уходит в запросе

```bash
python3 eval/run_eval.py --dry-run --limit 1 | sed '/^#/d' | jq -c '.stop'
```

**ожидаемо:** `["<|im_end|>"]` — ровно то, что в `config/generation.yaml`.

`--dry-run` печатает служебные строки, начинающиеся с `#` (заголовок с endpoint и
строку `# id:`), — перед `jq` их надо отфильтровать, иначе парсер ругнётся на `#`.
Источник правды один — конфиг; `build_payload()` в `eval/run_eval.py` и
`docs/contract.md` должны показывать одно и то же. Если здесь пусто/`null` — правился
код или конфиг, разбирайтесь до прогона eval.

### 2.3 Живая модель: три запроса, отличается только `stop`

```bash
set -a; . ./.env; set +a
API=http://127.0.0.1:8000
Q='Привет, ты кто? Ответь одной строкой.'
REAL=''    # ← строка конца хода из 2.1, если она НЕ <|im_end|>

ask() {  # $1 = stop-строка или none | $2 = метка в таблице
  jq -n --arg q "$Q" --arg s "$1" '
      {model:"eva32", messages:[{role:"user", content:$q}],
       temperature:0, max_tokens:96}
      + (if $s=="none" then {} else {stop:[$s]} end)' \
  | curl -sS -m 180 "$API/v1/chat/completions" \
        -H "Authorization: Bearer $VLLM_API_KEY" \
        -H 'Content-Type: application/json' -d @- \
  | jq -r --arg l "$2" '
      if .error then "ОШИБКА \($l): \(.error.message)"
      else [ $l,
             .choices[0].finish_reason,
             (.usage.completion_tokens|tostring),
             ((.choices[0].message.content|length)|tostring),
             (.choices[0].message.content|gsub("\\s+";" ")|.[0:45]) ] | @tsv
      end'
}

printf 'stop\tfinish\ttokens\tchars\tcontent\n'
ask none         none
ask '<|im_end|>' im_end
[ -n "$REAL" ] && ask "$REAL" real
```

**ожидаемо (читаем таблицу построчно):**

| строка | что значит |
|---|---|
| `none` → `finish=stop`, токенов немного | модель сама ставит EOS — это и есть настоящий конец хода |
| `none` → `finish=length`, текст «уезжает» дальше | EOS не применился: это и есть симптом «модель не останавливается», см. раздел «Если что-то не сошлось» |
| `im_end` ≈ `none` по `chars`/`tokens` | строка ни на что не влияет — инертный `stop` |
| `im_end` заметно короче, обрыв на полуслове | строка реально матчится и **режет** ответ — так оставлять нельзя |
| `real` ≈ `none` | маркер конца хода — служебный токен, в видимый текст не попадает (это норма) |

Отдельно проверьте, не протёк ли служебный маркер в видимый текст (запрос без `stop`):

```bash
curl -sS -m 180 -o /tmp/eva-nostop.json "$API/v1/chat/completions" \
  -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg q "$Q" '{model:"eva32",messages:[{role:"user",content:$q}],
                             temperature:0,max_tokens:96}')"

jq -r '.choices[0].message.content' /tmp/eva-nostop.json | grep -c 'im_end' || true
jq -r '.choices[0].message.content' /tmp/eva-nostop.json \
  | grep -oE '<[|｜][^>]{1,40}[|｜]>' || echo "служебных токенов в тексте нет"
```

**ожидаемо:** `grep -c 'im_end'` → `0`; вторая команда печатает «служебных токенов в
тексте нет». Если маркер виден в тексте — он не специальный токен для этой модели, и
`stop` по нему режет уже готовый текст.

### 2.4 ⚠️ требует сверки: посмотреть, как токенизируется строка

У vLLM есть служебные эндпоинты `/tokenize` и `/detokenize` (не часть OpenAI API,
наличие и форма ответа зависят от версии — если 404, шаг пропускается, факты из 2.1
достаточны):

```bash
curl -sS -m 60 "$API/tokenize" \
  -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"eva32","prompt":"<|im_end|>"}' | jq -c '{count:(.count // (.tokens|length)), tokens:(.tokens // .)}'

EOS_ID=$(python3 -c "import json;print(json.load(open('$SNAP/tokenizer_config.json')).get('eos_token_id'))")
curl -sS -m 60 "$API/detokenize" \
  -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"eva32\",\"tokens\":[$EOS_ID]}" | jq -c .
```

**ожидаемо:** `count=1` и `tokens` из одного id → строка — специальный токен модели;
`count>1` → это обычный текст, и как «стоп-последовательность» он работает только при
буквальном совпадении в тексте. `/detokenize` для eos печатает ту самую строку конца
хода — её и сравнивайте с `config/generation.yaml`.

### 2.5 Что делать с результатом (не править вслепую)

- `stop` инертен и `none` останавливается по `stop` — безопасно **убрать** `stop` из
  конфига либо заменить на маркер из 2.1; и то и другое — правка бандла, `bundle`
  изменится, нужен новый прогон eval и запись в `CHANGELOG.md`. Места, где маркер
  продублирован: `config/generation.yaml`, `docs/contract.md` §2 и §8, `dev/Modelfile`.
- `stop` режет ответ — правка обязательна до eval, иначе метрики меряют обрезанные ответы.
- Решение фиксируется в `MODEL_CARD.md`/`docs/contract.md` строкой с фактическим маркером,
  чтобы следующий человек не проверял это заново.

---

## 3. Режим размышлений: доезжает ли `thinking: off` до модели

`thinking: off` в `config/generation.yaml` — это поле **нашего** конфига, а не флаг vLLM.
Сначала доказываем, что оно никуда не отправляется, потом измеряем, что модель делает по
умолчанию, и только потом ищем выключатель.

### 3.1 В запросе его нет

```bash
python3 eval/run_eval.py --dry-run --limit 1 | grep -ci thinking
grep -n 'thinking' config/generation.yaml eval/run_eval.py
```

**ожидаемо:** `0` (в теле запроса слова `thinking` нет); `grep` по файлам показывает
только строку конфига. Вывод: связка «`thinking` → параметр запроса» нигде не реализована.
Тот, кто собирает запросы (orioncode), должен её реализовать — иначе `thinking: off`
не действует, а `eval` меряет **дефолтный** режим модели, а не заявленный в конфиге.

### 3.2 Что модель делает по умолчанию

```bash
set -a; . ./.env; set +a
API=http://127.0.0.1:8000
Q='Сколько будет 17*23? Ответь одним числом.'

jq -n --arg q "$Q" '{model:"eva32",messages:[{role:"user",content:$q}],
                     temperature:0,max_tokens:512}' \
| curl -sS -m 300 "$API/v1/chat/completions" \
    -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' -d @- \
| jq '{finish:.choices[0].finish_reason, usage:.usage,
       message_keys:(.choices[0].message|keys),
       content:(.choices[0].message.content|.[0:300])}'
```

**ожидаемо:** `finish=stop`, `message_keys` = `["content","role"]` (или с
`reasoning_content`, если сервер отдаёт reasoning отдельным полем — ⚠️ требует сверки:
поле появляется, когда включён reasoning-парсер vLLM; имя и наличие смотреть в рецепте),
ответ — короткое `391` и ничего больше. **Признак включённого reasoning:** `completion_tokens`
в сотнях на арифметику, в `content` длинное рассуждение до ответа, либо непустой
`reasoning_content`.

Прогоните тот же запрос три раза (`for i in 1 2 3`) и сравните `completion_tokens`:
разброс в 2–3 раза на одном и том же вопросе при `temperature:0` — тоже сигнал, что
модель «думает» по-разному, а не отвечает сразу.

### 3.3 Где искать настоящий выключатель (⚠️ требует сверки целиком)

Порядок такой: сначала шаблон, потом карточка модели/рецепт vLLM, и только потом запрос.

```bash
# 1) условия и переменные, которых ждёт chat_template
python3 - "$SNAP" <<'PY'
import json, pathlib, sys, re
snap = pathlib.Path(sys.argv[1]); tpl = ""
p = snap / "tokenizer_config.json"
if p.exists():
    tpl = json.loads(p.read_text(encoding="utf-8")).get("chat_template") or ""
j = snap / "chat_template.jinja"
if not tpl and j.exists():
    tpl = j.read_text(encoding="utf-8")
seen = set()
for m in re.finditer(r"\{\{[^}]*\}\}|\{%[^%]*%\}", tpl):
    frag = m.group(0)
    if any(k in frag.lower() for k in ("think", "reason", "mode")):
        if frag not in seen:
            seen.add(frag); print("TPL:", frag[:160])
PY
```

**ожидаемо:** список условий/переменных шаблона, относящихся к режиму. Имя ключа для
`chat_template_kwargs` берётся **отсюда** (например, из `{% if ... %}`), а не придумывается.
Если вывод пуст — шаблон режим не переключает, и выключать нечего: режим задан весами.
Тогда фиксируем это как факт и не пишем в контракт «thinking: off».

```bash
# 2) не включён ли reasoning на стороне сервера
docker compose -f serving/compose.yaml --env-file .env config | grep -i -E 'reason|think' || echo "флагов reasoning в compose нет"
docker compose -f serving/compose.yaml --env-file .env logs --tail=300 vllm \
  | grep -i -E 'reason|think' || echo "в логах упоминаний нет"
```

**ожидаемо:** в текущем compose таких флагов нет (это факт репозитория). ⚠️ требует сверки:
имя серверного флага (`--reasoning-parser` или иное) и его наличие зависят от версии
vLLM — сверять по рецепту, а не по этому документу.

```bash
# 3) A/B: тот же запрос с найденным ключом и без него
KW='<ключ из шаблона>'      # ⚠️ подставить найденное имя, не угадывать
probe() {  # $1 = 1 слать chat_template_kwargs, 0 — не слать
  jq -n --arg q "$Q" --arg kw "$KW" --argjson on "$1" '
      {model:"eva32", messages:[{role:"user", content:$q}],
       temperature:0, max_tokens:512}
      + (if $on==1 then {chat_template_kwargs:{($kw):false}} else {} end)' \
  | curl -sS -m 300 "$API/v1/chat/completions" \
        -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' -d @- \
  | tee /tmp/eva-probe-$1.json \
  | jq -r --arg on "$1" '
      if .error then "ОШИБКА on=\($on): \(.error.message)"
      else [ $on, .choices[0].finish_reason,
             (.usage.completion_tokens|tostring),
             ((.choices[0].message.content // "")|length|tostring),
             ((.choices[0].message|keys)|join(",")),
             ((.choices[0].message.content // "")|gsub("\\s+";" ")|.[0:60]) ] | @tsv
      end'
}
printf 'on\tfinish\ttokens\tchars\tkeys\tcontent\n'
probe 0
probe 1
```

**ожидаемо (критерий приёмки):** `on=0` показывает reasoning (большие `tokens`, длинный
content или `reasoning_content` в `keys`); `on=1` — короткий ответ, `tokens` в единицах,
`keys` = `content,role`, и то же число в ответе. Тогда выключатель найден и работает.

Возможные честные исходы:

- `on=1` → `400` с текстом «unexpected keyword argument …» / «unknown field» → **имя
  неверное**; текст ошибки — факт, имя брать из шаблона/карточки, а не подбирать перебором;
- `on=1` даёт тот же объём reasoning, что `on=0` → ключ принят, но не действует;
  смотреть шаблон внимательнее и версию vLLM;
- `400` на весь `chat_template_kwargs` → версия vLLM его не поддерживает; ⚠️ требует
  сверки: поддержка `chat_template_kwargs` версионная, сверять с рецептом. Тогда
  единственный честный путь — правка шаблона модели на стороне сервинга (отдельное
  решение, не «на глаз»).

### 3.4 Что это значит для бандла

Пока выключатель не подтверждён, `thinking: off` в конфиге — **заявление, а не факт**.
Прогон eval в таком состоянии измеряет дефолтный режим; в `MODEL_CARD.md` это надо
написать прямо (какой режим реально проверялся), а в `docs/contract.md` §8 — закрыть
открытый вопрос с фактическим именем ключа и ссылкой на шаг 3.3.

---

## 4. Контракт: нестриминг, SSE, `usage`

Тело запроса не сочиняем — берём то, что печатает `build_payload()`; это и есть сверка
с `docs/contract.md`.

```bash
set -a; . ./.env; set +a
API=http://127.0.0.1:8000
python3 eval/run_eval.py --dry-run --limit 1 | sed '/^#/d' > /tmp/eva-req.json
jq -e '.model=="eva32" and (.messages|length)==2 and .stop==["<|im_end|>"]' /tmp/eva-req.json \
  && echo "payload как в контракте"
```

**ожидаемо:** `payload как в контракте`; в `/tmp/eva-req.json` — system-промпт из
`prompts/system.ru.md`, одно user-сообщение, `temperature 1.0`, `top_p 1.0`,
`max_tokens 1024`, `stop`. У первой строки golden (`greet`) контекста нет, поэтому блока
«Контекст:» в ней не будет; строки с контекстом (а их в наборе большинство) начинают
user-сообщение с `Контекст:` — их видно в `--dry-run` без `--limit`.

### 4.1 Нестриминговый ответ

```bash
curl -sS -m 180 "$API/v1/chat/completions" \
  -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
  -d @/tmp/eva-req.json > /tmp/eva-resp.json

jq '{model, finish:.choices[0].finish_reason,
     content:(.choices[0].message.content|.[0:120]), usage}' /tmp/eva-resp.json
jq -e '.choices[0].message.content|length>0' /tmp/eva-resp.json >/dev/null && echo "content есть"
jq -e '.usage.total_tokens == (.usage.prompt_tokens + .usage.completion_tokens)' \
  /tmp/eva-resp.json >/dev/null && echo "usage сходится"
```

**ожидаемо:** `content есть`, `usage сходится`, `finish` = `stop` (или `length` — тогда
ответ упёрся в 1024 токена, что для короткой строки golden подозрительно), `model` =
`eva32`; `usage` содержит `prompt_tokens`, `completion_tokens`, `total_tokens`.
Если `usage` нет вовсе — это не контракт: клиент не сможет считать деньги (см. §4 контракта).

### 4.2 Ключ обязателен (без него — 401, а не ответ)

```bash
curl -sS -o /dev/null -w 'без ключа: %{http_code}\n' -m 30 \
  "$API/v1/chat/completions" -H 'Content-Type: application/json' -d @/tmp/eva-req.json
curl -sS -o /dev/null -w 'чужой ключ: %{http_code}\n' -m 30 \
  "$API/v1/chat/completions" -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer not-my-key' -d @/tmp/eva-req.json
```

**ожидаемо:** `401` и `401`. `200` без ключа = `--api-key` не применился — дальше не идти,
это дыра, а не мелочь.

### 4.3 Стриминг (SSE)

```bash
jq '. + {stream:true, temperature:0}' /tmp/eva-req.json > /tmp/eva-req-stream.json
curl -sS -N -m 180 "$API/v1/chat/completions" \
  -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
  -d @/tmp/eva-req-stream.json -o /tmp/eva-sse.txt

head -c 400 /tmp/eva-sse.txt; echo; echo '--- хвост:'; tail -2 /tmp/eva-sse.txt
grep -c '^data: ' /tmp/eva-sse.txt
grep -c '^data: \[DONE\]' /tmp/eva-sse.txt
```

**ожидаемо:** несколько строк `data: {"choices":[{"delta":...}]}` (первая — про `role`,
дальше куски текста), последняя строка — `data: [DONE]` (ровно одна). `Content-Type`
ответа — `text/event-stream` (`curl -sS -D - -o /dev/null`), если проверяете заголовки.

Соберите текст из дельт и убедитесь, что он не пустой и читается как ответ:

```bash
grep '^data: ' /tmp/eva-sse.txt | sed 's/^data: //' | grep -v '^\[DONE\]$' \
  | jq -r '.choices[0].delta.content // empty' | tr -d '\n' > /tmp/eva-sse-content.txt
wc -c < /tmp/eva-sse-content.txt; head -c 200 /tmp/eva-sse-content.txt; echo
```

**ожидаемо:** размер > 0, текст по-русски, обрывается логично. Совпадения с нестриминговым
ответом байт-в-байт не требуем даже при `temperature:0` (батчинг vLLM недетерминирован);
сверяем структуру и смысл, а не хеш.

### 4.4 `usage` в стриме (`stream_options.include_usage`)

```bash
jq '. + {stream:true, stream_options:{include_usage:true}}' /tmp/eva-req.json \
| curl -sS -N -m 180 "$API/v1/chat/completions" \
    -H "Authorization: Bearer $VLLM_API_KEY" -H 'Content-Type: application/json' \
    -d @- -o /tmp/eva-sse-usage.txt

tail -3 /tmp/eva-sse-usage.txt
grep '^data: ' /tmp/eva-sse-usage.txt | sed 's/^data: //' | grep -v '^\[DONE\]$' \
  | jq -c 'select(.usage != null) | .usage'
```

**ожидаемо:** один объект `usage` с `prompt_tokens`/`completion_tokens`/`total_tokens`,
`completion_tokens` близок к значению из 4.1. ⚠️ требует сверки: отдельный чанк с `usage`
(обычно с `"choices":[]`) — поведение версионное; если `usage` не приходит вовсе, клиенту
считать деньги нечем, и это надо чинить до прода, а не «примерно оценивать» длину.

### 4.5 Имя модели — только `eva32`

```bash
curl -sS -o /dev/null -w 'чужое имя модели: %{http_code}\n' -m 60 \
  "$API/v1/chat/completions" -H "Authorization: Bearer $VLLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "$(jq '.model="deepseek-ai/DeepSeek-V4-Flash"' /tmp/eva-req.json)"
```

**ожидаемо:** `404` (или `400`) — `--served-model-name=eva32` скрыл исходное имя. `200`
означает, что в проде доступны оба имени; контракт (`"model": "eva32"`) всё равно менять
не нужно, но это надо знать.

---

## 5. Старт и healthcheck

`start_period: 900s` — это 15 минут, в течение которых падения healthcheck **не**
считаются. Модель на ~160 ГБ с 8×H100 легко грузится дольше, чем интуитивные «пара минут»,
поэтому проверяем не «быстро ли», а «дошло ли до healthy, и не крутится ли рестарт».

### 5.1 Железо на месте

```bash
nvidia-smi -L | wc -l
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
```

**ожидаемо:** `8`; восемь строк `NVIDIA H100 ...` примерно по 80 ГБ (в `memory.total` это
обычно ~81000 MiB — точное число зависит от карты), занятая память до старта — сотни
мегабайт. `6` строк — tensor parallel 8 не поднимется, падать
будет на старте.

### 5.2 Старт

```bash
make compose-up        # = docker compose -f serving/compose.yaml --env-file .env up -d
docker compose -f serving/compose.yaml --env-file .env ps
```

**ожидаемо:** сервис `vllm` в статусе `Up ... (health: starting)`. Именно `starting` —
`start_period` ещё идёт; `unhealthy` в первые 15 минут означать ничего не должен.

### 5.3 Ждём `/v1/models` (с ключом)

```bash
set -a; . ./.env; set +a
API=http://127.0.0.1:8000
for i in $(seq 1 120); do
  if curl -fsS -m 4 -H "Authorization: Bearer $VLLM_API_KEY" "$API/v1/models" >/dev/null 2>&1; then
    echo "ответил через ~$((i*15)) c"; break
  fi
  sleep 15
done
curl -sS -m 10 -H "Authorization: Bearer $VLLM_API_KEY" "$API/v1/models" | jq .
```

**ожидаемо:** в течение ≤ 30 минут (обычно меньше) появляется `{"object":"list","data":[...]}`,
в `data[0].id` — `eva32`. Запрос без ключа должен давать `401` — healthcheck в compose
ходит именно с ключом, и это правильно: иначе контейнер «здоров» при закрытом API.

### 5.4 Healthcheck сошёлся, рестартов нет

```bash
CID=$(docker compose -f serving/compose.yaml --env-file .env ps -q vllm)
docker inspect -f 'health={{.State.Health.Status}} restarts={{.RestartCount}}' "$CID"
docker inspect -f '{{range .State.Health.Log}}{{.ExitCode}} {{end}}' "$CID"
```

**ожидаемо:** `health=healthy`, `restarts=0`, в журнале healthcheck — нули (успехи).
Если `restarts` растёт — контейнер падает и перезапускается (`restart: unless-stopped`
реагирует на выход процесса; статус `unhealthy` сам по себе перезапуск **не** вызывает —
это важно не перепутать при разборе).

### 5.5 Что в логах

```bash
make compose-logs      # или: docker compose -f serving/compose.yaml --env-file .env logs --tail=200 vllm
```

**ожидаемо:** строки о загрузке весов, о `tensor parallel size 8`, о `max_model_len 65536`,
о старте HTTP-сервера, и ни одной `OOM`/`CUDA out of memory`/`Killed`. ⚠️ требует сверки:
точные формулировки зависят от версии vLLM — ищите подстроки (`tensor parallel`,
`max_model_len`, `Application startup complete` или аналог), а не дословный текст.

### 5.6 Конфиг доехал до процесса

```bash
docker compose -f serving/compose.yaml --env-file .env exec vllm \
  python3 -c "import os; print('HF_HOME =', os.environ.get('HF_HOME')); print('VLLM_API_KEY задан:', bool(os.environ.get('VLLM_API_KEY')))"
```

**ожидаемо:** `HF_HOME = /models/hf` и `VLLM_API_KEY задан: True` (ключ не печатаем —
только факт наличия). Если `HF_HOME` другой, контейнер смотрит не в тот том; если
`VLLM_API_KEY задан: False`, healthcheck внутри контейнера упадёт на `KeyError` и
контейнер никогда не станет healthy. Не печатайте сам ключ в терминал/лог: он попадёт в
историю и в тикеты.

---

## 6. Eval против прода

```bash
set -a; . ./.env; set +a          # EVA_BASE/ключ берутся из окружения
python3 eval/run_eval.py --bundle          # bundle ДО прогона — для сверки
EVA_BASE=http://127.0.0.1:8000 VLLM_API_KEY="$VLLM_API_KEY" \
  python3 eval/run_eval.py --model eva32 --repeat 3
echo "exit=$?"
```

**ожидаемо:** шапка `endpoint http://127.0.0.1:8000  model eva32  bundle <12hex>`; по строке
`ok  3/3 <id>` на **каждую** запись `eval/golden.jsonl`; финал
`<N>/<N> ok (по 3 прогона на строку)  bundle <12hex>  -> eval/results/eval-<ts>.json`,
где `<N>` — число строк набора; `exit=0`.

`N` не берите из этого документа: набор правят, сейчас в нём 36 строк. Правильное число —
`wc -l < eval/golden.jsonl`; ожидаемое в выводе — `<N>/<N>`, то есть все строки прошли все
три прогона.

Важно: `make eval` в `Makefile` передаёт только `EVA_BASE` и **не** передаёт ключ — он
берётся из окружения. Запуск `make eval` без `export VLLM_API_KEY` даст 401 на всех
строках. Либо экспортируйте (`set -a; . ./.env; set +a`), либо используйте команду выше.

`--repeat 3` — это не «шум», а проверка устойчивости: строка, прошедшая 2/3, считается
проваленной (`ok=false`, `pass: "2/3"`), и итог будет `<N-1>/<N>`. Приёмка — только `<N>/<N>`.

### 6.1 Куда смотреть в `eval/results/*.json`

```bash
F=$(ls -t eval/results/eval-*.json | head -1)
python3 - "$F" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print("base       ", d["base"])
print("model      ", d["model"])
print("repeat     ", d["repeat"])
print("bundle     ", d["bundle"])
print("prompt_sha ", d["prompt_sha256"])
print("passed     ", f'{d["passed"]}/{d["total"]}')
bad = [r for r in d["rows"] if not r["ok"]]
print("не 3/3     ", len(bad))
for r in bad:
    why = next((x["problems"] for x in r["runs"] if not x["ok"]), [])
    print("  ", r["id"], r["pass"], "; ".join(why) or "—")
PY
```

**ожидаемо:** `base` = `http://127.0.0.1:8000`, `model` = `eva32`, `repeat` = 3,
`passed` = `<N>/<N>` (все строки), `не 3/3` = 0. Поля `bundle` и `prompt_sha256` — это идентичность
бандла, а не метрика качества: по ним доказывается, что мерили именно то, что крутится.

### 6.2 Сверка `bundle` и `prompt_sha256` с локальными значениями

```bash
python3 eval/run_eval.py --bundle
python3 - <<'PY'
import hashlib
p = open("prompts/system.ru.md", encoding="utf-8").read()
print("prompt_sha256(strip) =", hashlib.sha256(p.strip().encode()).hexdigest())
PY
git status --porcelain
git rev-parse --short HEAD
```

**ожидаемо:** `bundle` из вывода совпадает с `bundle` в JSON; `prompt_sha256(strip)`
совпадает с `prompt_sha256` в JSON; `git status` пуст (или вы осознанно везёте рабочее
дерево — тогда зафиксируйте diff, иначе воспроизвести релиз будет нечем).

Ловушка: `run_eval.py` считает sha от `read().strip()`, а не от файла целиком — `sha256sum
prompts/system.ru.md` даст другое значение из-за финального перевода строки. Сверяйте
только со `strip`-версией, как в команде выше, либо просто с полем в JSON.

---

## 7. Безопасность: эндпоинт только на 127.0.0.1

`serving/compose.yaml` публикует `127.0.0.1:8000:8000`, а внутри контейнера vLLM слушает
`--host=0.0.0.0`. Это нормально: `0.0.0.0` — внутри сетевого namespace контейнера,
публикация же ограничена loopback хоста. Проверяем с двух сторон: что снаружи не
отвечает и что изнутри ключ всё-таки требуется.

### 7.1 Как порт опубликован

```bash
docker compose -f serving/compose.yaml --env-file .env ps
CID=$(docker compose -f serving/compose.yaml --env-file .env ps -q vllm)
docker port "$CID" 8000
ss -ltnp | grep ':8000' || sudo ss -ltnp | grep ':8000'
```

**ожидаемо:** в `PORTS` — `127.0.0.1:8000->8000/tcp`; `docker port` печатает
`127.0.0.1:8000`; `ss` — одна строка `LISTEN ... 127.0.0.1:8000`, **без** `0.0.0.0:8000`
и без `[::]:8000`. Любой `*:8000`/`0.0.0.0:8000` — стоп, это уже не приватный сервис.

### 7.2 Снаружи не отвечает (с другой машины, не с этого сервера)

```bash
# на другой машине в той же сети и/или из интернета:
curl -sS --connect-timeout 3 -o /dev/null -w '%{http_code}\n' "http://<IP-сервера>:8000/v1/models"; echo "rc=$?"
nc -vz -w3 "<IP-сервера>" 8000; echo "nc_rc=$?"
```

**ожидаемо:** `curl` — `000` и ненулевой `rc` (connection refused/timeout), `nc` —
`Connection refused` или таймаут, `nc_rc` ≠ 0. Любой `200/401` снаружи — **инцидент**:
эндпоинт торчит, останавливайтесь и разбирайтесь (публикация, firewall, security group,
второй контейнер на том же порту).

Проверять надо и из локальной сети, и из интернета: `127.0.0.1`-биндинг закрывает оба
случая, но «в интернет не смотрит» и «в соседнюю подсеть не смотрит» — разные проверки
и разные люди их ломают.

### 7.3 Firewall и чужие публикации порта

```bash
sudo iptables -S 2>/dev/null | grep -E '8000|docker' | head -20
sudo ufw status 2>/dev/null | head -10
docker ps --format '{{.Names}}\t{{.Ports}}' | grep 8000
```

**ожидаемо:** в `iptables`/`ufw` нет правила, разрешающего 8000 снаружи; в `docker ps`
единственная публикация 8000 — `127.0.0.1:8000->8000/tcp`. Правило `ACCEPT ... dpt:8000`
без ограничения источника — находка, а не «на всякий случай».

### 7.4 IPv6 отдельно

```bash
curl -g -sS --connect-timeout 3 -o /dev/null -w '%{http_code}\n' "http://[::1]:8000/v1/models"; echo "rc=$?"
```

**ожидаемо:** `000`/ненулевой `rc` — наружу по IPv6 сервис не опубликован. Если ответил
`401`, значит слушает и `::1`; это ещё не интернет-экспозиция, но зафиксируйте факт
(`docker port` покажет, откуда взялось), потому что `[::]` и `::1` легко перепутать.

### 7.5 Что это значит для orioncode

`127.0.0.1:8000` доступен только процессам **на этом же хосте**. Если бэкенд orioncode
живёт на другой машине, «просто открыть порт» — не решение: между ними по контракту
(mTLS/JWT) и `MODEL_CARD.md` предполагается приватная сеть. Варианты — на уровне
топологии (тот же хост, либо туннель, либо прокси с mTLS), и это отдельное решение,
а не правка одной строки в compose. Проверка, что связка «orioncode → EVA32» вообще
возможна в текущей топологии, входит в приёмку; сама mTLS-обвязка — вне этого чек-листа.

---

## 8. Что зафиксировать после успешного прогона

- `MODEL_CARD.md`: настоящий commit в строке «Ревизия» (шаг 1.5).
- `CHANGELOG.md`: запись с ревизией, `bundle`, `passed/total` и путём к `eval/results/*.json`.
- `docs/contract.md` §8: закрыть `stop` и `thinking` фактическими значениями из шагов 2 и 3.
- `serving/compose.yaml`: закрепить тег образа вместо `latest` (в файле это помечено как
  долг первого подъёма).

---

## Требует сверки: сводка

| Где | Что не подтверждено | Чем проверяется |
|---|---|---|
| 1.3 | наличие `python3`/`huggingface_hub` в образе для предзагрузки | `--entrypoint python3 -c "import huggingface_hub"` |
| 1.5 | ревизии нет в `/v1/models`; набор полей версионный | лог старта + снапшот + `MODEL_CARD.md` |
| 1.6 | точный текст строки лога с commit | `grep -i revision` по логам, подстрока, не фраза |
| 2.1 | где лежит `chat_template` (в `tokenizer_config.json` или рядом файлом) | `ls` по `$SNAP`: искать `*template*` |
| 2.4 | `/tokenize`, `/detokenize` есть не во всех версиях vLLM | 404 → шаг пропускается, факты из 2.1 |
| 3.2 | поле `reasoning_content` (зависит от reasoning-парсера) | `jq` по `.choices[0].message` → `keys` |
| 3.3 | имя ключа в `chat_template_kwargs`; поддержка самого `chat_template_kwargs` | переменные шаблона + A/B; `400` с текстом ошибки |
| 3.3 | имя серверного флага reasoning | `docker compose config`, логи, рецепт vLLM |
| 4.4 | отдельный чанк `usage` в SSE | последние `data:` строки стрима |
| 5.5 | формулировки логов о старте/TP/max_model_len | подстроки в логах, не дословный текст |

---

## Если что-то не сошлось

**Модель не останавливается (ответ уезжает в `max_tokens`).**
Симптом: `finish_reason=length`, `completion_tokens` = `max_tokens`, в тексте после ответа
идёт продолжение «следующего хода» или повтор промпта, строки eval падают по таймауту/длине.
Смотреть по порядку: шаг 2.1 (настоящий eos и маркер в шаблоне), 2.2 (`stop` в запросе),
2.3 строка `none` (ставит ли модель EOS сама), логи vLLM на предмет применения stop-токенов.
Не лечить подбором строк наугад: неверная строка в `stop` либо инертна, либо режет текст.
Правка маркера — это правка бандла (`bundle` меняется), после неё обязателен новый eval.
Если `none` уже даёт `length` — проблема не в `stop`, а в EOS/шаблоне: фиксировать как
блокер, эскалировать, а не «подкручивать temperature».

**Reasoning лезет в ответ.**
Симптом: на простой вопрос в `content` длинное рассуждение перед ответом, либо непустой
`reasoning_content`, `completion_tokens` в сотнях, ответы приходят медленно, eval падает
по таймауту на коротких строках golden (приветствие, цены) — там ответ должен быть в
десятки токенов. Смотреть: 3.1 (в запросе `thinking` нет — это факт),
3.2 (что модель делает по умолчанию), 3.3 (шаблон и A/B с найденным ключом). Пока
выключатель не подтверждён — не писать в `docs/contract.md`, что режим выключен, и не
считать результаты eval метриками «чатового» режима: они сняты в дефолтном. Если
`chat_template_kwargs` сервер не принимает — это решение про версию vLLM/шаблон, а не
про «добавить ещё один параметр в конфиг».

**503 на старте.**
`503` (или отказ соединения) до готовности — **норма**, контракт это прямо разрешает,
healthcheck в это время `starting`, `start_period` = 900 с. Действия: смотреть
`make compose-logs` (загрузка весов, TP, OOM), `nvidia-smi` (память/`utilization 0.92`
вписывается ли), `docker inspect ... .RestartCount` (растёт — процесс падает, а не
«долго грузится»), скорость скачивания весов в `HF_HOME`. Дольше ~30 минут без healthy —
не ждать дальше, а разбирать лог: типичные причины — неверная ревизия (шаг 1.3),
не 8 GPU (5.1), мало `shm`/`ipc`, OOM при `gpu-memory-utilization 0.92`.

**401/403 на loopback, хотя ключ в `.env` есть.**
Симптом: `curl` изнутри даёт `401`, eval — 401 на всех строках, а ключ в `.env` заполнен.
Причины по частоте: ключ не попал в контейнер (`make compose-up` без `--env-file .env`,
или запуск `docker compose up` из другого каталога), в ключе пробел/кавычка (`.env`
читается буквально, `Bearer  abc` ломает заголовок), `make eval` без экспортированного
`VLLM_API_KEY` (Makefile его не передаёт). Проверять: шаг 4.2 (401 без ключа — значит
auth вообще работает), шаг 5.6 (наличие переменной в контейнере), `docker compose config`
(подставился ли ключ). Сам ключ в логи и тикеты не печатать.
