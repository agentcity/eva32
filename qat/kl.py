"""KL-дистилляция: формат логитов учителя, разбор файла и опорный расчёт.

Модуль **не импортирует torch**: и проверка формата, и эталонный KL считаются
на чистом Python, поэтому проверяются обычным ``python3 -m unittest`` без
GPU-стека. Torch-путь (``kl_div_sum_torch``) импортирует torch лениво — им
пользуется ``qat.train`` на реальном прогоне.

Формат строки файла логитов (JSONL, по строке на пару)::

    {"question": "…", "answer": "…",
     "logits": [[… V чисел …], … A строк …]}

и, начиная с задачи 0111, top-k вместо полного словаря::

    {"question": "…", "answer": "…",
     "topk": {"k": 64,
              "indices": [[… k индексов …], … A строк …],
              "logits":  [[… k логитов …], … A строк …]}}

* ``V`` — размер словаря ученика (он обязан совпадать со словарём учителя);
* ``P`` — длина промпта ученика (тот же токенизатор и тот же chat-шаблон,
  что и при обучении), ``A`` — число токенов ответа, включая eos;
* строка ``t`` — логиты учителя на позиции, которая предсказывает ``t``-й
  токен ответа, то есть ``logits(seq)[P - 1 + t]``. Именно с этой позиции
  ученик предсказывает тот же токен: ``student_logits[:, P-1]`` предсказывает
  метку с индексом ``P``.

Сдвиг на один токен (взять ``P + t`` вместо ``P - 1 + t``) даёт **молча
неверный** KL: длины сойдутся, обучение пойдёт, а дистилляция будет учить
ученика не тому, что сказал учитель. Поэтому ``PairDataset`` допускает
``A - 1 … A + 1`` строк (eos/без eos и обрезка), а согласованность позиций
проверяется отдельно: ``scripts/tiny_qat.py logits`` считает
``KL(teacher || student)`` на стартовых весах и пишет её в
``logits_metrics.json``. Для совпадающих весов она должна быть ~0
(на крошечной модели замер — 1e-9 на токен); ненулевая KL на старте означает
сдвиг позиций или другой промпт, а не «учитель плохой».

Почему top-k (задача 0111)
=========================
Полный словарь не масштабируется: на Qwen (V=151936) пара из ``A=402`` токенов
это 116 МиБ fp16 на пару и 4,9 ТБ на 40 тыс. пар, а ``PairDataset`` держит
логиты Python-списками (~1,7 ГБ на пару). top-k=64 — это ``A × k × (4 + 8)``
байт в памяти и ~100 КиБ на пару в двоичном виде.

Цена приближения честно: масса учителя за пределами top-k **отбрасывается**,
распределение учителя перенормируется на ``k`` индексов. Поэтому

* ``kl_divergence_topk_full`` (путь обучения: у ученика полный словарь, у
  учителя top-k) — это ``Σ_{i∈top-k} p̃_t(i)·(log p̃_t(i) − log p_s(i))``,
  где ``p̃_t`` — softmax по ``k`` логитам учителя, ``p_s`` — обычный softmax
  ученика по всему словарю. Хвост учителя не даёт вклада вовсе;
* ``kl_divergence_topk`` (усечён **и** ученик) считает KL по объединению
  носителей: индексы, которых нет в top-k стороны, получают ``-inf`` либо
  минимальный логит этой стороны (``missing=``). При ``-inf`` у ученика KL
  становится бесконечной — это честный сигнал «k ученика мало», а не ошибка.

Оба варианта — **оценка** полной KL, а не равенство; расхождение измеряется на
tiny (``scripts/tiny_qat.py topk``) и пишется в отчёт задачи.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


class LogitsFormatError(ValueError):
    """Логиты учителя не того формата, который ждёт обучение."""


def expected_rows(answer_len: int) -> tuple[int, int]:
    """Допустимое число строк логитов для ответа длиной ``answer_len``.

    ``answer_len`` — длина ответа **до** добавления eos (так её считает
    ``PairDataset``), поэтому верны оба варианта: с eos (``+1``) и без него.
    """
    if answer_len < 0:
        raise LogitsFormatError(f"длина ответа не может быть отрицательной: {answer_len}")
    return max(0, answer_len - 1), answer_len + 1


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_teacher_logits(
    raw: Any,
    answer_len: int,
    vocab_size: int | None = None,
    *,
    check_values: bool = False,
) -> list[list[float]] | None:
    """Проверить одну строку ``logits`` и вернуть её же (без копии).

    ``None`` — строка непригодна, и пример надо выбросить (так же считает
    ``PairDataset``); ``LogitsFormatError`` — ошибка, которую нельзя молча
    проглотить (словарь учителя не совпал со словарём ученика).

    Копии нет осознанно: для Qwen это ``A × 151936`` чисел на пример, и второй
    такой список в памяти Python стоил бы сотни мегабайт на пару. Поэтому
    ``check_values=True`` (полная проверка каждого числа) — только для
    маленьких словарей: файловый парсер и тесты.
    """
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        return None
    low, high = expected_rows(answer_len)
    if not (low <= len(raw) <= high):
        return None
    first = raw[0]
    if not isinstance(first, list) or not first:
        return None
    width = len(first)
    if vocab_size is not None and width != vocab_size:
        raise LogitsFormatError(
            f"logits учителя имеют размер {width}, а словарь ученика "
            f"{vocab_size}: KL возможен только при общем токенизаторе"
        )
    for row in raw:
        if not isinstance(row, list) or len(row) != width:
            return None
        if check_values:
            for value in row:
                if not _is_number(value) or not math.isfinite(value):
                    return None
    return raw


TOP_K_KEYS = ("topk", "top_k", "teacher_topk")


class TopKLogits:
    """top-k логиты одной пары: ``A`` строк по ``k`` индексов и логитов.

    Индексы отсортированы по убыванию логита (так пишет ``write_topk_file`` и
    ``torch.topk``), логиты конечные; ``indices[r][j]`` — номер токена словаря,
    ``logits[r][j]`` — логит учителя в этой позиции. Первый ``A``-токен
    по-прежнему отвечает позиции ``P - 1 + t`` (см. модульный docstring).
    """

    __slots__ = ("indices", "logits")

    def __init__(self, indices: Sequence[Sequence[int]],
                 logits: Sequence[Sequence[float]]) -> None:
        # Приведение типов не делается: ``_validate_topk`` уже проверил, что
        # индексы — целые, а логиты — конечные числа. На k=128 это ~10 мс на
        # пару, а данные приходят из JSON/torch и так нужных типов.
        self.indices = tuple(tuple(row) for row in indices)
        self.logits = tuple(tuple(row) for row in logits)

    @property
    def k(self) -> int:
        return len(self.indices[0]) if self.indices else 0

    def __len__(self) -> int:
        return len(self.indices)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TopKLogits):
            return NotImplemented
        return self.indices == other.indices and self.logits == other.logits

    def __repr__(self) -> str:
        return f"TopKLogits(rows={len(self)}, k={self.k})"

    def to_json(self, precision: int = 5) -> dict[str, Any]:
        return {
            "k": self.k,
            "indices": [list(row) for row in self.indices],
            "logits": [[round(value, precision) for value in row]
                       for row in self.logits],
        }


def split_teacher_record(record: dict[str, Any]) -> tuple[Any | None, Any | None]:
    """Достать из записи пары (полные логиты, top-k) — не больше одного.

    Канонический вид top-k — вложенный объект ``topk``; принимаются также
    ``top_k``/``teacher_topk`` и «плоские» поля ``topk_indices``/``topk_logits``.
    Приоритет у top-k: файл не должен содержать оба формата в одной строке, но
    если содержит — читается top-k (он и есть актуальный формат).
    """
    for key in TOP_K_KEYS:
        value = record.get(key)
        if value is not None:
            return None, value
    if record.get("topk_indices") is not None or record.get("topk_logits") is not None:
        return None, {
            "indices": record.get("topk_indices"),
            "logits": record.get("topk_logits"),
            "k": record.get("k"),
        }
    return record.get("logits", record.get("teacher_logits")), None


def _validate_topk(
    raw: Any,
    answer_len: int,
    vocab_size: int | None,
    max_k: int | None,
) -> tuple[TopKLogits | None, str | None]:
    """Проверить top-k объект; вернуть (значение, причина отказа).

    Причина — для строгого файлового парсера: в обучении такая строка просто
    выбрасывается, а в артефакте это ошибка с номером строки. Значения
    проверяются **всегда**: ``k`` на порядки меньше ``V``, экономить не на чем,
    зато NaN/inf и битые индексы ловятся здесь, а не в тензоре.
    """
    if raw is None:
        return None, "нет поля topk"
    if not isinstance(raw, dict):
        return None, "topk должен быть объектом {k, indices, logits}"
    indices, logits = raw.get("indices"), raw.get("logits")
    if indices is None or logits is None:
        return None, "в topk нет indices или logits"
    if not isinstance(indices, list) or not isinstance(logits, list) or not indices:
        return None, "indices/logits пусты или не списки"
    if len(indices) != len(logits):
        return None, f"строк индексов {len(indices)}, строк логитов {len(logits)}"
    low, high = expected_rows(answer_len)
    if not (low <= len(indices) <= high):
        return None, (f"строк {len(indices)}, а для ответа {answer_len} "
                      f"допустимо {low}…{high}")
    width = len(indices[0]) if isinstance(indices[0], list) else 0
    if width <= 0:
        return None, "indices[0] — не непустой список"
    declared = raw.get("k")
    if declared is not None:
        if not isinstance(declared, int) or isinstance(declared, bool):
            return None, f"k={declared!r} — не целое"
        if declared != width:
            return None, f"k={declared}, а в строке {width} индексов"
    if vocab_size is not None and width > vocab_size:
        raise LogitsFormatError(
            f"top-k шире словаря: k={width}, словарь ученика {vocab_size}"
        )
    if max_k is not None and width > max_k:
        raise LogitsFormatError(
            f"top-k шире ожидаемого: k={width} > max_top_k={max_k}"
        )
    for row_index, (index_row, logit_row) in enumerate(zip(indices, logits)):
        if not isinstance(index_row, list) or not isinstance(logit_row, list):
            return None, f"строка {row_index}: индексы/логиты — не списки"
        if len(index_row) != width or len(logit_row) != width:
            return None, (f"строка {row_index}: ширина {len(index_row)}/"
                          f"{len(logit_row)}, а в первой {width}")
        # Проверки на C-уровне: на пару A×k значений, и Python-цикл по каждому
        # съедал больше, чем разбор JSON (замер: k=128 — 40 мс/пара).
        # type(x) is int отсекает и bool (bool — подкласс int, но не int).
        if not all(type(index) is int for index in index_row):
            return None, f"строка {row_index}: индексы должны быть целыми"
        if min(index_row) < 0 or (vocab_size is not None and max(index_row) >= vocab_size):
            return None, f"строка {row_index}: индекс вне словаря"
        if len(set(index_row)) != width:
            return None, f"строка {row_index}: индекс повторяется"
        try:
            if not all(map(math.isfinite, logit_row)):
                return None, f"строка {row_index}: логит — не конечное число"
        except TypeError:
            return None, f"строка {row_index}: логит — не число"
        if logit_row != sorted(logit_row, reverse=True):
            return None, f"строка {row_index}: логиты не по убыванию"
    return TopKLogits(indices, logits), None


def parse_topk_logits(
    raw: Any,
    answer_len: int,
    vocab_size: int | None = None,
    *,
    max_k: int | None = None,
) -> TopKLogits | None:
    """Проверить top-k одной пары; ``None`` — строку выбросить.

    ``answer_len``, как и у ``parse_teacher_logits``, — длина ответа до
    добавления eos; допускаются ``A - 1 … A + 1`` строк. ``max_k`` — **верхняя**
    граница ширины: файл шире, чем ждёт прогон, читать нельзя (индексы уже
    ничем не ограничены). ``LogitsFormatError`` — только там, где «выбросить
    пару» было бы тихой порчей: ``k`` больше словаря или больше ``max_k``.
    """
    value, _reason = _validate_topk(raw, answer_len, vocab_size, max_k)
    return value


def iter_logits_file(
    path: str | Path,
    *,
    vocab_size: int | None = None,
    max_k: int | None = None,
    check_values: bool = False,
) -> Iterator[dict[str, Any]]:
    """Потоково читать JSONL с логитами (полными или top-k), по строке за раз.

    Возвращает записи ``{"question", "answer", "logits", "topk", "k", "lineno"}``:
    ровно одно из ``logits``/``topk`` заполнено. Память — одна строка файла;
    на 40 тыс. пар full-vocab файла это единственный способ не съесть RAM.
    """
    path = Path(path)
    if not path.exists():
        raise LogitsFormatError(f"файл логитов не найден: {path}")
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LogitsFormatError(f"{path}:{lineno}: битый JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise LogitsFormatError(f"{path}:{lineno}: ожидался объект JSON")
            question, answer = raw.get("question"), raw.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise LogitsFormatError(f"{path}:{lineno}: нет непустого question")
            if not isinstance(answer, str) or not answer.strip():
                raise LogitsFormatError(f"{path}:{lineno}: нет непустого answer")
            full, topk_raw = split_teacher_record(raw)
            topk: TopKLogits | None = None
            parsed_full: list[list[float]] | None = None
            if topk_raw is not None:
                try:
                    topk, reason = _validate_topk(
                        topk_raw, _topk_rows(topk_raw), vocab_size, max_k,
                    )
                except LogitsFormatError as exc:
                    raise LogitsFormatError(f"{path}:{lineno}: {exc}") from exc
                if topk is None:
                    raise LogitsFormatError(f"{path}:{lineno}: topk: {reason}")
            else:
                if full is None:
                    raise LogitsFormatError(f"{path}:{lineno}: нет поля logits или topk")
                if not isinstance(full, list) or not full:
                    raise LogitsFormatError(f"{path}:{lineno}: logits пусты или не список")
                width = len(full[0]) if isinstance(full[0], list) else 0
                if width <= 0:
                    raise LogitsFormatError(
                        f"{path}:{lineno}: logits[0] — не непустой список")
                try:
                    parsed_full = parse_teacher_logits(
                        full, len(full), vocab_size, check_values=check_values,
                    )
                except LogitsFormatError as exc:
                    raise LogitsFormatError(f"{path}:{lineno}: {exc}") from exc
                if parsed_full is None:
                    raise LogitsFormatError(
                        f"{path}:{lineno}: логиты не прямоугольные [A, {width}]"
                    )
            yield {
                "question": question.strip(),
                "answer": answer.strip(),
                "logits": parsed_full,
                "topk": topk,
                "k": topk.k if topk is not None else None,
                "lineno": lineno,
            }


def _topk_rows(raw: dict[str, Any]) -> int:
    """Число строк top-k (для допуска ``A - 1 … A + 1`` в файловом парсере).

    В файле длина ответа неизвестна, поэтому, как и у полного словаря,
    «длина ответа» = число строк: проверка сводится к ширине и форме.
    """
    if not isinstance(raw, dict):
        return 0
    indices = raw.get("indices")
    return len(indices) if isinstance(indices, list) and indices else 0


def read_logits_file(
    path: str | Path,
    *,
    vocab_size: int | None = None,
    limit: int = 0,
    check_values: bool = False,
    max_k: int | None = None,
) -> list[dict[str, Any]]:
    """Прочитать JSONL с логитами учителя; ошибки — с номером строки.

    Строгость здесь выше, чем в обучении: файл-артефакт должен быть целым
    целиком. Пара без ``question``/``answer``/``logits`` или с логитами не той
    формы — это ``LogitsFormatError``, а не «пропустим и пойдём дальше»:
    иначе оверхед на 40 тыс. пар обнаружится только по просевшему датасету.

    ``check_values`` проверяет каждое число полного словаря (``A × V`` на пару,
    так делает только артефактная проверка); у top-k значения проверяются
    всегда — ``A × k`` на порядки меньше.
    Для больших файлов есть потоковый ``iter_logits_file`` и
    ``qat.data.PairStore``: этот читатель держит в памяти весь файл.
    """
    rows: list[dict[str, Any]] = []
    for row in iter_logits_file(
        path, vocab_size=vocab_size, max_k=max_k, check_values=check_values,
    ):
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    if not rows:
        raise LogitsFormatError(f"в {path} нет ни одной строки с логитами")
    return rows


def read_topk_file(
    path: str | Path,
    *,
    vocab_size: int | None = None,
    limit: int = 0,
    max_k: int | None = None,
) -> list[dict[str, Any]]:
    """Прочитать файл, требуя top-k формат в каждой строке (проверка артефакта)."""
    rows: list[dict[str, Any]] = []
    for row in iter_logits_file(path, vocab_size=vocab_size, max_k=max_k):
        if row["topk"] is None:
            raise LogitsFormatError(
                f"{path}:{row['lineno']}: ожидался top-k, а строка с полным словарём"
            )
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    if not rows:
        raise LogitsFormatError(f"в {path} нет ни одной строки с top-k логитами")
    return rows


def write_logits_file(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Записать JSONL с логитами; вернуть число строк.

    Числа округляются до 5 знаков: ``PairDataset`` всё равно хранит логиты в
    float16, а файл на 40 тыс. пар иначе распухает на гигабайты. Округление
    5e-6 в логите даёт ошибку KL порядка 1e-11 на токен — на фоне float16
    (1e-3) это шум.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            packed = {
                "question": row["question"],
                "answer": row["answer"],
                "logits": [[round(float(value), 5) for value in logit_row]
                           for logit_row in row["logits"]],
            }
            handle.write(json.dumps(packed, ensure_ascii=False) + "\n")
            written += 1
    return written


def topk_from_logits(
    values: Sequence[float], k: int,
) -> tuple[list[int], list[float]]:
    """top-k одной строки полных логитов: индексы и логиты по убыванию.

    Хвост (всё за ``k``) **отбрасывается** — это и есть приближение формата.
    Ties разрешаются стабильной сортировкой (меньший индекс выше), чтобы
    запись файла была детерминированной.
    """
    if k <= 0:
        raise ValueError(f"k должен быть > 0, а не {k}")
    if not values:
        raise ValueError("пустой вектор логитов")
    if k > len(values):
        raise ValueError(f"k={k} больше словаря {len(values)}")
    order = sorted(range(len(values)), key=lambda index: float(values[index]),
                   reverse=True)[:k]
    logits = [float(values[index]) for index in order]
    for value in logits:
        if not math.isfinite(value):
            raise ValueError("в логитах NaN/inf: top-k из них не построить")
    return [int(index) for index in order], logits


def topk_from_rows(rows: Sequence[Sequence[float]], k: int) -> TopKLogits:
    """top-k для ``A`` строк полных логитов (чистый Python)."""
    pairs = [topk_from_logits(row, k) for row in rows]
    return TopKLogits([indices for indices, _ in pairs], [logits for _, logits in pairs])


def write_topk_file(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    k: int = 0,
    precision: int = 5,
) -> int:
    """Записать JSONL в top-k формате; вернуть число строк.

    ``rows`` — записи с готовым ``topk`` (``TopKLogits`` или объект
    ``{k, indices, logits}``) либо с полным словарём в ``logits``: тогда
    ``k`` обязателен, и хвост отбрасывается здесь (``topk_from_rows``).
    ``k=0`` допустим только для готовых top-k записей.

    ``precision`` — знаков после запятой: JSON — текст, и 5 знаков дают ошибку
    KL ~1e-11 на токен, то есть заведомо ниже fp16-шума. Меньше 5 не стоит:
    логиты у Qwen доходят до ±30, при 4 знаках ошибка KL уже ~1e-7.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            full, raw_topk = split_teacher_record(row)
            if isinstance(raw_topk, TopKLogits):
                topk = raw_topk
            elif raw_topk is not None:
                topk, reason = _validate_topk(
                    raw_topk, _topk_rows(raw_topk) or 1, None, None)
                if topk is None:
                    raise LogitsFormatError(f"topk записи негоден: {reason}")
            else:
                if full is None:
                    raise LogitsFormatError("в записи нет ни logits, ни topk")
                if k <= 0:
                    raise LogitsFormatError(
                        "k не задан, а логиты полные: укажите k для усечения")
                if not isinstance(full, list) or not full:
                    raise LogitsFormatError("logits пусты или не список")
                if not isinstance(full[0], list):
                    raise LogitsFormatError("logits[0] — не список")
                width = len(full[0])
                if len(full) and any(len(logit_row) != width for logit_row in full):
                    raise LogitsFormatError("логиты не прямоугольные: ширина строк разная")
                topk = topk_from_rows(full, k)
            if k and k != topk.k:
                # Из готового top-k хвост не восстановить: пере-усечение молча
                # записало бы файл с чужим k.
                raise LogitsFormatError(
                    f"k={k}, а в записи top-k с k={topk.k}: "
                    f"пере-усечь готовый top-k нельзя (хвост потерян)"
                )
            packed = {
                "question": row["question"],
                "answer": row["answer"],
                "topk": topk.to_json(precision),
            }
            # Ровно те же данные без пробелов после запятых/двоеточий: на k=64
            # это ~20% объёма файла, а формат остаётся обычным JSON.
            handle.write(json.dumps(packed, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Эталонный KL (чистый Python) и torch-путь (тот же расчёт)
# ---------------------------------------------------------------------------


def log_softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    """``log_softmax(values / T)`` — устойчиво, без потери на больших логитах."""
    if temperature <= 0:
        raise ValueError("temperature должен быть > 0")
    scaled = [float(value) / temperature for value in values]
    top = max(scaled)
    shifted = [value - top for value in scaled]
    log_z = math.log(sum(math.exp(value) for value in shifted))
    return [value - log_z for value in shifted]


def kl_divergence(
    teacher_logits: Sequence[float],
    student_logits: Sequence[float],
    temperature: float = 1.0,
) -> float:
    """KL(учитель ‖ ученик) для одного токена, с множителем ``T²``.

    Ровно то, что считает ``torch.nn.functional.kl_div(log_target=True)``:
    ``Σ_v p_t · (log p_t − log p_s)``, где ``p_t = softmax(teacher / T)``,
    ``p_s = softmax(student / T)``, результат умножен на ``T²`` (градиенты
    KL не зависят от температуры — Hinton et al.).
    """
    if len(teacher_logits) != len(student_logits):
        raise ValueError(
            f"словари не совпадают: у учителя {len(teacher_logits)}, "
            f"у ученика {len(student_logits)}"
        )
    if not teacher_logits:
        return 0.0
    log_t = log_softmax(teacher_logits, temperature)
    log_s = log_softmax(student_logits, temperature)
    total = 0.0
    for lt, ls in zip(log_t, log_s):
        total += math.exp(lt) * (lt - ls)
    return total * (temperature ** 2)


def kl_div_sum(
    teacher_rows: Sequence[Sequence[float]],
    student_rows: Sequence[Sequence[float]],
    temperature: float = 1.0,
) -> float:
    """Сумма KL по токенам ответа (так же суммирует ``train._kl_sum``)."""
    if len(teacher_rows) != len(student_rows):
        raise ValueError(
            f"строк логитов учителя {len(teacher_rows)}, ученика {len(student_rows)}"
        )
    return sum(
        kl_divergence(teacher, student, temperature)
        for teacher, student in zip(teacher_rows, student_rows)
    )


def kl_div_sum_torch(torch_module: Any, student: Any, teacher: Any,
                     temperature: float = 1.0) -> Any:
    """Тот же KL на torch-тензорах ``[n, vocab]`` (fp32), с ``T²``.

    ``torch_module`` передаётся аргументом, чтобы модуль импортировался без
    torch. Числа совпадают с ``kl_div_sum`` (проверяется в
    ``tests/test_kl_distill.py``).
    """
    import torch.nn.functional as F  # noqa: PLC0415 - ленивый импорт

    if student.size(-1) != teacher.size(-1):
        raise ValueError(
            f"словари не совпадают: у учителя {teacher.size(-1)}, "
            f"у ученика {student.size(-1)}"
        )
    return F.kl_div(
        F.log_softmax(student.float() / temperature, dim=-1),
        F.log_softmax(teacher.to(device=student.device, dtype=torch_module.float32)
                      / temperature, dim=-1),
        log_target=True,
        reduction="sum",
    ) * (temperature ** 2)


# ---------------------------------------------------------------------------
# KL по top-k: объединение носителей и путь обучения (у учителя top-k)
# ---------------------------------------------------------------------------


def _support_union(
    teacher_indices: Sequence[int], student_indices: Sequence[int],
) -> list[int]:
    """Объединение индексов top-k учителя и ученика, по возрастанию."""
    return sorted(set(int(index) for index in teacher_indices)
                  | set(int(index) for index in student_indices))


def kl_divergence_topk_full(
    teacher_indices: Sequence[int],
    teacher_logits: Sequence[float],
    student_logits: Sequence[float],
    temperature: float = 1.0,
) -> float:
    """KL(учитель ‖ ученик) одного токена: у учителя top-k, у ученика — словарь.

    Это **путь обучения**: модель ученика всё равно считает логиты по всему
    словарю, поэтому ``p_s`` берётся точной (softmax по ``V``), а от учителя
    остаются только ``k`` индексов, и его распределение перенормируется на них::

        KL ≈ Σ_{i∈top-k} p̃_t(i) · (log p̃_t(i) − log p_s(i))

    Масса учителя вне top-k отбрасывается (``-inf`` для остальных индексов),
    поэтому результат — оценка, а не равенство полной KL: при ``k ≪ V`` он
    смещён вниз, если хвост учителя тяжёлый. Если top-k покрывает весь словарь,
    результат совпадает с ``kl_divergence`` (проверяется тестом).

    ``teacher_indices``/``teacher_logits`` — одна строка top-k (индексы в
    ``[0, V)``, логиты по убыванию), ``student_logits`` — полный вектор ``V``.
    """
    if len(teacher_indices) != len(teacher_logits):
        raise ValueError(
            f"индексов {len(teacher_indices)}, логитов {len(teacher_logits)}"
        )
    if not teacher_indices:
        raise ValueError("пустой top-k: KL не из чего считать")
    if not student_logits:
        raise ValueError("пустой вектор логитов ученика")
    for index in teacher_indices:
        if not 0 <= int(index) < len(student_logits):
            raise ValueError(
                f"индекс учителя {index} вне словаря ученика ({len(student_logits)})"
            )
    log_t = log_softmax(teacher_logits, temperature)
    log_s = log_softmax(student_logits, temperature)
    total = 0.0
    for index, lt in zip(teacher_indices, log_t):
        total += math.exp(lt) * (lt - log_s[int(index)])
    return total * (temperature ** 2)


def kl_divergence_topk(
    teacher_indices: Sequence[int],
    teacher_logits: Sequence[float],
    student_indices: Sequence[int],
    student_logits: Sequence[float],
    temperature: float = 1.0,
    *,
    missing: str = "neg_inf",
) -> float:
    """KL(учитель ‖ ученик) одного токена по **объединению** двух top-k.

    Усечены обе стороны: учитель знает ``k_t`` индексов, ученик — ``k_s``.
    Считаем по объединению их носителей; для индекса, которого у стороны нет,
    логит подставляется по ``missing``:

    * ``"neg_inf"`` (по умолчанию) — ``-inf``, как и велит формат: масса там
      нулевая. Если у ученика нет индекса, где у учителя есть масса, KL
      **бесконечна**: это не сбой, а сигнал, что ``k_s`` мало для этого токена;
    * ``"min"`` — минимальный логит этой стороны (её ``k``-й). Хвост
      оценивается «сверху»: реальные логиты за top-k не больше ``k``-го,
      поэтому такая замена даёт конечную, но пессимистичную оценку.

    Так же, как ``kl_divergence_topk_full``, это приближение: хвост учителя
    (и хвост ученика) отброшен, распределения перенормированы на объединение.
    """
    if missing not in ("neg_inf", "min"):
        raise ValueError(f"missing={missing!r}: допустимо 'neg_inf' или 'min'")
    if len(teacher_indices) != len(teacher_logits):
        raise ValueError(
            f"у учителя индексов {len(teacher_indices)}, логитов {len(teacher_logits)}"
        )
    if len(student_indices) != len(student_logits):
        raise ValueError(
            f"у ученика индексов {len(student_indices)}, логитов {len(student_logits)}"
        )
    if not teacher_indices or not student_indices:
        raise ValueError("пустой top-k: KL не из чего считать")
    teacher_map = {int(index): float(value)
                   for index, value in zip(teacher_indices, teacher_logits)}
    student_map = {int(index): float(value)
                   for index, value in zip(student_indices, student_logits)}
    union = _support_union(teacher_indices, student_indices)
    if missing == "min":
        teacher_floor = min(teacher_map.values())
        student_floor = min(student_map.values())
        teacher_vector = [teacher_map.get(index, teacher_floor) for index in union]
        student_vector = [student_map.get(index, student_floor) for index in union]
    else:
        teacher_vector = [teacher_map.get(index, -math.inf) for index in union]
        student_vector = [student_map.get(index, -math.inf) for index in union]
    log_t = log_softmax(teacher_vector, temperature)
    log_s = log_softmax(student_vector, temperature)
    total = 0.0
    for lt, ls in zip(log_t, log_s):
        if lt == -math.inf:  # у учителя масса ноль — член KL нулевой
            continue
        if ls == -math.inf:
            return math.inf
        total += math.exp(lt) * (lt - ls)
    return total * (temperature ** 2)


def kl_div_sum_topk(
    teacher: TopKLogits,
    student: TopKLogits | Sequence[Sequence[float]],
    temperature: float = 1.0,
    *,
    missing: str = "neg_inf",
) -> float:
    """Сумма KL по токенам ответа для top-k учителя и top-k/словаря ученика.

    ``student`` — ``TopKLogits`` (тогда считается по объединению, см.
    ``kl_divergence_topk``) или ``A`` строк полных логитов (тогда точная
    сторона ученика, см. ``kl_divergence_topk_full`` — это путь обучения).
    """
    if len(teacher) != len(student):
        raise ValueError(
            f"строк top-k учителя {len(teacher)}, ученика {len(student)}"
        )
    if isinstance(student, TopKLogits):
        return sum(
            kl_divergence_topk(
                teacher.indices[row], teacher.logits[row],
                student.indices[row], student.logits[row],
                temperature, missing=missing,
            )
            for row in range(len(teacher))
        )
    return sum(
        kl_divergence_topk_full(
            teacher.indices[row], teacher.logits[row], student[row], temperature,
        )
        for row in range(len(teacher))
    )


def kl_div_sum_topk_torch(
    torch_module: Any,
    student: Any,
    teacher_indices: Any,
    teacher_values: Any,
    temperature: float = 1.0,
) -> Any:
    """``kl_div_sum_topk`` на тензорах: ученик ``[A, V]``, учитель ``[A, k]``.

    Ученик — полный словарь (так его и считает модель), поэтому хвост учителя
    единственное приближение; числа совпадают с ``kl_divergence_topk_full``.
    ``student.float()`` — чтобы fp16-логиты не теряли точность в softmax;
    ``teacher_values`` может быть fp16 (``PairDataset``) — приводится к fp32.
    """
    import torch.nn.functional as F  # noqa: PLC0415 - ленивый импорт

    if student.dim() != 2:
        raise ValueError(f"логиты ученика должны быть [A, V], а не {tuple(student.shape)}")
    if teacher_indices.shape != teacher_values.shape:
        raise ValueError(
            f"индексов {tuple(teacher_indices.shape)}, "
            f"логитов {tuple(teacher_values.shape)}"
        )
    if teacher_indices.shape[0] != student.shape[0]:
        raise ValueError(
            f"строк учителя {teacher_indices.shape[0]}, ученика {student.shape[0]}"
        )
    if teacher_indices.numel() == 0:
        raise ValueError("пустой top-k: KL не из чего считать")
    vocab = int(student.size(-1))
    lowest = int(teacher_indices.min().item())
    highest = int(teacher_indices.max().item())
    if lowest < 0 or highest >= vocab:
        raise ValueError(
            f"индексы учителя в [{lowest}, {highest}], а словарь ученика {vocab}"
        )
    log_student = F.log_softmax(student.float() / temperature, dim=-1)
    picked = log_student.gather(
        1, teacher_indices.to(device=student.device, dtype=torch_module.long)
    )
    log_teacher = F.log_softmax(
        teacher_values.to(device=student.device, dtype=torch_module.float32)
        / temperature,
        dim=-1,
    )
    return F.kl_div(picked, log_teacher, log_target=True, reduction="sum") * (
        temperature ** 2
    )
