"""Данные QAT: пары «вопрос → ответ» из JSONL в токены с маской на вопрос.

Формат строки (его пишет distill/run.py)::

    {"question": "...", "answer": "...", "meta": {...}}

Учим только на токенах ответа: вопрос и padding получают label = -100.
Поле ``logits`` (или ``teacher_logits``) читается только при
``data.use_teacher_logits: true`` — это распределения учителя по токенам
ответа для KL-лосса; вместо полного словаря допустим top-k формат
``{"topk": {"k", "indices", "logits"}}`` (задача 0111). Формат строк и
допуски — ``qat.kl`` (модуль без torch).

Память
======
Полный словарь на паре — это ``A × V`` чисел: для Qwen (``V=151936``) пара из
402 токенов занимает ~116 МиБ в fp16 и ~1,7 ГБ Python-списками, а 40 тыс. пар —
это 4,9 ТБ. Поэтому на обучении файл читается **потоково**: ``PairStore``
держит в памяти только ``question``/``answer`` и смещения строк, а логиты
(``A × k`` у top-k) поднимаются из файла на каждый пример и живут до конца
``encode``. ``load_pairs`` остаётся для маленьких файлов и тестов и умеет
``with_logits=False`` — тогда логиты в RAM не копируются вообще.

Поток данных детерминирован и возобновляем: позиция задаётся числом
обработанных примеров ``samples_seen``; порядок внутри эпохи — перестановка
с seed ``seed + epoch``. Никакого состояния DataLoader в чекпоинте нет.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

from .kl import LogitsFormatError, parse_teacher_logits, parse_topk_logits, split_teacher_record
from .util import file_fingerprint


class DataError(RuntimeError):
    """Данные не читаются или не подходят под формат."""


def load_pairs(
    path: str | Path,
    max_samples: int = 0,
    *,
    with_logits: bool = True,
) -> list[dict[str, Any]]:
    """Прочитать JSONL и оставить строки с непустыми question/answer.

    ``with_logits=False`` не кладёт логиты в RAM: для полного словаря это и
    есть разница между «1,7 ГБ на пару» и парой строк текста. Обучение
    логиты не теряет — оно читает их через ``PairStore`` по требованию.
    """
    path = Path(path)
    if not path.exists():
        raise DataError(
            f"файл данных не найден: {path}\n"
            f"соберите пары: make distill-mock (проверка) или make distill (учитель)"
        )
    rows: list[dict[str, Any]] = []
    skipped = 0
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{lineno}: битый JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise DataError(f"{path}:{lineno}: ожидался объект JSON")
            question, answer = row.get("question"), row.get("answer")
            if not isinstance(question, str) or not question.strip():
                skipped += 1
                continue
            if not isinstance(answer, str) or not answer.strip():
                skipped += 1
                continue
            pair: dict[str, Any] = {
                "question": question.strip(),
                "answer": answer.strip(),
                "logits": None,
            }
            if with_logits:
                full, topk = split_teacher_record(row)
                pair["logits"] = full
                pair["topk"] = topk
            rows.append(pair)
            if max_samples and len(rows) >= max_samples:
                break
    if skipped:
        print(f"[qat] данных: пропущено {skipped} строк без question/answer")
    if not rows:
        raise DataError(f"в {path} нет пригодных пар question/answer")
    return rows


class PairStore:
    """Потоковое чтение пар: в RAM только question/answer и смещения строк.

    Один проход по файлу на старте запоминает смещение и текст вопроса/ответа
    каждой пригодной строки; логиты читаются на каждый ``store[index]``
    (seek + ``json.loads`` одной строки) и после ``encode`` выбрасываются.
    Память не зависит от ширины логитов: 40 тыс. пар — это ~8 МБ смещений и
    строк, а не 4,9 ТБ логитов.

    Строка без ``question``/``answer`` пропускается (как в ``load_pairs``);
    битый JSON — ``DataError`` с номером строки. Отсутствие ``logits``/``topk``
    здесь не ошибка: ``PairDataset`` отбросит пару со счётчиком
    ``dropped_no_logits`` — так же, как при чтении через ``load_pairs``.
    """

    def __init__(self, path: str | Path, max_samples: int = 0) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise DataError(
                f"файл данных не найден: {self.path}\n"
                f"соберите пары: make distill-mock (проверка) или make distill (учитель)"
            )
        self._rows: list[tuple[int, int, str, str]] = []
        skipped = 0
        with self.path.open("rb") as handle:
            offset = handle.tell()
            lineno = 0
            while True:
                line = handle.readline()
                if not line:
                    break
                lineno += 1
                if line.strip():
                    record = self._decode(line, lineno)
                    question, answer = record.get("question"), record.get("answer")
                    if (not isinstance(question, str) or not question.strip()
                            or not isinstance(answer, str) or not answer.strip()):
                        skipped += 1
                    else:
                        self._rows.append(
                            (offset, lineno, question.strip(), answer.strip())
                        )
                        if max_samples and len(self._rows) >= max_samples:
                            break
                offset = handle.tell()
        self.skipped = skipped
        if skipped:
            print(f"[qat] данных: пропущено {skipped} строк без question/answer")
        if not self._rows:
            raise DataError(f"в {self.path} нет пригодных пар question/answer")

    def _decode(self, line: bytes, lineno: int) -> dict[str, Any]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"{self.path}:{lineno}: битый JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise DataError(f"{self.path}:{lineno}: ожидался объект JSON")
        return record

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self._rows)):
            yield self[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self._rows)
        if not 0 <= index < len(self._rows):
            raise IndexError(f"пара {index} вне файла (пар {len(self._rows)})")
        offset, lineno, question, answer = self._rows[index]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            line = handle.readline()
        record = self._decode(line, lineno)
        full, topk = split_teacher_record(record)
        return {
            "question": question,
            "answer": answer,
            "logits": full,
            "topk": topk,
            "lineno": lineno,
        }


def data_fingerprint(path: str | Path, rows: int) -> dict[str, Any]:
    """Отпечаток данных для проверки при --resume (файл не должен подменяться)."""
    fp = file_fingerprint(path)
    fp["samples"] = rows
    return fp


def collate_samples(samples: Sequence[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    """Склеить примеры в микро-батч, дополнив до самой длинной последовательности."""
    if not samples:
        raise DataError("пустой микро-батч")
    max_len = max(len(s["input_ids"]) for s in samples)
    batch_size = len(samples)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    for i, sample in enumerate(samples):
        length = len(sample["input_ids"])
        input_ids[i, :length] = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask[i, :length] = 1
        labels[i, :length] = torch.tensor(sample["labels"], dtype=torch.long)

    teacher_logits = None
    # Элемент списка — либо тензор ``[A, vocab]`` (полный словарь), либо
    # ``{"indices", "logits"}`` (top-k, задача 0111): ``train._kl_sum``
    # разбирает формат по типу элемента.
    present = [s.get("teacher_logits") is not None or s.get("teacher_topk") is not None
               for s in samples]
    if any(present):
        if not all(present):
            raise DataError("в батче смешаны примеры с logits учителя и без них")
        teacher_logits = [
            s["teacher_logits"] if s.get("teacher_logits") is not None
            else s["teacher_topk"]
            for s in samples
        ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "n_tokens": int((labels != -100).sum().item()),
        "teacher_logits": teacher_logits,
    }


class PairDataset:
    """Пары в токенах. Токенизация — лениво, в момент выборки.

    ``rows`` — список записей (``load_pairs``) **или** ``PairStore``: нужны
    только ``len()`` и ``rows[index]``, поэтому запись из файла не копируется
    в RAM целиком. Кеша нет осознанно: 30 млн токенов в Python-списках съедят
    гигабайты, а быстрый токенизатор повторно разбирает эпоху за
    секунды-минуты на фоне часов QAT.
    """

    def __init__(
        self,
        rows: Sequence[dict[str, Any]] | PairStore,
        tokenizer: Any,
        *,
        max_seq_len: int,
        long_pairs: str = "truncate",
        min_answer_tokens: int = 1,
        use_teacher_logits: bool = False,
        vocab_size: int | None = None,
        expected_top_k: int | None = None,
    ) -> None:
        # Без list(rows): у PairStore это значило бы прочитать и разобрать
        # логиты всех пар сразу — ровно то, от чего задача 0111 уходит.
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.long_pairs = long_pairs
        self.min_answer_tokens = min_answer_tokens
        self.use_teacher_logits = use_teacher_logits
        self.vocab_size = vocab_size
        # Ожидаемая ширина k: если задана, файл с другой шириной — ошибка, а не
        # «пропустим примеры» (иначе прогон молча теряет KL-путь целиком).
        self.expected_top_k = expected_top_k
        self.stats: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self.rows)

    def _prompt_ids(self, question: str) -> list[int]:
        """Chat-темплейт токенизатора + add_generation_prompt, как в инференсе."""
        messages = [{"role": "user", "content": question}]
        if getattr(self.tokenizer, "chat_template", None):
            ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            return list(ids)
        # Запасной вариант для базовой модели без chat_template.
        text = f"Пользователь: {question}\nАссистент:"
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _teacher_logits(self, raw: Any, answer_len: int) -> torch.Tensor | None:
        """Логиты учителя по токенам ответа → float16-тензор ``[A, vocab]``.

        Формат и допуски — в ``qat.kl`` (там же опорный расчёт и парсер файла):
        строка ``t`` обязана быть логитами позиции, предсказывающей ``t``-й
        токен ответа (``P - 1 + t``). Проверка формы общая с тестами без torch,
        а конвертация и проверка на NaN/inf — здесь, на тензоре.
        """
        if raw is None:
            self.stats["dropped_no_logits"] += 1
            return None
        try:
            rows = parse_teacher_logits(raw, answer_len, self.vocab_size)
        except LogitsFormatError as exc:
            raise DataError(str(exc)) from exc
        if rows is None:
            self.stats["dropped_bad_logits"] += 1
            return None
        try:
            tensor = torch.tensor(rows, dtype=torch.float16)
        except (TypeError, ValueError):
            self.stats["dropped_bad_logits"] += 1
            return None
        if tensor.ndim != 2 or tensor.shape[1] != len(rows[0]):
            self.stats["dropped_bad_logits"] += 1
            return None
        # NaN/inf в логитах учителя не ловятся формой, а обучение ими отравится.
        if not bool(torch.isfinite(tensor).all()):
            self.stats["dropped_bad_logits"] += 1
            return None
        return tensor

    def _teacher_topk(self, raw: Any, answer_len: int) -> dict[str, Any] | None:
        """top-k логиты учителя по токенам ответа → индексы/логиты тензорами.

        ``indices`` — ``int64 [A, k]``, ``logits`` — ``float32 [A, k]``. Логиты
        намеренно fp32, а не fp16, как у полного словаря: на ``k`` экономить
        нечего (``A × k × 4`` байт), а ошибка приближения не складывается с
        ошибкой округления. Позиции те же, что у полного формата (``P-1+t``),
        иначе KL выровняется не на те токены — см. ``qat.kl``.
        """
        if raw is None:
            self.stats["dropped_no_logits"] += 1
            return None
        try:
            topk = parse_topk_logits(raw, answer_len, self.vocab_size)
        except LogitsFormatError as exc:
            raise DataError(str(exc)) from exc
        if topk is None:
            self.stats["dropped_bad_topk"] += 1
            return None
        if self.expected_top_k is not None and topk.k != self.expected_top_k:
            # Ширина — часть интерфейса прогона: файл с другим k означает, что
            # данные собраны не той командой, и молча учить на них нельзя.
            raise DataError(
                f"top-k ширины {topk.k}, а прогон ожидает k={self.expected_top_k}: "
                f"пересоберите файл логитов с тем же k"
            )
        try:
            indices = torch.tensor(topk.indices, dtype=torch.long)
            logits = torch.tensor(topk.logits, dtype=torch.float32)
        except (TypeError, ValueError):
            self.stats["dropped_bad_topk"] += 1
            return None
        if (indices.ndim != 2 or logits.shape != indices.shape
                or indices.size(1) != topk.k):
            self.stats["dropped_bad_topk"] += 1
            return None
        # NaN/inf ловятся уже в парсере; повтор — потому что отравленный логит
        # не видно по форме, а обучение им испортится молча.
        if not bool(torch.isfinite(logits).all()):
            self.stats["dropped_bad_topk"] += 1
            return None
        self.stats[f"topk_k_{topk.k}"] += 1
        return {"indices": indices, "logits": logits}

    def encode(self, index: int) -> dict[str, Any] | None:
        """Пример в виде списков или None, если его надо выбросить."""
        row = self.rows[index]
        prompt_ids = self._prompt_ids(row["question"])
        answer_ids = list(self.tokenizer(row["answer"], add_special_tokens=False).input_ids)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and (not answer_ids or answer_ids[-1] != eos_id):
            answer_ids = answer_ids + [eos_id]
        if not answer_ids:
            self.stats["dropped_empty_answer"] += 1
            return None
        if len(prompt_ids) + len(answer_ids) > self.max_seq_len:
            if self.long_pairs == "drop" or len(prompt_ids) >= self.max_seq_len:
                self.stats["dropped_long"] += 1
                return None
            # Обрезаем ответ, оставляя eos в конце: иначе модель не учится
            # останавливаться, а потеря хвоста — цена, которую видно в метриках.
            keep = self.max_seq_len - len(prompt_ids)
            answer_ids = answer_ids[:keep]
            if eos_id is not None:
                answer_ids[-1] = eos_id
            self.stats["truncated_long"] += 1
        # Длина считается по обрезанному ответу: логиты учителя снимаются с
        # тех же токенов (см. scripts/tiny_qat.py logits). Раньше здесь стояла
        # длина до обрезки, и на парах длиннее max_seq_len (15,6% датасета)
        # любая строка логитов отбрасывалась как «не той длины» — KL-путь
        # молча оставался без данных.
        answer_tokens = len(answer_ids)
        labels = [-100] * len(prompt_ids) + answer_ids
        if sum(1 for x in labels if x != -100) < self.min_answer_tokens:
            self.stats["dropped_short"] += 1
            return None

        teacher = None
        teacher_topk = None
        if self.use_teacher_logits:
            raw_topk = row.get("topk")
            if raw_topk is not None:
                teacher_topk = self._teacher_topk(raw_topk, answer_tokens)
                if teacher_topk is None:
                    return None
            else:
                teacher = self._teacher_logits(row.get("logits"), answer_tokens)
                if teacher is None:
                    return None
        self.stats["encoded"] += 1
        return {
            "input_ids": prompt_ids + answer_ids,
            "labels": labels,
            "teacher_logits": teacher,
            "teacher_topk": teacher_topk,
        }

    def stats_snapshot(self) -> dict[str, int]:
        return dict(self.stats)


class PairStream:
    """Возобновляемый поток шагов: шаг = grad_accum микро-батчей.

    Позиция — абсолютное число попыток выборки (``samples_seen``). Эпоха и
    позиция внутри эпохи выводятся из него, поэтому чекпоинт содержит только
    одно число и не зависит от состояния итераторов.
    """

    def __init__(
        self,
        dataset: PairDataset,
        *,
        micro_batch_size: int,
        grad_accum_steps: int,
        seed: int,
        pad_token_id: int,
        start_sample: int = 0,
    ) -> None:
        if len(dataset) == 0:
            raise DataError("пустой датасет")
        self.dataset = dataset
        self.micro_batch_size = micro_batch_size
        self.grad_accum_steps = grad_accum_steps
        self.seed = seed
        self.pad_token_id = pad_token_id
        self.samples_seen = max(0, int(start_sample))
        self._epoch = self.samples_seen // len(dataset)
        self._pos = self.samples_seen % len(dataset)
        self._order: list[int] | None = None
        self._order_epoch = -1

    @property
    def epoch(self) -> int:
        """Номер эпохи для логов, с единицы."""
        return (self.samples_seen - 1) // len(self.dataset) + 1 if self.samples_seen else 1

    def _ensure_order(self) -> None:
        if self._order_epoch == self._epoch and self._order is not None:
            return
        order = list(range(len(self.dataset)))
        random.Random(self.seed + self._epoch).shuffle(order)
        self._order = order
        self._order_epoch = self._epoch

    def _next_index(self) -> int:
        self._ensure_order()
        if self._pos >= len(self.dataset):
            self._epoch += 1
            self._pos = 0
            self._ensure_order()
        assert self._order is not None
        index = self._order[self._pos]
        self._pos += 1
        self.samples_seen += 1
        return index

    def next_step(self) -> list[dict[str, Any]]:
        """Собрать один шаг: список микро-батчей длиной grad_accum_steps."""
        batches: list[dict[str, Any]] = []
        for _ in range(self.grad_accum_steps):
            samples: list[dict[str, Any]] = []
            attempts = 0
            limit = self.micro_batch_size * 100 + 100
            while len(samples) < self.micro_batch_size:
                if attempts >= limit:
                    raise DataError(
                        "не удалось собрать микро-батч: слишком много примеров "
                        f"отброшено ({self.dataset.stats_snapshot()}); проверьте данные "
                        "и data.max_seq_len"
                    )
                attempts += 1
                example = self.dataset.encode(self._next_index())
                if example is not None:
                    samples.append(example)
            batches.append(collate_samples(samples, self.pad_token_id))
        return batches


class RandomTokenStream:
    """Заглушка для --dry-run: те же формы тензоров, но без датасета."""

    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        micro_batch_size: int,
        grad_accum_steps: int,
        seed: int = 0,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.micro_batch_size = micro_batch_size
        self.grad_accum_steps = grad_accum_steps
        self.generator = torch.Generator().manual_seed(seed)
        self.samples_seen = 0

    @property
    def epoch(self) -> int:
        return 1

    def next_step(self) -> list[dict[str, Any]]:
        total = self.micro_batch_size * self.grad_accum_steps
        input_ids = torch.randint(
            0, self.vocab_size, (total, self.seq_len), generator=self.generator
        )
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        self.samples_seen += total
        batches = []
        for start in range(0, total, self.micro_batch_size):
            ids = input_ids[start:start + self.micro_batch_size]
            batches.append({
                "input_ids": ids,
                "attention_mask": attention_mask[start:start + self.micro_batch_size],
                "labels": labels[start:start + self.micro_batch_size],
                "n_tokens": int(ids.numel()),
                "teacher_logits": None,
            })
        return batches
