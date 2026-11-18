"""Данные QAT: пары «вопрос → ответ» из JSONL в токены с маской на вопрос.

Формат строки (его пишет distill/run.py)::

    {"question": "...", "answer": "...", "meta": {...}}

Учим только на токенах ответа: вопрос и padding получают label = -100.
Поле ``logits`` (или ``teacher_logits``) читается только при
``data.use_teacher_logits: true`` — это распределения учителя по токенам
ответа для KL-лосса. Формат строк и допуски — ``qat.kl`` (модуль без torch).

Поток данных детерминирован и возобновляем: позиция задаётся числом
обработанных примеров ``samples_seen``; порядок внутри эпохи — перестановка
с seed ``seed + epoch``. Никакого состояния DataLoader в чекпоинте нет.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch

from .kl import LogitsFormatError, parse_teacher_logits
from .util import file_fingerprint


class DataError(RuntimeError):
    """Данные не читаются или не подходят под формат."""


def load_pairs(path: str | Path, max_samples: int = 0) -> list[dict[str, Any]]:
    """Прочитать JSONL и оставить строки с непустыми question/answer."""
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
            rows.append({
                "question": question.strip(),
                "answer": answer.strip(),
                "logits": row.get("logits", row.get("teacher_logits")),
            })
            if max_samples and len(rows) >= max_samples:
                break
    if skipped:
        print(f"[qat] данных: пропущено {skipped} строк без question/answer")
    if not rows:
        raise DataError(f"в {path} нет пригодных пар question/answer")
    return rows


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
    if any(s.get("teacher_logits") is not None for s in samples):
        if not all(s.get("teacher_logits") is not None for s in samples):
            raise DataError("в батче смешаны примеры с logits учителя и без них")
        teacher_logits = [s["teacher_logits"] for s in samples]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "n_tokens": int((labels != -100).sum().item()),
        "teacher_logits": teacher_logits,
    }


class PairDataset:
    """Пары в токенах. Токенизация — лениво, в момент выборки.

    Кеша нет осознанно: 30 млн токенов в Python-списках съедят гигабайты,
    а быстрый токенизатор повторно разбирает эпоху за секунды-минуты на фоне
    часов QAT.
    """

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        *,
        max_seq_len: int,
        long_pairs: str = "truncate",
        min_answer_tokens: int = 1,
        use_teacher_logits: bool = False,
        vocab_size: int | None = None,
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.long_pairs = long_pairs
        self.min_answer_tokens = min_answer_tokens
        self.use_teacher_logits = use_teacher_logits
        self.vocab_size = vocab_size
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
        if self.use_teacher_logits:
            teacher = self._teacher_logits(row.get("logits"), answer_tokens)
            if teacher is None:
                return None
        self.stats["encoded"] += 1
        return {
            "input_ids": prompt_ids + answer_ids,
            "labels": labels,
            "teacher_logits": teacher,
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
