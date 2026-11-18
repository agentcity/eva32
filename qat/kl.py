"""KL-дистилляция: формат логитов учителя, разбор файла и опорный расчёт.

Модуль **не импортирует torch**: и проверка формата, и эталонный KL считаются
на чистом Python, поэтому проверяются обычным ``python3 -m unittest`` без
GPU-стека. Torch-путь (``kl_div_sum_torch``) импортирует torch лениво — им
пользуется ``qat.train`` на реальном прогоне.

Формат строки файла логитов (JSONL, по строке на пару)::

    {"question": "…", "answer": "…",
     "logits": [[… V чисел …], … A строк …]}

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
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


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


def read_logits_file(
    path: str | Path,
    *,
    vocab_size: int | None = None,
    limit: int = 0,
    check_values: bool = False,
) -> list[dict[str, Any]]:
    """Прочитать JSONL с логитами учителя; ошибки — с номером строки.

    Строгость здесь выше, чем в обучении: файл-артефакт должен быть целым
    целиком. Пара без ``question``/``answer``/``logits`` или с логитами не той
    формы — это ``LogitsFormatError``, а не «пропустим и пойдём дальше»:
    иначе оверхед на 40 тыс. пар обнаружится только по просевшему датасету.
    """
    path = Path(path)
    if not path.exists():
        raise LogitsFormatError(f"файл логитов не найден: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LogitsFormatError(f"{path}:{lineno}: битый JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise LogitsFormatError(f"{path}:{lineno}: ожидался объект JSON")
            question, answer = row.get("question"), row.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise LogitsFormatError(f"{path}:{lineno}: нет непустого question")
            if not isinstance(answer, str) or not answer.strip():
                raise LogitsFormatError(f"{path}:{lineno}: нет непустого answer")
            raw = row.get("logits", row.get("teacher_logits"))
            if raw is None:
                raise LogitsFormatError(f"{path}:{lineno}: нет поля logits")
            if not isinstance(raw, list) or not raw:
                raise LogitsFormatError(f"{path}:{lineno}: logits пусты или не список")
            width = len(raw[0]) if isinstance(raw[0], list) else 0
            if width <= 0:
                raise LogitsFormatError(f"{path}:{lineno}: logits[0] — не непустой список")
            try:
                parsed = parse_teacher_logits(
                    raw, len(raw), vocab_size, check_values=check_values,
                )
            except LogitsFormatError as exc:
                raise LogitsFormatError(f"{path}:{lineno}: {exc}") from exc
            if parsed is None:
                raise LogitsFormatError(
                    f"{path}:{lineno}: логиты не прямоугольные [A, {width}]"
                )
            rows.append({
                "question": question.strip(),
                "answer": answer.strip(),
                "logits": parsed,
            })
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise LogitsFormatError(f"в {path} нет ни одной строки с логитами")
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
