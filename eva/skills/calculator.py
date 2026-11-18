"""Скилл ``math``: арифметическое выражение без модели и без сети.

Считает **явные** выражения: ``2+2``, ``12 * (3 + 4)``, ``2^10``, ``10 / 4``,
``сколько будет 2,5 * 4``. Разбор — через :mod:`ast`, без ``eval``/``exec``:
разрешены только числа, ``+ - * / ^`` и скобки, степень ограничена.

Проценты («Сколько будет 15% от 800?») намеренно **не** перехватываются: в
золотом наборе это вопросы большой модели, а порог роутера не меняется без
замера (``ai/tasks/0091``). Такие вопросы уходят модели, как и раньше.

Ответ не из каталога, поэтому ``sources`` пустой.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from eva.catalog import normalize
from eva.skills.base import Skill, SkillResult, format_number, make_result

__all__ = ["MathSkill"]

#: Словесные операторы → символы.
_WORD_OPS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\s+умножить\s+на\s+"), " * "),
    (re.compile(r"\s+помножить\s+на\s+"), " * "),
    (re.compile(r"\s+разделить\s+на\s+"), " / "),
    (re.compile(r"\s+поделить\s+на\s+"), " / "),
    (re.compile(r"\s+делить\s+на\s+"), " / "),
    (re.compile(r"\s+в\s+степени\s+"), " ^ "),
    (re.compile(r"\s+плюс\s+"), " + "),
    (re.compile(r"\s+минус\s+"), " - "),
)

#: Кандидат в выражение: цифры, операторы, скобки, точки/запятые, пробелы.
_CANDIDATE_RE = re.compile(r"[0-9()+\-*/^.,\s]+")
_ALLOWED_RE = re.compile(r"^[0-9.+\-*/^()]+$")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_DATE_LIKE_RE = re.compile(r"\d{1,4}[./-]\d{1,2}[./-]\d{1,4}")

#: Проценты — не наша зона (см. docstring): это вопросы большой модели.
_PERCENT_RE = re.compile(r"(?:%|процент\w*)", re.IGNORECASE)

#: Каталожные вопросы не перехватываем: их считает скилл цены.
_GUARD_RE = re.compile(
    r"(?:"
    r"тариф\w*|каталог\w*|доставк\w*|возврат\w*|подписк\w*|скидк\w*"
    r"|стоимост\w*|цен[аыуе]\w*|стоит|руб\w*|₽|офис\w*|телефон\w*"
    r"|поддержк\w*|заказ\w*"
    r")",
    re.IGNORECASE,
)

_MAX_LENGTH = 64
_MAX_RESULT = 1e12
_MAX_DEPTH = 32
_MAX_EXPONENT = 12


class _BadExpression(Exception):
    """Выражение не арифметическое или слишком дорогое для скилла."""


class MathSkill(Skill):
    """Считает простые арифметические выражения."""

    name = "math"

    def match(self, text: str, ctx: dict[str, Any] | None = None) -> bool:
        expression = _expression(text)
        if expression is None:
            return False
        _value, error = _evaluate(expression)
        # Деление на ноль — валидный вопрос: ответим объяснением.
        return error in (None, "zero")

    def run(self, text: str, ctx: dict[str, Any]) -> SkillResult | None:
        expression = _expression(text)
        if expression is None:
            return None
        value, error = _evaluate(expression)
        shown = _display(expression)
        if error == "zero":
            return make_result(
                self.name,
                f"{shown} — деление на ноль, результат не определён.",
                sources=[],
            )
        if error is not None or value is None:
            return None
        rounded = round(value, 6)
        relation = "=" if abs(rounded - value) < 1e-12 else "≈"
        number = format_number(value, 0 if float(value).is_integer() else 6)
        return make_result(self.name, f"{shown} {relation} {number}.", sources=[])


# --------------------------------------------------------------------------- #
# разбор выражения
# --------------------------------------------------------------------------- #
def _expression(text: str) -> str | None:
    """Вытащить выражение из фразы или вернуть ``None``, если его нет."""
    if not text or not text.strip():
        return None
    value = normalize(text)
    if _PERCENT_RE.search(value) or _GUARD_RE.search(value) or _DATE_LIKE_RE.search(value):
        return None
    for pattern, replacement in _WORD_OPS:
        value = pattern.sub(replacement, value)
    value = (
        value.replace("×", "*")
        .replace("÷", "/")
        .replace("·", "*")
        .replace("−", "-")
    )

    best = ""
    for match in _CANDIDATE_RE.finditer(value):
        candidate = match.group(0).strip()
        if len(candidate) > _MAX_LENGTH or not _plausible(candidate):
            continue
        if len(candidate) > len(best):
            best = candidate
    if not best:
        return None
    return _sanitize(best)


def _plausible(candidate: str) -> bool:
    if not any(char.isdigit() for char in candidate):
        return False
    if not any(char in "+-*/^" for char in candidate):
        return False
    return len(_NUMBER_RE.findall(candidate)) >= 2


def _sanitize(candidate: str) -> str | None:
    expression = candidate.strip().rstrip(",.")
    expression = expression.strip()
    if not expression:
        return None
    expression = expression.replace(",", ".").replace(" ", "")
    if not _ALLOWED_RE.fullmatch(expression):
        return None
    if len(_NUMBER_RE.findall(expression)) < 2:
        return None
    return expression


def _display(expression: str) -> str:
    """Выражение для ответа: степень как ``^``, дробная часть через запятую."""
    return expression.replace("**", "^").replace(".", ",")


# --------------------------------------------------------------------------- #
# вычисление без eval
# --------------------------------------------------------------------------- #
def _evaluate(expression: str) -> tuple[float | None, str | None]:
    """Посчитать выражение: ``(значение, None)``, ``(None, "zero")`` или ошибка."""
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError:
        return None, "invalid"
    try:
        return _eval_node(tree, 0), None
    except _BadExpression:
        return None, "invalid"
    except ZeroDivisionError:
        return None, "zero"
    except (OverflowError, ValueError):
        return None, "invalid"


def _eval_node(node: ast.AST, depth: int) -> float:
    if depth > _MAX_DEPTH:
        raise _BadExpression
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, depth + 1)
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise _BadExpression
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand, depth + 1)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        return _apply(node.op, left, right)
    raise _BadExpression


def _apply(operator: ast.operator, left: float, right: float) -> float:
    if isinstance(operator, ast.Add):
        result = left + right
    elif isinstance(operator, ast.Sub):
        result = left - right
    elif isinstance(operator, ast.Mult):
        result = left * right
    elif isinstance(operator, ast.Div):
        if abs(right) < 1e-12:
            raise ZeroDivisionError
        result = left / right
    elif isinstance(operator, ast.Pow):
        if abs(right) > _MAX_EXPONENT or abs(left) > 1_000_000:
            raise _BadExpression
        result = left**right
    else:
        raise _BadExpression
    if result != result or abs(result) > _MAX_RESULT:  # NaN/∞/слишком много
        raise _BadExpression
    return result
