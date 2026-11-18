"""QAT-заготовка EVA32: тернарное дообучение Qwen2.5 на дистилляции.

Модули:

* :mod:`qat.ternary` — квантование, STE и упаковка тритов (чистая математика
  работает без torch);
* :mod:`qat.train` — цикл обучения, чекпоинты, возобновление;
* :mod:`qat.export` — экспорт в ``ternary.pt`` + ``manifest.json``;
* :mod:`qat.config` — YAML-конфиг;
* :mod:`qat.data` — пары «вопрос → ответ» и маска на вопрос.

Зависимости GPU-машины — ``qat/requirements-qat.txt``.
"""
from __future__ import annotations

__version__ = "0.1.0"
