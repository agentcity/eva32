"""Паритет токенов simd vs hybrid через C-ABI (задача 0199).

Симптом 0198: в ``vtm run`` числа гибрида совпадали с simd, а в чате/сервере
(C-ABI ``vtm_api_generate`` с кэшем префикса и ``prefill_batch``) гибрид выдавал
мусор. Причина — не гонка, а шаг строки результата GPU-батча: хост читал
staging с шагом ``nt`` (длина последнего chunk'а), а ядро писало его с шагом
варианта ``N`` (8/16/32). На длинах пачки 8/16/32 (и < 8, где GEMM отказывал)
это совпадало, поэтому прежние ``bench_backend --check`` и CLI-прогоны на
24/40/64 токенах баг не ловили.

Здесь проверяется ровно путь чата: ``bench_backend --api-tokens`` открывает
модель с ``cache_prefix=1``, ``prefill_batch=256``, жадным SAMP и гоняет три
промпта (длины 15/24/45; prefill 14/23/44 — все ломались до 0199), каждый
дважды (второй раз префикс берётся из кэша). Один бэкенд на процесс: режим
Metal глобален, и два контекста в одном процессе поделили бы его.

Тест пропускается, если нет собранного ``bench_backend``, 3B-контейнера или
Metal-устройства. Проверка «гибрид действительно считал на GPU» — по
``VTM_METAL_STATS`` (``gemm N``); иначе сравнение было бы simd с самим собой.

Запуск::

    python3 -m unittest tests.vtm.test_hybrid_api_parity -v
"""

from __future__ import annotations

import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "src" / "vtm" / "bench_backend"
MODEL = ROOT / "runs" / "q4" / "3b-q4k.vtm"

#: Сколько новых токенов просит ``--api-tokens`` (max_new_tokens в опциях).
NEW_TOKENS = 8
#: Три промпта: длины до prefill 14, 23 и 44 (chunk 32 + 12).
PROMPTS = 3


def _run(backend: str) -> tuple[dict[str, str], str]:
    """Прогнать --api-tokens одним бэкендом; вернуть (строки, stderr)."""
    env = dict(os.environ)
    env.pop("VTM_BACKEND", None)          # выбор задаёт --backend явно
    env["VTM_METAL_STATS"] = "1"          # нужен счётчик gemm
    proc = subprocess.run(
        [str(BENCH), "--api-tokens", str(MODEL), "--backend", backend],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"--api-tokens --backend {backend} -> {proc.returncode}\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    lines: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if parts and parts[0] in ("BACKEND", "PROMPT", "TOK"):
            lines[line] = line
    return lines, proc.stderr


class HybridApiParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not BENCH.is_file():
            raise unittest.SkipTest(f"нет {BENCH} (соберите make -C src/vtm test)")
        if not MODEL.is_file():
            raise unittest.SkipTest(f"нет {MODEL}")

    def test_tokens_match_simd(self) -> None:
        simd, _ = _run("simd")
        hybrid, err = _run("hybrid")

        backends = [l.split()[1] for l in hybrid if l.startswith("BACKEND ")]
        gemm = [int(m) for m in re.findall(r"gemm (\d+)", err)]
        if not backends or backends[0] != "hybrid":
            self.skipTest("hybrid собрался в simd (нет Metal/устройства)")
        if not gemm or max(gemm) <= 0:
            self.skipTest("гибрид не сделал ни одного GEMM (Metal недоступен)")

        toks = [l for l in simd if l.startswith("TOK ")]
        toks_h = [l for l in hybrid if l.startswith("TOK ")]
        self.assertEqual(len(toks), PROMPTS * 2, "simd: не все генерации дошли")
        self.assertEqual(len(toks_h), PROMPTS * 2, "hybrid: не все генерации дошли")

        for line in toks:
            self.assertIn(line, hybrid, f"hybrid разошёлся с simd: {line!r}")
        for line in toks_h:
            self.assertIn(line, simd, f"simd разошёлся с hybrid: {line!r}")

        # Повтор на том же контексте идёт через кэш префикса, снятый после
        # GPU-батча: он обязан дать те же токены, что первый прогон.
        first = {l.split()[1]: l for l in toks if l.split()[2] == "0"}
        again = {l.split()[1]: l for l in toks if l.split()[2] == "1"}
        for prompt in first:
            self.assertEqual(
                first[prompt].split()[3:],
                again[prompt].split()[3:],
                f"кэш префикса вернул не тот ответ для промпта {prompt}",
            )

        # Каждая генерация должна вернуть ровно NEW_TOKENS id: если batch
        # оборвался раньше (ошибка GPU/откат), сравнение было бы пустым.
        for line in toks:
            self.assertEqual(len(line.split()) - 3, NEW_TOKENS, line)


if __name__ == "__main__":
    unittest.main()
