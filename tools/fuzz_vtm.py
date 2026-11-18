#!/usr/bin/env python3
"""Фаззер контейнера `.vtm`: случайные мутации секций + прогон интерпретатора.

Задача 0069. Проверялись только обрезания файла; здесь мутируются байты
заголовка, таблицы тензоров, программы, меты и данных, а результат скармливается
настоящему CLI:

    src/vtm/vtm run <мутация>.vtm --tokens 1 --max-new 1

Ищутся сигналы (SIGSEGV/SIGABRT/...), коды возврата >= 128, таймауты и (если
передан санитайзерный бинарник) отчёты ASan/UBSan. Найденный случай целиком
сохраняется в `fuzz-findings/`: мутированный `.vtm`, JSON с абсолютными
смещениями и sha256, stdout/stderr.

    python3 tools/fuzz_vtm.py --iterations 5000 --seed 1
    python3 tools/fuzz_vtm.py --vtm src/vtm/vtm-asan --iterations 500

По умолчанию берётся `demo/model.vtm`; если его нет, он собирается
`scripts/demo_ternary_model.py --out demo/`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import struct
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER_SIZE = 32
TENSOR_ENTRY_SIZE = 52
INSTR_SIZE = 16
DATA_ALIGN = 32
MAGIC = b"VTM1"

HEADER_FIELDS = {
    "version": (4, 2),
    "flags": (6, 2),
    "n_instr": (8, 4),
    "n_tensors": (12, 4),
    "n_program": (16, 4),
    "program_offset": (20, 4),
    "meta_offset": (24, 4),
    "meta_len": (28, 4),
}

PROGRAM_FIELDS = {
    "opcode": (0, 1),
    "flags": (1, 1),
    "dst": (2, 2),
    "a": (4, 2),
    "b": (6, 2),
    "c": (8, 2),
    "imm": (12, 4),
}

TABLE_FIELDS = {
    "name": (0, 1),
    "kind": (24, 1),
    "ndim": (25, 1),
    "dim0": (26, 4),
    "dim1": (30, 4),
    "dim2": (34, 4),
    "dim3": (38, 4),
    "offset": (42, 4),
    "scale_offset": (46, 4),
    "reserved": (50, 2),
}

# Веса секций: программа и данные — самое интересное, заголовок/таблица —
# структурные границы, мета — самый маленький кусок.
SECTION_WEIGHTS = {
    "header": 18,
    "table": 22,
    "data": 25,
    "program": 30,
    "meta": 5,
}

EDGE_BYTES = [0x00, 0x01, 0x02, 0x03, 0x04, 0x07, 0x0F, 0x10, 0x1F, 0x20,
              0x3F, 0x40, 0x7F, 0x80, 0xFE, 0xFF]
EDGE_U16 = [0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 52, 0x7FFF, 0x8000,
            0xFFFE, 0xFFFF]
EDGE_U32 = [0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 44, 52, 64, 128, 256,
            0x7FFFFFFF, 0x80000000, 0xFFFFFFE0, 0xFFFFFFFE, 0xFFFFFFFF]
EDGE_I32 = [0, 1, -1, 2, -2, 3, -3, 4, 7, 8, 16, 31, 32, 64, 100, -100,
            1000, -1000, 128, 256, 151936, 0x7FFFFFFF, -0x80000000]
# Биты float: +0, -0, 1.0, -1.0, inf, -inf, NaN, денормаль.
EDGE_F32_BITS = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000,
                 0x7F800000, 0xFF800000, 0x7FC00000, 0x7F7FFFFF,
                 0x00000001, 0xFFFFFFFF]


def align_up(value: int, alignment: int = DATA_ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_demo(model_path: str) -> None:
    """Собрать demo/model.vtm, если его ещё нет (как в README задачи)."""
    if os.path.exists(model_path):
        return
    out_dir = os.path.dirname(os.path.abspath(model_path)) or "."
    script = os.path.join(ROOT, "scripts", "demo_ternary_model.py")
    print(f"demo-модели нет: собираю {model_path} из {script}")
    subprocess.run([sys.executable, script, "--out", out_dir], check=True)


def parse_layout(data: bytes) -> dict:
    """Разобрать базовый (валидный) контейнер на секции.

    Раскладка берётся из базового файла; мутации заголовка её уже не меняют,
    поэтому смещения в отчёте всегда абсолютные и воспроизводимые.
    """
    if len(data) < HEADER_SIZE or data[:4] != MAGIC:
        raise ValueError("не VTM1 или файл короче заголовка")
    version, _flags, n_instr, n_tensors, _n_program = struct.unpack_from(
        "<HHIII", data, 4)
    if version != 2:
        raise ValueError(f"ожидали версию 2, в файле {version}")
    table_end = HEADER_SIZE + TENSOR_ENTRY_SIZE * n_tensors
    if table_end > len(data):
        raise ValueError("таблица тензоров не влезает в файл")
    data_start = align_up(table_end)
    if data_start > len(data):
        raise ValueError("начало данных за файлом")

    program_bytes = n_instr * INSTR_SIZE
    if program_bytes > len(data):
        raise ValueError("программа не влезает в файл")
    program_start = len(data) - program_bytes
    program_offset, meta_offset, meta_len = struct.unpack_from("<III", data, 20)
    if program_offset not in (0, program_start):
        # 0 — контейнер без «резерва» (старая раскладка); C-загрузчик берёт
        # программу как file_size - n_instr*16, поэтому это допустимо.
        raise ValueError(
            f"program_offset={program_offset} не совпал с file_size-n_instr*16="
            f"{program_start}")

    meta_valid = (meta_offset != 0 and meta_len != 0 and
                  meta_offset >= data_start and
                  meta_offset + meta_len <= program_start)
    data_end = meta_offset if meta_valid else program_start
    if data_end < data_start:
        raise ValueError("данные кончаются раньше, чем начинаются")

    sections = {
        "header": (0, HEADER_SIZE),
        "table": (HEADER_SIZE, table_end),
        "data": (data_start, data_end),
        "program": (program_start, len(data)),
    }
    if meta_valid:
        sections["meta"] = (meta_offset, meta_offset + meta_len)
    else:
        sections["meta"] = (0, 0)

    # Начала TRT1-сегментов внутри данных: полезно мутировать их заголовки.
    trt1 = []
    pos = data_start
    while True:
        pos = data.find(b"TRT1", pos, data_end)
        if pos < 0:
            break
        if (pos - data_start) % DATA_ALIGN == 0 and pos + 18 <= data_end:
            n_trits, n_scales = struct.unpack_from("<QI", data, pos + 6)
            trt1.append((pos, n_trits, n_scales))
        pos += 1
    return {
        "size": len(data),
        "n_instr": n_instr,
        "n_tensors": n_tensors,
        "sections": sections,
        "trt1": trt1,
    }


class Mutator:
    """Детерминированный генератор мутаций по секциям базового файла."""

    def __init__(self, base: bytes, layout: dict, seed: int):
        self.base = base
        self.size = len(base)
        self.layout = layout
        self.sections = layout["sections"]
        self.rng = random.Random(seed)
        self.n_tensors = layout["n_tensors"]
        self.n_instr = layout["n_instr"]
        self.data_size = (self.sections["data"][1] -
                          self.sections["data"][0])
        self.trt1 = layout["trt1"]
        self.names = list(SECTION_WEIGHTS)
        self.weights = [SECTION_WEIGHTS[name] for name in self.names]

    # --- низкоуровневые помощники -------------------------------------
    def _pick(self, values):
        return self.rng.choice(values)

    def _int_value(self, size: int, current: int) -> int:
        if size == 1:
            pool = EDGE_BYTES + [self.rng.randrange(256)]
        elif size == 2:
            pool = EDGE_U16 + [self.rng.randrange(0x10000), current & 0xFFFF]
        else:
            pool = EDGE_U32 + [
                self.rng.randrange(0x100000000), current & 0xFFFFFFFF,
                self.size, self.size - 1, self.data_size, self.n_tensors,
                self.n_instr,
            ]
        return self._pick(pool) & ((1 << (8 * size)) - 1)

    def _write(self, buf: bytearray, offset: int, size: int, value: int,
               label: str, section: str, journal: list) -> None:
        if offset < 0 or offset + size > self.size:
            return
        old = bytes(buf[offset:offset + size])
        new = int(value).to_bytes(size, "little", signed=False)
        if old == new:
            # Всё равно фиксируем: мутация «тем же значением» неинтересна.
            return
        buf[offset:offset + size] = new
        journal.append({
            "section": section,
            "offset": offset,
            "size": size,
            "old": old.hex(),
            "new": new.hex(),
            "label": label,
        })

    def _section_offset(self, section: str) -> tuple[int, int]:
        start, end = self.sections[section]
        return start, end

    # --- мутации конкретных секций ------------------------------------
    def mutate_header(self, buf: bytearray, journal: list) -> None:
        if self.rng.random() < 0.70:
            name = self._pick(list(HEADER_FIELDS))
            rel, size = HEADER_FIELDS[name]
            current = int.from_bytes(self.base[rel:rel + size], "little")
            value = self._int_value(size, current)
            self._write(buf, rel, size, value, f"header.{name}", "header",
                        journal)
        else:
            rel = self.rng.randrange(HEADER_SIZE)
            value = self._pick(EDGE_BYTES + [self.rng.randrange(256),
                                             self.base[rel] ^ (1 << self.rng.randrange(8))])
            self._write(buf, rel, 1, value, "header.byte", "header", journal)

    def mutate_table(self, buf: bytearray, journal: list) -> None:
        if self.n_tensors <= 0:
            return
        entry = self.rng.randrange(self.n_tensors)
        base_off = HEADER_SIZE + entry * TENSOR_ENTRY_SIZE
        name = self._pick(list(TABLE_FIELDS))
        rel, size = TABLE_FIELDS[name]
        off = base_off + rel
        if name == "name":
            value = self._pick(EDGE_BYTES + [self.rng.randrange(256),
                                             ord("A"), ord("z"), 0])
        elif name == "kind":
            value = self._pick([0, 1, 2, 3, 4, 5, 0x7F, 0xFF])
        elif name == "ndim":
            value = self._pick([0, 1, 2, 3, 4, 5, 0x7F, 0xFF])
        elif name.startswith("dim"):
            current = int.from_bytes(self.base[off:off + 4], "little")
            value = self._pick(EDGE_U32 + [
                current, current + 1, current - 1, 0xFFFFFFFF, 1 << 20,
                1 << 24, 1 << 30,
            ]) & 0xFFFFFFFF
        elif name == "offset":
            choices = [0, 32, 64, 128, 256, 512, self.data_size,
                       self.data_size - 32, self.data_size + 32, 0xFFFFFFFF,
                       0xFFFFFFE0]
            choices += [self.rng.randrange(0, max(1, self.data_size // 32)) * 32]
            value = self._pick(choices) & 0xFFFFFFFF
        elif name == "scale_offset":
            value = self._pick([0, 32, 64, 0xFFFFFFFF, self.data_size,
                                self.data_size - 32]) & 0xFFFFFFFF
        else:
            value = self._int_value(size, 0)
        self._write(buf, off, size, value, f"table[{entry}].{name}", "table",
                    journal)

    def mutate_program(self, buf: bytearray, journal: list) -> None:
        if self.n_instr <= 0:
            return
        instr = self.rng.randrange(self.n_instr)
        base_off = self.sections["program"][0] + instr * INSTR_SIZE
        name = self._pick(list(PROGRAM_FIELDS))
        rel, size = PROGRAM_FIELDS[name]
        off = base_off + rel
        n = self.n_tensors
        if name == "opcode":
            value = self._pick([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                13, 14, 15, 16, 0x7F, 0xFF])
        elif name == "flags":
            value = self._pick([0, 1, 2, 3, 4, 0x7F, 0x80, 0xFF])
        elif name in ("dst", "a", "b", "c"):
            choices = [0, 1, n - 1, n, n + 1, 0xFFFF, 0xFFFE, 0x7FFF, 0x8000,
                       127, 128, 255, 256]
            choices += [self.rng.randrange(0, max(1, n))]
            value = self._pick(choices) & 0xFFFF
        else:  # imm
            value = self._pick(EDGE_I32 + [
                self.rng.randrange(-0x80000000, 0x80000000),
                self.rng.randrange(0, 128),
            ]) & 0xFFFFFFFF
        self._write(buf, off, size, value, f"instr[{instr}].{name}", "program",
                    journal)

    def mutate_data(self, buf: bytearray, journal: list) -> None:
        start, end = self.sections["data"]
        if end <= start:
            return
        if self.trt1 and self.rng.random() < 0.35:
            seg, n_trits, n_scales = self._pick(self.trt1)
            field = self._pick(["magic", "version", "flags", "n_trits",
                                "n_scales", "scale", "trits"])
            if field == "magic":
                rel = self.rng.randrange(4)
                self._write(buf, seg + rel, 1, self._pick(EDGE_BYTES),
                            f"trt1@{seg:#x}.magic", "data", journal)
            elif field == "version":
                self._write(buf, seg + 4, 1, self._pick([0, 1, 2, 0xFF]),
                            f"trt1@{seg:#x}.version", "data", journal)
            elif field == "flags":
                self._write(buf, seg + 5, 1, self._pick([0, 1, 2, 3, 0xFF]),
                            f"trt1@{seg:#x}.flags", "data", journal)
            elif field == "n_trits":
                value = self._pick([0, 1, n_trits - 1, n_trits + 1,
                                    n_trits + 160, n_trits * 2,
                                    0xFFFFFFFFFFFFFFFF])
                self._write(buf, seg + 6, 8, value,
                            f"trt1@{seg:#x}.n_trits", "data", journal)
            elif field == "n_scales":
                value = self._pick([0, 1, n_scales - 1, n_scales + 1,
                                    n_scales * 2, 0xFFFFFFFF])
                self._write(buf, seg + 14, 4, value,
                            f"trt1@{seg:#x}.n_scales", "data", journal)
            elif field == "scale":
                # Масштабы лежат сразу за 18-байтным заголовком TRT1.
                if n_scales > 0:
                    so = seg + 18 + self.rng.randrange(min(n_scales, 8)) * 4
                    if so + 4 <= end:
                        value = self._pick(EDGE_F32_BITS)
                        self._write(buf, so, 4, value,
                                    f"trt1@{seg:#x}.scale", "data", journal)
            else:  # trits
                so = seg + 18 + min(n_scales, 4096) * 4
                if so < end:
                    off = so + self.rng.randrange(end - so)
                    self._write(buf, off, 1, self._pick(EDGE_BYTES),
                                f"trt1@{seg:#x}.trits", "data", journal)
        else:
            off = self.rng.randrange(start, end)
            if self.rng.random() < 0.35 and off + 4 <= end:
                value = self._pick(EDGE_F32_BITS + EDGE_U32)
                self._write(buf, off, 4, value, "data.u32", "data", journal)
            else:
                value = self._pick(EDGE_BYTES + [self.rng.randrange(256),
                                                 self.base[off] ^ (1 << self.rng.randrange(8))])
                self._write(buf, off, 1, value, "data.byte", "data", journal)

    def mutate_meta(self, buf: bytearray, journal: list) -> None:
        start, end = self.sections["meta"]
        if end <= start:
            return
        off = self.rng.randrange(start, end)
        # Ломаем JSON: кавычки, скобки, цифры, запятые, экранирование.
        value = self._pick([ord(c) for c in '{}[]",:0123456789-.eE\\']
                           + EDGE_BYTES + [self.rng.randrange(256)])
        self._write(buf, off, 1, value, "meta.byte", "meta", journal)

    # --- один случай ---------------------------------------------------
    def case(self) -> tuple[bytearray, list]:
        buf = bytearray(self.base)
        journal = []
        roll = self.rng.random()
        count = 1 if roll < 0.72 else (2 if roll < 0.92 else 3)
        used = []
        for _ in range(count):
            # Стараемся не бить одну и ту же секцию дважды подряд — так
            # комбинации (например, программа + мета) разнообразнее.
            for _try in range(4):
                section = self.rng.choices(self.names, self.weights, k=1)[0]
                if not used or section != used[-1]:
                    break
            used.append(section)
            if section == "header":
                self.mutate_header(buf, journal)
            elif section == "table":
                self.mutate_table(buf, journal)
            elif section == "program":
                self.mutate_program(buf, journal)
            elif section == "data":
                self.mutate_data(buf, journal)
            else:
                self.mutate_meta(buf, journal)
        return buf, journal


def classify(returncode: int, stdout: bytes, stderr: bytes,
             timed_out: bool) -> str | None:
    if timed_out:
        return "timeout"
    text = (stderr + stdout).decode("utf-8", "replace")
    if "AddressSanitizer" in text or "runtime error:" in text:
        return "sanitizer"
    if returncode < 0:
        return "signal"
    if returncode >= 128:
        return "exit>=128"
    return None


def signal_name(returncode: int) -> str:
    if returncode >= 0:
        return ""
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"SIG{-returncode}"


class Findings:
    def __init__(self, directory: str, max_findings: int):
        self.directory = directory
        self.max_findings = max_findings
        self.entries = []
        self.signatures = {}
        self.duplicates = 0
        os.makedirs(directory, exist_ok=True)

    def save(self, iteration: int, seed: int, kind: str, returncode: int,
             timed_out: bool, elapsed: float, mutated: bytes,
             journal: list, stdout: bytes, stderr: bytes) -> bool:
        sig = (kind, returncode, stderr[:200].decode("utf-8", "replace"))
        if sig in self.signatures:
            self.duplicates += 1
            self.signatures[sig] += 1
            return False
        self.signatures[sig] = 1
        if len(self.entries) >= self.max_findings:
            return False
        digest = sha256(mutated)
        number = len(self.entries) + 1
        suffix = signal_name(returncode) if kind == "signal" else kind
        name = f"finding-{number:04d}-{suffix}-{iteration:05d}"
        vtm_path = os.path.join(self.directory, name + ".vtm")
        with open(vtm_path, "wb") as handle:
            handle.write(mutated)
        info = {
            "iteration": iteration,
            "seed": seed,
            "kind": kind,
            "returncode": returncode,
            "signal": signal_name(returncode),
            "timed_out": timed_out,
            "elapsed_s": round(elapsed, 6),
            "sha256": digest,
            "file": vtm_path,
            "size": len(mutated),
            "mutations": journal,
        }
        with open(os.path.join(self.directory, name + ".json"), "w",
                  encoding="utf-8") as handle:
            json.dump(info, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(self.directory, name + ".stdout"), "wb") as handle:
            handle.write(stdout)
        with open(os.path.join(self.directory, name + ".stderr"), "wb") as handle:
            handle.write(stderr)
        self.entries.append(info)
        print(f"  НАХОДКА {name}: {kind} rc={returncode} "
              f"sha256={digest[:16]}… (сохранено в {vtm_path})")
        return True

    def write_manifest(self, summary: dict) -> None:
        path = os.path.join(self.directory, "findings.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "findings": self.entries,
                       "duplicates": self.duplicates}, handle,
                      ensure_ascii=False, indent=2)
        print(f"манифест: {path} (находок {len(self.entries)}, "
              f"дублей {self.duplicates})")


def run_one(vtm: str, path: str, timeout: float, env: dict):
    started = time.monotonic()
    try:
        proc = subprocess.run(
            [vtm, "run", path, "--tokens", "1", "--max-new", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
            env=env,
        )
        return proc.returncode, proc.stdout, proc.stderr, False, \
            time.monotonic() - started
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        return -1, stdout, stderr, True, time.monotonic() - started


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.join(ROOT, "demo",
                                                        "model.vtm"))
    parser.add_argument("--vtm", default=os.path.join(ROOT, "src", "vtm",
                                                      "vtm"))
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--findings-dir",
                        default=os.path.join(ROOT, "fuzz-findings"))
    parser.add_argument("--max-findings", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--stop-on-finding", action="store_true")
    args = parser.parse_args(argv)

    if args.iterations < 0:
        parser.error("--iterations должен быть >= 0")
    if not os.path.exists(args.vtm):
        print(f"нет интерпретатора {args.vtm}: сначала `cd src/vtm && make`",
              file=sys.stderr)
        return 2
    ensure_demo(args.model)
    with open(args.model, "rb") as handle:
        base = handle.read()

    try:
        layout = parse_layout(base)
    except ValueError as exc:
        print(f"не разобрал {args.model}: {exc}", file=sys.stderr)
        return 2

    print(f"модель: {args.model} ({layout['size']} байт, "
          f"тензоров {layout['n_tensors']}, инструкций {layout['n_instr']})")
    for name in ("header", "table", "data", "meta", "program"):
        start, end = layout["sections"][name]
        if end > start:
            print(f"  {name:7s} 0x{start:08x}..0x{end:08x} "
                  f"({end - start} байт)")
    if layout["trt1"]:
        print(f"  TRT1-сегментов: {len(layout['trt1'])}")
    print(f"итераций: {args.iterations}, seed: {args.seed}, "
          f"таймаут: {args.timeout:g} с")

    env = dict(os.environ)
    # Фаззер не должен зависеть от чужих VTM_* в окружении вызывающего.
    env.pop("VTM_BACKEND", None)

    # Дымовой прогон: базовый файл обязан исполняться, иначе все «находки»
    # будут про сломанную сборку, а не про мутации.
    fd, tmp_path = tempfile.mkstemp(prefix="fuzz_vtm_", suffix=".vtm")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(base)
        rc, out, err, timed_out, elapsed = run_one(
            args.vtm, tmp_path, args.timeout, env)
        if timed_out or rc != 0:
            print(f"базовая модель не исполняется: rc={rc} "
                  f"timeout={timed_out}\n{err.decode('utf-8', 'replace')}",
                  file=sys.stderr)
            return 2
        print(f"дымовой прогон: ok ({elapsed * 1000:.1f} мс)")

        mutator = Mutator(base, layout, args.seed)
        findings = Findings(args.findings_dir, args.max_findings)
        counts = {"ok": 0, "rejected": 0, "other": 0}
        starting = time.monotonic()
        executed = 0
        for iteration in range(1, args.iterations + 1):
            executed = iteration
            mutated, journal = mutator.case()
            with open(tmp_path, "wb") as handle:
                handle.write(mutated)
            rc, out, err, timed_out, elapsed = run_one(
                args.vtm, tmp_path, args.timeout, env)
            kind = classify(rc, out, err, timed_out)
            if kind is None:
                if rc == 0:
                    counts["ok"] += 1
                elif rc == 1:
                    counts["rejected"] += 1
                else:
                    counts["other"] += 1
            else:
                findings.save(iteration, args.seed, kind, rc, timed_out,
                              elapsed, mutated, journal, out, err)
                if args.stop_on_finding:
                    print("останов по --stop-on-finding")
                    break
            if (args.progress_every > 0 and
                    iteration % args.progress_every == 0):
                spent = time.monotonic() - starting
                print(f"  [{iteration}/{args.iterations}] "
                      f"{iteration / spent:.1f} итер/с, "
                      f"находок {len(findings.entries)}")
        elapsed_total = time.monotonic() - starting
        total = executed
        summary = {
            "model": os.path.abspath(args.model),
            "model_sha256": sha256(base),
            "vtm": os.path.abspath(args.vtm),
            "iterations": total,
            "planned_iterations": args.iterations,
            "stopped_early": total < args.iterations,
            "seed": args.seed,
            "timeout_s": args.timeout,
            "elapsed_s": round(elapsed_total, 3),
            "iterations_per_s": round(total / elapsed_total, 1)
            if elapsed_total else None,
            "ok": counts["ok"],
            "rejected": counts["rejected"],
            "other": counts["other"],
            "signals": sum(1 for e in findings.entries
                           if e["kind"] == "signal"),
            "timeouts": sum(1 for e in findings.entries
                            if e["kind"] == "timeout"),
            "sanitizer": sum(1 for e in findings.entries
                             if e["kind"] == "sanitizer"),
            "findings": len(findings.entries),
            "duplicates": findings.duplicates,
            "sections": {name: layout["sections"][name]
                         for name in ("header", "table", "data", "meta",
                                      "program")},
        }
        findings.write_manifest(summary)
        print(f"итог: {total} итераций за {elapsed_total:.1f} с "
              f"({summary['iterations_per_s']} итер/с)")
        print(f"  rc=0 (исполнилось): {counts['ok']}, "
              f"rc=1 (отвергнуто загрузчиком/рантаймом): "
              f"{counts['rejected']}, прочие коды: {counts['other']}")
        print(f"  сигналов: {summary['signals']}, "
              f"таймаутов: {summary['timeouts']}, "
              f"санитайзер-отчётов: {summary['sanitizer']}, "
              f"уникальных находок: {len(findings.entries)}, "
              f"дублей: {findings.duplicates}")
        return 1 if findings.entries else 0
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
