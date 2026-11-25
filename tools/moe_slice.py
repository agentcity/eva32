#!/usr/bin/env python3
"""Срез MoE GGUF по HTTP Range — без скачивания 18,5 ГБ целиком.

Зачем
=====

Qwen3-30B-A3B Q4_K_M — 18,56 ГБ. Диск рабочей машины столько не отдаёт
(свободно ~19 ГиБ, и артефактам импорта нужно ещё ~2 ГБ). А для отладочного
пути «2–4 слоя + словарь + lm_head» хватает ~2 ГБ: слои MoE независимы (у них
нет общих весов), поэтому срез первых N слоёв — это честная модель той же
архитектуры с `block_count = N`.

Утилита делает три вещи:

1. скачивает префикс GGUF (метаданные + таблица тензоров) через HTTP Range;
2. выбирает нужные тензоры: `token_embd`, `output_norm`, `output` и
   `blk.{0..N-1}.*`;
3. пишет **валидный GGUF** с тем же словарём/метаданными, но `block_count = N`
   и только выбранными тензорами. Такой файл читает и `tools/moe_import.py`,
   и llama.cpp/ollama — то есть он же служит независимым эталоном для
   проверки паритета (одни и те же байты весов, другая реализация).

CLI
===

    python3 tools/moe_slice.py --out /tmp/moe/slice4.gguf --layers 4
    python3 tools/moe_slice.py --out /tmp/moe/slice4.gguf --layers 4 --no-output

Печатает размер среза и по тензору — сколько скачано.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gguf_import as G  # noqa: E402

HF_URL = "https://huggingface.co/{repo}/resolve/main/{file}"
DEFAULT_REPO = "Qwen/Qwen3-30B-A3B-GGUF"
DEFAULT_FILE = "Qwen3-30B-A3B-Q4_K_M.gguf"
#: GGUF пишет данные с выравниванием 32 (general.alignment); срез сохраняет его.
ALIGNMENT = 32
#: Префикс для разбора заголовка: метаданные Qwen3 (~6 МБ словаря) + таблица
#: тензоров (579 записей). 16 МБ хватает с запасом; если нет — берём больше.
PREFIX_STEPS = (16 << 20, 64 << 20, 256 << 20)


class SliceError(RuntimeError):
    """Срез не собрался — текст для человека."""


def _fetch(url: str, start: int, end: int, *, retries: int = 3) -> bytes:
    """HTTP Range: байты [start, end] включительно."""
    last: Exception | None = None
    for attempt in range(retries):
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{end}",
                          "User-Agent": "eva32-moe-slice/1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status not in (200, 206):
                    raise SliceError(
                        f"HTTP {response.status} на диапазон {start}-{end}"
                    )
                data = response.read()
            if response.status == 200 and len(data) > end - start + 1:
                # Сервер проигнорировал Range и отдал файл целиком — режем.
                data = data[start:end + 1]
            if len(data) != end - start + 1:
                raise SliceError(
                    f"диапазон {start}-{end}: получено {len(data)} байт"
                )
            return data
        except (urllib.error.URLError, OSError, SliceError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise SliceError(f"не скачался диапазон {start}-{end}: {last}")


def fetch_header(url: str, work: Path) -> tuple[G.GgufFile, bytes]:
    """Скачать префикс, достаточный для метаданных и таблицы тензоров."""
    last_error: Exception | None = None
    for size in PREFIX_STEPS:
        head = work / "head.bin"
        with open(head, "wb") as handle:
            handle.write(_fetch(url, 0, size - 1))
        try:
            gguf = G.read_gguf(head, check_data=False)
            return gguf, head.read_bytes()
        except G.GgufError as exc:
            last_error = exc
            continue
    raise SliceError(f"заголовок GGUF не разобрался: {last_error}")


def metadata_value_offsets(data: bytes) -> dict[str, int]:
    """Смещения значений метаданных в файле: key -> байт начала значения."""
    handle = io.BytesIO(data)
    handle.seek(8)
    n_tensors, n_kv = struct.unpack("<QQ", handle.read(16))
    offsets: dict[str, int] = {}
    lengths: dict[str, int] = {}
    for _ in range(int(n_kv)):
        key = G._read_string(handle, "метаданные")
        raw = handle.read(4)
        if len(raw) != 4:
            raise SliceError("метаданные обрезаны на типе значения")
        (value_type,) = struct.unpack("<I", raw)
        offsets[key] = handle.tell()
        G._read_metadata_value(handle, value_type, key, lengths)
    return offsets


def build_slice(
    url: str,
    out: str | Path,
    *,
    layers: int,
    keep_output: bool = True,
    keep_embed: bool = True,
    workers: int = 8,
    quiet: bool = False,
) -> dict[str, Any]:
    """Скачать и собрать GGUF-срез первых `layers` слоёв."""
    work = Path(out).resolve().parent
    work.mkdir(parents=True, exist_ok=True)
    if not 0 < layers <= 4096:
        raise SliceError(f"--layers={layers} вне разумных пределов")
    gguf, header = fetch_header(url, work)
    offsets = metadata_value_offsets(header)
    arch = gguf.metadata.get("general.architecture")
    key = f"{arch}.block_count"
    if key not in offsets:
        raise SliceError(f"в метаданных нет {key!r}")
    total_layers = int(gguf.metadata[key])
    if layers > total_layers:
        raise SliceError(f"--layers={layers} > слоёв в модели ({total_layers})")

    def wanted(name: str) -> bool:
        if name == "token_embd.weight":
            return keep_embed
        if name == "output.weight":
            return keep_output
        if name == "output_norm.weight":
            return True
        layer, _tail = G._split_layer_name(name)
        return layer is not None and layer < layers

    selected = [info for info in gguf.tensors if wanted(info.name)]
    if not selected:
        raise SliceError("не выбрано ни одного тензора")
    if len(selected) == len(gguf.tensors):
        # Срез «всей модели»: пересобирать незачем, но и вреда нет.
        pass

    # --- метаданные: копия байт с пропатченным block_count -------------------
    meta_start = 24
    table_start = _tensor_table_start(header)
    meta_bytes = bytearray(header[meta_start:table_start])
    patch_at = offsets[key] - meta_start
    if patch_at < 0 or patch_at + 4 > len(meta_bytes):
        raise SliceError("block_count вне скопированных метаданных")
    struct.pack_into("<I", meta_bytes, patch_at, layers)

    # --- таблица тензоров: пересчитываем смещения ---------------------------
    table = bytearray()
    data_offset = 0
    placed: list[tuple[G.TensorInfo, int]] = []
    for info in selected:
        data_offset = _align_up(data_offset, ALIGNMENT)
        start = data_offset            # смещение берётся ДО прибавления размера
        placed.append((info, start))
        data_offset += info.n_bytes
        name_bytes = info.name.encode("utf-8")
        table += struct.pack("<Q", len(name_bytes)) + name_bytes
        table += struct.pack("<I", len(info.dims))
        table += struct.pack(f"<{len(info.dims)}Q", *info.dims)
        table += struct.pack("<IQ", info.type_id, start)

    head = bytearray()
    head += b"GGUF" + struct.pack("<I", gguf.version)
    head += struct.pack("<QQ", len(selected), len(gguf.metadata))
    head += meta_bytes + table
    head += bytes((-len(head)) % ALIGNMENT)
    data_start = len(head)

    out_path = Path(out)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with open(tmp, "wb") as handle:
        handle.write(head)
        handle.truncate(data_start + data_offset)

    fd = os.open(tmp, os.O_WRONLY)
    try:
        total = data_start + data_offset
        done = [0]
        started = time.time()

        def fetch_one(job: tuple[G.TensorInfo, int]) -> None:
            info, offset = job
            start = gguf.data_offset + info.offset
            end = start + info.n_bytes - 1
            data = _fetch(url, start, end)
            os.pwrite(fd, data, data_start + offset)
            done[0] += len(data)
            if not quiet:
                print(
                    f"  {info.name:44s} {info.n_bytes / 1e6:8.1f} МБ  "
                    f"({done[0] / (time.time() - started) / 1e6:.1f} МБ/с)",
                    file=sys.stderr, flush=True,
                )

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(fetch_one, placed))
    except BaseException:
        os.close(fd)
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    os.close(fd)

    # Проверка ДО публикации: таблица должна сходиться с данными. Ошибка
    # здесь означает, что срез битый, и лучше не выдавать его за готовый.
    try:
        check = G.read_gguf(tmp)
    except G.GgufError as exc:
        raise SliceError(f"срез не прошёл проверку: {exc}") from exc
    os.replace(tmp, out_path)

    if int(check.metadata[key]) != layers:
        raise SliceError("block_count в срезе не совпал с --layers")
    summary = {
        "path": str(out_path),
        "bytes": total,
        "layers": layers,
        "tensors": len(selected),
        "skipped": len(gguf.tensors) - len(selected),
        "total_layers": total_layers,
        "meta_bytes": len(meta_bytes),
        "data_bytes": data_offset,
    }
    return summary


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _tensor_table_start(data: bytes) -> int:
    """Где кончаются метаданные и начинается таблица тензоров."""
    handle = io.BytesIO(data)
    handle.seek(8)
    _n_tensors, n_kv = struct.unpack("<QQ", handle.read(16))
    lengths: dict[str, int] = {}
    for _ in range(int(n_kv)):
        key = G._read_string(handle, "метаданные")
        raw = handle.read(4)
        (value_type,) = struct.unpack("<I", raw)
        G._read_metadata_value(handle, value_type, key, lengths)
    return handle.tell()


def read_vocab(path: str | Path) -> dict[str, Any]:
    """Прочитать словарь и типы токенов из метаданных GGUF.

    Нужно для паритета с ollama: её logprobs приходят байтами токена, а
    логиты VTM индексируются id. Байты -> id даёт обратная карта, собранная
    здесь (byte-level BPE Qwen — тот же алфавит GPT-2, см. `byte_decoder`).
    """
    data = Path(path).read_bytes()
    handle = io.BytesIO(data)
    handle.seek(8)
    _n_tensors, n_kv = struct.unpack("<QQ", handle.read(16))
    tokens: list[str] | None = None
    types: list[int] | None = None
    for _ in range(int(n_kv)):
        key = G._read_string(handle, "метаданные")
        raw = handle.read(4)
        if len(raw) != 4:
            raise SliceError("метаданные обрезаны на типе значения")
        (value_type,) = struct.unpack("<I", raw)
        if value_type == 9 and key in ("tokenizer.ggml.tokens",
                                       "tokenizer.ggml.token_type"):
            raw = handle.read(12)
            elem_type, count = struct.unpack("<IQ", raw)
            if elem_type == 8:
                items = [G._read_string(handle, key) for _ in range(int(count))]
                if key.endswith("tokens"):
                    tokens = items
            elif elem_type in (4, 5):
                fmt = "<I" if elem_type == 4 else "<i"
                size = 4 * int(count)
                payload = handle.read(size)
                if len(payload) != size:
                    raise SliceError(f"{key}: массив обрезан")
                if key.endswith("token_type"):
                    types = list(struct.unpack(f"<{int(count)}{fmt[1:]}",
                                               payload))
            else:
                raise SliceError(f"{key}: неизвестный тип элемента {elem_type}")
            continue
        G._read_metadata_value(handle, value_type, key, {})
    if not tokens:
        raise SliceError("в GGUF нет tokenizer.ggml.tokens")
    return {"tokens": tokens, "token_type": types or []}


def byte_decoder() -> dict[str, int]:
    """Обратная карта byte-level BPE (GPT-2/Qwen): символ -> байт."""
    printable = (list(range(ord("!"), ord("~") + 1))
                 + list(range(ord("¡"), ord("¬") + 1))
                 + list(range(ord("®"), ord("ÿ") + 1)))
    table = {chr(code): code for code in printable}
    extra = 0
    for code in range(256):
        if code not in printable:
            table[chr(256 + extra)] = code
            extra += 1
    return table


def token_bytes(token: str, table: dict[str, int] | None = None) -> bytes:
    """Байты токена словаря (byte-level BPE) — как их отдаёт ollama."""
    if table is None:
        table = byte_decoder()
    out = bytearray()
    for char in token:
        code = table.get(char)
        if code is None:
            out.extend(char.encode("utf-8"))
        else:
            out.append(code)
    return bytes(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moe_slice.py",
        description="Срез MoE GGUF (первые N слоёв) по HTTP Range",
    )
    parser.add_argument("--out", default=None,
                        help="куда писать срез (не нужен для --vocab-out)")
    parser.add_argument("--layers", type=int, default=4,
                        help="сколько первых слоёв взять (по умолчанию 4)")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--url", default=None, help="готовый URL GGUF")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-output", action="store_true",
                        help="не брать output.weight (lm_head)")
    parser.add_argument("--no-embed", action="store_true",
                        help="не брать token_embd.weight")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gguf", default=None,
                        help="готовый GGUF на диске (для --vocab-out)")
    parser.add_argument("--vocab-out", default=None,
                        help="записать словарь токенов GGUF в JSON "
                             "(нужен для паритета с ollama по logprobs)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.vocab_out and not args.out:
        print("нужен --out (или --vocab-out с --gguf)", file=sys.stderr)
        return 2
    if args.vocab_out:
        if not args.gguf:
            print("--vocab-out требует --gguf (откуда читать словарь)",
                  file=sys.stderr)
            return 2
        vocab = read_vocab(args.gguf)
        Path(args.vocab_out).write_text(
            json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
        )
        print(f"словарь: {len(vocab['tokens'])} токенов -> {args.vocab_out}")
        return 0
    url = args.url or HF_URL.format(repo=args.repo, file=args.file)
    summary = build_slice(
        url, args.out, layers=args.layers, keep_output=not args.no_output,
        keep_embed=not args.no_embed, workers=args.workers, quiet=args.quiet,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
