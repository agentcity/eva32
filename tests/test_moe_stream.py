"""Потоковый импорт GGUF -> `.vtm` без `q4.bin` — задача 0174.

Прежний путь собирал `q4.bin` (18,9 ГБ у полной 30B) только для того, чтобы
компилятор прочитал его ещё раз и записал `.vtm`: три копии весов, 56 ГБ диска.
`tools/moe_stream.py` пишет каркас контейнера через
`vtm_ref.write_vtm_stream` и кладёт payload'ы сегментов сразу в секцию данных,
не материализуя `q4.bin` (и не держа тензор в памяти — кусками по
`chunk_bytes`).

Тесты держат инварианты, а не гигабайты:

* **паритет с непоточным путём** — те же байты: таблица тензоров (имена, kind,
  dims, offset'ы), payload каждого сегмента (sha256), секция программы и мета
  (кроме полей, которых у потокового пути нет по построению);
* **блоки из GGUF**, а не из `q4.bin`: payload каждого kind 6/7 сегмента
  контейнера равен сырым байтам тензора GGUF;
* **срез словаря** (`limit_vocab`) совпадает с `moe_import --limit-vocab`;
* **HTTP Range** — тот же результат, что у локального файла: источник не
  влияет на план и раскладку (тест поднимает локальный Range-сервер);
* **память**: чтение идёт кусками не больше `chunk_bytes` (иначе на 19 ГБ
  контейнер не соберётся в 2 ГБ RSS);
* **за собой убирает**: после успеха/ошибки в каталоге нет `.part` и
  `.moe-head-*`; `--keep-temp` их оставляет;
* **проверка на диске** (`verify_container`) ловит подмену байта в payload;
* `--dry-run` не создаёт файл и оценивает размер контейнера ровно так, как его
  запишет писатель.

Запуск:
    python3 -m unittest tests.test_moe_stream -v
"""
from __future__ import annotations

import contextlib
import hashlib
import http.server
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.test_gguf_import import build_tiny_model  # noqa: E402
from tests.test_moe import TINY, build_tiny_moe_model  # noqa: E402
from tools import gguf_import as gi  # noqa: E402
from tools import moe_import as mi  # noqa: E402
from tools import moe_stream as ms  # noqa: E402
from tools import vtm_compile, vtm_ref  # noqa: E402

#: Короткие имена контейнера -> GGUF-имена (раскладка `compile_layer`).
_Q4K_NAMES = {
    "emb.w": "token_embd.weight",
    "lm_head.w": "output.weight",
}
_LAYER_NAMES = {
    "q_proj.w": "attn_q.weight",
    "k_proj.w": "attn_k.weight",
    "v_proj.w": "attn_v.weight",
    "o_proj.w": "attn_output.weight",
    "gate_bank": "ffn_gate_exps.weight",
    "up_bank": "ffn_up_exps.weight",
    "down_bank": "ffn_down_exps.weight",
    "shared_gate.w": "ffn_gate_shexp.weight",
    "shared_up.w": "ffn_up_shexp.weight",
    "shared_down.w": "ffn_down_shexp.weight",
}


def gguf_name_of(segment_name: str) -> str | None:
    """Имя тензора GGUF по имени сегмента контейнера (или None)."""
    if segment_name in _Q4K_NAMES:
        return _Q4K_NAMES[segment_name]
    if not segment_name.startswith("l") or "." not in segment_name:
        return None
    layer_text, tail = segment_name[1:].split(".", 1)
    if not layer_text.isdigit() or tail not in _LAYER_NAMES:
        return None
    return f"blk.{int(layer_text)}.{_LAYER_NAMES[tail]}"


def quiet_import(source: Path, out_dir: Path, **kwargs) -> dict:
    """`moe_import.import_moe` печатает по тензору в stderr — глушим."""
    with contextlib.redirect_stderr(io.StringIO()):
        return mi.import_moe(source, out_dir, **kwargs)


def build_reference(
    source: Path,
    out_dir: Path,
    *,
    limit_vocab: int | None = None,
    container_version: int | str = "auto",
) -> Path:
    """Непоточный путь: `moe_import` -> `load_export` -> `write_vtm`."""
    quiet_import(source, out_dir, limit_vocab=limit_vocab)
    export = vtm_compile.load_export(out_dir / "manifest.json")
    with contextlib.redirect_stderr(io.StringIO()):
        model = vtm_compile.compile_model(
            export.cfg, export.weights, export.meta
        )
    path = out_dir / "ref.vtm"
    vtm_ref.write_vtm(
        path, model.tensors, model.program, model.meta,
        container_version=container_version,
    )
    return path


def data_section_digest(path: Path) -> str:
    """sha256 секции данных контейнера (payload'ы + выравнивание)."""
    index = ms.read_container_index(path)
    end = index.meta_offset if index.meta_len else index.program_offset
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        handle.seek(index.data_start)
        left = end - index.data_start
        while left > 0:
            piece = handle.read(min(1 << 20, left))
            if not piece:
                break
            digest.update(piece)
            left -= len(piece)
    return digest.hexdigest()


class _TmpCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="moe-stream-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def leftovers(self) -> list[str]:
        return sorted(
            path.name for path in self.tmp.iterdir()
            if path.name.endswith(".part") or path.name.startswith(".moe-head-")
        )


class TestStreamParity(_TmpCase):
    """Потоковый контейнер байт-в-байт равен непоточному на срезе."""

    def test_segments_table_and_program_equal(self):
        source = build_tiny_moe_model(self.tmp / "tiny.gguf")
        reference = build_reference(source, self.tmp / "export")
        out = self.tmp / "stream.vtm"
        report = ms.stream_import(source, out, quiet=True)

        comparison = ms.compare_containers(out, reference)
        self.assertTrue(comparison["same"], comparison["mismatches"])
        self.assertEqual(comparison["mismatches"], [])
        self.assertEqual(comparison["meta_differences"], [])
        self.assertEqual(comparison["segments"], report["tensors"])

        # Сильнее, чем sha256 по сегментам: вся секция данных совпадает,
        # включая выравнивание и нулевые payload'ы активаций.
        self.assertEqual(
            data_section_digest(out), data_section_digest(reference)
        )
        left = ms.read_container_index(out)
        right = ms.read_container_index(reference)
        self.assertEqual(left.data_start, right.data_start)
        self.assertEqual(
            [segment.name for segment in left.segments],
            [segment.name for segment in right.segments],
        )
        self.assertEqual(left.program, right.program)
        report_q4 = sum(
            segment.payload_bytes for segment in left.segments
            if segment.kind in (vtm_ref.KIND_Q4K, vtm_ref.KIND_Q6K)
        )
        self.assertEqual(report["q4_bytes"], report_q4)
        manifest = json.loads(
            (self.tmp / "export" / "manifest.json").read_text(encoding="utf-8")
        )
        fp_names = {
            vtm_compile._short_weight_name(  # noqa: SLF001
                *vtm_compile._weight_role(entry["name"])  # noqa: SLF001
            ): entry["length"]
            for entry in manifest["non_ternary"]["tensors"]
        }
        self.assertEqual(report["fp32_bytes"], sum(fp_names.values()))
        self.assertEqual(
            {
                segment.name: segment.payload_bytes
                for segment in left.segments if segment.name in fp_names
            },
            fp_names,
        )

    def test_blocks_are_raw_gguf_bytes(self):
        """Payload kind 6/7 = заголовок сегмента + сырые блоки GGUF."""
        source = build_tiny_moe_model(self.tmp / "tiny.gguf")
        out = self.tmp / "stream.vtm"
        ms.stream_import(source, out, quiet=True)
        gguf = gi.read_gguf(source)
        index = ms.read_container_index(out)
        checked = 0
        with open(out, "rb") as handle:
            for segment in index.segments:
                if segment.kind not in (
                    vtm_ref.KIND_Q4K, vtm_ref.KIND_Q6K
                ):
                    continue
                name = gguf_name_of(segment.name)
                self.assertIsNotNone(name, segment.name)
                info = gguf.tensor(name)
                raw = gguf.raw_tensor(info)
                handle.seek(index.data_start + segment.offset)
                payload = handle.read(segment.payload_bytes)
                self.assertEqual(
                    payload[vtm_ref.Q4K_HEADER_SIZE:], raw, segment.name
                )
                checked += 1
        self.assertGreater(checked, 0)

    def test_limit_vocab_parity(self):
        source = build_tiny_moe_model(self.tmp / "tiny.gguf")
        reference = build_reference(
            source, self.tmp / "export", limit_vocab=64
        )
        out = self.tmp / "stream.vtm"
        ms.stream_import(source, out, limit_vocab=64, quiet=True)
        comparison = ms.compare_containers(out, reference)
        self.assertTrue(comparison["same"], comparison["mismatches"])
        self.assertEqual(
            data_section_digest(out), data_section_digest(reference)
        )

    def test_forced_container_version_3(self):
        source = build_tiny_moe_model(self.tmp / "tiny.gguf")
        reference = build_reference(
            source, self.tmp / "export", container_version=3
        )
        out = self.tmp / "stream.vtm"
        report = ms.stream_import(
            source, out, container_version=3, quiet=True
        )
        self.assertEqual(report["container_version"], 3)
        comparison = ms.compare_containers(out, reference)
        self.assertTrue(comparison["same"], comparison["mismatches"])
        self.assertEqual(
            data_section_digest(out), data_section_digest(reference)
        )


    def test_compare_detects_difference(self):
        """Сверка не вакуумная: подмена байта в эталоне ловится."""
        source = build_tiny_moe_model(self.tmp / "tiny.gguf")
        reference = build_reference(source, self.tmp / "export")
        out = self.tmp / "stream.vtm"
        ms.stream_import(source, out, quiet=True)
        self.assertTrue(ms.compare_containers(out, reference)["same"])
        index = ms.read_container_index(reference)
        segment = next(
            segment for segment in index.segments
            if segment.kind == vtm_ref.KIND_Q4K
        )
        position = index.data_start + segment.offset + vtm_ref.Q4K_HEADER_SIZE
        with open(reference, "r+b") as handle:
            handle.seek(position)
            original = handle.read(1)
            handle.seek(position)
            handle.write(bytes((original[0] ^ 0xFF,)))
        comparison = ms.compare_containers(out, reference)
        self.assertFalse(comparison["same"])
        self.assertTrue(
            any(segment.name in line for line in comparison["mismatches"]),
            comparison["mismatches"],
        )

    def test_dense_gguf_rejected(self):
        """Плотный Qwen2 — не MoE: отказ с текстом, а не «похожий» импорт."""
        dense = build_tiny_model(self.tmp / "dense.gguf")
        out = self.tmp / "stream.vtm"
        with self.assertRaises(ms.StreamImportError) as ctx:
            ms.stream_import(dense, out, quiet=True)
        self.assertIn("moe", str(ctx.exception).lower())
        self.assertFalse(out.exists())
        self.assertEqual(self.leftovers(), [])


class TestStreamReport(_TmpCase):
    def test_metrics_and_disk_budget(self):
        source = build_tiny_moe_model(self.tmp / "tiny.gguf")
        out = self.tmp / "stream.vtm"
        report = ms.stream_import(source, out, quiet=True)
        self.assertEqual(out.stat().st_size, report["container_bytes"])
        self.assertGreater(report["segments"], 0)
        self.assertGreater(report["q4_bytes"], 0)
        self.assertGreater(report["peak_rss_bytes"], 0)
        # Требование задачи: пик диска <= контейнер +10 %. У локального
        # источника временных файлов нет вовсе — пик равен контейнеру.
        self.assertEqual(report["peak_disk_bytes"], report["container_bytes"])
        self.assertLessEqual(
            report["peak_disk_bytes"], report["container_bytes"] * 1.10
        )
        self.assertIsNone(report["temp_path"])
        self.assertEqual(self.leftovers(), [])
        self.assertEqual(report["verify"]["mismatched"], [])
        self.assertEqual(report["verify"]["missing"], [])

    def test_dry_run_writes_nothing_and_estimates_exactly(self):
        source = build_tiny_moe_model(self.tmp / "tiny.gguf")
        out = self.tmp / "stream.vtm"
        estimate = ms.stream_import(source, out, dry_run=True, quiet=True)
        self.assertTrue(estimate["dry_run"])
        self.assertFalse(out.exists())
        self.assertEqual(self.leftovers(), [])
        self.assertEqual(estimate["architecture"]["n_layers"], TINY["layers"])
        actual = ms.stream_import(source, out, quiet=True)
        self.assertEqual(
            estimate["container_bytes"], actual["container_bytes"]
        )
        self.assertEqual(
            estimate["container_version"], actual["container_version"]
        )

    def test_reads_are_chunked(self):
        """Источник читается кусками ≤ chunk_bytes, а не тензором целиком."""
        source = build_tiny_moe_model(self.tmp / "tiny.gguf")

        class Recording(ms.LocalSource):
            def __init__(self, path):
                super().__init__(path)
                self.reads: list[int] = []

            def read(self, start, length):
                self.reads.append(int(length))
                return super().read(start, length)

        recorder = Recording(source)
        out = self.tmp / "stream.vtm"
        ms.stream_import(
            source, out, chunk_bytes=4096, quiet=True,
            source_factory=lambda _spec: recorder,
        )
        self.assertEqual(recorder.size, source.stat().st_size)
        self.assertGreater(len(recorder.reads), 2, "ожидалось несколько кусков")
        self.assertLessEqual(max(recorder.reads), 4096)
        # И ни один кусок не равен тензору целиком (это был бы не поток).
        biggest = max(
            segment.payload_bytes
            for segment in ms.read_container_index(out).segments
        )
        self.assertGreater(biggest, max(recorder.reads))


class TestStreamCleanup(_TmpCase):
    def test_truncated_source_cleans_part(self):
        source = build_tiny_moe_model(self.tmp / "tiny.gguf")
        broken = self.tmp / "broken.gguf"
        broken.write_bytes(source.read_bytes()[: source.stat().st_size // 2])
        out = self.tmp / "stream.vtm"
        with self.assertRaises((ms.StreamImportError, ms.SourceError)):
            ms.stream_import(broken, out, quiet=True)
        self.assertFalse(out.exists())
        self.assertEqual(self.leftovers(), [])

    def test_corrupted_payload_detected(self):
        source = build_tiny_moe_model(self.tmp / "tiny.gguf")
        out = self.tmp / "stream.vtm"
        ms.stream_import(source, out, quiet=True)
        index = ms.read_container_index(out)
        expected = ms.hash_segments(index)
        report = ms.verify_container(out, expected)
        self.assertEqual(report["mismatched"], [])
        self.assertEqual(report["checked"], report["segments"])
        # Портим один байт в payload первого Q4K-сегмента.
        segment = next(
            segment for segment in index.segments
            if segment.kind == vtm_ref.KIND_Q4K
        )
        position = index.data_start + segment.offset + segment.payload_bytes - 1
        with open(out, "r+b") as handle:
            handle.seek(position)
            original = handle.read(1)
            handle.seek(position)
            handle.write(bytes((original[0] ^ 0xFF,)))
        broken = ms.verify_container(out, expected)
        self.assertEqual(broken["mismatched"], [segment.name])


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """Минимальный HTTP-сервер с Range: как HF/CDN, но без сети."""

    file_path: Path

    def do_HEAD(self):  # noqa: N802 - имя требует BaseHTTPRequestHandler
        size = self.file_path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Length", str(size))
        self.end_headers()

    def do_GET(self):  # noqa: N802
        size = self.file_path.stat().st_size
        header = self.headers.get("Range") or ""
        start, end = 0, size - 1
        if header.startswith("bytes="):
            piece = header[len("bytes="):].split(",", 1)[0]
            first, _, last = piece.partition("-")
            if first:
                start = int(first)
            if last:
                end = min(int(last), size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(self.file_path, "rb") as handle:
            handle.seek(start)
            self.wfile.write(handle.read(end - start + 1))

    def log_message(self, *_args):  # pragma: no cover - не нужен в тестах
        return


@contextlib.contextmanager
def serve_range(path: Path):
    handler = type("RangeHandler", (_RangeHandler,), {"file_path": Path(path)})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}/{Path(path).name}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


class TestStreamHttp(_TmpCase):
    def test_http_range_matches_local_file(self):
        source = build_tiny_moe_model(self.tmp / "tiny.gguf")
        local_out = self.tmp / "local.vtm"
        ms.stream_import(source, local_out, quiet=True)
        with serve_range(source) as url:
            http_out = self.tmp / "http.vtm"
            report = ms.stream_import(url, http_out, quiet=True)
        self.assertEqual(report["source"]["kind"], "http")
        self.assertEqual(report["source"]["bytes"], source.stat().st_size)
        comparison = ms.compare_containers(http_out, local_out)
        self.assertTrue(comparison["same"], comparison["mismatches"])
        self.assertEqual(
            data_section_digest(http_out), data_section_digest(local_out)
        )
        self.assertEqual(self.leftovers(), [])

    def test_keep_temp_keeps_header_prefix(self):
        source = build_tiny_moe_model(self.tmp / "tiny.gguf")
        with serve_range(source) as url:
            out = self.tmp / "stream.vtm"
            report = ms.stream_import(url, out, keep_temp=True, quiet=True)
        self.assertIsNotNone(report["temp_path"])
        self.assertTrue(Path(report["temp_path"]).exists())
        self.assertGreater(report["temp_bytes"], 0)
        # Пик диска считается честно: контейнер + оставленный префикс.
        self.assertEqual(
            report["peak_disk_bytes"],
            report["container_bytes"] + report["temp_bytes"],
        )


if __name__ == "__main__":
    unittest.main()
