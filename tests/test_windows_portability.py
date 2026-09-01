import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_lcm.store as store_module
from hermes_lcm.ingest_protection import recover_hermes_persisted_output_with_file_stat
from hermes_lcm.store import MessageStore


@pytest.mark.skipif(os.name != "nt", reason="Windows text-mode newline expansion")
def test_text_mode_crlf_normalization_preserves_bare_carriage_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    directory = tmp_path / "hermes-results"
    directory.mkdir()
    original = "A" + chr(13) + "B" + chr(10) + "C"
    path = directory / "result.txt"
    on_disk = original.replace(chr(10), chr(13) + chr(10))
    path.write_bytes(on_disk.encode("utf-8"))
    lines = ["<persisted-output>"]
    lines += [
        f"This tool result was too large ({len(original)} characters, 0 KB).",
        f"Full output saved to: {path}",
    ]
    lines += ["Preview (first 1 chars):", "A", "...", "</persisted-output>"]
    marker = chr(10).join(lines)

    recovered = recover_hermes_persisted_output_with_file_stat(marker)

    assert recovered is not None
    assert recovered[0] == original


@pytest.mark.skipif(os.name != "nt", reason="Windows text-mode newline expansion")
def test_mixed_crlf_and_bare_lf_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    directory = tmp_path / "hermes-results"
    directory.mkdir()
    original = "A\nB\nC"
    path = directory / "mixed.txt"
    path.write_bytes(b"A\r\nB\nC")
    marker = (
        f"<persisted-output>\nThis tool result was too large ({len(original)} characters, 0 KB).\n"
        f"Full output saved to: {path}\nPreview (first 1 chars):\nA\n...\n</persisted-output>"
    )
    assert recover_hermes_persisted_output_with_file_stat(marker) is None


@pytest.mark.skipif(os.name != "nt", reason="Windows-only CLI behavior")
def test_backfill_cli_reports_posix_only_on_windows():
    script = Path(__file__).resolve().parents[1] / "scripts" / "backfill_externalized_tool_outputs.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
    )
    assert completed.returncode != 0
    assert "POSIX-only" in completed.stderr


def test_batch_ingest_timestamps_remain_strictly_monotonic_with_frozen_clock(
    tmp_path, monkeypatch
):
    store = MessageStore(tmp_path / "lcm.db")
    monkeypatch.setattr(store_module, "time", SimpleNamespace(time=lambda: 1234.0))
    try:
        store_ids = store.append_batch(
            "session-a",
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ],
        )
        rows = store._conn.execute(
            "SELECT timestamp FROM messages ORDER BY store_id"
        ).fetchall()
    finally:
        store.close()

    timestamps = [row[0] for row in rows]
    assert len(store_ids) == 3
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)
