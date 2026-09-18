"""Execute a notebook in-process with the repository Python 3.13 environment.

This small runner exists because the Python 3.13 environment containing econml does
not include Jupyter. It preserves stream outputs and tracebacks in the notebook JSON.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    args = parser.parse_args()

    notebook_path = args.notebook.resolve()
    workdir = args.cwd.resolve()
    os.chdir(workdir)
    os.environ.setdefault("MPLBACKEND", "Agg")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    execution_count = 0
    started = time.time()

    for cell_index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        execution_count += 1
        cell["execution_count"] = execution_count
        cell["outputs"] = []
        cell_source = "".join(cell.get("source", []))
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        print(f"\n[runner] cell {cell_index} / {len(notebook['cells']) - 1}", flush=True)
        try:
            with contextlib.redirect_stdout(Tee(sys.__stdout__, stdout_buffer)), contextlib.redirect_stderr(
                Tee(sys.__stderr__, stderr_buffer)
            ):
                exec(compile(cell_source, f"{notebook_path.name}:cell-{cell_index}", "exec"), namespace)
        except Exception as exc:
            stderr_buffer.write("".join(traceback.format_exception(exc)))
            if stdout_buffer.getvalue():
                cell["outputs"].append({"name": "stdout", "output_type": "stream", "text": stdout_buffer.getvalue()})
            cell["outputs"].append({
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "output_type": "error",
                "traceback": traceback.format_exception(exc),
            })
            notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"[runner] failed in cell {cell_index}: {exc}", file=sys.stderr, flush=True)
            return 1

        stdout_text = stdout_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()
        if stdout_text:
            cell["outputs"].append({"name": "stdout", "output_type": "stream", "text": stdout_text})
        if stderr_text:
            cell["outputs"].append({"name": "stderr", "output_type": "stream", "text": stderr_text})
        notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    elapsed = time.time() - started
    notebook.setdefault("metadata", {})["execution"] = {
        "runner": "Code/execute_notebook_py313.py",
        "python": sys.version,
        "elapsed_seconds": elapsed,
        "completed": True,
    }
    notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[runner] completed in {elapsed / 60:.1f} minutes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
