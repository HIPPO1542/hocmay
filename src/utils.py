import argparse
import csv
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from .config import PROJECT_ROOT


class ProjectError(RuntimeError):
    """Lỗi đầu vào hoặc trạng thái dự án có thể xử lý từ CLI."""


def workspace_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT, help="Thư mục chứa data, models và results.")
    return parser


def run_cli(main: Callable[[], None]) -> None:
    # Giữ thông báo tiếng Việt khi PowerShell hoặc công cụ khác chuyển hướng stdout.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        main()
    except (ProjectError, OSError, ValueError) as error:
        raise SystemExit(f"Lỗi: {error}") from error


@contextmanager
def atomic_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            yield handle
            handle.close()
            os.replace(temporary, path)
        finally:
            handle.close()
            temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: dict) -> None:
    with atomic_text(path) as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with atomic_text(path) as handle:
        writer = csv.DictWriter(handle, fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
