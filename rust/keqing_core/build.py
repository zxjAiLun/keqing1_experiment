#!/usr/bin/env python3
"""Build a wheel and rewrite it into the desired package layout.

The raw maturin wheel currently produces a top-level ``_native`` package:

    _native/__init__.py
    _native/_native.<platform-extension>

We need the final distribution to expose:

    keqing_core/__init__.py
    keqing_core/_native.<platform-extension>

This script runs maturin, rewrites the wheel contents, and regenerates
``RECORD`` so the output is pip-installable without manual site-packages edits.
"""

from __future__ import annotations

from base64 import urlsafe_b64encode
from csv import writer as csv_writer
from hashlib import sha256
from importlib import util as importlib_util
from io import StringIO
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def _record_line(name: str, data: bytes) -> tuple[str, str, str]:
    digest = urlsafe_b64encode(sha256(data).digest()).rstrip(b"=").decode("ascii")
    return name, f"sha256={digest}", str(len(data))


def _maturin_build_command() -> list[str]:
    if importlib_util.find_spec("maturin") is not None:
        return [sys.executable, "-m", "maturin", "build", "--release"]
    if shutil.which("maturin"):
        return ["maturin", "build", "--release"]
    if shutil.which("uvx"):
        return ["uvx", "--from", "maturin", "maturin", "build", "--release"]
    raise RuntimeError(
        "maturin is not available. Install it in the active environment or provide `uvx`."
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    wheel_dir = base_dir / "target" / "wheels"
    # The Python surface of the wheel mirrors src/keqing_core: every .py file
    # (currently __init__.py and cache_schema.py) is copied in at build time so
    # the installed package never imports from src/ or training/.
    repo_pkg_dir = base_dir.parents[1] / "src" / "keqing_core"
    repo_py_files = sorted(repo_pkg_dir.glob("*.py"))

    subprocess.run(
        _maturin_build_command(),
        check=True,
        cwd=base_dir,
    )

    wheels = sorted(wheel_dir.glob("keqing_core-*.whl"))
    if not wheels:
        raise FileNotFoundError(f"No wheel found under {wheel_dir}")
    wheel_path = max(wheels, key=lambda path: path.stat().st_mtime)
    temp_wheel = wheel_path.with_suffix(".rewritten.whl")
    repo_py_contents = {py.name: py.read_bytes() for py in repo_py_files}

    with zipfile.ZipFile(wheel_path, "r") as zf_in:
        entries = {name: zf_in.read(name) for name in zf_in.namelist()}

    rewritten: dict[str, bytes] = {}
    record_name = None
    for name, data in entries.items():
        if name.endswith(".dist-info/RECORD"):
            record_name = name
            continue
        if name.startswith("_native/"):
            target_name = "keqing_core/" + name[len("_native/") :]
        elif name == "keqing_core/__init__.py":
            target_name = name
            data = repo_py_contents["__init__.py"]
        else:
            target_name = name
        rewritten[target_name] = data

    for py_name, py_data in repo_py_contents.items():
        rewritten[f"keqing_core/{py_name}"] = py_data
    if record_name is None:
        record_name = "keqing_core-0.1.0.dist-info/RECORD"

    rows = []
    for name in sorted(rewritten):
        rows.append(_record_line(name, rewritten[name]))
    rows.append((record_name, "", ""))

    record_buffer = StringIO()
    csv = csv_writer(record_buffer, lineterminator="\n")
    csv.writerows(rows)
    rewritten[record_name] = record_buffer.getvalue().encode("utf-8")

    with zipfile.ZipFile(temp_wheel, "w", zipfile.ZIP_DEFLATED) as zf_out:
        for name in sorted(rewritten):
            zf_out.writestr(name, rewritten[name])

    temp_wheel.replace(wheel_path)
    print(f"Rewrote wheel: {wheel_path}")
    with zipfile.ZipFile(wheel_path, "r") as zf_check:
        for name in sorted(zf_check.namelist()):
            print(f"  {name}")


if __name__ == "__main__":
    main()
