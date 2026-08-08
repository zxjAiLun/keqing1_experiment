"""Test-suite env isolation.

用户级 `setx` 可能设置了 ``KEQING_LADDER_DATA_ROOT`` / ``KEQING_LADDER_CONFIG_DIR``
（正式 runtime 使用）。这些全局环境变量会污染依赖相对路径解析的测试
（``resolve_report_dir`` 优先使用 data root），因此每个测试开始前默认清除。

本 fixture 使用显式 ``os.environ`` save/restore，**不**依赖 pytest 内置
``monkeypatch`` fixture：autouse fixture 若把 ``monkeypatch`` 作为参数，
会延长其生命周期，使 rust 扩展测试的 ``_reset_rust_mode`` teardown 在
``monkeypatch`` undo 之前执行，导致 ``cache_clear`` 打在已替换的 lambda 上。
"""

from __future__ import annotations

import os

import pytest

_ENV_KEYS = ("KEQING_LADDER_DATA_ROOT", "KEQING_LADDER_CONFIG_DIR", "KEQING_PARTICIPANT_DATA_ROOT")


@pytest.fixture(autouse=True)
def _isolate_external_ladder_env() -> None:
    original = {key: os.environ.get(key) for key in _ENV_KEYS}
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
