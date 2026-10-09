"""Toolchain smoke test: the interpreter is 3.12 and every layout package imports."""

import importlib
import sys

import pytest

PACKAGES = [
    "services.ingest",
    "services.embed",
    "services.query",
    "services.rag",
    "libs.schemas",
    "libs.resilience",
    "libs.observability",
    "analytics.streaming",
    "analytics.batch",
    "analytics.features",
    "eval",
    "loadtest",
]


def test_python_version() -> None:
    assert sys.version_info[:2] == (3, 12)


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    importlib.import_module(name)
