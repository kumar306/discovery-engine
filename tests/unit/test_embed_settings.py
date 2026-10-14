"""Worker settings: sane defaults and environment overrides."""

import os

import pytest
from pydantic import ValidationError

from services.embed.settings import EmbedSettings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # a variable set in the developer's shell must not change what these tests see
    for name in [name for name in os.environ if name.startswith("EMBED_")]:
        monkeypatch.delenv(name)


def test_the_defaults_point_at_the_dev_stack_and_the_placeholder_model() -> None:
    settings = EmbedSettings()
    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.qdrant_collection == "chunks_esci"
    assert settings.dense_dim == 384
    # the bm25 parameters are part of the version, see the step 0 findings
    assert settings.model_version == "bge-small-en-v1.5_bm25-a256_1"


def test_environment_variables_override_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBED_QDRANT_COLLECTION", "chunks_blue")
    monkeypatch.setenv("EMBED_DENSE_DIM", "768")
    settings = EmbedSettings()
    assert (settings.qdrant_collection, settings.dense_dim) == ("chunks_blue", 768)


@pytest.mark.parametrize("name", ["EMBED_DENSE_DIM", "EMBED_QDRANT_TIMEOUT_S"])
def test_sizes_and_timeouts_must_be_positive(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    monkeypatch.setenv(name, "0")
    with pytest.raises(ValidationError):
        EmbedSettings()
