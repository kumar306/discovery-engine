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


# the model version is built from its parts, so none of them can change without it


def test_the_model_version_is_built_from_the_model_the_bm25_length_and_the_revision() -> None:
    assert EmbedSettings().model_version == "bge-small-en-v1.5_bm25-a256_1"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        pytest.param({"bm25_avg_len": 64.0}, "bge-small-en-v1.5_bm25-a64_1", id="bm25 length"),
        pytest.param({"bm25_avg_len": 64.5}, "bge-small-en-v1.5_bm25-a64.5_1", id="fractional"),
        pytest.param({"model_revision": 2}, "bge-small-en-v1.5_bm25-a256_2", id="revision"),
        pytest.param(
            {"dense_model": "BAAI/bge-base-en-v1.5"}, "bge-base-en-v1.5_bm25-a256_1", id="model"
        ),
    ],
)
def test_changing_any_part_changes_the_version(change: dict[str, object], expected: str) -> None:
    assert EmbedSettings(**change).model_version == expected  # type: ignore[arg-type]


def test_the_version_follows_an_environment_override_of_a_part(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBED_BM25_AVG_LEN", "48")
    assert EmbedSettings().model_version == "bge-small-en-v1.5_bm25-a48_1"


def test_the_machine_settings_have_sane_defaults() -> None:
    settings = EmbedSettings()
    assert (settings.embed_threads, settings.embed_sub_batch) == (2, 16)
    assert settings.embed_timeout_s > 0
