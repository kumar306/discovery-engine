"""The libraries whose versions decide the content of the index, as one text for a cache key."""

import importlib.metadata

import pytest

from eval.index_libs import INDEX_DEFINING_PACKAGES, index_library_versions


def test_every_listed_package_is_installed() -> None:
    # a typo in the list would make the key blind to that library
    for name in INDEX_DEFINING_PACKAGES:
        assert importlib.metadata.version(name)


def test_the_list_names_what_makes_the_vectors_and_the_sparse_terms() -> None:
    # the dense model runtime and tokenizer, the bm25 stemmer and term hash, the library around them
    assert set(INDEX_DEFINING_PACKAGES) == {
        "fastembed",
        "onnxruntime",
        "tokenizers",
        "py-rust-stemmers",
        "mmh3",
        "numpy",
    }


def test_the_text_holds_each_package_and_its_installed_version_in_a_fixed_order() -> None:
    text = index_library_versions()
    expected = "_".join(
        f"{name}-{importlib.metadata.version(name)}" for name in INDEX_DEFINING_PACKAGES
    )
    assert text == expected
    assert text.startswith("fastembed")


def test_a_changed_version_changes_the_text(monkeypatch: pytest.MonkeyPatch) -> None:
    before = index_library_versions()
    real = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: "9.9.9" if name == "onnxruntime" else real(name)
    )
    after = index_library_versions()
    assert after != before
    assert "onnxruntime-9.9.9" in after


def test_the_text_is_safe_in_a_cache_key() -> None:
    # no spaces or slashes, only what a cache key and a file name can hold
    assert all(character.isalnum() or character in "-._" for character in index_library_versions())
