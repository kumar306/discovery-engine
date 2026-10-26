"""The embedding cache: vectors of chunks on disk, exactly as the model made them."""

from array import array
from pathlib import Path

import pytest

from eval.embedding_cache import EmbeddingCache, embedding_cache_path
from libs.embedding.model import Embedding
from libs.embedding.settings import IndexSettings


def float32(values: list[float]) -> list[float]:
    """The same numbers as the real model returns them: float32, shown as python floats."""
    return array("f", values).tolist()


def embedding(seed: float) -> Embedding:
    return Embedding(
        # float32 values, as bge-small returns them
        dense=float32([seed / 3, seed / 7, -seed / 11, 0.1 + seed]),
        sparse_indices=[5, 2_000_000_000, 5_000_000_000],
        # bm25 returns float64, these are not exact in float32 on purpose
        sparse_values=[seed / 13, 1.0, seed * 1.0000001],
    )


def test_an_embedding_comes_back_exactly_as_it_was_put(tmp_path: Path) -> None:
    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        cache.put_many({"a": embedding(1.0)})
        assert cache.get_many(["a"]) == {"a": embedding(1.0)}


def test_the_dense_vector_is_exact_when_it_is_a_float32_as_the_model_returns_it(
    tmp_path: Path,
) -> None:
    vector = float32([0.1, 1 / 3, -2 / 7, 1e-8, 123456.789])
    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        cache.put_many({"a": Embedding(vector, [], [])})
        assert cache.get_many(["a"])["a"].dense == vector


def test_a_dense_value_that_does_not_survive_float32_is_refused_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    # a later model that returns float64 must not be rounded by the cache without anybody noticing
    good = embedding(1.0)
    bad = Embedding(dense=[0.1, 0.2], sparse_indices=[1], sparse_values=[1.0])
    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        with pytest.raises(ValueError, match="float32") as refused:
            cache.put_many({"good": good, "bad": bad})
        assert "bad" in str(refused.value)
        # the whole batch is refused before the first row is written
        assert cache.count() == 0


def test_a_dense_value_too_large_for_float32_is_refused_too(tmp_path: Path) -> None:
    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        with pytest.raises(ValueError, match="float32"):
            cache.put_many({"a": Embedding(dense=[1e300], sparse_indices=[], sparse_values=[])})
        assert cache.count() == 0


def test_the_sparse_values_stay_exact_at_64_bit_and_the_indices_at_64_bit_too(
    tmp_path: Path,
) -> None:
    sparse = Embedding(
        dense=float32([0.5]),
        sparse_indices=[2**40 + 7, 3],
        sparse_values=[0.1, 1 / 3],  # not exact in float32, must come back unchanged
    )
    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        cache.put_many({"a": sparse})
        again = cache.get_many(["a"])["a"]
    assert again.sparse_values == [0.1, 1 / 3]
    assert again.sparse_indices == [2**40 + 7, 3]


def test_only_the_known_ids_come_back_and_none_for_an_empty_question(tmp_path: Path) -> None:
    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        cache.put_many({"a": embedding(1.0), "b": embedding(2.0)})
        assert set(cache.get_many(["a", "x", "b", "y"])) == {"a", "b"}
        assert cache.get_many([]) == {}


def test_putting_the_same_id_again_keeps_one_entry_with_the_latest_value(tmp_path: Path) -> None:
    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        cache.put_many({"a": embedding(1.0)})
        cache.put_many({"a": embedding(2.0)})
        assert cache.count() == 1
        assert cache.get_many(["a"]) == {"a": embedding(2.0)}


def test_the_entries_are_still_there_when_the_file_is_opened_again(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    with EmbeddingCache(path) as cache:
        cache.put_many({"a": embedding(1.0), "b": embedding(2.0)})
    with EmbeddingCache(path) as again:
        assert again.count() == 2
        assert again.get_many(["b"]) == {"b": embedding(2.0)}


def test_the_directory_of_the_file_is_created(tmp_path: Path) -> None:
    with EmbeddingCache(tmp_path / "deep" / "er" / "cache.sqlite") as cache:
        assert cache.count() == 0


def test_a_text_with_no_terms_has_an_empty_sparse_vector_that_comes_back_empty(
    tmp_path: Path,
) -> None:
    empty = Embedding(dense=float32([0.5, 0.25]), sparse_indices=[], sparse_values=[])
    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        cache.put_many({"a": empty})
        assert cache.get_many(["a"]) == {"a": empty}


def test_two_caches_in_two_files_do_not_see_each_other(tmp_path: Path) -> None:
    with (
        EmbeddingCache(tmp_path / "one.sqlite") as one,
        EmbeddingCache(tmp_path / "two.sqlite") as two,
    ):
        one.put_many({"a": embedding(1.0)})
        assert two.get_many(["a"]) == {}


def test_a_large_question_is_answered_in_pieces_without_losing_an_id(tmp_path: Path) -> None:
    # sqlite limits the number of values in one query (32766 here, 999 on older builds), a build of
    # the distractor ladder asks about tens of thousands of chunks
    with EmbeddingCache(tmp_path / "cache.sqlite") as cache:
        cache.put_many({f"id{n}": Embedding([float(n)], [n], [1.0]) for n in range(40000)})
        found = cache.get_many([f"id{n}" for n in range(40000)])
        assert len(found) == 40000
        assert found["id39999"].dense == [39999.0]


# ---------------- which file: one per model version, and the average length is part of it


def test_the_file_is_named_after_the_model_version() -> None:
    settings = IndexSettings()
    path = embedding_cache_path(settings.model_version)
    assert path == Path("data") / "cache" / f"embeddings-{settings.model_version}.sqlite"


def test_a_changed_bm25_average_length_gives_another_model_version_and_another_file() -> None:
    # the sparse vectors depend on avg_len, so a cache made with 256 must never serve 118
    changed = IndexSettings(bm25_avg_len=118.0)
    assert changed.model_version != IndexSettings().model_version
    assert embedding_cache_path(changed.model_version) != embedding_cache_path(
        IndexSettings().model_version
    )


def test_a_changed_model_or_revision_gives_another_file_too() -> None:
    base = embedding_cache_path(IndexSettings().model_version)
    assert embedding_cache_path(IndexSettings(model_revision=2).model_version) != base
    bigger = IndexSettings(dense_model="BAAI/bge-base-en-v1.5", dense_dim=768)
    assert embedding_cache_path(bigger.model_version) != base
