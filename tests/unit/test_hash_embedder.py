"""The stub that stands in for the model in the pipeline tests."""

from services.embed.embedder import Embedder
from tests.support.hash_embedder import HashEmbedder, WhitespaceTokenizer


async def test_the_same_text_always_gives_the_same_vectors() -> None:
    stub = HashEmbedder()
    first = await stub.embed(["wireless mouse"], timeout_s=1)
    again = await HashEmbedder().embed(["wireless mouse"], timeout_s=1)
    assert first == again


async def test_different_texts_give_different_vectors_of_the_asked_size() -> None:
    stub = HashEmbedder(dense_dim=6)
    one, two = await stub.embed(["wireless mouse", "desk lamp"], timeout_s=1)
    assert len(one.dense) == len(two.dense) == 6
    assert one.dense != two.dense


async def test_results_keep_the_order_and_every_call_is_recorded() -> None:
    stub = HashEmbedder()
    results = await stub.embed(["a", "b", "a"], timeout_s=1)
    assert results[0] == results[2] != results[1]
    assert stub.calls == [["a", "b", "a"]]


async def test_texts_sharing_a_word_share_a_sparse_term() -> None:
    one, two, three = await HashEmbedder().embed(
        ["wireless mouse", "mouse pad", "desk lamp"], timeout_s=1
    )
    assert set(one.sparse_indices) & set(two.sparse_indices)
    assert not set(one.sparse_indices) & set(three.sparse_indices)
    assert one.sparse_values == [1.0, 1.0]


def test_the_tokenizer_counts_words_by_their_character_spans() -> None:
    assert WhitespaceTokenizer().spans("  wireless  mouse ") == [(2, 10), (12, 17)]


def test_the_stub_fits_what_the_worker_asks_of_an_embedder() -> None:
    embedder: Embedder = HashEmbedder()
    assert embedder.limits.max_tokens == 512
