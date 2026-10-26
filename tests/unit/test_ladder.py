"""The retrieval ladder: four ways of searching, run on the same queries and scored."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from eval.dataset import JudgementRow, Product, QueryRow, Selection, SelectionConfig, write_dataset
from eval.ladder import RUNGS, LadderConfig, LadderRefused, rank_query, run_ladder
from services.query.encoder import SparseQuery
from services.query.settings import QuerySettings
from tests.unit.test_eval_loader import REVISION, products
from tests.unit.test_query_api import FakeEncoder, FakeSearch, Rig, hit, search

NOW = datetime(2026, 10, 24, 9, 30, tzinfo=UTC)
# the products the tiny dataset holds, the index of the tests holds the same
INDEXED = {"P0", "P1", "P2", "P3"}


class NoTermsEncoder(FakeEncoder):
    """A query whose words bm25 does not know: the sparse vector has no terms."""

    async def encode_sparse(self, query: str) -> SparseQuery:
        self.sparse_queries.append(query)
        return SparseQuery([], [])


def ranked(answer: dict[str, Any], rung: str) -> list[tuple[str, float]]:
    return [(item.product_id, item.score) for item in answer[rung]]


def worked_example_search() -> FakeSearch:
    # the example of the spec: dense [A, B, C], bm25 [B, D, A]
    fake = FakeSearch()
    fake.grouped["dense"] = [hit("A", score=0.9), hit("B", score=0.8), hit("C", score=0.7)]
    fake.grouped["bm25"] = [hit("B", score=12.0), hit("D", score=9.0), hit("A", score=5.0)]
    fake.chunks["dense"] = [hit("A", 0), hit("A", 1), hit("B", 0)]
    fake.chunks["bm25"] = [hit("B", 0), hit("D", 0), hit("A", 1)]
    return fake


# ---------------- one query through the four rungs


async def test_the_single_retriever_rungs_are_the_grouped_products_with_their_own_scores() -> None:
    answer = await rank_query(
        "wireless mouse",
        FakeEncoder(),
        worked_example_search(),
        rrf_k=60,
        depth=100,
        chunk_depth=200,  # type: ignore[arg-type]
    )
    assert ranked(answer, "dense") == [("A", 0.9), ("B", 0.8), ("C", 0.7)]
    assert ranked(answer, "bm25") == [("B", 12.0), ("D", 9.0), ("A", 5.0)]


async def test_the_hybrid_rung_is_reciprocal_rank_fusion_of_the_two_product_lists() -> None:
    answer = await rank_query(
        "wireless mouse",
        FakeEncoder(),
        worked_example_search(),
        rrf_k=60,
        depth=100,
        chunk_depth=200,  # type: ignore[arg-type]
    )
    # A = 1/61 + 1/63, B = 1/62 + 1/61, C = 1/63, D = 1/62
    assert [product for product, _ in ranked(answer, "hybrid")] == ["B", "A", "D", "C"]
    scores = dict(ranked(answer, "hybrid"))
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["C"] == pytest.approx(1 / 63)


async def test_the_hybrid_rung_keeps_only_depth_products() -> None:
    answer = await rank_query(
        "q",
        FakeEncoder(),
        worked_example_search(),
        rrf_k=60,
        depth=2,
        chunk_depth=200,  # type: ignore[arg-type]
    )
    assert [product for product, _ in ranked(answer, "hybrid")] == ["B", "A"]


async def test_the_chunk_rung_fuses_chunks_then_keeps_each_product_at_its_first_place() -> None:
    answer = await rank_query(
        "q",
        FakeEncoder(),
        worked_example_search(),
        rrf_k=60,
        depth=100,
        chunk_depth=200,  # type: ignore[arg-type]
    )
    # chunks: B-0 = 1/63 + 1/61, A-1 = 1/62 + 1/63, A-0 = 1/61, D-0 = 1/62
    # in that order B, A (through A-1), A again (dropped), D
    assert [product for product, _ in ranked(answer, "hybrid_chunk")] == ["B", "A", "D"]
    scores = dict(ranked(answer, "hybrid_chunk"))
    assert scores["B"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["A"] == pytest.approx(1 / 62 + 1 / 63)


async def test_the_chunk_rung_stops_at_depth_products() -> None:
    answer = await rank_query(
        "q",
        FakeEncoder(),
        worked_example_search(),
        rrf_k=60,
        depth=2,
        chunk_depth=200,  # type: ignore[arg-type]
    )
    assert [product for product, _ in ranked(answer, "hybrid_chunk")] == ["B", "A"]


async def test_qdrant_is_asked_four_times_and_the_hybrid_rungs_cost_no_search_of_their_own() -> (
    None
):
    fake = worked_example_search()
    await rank_query("q", FakeEncoder(), fake, rrf_k=60, depth=100, chunk_depth=200)  # type: ignore[arg-type]
    asked = [(shape, using, limit) for shape, using, _, limit in fake.calls]
    assert sorted(asked) == [
        ("chunks", "bm25", 200),
        ("chunks", "dense", 200),
        ("grouped", "bm25", 100),
        ("grouped", "dense", 100),
    ]
    # no filters: the ladder searches the whole catalogue
    assert all(filters.brand_key is None and not filters.doc_ids for _, _, filters, _ in fake.calls)


async def test_a_query_without_bm25_terms_has_empty_bm25_rungs_and_no_bm25_search() -> None:
    fake = worked_example_search()
    answer = await rank_query("zzz", NoTermsEncoder(), fake, rrf_k=60, depth=100, chunk_depth=200)  # type: ignore[arg-type]
    assert answer["bm25"] == []
    # the hybrid rung then is the dense list alone, as the service answers when sparse is degraded
    assert [product for product, _ in ranked(answer, "hybrid")] == ["A", "B", "C"]
    assert all(using != "bm25" for _, using, _, _ in fake.calls)


async def test_the_hybrid_rung_is_what_post_search_answers_for_the_same_retrievers() -> None:
    rig = Rig()
    rig.search.grouped = worked_example_search().grouped
    answer = await rank_query(
        "wireless mouse",
        rig.encoder,
        rig.search,
        rrf_k=rig.settings.rrf_k,
        depth=100,
        chunk_depth=200,  # type: ignore[arg-type]
    )
    response = await search(rig, {"query": "wireless mouse", "k": 100})
    assert [item["doc_id"] for item in response.json()["results"]] == [
        product for product, _ in ranked(answer, "hybrid")
    ]


# ---------------- a whole run


def tiny_dataset(
    directory: Path, *, judged_queries: int = 2, items: list[Product] | None = None
) -> None:
    # two judged queries, judged like the spec example: A exact, B substitute, C irrelevant, D exact
    directory.mkdir(parents=True, exist_ok=True)
    items = items or products(4)
    labels = {"P0": "E", "P1": "S", "P2": "I", "P3": "E"}
    queries = tuple(QueryRow(n, f"query {n}", "test") for n in (1, 2, 3))
    judgements = tuple(
        JudgementRow(n, product_id, label)
        for n in range(1, judged_queries + 1)
        for product_id, label in labels.items()
    )
    selection = Selection(queries, judgements, tuple(sorted(labels)))
    write_dataset(directory, selection, items, SelectionConfig(seed=1), REVISION)
    sha = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    (directory / "loaded.json").write_text(json.dumps({"manifest_sha256": sha}), encoding="utf-8")


def ladder_search() -> FakeSearch:
    # the example again, with the product ids of the dataset: A = P0, B = P1, C = P2, D = P3
    fake = FakeSearch()
    fake.grouped["dense"] = [hit("P0"), hit("P1"), hit("P2")]
    fake.grouped["bm25"] = [hit("P1"), hit("P3"), hit("P0")]
    fake.chunks["dense"] = [hit("P0", 0), hit("P0", 1), hit("P1", 0)]
    fake.chunks["bm25"] = [hit("P1", 0), hit("P3", 0), hit("P0", 1)]
    return fake


def ladder_config(tmp_path: Path, **overrides: Any) -> LadderConfig:
    arguments: dict[str, Any] = {
        "dataset_directory": tmp_path / "mini",
        "settings": QuerySettings(),
        "results_root": tmp_path / "results",
        "resamples": 200,
        "clock": lambda: NOW,
    }
    return LadderConfig(**{**arguments, **overrides})


def metrics_of(folder: Path) -> dict[str, float]:
    loaded: dict[str, Any] = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    metrics: dict[str, float] = loaded["metrics"]
    return metrics


async def test_every_rung_is_scored_against_the_judgements_with_the_hand_computed_values(
    tmp_path: Path,
) -> None:
    tiny_dataset(tmp_path / "mini")
    folder = await run_ladder(ladder_config(tmp_path), FakeEncoder(), ladder_search(), INDEXED)  # type: ignore[arg-type]
    metrics = metrics_of(folder)
    # dense [A, B, C]: (1 + 0.1 / log2(3)) / 1.6809, bm25 [B, D, A] and hybrid [B, A, D, C]:
    # 1.2309 / 1.6809
    assert metrics["dense_ndcg_at_10"] == pytest.approx(0.63244, abs=1e-4)
    assert metrics["bm25_ndcg_at_10"] == pytest.approx(0.73229, abs=1e-4)
    assert metrics["hybrid_ndcg_at_10"] == pytest.approx(0.73229, abs=1e-4)
    assert metrics["queries_scored"] == 2
    # both queries are identical, so the gain of hybrid over dense is the same everywhere and sure
    assert metrics["hybrid_minus_dense_ndcg_mean"] == pytest.approx(0.73229 - 0.63244, abs=1e-4)
    assert metrics["hybrid_minus_dense_ndcg_low"] == pytest.approx(
        metrics["hybrid_minus_dense_ndcg_mean"]
    )
    assert metrics["hybrid_minus_bm25_ndcg_mean"] == pytest.approx(0.0, abs=1e-9)


async def test_a_query_nobody_judged_is_left_out_of_the_scores_and_counted(tmp_path: Path) -> None:
    tiny_dataset(tmp_path / "mini", judged_queries=2)
    folder = await run_ladder(ladder_config(tmp_path), FakeEncoder(), ladder_search(), INDEXED)  # type: ignore[arg-type]
    metrics = metrics_of(folder)
    # three queries in queries.jsonl, the third has no judgements
    assert (metrics["queries_scored"], metrics["queries_unjudged"]) == (2, 1)


async def test_the_run_files_are_trec_lines_one_file_per_rung(tmp_path: Path) -> None:
    tiny_dataset(tmp_path / "mini")
    folder = await run_ladder(ladder_config(tmp_path), FakeEncoder(), ladder_search(), INDEXED)  # type: ignore[arg-type]
    raw = folder / "raw"
    assert sorted(path.name for path in raw.iterdir()) == sorted(
        [f"{rung}.txt" for rung in RUNGS] + ["per_query.jsonl"]
    )
    lines = (raw / "hybrid.txt").read_text(encoding="utf-8").splitlines()
    # query 1, then query 2, products B A D C = P1 P0 P3 P2, ranks from 1, scores falling
    assert [line.split()[:4] for line in lines[:4]] == [
        ["1", "Q0", "P1", "1"],
        ["1", "Q0", "P0", "2"],
        ["1", "Q0", "P3", "3"],
        ["1", "Q0", "P2", "4"],
    ]
    assert all(line.split()[5] == "hybrid" for line in lines)
    scores = [float(line.split()[4]) for line in lines[:4]]
    assert scores == sorted(scores, reverse=True)
    assert lines[4].startswith("2 Q0 P1 1 ")
    assert b"\r" not in (raw / "hybrid.txt").read_bytes()


async def test_the_result_names_the_dataset_the_model_and_the_depths(tmp_path: Path) -> None:
    tiny_dataset(tmp_path / "mini")
    folder = await run_ladder(ladder_config(tmp_path), FakeEncoder(), ladder_search(), INDEXED)  # type: ignore[arg-type]
    loaded = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    assert folder.parent.name == "g6-ladder-mini"
    assert loaded["command"] == "make eval"
    assert loaded["dataset"]["model_version"] == QuerySettings().model_version
    assert loaded["config"]["depth"] == 100
    assert loaded["config"]["chunk_depth"] == 200
    assert loaded["config"]["resamples"] == 200
    assert set(loaded["raw_files"]) == {f"{rung}.txt" for rung in RUNGS} | {"per_query.jsonl"}


async def test_an_index_that_misses_a_product_of_the_dataset_is_refused(tmp_path: Path) -> None:
    tiny_dataset(tmp_path / "mini")
    with pytest.raises(LadderRefused, match=r"1 of the dataset are missing"):
        await run_ladder(
            ladder_config(tmp_path), FakeEncoder(), ladder_search(), {"P0", "P1", "P2"}
        )  # type: ignore[arg-type]
    assert not (tmp_path / "results").exists()


async def test_an_index_that_holds_a_product_the_dataset_does_not_have_is_refused(
    tmp_path: Path,
) -> None:
    tiny_dataset(tmp_path / "mini")
    with pytest.raises(LadderRefused, match=r"2 in the index are not in the dataset"):
        await run_ladder(
            ladder_config(tmp_path),
            FakeEncoder(),  # type: ignore[arg-type]
            ladder_search(),  # type: ignore[arg-type]
            INDEXED | {"X1", "X2"},
        )
    assert not (tmp_path / "results").exists()


async def test_no_loaded_marker_is_needed_because_the_index_itself_is_checked(
    tmp_path: Path,
) -> None:
    tiny_dataset(tmp_path / "mini")
    (tmp_path / "mini" / "loaded.json").unlink()
    folder = await run_ladder(ladder_config(tmp_path), FakeEncoder(), ladder_search(), INDEXED)  # type: ignore[arg-type]
    assert folder.exists()


async def test_the_hybrid_rung_is_compared_with_the_two_single_retrievers_only(
    tmp_path: Path,
) -> None:
    tiny_dataset(tmp_path / "mini")
    folder = await run_ladder(ladder_config(tmp_path), FakeEncoder(), ladder_search(), INDEXED)  # type: ignore[arg-type]
    names = {name for name in metrics_of(folder) if name.startswith("hybrid_minus_")}
    assert names == {
        f"hybrid_minus_{other}_ndcg_{part}"
        for other in ("bm25", "dense")
        for part in ("mean", "low", "high", "sd")
    }


async def test_a_unicode_line_separator_inside_a_product_text_does_not_split_a_line(
    tmp_path: Path,
) -> None:
    # real product texts hold U+2028, which str.splitlines treats as a line break, the dataset
    # files end a record with a newline only, so they must be read line by line from the file
    items = products(4)
    items[0] = replace(items[0], title="wire" + chr(0x2028) + "less mouse")
    tiny_dataset(tmp_path / "mini", items=items)
    folder = await run_ladder(ladder_config(tmp_path), FakeEncoder(), ladder_search(), INDEXED)  # type: ignore[arg-type]
    assert folder.exists()


async def test_the_expected_products_can_be_a_subset_of_the_dataset_for_a_smaller_corpus(
    tmp_path: Path,
) -> None:
    # the distractor ladder searches a corpus of the judged products plus some of the distractors,
    # the index must then hold exactly those, not the whole dataset
    tiny_dataset(tmp_path / "mini")
    folder = await run_ladder(
        ladder_config(tmp_path),
        FakeEncoder(),  # type: ignore[arg-type]
        ladder_search(),  # type: ignore[arg-type]
        {"P0", "P1", "P2"},
        expected_product_ids={"P0", "P1", "P2"},
    )
    assert folder.exists()


async def test_an_index_that_differs_from_the_expected_subset_is_refused(tmp_path: Path) -> None:
    tiny_dataset(tmp_path / "mini")
    with pytest.raises(LadderRefused, match=r"1 in the index are not in the dataset"):
        await run_ladder(
            ladder_config(tmp_path),
            FakeEncoder(),  # type: ignore[arg-type]
            ladder_search(),  # type: ignore[arg-type]
            {"P0", "P1", "P2"},
            expected_product_ids={"P0", "P1"},
        )
