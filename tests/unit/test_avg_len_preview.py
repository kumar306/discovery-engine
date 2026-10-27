"""The avg_len preview: does the length constant of BM25 change the ranking, offline, no index."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval.avg_len_preview import (
    AvgLenPreviewConfig,
    preview_avg_len,
    run_avg_len_preview,
    verdict_of,
)
from eval.dataset import (
    JudgementRow,
    Product,
    QueryRow,
    Selection,
    SelectionConfig,
    write_dataset,
)
from eval.metrics import PairedGain
from tests.support.hash_embedder import HashEmbedder

NOW = datetime(2026, 10, 27, 20, 0, tzinfo=UTC)
K1, B = 1.2, 0.75


def config(tmp_path: Path, **changes: object) -> AvgLenPreviewConfig:
    values: dict[str, object] = {
        "dataset_directory": tmp_path / "set",
        "distractors": 0,
        "k1": K1,
        "b": B,
        "results_root": tmp_path / "results",
        "resamples": 200,
        "clock": lambda: NOW,
    }
    return AvgLenPreviewConfig(**{**values, **changes})  # type: ignore[arg-type]


# ---------------- the verdict rule: the minimum gain, and an interval above 0


def gain(mean: float, low: float, high: float) -> PairedGain:
    return PairedGain(mean=mean, low=low, high=high, sd=0.1)


@pytest.mark.parametrize(
    ("mean", "low", "expected"),
    [
        (0.02, 0.005, "worth a hybrid experiment"),
        (0.01, 0.0001, "worth a hybrid experiment"),  # exactly the minimum gain counts
        (0.0099, 0.005, "keep the baseline"),  # a hair under the minimum gain
        (0.02, 0.0, "keep the baseline"),  # an interval that touches 0 is not above 0
        (0.02, -0.001, "keep the baseline"),
        (-0.02, -0.03, "keep the baseline"),  # a worse value never wins
    ],
)
def test_the_verdict_needs_the_minimum_gain_and_an_interval_above_zero(
    mean: float, low: float, expected: str
) -> None:
    assert verdict_of(gain(mean, low, mean + 0.01), min_gain=0.01) == expected


# ---------------- the ranking, hand computed

# one query "shoe" and two chunks of two products with the same single query word:
#   A has 20 tokens and the word twice, B has 2 tokens and the word once
#   score = idf * n (k1 + 1) / (n + k1 (1 - b + b len / avg)), the idf is the same for both
#   each cell is the weight of the word in the chunk, the bigger one ranks first
#   avg 1:    A 4.4 / (2 + 1.2 * 15.25)   = 0.2167    B 2.2 / (1 + 1.2 * 1.75)    = 0.7097
#             B first
#   avg 11:   A 4.4 / (2 + 1.2 * 1.6136)  = 1.1179    B 2.2 / (1 + 1.2 * 0.3864)  = 1.5030
#             B first
#   avg 1000: A 4.4 / (2 + 1.2 * 0.265)   = 1.8981    B 2.2 / (1 + 1.2 * 0.2515)  = 1.6902
#             A first
# A is the exact answer, B is irrelevant: A first scores nDCG 1.0, B first 1 / log2(3) = 0.6309
TWO_CHUNKS = {
    "A-0": ["shoe", "shoe"] + [f"filler{n}" for n in range(18)],
    "B-0": ["shoe", "other"],
}
PRODUCT_OF = {"A-0": "A", "B-0": "B"}
QUERY = {"q1": ["shoe"]}
JUDGED = {"q1": {"A": "E", "B": "I"}}


def test_the_order_of_two_products_flips_with_the_length_constant_and_so_does_ndcg(
    tmp_path: Path,
) -> None:
    rows = preview_avg_len(
        config(tmp_path, fixed_values=(1.0, 1000.0), baseline=1000.0),
        TWO_CHUNKS,
        PRODUCT_OF,
        QUERY,
        JUDGED,
    )
    by_value = {row.average_length: row for row in rows}
    # the true mean of the two chunks is (20 + 2) / 2 = 11, added to the fixed values
    assert sorted(by_value) == [1.0, 11.0, 1000.0]
    assert by_value[1.0].ndcg_at_10 == pytest.approx(0.6309, abs=1e-4)
    assert by_value[11.0].ndcg_at_10 == pytest.approx(0.6309, abs=1e-4)
    assert by_value[1000.0].ndcg_at_10 == pytest.approx(1.0)


def test_a_product_is_ranked_at_its_best_chunk_and_only_once(tmp_path: Path) -> None:
    # P1 has a weak chunk (the word once) and a strong chunk (three times), P2 one chunk (twice),
    # all the same length, so the order of the chunks is P1-strong, P2, P1-weak
    chunks = {
        "P1-weak": ["shoe", "x1", "x2", "x3"],
        "P1-strong": ["shoe", "shoe", "shoe", "x4"],
        "P2-0": ["shoe", "shoe", "x5", "x6"],
    }
    product_of = {"P1-weak": "P1", "P1-strong": "P1", "P2-0": "P2"}
    # P2 is the only exact answer: with P1 first (its best chunk) nDCG is 1 / log2(3);
    # had P1 been ranked at its WORST chunk, P2 would be first and nDCG would be 1.0
    rows = preview_avg_len(
        config(tmp_path, fixed_values=(256.0,), baseline=256.0),
        chunks,
        product_of,
        QUERY,
        {"q1": {"P1": "I", "P2": "E"}},
    )
    baseline = next(row for row in rows if row.average_length == 256.0)
    assert baseline.ndcg_at_10 == pytest.approx(0.6309, abs=1e-4)


def test_a_query_that_shares_no_word_with_any_chunk_scores_zero_and_is_kept(
    tmp_path: Path,
) -> None:
    rows = preview_avg_len(
        config(tmp_path, fixed_values=(256.0,), baseline=256.0),
        TWO_CHUNKS,
        PRODUCT_OF,
        {"q1": ["shoe"], "q2": ["zzz"]},
        {"q1": {"A": "E", "B": "I"}, "q2": {"A": "E"}},
    )
    assert all(len(row.ndcg_per_query) == 2 for row in rows)
    assert all(0.0 in row.ndcg_per_query for row in rows)


# ---------------- the rows


def test_the_baseline_row_has_no_gain_and_every_other_row_a_paired_interval(
    tmp_path: Path,
) -> None:
    rows = preview_avg_len(
        config(tmp_path, fixed_values=(1.0, 1000.0), baseline=1000.0),
        TWO_CHUNKS,
        PRODUCT_OF,
        QUERY,
        JUDGED,
    )
    baseline = next(row for row in rows if row.average_length == 1000.0)
    assert baseline.gain is None
    assert baseline.verdict == "baseline"
    other = next(row for row in rows if row.average_length == 1.0)
    assert other.gain is not None
    # one query: the gain is the single point 0.6309 - 1.0, below zero
    assert other.gain.mean == pytest.approx(-0.3691, abs=1e-4)
    assert other.verdict == "keep the baseline"


def test_a_value_that_wins_by_the_minimum_with_an_interval_above_zero_is_worth_a_hybrid_experiment(
    tmp_path: Path,
) -> None:
    # ten identical queries, so the interval is the single point +0.3691
    rows = preview_avg_len(
        config(tmp_path, fixed_values=(1.0, 1000.0), baseline=1.0),
        TWO_CHUNKS,
        PRODUCT_OF,
        {f"q{n}": ["shoe"] for n in range(10)},
        {f"q{n}": {"A": "E", "B": "I"} for n in range(10)},
    )
    winner = next(row for row in rows if row.average_length == 1000.0)
    assert winner.verdict == "worth a hybrid experiment"


def test_the_same_input_gives_the_same_rows(tmp_path: Path) -> None:
    arguments = (TWO_CHUNKS, PRODUCT_OF, QUERY, JUDGED)
    cfg = config(tmp_path, fixed_values=(1.0, 1000.0), baseline=1000.0)
    assert preview_avg_len(cfg, *arguments) == preview_avg_len(cfg, *arguments)


def test_the_baseline_must_be_one_of_the_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="baseline"):
        preview_avg_len(
            config(tmp_path, fixed_values=(64.0, 512.0), baseline=256.0),
            TWO_CHUNKS,
            PRODUCT_OF,
            QUERY,
            JUDGED,
        )


def test_a_judged_query_without_tokens_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="q2"):
        preview_avg_len(
            config(tmp_path, fixed_values=(256.0,), baseline=256.0),
            TWO_CHUNKS,
            PRODUCT_OF,
            QUERY,
            {"q1": {"A": "E"}, "q2": {"A": "E"}},
        )


def test_the_length_constant_is_never_zero_or_negative(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="above zero"):
        preview_avg_len(
            config(tmp_path, fixed_values=(0.0, 256.0), baseline=256.0),
            TWO_CHUNKS,
            PRODUCT_OF,
            QUERY,
            JUDGED,
        )


# ---------------- the whole run on a dataset directory


def make_dataset(directory: Path, distractors: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    items = [
        Product("J0", "us", "red running shoes", None, None, None, None),
        Product("J1", "us", "blue running socks", None, None, None, None),
    ]
    items += [
        Product(f"D{n:02d}", "us", f"garden hose number {n}", None, None, None, None)
        for n in range(distractors)
    ]
    selection = Selection(
        (QueryRow(1, "Running   SHOES", "validation"),),
        (JudgementRow(1, "J0", "E"), JudgementRow(1, "J1", "S")),
        tuple(sorted(item.product_id for item in items)),
    )
    write_dataset(directory, selection, items, SelectionConfig(seed=1), 1)


def run(tmp_path: Path, distractors: int, **changes: object) -> Path:
    make_dataset(tmp_path / "set", distractors=20)
    embedder = HashEmbedder()
    return run_avg_len_preview(
        config(tmp_path, distractors=distractors, **changes),
        tokenizer=embedder.tokenizer,
        limits=embedder.limits,
        tokenise_text=lambda text: text.split(),
    )


def test_a_run_writes_one_result_with_the_rows_and_the_per_query_table(tmp_path: Path) -> None:
    folder = run(tmp_path, distractors=5)
    assert folder.parent.name == "g12-avglen-preview"
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    metrics = result["metrics"]
    # 2 judged products and 5 distractors, one chunk each
    assert (metrics["products"], metrics["chunks"], metrics["queries"]) == (7.0, 7.0, 1.0)
    assert metrics["corpus_average_length"] > 0
    assert "avglen_256_ndcg_at_10" in metrics
    # the capitals and the extra spaces of the query were folded, so it found its judged products
    assert metrics["avglen_256_ndcg_at_10"] > 0
    assert "avglen_64_gain_mean" in metrics
    assert "avglen_64_gain_low" in metrics
    assert set(result["raw_files"]) == {"rows.jsonl", "per_query.jsonl"}
    assert result["config"]["k1"] == K1
    assert result["config"]["b"] == B
    assert result["config"]["distractors"] == 5
    rows = (folder / "raw" / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert {json.loads(line)["verdict"] for line in rows} <= {
        "baseline",
        "keep the baseline",
        "worth a hybrid experiment",
    }
    assert b"\r" not in (folder / "raw" / "rows.jsonl").read_bytes()


def test_the_corpus_is_the_judged_products_plus_the_first_distractors_only(tmp_path: Path) -> None:
    none = json.loads((run(tmp_path, 0) / "result.json").read_text(encoding="utf-8"))
    assert none["metrics"]["products"] == 2.0
    other = tmp_path / "second"
    other.mkdir()
    more = json.loads((run(other, 12) / "result.json").read_text(encoding="utf-8"))
    assert more["metrics"]["products"] == 14.0
    # a distractor is a product that no query judged: the judged ones are always there
    assert none["metrics"]["queries"] == more["metrics"]["queries"] == 1.0


def test_more_distractors_than_the_dataset_has_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distractors"):
        run(tmp_path, distractors=500)


def test_the_order_in_which_the_judgements_arrive_does_not_change_the_rows(tmp_path: Path) -> None:
    cfg = config(tmp_path, fixed_values=(1.0, 1000.0), baseline=1000.0)
    tokens = {"q1": ["shoe"], "q2": ["zzz"]}
    forward = {"q1": {"A": "E", "B": "I"}, "q2": {"A": "E"}}
    backward = {"q2": {"A": "E"}, "q1": {"A": "E", "B": "I"}}
    one = preview_avg_len(cfg, TWO_CHUNKS, PRODUCT_OF, tokens, forward)
    two = preview_avg_len(cfg, TWO_CHUNKS, PRODUCT_OF, tokens, backward)
    assert one == two
