"""Building the dataset from the parquet files, end to end, on tiny local copies of the files.

Two files stand in for the real ones: a train file with queries 1 to 12 and a test file with
queries 101 to 110. Rows are grouped five to a row group, so the planner has several groups to
choose from. Nothing is downloaded.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO

import pytest

from eval.build import build_full_dataset, build_mini_dataset, build_validation_dataset
from eval.dataset import SelectionConfig, verify_dataset
from eval.source import SourceFile
from tests.support.esci_fixture import esci_row, write_esci_parquet

POOL = 40
LABELS = "ESCI"


def rows_for(queries: range) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for query in queries:
        for j in range(4):
            product = f"A{(query * 3 + j * 5) % POOL:03d}"
            rows.append(esci_row(query, product, LABELS[(query + j) % 4]))
    # rows the build must ignore: another language, outside the small version
    rows.append(esci_row(queries.start, "A900", locale="es"))
    rows.append(esci_row(queries.start, "A901", small_version=0))
    return rows


TEST_QUERIES = range(101, 141)
TRAIN = SourceFile("data/train-00000-of-00001-aa.parquet", "train", 1, "a" * 64)
TEST = SourceFile("data/test-00000-of-00001-bb.parquet", "test", 1, "b" * 64)
FILES = [TEST, TRAIN]

CONFIG = SelectionConfig(
    seed=5, target_products=30, validation_share=0.25, mini_queries=2, mini_products=12
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write_esci_parquet(tmp_path / TRAIN.name, rows_for(range(1, 13)), row_group_size=5)
    write_esci_parquet(tmp_path / TEST.name, rows_for(TEST_QUERIES), row_group_size=5)
    return tmp_path


class Opener:
    """Opens the copies on disk, and remembers which files were opened."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.opened: list[str] = []

    def __call__(self, source: SourceFile) -> BinaryIO:
        self.opened.append(source.name)
        return (self.root / source.name).open("rb")


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


# the mini set


def test_the_mini_set_is_built_verified_and_has_the_asked_sizes(root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    manifest = build_mini_dataset(FILES, Opener(root), CONFIG, out, dataset_revision=1)
    verify_dataset(out)
    assert manifest["counts"]["queries"] == CONFIG.mini_queries
    assert manifest["counts"]["test"] == CONFIG.mini_queries
    assert manifest["counts"]["products"] == CONFIG.mini_products


def test_the_mini_set_never_opens_the_train_file(root: Path, tmp_path: Path) -> None:
    opener = Opener(root)
    out = tmp_path / "out"
    out.mkdir()
    build_mini_dataset(FILES, opener, CONFIG, out, dataset_revision=1)
    assert set(opener.opened) == {TEST.name}


def test_every_product_of_the_mini_set_has_its_text_from_the_file(
    root: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    build_mini_dataset(FILES, Opener(root), CONFIG, out, dataset_revision=1)
    import json

    products = [json.loads(line) for line in read_lines(out / "products.jsonl")]
    assert len(products) == CONFIG.mini_products
    for product in products:
        assert product["title"] == f"title {product['product_id']}"
        assert product["description"] == f"description {product['product_id']}"


def test_the_manifest_lists_the_source_files_that_were_used(root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    manifest = build_mini_dataset(FILES, Opener(root), CONFIG, out, dataset_revision=1)
    assert [f["name"] for f in manifest["source_files"]] == [TEST.name]
    assert manifest["source_files"][0]["sha256"] == "b" * 64


def test_building_twice_gives_the_same_files(root: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    a = build_mini_dataset(FILES, Opener(root), CONFIG, first, dataset_revision=1)
    b = build_mini_dataset(FILES, Opener(root), CONFIG, second, dataset_revision=1)
    assert a["files"] == b["files"]


def test_the_mini_text_is_read_from_few_row_groups(root: Path, tmp_path: Path) -> None:
    # 40 test queries x 4 rows + 2 noise rows = 162 rows = 33 row groups of 5
    # the mini set (2 queries and about 4 distractors) must need only a few of them
    import eval.source as source_module

    reads: list[int] = []
    original = source_module.read_product_text_heavy_pass

    def spying(*args: object, **kwargs: object) -> object:
        row_group_numbers = kwargs["row_group_numbers"]
        reads.append(len(row_group_numbers))  # type: ignore[arg-type]
        return original(*args, **kwargs)  # type: ignore[arg-type]

    import eval.build as build_module

    build_module.read_product_text_heavy_pass = spying  # type: ignore[assignment]
    try:
        out = tmp_path / "out"
        out.mkdir()
        build_mini_dataset(FILES, Opener(root), CONFIG, out, dataset_revision=1)
    finally:
        build_module.read_product_text_heavy_pass = original
    assert 0 < sum(reads) <= 5


# the full selection


def test_the_full_set_reads_both_files_and_has_all_three_buckets(
    root: Path, tmp_path: Path
) -> None:
    opener = Opener(root)
    out = tmp_path / "out"
    out.mkdir()
    config = replace(CONFIG, target_products=1000, validation_share=0.3)
    manifest = build_full_dataset(FILES, opener, config, out, dataset_revision=1)
    verify_dataset(out)
    assert set(opener.opened) == {TEST.name, TRAIN.name}
    counts = manifest["counts"]
    assert counts["train"] > 0
    assert counts["validation"] > 0
    assert counts["test"] == len(TEST_QUERIES)
    assert counts["queries"] == len(TEST_QUERIES) + 12
    assert [f["name"] for f in manifest["source_files"]] == [TEST.name, TRAIN.name]


# the validation set: validation queries from the train file, nested, with the asked sizes

# half of the train queries are validation queries, so the 12 queries of the train file hold about 6
VALIDATION_CONFIG = replace(CONFIG, validation_share=0.5, mini_queries=3, mini_products=14)


def ids_of(path: Path, key: str) -> list[object]:
    return [json.loads(line)[key] for line in read_lines(path)]


def build_validation(root: Path, out: Path, **changes: int) -> dict[str, object]:
    out.mkdir(parents=True, exist_ok=True)
    config = replace(VALIDATION_CONFIG, **changes)
    return build_validation_dataset(FILES, Opener(root), config, out, dataset_revision=1)


def test_the_validation_set_is_built_verified_and_holds_only_validation_queries(
    root: Path, tmp_path: Path
) -> None:
    manifest = build_validation(root, tmp_path / "out")
    verify_dataset(tmp_path / "out")
    counts = manifest["counts"]
    assert isinstance(counts, dict)
    assert (counts["queries"], counts["validation"]) == (3, 3)
    assert (counts["test"], counts["train"]) == (0, 0)
    assert counts["products"] == VALIDATION_CONFIG.mini_products


def test_the_validation_set_opens_only_the_train_file_because_the_test_file_cannot_hold_one(
    root: Path, tmp_path: Path
) -> None:
    opener = Opener(root)
    out = tmp_path / "out"
    out.mkdir()
    build_validation_dataset(FILES, opener, VALIDATION_CONFIG, out, dataset_revision=1)
    assert set(opener.opened) == {TRAIN.name}


def test_every_query_of_the_validation_set_is_a_validation_query_of_the_whole_selection(
    root: Path, tmp_path: Path
) -> None:
    # the whole selection of the same config, built the other way, says which queries are validation
    full = tmp_path / "full"
    full.mkdir()
    build_full_dataset(
        FILES, Opener(root), replace(VALIDATION_CONFIG, target_products=1000), full, 1
    )
    validation_in_full = {
        row["query_id"]
        for row in map(json.loads, read_lines(full / "queries.jsonl"))
        if row["split"] == "validation"
    }
    build_validation(root, tmp_path / "out")
    chosen = set(ids_of(tmp_path / "out" / "queries.jsonl", "query_id"))
    assert chosen < validation_in_full
    assert len(validation_in_full) > 3


def test_a_smaller_validation_set_is_inside_a_bigger_one(root: Path, tmp_path: Path) -> None:
    build_validation(root, tmp_path / "small", mini_queries=2, mini_products=14)
    build_validation(root, tmp_path / "big", mini_queries=4, mini_products=20)
    small = set(ids_of(tmp_path / "small" / "queries.jsonl", "query_id"))
    big = set(ids_of(tmp_path / "big" / "queries.jsonl", "query_id"))
    assert small < big
    small_judged = {
        (r["query_id"], r["product_id"])
        for r in map(json.loads, read_lines(tmp_path / "small" / "judgements.jsonl"))
    }
    big_judged = {
        (r["query_id"], r["product_id"])
        for r in map(json.loads, read_lines(tmp_path / "big" / "judgements.jsonl"))
    }
    assert small_judged < big_judged


def test_every_judged_product_of_a_chosen_query_is_in_the_products_file(
    root: Path, tmp_path: Path
) -> None:
    build_validation(root, tmp_path / "out")
    judged = set(ids_of(tmp_path / "out" / "judgements.jsonl", "product_id"))
    products = set(ids_of(tmp_path / "out" / "products.jsonl", "product_id"))
    assert judged < products


def test_the_validation_set_is_the_same_on_every_build(root: Path, tmp_path: Path) -> None:
    a = build_validation(root, tmp_path / "one")
    b = build_validation(root, tmp_path / "two")
    assert a["files"] == b["files"]


def test_the_manifest_of_the_validation_set_records_the_asked_sizes_and_only_the_train_file(
    root: Path, tmp_path: Path
) -> None:
    manifest = build_validation(root, tmp_path / "out", mini_queries=2, mini_products=13)
    config = manifest["config"]
    assert isinstance(config, dict)
    assert (config["mini_queries"], config["mini_products"]) == (2, 13)
    sources = manifest["source_files"]
    assert isinstance(sources, list)
    assert [source["name"] for source in sources] == [TRAIN.name]


def test_asking_for_more_validation_queries_than_there_are_says_how_many_there_are(
    root: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=r"only \d+ validation queries, 99 are needed"):
        build_validation(root, tmp_path / "out", mini_queries=99)


def test_a_validation_set_too_small_for_its_judged_products_is_refused(
    root: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="mini_products"):
        build_validation(root, tmp_path / "out", mini_products=2)


def test_the_whole_validation_bucket_is_available_whatever_the_coverage_target(
    root: Path, tmp_path: Path
) -> None:
    # the mini set picks its queries from a pool that covers a few times its products, which on
    # real data holds a few dozen validation queries; a target of one product would leave one
    # query in such a pool, and the validation set must still be able to choose among all of them
    manifest = build_validation(root, tmp_path / "out", target_products=1)
    counts = manifest["counts"]
    assert isinstance(counts, dict)
    assert counts["validation"] == VALIDATION_CONFIG.mini_queries
