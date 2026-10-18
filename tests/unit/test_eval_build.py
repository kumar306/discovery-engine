"""Building the dataset from the parquet files, end to end, on tiny local copies of the files.

Two files stand in for the real ones: a train file with queries 1 to 12 and a test file with
queries 101 to 110. Rows are grouped five to a row group, so the planner has several groups to
choose from. Nothing is downloaded.
"""

from dataclasses import replace
from pathlib import Path
from typing import BinaryIO

import pytest

from eval.build import build_full_dataset, build_mini_dataset
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
    original = source_module.read_products

    def spying(*args: object, **kwargs: object) -> object:
        row_group_numbers = kwargs["row_group_numbers"]
        reads.append(len(row_group_numbers))  # type: ignore[arg-type]
        return original(*args, **kwargs)  # type: ignore[arg-type]

    import eval.build as build_module

    build_module.read_products = spying  # type: ignore[assignment]
    try:
        out = tmp_path / "out"
        out.mkdir()
        build_mini_dataset(FILES, Opener(root), CONFIG, out, dataset_revision=1)
    finally:
        build_module.read_products = original
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
