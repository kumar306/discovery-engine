"""Reading the ESCI parquet files without downloading them whole, and planning which parts to read.

The files here are tiny local copies with the real column names (tests/support/esci_fixture.py).
Production opens the same files over the network, the code that reads them is the same.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO

import pytest

from eval.dataset import Pair
from eval.source import (
    SourceFile,
    plan_reads,
    read_pairs,
    read_products,
    source_files,
)
from tests.support.esci_fixture import esci_row, write_esci_parquet

# parse the listing the hugging face api returns


def sibling(name: str, size: int | None = None, sha: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        rfilename=name, size=size, lfs=SimpleNamespace(sha256=sha) if sha else None
    )


def test_the_parquet_files_are_listed_with_size_checksum_and_split_other_files_are_ignored() -> (
    None
):
    listing = [
        sibling(".gitattributes"),
        sibling("README.md"),
        sibling("data/train-00001-of-00002-bb.parquet", 20, "b" * 64),
        sibling("data/test-00000-of-00001-aa.parquet", 10, "a" * 64),
        sibling("data/train-00000-of-00002-cc.parquet", 30, "c" * 64),
    ]
    files = source_files(listing)
    assert [f.name for f in files] == [
        "data/test-00000-of-00001-aa.parquet",
        "data/train-00000-of-00002-cc.parquet",
        "data/train-00001-of-00002-bb.parquet",
    ]
    assert [f.split for f in files] == ["test", "train", "train"]
    assert files[0] == SourceFile("data/test-00000-of-00001-aa.parquet", "test", 10, "a" * 64)


def test_a_parquet_file_without_a_checksum_is_refused() -> None:
    with pytest.raises(ValueError, match="checksum"):
        source_files([sibling("data/test-00000-of-00001-aa.parquet", 10, None)])


def test_a_parquet_file_that_is_neither_train_nor_test_is_refused() -> None:
    with pytest.raises(ValueError, match="train or test"):
        source_files([sibling("data/other-00000-of-00001-aa.parquet", 10, "a" * 64)])


# reading the light columns


def opener(root: Path):  # type: ignore[no-untyped-def]
    opened: list[str] = []

    def open_file(source: SourceFile) -> BinaryIO:
        opened.append(source.name)
        return (root / source.name).open("rb")

    open_file.opened = opened  # type: ignore[attr-defined]
    return open_file


@pytest.fixture
def test_file(tmp_path: Path) -> SourceFile:
    # nine rows in row groups of three: rows 0 to 2 are group 0, rows 3 to 5 group 1, 6 to 8 group 2
    rows = [
        esci_row(1, "P1", "E"),
        esci_row(1, "P2", "S"),
        esci_row(1, "P3", "I", locale="es"),  # another language
        esci_row(2, "P2", "C"),  # P2 is judged again, in group 1
        esci_row(2, "P4", "E", small_version=0),  # outside the small version
        esci_row(2, "P5", "E"),
        esci_row(3, "P1", "S"),  # P1 again, in group 2
        esci_row(3, "P6", "E"),
        esci_row(3, "P7", "I"),
    ]
    write_esci_parquet(tmp_path / "data" / "test-0.parquet", rows, row_group_size=3)
    return SourceFile("data/test-0.parquet", "test", 1, "a" * 64)


def test_only_english_rows_of_the_small_version_become_pairs(
    tmp_path: Path, test_file: SourceFile
) -> None:
    pairs, _ = read_pairs(opener(tmp_path), test_file, locale="us", small_version=1)
    assert [(p.query_id, p.product_id, p.label) for p in pairs] == [
        (1, "P1", "E"),
        (1, "P2", "S"),
        (2, "P2", "C"),
        (2, "P5", "E"),
        (3, "P1", "S"),
        (3, "P6", "E"),
        (3, "P7", "I"),
    ]


def test_a_pair_carries_the_search_text_the_locale_and_the_file_it_came_from(
    tmp_path: Path, test_file: SourceFile
) -> None:
    pairs, _ = read_pairs(opener(tmp_path), test_file, locale="us", small_version=1)
    assert pairs[0] == Pair(
        query_id=1,
        query="query 1",
        product_id="P1",
        locale="us",
        label="E",
        small_version=1,
        dataset_split="test",
    )


def test_it_remembers_in_which_row_groups_each_kept_product_was_found(
    tmp_path: Path, test_file: SourceFile
) -> None:
    _, groups = read_pairs(opener(tmp_path), test_file, locale="us", small_version=1)
    # row group number -> the products with a kept row in it
    assert groups == {0: {"P1", "P2"}, 1: {"P2", "P5"}, 2: {"P1", "P6", "P7"}}


# reading the heavy text columns of chosen row groups only


def test_the_text_is_read_only_for_the_wanted_products_in_the_chosen_row_groups(
    tmp_path: Path, test_file: SourceFile
) -> None:
    products = read_products(
        opener(tmp_path),
        test_file,
        wanted_product_ids={"P1", "P2", "P6"},
        row_group_numbers={0, 2},
        locale="us",
    )
    # P1 is in groups 0 and 2 and appears twice, P2 is also in group 1 but that group is not read
    assert sorted(p.product_id for p in products) == ["P1", "P1", "P2", "P6"]
    first = next(p for p in products if p.product_id == "P6")
    assert first.title == "title P6"
    assert first.description == "description P6"
    assert first.bullet_point == "bullet one P6\nbullet two P6"
    assert (first.brand, first.color, first.locale) == ("brand P6", "black", "us")


def test_missing_text_fields_become_none_and_a_missing_title_becomes_empty(
    tmp_path: Path,
) -> None:
    rows = [
        esci_row(
            1,
            "P1",
            product_title=None,
            product_description=None,
            product_bullet_point=None,
            product_brand=None,
            product_color=None,
        )
    ]
    write_esci_parquet(tmp_path / "data" / "f.parquet", rows, row_group_size=3)
    source = SourceFile("data/f.parquet", "train", 1, "a" * 64)
    (product,) = read_products(
        opener(tmp_path), source, wanted_product_ids={"P1"}, row_group_numbers={0}, locale="us"
    )
    assert (product.title, product.description, product.bullet_point) == ("", None, None)
    assert (product.brand, product.color) == (None, None)


def test_the_product_text_column_of_the_mirror_is_never_read(
    tmp_path: Path, test_file: SourceFile
) -> None:
    # the mirror adds a column whose meaning is unknown, we compose our own text from the fields
    import eval.source as source_module

    assert "product_text" not in source_module.LIGHT_COLUMNS
    assert "product_text" not in source_module.TEXT_COLUMNS


# planning what to read


Where = tuple[str, int]


def groups_with(*products: str) -> set[str]:
    return set(products)


PRODUCT_IDS_BY_ROW_GROUP: dict[Where, set[str]] = {
    ("f", 0): groups_with("a", "b", "c"),
    ("f", 1): groups_with("c", "d"),
    ("f", 2): groups_with("e", "f", "g", "h"),
    ("f", 3): groups_with("i", "j"),
    ("g", 0): groups_with("k", "l", "m"),
}


def test_every_judged_product_gets_one_row_group_to_be_read_from() -> None:
    groups, pool = plan_reads(
        PRODUCT_IDS_BY_ROW_GROUP, must_have_product_ids={"a", "d"}, distractors_needed=0, seed=1
    )
    assert ("f", 0) in groups  # a
    assert ("f", 1) in groups  # d
    assert {"a", "d"} <= pool


def test_a_judged_product_found_in_several_row_groups_uses_only_the_first_of_them() -> None:
    groups, _ = plan_reads(
        PRODUCT_IDS_BY_ROW_GROUP, must_have_product_ids={"c"}, distractors_needed=0, seed=1
    )
    assert groups == {("f", 0)}  # c is in groups 0 and 1, one is enough


def test_more_row_groups_are_added_until_there_are_enough_distractors() -> None:
    groups, pool = plan_reads(
        PRODUCT_IDS_BY_ROW_GROUP, must_have_product_ids={"a"}, distractors_needed=4, seed=1
    )
    assert len(pool - {"a"}) >= 4
    assert ("f", 0) in groups
    # but not every group, only as many as it took
    assert len(groups) < len(PRODUCT_IDS_BY_ROW_GROUP)


def test_the_plan_is_the_same_on_every_run_and_changes_with_the_seed() -> None:
    first = plan_reads(
        PRODUCT_IDS_BY_ROW_GROUP, must_have_product_ids={"a"}, distractors_needed=4, seed=1
    )
    assert (
        plan_reads(
            PRODUCT_IDS_BY_ROW_GROUP, must_have_product_ids={"a"}, distractors_needed=4, seed=1
        )
        == first
    )
    others = {
        frozenset(plan_reads(PRODUCT_IDS_BY_ROW_GROUP, {"a"}, 4, seed)[0]) for seed in range(2, 12)
    }
    assert len(others | {frozenset(first[0])}) > 1


def test_the_pool_is_exactly_the_products_in_the_chosen_row_groups() -> None:
    groups, pool = plan_reads(
        PRODUCT_IDS_BY_ROW_GROUP, must_have_product_ids={"a"}, distractors_needed=4, seed=1
    )
    assert pool == set().union(*(PRODUCT_IDS_BY_ROW_GROUP[where] for where in groups))


def test_not_enough_products_anywhere_is_an_error() -> None:
    with pytest.raises(ValueError, match="distractors"):
        plan_reads(
            PRODUCT_IDS_BY_ROW_GROUP, must_have_product_ids={"a"}, distractors_needed=99, seed=1
        )


def test_a_judged_product_that_is_in_no_row_group_is_an_error() -> None:
    with pytest.raises(ValueError, match="not found"):
        plan_reads(
            PRODUCT_IDS_BY_ROW_GROUP, must_have_product_ids={"zzz"}, distractors_needed=0, seed=1
        )


def test_the_label_words_of_the_dataset_become_the_letters_the_code_uses(tmp_path: Path) -> None:
    rows = [
        esci_row(1, "P1", "E"),
        esci_row(1, "P2", "S"),
        esci_row(1, "P3", "C"),
        esci_row(1, "P4", "I"),
    ]
    write_esci_parquet(tmp_path / "data" / "f.parquet", rows, row_group_size=10)
    source = SourceFile("data/f.parquet", "train", 1, "a" * 64)
    pairs, _ = read_pairs(opener(tmp_path), source, locale="us", small_version=1)
    assert [p.label for p in pairs] == ["E", "S", "C", "I"]


def test_a_label_word_nobody_knows_is_refused_with_the_file_name(tmp_path: Path) -> None:
    rows = [esci_row(1, "P1", "E", esci_label="Maybe")]
    write_esci_parquet(tmp_path / "data" / "f.parquet", rows, row_group_size=10)
    source = SourceFile("data/f.parquet", "train", 1, "a" * 64)
    with pytest.raises(ValueError, match=r"Maybe.*f\.parquet"):
        read_pairs(opener(tmp_path), source, locale="us", small_version=1)


def test_the_text_of_a_product_row_in_another_language_is_not_read(
    tmp_path: Path, test_file: SourceFile
) -> None:
    # P3 has a row in group 0 but it is a spanish row, and P3 has no english row at all
    products = read_products(
        opener(tmp_path), test_file, wanted_product_ids={"P3"}, row_group_numbers={0}, locale="us"
    )
    assert products == []


def test_a_judged_product_already_covered_by_a_chosen_group_adds_no_other_group() -> None:
    # "a" is only in group 2, "b" is in groups 0 and 2, group 2 is chosen for "a" and has "b" too
    locations: dict[Where, set[str]] = {("f", 0): {"b", "z"}, ("f", 2): {"a", "b"}}
    groups, _ = plan_reads(
        locations, must_have_product_ids={"a", "b"}, distractors_needed=0, seed=1
    )
    assert groups == {("f", 2)}


def test_only_products_on_the_usable_list_count_as_distractors() -> None:
    usable = {"a", "k", "l"}
    for seed in range(1, 6):
        groups, pool = plan_reads(
            PRODUCT_IDS_BY_ROW_GROUP,
            must_have_product_ids={"a"},
            distractors_needed=2,
            seed=seed,
            allowed_distractor_product_ids=usable,
        )
        assert ("g", 0) in groups  # the only group with two usable products
        assert len((pool - {"a"}) & usable) >= 2


def test_too_few_usable_products_is_an_error_even_when_other_products_exist() -> None:
    with pytest.raises(ValueError, match="distractors"):
        plan_reads(
            PRODUCT_IDS_BY_ROW_GROUP,
            must_have_product_ids={"a"},
            distractors_needed=5,
            seed=1,
            allowed_distractor_product_ids={"a", "k", "l"},
        )
