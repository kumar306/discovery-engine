from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from eval.dataset import (
    Pair,
    Product,
    SelectionConfig,
    build_mini,
    choose_mini,
    collect_products,
    select,
    write_dataset,
)
from eval.source import OpenFile, SourceFile, Where, plan_reads, read_pairs, read_products

# the two ways to build the dataset on disk from the parquet files
#   build_mini_dataset: the small set for development and CI, it needs only the test files
#   build_full_dataset: the large selection (about 100 thousand products), run at will
# both do the same four things: read the light columns, choose with eval/dataset.py, read the
# text of the chosen products from as few row groups as possible, and write the files


def _read_light(
    open_file: OpenFile, files: Sequence[SourceFile], config: SelectionConfig
) -> tuple[list[Pair], dict[Where, set[str]]]:
    """The first pass over the files: all kept pairs, and which products are in which row group."""
    kept_pairs: list[Pair] = []
    product_ids_by_row_group: dict[Where, set[str]] = {}
    for source in files:
        pairs_of_file, product_ids_by_row_group_number = read_pairs(
            open_file, source, locale=config.locale, small_version=config.small_version
        )
        kept_pairs.extend(pairs_of_file)
        for row_group_number, product_ids in product_ids_by_row_group_number.items():
            product_ids_by_row_group[(source.name, row_group_number)] = product_ids
    return kept_pairs, product_ids_by_row_group


def _read_text(
    open_file: OpenFile,
    files: Sequence[SourceFile],
    row_groups_to_download: set[Where],
    product_ids_to_fetch: set[str],
    locale: str,
) -> list[Product]:
    """The second pass: the text of the wanted products, from the chosen row groups only."""
    file_by_name = {source.name: source for source in files}
    products: list[Product] = []
    for file_name in sorted({file_name for file_name, _ in row_groups_to_download}):
        row_group_numbers = {number for name, number in row_groups_to_download if name == file_name}
        products.extend(
            read_products(
                open_file,
                file_by_name[file_name],
                wanted_product_ids=product_ids_to_fetch,
                row_group_numbers=row_group_numbers,
                locale=locale,
            )
        )
    return products


def build_mini_dataset(
    files: Sequence[SourceFile],
    open_file: OpenFile,
    config: SelectionConfig,
    directory: Path,
    dataset_revision: int,
) -> dict[str, Any]:
    # the mini set is made of test queries, so only the test files are read at all
    test_split_files = [source for source in files if source.split == "test"]
    kept_pairs, product_ids_by_row_group = _read_light(open_file, test_split_files, config)
    # a pool of queries three times bigger than the mini set needs, the mini queries and the
    # distractors are taken from it
    query_pool = select(kept_pairs, replace(config, target_products=config.mini_products * 3))
    _, mini_judgements = choose_mini(query_pool, config)
    mini_query_product_ids = {j.product_id for j in mini_judgements}
    # decide which row groups to read the text from, the products of the mini queries first,
    # then only as many more as it takes to have enough distractors
    distractors_needed = max(config.mini_products - len(mini_query_product_ids), 0)
    # only products of the query pool can be distractors, the plan counts only those
    row_groups_to_download, product_ids_in_planned_blocks = plan_reads(
        product_ids_by_row_group,
        mini_query_product_ids,
        distractors_needed,
        config.seed,
        allowed_distractor_product_ids=query_pool.product_ids,
    )
    mini_selection = build_mini(query_pool, config, product_ids_in_planned_blocks)
    product_ids_to_fetch = set(mini_selection.product_ids)
    mini_products = collect_products(
        mini_selection.product_ids,
        _read_text(
            open_file,
            test_split_files,
            row_groups_to_download,
            product_ids_to_fetch,
            config.locale,
        ),
    )
    directory.mkdir(parents=True, exist_ok=True)
    sources = [asdict(source) for source in test_split_files]
    return write_dataset(
        directory, mini_selection, mini_products, config, dataset_revision, sources=sources
    )


def build_full_dataset(
    files: Sequence[SourceFile],
    open_file: OpenFile,
    config: SelectionConfig,
    directory: Path,
    dataset_revision: int,
) -> dict[str, Any]:
    kept_pairs, product_ids_by_row_group = _read_light(open_file, files, config)
    selection = select(kept_pairs, config)
    product_ids_to_fetch = set(selection.product_ids)
    # no distractors to find here, every selected product needs its text
    row_groups_to_download, _ = plan_reads(
        product_ids_by_row_group, product_ids_to_fetch, 0, config.seed
    )
    selection_products = collect_products(
        selection.product_ids,
        _read_text(open_file, files, row_groups_to_download, product_ids_to_fetch, config.locale),
    )
    directory.mkdir(parents=True, exist_ok=True)
    sources = [asdict(source) for source in files]
    return write_dataset(
        directory, selection, selection_products, config, dataset_revision, sources=sources
    )
