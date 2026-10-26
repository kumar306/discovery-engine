import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from eval.dataset import (
    Pair,
    Product,
    SelectionConfig,
    add_distractors_to_mini_selection,
    dedupe_and_check_products,
    pick_mini_queries_of_split,
    pick_queries_and_assign_buckets,
    write_dataset,
)
from eval.source import (
    OpenFile,
    SourceFile,
    Where,
    plan_reads,
    read_judgement_rows_light_pass,
    read_product_text_heavy_pass,
)

# the three ways to build the dataset on disk from the parquet files
#   build_mini_dataset: the small set for development and CI, it needs only the test files
#   build_validation_dataset: the set that decisions are made on (the dense model, avg_len),
#       made of validation queries, which live in the train files, so only those are read
#   build_full_dataset: the large selection (about 100 thousand products), run at will
# both do the same four things: read the light columns, choose with eval/dataset.py, read the
# text of the chosen products from as few row groups as possible, and write the files


def _light_pass_over_files(
    open_file: OpenFile, files: Sequence[SourceFile], config: SelectionConfig
) -> tuple[list[Pair], dict[Where, set[str]]]:
    """The first pass over the files: all kept pairs, and which products are in which row group."""
    kept_pairs: list[Pair] = []
    product_ids_by_row_group: dict[Where, set[str]] = {}
    for source in files:
        pairs_of_file, product_ids_by_row_group_number = read_judgement_rows_light_pass(
            open_file, source, locale=config.locale, small_version=config.small_version
        )
        kept_pairs.extend(pairs_of_file)
        for row_group_number, product_ids in product_ids_by_row_group_number.items():
            product_ids_by_row_group[(source.name, row_group_number)] = product_ids
    return kept_pairs, product_ids_by_row_group


def _text_pass_over_files(
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
            read_product_text_heavy_pass(
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
    kept_pairs, product_ids_by_row_group = _light_pass_over_files(
        open_file, test_split_files, config
    )
    # a pool of queries three times bigger than the mini set needs, the mini queries and the
    # distractors are taken from it
    query_pool = pick_queries_and_assign_buckets(
        kept_pairs, replace(config, target_products=config.mini_products * 3)
    )
    _, mini_judgements = pick_mini_queries_of_split(query_pool, config, "test")
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
    mini_selection = add_distractors_to_mini_selection(
        query_pool, config, product_ids_in_planned_blocks
    )
    product_ids_to_fetch = set(mini_selection.product_ids)
    mini_products = dedupe_and_check_products(
        mini_selection.product_ids,
        _text_pass_over_files(
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


def build_validation_dataset(
    files: Sequence[SourceFile],
    open_file: OpenFile,
    config: SelectionConfig,
    directory: Path,
    dataset_revision: int,
) -> dict[str, Any]:
    """The validation set: config.mini_queries validation queries, their judged products, and
    distractors up to config.mini_products products in all."""
    # a validation query is a train query that a seeded hash put in the validation bucket, so the
    # validation set needs the train files and never opens a test file
    train_files = [source for source in files if source.split == "train"]
    kept_pairs, product_ids_by_row_group = _light_pass_over_files(open_file, train_files, config)
    # EVERY train query is picked: the mini set takes its queries from a pool that only covers
    # three times its products, which would hold a few dozen validation queries, and a set of 500
    # needs the whole bucket (about one train query in ten) to choose from
    every_train_query = pick_queries_and_assign_buckets(
        kept_pairs, replace(config, target_products=sys.maxsize)
    )
    _, validation_judgements = pick_mini_queries_of_split(every_train_query, config, "validation")
    judged_product_ids = {j.product_id for j in validation_judgements}
    # the text of the judged products first, then only as many more row groups as it takes to have
    # enough distractors; distractors are products judged for other train queries, never test ones
    distractors_needed = max(config.mini_products - len(judged_product_ids), 0)
    row_groups_to_download, product_ids_in_planned_blocks = plan_reads(
        product_ids_by_row_group,
        judged_product_ids,
        distractors_needed,
        config.seed,
        allowed_distractor_product_ids=every_train_query.product_ids,
    )
    validation_selection = add_distractors_to_mini_selection(
        every_train_query, config, product_ids_in_planned_blocks, split="validation"
    )
    products = dedupe_and_check_products(
        validation_selection.product_ids,
        _text_pass_over_files(
            open_file,
            train_files,
            row_groups_to_download,
            set(validation_selection.product_ids),
            config.locale,
        ),
    )
    directory.mkdir(parents=True, exist_ok=True)
    sources = [asdict(source) for source in train_files]
    return write_dataset(
        directory, validation_selection, products, config, dataset_revision, sources=sources
    )


def build_full_dataset(
    files: Sequence[SourceFile],
    open_file: OpenFile,
    config: SelectionConfig,
    directory: Path,
    dataset_revision: int,
) -> dict[str, Any]:
    kept_pairs, product_ids_by_row_group = _light_pass_over_files(open_file, files, config)
    selection = pick_queries_and_assign_buckets(kept_pairs, config)
    product_ids_to_fetch = set(selection.product_ids)
    # no distractors to find here, every selected product needs its text
    row_groups_to_download, _ = plan_reads(
        product_ids_by_row_group, product_ids_to_fetch, 0, config.seed
    )
    selection_products = dedupe_and_check_products(
        selection.product_ids,
        _text_pass_over_files(
            open_file, files, row_groups_to_download, product_ids_to_fetch, config.locale
        ),
    )
    directory.mkdir(parents=True, exist_ok=True)
    sources = [asdict(source) for source in files]
    return write_dataset(
        directory, selection, selection_products, config, dataset_revision, sources=sources
    )
