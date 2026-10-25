from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, BinaryIO

import pyarrow.parquet as pq

from eval.dataset import DATASET_REPO, DATASET_REVISION, Pair, Product, stable_rank

# how we read the dataset without downloading all of it
# the real dataset is about 2.5 GB in 15 parquet files. a parquet file is stored in column pieces
# and in row groups (here about 1000 rows each), so we can fetch just some columns, and just some
# row groups, over the network. two passes:
#   1. the light pass: a few small columns (ids, the letter) of every row of the files we need,
#      about 4 MB per file, to learn which searches and products exist and where each product is
#   2. the text pass: the big text columns, but only for the row groups that hold the products we
#      decided to keep
# the file with the text of a product is far bigger than the file with its labels, so the second
# pass is the part that must stay small

# the small columns of the first pass
LIGHT_COLUMNS = (
    "query_id",
    "query",
    "product_id",
    "product_locale",
    "esci_label",
    "small_version",
)
# the columns of the second pass, we build our own text from these fields
# the mirror also has a column "product_text" whose meaning is unknown, so it is never read
TEXT_COLUMNS = (
    "product_id",
    "product_locale",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
)

# in this copy of the dataset the judges' label is written as a word, the rest of our code uses
# the first letter of it, so the reader translates, found by running against the real files
LABEL_LETTERS = {"Exact": "E", "Substitute": "S", "Complement": "C", "Irrelevant": "I"}

# a place in the data: (name of the file, number of the row group inside it)
# example: ("data/test-00000-of-00004-d484.parquet", 17) is the 18th group of 1000 rows of that file
Where = tuple[str, int]


@dataclass(frozen=True)
class SourceFile:
    """One parquet file of the dataset, as the hugging face listing describes it."""

    name: str
    # train or test, taken from the file name, these are the dataset's own two splits
    split: str
    size: int
    # the sha256 of the file, given by hugging face for this exact revision
    sha256: str


# a function that opens a source file for reading: it gives a file object, and reading from it
# is what costs network traffic when the file is remote. tests pass a function that opens a copy
# on disk, production passes one that opens the file on hugging face
OpenFile = Callable[[SourceFile], BinaryIO]


def source_files(siblings: Iterable[Any]) -> list[SourceFile]:
    """Turn the file listing of the dataset into SourceFile entries, ignoring non parquet files."""
    files: list[SourceFile] = []
    for entry in siblings:
        if not entry.rfilename.endswith(".parquet"):
            continue
        if entry.lfs is None or not entry.lfs.sha256:
            raise ValueError(f"{entry.rfilename} has no checksum in the listing")
        base = entry.rfilename.rsplit("/", 1)[-1]
        if base.startswith("train-"):
            split = "train"
        elif base.startswith("test-"):
            split = "test"
        else:
            raise ValueError(f"{entry.rfilename} is not a train or test file")
        files.append(SourceFile(entry.rfilename, split, int(entry.size), entry.lfs.sha256))
    return sorted(files, key=lambda f: f.name)


def fetch_source_files() -> list[SourceFile]:
    """Ask hugging face for the files of the pinned revision, with their sizes and checksums."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATASET_REPO, revision=DATASET_REVISION, files_metadata=True)
    return source_files(info.siblings or [])


def hf_opener() -> OpenFile:
    """A function that opens the files of the pinned revision on hugging face, reading on demand."""
    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()

    def open_file(source: SourceFile) -> BinaryIO:
        path = f"datasets/{DATASET_REPO}@{DATASET_REVISION}/{source.name}"
        handle: BinaryIO = filesystem.open(path, "rb")
        return handle

    return open_file


def read_judgement_rows_light_pass(
    open_file: OpenFile, source: SourceFile, *, locale: str, small_version: int
) -> tuple[list[Pair], dict[int, set[str]]]:
    """The first pass over one file: its kept rows as Pair, and where its products are.

    Returns two things:
      the pairs, one per kept row (english, small version), with dataset_split set from the file
      product_ids_by_row_group_number: row group number -> the product ids that have a kept row
      in that row group
    """
    with open_file(source) as handle:
        parquet = pq.ParquetFile(handle)
        # the whole file, but only the six light columns, in one request
        table = parquet.read(columns=list(LIGHT_COLUMNS))
        # a parquet file is stored as row groups, parquet.num_row_groups of them, each holding a
        # block of rows (about 1000), rows_in_each_row_group is how many rows each group holds
        # example: a file of 8 rows in 3 groups gives rows_in_each_row_group = [3, 3, 2]
        rows_in_each_row_group = [
            parquet.metadata.row_group(g).num_rows for g in range(parquet.num_row_groups)
        ]
    # table has all the rows one after another, without saying which group a row came from, so
    # we work it out from the sizes: row_group_number_of_each_row[i] is the group number of row i
    # example: sizes [3, 3, 2] give [0, 0, 0, 1, 1, 1, 2, 2]
    row_group_number_of_each_row = [
        group for group, size in enumerate(rows_in_each_row_group) for _ in range(size)
    ]
    # the table is stored by column, so first turn each column into a plain python list
    # values_by_column_name["product_id"][5] is then the product id of row 5
    values_by_column_name = {name: table.column(name).to_pylist() for name in LIGHT_COLUMNS}
    pairs: list[Pair] = []
    product_ids_by_row_group_number: dict[int, set[str]] = {}
    for row in range(table.num_rows):
        if (
            values_by_column_name["product_locale"][row] != locale
            or values_by_column_name["small_version"][row] != small_version
        ):
            continue
        product_id = values_by_column_name["product_id"][row]
        label = LABEL_LETTERS.get(values_by_column_name["esci_label"][row])
        if label is None:
            raise ValueError(
                f"unknown esci_label {values_by_column_name['esci_label'][row]!r} in {source.name}"
            )
        pairs.append(
            Pair(
                query_id=values_by_column_name["query_id"][row],
                query=values_by_column_name["query"][row],
                product_id=product_id,
                locale=values_by_column_name["product_locale"][row],
                label=label,
                small_version=values_by_column_name["small_version"][row],
                dataset_split=source.split,
            )
        )
        product_ids_by_row_group_number.setdefault(row_group_number_of_each_row[row], set()).add(
            product_id
        )
    return pairs, product_ids_by_row_group_number


def read_product_text_heavy_pass(
    open_file: OpenFile,
    source: SourceFile,
    *,
    wanted_product_ids: Collection[str],
    row_group_numbers: Collection[int],
    locale: str,
) -> list[Product]:
    """The second pass: the text of the wanted products, read from the chosen row groups only."""
    wanted_product_id_set = set(wanted_product_ids)
    products: list[Product] = []
    with open_file(source) as handle:
        parquet = pq.ParquetFile(handle)
        for row_group_number in sorted(row_group_numbers):
            table = parquet.read_row_group(row_group_number, columns=list(TEXT_COLUMNS))
            for row in table.to_pylist():
                if (
                    row["product_id"] not in wanted_product_id_set
                    or row["product_locale"] != locale
                ):
                    continue
                products.append(
                    Product(
                        product_id=row["product_id"],
                        locale=row["product_locale"],
                        # a product with no title would break the composer later, an empty
                        # title is the honest value, the other fields are allowed to be missing
                        title=row["product_title"] or "",
                        description=row["product_description"],
                        bullet_point=row["product_bullet_point"],
                        brand=row["product_brand"],
                        color=row["product_color"],
                    )
                )
    return products


def plan_reads(
    product_ids_by_row_group: Mapping[Where, set[str]],
    must_have_product_ids: Collection[str],
    distractors_needed: int,
    seed: int,
    allowed_distractor_product_ids: Collection[str] | None = None,
) -> tuple[set[Where], set[str]]:
    """Choose which row groups to read the text from, so the second pass stays small.

    product_ids_by_row_group: for every row group, the products that have a kept row in it
    must_have_product_ids: the products we must have the text of (those of the mini queries)
    distractors_needed: how many more products we want as distractors
    allowed_distractor_product_ids: the only products that may count as distractors, None means
        any product, the mini build passes the products of its query pool, the same list that
        add_distractors_to_mini_selection filters on, so the plan never counts a product that
        add_distractors_to_mini_selection would refuse
    Returns the row groups to download, and every product that has text in those row groups.
    """
    must_have_set = set(must_have_product_ids)
    # None means any product may be a distractor, so allow every product of every row group
    allowed_set = (
        set(allowed_distractor_product_ids)
        if allowed_distractor_product_ids is not None
        else set[str]().union(*product_ids_by_row_group.values())
    )
    row_groups_to_download: set[Where] = set()
    product_ids_in_row_groups: set[str] = set()

    # 1. every must have product needs the text from at least one row group, take the first
    #    row group that has it, unless a row group we already chose has it, so we read as few
    #    row groups as we can
    # first_row_group_of_product: built once, a scan for every product would be far too slow
    # for 100 thousand products
    first_row_group_of_product: dict[str, Where] = {}
    for row_group in sorted(product_ids_by_row_group):
        for product_id in product_ids_by_row_group[row_group]:
            first_row_group_of_product.setdefault(product_id, row_group)
    for product_id in sorted(must_have_set):
        if product_id in product_ids_in_row_groups:
            continue
        first_row_group = first_row_group_of_product.get(product_id)
        if first_row_group is None:
            raise ValueError(f"product {product_id} not found in any row group")
        row_groups_to_download.add(first_row_group)
        product_ids_in_row_groups |= product_ids_by_row_group[first_row_group]

    # 2. the other products in the row groups we chose can already serve as distractors, if
    #    that is not enough add more row groups, in a repeatable shuffled order, until it is
    # usable_distractor_ids: products in the chosen row groups that are not must have and are
    # allowed, only these count towards distractors_needed
    usable_distractor_ids = (product_ids_in_row_groups - must_have_set) & allowed_set
    remaining_row_groups_shuffled = sorted(
        (
            row_group
            for row_group in product_ids_by_row_group
            if row_group not in row_groups_to_download
        ),
        key=lambda row_group: stable_rank(seed, "group", f"{row_group[0]}:{row_group[1]}"),
    )
    for row_group in remaining_row_groups_shuffled:
        if len(usable_distractor_ids) >= distractors_needed:
            break
        row_groups_to_download.add(row_group)
        new_product_ids = product_ids_by_row_group[row_group]
        product_ids_in_row_groups |= new_product_ids
        usable_distractor_ids |= (new_product_ids - must_have_set) & allowed_set
    if len(usable_distractor_ids) < distractors_needed:
        raise ValueError(
            f"only {len(usable_distractor_ids)} distractors are available, "
            f"{distractors_needed} needed"
        )
    return row_groups_to_download, product_ids_in_row_groups
