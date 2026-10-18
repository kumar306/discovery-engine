"""A tiny stand in for the ESCI parquet files, written to disk with the real column names.

The files have the same 14 columns as the real dataset, a handful of rows, and small row groups,
so the readers and the planner can be tested without any download.
"""

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

COLUMNS = [
    "example_id",
    "query",
    "query_id",
    "product_id",
    "product_locale",
    "esci_label",
    "small_version",
    "large_version",
    "product_title",
    "product_description",
    "product_bullet_point",
    "product_brand",
    "product_color",
    "product_text",
]


def esci_row(
    query_id: int,
    product_id: str,
    label: str = "E",
    locale: str = "us",
    small_version: int = 1,
    **text: Any,
) -> dict[str, Any]:
    """One row of the raw table. The text fields default to something derived from the id."""
    # the real files write the judge's label as a word
    words = {"E": "Exact", "S": "Substitute", "C": "Complement", "I": "Irrelevant"}
    row: dict[str, Any] = {
        "example_id": query_id * 1000 + int(product_id[1:]),
        "query": f"query {query_id}",
        "query_id": query_id,
        "product_id": product_id,
        "product_locale": locale,
        "esci_label": words[label],
        "small_version": small_version,
        "large_version": 1,
        "product_title": f"title {product_id}",
        "product_description": f"description {product_id}",
        "product_bullet_point": f"bullet one {product_id}\nbullet two {product_id}",
        "product_brand": f"brand {product_id}",
        "product_color": "black",
        "product_text": "never used, the mirror has this column and we must not read it",
    }
    row.update(text)
    return row


def write_esci_parquet(path: Path, rows: list[dict[str, Any]], row_group_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=None)
    # keep the column order of the real files
    table = table.select(COLUMNS)
    pq.write_table(table, path, row_group_size=row_group_size)
