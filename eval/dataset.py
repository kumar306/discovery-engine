import hashlib
import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# what this file does
# the ESCI dataset is a big table where every row says "for this search, the judges labelled this
# product with this letter". we do not use all of it. this file picks a smaller set of searches,
# splits them into train, validation and test, builds a tiny version for fast tests (the mini set),
# and writes the result as plain text files plus a manifest that records how it was made.
#
# the steps, in order:
#   restrict()         keep only english rows of the small version, drop bad rows
#   select()           pick searches (queries) until their products cover enough distinct products
#   (inside select)    give every picked query a bucket: train, validation or test
#   build_mini()       from the selection, a small set for development and CI
#   collect_products() the text of the chosen products, one entry per product
#   write_dataset()    write the files and the manifest, verify_dataset() checks them later
#
# the repository never contains the rows themselves, only this code, the seed, the pinned
# revision and the checksums in the manifest, the rows are downloaded and rebuilt from them

# the exact copy of the dataset we use, pinned so nobody gets a changed dataset by surprise
DATASET_REPO = "tasksource/esci"
DATASET_REVISION = "8113b17a5d4099e20243282c926f1bc1a08a4d13"

# the four letters the judges use, see eval/metrics.py for what each one means
LABELS = frozenset("ESCI")
# the dataset ships two files: the train file and the test file
DATASET_SPLITS = frozenset({"train", "test"})


@dataclass(frozen=True)
class SelectionConfig:
    """Every number that decides what is selected. The same config and seed give the same set."""

    # the seed makes the choices repeatable, change it and you get another (equally valid) set
    seed: int
    locale: str = "us"
    # the dataset marks the rows of its smaller "reduced" version with small_version = 1
    small_version: int = 1
    # keep adding queries until their judged products cover at least this many distinct products
    target_products: int = 100_000
    # the share of the train queries that become validation queries, 0.10 means about one in ten
    validation_share: float = 0.10
    # the mini set: this many test queries, and this many products in total
    mini_queries: int = 50
    mini_products: int = 5_000


@dataclass(frozen=True, slots=True)
class Pair:
    """One row of the raw data: for this search, the judges labelled this product."""

    query_id: int
    query: str
    product_id: str
    locale: str
    label: str
    small_version: int
    # which of the two files of the dataset the row came from: "train" or "test"
    dataset_split: str


@dataclass(frozen=True, slots=True)
class Product:
    """The text of one product, as the raw data has it (any field can be missing)."""

    product_id: str
    locale: str
    title: str
    description: str | None
    bullet_point: str | None
    brand: str | None
    color: str | None


@dataclass(frozen=True)
class QueryRow:
    query_id: int
    query: str
    # train, validation or test
    split: str


@dataclass(frozen=True)
class JudgementRow:
    query_id: int
    product_id: str
    label: str


@dataclass(frozen=True)
class Selection:
    queries: tuple[QueryRow, ...]
    judgements: tuple[JudgementRow, ...]
    # every distinct product of the selection, sorted
    product_ids: tuple[str, ...]


class ManifestMismatch(Exception):
    """A file on disk is missing or is not the file the manifest was made for."""


def stable_rank(seed: int, salt: str, key: str) -> int:
    """A number for a key that is the same on every machine, and looks random.

    We need a repeatable shuffle: the same seed must pick the same queries tomorrow, on a
    colleague's laptop and in CI. A random number library cannot promise that across versions, so
    we do not use one. We hash the text "seed:salt:key" with sha256 and read the first 8 bytes of
    the hash as a number. Sorting keys by this number puts them in a shuffled order that depends
    only on the seed. The salt keeps different decisions independent: the same query gets a
    different number when we pick queries than when we decide validation.
    example: stable_rank(7, "queries", "42") is the number from the hash of "7:queries:42"
    """
    digest = hashlib.sha256(f"{seed}:{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def restrict(pairs: Iterable[Pair], config: SelectionConfig) -> list[Pair]:
    """Keep the english rows of the small version, once each, with a known label."""
    kept: dict[tuple[int, str], Pair] = {}
    for p in pairs:
        # rows of other languages and of the large version are not part of our dataset
        if p.locale != config.locale or p.small_version != config.small_version:
            continue
        if p.label not in LABELS:
            raise ValueError(f"unknown label {p.label!r}, expected one of {sorted(LABELS)}")
        key = (p.query_id, p.product_id)
        earlier = kept.get(key)
        if earlier is None:
            kept[key] = p
        elif earlier.label != p.label:
            # the same search and product judged twice with different letters, we cannot know which
            raise ValueError(f"two labels for query {p.query_id} and product {p.product_id}")
        # the same row twice with the same label is just a duplicate, it collapses to one
    return list(kept.values())


def _check_each_query_is_in_one_file(pairs: Iterable[Pair]) -> None:
    # the whole split rests on this: a search must be either in the train file or in the test
    # file, never both, otherwise a test query would also be trained on
    files: dict[int, set[str]] = {}
    for p in pairs:
        if p.dataset_split not in DATASET_SPLITS:
            raise ValueError(f"unknown dataset split {p.dataset_split!r}")
        files.setdefault(p.query_id, set()).add(p.dataset_split)
    for query_id, seen in files.items():
        if len(seen) > 1:
            raise ValueError(f"query {query_id} is in both the train and the test file")


def _bucket(query_id: int, dataset_split: str, config: SelectionConfig) -> str:
    # a query from the test file is a test query, we keep the dataset's own test set
    if dataset_split == "test":
        return "test"
    # a query from the train file is train, except a share of them become validation
    # the choice is a hash of the seed and the query id turned into a fraction between 0 and 1
    # (the number divided by 2**64, the biggest value it can have), and it is validation when
    # that fraction is below the share: with a share of 0.10 about one query in ten, with 0
    # none, with 1.0 all of them
    fraction = stable_rank(config.seed, "validation", str(query_id)) / 2**64
    return "validation" if fraction < config.validation_share else "train"


def _group_by_query(rows: list[Pair]) -> dict[int, list[Pair]]:
    # turns a flat list of rows into: query id -> the rows judged for that query
    # example: [row(q=1, P1), row(q=1, P2), row(q=2, P2)] becomes {1: [row(q=1, P1), row(q=1, P2)],
    # 2: [row(q=2, P2)]}
    by_query: dict[int, list[Pair]] = {}
    for row in rows:
        by_query.setdefault(row.query_id, []).append(row)
    return by_query


def _pick_queries(
    by_query: dict[int, list[Pair]], config: SelectionConfig
) -> tuple[list[int], set[str]]:
    """Choose queries one by one, in a repeatable shuffled order, until enough products are covered.

    Returns the chosen query ids, and the set of distinct product ids those queries cover.
    """
    # the queries in shuffled order, the same order for the same seed
    order = sorted(by_query, key=lambda q: stable_rank(config.seed, "select", str(q)))
    chosen: list[int] = []
    covered: set[str] = set()
    for query_id in order:
        # enough distinct products are covered, stop here
        if len(covered) >= config.target_products:
            break
        # a query is taken whole: all the products the judges labelled for it come with it, a
        # query with only some of its labels would give unfair scores
        chosen.append(query_id)
        covered.update(p.product_id for p in by_query[query_id])
    return chosen, covered


def _query_row(query_id: int, rows: list[Pair], config: SelectionConfig) -> QueryRow:
    # all the rows of one query have the same search text and come from the same file,
    # so the first row speaks for all of them
    first = rows[0]
    return QueryRow(
        query_id=query_id,
        query=first.query,
        # this is where a query gets its bucket: train, validation or test
        split=_bucket(query_id, first.dataset_split, config),
    )


def select(pairs: Iterable[Pair], config: SelectionConfig) -> Selection:
    """From the raw rows, pick queries, give each one a bucket, and collect what they cover."""
    rows = restrict(pairs, config)
    _check_each_query_is_in_one_file(rows)
    by_query = _group_by_query(rows)
    chosen, covered = _pick_queries(by_query, config)
    # the bucket is given to each chosen QUERY, a product has no bucket of its own
    queries = [_query_row(query_id, by_query[query_id], config) for query_id in chosen]
    judgements = [
        JudgementRow(query_id=row.query_id, product_id=row.product_id, label=row.label)
        for query_id in chosen
        for row in by_query[query_id]
    ]
    # sorted, so the result does not depend on the order the rows arrived in
    return Selection(
        queries=tuple(sorted(queries, key=lambda row: row.query_id)),
        judgements=tuple(sorted(judgements, key=lambda row: (row.query_id, row.product_id))),
        product_ids=tuple(sorted(covered)),
    )


def choose_mini(
    selection: Selection, config: SelectionConfig
) -> tuple[tuple[QueryRow, ...], tuple[JudgementRow, ...]]:
    """The queries of the mini set, and every judgement of those queries."""
    test_queries = [q for q in selection.queries if q.split == "test"]
    if len(test_queries) < config.mini_queries:
        raise ValueError(
            f"the selection has only {len(test_queries)} test queries, "
            f"{config.mini_queries} are needed"
        )
    shuffled = sorted(test_queries, key=lambda q: stable_rank(config.seed, "mini", str(q.query_id)))
    chosen = sorted(shuffled[: config.mini_queries], key=lambda q: q.query_id)
    chosen_ids = {q.query_id for q in chosen}
    # all the judgements of the chosen queries, the selection is already sorted
    judgements = tuple(j for j in selection.judgements if j.query_id in chosen_ids)
    return tuple(chosen), judgements


def build_mini(
    selection: Selection,
    config: SelectionConfig,
    product_ids_in_planned_blocks: Collection[str] | None = None,
) -> Selection:
    """A small set for fast tests: test queries, their labelled products, and distractors.

    product_ids_in_planned_blocks, when given, limits the distractors to those products. The
    download script gives the products whose text it will read anyway, so the mini set
    needs few row groups.
    """
    chosen, judgements = choose_mini(selection, config)
    mini_query_product_ids = {j.product_id for j in judgements}
    if len(mini_query_product_ids) > config.mini_products:
        raise ValueError(
            f"mini_products {config.mini_products} is smaller than the "
            f"{len(mini_query_product_ids)} products judged for the mini queries"
        )
    # distractors: products of the selection that the judges did NOT label for these queries
    #
    # why we need them: later we load these products into the search index (Qdrant) and run the
    # test searches against it, then score the results with ndcg
    # if we loaded ONLY the products the judges labelled for these searches, the index would hold
    # nothing else, so every search could only ever return those few products, and almost any
    # ordering would put the good ones near the top, so ndcg and recall would come out high and
    # look better than the system really is, a false score
    # a real catalogue is full of products that do not belong to a given search, and the system has
    # to rank the right ones above all of them
    # the distractors play that role: they compete for the top places, so a search that cannot
    # tell good from bad scores low here, the way it would in production
    # the more realistic the competition, the more we can trust the score
    planned_block_product_ids = (
        None if product_ids_in_planned_blocks is None else set(product_ids_in_planned_blocks)
    )
    distractor_candidates_ranked = sorted(
        (
            p
            for p in selection.product_ids
            if p not in mini_query_product_ids
            and (planned_block_product_ids is None or p in planned_block_product_ids)
        ),
        key=lambda p: stable_rank(config.seed, "distractor", p),
    )
    distractors_needed = config.mini_products - len(mini_query_product_ids)
    if distractors_needed > len(distractor_candidates_ranked):
        raise ValueError(
            f"only {len(distractor_candidates_ranked)} products are left to use as distractors"
        )
    product_ids = tuple(
        sorted(mini_query_product_ids.union(distractor_candidates_ranked[:distractors_needed]))
    )
    return Selection(chosen, judgements, product_ids)


def collect_products(
    product_ids: Iterable[str], products: Iterable[Product]
) -> tuple[Product, ...]:
    """One entry per wanted product, sorted by id. The raw data repeats a product once per query."""
    wanted = set(product_ids)
    found: dict[str, Product] = {}
    for p in products:
        if p.product_id in wanted and p.product_id not in found:
            found[p.product_id] = p
    missing = wanted.difference(found)
    if missing:
        raise ValueError(f"{len(missing)} product(s) missing, for example {sorted(missing)[:3]}")
    return tuple(found[product_id] for product_id in sorted(found))


def _lines(rows: Sequence[dict[str, Any]]) -> bytes:
    # one json object per line, keys sorted, no spaces, utf 8, the same bytes every time
    # so the same data always gives the same file and therefore the same checksum
    text = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    return text.encode("utf-8")


def write_dataset(
    directory: Path,
    selection: Selection,
    products: Iterable[Product],
    config: SelectionConfig,
    dataset_revision: int,
    sources: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Write queries, judgements and products as text files, and a manifest that describes them.

    dataset_revision is our own whole number for this build of the dataset, it is the version
    the bulk loader gives every document, raise it by hand when the data or the selection changes.
    """
    sorted_products = sorted(products, key=lambda p: p.product_id)
    contents = {
        "queries.jsonl": _lines([asdict(q) for q in selection.queries]),
        "judgements.jsonl": _lines([asdict(j) for j in selection.judgements]),
        "products.jsonl": _lines([asdict(p) for p in sorted_products]),
    }
    for name, data in contents.items():
        (directory / name).write_bytes(data)
    manifest: dict[str, Any] = {
        "schema": 1,
        "dataset": {"repo": DATASET_REPO, "revision": DATASET_REVISION},
        "dataset_revision": dataset_revision,
        "config": asdict(config),
        # the files of the dataset this was built from, with the checksums huggingface gives them
        "source_files": [dict(source) for source in sources],
        "counts": {
            "queries": len(selection.queries),
            "train": sum(q.split == "train" for q in selection.queries),
            "validation": sum(q.split == "validation" for q in selection.queries),
            "test": sum(q.split == "test" for q in selection.queries),
            "judgements": len(selection.judgements),
            "products": len(sorted_products),
        },
        # the sha256 of each file: if one byte of a file changes, its checksum no longer matches
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (directory / "manifest.json").write_bytes(text.encode("utf-8"))
    return manifest


def verify_dataset(directory: Path) -> None:
    """Check every file against the manifest, raise ManifestMismatch naming the first bad file."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.exists():
            raise ManifestMismatch(f"{name} is missing")
        # file_digest reads the file in blocks, a big products.jsonl is never held whole
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ManifestMismatch(f"{name} does not match the manifest")
