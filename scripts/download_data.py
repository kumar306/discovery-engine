"""Download the pinned ESCI dataset in pieces and build the repeatable selection on disk.

  python scripts/download_data.py                 the mini set, 50 test queries (make data)
  python scripts/download_data.py --size validation --queries 500 --products 14263
                                                  the validation set (make data-validation)
  python scripts/download_data.py --size full     the 100 thousand product set (make data-full)

Nothing is committed: the output goes to data/processed/<directory>, which git ignores. The
repository holds this script, the seed, the pinned revision and the checksums of a build in its
manifest, never the rows.
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

from eval.build import build_full_dataset, build_mini_dataset, build_validation_dataset
from eval.dataset import SelectionConfig, verify_dataset
from eval.source import fetch_source_files, hf_opener

# the builder and the folder under data/processed of each size
BUILDERS = {
    "mini": build_mini_dataset,
    "validation": build_validation_dataset,
    "full": build_full_dataset,
}
DIRECTORIES = {"mini": "mini", "validation": "validation", "full": "full"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--size", choices=tuple(BUILDERS), default="mini")
    parser.add_argument(
        "--queries",
        type=int,
        default=None,
        help="validation only: how many validation queries (the CI mini set is always 50)",
    )
    parser.add_argument(
        "--products",
        type=int,
        default=None,
        help="validation only: products in all, the judged ones plus distractors",
    )
    parser.add_argument("--out", type=Path, default=None, help="default: data/processed/<folder>")
    parser.add_argument("--seed", type=int, default=7, help="changes which queries are picked")
    parser.add_argument(
        "--dataset-revision",
        type=int,
        default=1,
        help="our own whole number for this build, it becomes the version of every document",
    )
    args = parser.parse_args(argv)
    if args.size != "validation" and (args.queries is not None or args.products is not None):
        parser.error("--queries and --products are only for --size validation")

    config = SelectionConfig(seed=args.seed)
    if args.queries is not None:
        config = replace(config, mini_queries=args.queries)
    if args.products is not None:
        config = replace(config, mini_products=args.products)
    out: Path = args.out or Path("data") / "processed" / DIRECTORIES[args.size]
    started = time.monotonic()
    files = fetch_source_files()
    print(f"{len(files)} parquet files listed at the pinned revision", flush=True)
    manifest = BUILDERS[args.size](files, hf_opener(), config, out, args.dataset_revision)
    verify_dataset(out)
    counts = manifest["counts"]
    print(
        f"{args.size} dataset written to {out} in {time.monotonic() - started:.0f} s: "
        f"{counts['queries']} queries ({counts['train']} train, {counts['validation']} validation, "
        f"{counts['test']} test), {counts['judgements']} judgements, {counts['products']} products",
        flush=True,
    )


if __name__ == "__main__":
    main()
