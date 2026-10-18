"""Download the pinned ESCI dataset in pieces and build the repeatable selection on disk.

  python scripts/download_data.py                 the mini set (make data)
  python scripts/download_data.py --size full     the 100 thousand product set (make data-full)

Nothing is committed: the output goes to data/processed/<size>, which git ignores. The
repository holds this script, the seed, the pinned revision and the checksums of a build in its
manifest, never the rows.
"""

import argparse
import time
from pathlib import Path

from eval.build import build_full_dataset, build_mini_dataset
from eval.dataset import SelectionConfig, verify_dataset
from eval.source import fetch_source_files, hf_opener


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--size", choices=("mini", "full"), default="mini")
    parser.add_argument("--out", type=Path, default=None, help="default: data/processed/<size>")
    parser.add_argument("--seed", type=int, default=7, help="changes which queries are picked")
    parser.add_argument(
        "--dataset-revision",
        type=int,
        default=1,
        help="our own whole number for this build, it becomes the version of every document",
    )
    args = parser.parse_args(argv)

    config = SelectionConfig(seed=args.seed)
    out: Path = args.out or Path("data") / "processed" / args.size
    started = time.monotonic()
    files = fetch_source_files()
    print(f"{len(files)} parquet files listed at the pinned revision", flush=True)
    build = build_mini_dataset if args.size == "mini" else build_full_dataset
    manifest = build(files, hf_opener(), config, out, args.dataset_revision)
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
