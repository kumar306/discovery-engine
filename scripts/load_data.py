"""Load the processed ESCI dataset through the running ingest api.

  python scripts/load_data.py                 the mini set (make load-mini)
  python scripts/load_data.py --size full     the 100 thousand product set (make load-full)

The api, the outbox relay and the worker must be running. The exit code is 0 only when every
product was reported indexed, and then loaded.json exists next to the dataset. Exit code 2
means the system was not ready and nothing was sent.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from eval.loader import LoadConfig, LoaderNotReady, load_dataset

# the readiness routes of the three processes, on their default ports: the api itself, the worker
# (it also checks that qdrant has its collection) and the relay
DEFAULT_READY_URLS = (
    "http://127.0.0.1:8000/readyz",
    "http://127.0.0.1:8081/readyz",
    "http://127.0.0.1:8083/readyz",
)


async def run(arguments: argparse.Namespace) -> int:
    directory: Path = arguments.dir or Path("data/processed") / arguments.size
    config = LoadConfig(
        max_in_flight=arguments.max_in_flight,
        wait_timeout_s=arguments.wait_timeout_s,
        ready_urls=tuple(arguments.ready_url or DEFAULT_READY_URLS),
    )
    # every request has a timeout, a hung connection must not hang the load
    async with httpx.AsyncClient(
        base_url=arguments.base_url, timeout=httpx.Timeout(10.0)
    ) as client:
        try:
            report = await load_dataset(client, directory, config)
        except LoaderNotReady as error:
            print(f"not ready, nothing was sent: {error}", file=sys.stderr)
            return 2
    if report.failures:
        print(
            f"{len(report.failures)} of {report.documents} documents failed, no marker written",
            file=sys.stderr,
        )
        for doc_id, reason in sorted(report.failures.items())[:10]:
            print(f"  {doc_id}: {reason}", file=sys.stderr)
        return 1
    print(f"{report.documents} documents indexed, wrote {directory / 'loaded.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--size", choices=("mini", "full"), default="mini")
    parser.add_argument("--dir", type=Path, default=None, help="default: data/processed/<size>")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-in-flight", type=int, default=16)
    parser.add_argument("--wait-timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--ready-url",
        action="append",
        help="a url that must answer 200 before anything is sent, repeat for several",
    )
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
