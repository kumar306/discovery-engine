"""The dev setup script: what it refuses to touch and how it fails. No stack needed."""

import time

import pytest
from botocore.exceptions import BotoCoreError

from scripts.dev_init import ensure_bucket
from services.ingest.settings import IngestSettings


def test_it_refuses_to_create_a_bucket_when_no_endpoint_is_set() -> None:
    # no endpoint url means real aws, and real buckets belong to terraform
    with pytest.raises(SystemExit, match="terraform owns those"):
        ensure_bucket(IngestSettings(s3_endpoint_url=None))


def test_an_unreachable_stand_in_fails_fast_instead_of_hanging() -> None:
    settings = IngestSettings(s3_endpoint_url="http://127.0.0.1:9", s3_timeout_s=0.5)
    started = time.monotonic()
    # windows times out on a closed port and linux refuses it, both are botocore errors
    with pytest.raises(BotoCoreError):
        ensure_bucket(settings)
    assert time.monotonic() - started < 3
