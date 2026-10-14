"""Bring a fresh dev stack to a usable state: tables, bucket and collection, safe to repeat."""

import asyncio

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from libs.db.migrate import migrate
from services.embed.qdrant import ensure_collection, new_client
from services.embed.settings import EmbedSettings
from services.ingest.settings import IngestSettings


def ensure_bucket(settings: IngestSettings) -> bool:
    """Create the bucket on the local stand in, return False when it was already there."""
    # a real aws bucket belongs to terraform, so with no endpoint url set this script stops
    if settings.s3_endpoint_url is None:
        raise SystemExit("refusing to create a bucket on real aws, terraform owns those")
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        region_name=settings.s3_region,
        # the stand in accepts any credentials, and only the stand in is ever reached here
        aws_access_key_id="test",
        aws_secret_access_key="test",  # noqa: S106
        config=Config(
            connect_timeout=settings.s3_timeout_s,
            read_timeout=settings.s3_timeout_s,
            retries={"total_max_attempts": 1},
        ),
    )
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
    except ClientError as error:
        # a missing bucket answers 404 and is created below, any other answer is a real failure
        if error.response.get("Error", {}).get("Code") != "404":
            raise
    else:
        return False
    client.create_bucket(Bucket=settings.s3_bucket)
    return True


async def ensure_qdrant_collection(settings: EmbedSettings) -> bool:
    """Create the collection when it is missing, a wrong existing one raises and stops make init."""
    client = new_client(settings)
    try:
        return await ensure_collection(client, settings)
    finally:
        await client.close()


def main() -> None:
    settings = IngestSettings()
    applied = migrate(settings.database_dsn)
    print(f"migrations applied: {', '.join(applied) if applied else 'none, already up to date'}")
    created = ensure_bucket(settings)
    print(f"bucket {settings.s3_bucket}: {'created' if created else 'already there'}")
    embed = EmbedSettings()
    made = asyncio.run(ensure_qdrant_collection(embed))
    print(f"collection {embed.qdrant_collection}: {'created' if made else 'already there'}")


if __name__ == "__main__":
    main()
