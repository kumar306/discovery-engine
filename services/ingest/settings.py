from pydantic import PositiveFloat, PositiveInt
from pydantic_settings import SettingsConfigDict

from libs.settings import KafkaSettings, PostgresSettings, S3Settings


# these values are only fallbacks for when an environment variable is not present in the pod
# on the cluster every one of them is injected from a config map and wins over what is written here
# postgres, s3 and kafka come from the shared settings, the rest belongs to the ingest service
class IngestSettings(PostgresSettings, S3Settings, KafkaSettings):
    model_config = SettingsConfigDict(env_prefix="INGEST_", env_file=None)

    # http, the host is only widened to all interfaces inside a container
    host: str = "127.0.0.1"
    port: PositiveInt = 8000
    max_body_bytes: PositiveInt = 256 * 1024
    # seconds a client is told to wait before retrying after a 503
    retry_after_s: PositiveInt = 1
    # the outermost limit: requests being served at once, a full house answers 503 at once
    max_inflight_requests: PositiveInt = 32
    admission_wait_s: PositiveFloat = 0.05

    # the outbox relay
    relay_batch_size: PositiveInt = 100
    relay_poll_interval_s: PositiveFloat = 0.5
    relay_produce_timeout_s: PositiveFloat = 5.0
