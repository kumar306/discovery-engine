from pydantic import PositiveFloat, PositiveInt
from pydantic_settings import SettingsConfigDict

from libs.embedding.settings import IndexSettings
from libs.settings import (
    HealthSettings,
    KafkaSettings,
    PostgresSettings,
    S3Settings,
)


# these values are only fallbacks for when an environment variable is not present in the pod
# on the cluster every one of them is injected from a config map and wins over what is written here
class EmbedSettings(IndexSettings, PostgresSettings, S3Settings, KafkaSettings, HealthSettings):
    model_config = SettingsConfigDict(env_prefix="EMBED_", env_file=None)

    # the consumer: one group of workers shares the partitions of the events topic
    consumer_group: str = "embed-worker"
    # how long one poll waits for a message before the loop looks at its timers and the stop signal
    consumer_poll_ms: PositiveInt = 500
    # tries for one message before it is given up on, a transient error then stalls its partition
    message_attempts: PositiveInt = 3
    # a stalled partition is held and tried again, the wait grows from the first number to the cap
    stall_base_delay_s: PositiveFloat = 1.0
    stall_max_delay_s: PositiveFloat = 30.0
    dlq_produce_timeout_s: PositiveFloat = 5.0
