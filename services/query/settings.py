from pydantic import PositiveFloat, PositiveInt
from pydantic_settings import SettingsConfigDict

from libs.embedding.settings import IndexSettings
from libs.settings import HealthSettings


# these values are only fallbacks for when an environment variable is not present in the pod
# on the cluster every one of them is injected from a config map and wins over what is written here
# the numbers about depth, fusion and timeouts are starting guesses, quality is tuned on ESCI-mini
# in the evaluation ladder and what they cost in latency is measured only on the aws deployment
class QuerySettings(IndexSettings, HealthSettings):
    model_config = SettingsConfigDict(env_prefix="QUERY_", env_file=None)

    # http, the host is only widened to all interfaces inside a container
    host: str = "127.0.0.1"
    port: PositiveInt = 8010
    max_body_bytes: PositiveInt = 16 * 1024
    # seconds a client is told to wait before retrying after a 503
    retry_after_s: PositiveInt = 1
    # the outermost limit: requests being served at once, a full house answers 503 at once
    max_inflight_requests: PositiveInt = 32
    admission_wait_s: PositiveFloat = 0.05

    # how many products each retriever returns in the search flow, and how many chunks in the
    # chunk flow when the request does not say
    candidates_per_retriever: PositiveInt = 100
    chunks_per_retriever: PositiveInt = 20
    # the constant of reciprocal rank fusion, 60 is the value of the original paper
    rrf_k: PositiveInt = 60
    # a retriever is encode plus search, this long for both together, then it is left out
    retriever_timeout_s: PositiveFloat = 2.0
    # the dense model call alone, it cannot be interrupted, only waited for
    encode_timeout_s: PositiveFloat = 2.0
    # whether the dense model gets its query prefix, the ladder decides which is better
    dense_query_instruction: bool = False
    # characters of the best chunk shown as the snippet of a result
    snippet_chars: PositiveInt = 300
