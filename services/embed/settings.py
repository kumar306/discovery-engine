from pydantic import PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


# these values are only fallbacks for when an environment variable is not present in the pod
# on the cluster every one of them is injected from a config map and wins over what is written here
class EmbedSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBED_", env_file=None)

    qdrant_url: str = "http://127.0.0.1:6333"
    # the collection is created by make init, the worker only verifies it at startup
    qdrant_collection: str = "chunks_esci"
    qdrant_timeout_s: PositiveInt = 10

    # the size of the dense vectors, it must match the model, the startup check compares them
    dense_dim: PositiveInt = 384
    # the model and the bm25 parameters are baked into every chunk id, so changing this string
    # makes every chunk of every document new, and a collection only ever holds one value
    model_version: str = "bge-small-en-v1.5_bm25-a256_1"
