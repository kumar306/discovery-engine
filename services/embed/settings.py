from pydantic import PositiveFloat, PositiveInt, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


# these values are only fallbacks for when an environment variable is not present in the pod
# on the cluster every one of them is injected from a config map and wins over what is written here
class EmbedSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBED_", env_file=None)

    qdrant_url: str = "http://127.0.0.1:6333"
    # the collection is created by make init, the worker only verifies it at startup
    qdrant_collection: str = "chunks_esci"
    qdrant_timeout_s: PositiveInt = 10

    # the dense model and the size of its vectors, the startup check compares the two
    dense_model: str = "BAAI/bge-small-en-v1.5"
    dense_dim: PositiveInt = 384
    # the bm25 length normalisation assumes this average chunk length in tokens
    # k and b stay at the fastembed defaults, changing them is a code change that bumps the revision
    bm25_avg_len: PositiveFloat = 256.0
    # bump this by hand for any other change that alters stored vectors
    model_revision: PositiveInt = 1

    # onnx threads, a plain number on purpose, the default is a guess until the aws benchmark
    embed_threads: PositiveInt = 2
    embed_sub_batch: PositiveInt = 16
    embed_timeout_s: PositiveFloat = 30.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_version(self) -> str:
        # the model and the bm25 parameters are baked into every chunk id and into the collection
        # built from the parts, so changing a part changes the version, and nobody can change the
        # parameter and forget the version, which would mix two kinds of vectors in one collection
        name = self.dense_model.split("/")[-1]
        return f"{name}_bm25-a{self.bm25_avg_len:g}_{self.model_revision}"
