"""The versions of the libraries that decide what vectors an index holds, as text for a cache key.

The stored index of the CI quality job is keyed by the code that defines it and by these versions,
not by the whole lock file: a bump of any dependency would otherwise rebuild a 25 minute index.
These six decide the numbers: the dense model runs in onnxruntime on arrays of numpy, its text goes
through tokenizers, and the sparse (bm25) terms come from the stemmer and the murmur hash that
fastembed calls. A change to one of them can change a vector, so it must change the key.

    python -m eval.index_libs      prints, for example:
    fastembed-0.8.1_onnxruntime-1.30.0_tokenizers-0.23.2_py-rust-stemmers-0.1.8_mmh3-5.3.1_numpy-2.5.3
"""

import importlib.metadata

INDEX_DEFINING_PACKAGES: tuple[str, ...] = (
    "fastembed",
    "onnxruntime",
    "tokenizers",
    "py-rust-stemmers",
    "mmh3",
    "numpy",
)


def index_library_versions() -> str:
    """Name-version of each package, joined with underscores, always in the same order."""
    return "_".join(
        f"{name}-{importlib.metadata.version(name)}" for name in INDEX_DEFINING_PACKAGES
    )


if __name__ == "__main__":
    print(index_library_versions())
