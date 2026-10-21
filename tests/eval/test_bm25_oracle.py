"""G10: our sparse (bm25) search against a reference bm25, on the real index of ESCI-mini.

needs the dev stack with ESCI-mini loaded (make load-mini), marker eval, so make test skips it
    make oracle
writes benchmarks/results/g10-bm25-oracle/<time>/ with the numbers and the rankings
the production variant must have the same scores as ours at every rank (gap at most MAX_SCORE_GAP),
the order measures and the textbook variant are written to the result but not gated
"""

import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models

from eval.bm25 import Bm25Corpus, tokenise
from eval.metrics import overlap_at, rank_correlation, score_gap_by_rank, top1_equal
from eval.results import write_result
from libs.embedding.collection import QdrantPointState, normalise, verify_collection
from libs.embedding.model import OnnxEmbedder, build_onnx_embedder
from services.query.encoder import OnnxQueryEncoder
from services.query.search import Filters, QdrantSearch
from services.query.settings import QuerySettings

pytestmark = pytest.mark.eval

DATASET = Path("data/processed/mini")
RESULTS = Path("benchmarks/results")
# how many chunks of each ranking are compared, and the k of overlap_at
DEPTH = 50
OVERLAP_K = 10
# qdrant keeps float32 and the oracle float64, the measured gap was 1.6e-07 and float32 spacing is
# about 6e-8, so this leaves about 60 times of margin, revise it if a library update moves it
MAX_SCORE_GAP = 1e-5


async def test_our_sparse_search_agrees_with_a_reference_bm25(tmp_path: Path) -> None:
    settings = QuerySettings()

    # 1. the index must be the one loaded from this dataset
    manifest_sha256 = hashlib.sha256((DATASET / "manifest.json").read_bytes()).hexdigest()
    marker_path = DATASET / "loaded.json"
    assert marker_path.exists(), "no loaded.json, run make load-mini first"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["manifest_sha256"] == manifest_sha256, (
        "the index was not loaded from this dataset"
    )

    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url, timeout=settings.qdrant_timeout_s)
    embedder: OnnxEmbedder | None = None
    # everything from here on is inside the try, the client and the model close whatever fails
    try:
        # the collection must be the one built for the model version of these settings, the stored
        # model_version holds the bm25 length constant (bm25-a256), so this is what shows that the
        # index was made with the length that the production variant of the oracle assumes
        # it also checks the vector sizes, the idf modifier and the payload indexes
        await verify_collection(qdrant_client, settings)

        # 2. every live chunk of the index, point id to text
        live = models.Filter(
            must=[
                models.FieldCondition(
                    key="state",
                    match=models.MatchValue(value=QdrantPointState.QDRANT_POINT_LIVE.value),
                )
            ]
        )
        chunk_text_by_point_id: dict[str, str] = {}
        cursor = None
        while True:
            records, cursor = await qdrant_client.scroll(
                settings.qdrant_collection,
                scroll_filter=live,
                limit=500,
                with_payload=["text"],
                with_vectors=False,
                offset=cursor,
            )
            for record in records:
                chunk_text_by_point_id[str(record.id)] = (record.payload or {})["text"]
            if cursor is None:
                break
        assert chunk_text_by_point_id, "the index holds no live chunk"

        # 3. the tokens of every chunk and the reference corpus
        # k and b are not part of model_version, they are the defaults of the installed fastembed,
        # the same library the worker uses, and they are read from its object, not typed again
        bm25_model = SparseTextEmbedding("Qdrant/bm25", avg_len=settings.bm25_avg_len).model
        k1, b, production_average = bm25_model.k, bm25_model.b, bm25_model.avg_len
        corpus = Bm25Corpus(
            {
                point_id: tokenise(bm25_model, text)
                for point_id, text in chunk_text_by_point_id.items()
            }
        )
        assert corpus.average_length > 0

        # 4 to 6. every query through our retriever and through the oracle, then compared
        embedder = build_onnx_embedder(settings)
        encoder = OnnxQueryEncoder(embedder, settings.encode_timeout_s)
        search = QdrantSearch(qdrant_client, settings.qdrant_collection)
        queries = [
            json.loads(line)
            for line in (DATASET / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        average_of = {"production": production_average, "textbook": corpus.average_length}
        run_lines: dict[str, list[str]] = {"ours": [], "production": [], "textbook": []}
        per_query: list[dict[str, object]] = []
        measures: dict[str, dict[str, list[float]]] = {
            variant: {"overlap": [], "top1": [], "rho": [], "gap": []} for variant in average_of
        }
        for query in queries:
            text = query["query"]
            sparse_query = await encoder.encode_sparse(text)
            hits = (
                await search.chunk_search("bm25", sparse_query, Filters(), DEPTH)
                if sparse_query.indices
                else []
            )
            ours = [hit.point_id for hit in hits]
            run_lines["ours"] += [
                f"{query['query_id']} Q0 {hit.point_id} {rank} {hit.score} ours"
                for rank, hit in enumerate(hits, start=1)
            ]
            query_tokens = tokenise(bm25_model, normalise(text).casefold())
            row: dict[str, object] = {
                "query_id": query["query_id"],
                "query": text,
                "ours": len(ours),
            }
            for variant, average_length in average_of.items():
                ranked = corpus.rank(
                    query_tokens, k1=k1, b=b, average_length=average_length, limit=DEPTH
                )
                reference = [item.doc_id for item in ranked]
                run_lines[variant] += [
                    f"{query['query_id']} Q0 {item.doc_id} {rank} {item.score} {variant}"
                    for rank, item in enumerate(ranked, start=1)
                ]
                overlap = overlap_at(OVERLAP_K, ours, reference)
                first = top1_equal(ours, reference)
                rho = rank_correlation(ours, reference)
                # the scores standing at each rank, not the chunks: tie order does not matter and
                # a chunk that ours lost moves every score below it up a rank and shows as a gap
                gap = score_gap_by_rank(
                    [hit.score for hit in hits], [item.score for item in ranked]
                )
                # a query that both sides answered with nothing is skipped, it is not agreement
                if overlap is not None:
                    measures[variant]["overlap"].append(overlap)
                if first is not None:
                    measures[variant]["top1"].append(1.0 if first else 0.0)
                if rho is not None:
                    measures[variant]["rho"].append(rho)
                if gap is not None:
                    measures[variant]["gap"].append(gap)
                row[f"{variant}_results"] = len(reference)
                row[f"{variant}_overlap_at_10"] = overlap
                row[f"{variant}_top1_equal"] = first
                row[f"{variant}_rank_correlation"] = rho
                row[f"{variant}_score_gap_by_rank"] = gap
            per_query.append(row)
    finally:
        if embedder is not None:
            embedder.close()
        await qdrant_client.close()

    # the setup was sound: there is something to average for every measure and variant
    for variant, values in measures.items():
        for name, numbers in values.items():
            assert numbers, f"no query could be compared for {variant} {name}"

    # 7. the averages per variant
    compared = len(measures["production"]["overlap"])
    metrics: dict[str, float] = {
        "corpus_chunks": float(len(chunk_text_by_point_id)),
        "corpus_average_length": corpus.average_length,
        "assumed_average_length": production_average,
        "queries_compared": float(compared),
        "queries_skipped": float(len(queries) - compared),
    }
    for variant, values in measures.items():
        metrics[f"{variant}_overlap_at_10"] = statistics.fmean(values["overlap"])
        metrics[f"{variant}_overlap_at_10_min"] = min(values["overlap"])
        metrics[f"{variant}_top1_equal_rate"] = statistics.fmean(values["top1"])
        metrics[f"{variant}_rank_correlation"] = statistics.fmean(values["rho"])
        metrics[f"{variant}_score_gap_max"] = max(values["gap"])
        metrics[f"{variant}_score_gap_mean"] = statistics.fmean(values["gap"])

    # 8. one result folder, with the three rankings and the per query numbers as raw files
    raw_files = []
    for name, lines in run_lines.items():
        path = tmp_path / f"{name}.txt"
        # unix line endings, the checksum in result.json must survive a commit on windows
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        raw_files.append(path)
    per_query_path = tmp_path / "per_query.jsonl"
    per_query_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_query),
        encoding="utf-8",
        newline="\n",
    )
    raw_files.append(per_query_path)
    folder = write_result(
        RESULTS,
        "g10-bm25-oracle",
        command="make oracle",
        environment_kind="local",
        dataset_directory=DATASET,
        model_version=settings.model_version,
        config={
            "k1": k1,
            "b": b,
            "avg_len_production": production_average,
            "depth": DEPTH,
            "overlap_k": OVERLAP_K,
            "queries": len(queries),
        },
        metrics=metrics,
        clock=lambda: datetime.now(UTC),
        raw_files=raw_files,
    )
    print(f"\nresult written to {folder}")
    for name, value in sorted(metrics.items()):
        print(f"  {name}: {value:.4g}")

    # the gate comes after the write, so a failing run still leaves its numbers to look at
    assert metrics["production_score_gap_max"] <= MAX_SCORE_GAP, (
        "our scores differ from the reference bm25 with the assumed length, see " + str(folder)
    )
