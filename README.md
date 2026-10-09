# search-platform

Streaming ingestion, hybrid retrieval (dense + BM25), learned ranking, a RAG shopping assistant with
citations, and Spark analytics over Amazon ESCI product data.
Built on Qdrant, Redpanda, Postgres, S3, Redis, FastAPI, gRPC, LightGBM and PySpark. Deployed on AWS (EKS) with
Terraform, not running in production.

Status: design phase; no service code yet.

## Quick start

```bash
uv sync            # Python 3.12 + dependencies
uv run pytest      # fast tests, no Docker
make up            # Redpanda + Qdrant + Redis (needs Docker and GNU make)
```
