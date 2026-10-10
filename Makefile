.DEFAULT_GOAL := help
COMPOSE := docker compose -f deploy/compose/docker-compose.yml
UV := uv run

.PHONY: help setup up down test test-integration test-spark lint typecheck eval bench chaos deploy data

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n",$$1,$$2}'

setup:           ## install Python 3.12 deps (uv)
	uv sync --group ml --group dev

up:              ## start Redpanda, Qdrant, Redis, Postgres, Moto (S3)
	$(COMPOSE) up -d --wait

down:            ## stop and delete local stack volumes
	$(COMPOSE) down -v

lint:            ## ruff
	$(UV) ruff check . && $(UV) ruff format --check .

typecheck:       ## mypy --strict
	$(UV) mypy

test: lint typecheck  ## fast tests (unit + property + contract), no Docker
	$(UV) pytest

test-integration: ## Testcontainers: real Redpanda + Qdrant + Redis
	$(UV) pytest -m integration -o addopts=""

test-spark:      ## Spark tests inside Docker (Windows hosts lack winutils)
	docker build -f deploy/compose/spark-test.Dockerfile -t sp-spark-test . && docker run --rm sp-spark-test

data:            ## download ESCI into ./data (never committed; FiQA is an optional later check)
	$(UV) python scripts/download_data.py

eval:            ## retrieval ladder + ranking + RAG eval, writes benchmarks/results/
	$(UV) python -m eval.run --all

bench:           ## load-test scenarios, writes benchmarks/results/
	$(UV) python -m loadtest.run --all

chaos:           ## fault-injection suite through Toxiproxy
	$(COMPOSE) --profile chaos up -d --wait && $(UV) pytest -m failure -o addopts=""

deploy:          ## terraform apply + post-deploy smoke test
	cd deploy/terraform && terraform apply && cd ../.. && $(UV) python scripts/smoke_test.py
