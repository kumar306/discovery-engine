.DEFAULT_GOAL := help
COMPOSE := docker compose -f deploy/compose/docker-compose.yml
UV := uv run

.PHONY: help setup up down topics init test test-integration test-crash test-spark lint typecheck eval bench chaos deploy data data-full load-mini load-full oracle build-index gate baseline

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n",$$1,$$2}'

setup:           ## install Python 3.12 deps (uv)
	uv sync --group ml --group dev

up:              ## start Redpanda, Qdrant, Redis, Postgres, PgBouncer, Moto (S3)
	$(COMPOSE) up -d --wait

down:            ## stop and delete local stack volumes
	$(COMPOSE) down -v

topics:          ## create the Kafka topics (doc.events with 24 partitions, and its dead letter topic)
	$(COMPOSE) exec -T redpanda sh -c 'rpk topic describe doc.events >/dev/null 2>&1 || rpk topic create doc.events -p 24 -r 1'
	$(COMPOSE) exec -T redpanda sh -c 'rpk topic describe doc.events.dlq >/dev/null 2>&1 || rpk topic create doc.events.dlq -p 24 -r 1'

init: up topics  ## one command from nothing to a usable dev stack, safe to repeat
	$(UV) python scripts/dev_init.py

lint:            ## ruff
	$(UV) ruff check . && $(UV) ruff format --check .

typecheck:       ## mypy --strict
	$(UV) mypy

test: lint typecheck  ## fast tests (unit + property + contract), no Docker
	$(UV) pytest

test-integration: ## tests that need the compose stack, run make up first, the slow crash tests are separate
	$(UV) pytest -m "integration and not crash" -o addopts=""

test-crash:      ## kill the worker at every call and in random histories, then check it recovers
	$(UV) pytest -m crash -o addopts=""

test-spark:      ## Spark tests inside Docker (Windows hosts lack winutils)
	docker build -f deploy/compose/spark-test.Dockerfile -t sp-spark-test . && docker run --rm sp-spark-test

data:            ## build the small ESCI set in ./data (never committed), reads only the test files, in pieces
	$(UV) python scripts/download_data.py --size mini

data-full:       ## build the 100 thousand product selection from all files, run at will
	$(UV) python scripts/download_data.py --size full

load-mini:       ## send the mini set through the running ingest api and wait until every product is indexed
	$(UV) python scripts/load_data.py --size mini

load-full:       ## the same for the 100 thousand product set, run at will
	$(UV) python scripts/load_data.py --size full

oracle:          ## compare our sparse search with a reference bm25 on the real index, needs make load-mini
	$(UV) pytest -m eval tests/eval/test_bm25_oracle.py -o addopts="" -s

build-index:     ## fill a Qdrant collection straight from a dataset (no Kafka, Postgres or S3), see eval/index_builder.py
	$(UV) python -m eval.index_builder

gate:            ## run the ladder, then fail if a rung lost more than 2 percent against benchmarks/baseline/ladder-mini.json
	$(UV) python -m eval.ladder && $(UV) python -m eval.gate check

baseline:        ## write a PROPOSED baseline from the newest ladder result, read it and commit it on purpose
	$(UV) python -m eval.gate propose

eval:            ## retrieval ladder on the loaded mini set, writes benchmarks/results/g6-ladder-mini
	$(UV) python -m eval.ladder

bench:           ## load-test scenarios, writes benchmarks/results/
	$(UV) python -m loadtest.run --all

chaos:           ## fault-injection suite through Toxiproxy
	$(COMPOSE) --profile chaos up -d --wait && $(UV) pytest -m failure -o addopts=""

deploy:          ## terraform apply + post-deploy smoke test
	cd deploy/terraform && terraform apply && cd ../.. && $(UV) python scripts/smoke_test.py
