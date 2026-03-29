.PHONY: dev docker-up docker-down seed lint clean

# ── Local dev ──────────────────────────────────────────────────────────────
dev:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# ── Docker ─────────────────────────────────────────────────────────────────
docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f app

# ── Database seeding ───────────────────────────────────────────────────────
seed:
	python -m backend.ingestion.indexer

# ── Quality ────────────────────────────────────────────────────────────────
lint:
	ruff check backend/
	mypy backend/ --ignore-missing-imports

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	rm -rf data/ uploads/ __pycache__ .pytest_cache
	find . -name "*.pyc" -delete

