.PHONY: help install format lint test run-api run-ui docker-up docker-down clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies using uv sync
	uv sync

format: ## Format code using ruff
	uv run ruff format .

lint: ## Lint code and fix issues using ruff
	uv run ruff check --fix .

test: ## Run the test suite with coverage
	uv run pytest --cov=src -v

run-api: ## Run the FastAPI backend server locally
	uv run uvicorn src.main:app --reload --port 8000

run-ui: ## Run the Streamlit frontend locally
	PYTHONPATH=. uv run streamlit run ui/app.py

docker-up: ## Start the full stack using Docker Compose
	docker compose up --build -d

docker-db-up: ## Start only the database container (pgvector) using Docker Compose
	docker compose up db -d

docker-down: ## Stop the full stack Docker Compose environment
	docker compose down

docker-test: ## Run the test suite natively inside a Docker container
	docker build --target tester -t pdf-rag-test .
	docker run --rm \
		-e LLM_API_KEY="test" \
		-e LLM_BASE_URL="http://test" \
		-e LLM_PROVIDER="ollama" \
		-e LLM_MODEL_NAME="test" \
		-e LLM_TEMPERATURE="0.0" \
		-e LLM_EMBEDDING_PROVIDER="ollama" \
		-e LLM_EMBEDDING_MODEL="test" \
		-e DATABASE_URL="postgresql+asyncpg://test:test@test/test" \
		-e DATABASE_COLLECTION_NAME="test" \
		pdf-rag-test

clean: ## Remove standard python artifacts (pycache, pytest_cache, coverage, etc)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage
