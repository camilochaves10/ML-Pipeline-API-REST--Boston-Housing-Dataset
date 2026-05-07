install:
	uv sync

train:
	uv run python -m src.train

api:
	uv run uvicorn app.main:app --reload

test:
	uv run pytest

lint:
	uv run ruff check .

docker-build:
	docker build -t housing-api .

docker-run:
	docker run -p 8000:8000 housing-api