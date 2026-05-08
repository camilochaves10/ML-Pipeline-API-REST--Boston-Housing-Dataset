.PHONY: run run-api run-streamlit test run-docker-compose stop-docker-compose logs

run-api:
	uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

run-streamlit:
	streamlit run app/streamlit_app.py

test:
	pytest

run-docker-compose:
	docker compose up --build

stop-docker-compose:
	docker compose down

logs:
	docker compose logs -f