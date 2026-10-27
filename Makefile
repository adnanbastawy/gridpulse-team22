.PHONY: install test api ui up down mlflow sync

install:            ## local dev environment
	python -m pip install -r requirements/dev.txt

test:               ## parity with the notebook, leakage, detection and API tests
	python -m pytest -q tests

api:                ## API on http://localhost:8000 (docs at /docs)
	PYTHONPATH=src:. uvicorn api.main:app --reload --port 8000

ui:                 ## Streamlit app on http://localhost:8501 (needs the API)
	GRIDPULSE_API_URL=http://localhost:8000 streamlit run ui/app.py

up:                 ## both services in Docker
	docker compose up --build -d

mlflow:             ## also the MLflow UI on http://localhost:5000
	docker compose --profile mlops up --build -d

down:
	docker compose --profile mlops down

sync:               ## make sync NB_OUT=/path/to/gridpulse_outputs
	python scripts/sync_artifacts.py $(NB_OUT)
