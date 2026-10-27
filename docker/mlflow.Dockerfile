# MLflow UI over the tracking store written by the notebook (runs, Optuna studies, Model Registry).
ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements/mlflow.txt /tmp/mlflow.txt
RUN python -m pip install -r /tmp/mlflow.txt
EXPOSE 5000
# The notebook writes artifact paths under /content/gridpulse_outputs (Colab), so the store is mounted there.
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////content/gridpulse_outputs/mlflow.db", \
     "--default-artifact-root", "file:///content/gridpulse_outputs/mlartifacts", "--host", "0.0.0.0", "--port", "5000"]
