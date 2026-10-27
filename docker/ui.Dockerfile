# GridPulse dispatcher app (Streamlit). Talks to the API only (GRIDPULSE_API_URL).
ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRIDPULSE_API_URL=http://api:8000 HOME=/tmp
WORKDIR /app
COPY requirements/ui.txt requirements/ui.txt
RUN python -m pip install -r requirements/ui.txt
COPY .streamlit ./.streamlit
COPY ui ./ui

USER 10001
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status != 200)"]
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
