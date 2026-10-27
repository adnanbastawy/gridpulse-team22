# GridPulse API: FastAPI + Uvicorn. Artifacts are baked in (versioned with the image) and verified at start-up.
ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app GRIDPULSE_ARTIFACTS=/app/artifacts
# LightGBM needs the OpenMP runtime (libgomp1). Skipped when the base image already has it.
# The Debian mirror is switched to HTTPS: some networks block plain HTTP (port 80) from containers.
RUN if ! ldconfig -p 2>/dev/null | grep -q "libgomp.so.1" && command -v apt-get >/dev/null 2>&1; then \
      sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
      apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*; \
    fi
WORKDIR /app
COPY requirements/api.txt requirements/api.txt
RUN python -m pip install -r requirements/api.txt
COPY src ./src
COPY api ./api
COPY artifacts ./artifacts

USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
