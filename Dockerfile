# AuditLens API image. Build: docker build -t auditlens:latest .
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 HF_HOME=/app/.cache/hf
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA wheel
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install ".[genai,monitoring]"

COPY configs ./configs
COPY docs/policies ./docs/policies
COPY eval ./eval
COPY sql ./sql
COPY dashboard ./dashboard
# Trained bundle is baked in when present (CI trains before building); otherwise mount it.
COPY artifacts ./artifacts

RUN useradd -m -u 1000 app && mkdir -p logs reports artifacts && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "auditlens.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
