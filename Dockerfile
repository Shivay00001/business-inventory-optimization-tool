# Inventory Optimization API — production image
FROM python:3.11-slim

WORKDIR /app

# Install Python deps in their own layer (rebuild only when they change)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime config comes only from env vars (see .env.example) — no secrets baked in.
# API_KEY must be provided at runtime; without it the container starts with the
# dev placeholder and logs a warning (auth still enforced on protected routes).
ENV PORT=8000 \
    LOG_LEVEL=info \
    CORS_ALLOWED_ORIGINS=""

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT}"]
