# ============================================================
# Coin Trading Bot — Multi-stage Docker build
# ============================================================

# Stage 1: Frontend build
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run build

# Stage 2: Python application
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy built frontend assets
COPY --from=frontend-builder /app/src/bot/dashboard/static/ src/bot/dashboard/static/

# Create data directory for SQLite
RUN mkdir -p /app/data

# Create non-root user
RUN groupadd -r botuser && useradd -r -g botuser -d /app -s /sbin/nologin botuser \
    && chown -R botuser:botuser /app
USER botuser

# Default environment variables
ENV TRADING_MODE=paper \
    TRADING_ENV=production \
    LOG_LEVEL=INFO \
    DATABASE_URL=sqlite+aiosqlite:///data/trading.db \
    DASHBOARD_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "-m", "bot.main"]
