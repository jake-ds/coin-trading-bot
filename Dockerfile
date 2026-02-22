# Stage 1: Frontend build
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run build

# Stage 2: Python dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY src/ src/
COPY pyproject.toml .

RUN pip install --no-cache-dir --prefix=/install -e .

# Stage 3: Runtime
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ src/
COPY pyproject.toml .

# Copy built frontend assets
COPY --from=frontend-builder /app/src/bot/dashboard/static/ src/bot/dashboard/static/

# Create data directory for SQLite
RUN mkdir -p /app/data

# Default environment variables
ENV TRADING_MODE=paper
ENV LOG_LEVEL=INFO
ENV DATABASE_URL=sqlite+aiosqlite:///data/trading.db
ENV DASHBOARD_PORT=8000
ENV NODE_ENV=production

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "bot.main"]
