FROM node:20-alpine AS frontend
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    TEMP_DIR=/app/data/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY app ./app
COPY samples ./samples
COPY run.py wsgi.py ./
COPY --from=frontend /ui/dist ./frontend/dist
RUN mkdir -p /app/data/jobs /app/data/cache /app/data/tmp

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=8s --retries=3 CMD curl -fsS http://127.0.0.1:5000/api/health >/dev/null || exit 1
CMD ["python", "run.py"]
