FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /app appuser \
    && mkdir -p /app/data \
    && chown appuser:appuser /app/data

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"]

CMD ["sh", "-c", "python -I scripts/check_runtime_env.py && exec python -I -u -B -m uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8080}\" --no-proxy-headers --workers 1 --lifespan on --timeout-graceful-shutdown 8"]
