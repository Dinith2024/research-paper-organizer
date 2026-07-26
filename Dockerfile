FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN mkdir -p /app/data/chroma \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
