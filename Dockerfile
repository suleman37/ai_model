FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    APP_HOME=/app \
    MODEL_PATH=/app/FASTAPI/best.pt \
    PORT=8000

WORKDIR ${APP_HOME}

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir /home/appuser appuser

COPY FASTAPI/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --retries 10 -r /tmp/requirements.txt

COPY docker ${APP_HOME}/docker
COPY FASTAPI ${APP_HOME}/FASTAPI
COPY fastapi_module_2 ${APP_HOME}/fastapi_module_2
COPY API_updated ${APP_HOME}/API_updated
COPY main.py ${APP_HOME}/main.py
COPY start.sh ${APP_HOME}/start.sh

RUN chmod +x ${APP_HOME}/start.sh \
    && mkdir -p /tmp \
    && chown -R appuser:appuser ${APP_HOME} /tmp /home/appuser

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["sh", "-c", "python /app/docker/healthcheck.py http://127.0.0.1:${PORT:-8000}/health"]

CMD ["/app/start.sh"]
