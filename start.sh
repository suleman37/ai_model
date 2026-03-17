#!/usr/bin/env sh
set -eu

exec uvicorn FASTAPI.app:app --host 0.0.0.0 --port "${PORT:-8000}"
