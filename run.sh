#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT
streamlit run frontend/app.py --server.port 8501 --server.headless true
