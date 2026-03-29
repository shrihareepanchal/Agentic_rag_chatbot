#!/bin/bash
export WATCHFILES_FORCE_POLLING=false
uvicorn backend.main:app \
  --reload \
  --reload-dir backend \
  --reload-dir frontend \
  --port 8000
