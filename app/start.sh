#!/bin/bash
bash backend/download_models.sh
uvicorn backend.app.main:app --host 0.0.0.0 --port 10000