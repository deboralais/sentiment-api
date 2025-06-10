#!/bin/bash
bash ../download_models.sh
uvicorn app.main:app --host 0.0.0.0 --port 10000
