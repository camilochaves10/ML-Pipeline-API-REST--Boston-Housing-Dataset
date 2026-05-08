#!/bin/bash

set -e

echo "Starting retraining job..."

uv run python -m src.train

echo "Retraining completed."