#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  SPACEIND // GREEN BUILDING DIGITAL TWIN PLATFORM"
echo "  Industrial Brutalist Telemetry, 3D BIM & Facility AI"
echo "========================================================"
echo ""

if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[ERROR] Python is not installed or not in PATH."
    exit 1
fi

PY_CMD=python
if command -v python3 &> /dev/null; then
    PY_CMD=python3
fi

echo "[INFO] Starting Green Building Digital Twin on http://127.0.0.1:8000 ..."
$PY_CMD server.py
