#!/usr/bin/env bash
# Download the National Burned Area Composite (NRCan CWFIS datamart, public) for the fire years 2020-2025 (2025 optional) and unzip them.
# Source: https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/ (release 20260513, read on 12 Oct 2026). Usage: bash fetch_nbac.sh [target_dir]
set -euo pipefail
D="${1:-nbac}"; mkdir -p "$D"; cd "$D"
BASE=https://cwfis.cfs.nrcan.gc.ca/downloads/nbac
for Y in 2020 2021 2022 2023 2024 2025; do
  F="NBAC_${Y}_20260513.zip"
  if [ ! -f "$F" ]; then
    wget -q --show-progress "$BASE/$F" || { echo "NBAC $Y not available on the server (fine for 2025: only used to mask next-season fires)"; rm -f "$F"; continue; }
  fi
  mkdir -p "NBAC_${Y}" && unzip -oq "$F" -d "NBAC_${Y}"
done
[ -f NBAC_1972to2025_20260513_shp_metadata.pdf ] || wget -q "$BASE/NBAC_1972to2025_20260513_shp_metadata.pdf"
ls -R | grep -E '\.shp$'
echo "NBAC ready in $(pwd)"
