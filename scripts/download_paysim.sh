#!/usr/bin/env bash
# Download the real PaySim dataset (~470 MB) from Kaggle and load it into PostgreSQL.
# Needs a Kaggle API token: https://www.kaggle.com/settings -> API -> "Create New Token",
# then `export KAGGLE_API_TOKEN=KGAT_...` (older kaggle.json tokens in ~/.kaggle/ also work).
set -euo pipefail
pip install -q "kaggle>=1.8"
mkdir -p data/raw
kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip
mv data/raw/PS_*.csv data/raw/paysim.csv
echo "Saved data/raw/paysim.csv ($(wc -l < data/raw/paysim.csv) lines)"
if [[ "${1:-}" == "--load-db" ]]; then auditlens load-db; fi
