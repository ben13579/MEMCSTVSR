#!/usr/bin/env bash
set -e

# ====== default args ======
MODE="frames"
K=9

# ====== parse args ======
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --k)
      K="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "[INFO] mode=${MODE}, k=${K}"

SCRIPT=python
GEN=diffsynth/core/data/metadata_gennerator.py

ROOT=../Adobe240/frame

# ====== train ======
$SCRIPT $GEN \
  --base_dir ${ROOT}/train \
  --out_jsonl ${ROOT}/train/metadata.jsonl \
  --mode ${MODE} \
  --k ${K}

# ====== valid ======
$SCRIPT $GEN \
  --base_dir ${ROOT}/valid \
  --out_jsonl ${ROOT}/valid/metadata.jsonl \
  --mode ${MODE} \
  --k ${K}

# ====== test ======
$SCRIPT $GEN \
  --base_dir ${ROOT}/test \
  --out_jsonl ${ROOT}/test/metadata.jsonl \
  --mode ${MODE} \
  --k ${K}

echo "[DONE] metadata generation finished"
