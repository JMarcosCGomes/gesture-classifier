#!/bin/bash
set -e

DATASETS=("landmarks" "angles" "distances" "combined")
#DATASETS=("landmarks" "distances" "combined")

MODELS=("linear" "onelayer" "twolayer")

#TRAIN_FRACS=(0.1 0.25 0.5 1.0)
TRAIN_FRACS=(0.1 0.5 1.0)

DROPOUTS=(0.0 0.2)
#DROPOUT=0.0

total=$((${#DATASETS[@]} * ${#MODELS[@]} * ${#TRAIN_FRACS[@]} * ${#DROPOUTS[@]}))
count=0

for train_frac in "${TRAIN_FRACS[@]}"; do
  for dropout in "${DROPOUTS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      for model in "${MODELS[@]}"; do
        count=$((count + 1))
        echo "=== [$count/$total] dataset=$dataset model=$model train_frac=$train_frac dropout=$dropout ==="
        python3 -m src.train --dataset "$dataset" --model "$model" --train-frac "$train_frac" --dropout "$dropout"
      done
    done
  done
done

echo "Todos os experimentos finalizados ($total runs)"