#!/usr/bin/env bash
set -e

# Define the log folder which saves the pretrained models
LOGDIR=$(pwd)/logs/se_resnext101_50k_2/
echo "Inference..."
PYTHONPATH=. python catalyst/dl/scripts/inference.py \
   --model-dir=. \
   --resume=$LOGDIR/checkpoint.best.pth.tar \
   --out-prefix=$LOGDIR/dataset.predictions.{suffix}.npy \
   --config=$LOGDIR/config.json,inference.yml \
   --verbose
