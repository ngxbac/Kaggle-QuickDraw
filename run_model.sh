
#!/usr/bin/env bash
set -e

LOGDIR=$(pwd)/logs/se_resnet50/

echo "Training..."
PYTHONPATH=. python catalyst/dl/scripts/train.py \
   --model-dir=. \
   --config=train.yml \
   --logdir=$LOGDIR --verbose
