#!/usr/bin/env bash
#!/usr/bin/env bash
set -e

LOGDIR=$(pwd)/logs/fusion_50k_imgsize64/
rm -rf $LOGDIR

echo "Training..."
PYTHONPATH=. python catalyst/dl/scripts/train.py \
   --model-dir=. \
   --config=train_lstm.yml \
   --logdir=$LOGDIR --verbose