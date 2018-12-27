
#!/usr/bin/env bash
set -e

# Define the log folder which saves the pretrained models
LOGDIR=$(pwd)/logs/inception_v4/
# rm -rf $LOGDIR

echo "Inference..."


# List 5 best checkpoints of model
for best_checkpoint in 1 2 3 4 5
#for best_checkpoint in 1
do
echo "Inference..."
PYTHONPATH=. python catalyst/dl/scripts/inference.py \
   --model-dir=. \
   --resume=$LOGDIR/checkpoint.stage1.$best_checkpoint.pth.tar \
   --out-prefix=$LOGDIR/dataset.predictions.{suffix}.satge1.$best_checkpoint.npy \
   --config=$LOGDIR/config.json,inference.yml \
   --verbose
 done

# docker trick
if [ "$EUID" -eq 0 ]; then
  chmod -R 777 ${LOGDIR}
fi
