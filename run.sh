#!/bin/sh
source "${HOME}/bin/activate-tf1"
set -x
if [ -z $TPU_HOST ]
then
  1>&2 echo "Set \$TPU_HOST"
  exit 1
fi
exec python3 001_sharing.py --tpu tpu-euw4a-69 --model_dir gs://danbooru-euw4a/checkpoint/test117m-0/ "$@"
