#!/bin/sh
source "${HOME}/bin/activate-tf1"
set -x
if [ -z $TPU_HOST ]
then
  1>&2 echo "Set \$TPU_HOST"
  exit 1
fi

model_dir=gs://danbooru-euw4a/checkpoint/test117m-0
restore_dir="${model_dir}"

exec python3 001_sharing.py --tpu tpu-euw4a-69 --model_dir "${model_dir}" --restore_dir "${restore_dir}" "$@"
