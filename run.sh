#!/bin/sh
source "${HOME}/bin/activate-tf1"
set -x
if [ -z $TPU_HOST ]
then
  1>&2 echo "Set \$TPU_HOST"
  exit 1
fi

export TPU_CORES=8
params=117M.json
model_dir=gs://danbooru-euw4a/checkpoint/test117m-0
tpu=tpu-euw4a-69

params=1.5B.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-0
tpu=tpu-euw4a-69

params=1558M.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-1
tpu=tpu-euw4a-68

params=1558M.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-2
tpu=tpu-euw4a-67

params=117M_memory_saving_gradients.json
model_dir=gs://danbooru-euw4a/checkpoint/test117m-1
tpu=tpu-euw4a-66

params=117M.json
model_dir=gs://danbooru-euw4a/checkpoint/test117m-2
tpu=tpu-euw4a-65

params=1558M.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-3
tpu=tpu-euw4a-65

params=1558M.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-65
tpu=tpu-euw4a-65
#export TPU_CORES=2

params=1.5B.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-69
tpu=tpu-euw4a-69

params=1.5B_adam.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-67
tpu=tpu-euw4a-67

params=1.5B_adam.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-65
tpu=tpu-euw4a-65

params=1.5B_adam.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-70
tpu=tpu-euw4a-70

params=1.5B_adam.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-71
tpu=tpu-euw4a-71
restore_dir=gs://gpt-2/models/1558M

params=1.5B_adam.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-78
tpu=tpu-euw4a-78
restore_dir=gs://danbooru-euw4a/models/1558M

params=1.5B.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-76
tpu=tpu-euw4a-76
#restore_dir=gs://danbooru-euw4a/models/1558M
unset restore_dir

params=1.5B.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-69
tpu=tpu-euw4a-69
restore_dir=gs://danbooru-euw4a/models/1558M
#unset restore_dir
#gsutil -m rm -rf "${model_dir}"

params=1.5B.json
model_dir=gs://danbooru-euw4a/checkpoint/test1558m-77
tpu=tpu-euw4a-77
restore_dir=gs://danbooru-euw4a/models/1558M
unset restore_dir

params=117M.json
model_dir=gs://danbooru-euw4a/checkpoint/test117m-71-2
tpu=tpu-euw4a-71
restore_dir=gs://danbooru-euw4a/models/117M
dataset="--dataset combined-pgpf-ftfy.txt.npz"
#unset restore_dir

params=117M.json
model_dir=gs://danbooru-euw4a/checkpoint/test117m-76
tpu=tpu-euw4a-76
#restore_dir=gs://danbooru-euw4a/models/117M
#restore_trainable="--restore_trainable_variables true"
restore_dir="${model_dir}"
dataset="--dataset combined-pgpf-ftfy.txt.npz"
#unset restore_dir
#gsutil -m rm -rf "${model_dir}"

if [ ! -z "$restore_dir" ]
then
  restore_dir="--restore_dir ${restore_dir} ${restore_trainable}"
fi
#exec python3 001_sharing.py --tpu "${tpu}" --model_dir "${model_dir}" --restore_dir "${restore_dir}" --params "${params}" "$@"
exec python3 -m pdb -c continue main_gpt2.py --tpu "${tpu}" --model_dir "${model_dir}" ${restore_dir} --params "${params}" --num_cores "${TPU_CORES}" ${dataset} "$@"
