#!/bin/bash
set -ex
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
export TPU_HOST=${TPU_HOST:-10.254.128.2}
export TPU_NAME="${TPU_NAME:-tpu-v2-512-usc1a-2}"

export RUN_ID="${RUN_ID:-c}"
export RUN_NAME="${RUN_NAME:-revnet01}"
export RUN_DESC="${RUN_DESC:-117M revnet run}"
tmux-set-title "${RUN_NAME}/${RUN_ID} ${TPU_NAME}"
export MODEL_DIR="${MODEL_DIR:-gs://tpu-usc1/runs/gpt-2/${RUN_NAME}/${RUN_ID}/}"
export MODEL_DIR="$(printf '%s' "${MODEL_DIR}" | sed 's/\/$//')" # normalize model dir; ensure it does *not* end with a slash
export GIN_CONFIG="cfg/${RUN_NAME}.gin"


export MODEL="${MODEL:-GPT2Rev}"
export MODEL_NAME="${MODEL_NAME:-117M}"
export DATASET="${DATASET:-gs://tpu-usc1/datasets/novels.tok16}"
export RESTORE_DIR="${RESTORE_DIR:-gs://tpu-usc1/models/gpt-2/${MODEL_NAME}}"

export WRAPPER="${WRAPPER:-wrapper.py}"



date="$(python3 -c 'import datetime; print(datetime.datetime.now().strftime("%Y-%m-%d-%H"))')"
logfile="logs/${RUN_NAME}-${RUN_ID}-${date}.txt"
cloud_log_file="${MODEL_DIR}/logs-${date}-${RUN_NAME}-${RUN_ID}.txt"
cloud_description_file="${MODEL_DIR}/description.txt"
mkdir -p logs

export DATASET="--dataset ${DATASET}"
#export RESTORE_DIR="--restore_dir ${RESTORE_DIR} --restore_trainable_variables true"
export RESTORE_DIR="--restore_dir ${MODEL_DIR} --restore_trainable_variables true"
export RUN_DESC="
name: ${RUN_NAME}/${RUN_ID}
date: ${date}
tpu: ${TPU_NAME}
model_dir: ${MODEL_DIR}
dataset: ${DATASET}
model_name: ${MODEL_NAME}

${RUN_DESC}"

printf "%s" "${RUN_DESC}"

#pu list -s -t $TPU_NAME | sed 's/\x1b\[[0-9;]*m//g'


export TPU_SPLIT_COMPILE_AND_EXECUTE=1
export TF_TPU_WATCHDOG_TIMEOUT=1800

if [ -z "$TPU_CORES" ]
then
  cores="$(echo $TPU_NAME | sed 's/^tpu-v[23][-]\([0-9]*\).*$/\1/g')"
  if [ -z "$cores" ]
  then
    1>&2 echo "Failed to parse TPU core count from $TPU_NAME"
    exit 1
  fi
  export TPU_CORES=$cores
fi


if [ ! -z "${DEV}" ]
then
  exec python3 -m pdb -c continue $WRAPPER main_gpt2.py --tpu "${TPU_NAME}" --model_dir "${MODEL_DIR}" ${RESTORE_DIR} --params "${MODEL_NAME}.json" --num_cores "${TPU_CORES}" ${DATASET} "$@"
  exit -1
fi


while true; do
  echo "Saving description to ${cloud_description_file} ..."
  printf "%s" "${RUN_DESC}" | gsutil cp - "${cloud_description_file}"

  echo "Starting production training run in 10s ..."
  sleep 10

  timeout --signal=SIGKILL 4h python3 $WRAPPER main_gpt2.py --tpu "${TPU_NAME}" --model_dir "${MODEL_DIR}" ${RESTORE_DIR} --params "${MODEL_NAME}.json" --num_cores "${TPU_CORES}" ${DATASET} "$@" 2>&1 | tee -a "${logfile}" | tee /dev/fd/2 | gsutil cp - "${cloud_log_file}"
  if [ ! -z "$TPU_NO_RECREATE" ]
  then
    echo "Not recreating TPU. Waiting 30s."
    sleep 30
  else
    echo "Recreating TPU in 30."
    sleep 30
    # sudo pip3 install -U tpudiepie
    pu recreate "$TPU_NAME" --yes --retry 300
  fi
done
