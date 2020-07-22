#!/bin/bash
set -ex
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
export TPU_HOST=${TPU_HOST:-10.255.128.2}
export TPU_NAME="${TPU_NAME:-tpu-v3-128-euw4a-50}"

export RUN_NAME="${RUN_NAME:-run0-117m-tensorflow}"
tmux-set-title "${RUN_NAME} ${TPU_NAME}"
export MODEL_DIR="${MODEL_DIR:-gs://dota-euw4a/runs/gpt-2/${RUN_NAME}/}"
export GIN_CONFIG="cfg/${RUN_NAME}.gin"

date="$(python3 -c 'import datetime; print(datetime.datetime.now().strftime("%Y-%m-%d"))')"
logfile="logs/${RUN_NAME}-${date}.txt"
mkdir -p logs

export TPU_SPLIT_COMPILE_AND_EXECUTE=1
export TF_TPU_WATCHDOG_TIMEOUT=1800

cores="$(echo $TPU_NAME | sed 's/^tpu-v[23][-]\([0-9]*\).*$/\1/g')"
if [ -z "$cores" ]
then
  1>&2 echo "Failed to parse TPU core count from $TPU_NAME"
  exit 1
fi
export TPU_CORES=$cores

if [ ! -z "${DEV}" ]
then
  exec python3 -m pdb -c continue wrapper.py main_gpt2.py --tpu "${tpu}" --model_dir "${model_dir}" ${restore_dir} --params "${params}" --num_cores "${TPU_CORES}" ${dataset} "$@"
  exit -1
fi

while true; do
  timeout --signal=SIGKILL 19h python3 wrapper.py main_gpt2.py --tpu "${tpu}" --model_dir "${model_dir}" ${restore_dir} --params "${params}" --num_cores "${TPU_CORES}" ${dataset} "$@" | tee -a "${logfle}"
  if [ ! -z "$TPU_NO_RECREATE" ]
  then
    echo "Not recreating TPU. Waiting 120s."
    sleep 120
  else
    echo "Recreating TPU in 120."
    sleep 120
    # sudo pip3 install -U tpudiepie
    pu recreate "$TPU_NAME" --yes
  fi
done
