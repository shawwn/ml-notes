#set -ex
#TPU_HOST=10.255.0.2 TPU_NAME=tpu-v2-8-usc1f-0 PYTHONSTARTUP=wrapper.py python3 "$@"

set -ex
#TPU_HOST=35.239.110.222 NUM_CORES=8 TPU_NAME='grpc://35.239.110.222:48000' PYTHONPATH=src PYTHONSTARTUP=wrapper.py exec python3 "$@"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/tfk/lib}"
export TPU_HOST=10.255.0.2
export NUM_CORES=8
#export TPU_NAME="grpc://${TPU_HOST}:48000"
export TPU_NAME=tpu-v2-8-usc1f-0
export PYTHONPATH=src
if [ ! -z "$*" ]
then
  exec python3 wrapper.py "$@"
else
  PYTHONSTARTUP=wrapper.py exec python3pdb "$@"
fi

