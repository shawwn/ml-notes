set -ex
# export PRECISION=bfloat16
# export MODEL=GPT2
# #export N_EMBD=1536
# export N_EMBD=768
# DEV=1 RUN_ID=e TPU_CORES=8 TPU_NAME=grpc://35.239.110.222:48004 BATCH_PER_CORE=8 exec bash runs/train-rev-117m.sh  "$@"

#export PRECISION=bfloat16
export PRECISION=float32
export ITERATIONS=10
#export BATCH_PER_CORE=8
#export BATCH_PER_CORE=1
export BATCH_PER_CORE=16
export N_EMBD=1536 
export RUN_ID=e3
export DATASET=gs://tpu-usc1/datasets/books3.tok16
#export LR=0.000025 # diverged?!
export LR=0.00025
#DEV=1 TPU_CORES=8 TPU_NAME=grpc://35.239.110.222:48004 exec bash runs/train-rev-117m.sh  "$@"
#DEV=1 TPU_CORES=8 TPU_NAME=grpc://104.197.254.29:48200 exec bash runs/train-rev-117m.sh  "$@"
export WRAPPER=" "
export TPU_CORES=8
export TPU_NAME=tpu-v3-8-usc1a-200
exec bash runs/train-rev-117m.sh  "$@"

