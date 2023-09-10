#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
CONFIG=$3
NODES=$4
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

module switch compiler/rocm/2.9 compiler/rocm/4.0.1
module load apps/anaconda3/5.2.0

export PATH=/public/home/zhangsongy/.conda/envs/mmcv_env/bin:$PATH
export LD_LIBRARY_PATH=/public/home/zhangsongy/.conda/envs/mmcv_env/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=dcu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --nodes=${NODES} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
