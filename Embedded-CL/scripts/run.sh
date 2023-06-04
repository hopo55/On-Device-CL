#!/usr/bin/env bash
SAVE_DIR=results
IN_MEMORY=0
NUM_WORKERS=8
DATASET=CIFAR10
LR=0.001
POOL='avg'

MODEL='resnet18'
PERMUTATION_SEED=0
CL_MODEL=ncm
DATA_ORDER=class_iid

echo "Model: ${MODEL}; Dataset: ${DATASET}; Pool: ${POOL}"

EXPT_NAME=streaming_${CL_MODEL}_${MODEL}_${DATASET}_${POOL}_${DATA_ORDER}

LOG_FILE=${SAVE_DIR}/logs/${EXPT_NAME}.log

python -u main.py \
  --arch ${MODEL} \
  --cl_model ${CL_MODEL} \
  --dataset ${DATASET} \
  --expt_name ${EXPT_NAME} \
  --save_dir ${SAVE_DIR}${EXPT_NAME} \
  --data_ordering ${DATA_ORDER} \
  --dataset_in_memory ${IN_MEMORY} \
  --num_workers ${NUM_WORKERS} \
  --permutation_seed ${PERMUTATION_SEED}

