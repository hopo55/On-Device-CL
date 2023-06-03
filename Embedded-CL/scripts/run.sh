#!/usr/bin/env bash
SAVE_DIR=results
FEATURES_DIR=features
IN_MEMORY=0
NUM_WORKERS=8
DATASET=places_lt
LR=0.001
POOL='avg'

MODEL='resnet18'
PERMUTATION_SEED=0
CL_MODEL=ncm
DATA_ORDER=class_iid

echo "Model: ${MODEL}; Dataset: ${DATASET}; Pool: ${POOL}"

CACHE=${FEATURES_DIR}/${DATASET}/supervised_${MODEL}_${DATASET}_${POOL}

EXPT_NAME=streaming_${CL_MODEL}_LR_${LR}_${MODEL}_${DATASET}_${POOL}_${DATA_ORDER}_seed_${PERMUTATION_SEED}

LOG_FILE=${SAVE_DIR}/logs/${EXPT_NAME}.log

python -u streaming_places_experiment.py \
  --arch ${MODEL} \
  --cl_model ${CL_MODEL} \
  --dataset ${DATASET} \
  --h5_features_dir ${CACHE} \
  --expt_name ${EXPT_NAME} \
  --save_dir ${SAVE_DIR}${EXPT_NAME} \
  --data_ordering ${DATA_ORDER} \
  --dataset_in_memory ${IN_MEMORY} \
  --num_workers ${NUM_WORKERS} \
  --permutation_seed ${PERMUTATION_SEED} >${LOG_FILE}

