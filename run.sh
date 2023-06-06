#!/usr/bin/env bash

DATASET=CIFAR100
CACHE_PATH=features
DATASET_PATH=data
POOL='avg'
BATCH_SIZE=1024
DATA_ORDER=class_iid
NUM_CLASSES=100
VAL_INCRE=1
IN_MEMORY=0
NUM_WORKERS=8
PERMUTATION_SEED=0
SAVE_DIR=results/

for MODEL in mobilenet_v3_small mobilenet_v3_large efficientnet_b0 efficientnet_b1 resnet18; do
  CACHE=${CACHE_PATH}/${DATASET}/${MODEL}_${POOL}

  python -u cache_features.py \
    --arch ${MODEL} \
    --dataset ${DATASET} \
    --cache_h5_dir ${CACHE} \
    --images_dir ${DATASET_PATH} \
    --pooling_type ${POOL} \
    --batch_size ${BATCH_SIZE}

  for CL_MODEL in ncm nb slda replay fine_tune ovr perceptron; do
    EXPT_NAME=${CL_MODEL}_${MODEL}_${DATASET}_${POOL}_${DATA_ORDER}

    python -u main.py \
      --arch ${MODEL} \
      --cl_model ${CL_MODEL} \
      --dataset ${DATASET} \
      --num_classes ${NUM_CLASSES} \
      --evaluate_increment ${VAL_INCRE} \
      --h5_features_dir ${CACHE} \
      --expt_name ${EXPT_NAME} \
      --save_dir ${SAVE_DIR}${EXPT_NAME} \
      --data_ordering ${DATA_ORDER} \
      --dataset_in_memory ${IN_MEMORY} \
      --num_workers ${NUM_WORKERS} \
      --permutation_seed ${PERMUTATION_SEED}

  done
done