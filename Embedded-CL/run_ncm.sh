#!/usr/bin/env bash

# Dataset List = ['MNIST', 'CIFAR10', 'CIFAR100']
# Update Dataset List = ['places', 'imagenet', 'places_lt']
DATASET=CIFAR100
CACHE_PATH=features
DATASET_PATH=data
POOL='avg'
BATCH_SIZE=512
DATA_ORDER=class_iid
NUM_CLASSES=100
CLASS_INCRE=10
IN_MEMORY=0
NUM_WORKERS=8
PERMUTATION_SEED=0
SAVE_DIR=results/

# (Jetson) torch 1.8 does not support 'efficientnet_b0', 'efficientnet_b1'
MODEL=resnet18
CACHE=${CACHE_PATH}/${DATASET}/${MODEL}_${POOL}

python3 -u cache_features.py \
  --arch ${MODEL} \
  --dataset ${DATASET} \
  --cache_h5_dir ${CACHE} \
  --images_dir ${DATASET_PATH} \
  --pooling_type ${POOL} \
  --num_workers ${NUM_WORKERS} \
  --batch_size ${BATCH_SIZE}

CL_MODEL=ncm
EXPT_NAME=${DATASET}/${CL_MODEL}/${MODEL}_${POOL}_${DATA_ORDER}

python3 -u main.py \
  --arch ${MODEL} \
  --cl_model ${CL_MODEL} \
  --dataset ${DATASET} \
  --batch_size ${BATCH_SIZE} \
  --num_classes ${NUM_CLASSES} \
  --class_increment ${CLASS_INCRE} \
  --h5_features_dir ${CACHE} \
  --expt_name ${EXPT_NAME} \
  --save_dir ${SAVE_DIR}${EXPT_NAME} \
  --data_ordering ${DATA_ORDER} \
  --dataset_in_memory ${IN_MEMORY} \
  --num_workers ${NUM_WORKERS} \
  --permutation_seed ${PERMUTATION_SEED}