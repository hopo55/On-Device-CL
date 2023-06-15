#!/usr/bin/env bash

# Dataset List = ['MNIST', 'CIFAR10', 'CIFAR100']
DATASET=CIFAR10
CACHE_PATH=features
DATASET_PATH=data
POOL='avg'
BATCH_SIZE=1
DATA_ORDER=class_iid
NUM_CLASSES=10
CLASS_INCRE=1
IN_MEMORY=0
NUM_WORKERS=4
PERMUTATION_SEED=0
SAVE_DIR=results/
MODEL='resnet18,mobilenet_v3_large'
# MODEL='resnet18'
CL_MODEL=ncm

CACHE=${CACHE_PATH}/${DATASET}/${MODEL}_${POOL}
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