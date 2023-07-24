#!/usr/bin/env bash

# Dataset List = ['CIFAR10', 'CIFAR100', 'CUB200', 'TinyImageNet'] # TinyImageNet(200)
DATASET=CIFAR10
CACHE_PATH=Feature-CL/features
DATASET_PATH=data
POOL='avg'
BATCH_SIZE=512
DATA_ORDER=class_iid
NUM_CLASSES=10
CLASS_INCRE=1
IN_MEMORY=0
NUM_WORKERS=8
PERMUTATION_SEED=0
SAVE_DIR=Feature-CL/results
DEVICE='1'

# ['resnet18', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b1']
# torchvision.models = [alexnet, convnext, densenet, efficientnet, googlenet, inception, mnasnet, mobilenet, regnet, resnet, shufflenetv2, squeezenet, vgg]
MODEL='resnet18 mobilenet_v3_large mobilenet_v3_small efficientnet_b0 efficientnet_b1'
MODEL_NAME='resnet18_mobilenet_v3_large_mobilenet_v3_small_efficientnet_b0_efficientnet_b1'
CACHE=${CACHE_PATH}/${DATASET}/

# Extract Features
python -u Feature-CL/cache_features.py \
  --arch ${MODEL} \
  --dataset ${DATASET} \
  --cache_h5_dir ${CACHE} \
  --images_dir ${DATASET_PATH} \
  --pooling_type ${POOL} \
  --num_workers ${NUM_WORKERS} \
  --batch_size ${BATCH_SIZE} \
  --device ${DEVICE}

# Ensemble Model
CL_MODEL=ncm
EXPT_NAME=${DATASET}/${CL_MODEL}/

python -u Feature-CL/main.py \
  --arch ${MODEL} \
  --cl_model ${CL_MODEL} \
  --dataset ${DATASET} \
  --batch_size ${BATCH_SIZE} \
  --num_classes ${NUM_CLASSES} \
  --class_increment ${CLASS_INCRE} \
  --h5_features_dir ${CACHE} \
  --expt_name ${EXPT_NAME} \
  --save_dir ${SAVE_DIR}/${EXPT_NAME} \
  --data_ordering ${DATA_ORDER} \
  --num_workers ${NUM_WORKERS} \
  --permutation_seed ${PERMUTATION_SEED} \
  --device ${DEVICE}

  # --dataset_in_memory ${IN_MEMORY} \
