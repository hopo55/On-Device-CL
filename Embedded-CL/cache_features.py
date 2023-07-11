import os
import argparse
import h5py
import numpy as np
import json

import torch

from dataset_utils import *
from utils import get_backbone, makedirs


def get_data_loaders(args):
    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        train_loader, val_loader = load_cifar_dataset(args, batch_size=args.batch_size)
    elif args.dataset == 'CUB200':
        train_loader, val_loader = load_cub_dataset(args, batch_size=args.batch_size)
    elif args.dataset == 'TinyImageNet':
        train_loader, val_loader = load_tiny_imagenet_dataset(args, batch_size=args.batch_size)
    else:
        raise NotImplementedError
    return train_loader, val_loader


def make_h5_feature_file(dataset, model, loader, h5_file_full_path, data_type, feature_size, device):
    if os.path.exists(h5_file_full_path):
        # os.remove(h5_file_full_path)
        # print('removed old h5 file')
        print('file already exists')
        return
    h5_file = h5py.File(h5_file_full_path, 'w')

    # preset array sizes
    if dataset == 'TinyImageNet':
        num_train = 100000
        num_val = 10000
    elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
        num_train = 50000
        num_val = 10000
    elif dataset == 'CUB200':
        num_train = 9430
        num_val = 2358
    else:
        raise NotImplementedError

    if data_type == 'train':
        h5_file.create_dataset("features", shape=(num_train, feature_size), dtype=np.float32)
        h5_file.create_dataset("labels", shape=(num_train,), dtype=np.int64)
    elif data_type == 'val':
        h5_file.create_dataset("features", shape=(num_val, feature_size), dtype=np.float32)
        h5_file.create_dataset("labels", shape=(num_val,), dtype=np.int64)
    else:
        raise NotImplementedError

    with torch.no_grad():

        # switch to evaluate mode
        model.eval().to(device)
        start = 0

        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                images = images.to(device)
                cur_feats = model(images).cpu()
                cur_targets = target.cpu()
                B, D = cur_feats.shape

                h5_file['features'][start:start + B, :] = cur_feats.numpy()
                h5_file['labels'][start:start + B] = cur_targets.numpy()

                start += B
    h5_file.close()


def cache_features(args):
    args.device = 'cuda:' + args.device
    args.device = torch.device(args.device)

    print('\nmodel : ', args.arch)
    train_loader, val_loader = get_data_loaders(args)
    backbone, feature_size = get_backbone(args.arch, args.pooling_type)

    print('\ncaching val features...')
    make_h5_feature_file(args.dataset, backbone, val_loader, os.path.join(args.cache_h5_dir, 'val_features.h5'), 'val', feature_size, args.device)
    print('\ncaching train features...')
    make_h5_feature_file(args.dataset, backbone, train_loader, os.path.join(args.cache_h5_dir, 'train_features.h5'), 'train', feature_size, args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # directory parameters
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'CUB200', 'TinyImageNet'])
    parser.add_argument('--images_dir', type=str)  # path to images (folder with 'train' and 'val' for data)
    parser.add_argument('--cache_h5_dir', type=str, default=None)
    parser.add_argument('--lt_txt_file', type=str, default='/media/tyler/Data/datasets/Places-LT/Places_LT_%s.txt')

    # other parameters
    # (Jetson) torch 1.8 does not support 'efficientnet_b0', 'efficientnet_b1'
    parser.add_argument('--arch', type=str, choices=['resnet18', 'mobilenet_v3_small', 'mobilenet_v3_large'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--pooling_type', type=str, default='avg', choices=['avg', 'max'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='0') 
    parser.add_argument('--img_size', type=int, default=32)

    args = parser.parse_args()
    # print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    # if not os.path.exists(args.cache_h5_dir):
    #     os.mkdir(args.cache_h5_dir)
    makedirs(args.cache_h5_dir)

    cache_features(args)