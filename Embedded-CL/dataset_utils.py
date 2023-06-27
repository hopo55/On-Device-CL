import os
from PIL import Image
import h5py
import numpy as np

import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from avalanche.benchmarks.datasets import CUB200, TinyImagenet


def load_cub_dataset(args, batch_size=256):
    img_size = args.img_size
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    data_path = args.images_dir + '/' + args.dataset

    trainset = CUB200(root=data_path, train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    valset = CUB200(root=data_path, train=False, download=True, transform=eval_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader

def load_tiny_imagenet_dataset(args, batch_size=256):
    img_size = args.img_size
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    data_path = args.images_dir + '/' + args.dataset

    trainset = TinyImagenet(root=data_path, train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    valset = TinyImagenet(root=data_path, train=False, download=True, transform=eval_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader
    
def load_cifar_dataset(args, batch_size=256):
    dataset_stats = {
        'CIFAR10' : {'mean': (0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
                    'std' : (0.2470322324632819, 0.24348512800005573, 0.26158784172796434),
                    'size' : 32},
        'CIFAR100': {'mean': (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                    'std' : (0.2673342858792409, 0.25643846291708816, 0.2761504713256834),
                    'size' : 32}
        }

    tarin_transform = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(dataset_stats[args.dataset]['mean'], dataset_stats[args.dataset]['std']),]
                )
    val_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(dataset_stats[args.dataset]['mean'], dataset_stats[args.dataset]['std']),]
                )
    
    data_path = args.images_dir + '/' + args.dataset

    if args.dataset == 'CIFAR10':
        trainset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=tarin_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        valset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        trainset = datasets.CIFAR100(root=data_path, train=True, download=True, transform=tarin_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

        valset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader

class FeaturesDatasetInMemory(Dataset):
    def __init__(self, h5_file_path, return_item_ix=False):
        super(FeaturesDatasetInMemory, self).__init__()
        self.h5_file_path = h5_file_path
        with h5py.File(self.h5_file_path, 'r') as h5:
            self.features = np.array(h5['features'])
            self.labels = np.array(h5['labels'], dtype=np.int64)
            self.dataset_len = len(self.features)

        self.return_item_ix = return_item_ix

    def __getitem__(self, index):

        if self.return_item_ix:
            return self.features[index], self.labels[index], int(index)
        else:
            return self.features[index], self.labels[index]

    def __len__(self):
        return self.dataset_len


class FeaturesDataset(Dataset):
    def __init__(self, h5_file_path, return_item_ix=False, transform=None):
        super(FeaturesDataset, self).__init__()
        self.h5_file_path = h5_file_path
        with h5py.File(self.h5_file_path, 'r') as h5:
            # keys = list(h5.keys())

            self.features_key = 'features'
            features = h5['features']
            self.dataset_len = len(features)

        self.return_item_ix = return_item_ix
        self.transform = transform

    def __getitem__(self, index):
        if not hasattr(self, 'features'):
            self.h5 = h5py.File(self.h5_file_path, 'r')
            self.features = self.h5[self.features_key]
            self.labels = self.h5['labels']

        feat = self.features[index]
        if self.transform is not None:
            feat = self.transform(feat)

        if self.return_item_ix:
            return feat, self.labels[index], int(index)
        else:
            return feat, self.labels[index]

    def __len__(self):
        return self.dataset_len


def make_features_dataloader(h5_file_path, batch_size, num_workers=8, shuffle=False, return_item_ix=False,
                             in_memory=True):
    if in_memory:
        dataset = FeaturesDatasetInMemory(h5_file_path, return_item_ix=return_item_ix)
    else:
        dataset = FeaturesDataset(h5_file_path, return_item_ix=return_item_ix)
    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, shuffle=shuffle, batch_size=batch_size)
    return loader


def make_incremental_features_dataloader(class_remap, h5_file_path, min_class, max_class, batch_size, num_workers=8,
                                         shuffle=False, return_item_ix=False, in_memory=True):
    # filter labels between min_class and max_class with class_remap
    h5 = h5py.File(h5_file_path, 'r')
    labels = np.array(h5['labels'], dtype=np.int64)

    class_list = []
    for i in range(min_class, max_class):
        class_list.append(class_remap[i])

    curr_idx = filter_by_class(labels, np.array(class_list))

    # make subset dataset with selected classes
    if in_memory:
        dataset = FeaturesDatasetInMemory(h5_file_path, return_item_ix=False)
    else:
        dataset = FeaturesDataset(h5_file_path, return_item_ix=False)
    loader = setup_subset_dataloader(dataset, curr_idx, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                     return_item_ix=return_item_ix)

    return loader


def filter_by_class(labels, class_list):
    ixs = []
    for c in class_list:
        curr_ix = np.where(labels == c)[0]
        ixs.extend(curr_ix.tolist())
    return ixs


class IndexSampler(torch.utils.data.Sampler):
    """Samples elements sequentially, always in the same order.
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class PartialDataset(Dataset):
    def __init__(self, data, indices, return_item_ix):
        self.data = data
        self.indices = indices
        self.return_item_ix = return_item_ix

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, y = self.data[index]
        if self.return_item_ix:
            return x, y, index
        else:
            return x, y


def setup_subset_dataloader(dataset, idxs, batch_size=256, shuffle=False, sampler=None, batch_sampler=None,
                            num_workers=8, return_item_ix=False):
    if batch_sampler is None and sampler is None:
        if shuffle:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(idxs)
        else:
            sampler = IndexSampler(idxs)

    dataset = PartialDataset(dataset, idxs, return_item_ix)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=True, batch_sampler=batch_sampler, sampler=sampler)
    return loader


def remap_classes(num_classes, seed):
    # get different class permutations

    np.random.seed(seed)
    ix = np.arange(num_classes)
    np.random.shuffle(ix)
    d = {}
    for i, v in enumerate(ix):
        d[i] = v
    return d


if __name__ == '__main__':
    h5_features_dir = '/media/tyler/Data/codes/edge-cl/features/places/supervised_resnet18_places_avg'
    h5_file_path = os.path.join(h5_features_dir, '%s_features.h5') % 'val'
    class_remap = remap_classes(365, 0)
    loader = make_incremental_features_dataloader(class_remap, h5_file_path, 0, 5, 256, num_workers=8, shuffle=False,
                                                  return_item_ix=False)
    print()