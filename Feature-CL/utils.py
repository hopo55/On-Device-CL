import shutil
import json
import logging
import os
import numpy as np
from pathlib import Path
from typing import Union

import torch
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models as models


class CMA(object):
    """
    A continual moving average for tracking loss updates.
    """

    def __init__(self):
        self.N = 0
        self.avg = 0.0

    def update(self, X):
        self.avg = (X + self.N * self.avg) / (self.N + 1)
        self.N = self.N + 1

def randint(max_val, num_samples):
    """
    return num_samples random integers in the range(max_val)
    """
    rand_vals = {}
    _num_samples = min(max_val, num_samples)
    while True:
        _rand_vals = np.random.randint(0, max_val, num_samples)
        for r in _rand_vals:
            rand_vals[r] = r
            if len(rand_vals) >= _num_samples:
                break

        if len(rand_vals) >= _num_samples:
            break
    return rand_vals.keys()


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_each_checkpoint(state, epoch, save_dir):
    ckpt_path = os.path.join(save_dir, 'ckpt_%d.pth.tar' % epoch)
    torch.save(state, ckpt_path)


def save_checkpoint(state, is_best, save_dir):
    ckpt_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, ckpt_path)
    if is_best:
        best_ckpt_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(ckpt_path, best_ckpt_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,), output_has_class_ids=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    output = output.cpu()
    target = target.cpu()
    if not output_has_class_ids:
        output = torch.Tensor(output)
    else:
        output = torch.LongTensor(output)
    target = torch.LongTensor(target)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.shape[0]
        if not output_has_class_ids:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
        else:
            pred = output[:, :maxk].t()
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class TFSubLogger:
    def __init__(self, parent_logger, prefix):
        self.parent_logger = parent_logger
        self.prefix = prefix

    def add_scalar(self, name, value, iteration):
        self.parent_logger.add_scalar(self.prefix + "/" + name, value, iteration)

    def message(self, message, name=""):
        self.parent_logger.message(message, self.prefix + "/" + name)

'''
class Logger:
    def add_scalar(self, name, value, iteration):
        raise NotImplementedError

    def add_scalars(self, name, value, iteration):
        raise NotImplementedError

    def close(self):
        pass

    def get_logger(self, name):
        raise NotImplementedError

    def message(self, message, name=""):
        print("[" + name + "] " + message)


class TFLogger(SummaryWriter, Logger):
    def __init__(self, log_dir=None, verbose=False, **args):
        SummaryWriter.__init__(self, log_dir=log_dir)
        self.verbose = verbose

    def add_scalar(self, name, value, iteration):
        if self.verbose:
            print("[LOG]: At " + str(iteration) + ": " + name + " = " + str(value))
        SummaryWriter.add_scalar(self, name, value, iteration)

    def add_scalars(self, name, value, iteration):
        if self.verbose:
            print("[LOG]: At " + str(iteration) + ": " + name + " = " + str(value))
        SummaryWriter.add_scalars(self, name, value, iteration)

    def get_logger(self, name):
        return TFSubLogger(self, name)
'''

def save_predictions(y_pred, min_class_trained, max_class_trained, save_path, suffix=''):
    name = 'preds_min_trained_' + str(min_class_trained) + '_max_trained_' + str(max_class_trained) + suffix
    torch.save(y_pred, save_path + '/' + name + '.pth')


def save_accuracies(accuracies, min_class_trained, max_class_trained, save_path, suffix=''):
    name = 'accuracies_min_trained_' + str(min_class_trained) + '_max_trained_' + str(
        max_class_trained) + suffix + '.json'
    json.dump(accuracies, open(os.path.join(save_path, name), 'w'))


def get_feature_size(arch):
    c = {
        'resnet18': 512,
        'resnet50': 1000,
        'mobilenet_v3_small': 576,
        'mobilenet_v3_large': 960,
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
    }

    if len(arch) == 1:
        return c[arch[0]]  # Use the first element of arch as the key
    else:
        # avg_size = 0
        # for model in arch:
        #     avg_size += c[model]
            
        # avg_size /= len(arch)

        # return int(avg_size)
        size_list = []
        for model in arch:
            size_list.append(c[model])

        f_size = min(size_list)

        return f_size


def get_backbone(arch_list, pooling_type):
    model_list = []
    feature_list = []

    for arch in arch_list:
        feature_size = get_feature_size([arch])
        feature_list.append(feature_size)

        model = models.__dict__[arch](pretrained=True)
        model.classifier = nn.Sequential()
        if arch == 'resnet18':
            model.fc = nn.Sequential()
        if pooling_type == 'max':
            print('replacing global average pooling with max pooling...')
            model.avgpool = nn.AdaptiveMaxPool2d(output_size=1)

        model_list.append(model)

    return model_list, feature_list


def bool_flag(s):
    if s == '1' or s == 'True' or s == 'true':
        return True
    elif s == '0' or s == 'False' or s == 'false':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0/1 or True/False or true/false)'
    raise ValueError(msg % s)

def _is_aws_or_gcloud_path(tb_log_dir: str) -> bool:
    return tb_log_dir.startswith("gs://") or tb_log_dir.startswith("s3://")

def _make_path_if_local(tb_log_dir: Union[str, Path]) -> Union[str, Path]:
    if isinstance(tb_log_dir, str) and _is_aws_or_gcloud_path(tb_log_dir):
        return tb_log_dir

    tb_log_dir = Path(tb_log_dir)
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    return tb_log_dir

class Tensor_Logger():
    def __init__(self, path):
        self.path = path
        tb_log_dir = _make_path_if_local(self.path)
        self.logger = SummaryWriter(tb_log_dir)

    def result(self, title, log_data, n_iter):
        self.logger.add_scalar(title, log_data, n_iter)

    def config(self, config, metric_dict):
        config = vars(config)
        self.logger.add_hparams(config, metric_dict, run_name=None)


def compute_size(model):
    from torchsummary import summary
    device = 'cuda'
    model = model.to(device)
    summary(model, (3, 224, 224))

    """ How to obtain the parameters of the output layer
        :Perceptron = feature x number of classes
        :Fine-Tune = feature x number of classes
        :Naive Bayes = feature x number of classes x 3(muK, muK2 varK)
        :SOvR = feature x number of classes
        :NCM = feature x number of classes
        :Replay = feature x number of classes x (samples per class + 1) why + 1???
        :SLDA = (feature x number of classes) + (feature x feature)
    """

if __name__ == '__main__':
    device = 'cuda'
    mb_small, _ = get_backbone('mobilenet_v3_small', pooling_type='avg')
    mb_large, _ = get_backbone('mobilenet_v3_large', pooling_type='avg')
    e0, _ = get_backbone('efficientnet_b0', pooling_type='avg')
    e1, _ = get_backbone('efficientnet_b1', pooling_type='avg')
    rn18, _ = get_backbone('resnet18', pooling_type='avg')

    compute_size(mb_small)
    print()
