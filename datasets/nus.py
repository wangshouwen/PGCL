import os
import logging
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def get_logger(log_dir):
    path = os.path.join(log_dir, "log.txt")
    logger = logging.getLogger("logger")
    logger.addHandler(logging.FileHandler(path))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    return logger


def read_txt(file):
    tmp = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            if '\\' in line:
                line = line.replace('\\', '/')
            tmp.append(line)
    return tmp


def cat2idx(file):
    cat_dict = dict()
    count = 0
    with open(file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            cat_dict[line] = count
            count += 1
    return cat_dict


def idx_to_category(cat2idx):
    idx2cat = {}
    for k, v in cat2idx.items():
        idx2cat[v] = k
    class_names = []
    for i in range(len(cat2idx)):
        class_names.append(idx2cat[i])

    return class_names


class Nuswide(Dataset):
    def __init__(self, root, phase, transform=None, target_transform=None):
        root_dir = os.path.abspath(root)
        self.root = os.path.join(root_dir, 'NUS-WIDE')
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform
        file_path = os.path.join(self.root, 'dpt_{}_set.npy'.format(self.phase))

        self.images = np.load(file_path, allow_pickle=True)
        self.cat2idx = json.load(open(os.path.join(self.root, 'category.json'), 'r'))
        self.class_names = idx_to_category(self.cat2idx)
        self.num_classes = len(self.cat2idx)
        names, labels = list(zip(*self.images))

        labels = np.stack(labels, 0)
        neg_idx = np.where(labels < 0)
        labels[neg_idx] = 0
        labels = 2 * labels - 1

        self.img_names = names
        self.labels = labels
        print('[dataset] NUS-WIDE classification phase={} number of classes={}  number of images={}'.format(
            self.phase, self.num_classes, len(self.images)))

    def class_sample_num(self):
        known_class_num = np.abs(self.labels).sum(0)
        known_class_pos_num = np.clip(self.labels, 0, 1).sum(0)
        self.known_class_label = np.stack((known_class_pos_num, known_class_num), axis=0)
        print('Number of pos-label per class: {}\n'
              'Number of known-label per class: {}'.format(self.known_class_label[0], self.known_class_label[1]))

    def __getitem__(self, index):
        path, target = self.img_names[index], self.labels[index]
        img = Image.open(os.path.join(self.root, 'Flickr', path)).convert('RGB')
        target = torch.from_numpy(target).float()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, index, path), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return self.num_classes