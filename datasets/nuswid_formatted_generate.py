"""Write the NUS-WIDE files the partial-label loader (datasets/nus.py) reads.

    python datasets/nuswid_formatted_generate.py --root ./datasets/dataset

Reads, from <root>/NUS-WIDE/ (the official NUS-WIDE download):
    Concepts81.txt, ImageList/TrainImagelist.txt, ImageList/TestImagelist.txt,
    TrainTestLabels/Labels_<concept>_{Train,Test}.txt
Writes, to the same folder: dpt_train_set.npy, dpt_test_set.npy, category.json
"""
import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm


def read_name_list(path, if_split=True):
    ret = []
    with open(path, 'r') as f:
        for line in f:
            if if_split:
                tmp = line.strip().split(' ')
                ret.append(tmp[0])
            else:
                tmp = line.strip()
                ret.append(tmp)
    return ret


def pairs(paths, labels):
    """An (N, 2) object array of [image path, label vector], as the loader expects."""
    out = np.empty((len(paths), 2), dtype=object)
    for i, (path, label) in enumerate(zip(paths, labels)):
        out[i, 0], out[i, 1] = path, np.array(label)
    return out


parser = argparse.ArgumentParser(description='Format NUS-WIDE for the partial-label setting')
parser.add_argument('--root', default='./datasets/dataset', help='folder that holds NUS-WIDE/')
root_dir = os.path.abspath(parser.parse_args().root)
root = os.path.join(root_dir, 'NUS-WIDE')
image_dir = os.path.join(root, "Flickr")

cls_name_list = read_name_list(os.path.join(root, 'Concepts81.txt'), False)
im_name_list = read_name_list(os.path.join(root, 'ImageList/TrainImagelist.txt'), False)
im_name_list_test = read_name_list(os.path.join(root, 'ImageList/TestImagelist.txt'), False)

path_labels = os.path.join(root, 'TrainTestLabels')
num_classes = len(cls_name_list)

# label columns follow Concepts81.txt, so that is the class order
with open(os.path.join(root, 'category.json'), 'w') as f:
    json.dump({name: i for i, name in enumerate(cls_name_list)}, f)

print('NUS-WIDE total {} images, test {} images. '.format(len(im_name_list), len(im_name_list_test)))

#################################################################################### Ttrain
train_labels = []
for i in tqdm(range(num_classes)):
    file_ = os.path.join(path_labels, 'Labels_' + cls_name_list[i] + '_Train.txt')
    cls_labels = []
    with open(file_, 'r') as f:
        for j, line in enumerate(f):
            tmp = line.strip()
            cls_labels.append(1 if int(tmp) > 0 else -1)
    train_labels.append(torch.tensor(cls_labels, dtype=torch.long))
train_labels = torch.stack(train_labels, dim=1)  # [161789, 81]
train_labels = np.array(train_labels)

train = []
for i, name in enumerate(im_name_list):
    file_ = '/'.join(name.split('\\'))
    train.append(file_)

np.save(os.path.join(root, 'dpt_train_set.npy'), pairs(train, train_labels))

#################################################################################### Test
test_labels = []
for i in tqdm(range(num_classes)):
    file_ = os.path.join(path_labels, 'Labels_' + cls_name_list[i] + '_Test.txt')
    cls_labels = []
    with open(file_, 'r') as f:
        for j, line in enumerate(f):
            tmp = line.strip()
            cls_labels.append(int(tmp))
    test_labels.append(torch.tensor(cls_labels, dtype=torch.long))
test_labels = torch.stack(test_labels, dim=1)
test_labels = np.array(test_labels)

test = []
for i, name in enumerate(im_name_list_test):
    file_ = '/'.join(name.split('\\'))
    test.append(file_)

np.save(os.path.join(root, 'dpt_test_set.npy'), pairs(test, test_labels))