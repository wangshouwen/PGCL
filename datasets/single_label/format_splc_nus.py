"""Turn the shipped SPLC NUS-WIDE lists into the arrays `--setup SPLC` loads.

    python datasets/single_label/format_splc_nus.py

Reads datasets/dataset/NUS-WIDE/data/SPLC_nus_train_singlelabel.txt ("image,class" per line,
one known positive per image) and SPLC_nus_val.txt ("image,class class ..." with every
positive), and writes splc_formatted_{train,val}_{images,labels,labels_obs}.npy next to them,
in the formats of the LargeLoss files: labels 0/1, labels_obs +1 for the known positive and
0 (unknown) elsewhere. Class indices follow NUSWIDE_class_names.npy.
"""
import argparse
import os

import numpy as np

pp = argparse.ArgumentParser(description='Format the SPLC NUS-WIDE split')
pp.add_argument('--data-path', type=str, default='./datasets/dataset/NUS-WIDE/data',
                help='folder with the SPLC_nus_*.txt files; the output goes there too')
pp.add_argument('--num-classes', type=int, default=81)
args = pp.parse_args()


def read_split(path):
    images, positives = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, classes = line.split(',', 1)
            images.append(name)
            positives.append([int(c) for c in classes.split()])
    return np.array(images), positives


def to_matrix(positives, num_classes):
    labels = np.zeros((len(positives), num_classes), dtype=np.float32)
    for i, cls in enumerate(positives):
        labels[i, cls] = 1.0
    return labels


for phase, filename in [('train', 'SPLC_nus_train_singlelabel.txt'), ('val', 'SPLC_nus_val.txt')]:
    images, positives = read_split(os.path.join(args.data_path, filename))
    labels = to_matrix(positives, args.num_classes)
    # the train list holds exactly the one observed positive; val is only scored against `labels`
    labels_obs = labels.copy()
    for suffix, array in [('images', images), ('labels', labels), ('labels_obs', labels_obs)]:
        np.save(os.path.join(args.data_path, 'splc_formatted_{}_{}.npy'.format(phase, suffix)), array)
    print('{}: {} images'.format(phase, len(images)))
