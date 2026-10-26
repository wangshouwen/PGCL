from torch.utils.data.dataset import Dataset
import torch
import os
from PIL import Image
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader


def get_metadata(dataset_name):
    if dataset_name == 'VOC2012':
        meta = {
            'num_classes': 20,
            'path_to_dataset': 'VOCdevkit/data',
            'path_to_images': 'VOCdevkit/VOC2012/JPEGImages'
        }
    elif dataset_name == 'MSCOCO':
        meta = {
            'num_classes': 80,
            'path_to_dataset': 'MSCOCO_2014/data',
            'path_to_images': 'MSCOCO_2014/'
        }
    elif dataset_name == 'NUSWIDE':
        meta = {
            'num_classes': 81,
            'path_to_dataset': 'NUS-WIDE/data',
            'path_to_images': 'NUS-WIDE/Flickr'
        }
    elif dataset_name == 'CUB':
        meta = {
            'num_classes': 312,
            'path_to_dataset': 'CUB_200_2011/data',
            'path_to_images': 'CUB_200_2011/images'
        }
    else:
        raise NotImplementedError('Metadata dictionary not implemented.')
    return meta


class ds_multilabel(Dataset):

    def __init__(self, dataset_name, phase, image_ids, label_matrix, label_matrix_obs, transform=None,
                 target_transform=None):
        meta = get_metadata(dataset_name)
        self.dataset_name = dataset_name
        self.phase = phase
        self.num_classes = meta['num_classes']
        self.path_to_images = meta['path_to_images']
        self.path_to_dataset = meta['path_to_dataset']

        self.image_ids = image_ids
        self.label_matrix = label_matrix
        self.label_matrix_obs = label_matrix_obs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        path = self.image_ids[idx]
        base_path = os.path.join(self.root_dir, self.path_to_images)
        path = os.path.join(base_path, path)
        if self.phase == 'train':
            target = self.label_matrix_obs[idx]
        else:
            target = self.label_matrix[idx]

        img = Image.open(path).convert('RGB')
        target = torch.from_numpy(target).float()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, idx, path), target

    def get_class_names(self):
        name_path = os.path.join(self.root_dir, self.path_to_dataset)
        class_names = np.load(os.path.join(name_path, '{}_class_names.npy'.format(self.dataset_name)))
        class_names = class_names.tolist()
        return class_names

    def class_sample_num(self):
        if self.phase == 'train':
            known_class_num = np.abs(self.label_matrix_obs).sum(0)
            known_class_pos_num = np.clip(self.label_matrix_obs, 0, 1).sum(0)
        else:
            known_class_num = np.abs(self.label_matrix).sum(0)
            known_class_pos_num = np.clip(self.label_matrix, 0, 1).sum(0)
        self.known_class_label = np.stack((known_class_pos_num, known_class_num), axis=0)

        print('Number of pos-label per class: {}'.format(self.known_class_label[0]))


def load_data(args, base_path):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        # NUS-WIDE has its own SPLC split (datasets/single_label/format_splc_nus.py writes it)
        prefix = 'splc_formatted' if args.setup == 'SPLC' and 'NUS-WIDE' in base_path else 'formatted'
        data[phase]['labels'] = np.load(os.path.join(base_path, '{}_{}_labels.npy'.format(prefix, phase)))
        data[phase]['labels_obs'] = np.load(os.path.join(base_path, '{}_{}_labels_obs.npy'.format(prefix, phase)))
        data[phase]['images'] = np.load(os.path.join(base_path, '{}_{}_images.npy'.format(prefix, phase)))

    return data


def generate_split(num_ex, frac, rng):
    n_2 = int(np.round(frac * num_ex))
    n_1 = num_ex - n_2

    # assign indices to splits:
    idx_rand = rng.permutation(num_ex)
    idx_1 = np.sort(idx_rand[:n_1])
    idx_2 = np.sort(idx_rand[-n_2:])

    return (idx_1, idx_2)


def make_spl_data_loader(args, is_train=True, with_validation=True):
    from .. import get_transform, TwoCropsTransform
    root_dir = args.data_root_dir
    transform = get_transform(args)
    meta = get_metadata(args.dataset)
    base_path = os.path.join(root_dir, meta['path_to_dataset'])
    source_data = load_data(args, base_path)

    test_dataset = ds_multilabel(dataset_name=args.dataset, phase='test',
                                 image_ids=source_data['val']['images'],
                                 label_matrix=source_data['val']['labels'],
                                 label_matrix_obs=source_data['val']['labels_obs'],
                                 transform=transform['test_augment'])
    test_dataset.root_dir = root_dir
    class_names = test_dataset.get_class_names()
    num_classes = test_dataset.num_classes
    test_dataset.class_sample_num()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)
    if not is_train:
        return None, None, test_loader, num_classes, class_names

    split_idx = {}
    split_idx['train'], split_idx['val'] = generate_split(len(source_data['train']['images']),
                                                          0.2 if with_validation else 0,
                                                          np.random.RandomState(1200))

    ss_rng = np.random.RandomState(1200)
    for phase in ['train', 'val']:
        num_initial = len(split_idx[phase])
        num_final = int(np.round(1.0 * num_initial))
        split_idx[phase] = split_idx[phase][np.sort(ss_rng.permutation(num_initial)[:num_final])]
    # Build train_loader
    if args.two_trans:
        transform_train = TwoCropsTransform(trans_weak=transform['weak_augment'],
                                            trans_strong=transform['strong_augment'])
    else:
        transform_train = transform['weak_augment']

    train_dataset = ds_multilabel(dataset_name=args.dataset, phase='train',
                                  image_ids=source_data['train']['images'][split_idx['train']],
                                  label_matrix=source_data['train']['labels'][split_idx['train'], :],
                                  label_matrix_obs=source_data['train']['labels_obs'][split_idx['train'], :],
                                  transform=transform_train)
    if with_validation:
        val_dataset = ds_multilabel(dataset_name=args.dataset, phase='test',
                                    image_ids=source_data['train']['images'][split_idx['val']],
                                    label_matrix=source_data['train']['labels'][split_idx['val'], :],
                                    label_matrix_obs=source_data['train']['labels_obs'][split_idx['val'], :],
                                    transform=transform['test_augment'])
    else:
        val_dataset = test_dataset
    train_dataset.root_dir = root_dir
    val_dataset.root_dir = root_dir
    train_dataset.class_sample_num()
    val_dataset.class_sample_num()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return train_loader, val_loader, test_loader, num_classes, class_names
