import random
import math
import numpy as np
from PIL import Image
from PIL import ImageFilter
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from .coco import COCO2014
from .voc import Voc2007
from .nus import Nuswide 
from .generate_observed_labels import generate_observed_labels
from .single_label import make_spl_data_loader


dataset_dict = {'VOC2007': Voc2007,
                'COCO2014': COCO2014,
                'NUSWIDE': Nuswide
                }


class Grid(object):
    def __init__(self, d1, d2, rotate=360, ratio=0.6, mode=1, p=0.8):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode=mode
        self.prob = p

    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)
        
        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image 
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h*h + w*w)))
        
        d = np.random.randint(self.d1, self.d2)
        
        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d*self.ratio)
        
        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh//d+1):
                s = d*i + st_h
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[s:t,:] *= 0
        for i in range(-1, hh//d+1):
                s = d*i + st_w
                t = s+self.l
                s = max(min(s, hh), 0)
                t = max(min(t, hh), 0)
                mask[:,s:t] *= 0
        r = np.random.randint(self.rotate)

        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]
        mask = torch.tensor(mask).float()

        if self.mode == 1:
            mask = 1-mask

        mask = mask.expand_as(img)
        img = img * mask 

        return img


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class TwoCropsTransform:
    """Take 2 random augmentations of one image."""

    def __init__(self,trans_weak, trans_strong):       
        self.trans_weak = trans_weak
        self.trans_strong = trans_strong
    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong(x)
        return [x1, x2]


def get_transform(args):
    transform = dict()

    # NOTE: the weak branch normalises with ImageNet statistics while the strong
    # and test branches use CLIP's. This is what the published results were
    # produced with, so it is left as-is; see README "Known issues".
    transform['weak_augment'] = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform['strong_augment'] = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        Grid(128, 448, rotate=360, ratio=0.6, mode=1, p=0.8)
    ])

    transform['test_augment'] = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    return transform


def make_pl_data_loader(args, is_train=True):
    root_dir = args.data_root_dir

    # Build val_loader
    transform = get_transform(args)
    if args.dataset in dataset_dict:
        val_dataset = dataset_dict[args.dataset](root_dir, phase='test', transform=transform['test_augment'])
    else:
        raise NotImplementedError('Value error: No matched dataset!')
    num_classes = val_dataset.get_number_classes()
    val_dataset.class_sample_num()
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    if not is_train:        
        class_names = val_dataset.class_names
        return None, val_loader, num_classes, class_names

    # Build train_loader
    if args.two_trans:
        transform_train = TwoCropsTransform(trans_weak=transform['weak_augment'], trans_strong=transform['strong_augment'])
    else:
        transform_train = transform['weak_augment']

    if args.dataset in dataset_dict:
        train_dataset = dataset_dict[args.dataset](root_dir, phase='train', transform=transform_train)
    else:
        raise NotImplementedError('Value error: No matched dataset!')

    if args.sample_partial_mode is not None:
        all_labels = train_dataset.labels
        obs_labels = generate_observed_labels(all_labels, sample_mode=args.sample_partial_mode, num_obs=args.sample_partial_param)
        train_dataset.labels = obs_labels
        
    train_dataset.class_sample_num()
    class_names = train_dataset.class_names
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    return train_loader, val_loader, num_classes, class_names