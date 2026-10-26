import argparse
import time
import os
import random
from datetime import datetime
from tqdm import tqdm

import torch

from datasets import make_pl_data_loader
from datasets import make_spl_data_loader
from models import build_PGCL_model
from utils.metrics import AverageValueMeter, AveragePrecisionMeter
from utils.helper import load_checkpoint

parser = argparse.ArgumentParser(description='PyTorch Training for Multi-label Image Classification with partial labels')

''' Fixed in general '''
parser.add_argument('--data_root_dir', default='./datasets/dataset', type=str, help='save path')
parser.add_argument('--image_size', '-i', default=448, type=int)
parser.add_argument('-b', '--batch_size', default=32, type=int)
parser.add_argument('--setting', default=None, type=str)
parser.add_argument('--setup', default='LargeLoss', type=str) # either LargeLoss or SPLC has the same test_loader
parser.add_argument('--two_trans', default=False, type=str)

''' Train setting '''
parser.add_argument('--dataset', metavar='NAME', help='dataset name (e.g. COCO2014')
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='INT', help='number of data loading workers (default: 4)')
parser.add_argument('--backbone', type=str, default='RN101')
parser.add_argument('--proj', default=True, action='store_true')
parser.add_argument('--save_dir', default='./checkpoint/test/', type=str, help='save path')

''' Val or Test setting '''
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


def main():
    args = parser.parse_args()
    print('---------------Args Items--------------')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print('---------------Args Items--------------\n')

    if args.setting == 'pl':
        _, test_loader, num_classes, class_names = make_pl_data_loader(args, is_train=False)
    elif args.setting == 'spl':
        _, _, test_loader, num_classes, class_names = make_spl_data_loader(args, with_validation=False)
    else:
        raise NotImplementedError

    args.num_classes = num_classes
    model = build_PGCL_model(args, classnames=class_names)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if not args.resume:
        raise SystemExit('test.py needs --resume path/to/checkpoint_best.pth')
    print("* Loading checkpoint '{}'".format(args.resume))
    checkpoint = load_checkpoint(args.resume)
    message = model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
    print(message)
    
    validate(test_loader, model, args.setting)
    return


def validate(val_loader, model, setting):

    model.eval()

    ap_meter = AveragePrecisionMeter()
    data_time = AverageValueMeter('data_time')
    batch_time = AverageValueMeter('batch_time')
    
    val_loader = tqdm(val_loader, desc='Validate')

    for i, (inputs, targets) in enumerate(val_loader):

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        imgs = inputs[0].to(device)

        targets = targets.float().to(device)
        
        if setting == 'spl':
            targets[targets == 0] = -1

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        ap_meter.add(outputs.cpu().data, targets.cpu())
    
    ap = ap_meter.value()
    val_map = ap.mean()
    print(ap)
    OP, OR, OF1, CP, CR, CF1 = ap_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = ap_meter.overall_topk(3)
    print('* Test\nmAP: {map:.4f}\t'
            'Data_time: {data_time:.4f}\t Batch_time: {batch_time:.4f}'.format(
            map=val_map, data_time=data_time.average(), batch_time=batch_time.average()))
    print('OP: {OP:.3f}\t OR: {OR:.3f}\t OF1: {OF1:.3f}\t'
          'CP: {CP:.3f}\t CR: {CR:.3f}\t CF1: {CF1:.3f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
    print('OP_3: {OP:.3f}\t OR_3: {OR:.3f}\t OF1_3: {OF1:.3f}\t'
          'CP_3: {CP:.3f}\t CR_3: {CR:.3f}\t CF1_3: {CF1:.3f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))

    return val_map

        
if __name__ == "__main__":
    main()