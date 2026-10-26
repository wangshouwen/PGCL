import argparse
import time
import os
import shutil
import random
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from datasets import make_pl_data_loader
from models import build_PGCL_model
from models import AsymmetricLoss
from models import IntraCategoryContrastiveLoss
from utils.metrics import AverageValueMeter, AveragePrecisionMeter
from utils.helper import save_checkpoint, load_checkpoint, print_learning_rate, ModelEma

parser = argparse.ArgumentParser(description='PyTorch Training for Multi-label Image Classification with partial labels')

''' Fixed in general '''
parser.add_argument('--data_root_dir', default='./datasets/dataset', type=str, help='data root path')
parser.add_argument('--image_size', '-i', default=448, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('-b', '--batch_size', default=16, type=int)
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='INT', help='number of data loading workers (default: 4)')
parser.add_argument('--display_interval', default=100, type=int, metavar='M', help='display_interval')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--w_d', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')

''' Train setting '''
parser.add_argument('--dataset', metavar='NAME', help='dataset name (e.g. COCO2014')
parser.add_argument('--two_trans', action='store_true')
parser.add_argument('--backbone', type=str, default='RN101')
parser.add_argument('--proj', action='store_true')
parser.add_argument('--model_ema', action='store_true', help='whether the model adopts exponential moving average method')
parser.add_argument('--ema_decay', type=float, default=0.9997)
parser.add_argument('--t', type=float, default=0.6)
parser.add_argument('--lambda_u', type=float, default=1.0)
parser.add_argument('--lambda_c', type=float, default=0.01)
parser.add_argument('--tau', type=float, default=1)
parser.add_argument('--T', type=float, default=0.8)
parser.add_argument('--save_dir', default='./checkpoint/COCO2014/', type=str, help='save path')
parser.add_argument('--sample_partial_mode', default='global_random', type=str, metavar='MODE', help='sample mode of label-observe-matrix, including class and img')
parser.add_argument('--sample_partial_param', default=[0.1], nargs='+', help='observed number of labels, it can be (0,1], -1 or int(>=0)')
parser.add_argument('--early_stop', action='store_true')

''' Val or Test setting '''
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


def main():
    global best_mAP
    args = parser.parse_args()
    if not args.resume:
        raise SystemExit('stage 2 needs --resume pointing at the stage-1 checkpoint_best.pth')
    if not (args.two_trans and args.proj):
        raise SystemExit('stage 2 needs --two_trans and --proj')
    print('---------------Args Items--------------')
    for k, v in vars(args).items():
        print('{}: {}'.format(k, v))
    print('---------------Args Items--------------\n')

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    train_loader, test_loader, num_classes, class_names = make_pl_data_loader(args)
    args.num_classes = num_classes
    model = build_PGCL_model(args, classnames=class_names)
    model.cuda()

    start_epoch = 0
    end_epoch = args.epochs
    best_mAP = 0
    best_epoch = -1

    if args.resume:
        print("* Loading checkpoint '{}'".format(args.resume))
        checkpoint = load_checkpoint(args.resume)
        best_mAP = checkpoint['best_score']
        if 'ema_state_dict' not in checkpoint:
            raise SystemExit('stage 2 starts from the EMA weights; run stage 1 with --model_ema')
        model.load_state_dict(checkpoint['ema_state_dict'], strict=False)
        # stage 2 keeps stage 1's best score as the bar to beat; if it never beats it,
        # the best checkpoint is still stage 1's, so put that one where stage 2 saves
        best_path = os.path.join(args.save_dir, 'checkpoint_best.pth')
        if os.path.abspath(args.resume) != os.path.abspath(best_path):
            os.makedirs(args.save_dir, exist_ok=True)
            shutil.copyfile(args.resume, best_path)
        print("==> Checkpoint Epoch: {0}, mAP: {1}".format(start_epoch, best_mAP))

    ema_model = None
    if args.model_ema:
        ema_model = ModelEma(model, decay=args.ema_decay)


    backbone_params, attn_params = model.backbone_params()
    prompt_params = model.prompt_params()
    for param in prompt_params:
        param.requires_grad = False
    tuned_params = model.tuned_params()

    gd_polices = [
                    {'params': backbone_params},
                    {'params': attn_params},
                    {'params': tuned_params}
                  ]
    optimizer = optim.AdamW(gd_polices, args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.w_d)
    args.steps_per_epoch = len(train_loader)
    lr_sched = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=args.steps_per_epoch, 
                                       epochs=args.epochs, pct_start=0.2) 
   
    criterion = {'ASLoss': AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05).cuda(),
                 'ClassContrastLoss': IntraCategoryContrastiveLoss(args=args, temperature=args.tau, contrast_th=args.T, size_average=True).cuda()
                }
    
    if args.evaluate:
        if args.model_ema:
            eva_model = ema_model.ema
        else:
            eva_model = model
        validate(test_loader, eva_model, criterion)
        return
        
    for epoch in range(start_epoch, end_epoch):
        print_learning_rate(optimizer)

        train(args, train_loader, model, ema_model, criterion, optimizer, epoch, scheduler=lr_sched)
        
        if args.model_ema:
            test_model = ema_model.ema
        else:
            test_model = model
        mAP = validate(test_loader, test_model, criterion)
        is_best = mAP > best_mAP
        if is_best:
            best_epoch = epoch
        best_mAP = max(mAP, best_mAP)
        checkpoint = {
                'epoch': epoch + 1,
                'model_name': args.backbone,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_sched.state_dict(),
                'best_score': best_mAP,
                'class_names': list(class_names),
                'image_size': args.image_size,
                }
        if args.model_ema:
            checkpoint['ema_state_dict'] = ema_model.ema.state_dict()
        model_dir = args.save_dir
        save_checkpoint(checkpoint, model_dir, is_best)

        print(' * best mAP={best:.4f}'.format(best=best_mAP))

        if args.early_stop:
            if best_epoch >= 0 and (epoch - best_epoch) == 5:
                print("epoch - best_epoch = {}, stop!".format(epoch - best_epoch)) 
                break
    

def train(args, train_loader, model, ema_model, criterion, optimizer, epoch, scheduler=None):

    sup_loss_meter = AverageValueMeter('sup_loss')
    unsup_loss_meter = AverageValueMeter('unsup_loss')
    loss_meter = AverageValueMeter('loss')
    data_time = AverageValueMeter('data_time')
    batch_time = AverageValueMeter('batch_time')

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    model.train()
    model.image_encoder.eval()
    model.image_encoder.layer4.train()
    model.image_encoder.attnpool.train()
    model.prompt_learner.eval()

    st_time = time.time()

    for i, (inputs, targets) in enumerate(train_loader):
        dt_time = time.time() - st_time
        data_time.add(dt_time)

        if args.two_trans:
            imgs_w, imgs_s = inputs[0]
            imgs = torch.cat((imgs_w, imgs_s), dim=0)
        else:
            imgs_w, imgs_s = inputs[0], None
            imgs = imgs_w
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        targets = targets.float().to(device)
        imgs = imgs.to(device)
        bs = imgs_w.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(imgs)
        if isinstance(outputs, tuple):
            logits, class_features = outputs
        else:
            logits = outputs
        logits_w = logits[:bs]
        logits_s = logits[bs:]

        pseudo_hard_labels, pseudo_soft_labels = unlabel_pseudo_labeling(logits_w, targets, args.t, args.t)

        sup_loss = criterion['ASLoss'](logits_w, targets) + criterion['ASLoss'](logits_s, targets)
        unsup_loss = criterion['ASLoss'](logits_s, pseudo_hard_labels.float())

        targets_ = targets + pseudo_soft_labels
        loss = sup_loss + args.lambda_u * unsup_loss
        contras_loss = args.lambda_c * criterion['ClassContrastLoss'](class_features, targets_)
        loss = loss + contras_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if args.model_ema:
            ema_model.update(model)
        loss_meter.add(loss.item(), imgs.size(0))
        sup_loss_meter.add(sup_loss.item(), imgs.size(0))
        unsup_loss_meter.add(unsup_loss.item() if isinstance(unsup_loss, torch.Tensor) else unsup_loss, imgs.size(0))

        bt_time = time.time() - st_time
        batch_time.add(bt_time)
        st_time = time.time()

        if i % args.display_interval == 0:
            print ('{0}\t''Epoch: [{1}] [{2}/{3}]\t'
                       'Batch_Time: {batch_time_cur:.3f} ({batch_time:.3f})\t'
                       'Data_Time: {data_time_cur:.3f} ({data_time:.3f})\t'
                       'Loss: {loss_cur:.4f} ({loss:.4f})\t'
                       'SupLoss: {loss1_cur:.4f} ({loss1:.4f})\t'
                       'UnsupLoss: {loss2_cur:.4f} ({loss2:.4f})\t'
                       .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, i, len(train_loader),
                    batch_time_cur=batch_time.value(), batch_time=batch_time.average(),
                    data_time_cur=data_time.value(), data_time=data_time.average(),
                    loss_cur=loss_meter.value(), loss=loss_meter.average(),
                    loss1_cur=sup_loss_meter.value(), loss1=sup_loss_meter.average(),
                    loss2_cur=unsup_loss_meter.value(), loss2=unsup_loss_meter.average()
                    ))

    loss = loss_meter.average()
    print('* Train-Epoch:[{0}]\t Loss: {loss:.4f}\t '.format(epoch, loss=loss))


def unlabel_pseudo_labeling(logits_w, targets, pos_th, neg_th):
    pseudo_hard_labels = torch.zeros_like(targets)
    with torch.no_grad():
        unlabel_part_mask = (targets == 0)
        p_pos = torch.sigmoid(logits_w.detach())
        p_neg = 1 - p_pos
        pos_mask = (p_pos * unlabel_part_mask) >= pos_th
        neg_mask = (p_neg * unlabel_part_mask) >= neg_th
        pseudo_hard_labels[pos_mask] = 1
        pseudo_hard_labels[neg_mask] = -1
        pseudo_soft_labels = pos_mask * p_pos + neg_mask * (-1) * p_neg

    return pseudo_hard_labels, pseudo_soft_labels


def validate(val_loader, model, criterion):

    model.eval()
    ap_meter = AveragePrecisionMeter()
    loss_meter = AverageValueMeter('loss')
    data_time = AverageValueMeter('data_time')
    batch_time = AverageValueMeter('batch_time')
    
    st_time = time.time()
    val_loader = tqdm(val_loader, desc='Validate')
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    for i, (inputs, targets) in enumerate(val_loader):
        dt_time = time.time() - st_time
        data_time.add(dt_time)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        imgs = inputs[0].to(device)
        targets = targets.float().to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion['ASLoss'](outputs, targets)
        loss_meter.add(loss.item(), imgs.size(0))

        bt_time = time.time() - st_time
        batch_time.add(bt_time)
        ap_meter.add(outputs.cpu().data, targets.cpu())
        st_time = time.time()
    
    ap = ap_meter.value()
    val_map = ap.mean()
    print(ap)

    OP, OR, OF1, CP, CR, CF1 = ap_meter.overall()
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = ap_meter.overall_topk(3)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('* Test\nLoss: {loss:.4f}\t mAP: {map:.4f}\t'
            'Data_time: {data_time:.4f}\t Batch_time: {batch_time:.4f}'.format(
            loss=loss_meter.average(), map=val_map, data_time=data_time.average(), batch_time=batch_time.average()))
    print('OP: {OP:.3f}\t OR: {OR:.3f}\t OF1: {OF1:.3f}\t'
          'CP: {CP:.3f}\t CR: {CR:.3f}\t CF1: {CF1:.3f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
    print('OP_3: {OP:.3f}\t OR_3: {OR:.3f}\t OF1_3: {OF1:.3f}\t'
          'CP_3: {CP:.3f}\t CR_3: {CR:.3f}\t CF1_3: {CF1:.3f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))

    return val_map

        
if __name__ == "__main__":
    main()