"""Run a trained PGCL checkpoint on your own images.

    python predict.py --checkpoint ./checkpoint/COCO2014_pl/0.1_stage2/checkpoint_best.pth --images photo.jpg

Nothing here touches the benchmark loaders, so it works on arbitrary images.

Class names are not cosmetic in PGCL: they are fed through CLIP's text encoder to
build the class-specific queries, so they must be the SAME names in the SAME order
used at training. Checkpoints written by this repo store them (with the backbone and
input size), so nothing else is needed. For an older checkpoint without them, pass
--classes (.txt with one name per line, .npy, or a category.json) plus --backbone
and --image_size.

Scores are sigmoid probabilities, so --threshold is meaningful. The label
convention the model was trained with: +1 present, -1 absent, 0 unknown.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import build_PGCL_model
from utils.helper import load_checkpoint

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--checkpoint', required=True, type=str,
                    help='a checkpoint written by train_pl2.py / train_spl2.py')
parser.add_argument('--images', required=True, type=str,
                    help='an image file or a directory of images')
parser.add_argument('--classes', default=None, type=str,
                    help='only for checkpoints without class names: .txt, .npy or category.json, in training order')
parser.add_argument('--backbone', default=None, type=str,
                    help='default: from the checkpoint (RN101 partial-label, RN50 single-positive)')
parser.add_argument('--image_size', default=None, type=int,
                    help='default: from the checkpoint (448 partial-label, 224 single-positive)')
parser.add_argument('--threshold', default=0.5, type=float, help='report labels scoring above this')
parser.add_argument('--top_n', default=0, type=int,
                    help='if > 0, report this many highest-scoring labels instead of thresholding')
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)


def list_images(path):
    if os.path.isfile(path):
        return [path]
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        raise SystemExit('no images found in %s' % path)
    return [os.path.join(path, f) for f in files]


def load_class_names(path):
    if path.endswith('.npy'):
        return [str(n) for n in np.load(path, allow_pickle=True)]
    if path.endswith('.json'):  # category.json: {name: index}
        with open(path) as f:
            cat2idx = json.load(f)
        return sorted(cat2idx, key=cat2idx.get)
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    if args.classes:
        class_names = load_class_names(args.classes)
    elif 'class_names' in checkpoint:
        class_names = list(checkpoint['class_names'])
    else:
        raise SystemExit('this checkpoint stores no class names; pass --classes')
    args.backbone = args.backbone or checkpoint.get('model_name', 'RN101')
    args.image_size = args.image_size or checkpoint.get('image_size', 448)
    args.num_classes = len(class_names)
    args.proj = False  # the projection head is only used by the contrastive loss during training

    model = build_PGCL_model(args, classnames=class_names)
    state = checkpoint.get('ema_state_dict') or checkpoint.get('state_dict') or checkpoint
    # strict=False: the checkpoint also carries the training-only projection head.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print('note: %d keys missing from the checkpoint (first: %s)' % (len(missing), missing[0]))
    model.to(args.device).eval()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])

    for path in list_images(args.images):
        image = Image.open(path).convert('RGB')
        batch = transform(image).unsqueeze(0).to(args.device)

        with torch.no_grad():
            outputs = model(batch)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            scores = torch.sigmoid(logits)[0].float().cpu()

        if args.top_n > 0:
            values, indices = scores.topk(min(args.top_n, len(class_names)))
        else:
            indices = (scores >= args.threshold).nonzero(as_tuple=True)[0]
            order = scores[indices].argsort(descending=True)
            indices = indices[order]
            values = scores[indices]

        print(os.path.basename(path))
        if len(indices) == 0:
            print('  (nothing above %.2f)' % args.threshold)
        for score, idx in zip(values.tolist(), indices.tolist()):
            print('  %.3f  %s' % (score, class_names[idx]))


if __name__ == '__main__':
    main()
