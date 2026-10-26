import os
import shutil
import torch
import numpy as np
from copy import deepcopy


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.ema.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def load_checkpoint(path):
    """Load a checkpoint written by this repo onto the CPU.

    It also holds optimizer and scheduler state, which PyTorch >= 2.6 refuses to unpickle
    by default, hence weights_only=False (older PyTorch has no such argument).
    """
    if not os.path.isfile(path):
        raise SystemExit('--resume: no such file %s' % path)
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def save_checkpoint(checkpoint, model_dir, is_best=False, epoch=1000):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    filename = 'checkpoint.pth'
    res_path = os.path.join(model_dir, filename)
    print('Save checkpoint to {}'.format(res_path))
    torch.save(checkpoint, res_path)
    if is_best:
        filename_best = 'checkpoint_best.pth'
        res_path_best = os.path.join(model_dir, filename_best)
        shutil.copyfile(res_path, res_path_best)


def print_learning_rate(optimizer):
    """ Sets learning rate if it is needed """
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    lr_now = np.unique(lr_list)
    print('Lr: {}'.format(lr_now))

