import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self, reduce=None, size_average=None):
        super(BCELoss, self).__init__()
        self.reduce = reduce
        self.size_average = size_average
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y):
        weight = torch.abs(torch.clamp(y, -1, 1))
        y = torch.clamp(y, 0, 1)
        loss = self.BCEWithLogitsLoss(x, y)
        loss = loss * weight

        if self.reduce:
            if self.size_average:
                return torch.mean(loss[weight > 0]) if torch.sum(weight) != 0 else torch.mean(loss)
            return torch.sum(loss)
        return loss


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-6, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        y = y.reshape(-1)
        xs_pos = xs_pos.reshape(-1)
        xs_neg = xs_neg.reshape(-1)

        xs_pos = xs_pos[y!=0]
        xs_neg = xs_neg[y!=0]
        y = y[y!=0]
        y[y==-1] = 0

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            grad_on = torch.is_grad_enabled() and not self.disable_torch_grad_focal_loss
            with torch.set_grad_enabled(grad_on):
                pt0 = xs_pos * y
                pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
                pt = pt0 + pt1
                one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()


class IntraCategoryContrastiveLoss(nn.Module):

    def __init__(self, args, temperature=0.07, contrast_th=None, size_average=True):
        super(IntraCategoryContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_th = contrast_th
        self.size_average = size_average
        self.cls_num = args.num_classes
        

    def forward(self, features, labels):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) != 3:
            raise ValueError('`features` needs to be [bsz, ncls, ...],'
                             '3 dimensions are required')

        batch_size = labels.shape[0]
        w_features = features[:batch_size] # BxCxD
        s_features = features[batch_size:]
        features = torch.stack([w_features, s_features], dim=2).transpose(0, 1) #CxBx2xD

        pos_labels_mask = (labels >= self.contrast_th)
        neg_labels_mask = (labels <= (-self.contrast_th))
        
        total_loss = 0
        step = 0
        for c in range(self.cls_num):
            class_features = features[c]
            class_pos_mask = pos_labels_mask[:, c]
            class_neg_mask = neg_labels_mask[:, c]
            neg_features = class_features[class_neg_mask]

            if class_pos_mask.sum() == 0:
                continue
            else:
                pos_features = class_features[class_pos_mask]
                mix_features = torch.cat([pos_features, neg_features], dim=0) # Ax2xD
                anchor_features = mix_features
                mask = torch.cat([class_pos_mask[class_pos_mask], class_pos_mask[class_neg_mask]], dim=0)

                bsz = mask.size(0)
                anchor_features = torch.cat(torch.unbind(anchor_features, dim=1), dim=0)
                contrast_features = anchor_features
                anchor_count = 2

                mask = mask.view(-1, 1)
                mask = mask * mask.T
                mask = mask.float().fill_diagonal_(1)
                mask = mask.repeat(anchor_count, anchor_count)

                anchor_dot_contrast = torch.div(
                torch.matmul(anchor_features, contrast_features.T),
                self.temperature)

                logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
                logits = anchor_dot_contrast - logits_max.detach()

                logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                            torch.arange(bsz * anchor_count).view(-1, 1).to(device), 0)
                mask = mask * logits_mask
                exp_logits = torch.exp(logits) * logits_mask

                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  
                mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

                loss = - mean_log_prob_pos
                loss = loss.view(anchor_count, bsz).mean()

                total_loss = total_loss + loss
                step = step + 1
  
        if step == 0:
            # No category had a reliable positive in this batch, so there is
            # nothing to contrast. Return a differentiable zero.
            return features.sum() * 0.0

        if self.size_average:
            total_loss = total_loss / step

        return total_loss