import torch

from dpipe.torch.functional import weighted_cross_entropy_with_logits


def dice_loss_with_logits(logit: torch.Tensor, target: torch.Tensor):
    """
    References
    ----------
    `Dice Loss <https://arxiv.org/abs/1606.04797>`_
    """
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    preds = torch.sigmoid(logit)

    sum_dims = list(range(1, logit.dim()))

    dice = 2 * torch.sum(preds * target, dim=sum_dims) / torch.sum(preds ** 2 + target ** 2, dim=sum_dims)
    loss = 1 - dice

    return loss.mean()


def iwbce(logit: torch.Tensor, target: torch.Tensor, cc: torch.Tensor = None, w_background=0.5, adaptive=False):
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    # re-weighting components
    weight = torch.ones_like(logit)
    for i, (target_single, cc_single) in enumerate(zip(target, cc)):
        n_cc = int(torch.max(cc_single).data)
        if n_cc > 0:
            n_positive = torch.sum(cc_single > 0).type(torch.FloatTensor)
            for n in range(1, n_cc + 1):
                weight[i][cc_single == n] = n_positive / (n_cc * torch.sum(cc_single == n))

        weight[i][cc_single == 0] = torch.min(weight[i]) * w_background

    loss = weighted_cross_entropy_with_logits(logit, target, weight, adaptive=adaptive)
    return loss


def iwdl(logit: torch.Tensor, target: torch.Tensor, cc: torch.Tensor = None, w_background=0.5):
    if not (target.size() == logit.size()):
        raise ValueError("Target size ({}) must be the same as logit size ({})".format(target.size(), logit.size()))

    preds = torch.sigmoid(logit)

    # re-weighting components
    weight = torch.ones_like(logit)
    for i, (target_single, cc_single) in enumerate(zip(target, cc)):
        n_cc = int(torch.max(cc_single).data)
        if n_cc > 0:
            n_positive = torch.sum(cc_single > 0).type(torch.FloatTensor)
            for n in range(1, n_cc + 1):
                weight[i][cc_single == n] = n_positive / (n_cc * torch.sum(cc_single == n))

        weight[i][cc_single == 0] = torch.min(weight[i]) * w_background

    sum_dims = list(range(1, logit.dim()))

    dice = 2 * torch.sum(weight * preds * target, dim=sum_dims) \
           / torch.sum(weight * (preds ** 2 + target ** 2), dim=sum_dims)

    loss = 1 - dice

    return loss.mean()
