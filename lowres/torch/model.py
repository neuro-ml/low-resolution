from functools import partial

import torch
from torch.nn.functional import max_pool3d

from dpipe.medim.utils import identity
from dpipe.torch.utils import sequence_to_var, to_np
from dpipe.torch.model import optimizer_step
from .torch_utils import get_one_batch_elem_from_seq


def train_step_with_x8(*inputs, architecture, criterion, optimizer, scale_factor: int = None, **optimizer_params):
    architecture.train()

    n_targets = 1
    n_inputs = len(inputs) - n_targets

    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, target = inputs[:n_inputs], inputs[-1]

    if scale_factor is not None:
        pool_fn = partial(max_pool3d, kernel_size=scale_factor, ceil_mode=True)
        loss = criterion(architecture(*inputs), pool_fn(target))
    else:
        loss = criterion(architecture(*inputs), target)

    optimizer_step(optimizer, loss, **optimizer_params)
    return to_np(loss)


def train_step_lowres(*inputs, model_x8, model_lowres, optimizer, criterion_x8, criterion_lowres, scale_factor=8,
                      lowres_alpha=0.5, bbox_loss=False, margin=0, **optimizer_params):
    model_x8.train()
    model_lowres.train()

    n_targets = 1
    n_inputs = len(inputs) - n_targets
    inputs = sequence_to_var(*inputs, device=model_lowres)
    inputs, target = inputs[:n_inputs], inputs[-1]

    *x_features, y8 = model_x8.forward_features(*inputs)
    y = model_lowres.restore_shape(y8=y8)

    loss_lowres = 0
    bbox_counter = 0
    for i, y8_single in enumerate(y8):
        x_features_single = get_one_batch_elem_from_seq(seq=x_features, num_of_elem=i)
        y_pred_single, y8_masks, y8_mask_shapes = model_lowres.prepare_masks(y8_single=y8_single, margin=margin)

        if y8_masks is not None:
            for y8_mask, y8_mask_shape in zip(y8_masks, y8_mask_shapes):

                local_pred, y_mask = model_lowres(
                    x_features_single=x_features_single, y8_mask=y8_mask, y8_mask_shape=y8_mask_shape
                )

                if bbox_loss:
                    loss_on_elem = criterion_lowres(local_pred.flatten(), target[i][y_mask[0]])
                    loss_lowres = loss_lowres + loss_on_elem
                    bbox_counter += 1.
                else:
                    y_pred_single[y_mask] = local_pred.flatten()

            # end for
            if not bbox_loss:
                y[i] = y_pred_single[0]

    # end for

    if bbox_loss and bbox_counter > 0:
        loss_lowres /= bbox_counter
    else:
        loss_lowres = criterion_lowres(y, target)

    pool_x8 = partial(max_pool3d, kernel_size=scale_factor, ceil_mode=True)
    loss_x8 = criterion_x8(y8, pool_x8(target))

    loss = lowres_alpha * loss_lowres + (1 - lowres_alpha) * loss_x8

    optimizer_step(optimizer, loss, **optimizer_params)

    return to_np(loss)


def inference_step_lowres(*inputs, model_x8, model_lowres, activation=identity, margin=0):
    model_x8.eval()
    model_lowres.eval()
    with torch.no_grad():
        inputs = sequence_to_var(*inputs, device=model_lowres)

        *x_features, y8 = model_x8.forward_features(*inputs)
        y = model_lowres.restore_shape(y8=y8)

        x_features_single = get_one_batch_elem_from_seq(seq=x_features, num_of_elem=0)
        _, y8_masks, y8_mask_shapes = model_lowres.prepare_masks(y8_single=y8[0], margin=margin)

        if y8_masks is not None:
            for y8_mask, y8_mask_shape in zip(y8_masks, y8_mask_shapes):
                local_pred, y_mask = model_lowres(
                    x_features_single=x_features_single, y8_mask=y8_mask, y8_mask_shape=y8_mask_shape
                )
                y[y_mask] = local_pred.flatten()

        return to_np(activation(y))
