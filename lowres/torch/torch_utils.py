import torch


def get_one_batch_elem_from_seq(seq, num_of_elem):
    new_seq = []
    for x in seq:
        new_elem = x[num_of_elem]
        new_seq.append(new_elem.reshape(torch.Size([1]) + new_elem.shape))
    return new_seq


def freeze_model(model):
    for params in model.parameters():
        params.requires_grad = False


def unfreeze_model(model):
    for params in model.parameters():
        params.requires_grad = True
