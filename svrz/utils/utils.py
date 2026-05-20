import torch


def _ffd(forward_values, current_values, directions, h, nrm_const = 1.0):
    if current_values.ndim == 1:
        current_values = current_values[:, None]
    diff = (forward_values - current_values) /  h
#    print(diff.shape, directions.shape)
    grad_contrib = diff[...,None] * directions                  
    b, l = directions.shape[:2]

    return nrm_const * torch.sum(grad_contrib, dim=(0, 1)) / (b * l)