import torch
import torch.nn.functional as F


def dino_kl_loss(pred, target):
    # pred and target are of sizes Nx384xHxW
    # Each feature vector is normalized to 1
    return 1 - torch.sum(pred * target, dim=1)


def total_variation_loss(delta_flow, reduce_batch=True):
    # flow should be size (N, H, W, 2)
    reduce_dims = (0, 1, 2, 3) if reduce_batch else (1, 2, 3)
    distance_fn = lambda a: torch.where(a <= 1.0, 0.5 * a.pow(2), a - 0.5).mean(dim=reduce_dims)
    # assert delta_flow.size(-1) == 2
    diff_y = distance_fn((delta_flow[:, :-1, :, :] - delta_flow[:, 1:, :, :]).abs())
    diff_x = distance_fn((delta_flow[:, :, :-1, :] - delta_flow[:, :, 1:, :]).abs())
    loss = diff_x + diff_y
    return loss

