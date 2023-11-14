import argparse
import numpy as np
import torch
from torchvision.utils import make_grid
from itertools import permutations
import torch.nn.functional as F


""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def map_minmax(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def sample_tuples(N, k=1, count=None, seed=None):

    if seed is not None:
        np.random.seed(seed)

    if count is None:  # return all possible (k+1) permutations
        # (N!/(N-k)!) x k array
        samples = np.array(list(permutations(range(N), k+1)))

    elif k == 1:
        p1 = np.random.choice(N, count)
        p2 = np.random.choice(N, count)
        return np.stack([p1, p2], axis=1)

    elif count == -1:
        samples = np.array(list(permutations(range(N), k)))
        samples = np.concatenate([samples, samples[:, 0].reshape(-1, 1)], axis=1)

    else: # sample count number of permutations
        # count x k array
        samples = np.zeros((count, k+1), dtype=int)
        for i in range(count):
            samples[i, :k] = np.random.choice(N, k, replace=False)
        # Force the last column to be same as the first column
        samples[:, k] = samples[:, 0]

    return samples


def images2grid(images, **grid_kwargs):
    # images should be (N, C, H, W)
    grid = make_grid(images, **grid_kwargs)
    out = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return out


def compute_pck(pred, target, vis=None, thresholds=None, img_size=256,
                alphas=None):
    if type(target) == list:
        target = torch.cat(target, dim=0).float().cpu()
    else:
        target = target.float().cpu()
    if type(pred) == list:
        pred = torch.cat(pred, dim=0).float().cpu()
    else:
        pred = pred.float().cpu()
    if vis is not None and type(vis) == list:
        vis = torch.cat(vis, dim=0).bool().cpu()
    elif vis is not None:
        vis = vis.bool().cpu()
    else:
        vis = torch.ones(target.size(0)).bool()
    target = target[vis]
    pred = pred[vis]

    if alphas is None:
        alphas = torch.arange(0.1, 0.009, -0.01)
    else:
        alphas = torch.tensor(alphas)
    correct = torch.zeros(len(alphas))

    err = (pred- target).norm(dim=-1)
    err = err.unsqueeze(0).repeat(len(alphas), 1)

    if thresholds is None:
        thresholds = alphas.unsqueeze(-1).repeat(1, err.size(1)) * img_size
    else:
        # Each keypoint within an image pair get same threshold
        # First get threshold (bbox) for all the visible keypoints
        if type(thresholds) == list:
            thresholds = torch.cat(thresholds, dim=0).float().cpu()
        thresholds = thresholds.unsqueeze(-1).repeat(1, vis.size(1))
        thresholds = thresholds[vis]
        # Next compute alpha x threshold for all the keypoints
        thresholds = thresholds.unsqueeze(0).repeat(len(alphas), 1)
        thresholds = thresholds * alphas.unsqueeze(-1)

    correct = err < thresholds
    correct = correct.sum(dim=-1) / len(target)

    print("PCK-Transfer: ", ','.join([f'{pck * 100:.2f}' for pck in correct]))
    return correct


def pck_loop(tuples, kps_all, transfer_fn, *args, ignore_interim=False, **kwargs):
    chain_length = tuples.shape[1] - 1
    gt_kps_all = []
    pred_kps_all = []
    vis_all = []
    for ch in range(chain_length):
        src_idx = tuples[:, ch]
        trg_idx = tuples[:, ch+1]

        if ch == 0:
            src_kps = kps_all[src_idx]
        else:
            src_kps = pred_kps

        pred_kps = transfer_fn(src_kps[..., :2], src_idx, trg_idx,
                                *args, **kwargs)

        gt_kps_all.append(kps_all[trg_idx][..., :2])
        pred_kps_all.append(pred_kps)
        
        if ch == 0:
            vis = kps_all[src_idx][..., 2] * kps_all[trg_idx][..., 2] > 0
        else:
            vis = vis * kps_all[trg_idx][..., 2] > 0
        vis_all.append(vis)

    if ignore_interim:
        return gt_kps_all[-1], pred_kps_all[-1], vis_all[-1]
    else:
        vis_all = torch.cat(vis_all)
        gt_kps_all = torch.cat(gt_kps_all)
        pred_kps_all = torch.cat(pred_kps_all)
        return gt_kps_all, pred_kps_all, vis_all
