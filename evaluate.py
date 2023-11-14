from pathlib import Path
import argparse
import numpy as np
import torch
from sklearn.preprocessing import normalize

from datasets.cub import CUBDataset
from datasets.spair import SpairDataset
from models.asic import Asic
from thirdparty.dino_vit_features.extractor import str2bool
from commons.utils import compute_pck, sample_tuples, pck_loop


@torch.no_grad()
def main(args):
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    if args.dset.lower() == 'cub':
        interim_dir = f'{args.dset.lower()}/{args.split}/{args.cub_idx:03d}'
        flow_dir = Path(args.flow_dir) / interim_dir / f'{args.model_type}_s{args.stride}'
        dset = CUBDataset(
            args.img_dir, split=args.split, img_size=args.img_size,
            cls_idx=args.cub_idx, flow_dir=flow_dir, num_parts=args.num_parts,
            mask_threshold=args.mask_threshold)
    elif args.dset.lower() == 'spair':
        interim_dir = f'{args.dset.lower()}/{args.split}/{args.spair_cat}'
        flow_dir = Path(args.flow_dir) / interim_dir / f'{args.model_type}_s{args.stride}'
        dset = SpairDataset(
            args.img_dir, split=args.split, img_size=args.img_size,
            spair_cat=args.spair_cat, flow_dir=flow_dir,
            num_parts=args.num_parts, mask_threshold=args.mask_threshold)
    else:
        raise NotImplementedError

    N = len(dset)
    print(f"{N} points in the dataset")
    kps_all = dset.kps.to(device)
    imgs_all = dset.imgs.to(device)
    masks_all = dset.masks.unsqueeze(1).to(device)

    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    ckpt_args = ckpt['args']
    stn = Asic(3, args.img_size, mf=ckpt_args.channel_multiplier,
                        bilinear=ckpt_args.bilinear,
                        padding_mode=ckpt_args.padding_mode,
                        use_tanh=ckpt_args.use_tanh).to(device)
    stn.load_state_dict(ckpt["t_ema"])
    transfer_fn = stn.transfer_points
    transfer_args = [imgs_all, masks_all, args.img_size]

    # First compute PCK for all 2-pairs
    if dset.fixed_pairs is None:
        tuples = sample_tuples(N)
        thresholds = None
    else:
        # For SPair-71k (thresholds are max size of bounding boxes)
        tuples = dset.fixed_pairs
        thresholds = [torch.from_numpy(dset.thresholds)[tuples[:, 1]]]

    print(f"First computing 2-point PCK for {len(tuples)} pairs")
    gt_corrs, pred_corrs, vis = pck_loop(tuples, kps_all, transfer_fn, *transfer_args)
    compute_pck(pred_corrs, gt_corrs, vis, thresholds, img_size=args.img_size)

    # Compute k-cycle PCK
    for k in args.k:
        tuples = sample_tuples(N, k=k, count=args.num_samples)
        if dset.fixed_pairs is None:
            thresholds = None
        else:
            thresholds = torch.from_numpy(dset.thresholds[tuples[:, 1:]])
            thresholds = thresholds.reshape(-1)
        print(f"Next computing {k}-cycle PCK for {len(tuples)} tuples")
        gt_corrs, pred_corrs, vis = pck_loop(tuples, kps_all, transfer_fn, *transfer_args)
        compute_pck(pred_corrs, gt_corrs, vis, thresholds, img_size=args.img_size)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess images')
    # DINO Hyperparameters
    parser.add_argument('--stride', default=2, type=int,
                        help="""stride of first convolution layer.
                                small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8',
                        choices=['dino_vits8', 'dino_vits16', 'dino_vitb8',
                                 'dino_vitb16', 'vit_small_patch8_224',
                                 'vit_small_patch16_224', 'vit_base_patch8_224',
                                 'vit_base_patch16_224', 'dinov2_vits14', 'dinov2_vitb14'],
                        help='type of model to extract.')
    parser.add_argument('--nbb_layer', default=9, type=int)
    parser.add_argument('--nbb_facet', default='key')
    parser.add_argument('--nbb_bin', default=False, type=str2bool)

    # Input
    parser.add_argument("--dset", type=str, default='cub',
                        choices=['cub', 'spair'],
                        help="data type")
    parser.add_argument("--img_dir", type=str, default='raw_data',
                        help="Path to images")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image size")
    parser.add_argument('--num_parts', default=0, type=int, help="number of parts")
    parser.add_argument("--mask_threshold", type=int, default=1,
                        help="Threshold for masking")

    # Cub dataset
    parser.add_argument("--cub_idx", type=int, default=1, help="cub category")
    parser.add_argument("--spair_cat", default='cat', help="spair category")
    parser.add_argument("--split", default='test', help="spair category")

    # Output
    parser.add_argument("--flow_dir", type=str, default="processed_data",
                        help="Output path")

    # Correspondence Hyperparam
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Output path")
    parser.add_argument("--k", type=int, default=[2], nargs='+',
                        help="Output path")

    # Load pretrained checkpoint
    parser.add_argument("--ckpt", default=None)

    args = parser.parse_args()

    main(args=args)
