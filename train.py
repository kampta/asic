import argparse
import torch
from torch import nn, optim
import numpy as np
from tqdm import tqdm
import json
import os
import torch.nn.functional as F
import wandb
from pathlib import Path

from commons.logger import Logger, log_visuals
from commons.distributed import get_rank, setup_distributed, reduce_loss_dict,\
    get_world_size, primary
from commons.utils import sample_tuples
from datasets.cub import CUBDataset
from datasets.in_memory import InMemoryDataset
from datasets.spair import SpairDataset
from datasets.utils import Augmentor
from models.utils import accumulate, requires_grad
from models.canonical import Canonical, CanonicalMLP
from models.asic import Asic
from losses.reg_losses import total_variation_loss
from thirdparty.lpips.lpips import get_perceptual_loss
from losses.matching_losses import LossCorrsSparse
from thirdparty.gangealing.annealing import DecayingCosineAnnealingWarmRestarts,\
    lr_cycle_iters


def save_state_dict(ckpt_name, c_module, t_module, c_ema, t_ema, canon_optim,
                    canon_sched, t_optim, t_sched, args, step,
                    add_step_to_name=False):
    ckpt_dict = {
        "canon": c_module.state_dict(),
        "t": t_module.state_dict(),
        "c_ema": c_ema.state_dict(),
        "t_ema": t_ema.state_dict(),
        "t_optim": t_optim.state_dict(),
        "t_sched": t_sched.state_dict(),
        "canon_optim": canon_optim.state_dict()
        if canon_optim is not None else None,
        "canon_sched": canon_sched.state_dict()
        if canon_sched is not None else None,
        "args": args,
        "iter": step
    }
    torch.save(ckpt_dict, f'{results_path}/{ckpt_name}.pt')
    if add_step_to_name:
        torch.save(ckpt_dict, f'{results_path}/{ckpt_name}_{step:07d}.pt')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def base_training_argparse():
    parser = argparse.ArgumentParser(description="Training")
    # Main training arguments:
    parser.add_argument("--exp-name", type=str, required=True,
                        help="Name for experiment run (used for logging)")
    parser.add_argument("--results", type=str, default='logs',
                        help='path to the results directory')
    parser.add_argument("--seed", default=0, type=int,
                        help='Random seed for this experiment')
    parser.add_argument("--dset", type=str, default='cub',
                        choices=["cub", "spair"])
    parser.add_argument("--img_dir", type=str, required=True,
                        help="Path to real data")
    parser.add_argument("--flow_dir", type=str, default='processed_data',
                        help="Path to preprocessed flows")
    parser.add_argument("--mask_threshold", type=int, default=1,
                        help="Threshold for masking")
    parser.add_argument("--mask_bbox_pad", type=int, default=4,
                        help="Crop with some padding")
    parser.add_argument("--img_size", default=256, type=int,
                        help='resolution of real images')
    parser.add_argument("--iter", type=int, default=20000,
                        help="total training iterations")
    parser.add_argument("--batch", type=int, default=20,
                        help="batch size per-GPU")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="num workers for dataloader")

    # Dataset hyperparameters:
    parser.add_argument("--cub_idx", type=int, default=1, help="cub category")
    parser.add_argument("--split", default='test',
                        choices=['test', 'val'],
                        help='splits for training and validation')
    parser.add_argument("--use_coseg_masks", action='store_true')
    parser.add_argument("--num_parts", default=4, type=int)
    parser.add_argument("--spair_cat", default='cat', help="cub category")

    # Loss hyperparameters:
    parser.add_argument("--loss_fn", type=str, default='vgg_ssl',
                        choices=['lpips', 'vgg_ssl'],
                        help="The perceptual loss to use.")
    parser.add_argument("--rec_weight", type=float, default=1.,
                        help='weight for reconstruction loss')
    parser.add_argument("--nbb_weight", type=float, default=30.,
                        help='weight for nbb loss')
    parser.add_argument("--flow_tv_weight", default=15000.0, type=float,
                        help="""Loss weighting of the Total Variation
                        smoothness regularizer on the residual flow""")
    parser.add_argument("--equi_weight", default=1.0, type=float,
                        help='Loss weighting for equivariance')
    parser.add_argument("--sparse_topk", type=int, default=None,
                        help='number of sparse correspondences for loss')
    parser.add_argument("--sparse_temp", type=float, default=1,
                        help='temperature for sparse loss')
    parser.add_argument("--mask_weight", default=0.1, type=float,
                        help="""Loss weighting of the mask""")
    parser.add_argument("--parts_weight", default=10.0, type=float,
                        help="""Loss weighting of the Parts Mask""")
    parser.add_argument("--use_nbb_parts", action='store_true')

    # Augmentation hyperparameters
    parser.add_argument("--jitter", default=[0.4, 0.4, 0.2, 0.1], type=float,
                        nargs='+', help='augmentation mode')
    parser.add_argument("--jitter_prob", default=0.8, type=float)
    parser.add_argument("--gray_prob", default=0.2, type=float)
    parser.add_argument("--solar_prob", default=0.2, type=float)
    parser.add_argument("--tps_scale", default=0.4, type=float)

    # Canonical space
    parser.add_argument("--unwarp_size", type=int, default=128,
                    help="resolution for unwarping")

    # Learned Grid hyperparameters
    parser.add_argument("--canon_size", type=int, default=256,
                        help="resolution of canonical space")
    parser.add_argument("--clamp", action='store_true',
                        help="clamp values of canonical space (-1, 1)")

    # MLP Hyperparams
    parser.add_argument("--use_mlp", action='store_true')
    parser.add_argument("--mlp_hidden_dim", type=int, default=256,
                        help="number of hidden units per layer")
    parser.add_argument("--mlp_num_layers", type=int, default=8,
                        help="number of layers")
    parser.add_argument("--mlp_skip_layers", type=int, nargs='+',
                        default=[4, 7], help="skip layers")

    # Model hyperparameters:
    parser.add_argument("--canon_lr", type=float, default=0.003,
                        help="base learning rate of canonical space")
    parser.add_argument("--canon_ema", action='store_true',
                        help='Enable ema for canonical space')
    parser.add_argument("--stn_ema", action='store_true',
                        help='Enable ema for canonical space')
    parser.add_argument("--stn_lr", type=float, default=0.003,
                        help="base learning rate of SpatialTransformer")
    parser.add_argument("--flow_ssl", action='store_true',
                        help="""If specified, apply STN on SSL features)""")
    parser.add_argument("--channel_multiplier", default=0.5, type=float,
                        help='channel multiplier for smaller models')
    parser.add_argument("--bilinear", action='store_true',
                        help='Apply bilinear upsample/downsample')
    parser.add_argument("--padding_mode", default='border',
                        choices=['border', 'zeros', 'reflection'], type=str,
                        help="""Padding algorithm for when the STN samples
                        beyond image boundaries""")
    parser.add_argument("--use_tanh", action='store_true',
                        help='Use tanh activation at the flow output')
    parser.add_argument("--disable_tps", action='store_true',
                        help='disable tps transformations')

    # Backbone parameters
    parser.add_argument("--bb", default='dino_vits8',
                        choices=['dino_vits8', 'dino_vits16', 'dino_vitb8',
                                 'dino_vitb16', 'vit_small_patch8_224',
                                 'vit_small_patch16_224',
                                 'vit_base_patch16_224'],
                        help='backbone models')
    parser.add_argument('--bb_stride', default=2, type=int,
                        help="stride.")

    # Visualization hyperparameters:
    parser.add_argument("--vis_every", type=int, default=500,
                        help="""frequency with which visualizations are
                        generated during training""")
    parser.add_argument("--vis_denseres", type=int, default=32,
                        help='number of sparse correspondences to visualize')
    parser.add_argument("--ckpt_every", type=int, default=10000,
                        help='frequency of checkpointing during training')
    parser.add_argument("--log_every", default=25, type=int,
                        help='How frequently to log data to TensorBoard')
    parser.add_argument("--n_sample", type=int, default=4,
                        help="""number of images (real and fake) to
                        generate visuals for""")
    parser.add_argument("--disable_wandb", action='store_true',
                    help='Disable wandb for debugging')


    # Learning Rate scheduler hyperparameters:
    parser.add_argument("--period", default=10000, type=float,
                        help="""Period for cosine learning rate scheduler
                        (measured in gradient steps)""")
    parser.add_argument("--decay", default=0.9, type=float,
                        help="""Decay factor for the cosine learning rate
                        scheduler""")
    parser.add_argument("--tm", default=2, type=int,
                        help="""Period multiplier for the cosine learning
                        rate scheduler""")

    return parser


def train(args, train_dset, canon, stn, c_ema, t_ema, canon_optim,
          canon_sched, t_optim, t_sched, loss_fn, nbb_loss_fn, device, writer):

    # Record modules to make saving checkpoints easier:
    if args.distributed:
        t_module = stn.module
        c_module = canon.module
    else:
        t_module = stn
        c_module = canon

    # Initialize Spatial Transformation Generator (Thin Plate Spline)
    aug = Augmentor(jitter=args.jitter, jitter_prob=args.jitter_prob,
                    gray_prob=args.gray_prob, solar_prob=args.solar_prob,
                    tps_scale=args.tps_scale).to(device)

    # A model checkpoint will be saved whenever the learning rate is zero:
    zero_lr_iters = lr_cycle_iters(0, args.period, args.iter, args.tm)
    early_ckpt_iters = set(zero_lr_iters)
    early_vis_iters = {100}
    early_vis_iters.update(early_ckpt_iters)

    # Initialize various training variables and constants:
    rec_loss = torch.tensor(0.0, device='cuda')
    flow_tv_loss = torch.tensor(0.0, device='cuda')
    nbb_loss = torch.tensor(0.0, device='cuda')
    equi_loss = torch.tensor(0.0, device='cuda')
    mask_loss = torch.tensor(0.0, device='cuda')
    parts_loss = torch.tensor(0.0, device='cuda')
    accum = 0.5 ** (32 / (10 * 1000))

    # Resize function for perceptual loss
    if args.unwarp_size != args.img_size:
        scale_factor = args.unwarp_size / args.img_size
        resize_fn = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                                align_corners=True)
    else:
        resize_fn = nn.Identity()

    # Pre-load on GPU
    # Assuming ~30 images of size 256x256, takes up ~23 MB device memory
    has_gt_kp = train_dset.kps is not None
    all_imgs = train_dset.imgs = train_dset.imgs.to(device)  # / 127.5 - 1.0
    all_masks = train_dset.masks = train_dset.masks.unsqueeze(1).to(device)
    all_parts = train_dset.parts = train_dset.parts.to(device)
    if has_gt_kp:
        all_kps = train_dset.kps = train_dset.kps.to(device)

    # Pseudo GT
    pseudo_kps = train_dset.pseudo_kps = torch.from_numpy(train_dset.pseudo_kps).to(device)
    num_parts = train_dset.num_parts
    loss_topk = pseudo_kps.shape[2] if args.sparse_topk is None else min(args.sparse_topk, pseudo_kps.shape[2])

    # Progress bar for monitoring training:
    pbar = range(args.start_iter, args.iter)
    if primary():
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True,
                    smoothing=0.2)
        pck_pairs, pck_cycles = log_visuals(
            c_ema, t_ema, train_dset, 0, writer, vis_sample=args.n_sample,
            vis_denseres=args.vis_denseres)
        best_pck_pairs = pck_pairs
        best_pck_cycles = pck_cycles

    requires_grad(stn, True)
    requires_grad(canon, True)

    for idx in pbar:  # main training loop
        i = idx + args.start_iter + 1

        ####################################
        #          TRAIN STN and CANON     #
        ####################################

        N = args.batch
        pairs = sample_tuples(len(train_dset), count=N // 2)
        src_idx, trg_idx = pairs[:, 0], pairs[:, 1]
        all_idx = np.concatenate([src_idx, trg_idx])
        batch_imgs = all_imgs[all_idx]
        batch_parts = all_parts[all_idx]

        if args.use_nbb_parts:
            batch_masks = (batch_parts != num_parts).unsqueeze(1).float()
            batch_masks_resized = resize_fn(batch_masks)
        else:
            batch_masks = all_masks[all_idx]
        batch_masks_resized = resize_fn(batch_masks)

        kp1 = pseudo_kps[src_idx, trg_idx][:, :loss_topk]      # (N/2, K, 4)
        kp2 = pseudo_kps[trg_idx, src_idx][:, :loss_topk]      # (N/2, K, 4)
        batch_kps_vis = kp1[..., 2] > 0                        # (N/2, K)
        batch_kps_wt = torch.ones_like(batch_kps_vis).float()  # (N/2, K)
        batch_kps = torch.cat([kp1, kp2])[..., :2]             # (N, K, 2)
        if args.use_nbb_parts:
            nbb_parts_vis = (kp1[..., 3] != args.num_parts) * (kp2[..., 3] != args.num_parts)
            batch_kps_wt *= nbb_parts_vis

        # Map the images to the canonical space
        flow, delta_flow = stn(batch_imgs)
        unwarped = canon.unwarp(flow, args.unwarp_size)

        # NBB weight
        if args.nbb_weight > 0.:
            nbb_loss = nbb_loss_fn(flow[:N//2], flow[N//2:],
                                   batch_kps[:N//2], batch_kps[N//2:],
                                   batch_kps_vis, batch_kps_wt)

        if args.equi_weight > 0.:
            # Apply tps transformations
            if args.disable_tps:
                batch_imgs_t = aug.forward_geom(aug.forward_color(batch_imgs))
                batch_masks_t = aug.forward_geom(batch_masks, fixed=True)
                # Apply tps to flow
                flow_tf = aug.forward_geom(flow.permute(0, 3, 1, 2), fixed=True).permute(0, 2, 3, 1)
            else:
                batch_imgs_t = aug.forward_tps(aug.forward_color(batch_imgs))
                batch_masks_t = aug.forward_tps(batch_masks, fixed=True)
                # Apply tps to flow
                flow_tf = aug.forward_tps(flow.permute(0, 3, 1, 2), fixed=True).permute(0, 2, 3, 1)

            batch_masks_t = torch.where(batch_masks_t > 0.5, 1., 0.)
            batch_masks_t_resized = resize_fn(batch_masks_t)
            vis = batch_masks_t * batch_masks


            # Flow of tps image
            flow_ft, _ = stn(batch_imgs_t)
            unwarped_ft = canon.unwarp(flow_ft, args.unwarp_size)

            equi_loss = F.l1_loss(flow_ft, flow_tf.detach(), reduction='none') \
                + F.l1_loss(flow_tf, flow_ft.detach(), reduction='none')
            equi_loss = (equi_loss * vis.squeeze(1).unsqueeze(-1)).mean()

        if args.mask_weight > 0:
            unwarped_mask = unwarped[:, [3]]
            mask_loss = F.binary_cross_entropy_with_logits(unwarped_mask, batch_masks_resized)

            if args.equi_weight > 0.:
                unwarped_ft_mask = unwarped_ft[:, [3]]
                mask_loss = 0.5 * mask_loss + \
                    0.5 * F.binary_cross_entropy_with_logits(
                        unwarped_ft_mask, batch_masks_t_resized)

        # Get Total Variation Loss on flow
        if args.flow_tv_weight > 0:
            flow_tv_loss = total_variation_loss(delta_flow)

        # Reconstruction loss
        if args.rec_weight > 0:
            unwarped = unwarped * batch_masks_resized
            resized_img = resize_fn(batch_imgs) * batch_masks_resized

            rec_loss = loss_fn(unwarped[:, :3], resized_img).mean()

            if args.equi_weight > 0.:
                unwarped_ft = unwarped_ft * batch_masks_t_resized
                resized_img = resize_fn(batch_imgs_t) * batch_masks_t_resized

                rec_loss = 0.5*rec_loss + 0.5 * loss_fn(unwarped_ft[:, :3], resized_img).mean()

        # Parts Loss
        if args.parts_weight > 0.:
            # Calculate the centroid of each part
            part_centroids = torch.zeros(num_parts+1, 2, dtype=torch.float,
                                         device=device)
            part_centroids.index_add_(0, batch_parts.reshape(-1),
                                      flow.reshape(-1, 2))
            part_counts = torch.bincount(batch_parts.reshape(-1)).float()
            part_centroids = (part_centroids/part_counts.unsqueeze(-1)).detach()

            # Compute the loss as the distance of the centroid from the flows
            parts_loss = F.l1_loss(flow, part_centroids[batch_parts],
                                   reduction='none')
            parts_loss = (parts_loss * batch_masks.squeeze(1).unsqueeze(-1)).mean()

        loss_dict = {"p": rec_loss, "ftv": flow_tv_loss,
                     "nbb": nbb_loss, "equi": equi_loss,  "mask": mask_loss,
                     'parts': parts_loss}

        canon.zero_grad()
        stn.zero_grad()
        full_stn_loss = args.rec_weight * rec_loss + \
            args.flow_tv_weight * flow_tv_loss + \
            args.nbb_weight * nbb_loss + args.equi_weight * equi_loss + \
            args.mask_weight * mask_loss + args.parts_weight * parts_loss
        full_stn_loss.backward()
        t_optim.step()
        epoch = max(0, i / args.period)
        t_sched.step(epoch)
        if args.canon_lr > 0:
            canon_optim.step()
            canon_sched.step(epoch)

        if args.stn_ema:
            accumulate(t_ema, t_module, accum)
        if args.canon_ema:
            accumulate(c_ema, c_module, accum)

        # Aggregate loss information across GPUs
        loss_reduced = reduce_loss_dict(loss_dict)

        if primary():
            # Display losses on the progress bar:
            perceptual_loss_val = loss_reduced["p"].mean().item()
            flow_tv_loss_val = loss_reduced["ftv"].mean().item()
            nbb_loss_val = loss_reduced["nbb"].mean().item()
            equi_loss_val = loss_reduced["equi"].mean().item()
            mask_loss_val = loss_reduced["mask"].mean().item()
            parts_loss_val = loss_reduced["parts"].mean().item()
            p_str = f"rec: {perceptual_loss_val:.4f}; " \
                if args.rec_weight > 0 else ""
            ftv_str = f"ftv: {flow_tv_loss_val:.6f}; " \
                if args.flow_tv_weight > 0 else ""
            nbb_str = f"nbb: {nbb_loss_val:.6f}; " \
                if args.nbb_weight > 0 else ""
            equi_str = f"equi: {equi_loss_val:.6f}; " \
                if args.equi_weight > 0 else ""
            mask_str = f"mask: {mask_loss_val:.6f}; " \
                if args.mask_weight > 0 else ""
            parts_str = f"parts: {parts_loss_val:.6f}; " \
                if args.parts_weight > 0 else ""
            pbar.set_description(
                f"{p_str}{nbb_str}{equi_str}{mask_str}{ftv_str}{parts_str}")

            # Log losses and others metrics to TensorBoard:
            if i % args.log_every == 0 or i in early_ckpt_iters or i == 1:
                writer.add_scalars('', {
                    'Loss/Full': full_stn_loss.item(),
                    'Loss/Reconstruction': perceptual_loss_val,
                    'Loss/TotalVariation': flow_tv_loss_val,
                    'Loss/NBB': nbb_loss_val,
                    'Loss/Equi': equi_loss_val,
                    'Loss/Mask': mask_loss_val,
                    'Loss/Parts': parts_loss_val,
                    'Progress/STN_LearningRate': t_sched.get_last_lr()[0],
                    'Progress/Canon_LearningRate': canon_sched.get_last_lr()[0] if args.canon_lr > 0 else 0.
                }, i)

            if (i % args.ckpt_every == 0 or i in early_ckpt_iters):
                save_state_dict(
                    'checkpoint', c_module, t_module, c_ema, t_ema,
                    canon_optim, canon_sched, t_optim, t_sched, args, i, True)

            if i % args.vis_every == 0 or i in early_vis_iters or i == 1:
                # Save visualizations to Tens orBoard
                if i in early_ckpt_iters:
                    pbar.write(f'{i:07}: LR = {t_sched.get_last_lr()[0]}')

                pck_pairs, pck_cycles = log_visuals(
                    c_ema, t_ema, train_dset, i, writer,
                    vis_sample=args.n_sample, vis_denseres=args.vis_denseres)

                if has_gt_kp and best_pck_cycles[2][0] < pck_cycles[2][0]:
                    best_pck_pairs = pck_pairs
                    for k, pck_cycle in enumerate(pck_cycles):
                        best_pck_cycles[k] = pck_cycle
                    save_state_dict(
                        'best', c_module, t_module, c_ema, t_ema,
                        canon_optim, canon_sched, t_optim, t_sched, args, i)

                pck_summary = {}
                if has_gt_kp:
                    pck_summary.update({
                        'Progress/PCK@0.10': pck_pairs[0] * 100,
                        'Progress/PCK@0.01': pck_pairs[-1] * 100,
                        'Progress/BestPCK@0.10': best_pck_pairs[0] * 100,
                        'Progress/BestPCK@0.01': best_pck_pairs[-1] * 100,
                    })

                for k, pck_cycle in enumerate(pck_cycles):
                    pck_summary[f'Progress/{k+2}-PCK@0.10'] = pck_cycle[0] * 100
                    pck_summary[f'Progress/{k+2}-PCK@0.01'] = pck_cycle[-1] * 100
                
                    if has_gt_kp:
                        pck_summary[f'Progress/Best{k+2}-PCK@0.10'] = best_pck_cycles[k][0] * 100
                        pck_summary[f'Progress/Best{k+2}-PCK@0.01'] = best_pck_cycles[k][-1] * 100

                writer.add_scalars('', pck_summary, i)


if __name__ == "__main__":
    device = "cuda"
    parser = base_training_argparse()
    args = parser.parse_args()

    # Setup distributed PyTorch and create results directory:
    args.distributed = setup_distributed()
    results_path = os.path.join(args.results, args.exp_name)
    if primary():
        # exp_id = hashlib.md5(args.exp_name.encode('utf-8')).hexdigest()
        use_wandb = not args.disable_wandb
        if use_wandb:
            wandb.init(project="asic", entity="kampta", name=args.exp_name,
                    reinit=True)
            wandb.config.update(args)
        writer = Logger(results_path, log_to_wandb=use_wandb)
        with open(f'{results_path}/opt.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        writer = None

    # Seed RNG:
    torch.manual_seed(args.seed * get_world_size() + get_rank())
    np.random.seed(args.seed * get_world_size() + get_rank())

    # UNet output is same size as input by default
    # When input are SSL features, we want to upsample
    # the flow when loss is computed in the image space
    # not upsammple the flow when loss is computed in the
    # SSL featuremap space

    # Initialize U-Net for regressing flow
    if args.flow_ssl:
        # in_size = extractor.num_patches
        # in_ch = extractor.feat_dim
        # TODO: read from the file and modfiy accordingly
        raise NotImplementedError
    else:
        in_size = args.img_size
        in_ch = 3
    stn = Asic(
        in_ch, in_size, mf=args.channel_multiplier, bilinear=args.bilinear,
        padding_mode=args.padding_mode, use_tanh=args.use_tanh).to(device)
    if args.stn_ema:
        t_ema = Asic(
            in_ch, in_size, mf=args.channel_multiplier, bilinear=args.bilinear,
            padding_mode=args.padding_mode).to(device)
        accumulate(t_ema, stn, 0)
    else:
        t_ema = stn

    if args.mask_weight > 0:
        num_ch = 4
    else:
        num_ch = 3

    if args.use_mlp:
        canon = CanonicalMLP(
                input_dim=2, output_dim=num_ch, hidden_dim=args.mlp_hidden_dim,
                skip_layers=args.mlp_skip_layers, num_layers=args.mlp_num_layers,
                resolution=args.canon_size).to(device)

    else:
        canon = Canonical((1, num_ch, args.canon_size, args.canon_size),
                          clamp=args.clamp).to(device)

    if args.canon_ema:
        if args.use_mlp:
            c_ema = CanonicalMLP(
                input_dim=2, output_dim=num_ch, hidden_dim=args.mlp_hidden_dim,
                skip_layers=args.mlp_skip_layers, num_layers=args.mlp_num_layers,
                resolution=args.canon_size).to(device)
        else:
            c_ema = Canonical((1, num_ch, args.canon_size, args.canon_size),
                              clamp=args.clamp).to(device)
        accumulate(c_ema, canon, 0)
    else:
        c_ema = canon

    # Setup the perceptual loss function:
    loss_fn = get_perceptual_loss(args.loss_fn, device)

    if args.nbb_weight > 0.:
        nbb_loss_fn = LossCorrsSparse(flow_size=in_size, T=args.sparse_temp)
        nbb_loss_fn = nbb_loss_fn.to(device)
    else:
        nbb_loss_fn = None

    if args.canon_lr == 0:
        requires_grad(canon, False)
        canon_optim = None
        canon_sched = None
    else:
        canon_optim = optim.Adam(canon.parameters(), lr=args.canon_lr,
                                 betas=(0.9, 0.999), eps=1e-8)
        canon_sched = DecayingCosineAnnealingWarmRestarts(
            canon_optim, T_0=1, T_mult=args.tm, decay=args.decay)

    if primary():
        print(f"{count_parameters(stn)} parameters in STN")
        print(f"{count_parameters(canon)} parameters in Canonical")
    # Setup optimizers and learning rate schedulers:
    t_optim = optim.Adam(stn.parameters(), lr=args.stn_lr, betas=(0.9, 0.999),
                         eps=1e-8)
    t_sched = DecayingCosineAnnealingWarmRestarts(
        t_optim, T_0=1, T_mult=args.tm, decay=args.decay)

    # See if the start iteration can be recovered when resuming training:
    args.start_iter = 0

    # Load pre-trained generator (and optionally resume from a GANgealing checkpoint):
    ckpt_path = Path(args.results) / args.exp_name / 'checkpoint.pt'
    try:
        print(f"Loading model from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        canon.load_state_dict(ckpt["canon"])
        c_ema.load_state_dict(ckpt["c_ema"])

        stn.load_state_dict(ckpt["t"])
        t_ema.load_state_dict(ckpt["t_ema"])

        t_optim.load_state_dict(ckpt["t_optim"])
        t_sched.load_state_dict(ckpt["t_sched"])

        if canon_optim is not None:
            canon_optim.load_state_dict(ckpt["canon_optim"])
        if canon_optim is not None:
            canon_sched.load_state_dict(ckpt["canon_sched"])

        args.start_iter = ckpt['iter']
        print(f"Checkpoint found. Resuming from {args.start_iter} iterations")

    except FileNotFoundError:
        print("No checkpoint found. Training from scratch.")

    except KeyError:
        raise Exception

    # Move models to DDP if distributed training is enabled:
    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        stn = nn.parallel.DistributedDataParallel(
            stn, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False)
        canon = nn.parallel.DistributedDataParallel(
            canon, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False)

    # Setup data
    if args.dset.lower() == 'folder':
        interim_dir = Path(args.img_dir).stem
        flow_dir = Path(args.flow_dir) / interim_dir / f'{args.bb}_s{args.bb_stride}'
        train_dset = InMemoryDataset(
            args.img_dir, img_size=args.img_size, num_parts=args.num_parts,
            mask_threshold=args.mask_threshold, flow_dir=flow_dir,
            use_coseg_masks=args.use_coseg_masks, every_k=args.every_k)
    elif args.dset.lower() == 'cub':
        interim_dir = f'{args.dset.lower()}/{args.split}/{args.cub_idx:03d}'
        flow_dir = Path(args.flow_dir) / interim_dir / f'{args.bb}_s{args.bb_stride}'
        train_dset = CUBDataset(
            args.img_dir, split=args.split, img_size=args.img_size,
            cls_idx=args.cub_idx, flow_dir=flow_dir,
            num_parts=args.num_parts,
            mask_threshold=args.mask_threshold,
            use_coseg_masks=args.use_coseg_masks)
    elif args.dset.lower() == 'spair':
        interim_dir = f'{args.dset.lower()}/{args.split}/{args.spair_cat}'
        flow_dir = Path(args.flow_dir) / interim_dir / f'{args.bb}_s{args.bb_stride}'
        train_dset = SpairDataset(
            args.img_dir, split=args.split, img_size=args.img_size,
            spair_cat=args.spair_cat, flow_dir=flow_dir,
            num_parts=args.num_parts, mask_threshold=args.mask_threshold,
            use_coseg_masks=args.use_coseg_masks)
    else:
        raise NotImplementedError

    # Begin training:
    train(args, train_dset, canon, stn, c_ema, t_ema,
          canon_optim, canon_sched, t_optim, t_sched, loss_fn,
          nbb_loss_fn, device, writer)
