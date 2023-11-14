from torch.utils.tensorboard.writer import SummaryWriter

from PIL import Image
import torch
import torch.nn.functional as F
import wandb
import numpy as np

from commons.utils import images2grid, map_minmax, compute_pck, sample_tuples, \
    pck_loop
from commons.draw import splat_points, load_fg_points, \
    concat_v, get_colors, get_dense_colors, load_text_points
from thirdparty.colormap.colormap_flow import color_wheel_fast_smooth


@torch.inference_mode()
def log_visuals(canon, stn, dset, train_idx, writer, vis_sample=2,
                vis_denseres=32):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pseudo_kps = dset.pseudo_kps
    parts = dset.parts
    vis_sample = min(vis_sample, len(dset))
    res = dset.img_size
    has_gt_kp = dset.kps is not None
    has_fixed_pairs = dset.fixed_pairs is not None  # SPair

    # Run full test dataloader (assuming small dataset)
    all_imgs = dset.imgs
    all_masks = dset.masks
    all_kps = dset.kps
    all_flows, _ = stn(all_imgs)

    if has_gt_kp:
        kps_cols = torch.from_numpy(get_colors(all_kps.size(1))).float()
        kps_cols = map_minmax(kps_cols, 0, 1, -1, 1).to(device).unsqueeze(0)

    parts_cols = torch.from_numpy(get_colors(dset.num_parts+1)).float()
    parts_cols = map_minmax(parts_cols, 0, 1, -1, 1).to(device)
    parts_cols[-1] = 0

    # Text logging
    text_kp, text_kp_col = load_text_points('CVPR')
    text_kp = text_kp.to(device).unsqueeze(0)
    text_kp_col = text_kp_col.to(device).unsqueeze(0)

    pairs = sample_tuples(len(dset), count=vis_sample, seed=0)
    src_idx, trg_idx = pairs[:, 0], pairs[:, 1]

    # Log only once during the training
    if train_idx == 0:
        # Log images and the mask
        writer.log_image_grid(all_imgs[:vis_sample], 'img', train_idx,
                              vis_sample, nrow=vis_sample)
        writer.log_image_grid(all_imgs[:vis_sample]*all_masks[:vis_sample],
                              'img_mask', train_idx, vis_sample, nrow=vis_sample)

        # Log neural best buddies (sparse)
        kp1 = pseudo_kps[src_idx, trg_idx]
        kp2 = pseudo_kps[trg_idx, src_idx]
        kp_vis = kp1[..., -1] * kp2[..., -1]
        kp1, kp2 = kp1[..., :2], kp2[..., :2]
        colors = map_minmax(get_dense_colors(kp1), 0, 1, -1, 1)

        blend_src = splat_points(
            all_imgs[src_idx], kp1, sigma=3., opacity=1.0, colors=colors,
            alpha_channel=kp_vis.unsqueeze(-1))
        blend_trg = splat_points(
            all_imgs[trg_idx], kp2, sigma=3., opacity=1.0, colors=colors,
            alpha_channel=kp_vis.unsqueeze(-1))
        stacked = torch.stack([blend_src, blend_trg], dim=1).flatten(0, 1)

        writer.log_image_grid(stacked, 'kp_pseudo_gt', train_idx, 2*vis_sample,
                              log_mean_img=False, nrow=2)

        # Log parts
        parts_img = parts_cols[parts[:vis_sample]].permute(0, 3, 1, 2)
        writer.log_image_grid(parts_img, 'parts', train_idx, vis_sample,
                              nrow=vis_sample, log_mean_img=False)

        # Log groundtruth kp
        if has_gt_kp:
            kp1, kp2 = all_kps[src_idx], all_kps[trg_idx]
            kp_vis = kp1[..., -1] * kp2[..., -1]
            kp1, kp2 = kp1[..., :2], kp2[..., :2]

            colors = kps_cols.expand(vis_sample, -1, -1)
            blend_src = splat_points(
                all_imgs[src_idx], kp1, sigma=3., opacity=1.0, colors=colors,
                alpha_channel=kp_vis.unsqueeze(-1))
            blend_trg = splat_points(
                all_imgs[trg_idx], kp2, sigma=3., opacity=1.0, colors=colors,
                alpha_channel=kp_vis.unsqueeze(-1))
            stacked = torch.stack([blend_src, blend_trg], dim=1).flatten(0, 1)

            stacked = torch.stack([blend_src, blend_trg], dim=1).flatten(0, 1)
            writer.log_image_grid(stacked, 'kp_gt', train_idx, 2*vis_sample,
                                log_mean_img=False, nrow=2)

    # Log kp and top predictions by STN (if kp are available)
    if has_gt_kp:
        kp1 = all_kps[src_idx][..., :2]
        kp_vis = all_kps[src_idx][..., 2]

        kp_pred = stn.transfer_points(
            kp1, src_idx, trg_idx, all_flows, mask=all_masks, res=res, is_flow=True)
        colors = kps_cols.expand(vis_sample, -1, -1)

        blend_src = splat_points(
            all_imgs[src_idx], kp1, sigma=3., opacity=1.0,
            colors=colors, alpha_channel=kp_vis.unsqueeze(-1))
        blend_trg = splat_points(
            all_imgs[trg_idx], kp_pred.float(), sigma=3., opacity=1.0,
            colors=colors, alpha_channel=kp_vis.unsqueeze(-1))

        stacked = torch.stack([blend_src, blend_trg], dim=1).flatten(0, 1)
        writer.log_image_grid(stacked, 'kp_pred_sparse', train_idx,
                            2*vis_sample, log_mean_img=False, nrow=2)

    # Log current canon image
    canon_grid = canon.get_grid(vis_sample)
    if canon_grid.size(1) > 3:
        canon_grid = canon_grid[:, :3]
    scale_factor = res / canon_grid.size(-1)
    canon_grid = F.interpolate(
        canon_grid, scale_factor=scale_factor, mode='bilinear')
    writer.log_image_grid(canon_grid, 'canon', train_idx, 1, log_mean_img=False)

    # Log dense correspondences
    kp, kp_vis, kp_col_dense = load_fg_points(all_masks[src_idx],
                                              resolution=vis_denseres)
    kp_pred, kp_canon = stn.transfer_points(
        kp, src_idx, trg_idx, all_flows, mask=all_masks, res=res,
        return_canon=True, is_flow=True)
    colors = map_minmax(kp_col_dense, 0, 1, -1, 1)

    blend_src = splat_points(
        all_imgs[src_idx], kp, sigma=4., opacity=0.75,
        colors=colors, alpha_channel=kp_vis.unsqueeze(-1))

    blend_trg = splat_points(
        all_imgs[trg_idx], kp_pred.float(), sigma=4., opacity=0.75,
        colors=colors, alpha_channel=kp_vis.unsqueeze(-1))

    blend_canon = splat_points(
        torch.ones_like(canon_grid) * -1, kp_canon, sigma=1.3, opacity=1.0,
        colors=colors, alpha_channel=kp_vis.unsqueeze(-1))
    stacked = torch.stack([blend_src, blend_canon, blend_trg], dim=1).\
        flatten(0, 1)
    writer.log_image_grid(
        stacked, 'kp_pred_dense', train_idx, 3*vis_sample,
        log_mean_img=False, nrow=3)

    # # Log dense correspondences with text
    # text_kp = text_kp.expand(vis_sample, -1, -1)
    # text_kp_col = text_kp_col.expand(vis_sample, -1, -1)
    # kp_pred, kp_canon = stn.transfer_points(
    #     text_kp, src_idx, trg_idx, all_flows, mask=all_masks, res=res,
    #     return_canon=True, is_flow=True)

    # blend_src = splat_points(all_imgs[src_idx], text_kp, sigma=0.7, opacity=1.,
    #                          colors=text_kp_col)

    # blend_trg = splat_points(all_imgs[trg_idx], kp_pred.float(), sigma=0.7,
    #                          opacity=1., colors=text_kp_col)

    # blend_canon = splat_points(torch.ones_like(canon_grid) * -1, kp_canon,
    #                            sigma=0.7, opacity=1., colors=text_kp_col)

    # stacked = torch.stack([blend_src, blend_canon, blend_trg], dim=1).\
    #     flatten(0, 1)
    # writer.log_image_grid(
    #     stacked, 'kp_pred_text', train_idx, 3*vis_sample,
    #     log_mean_img=False, nrow=3)

    # Log dense mapping from canonical space to Image space
    wheel = color_wheel_fast_smooth(res).permute(2, 0, 1).unsqueeze(0).to(device)
    colors = wheel.expand(vis_sample, -1, -1, -1)
    flow, _ = stn(all_imgs[src_idx])
    colors = F.grid_sample(colors, flow, padding_mode='border',
                           align_corners=True)
    colors = map_minmax(colors, 0, 1, -1, 1)
    alpha = 0.5
    blend_img = alpha * all_imgs[src_idx] * (1-all_masks[src_idx]) + \
        (all_imgs[src_idx] * alpha + colors * (1-alpha)) * all_masks[src_idx]
    blend_img = torch.cat([wheel, blend_img, wheel, colors* all_masks[src_idx]])
    writer.log_image_grid(blend_img, 'canon_map', train_idx, len(blend_img),
                          log_mean_img=False, nrow=len(blend_img)//2)

    # Log keypoints from Image space to canonical space
    if has_gt_kp:
        canon_corrs = stn.transfer_forward(all_flows, all_kps[..., :2], res, is_flow=True)
        canon_corrs = stn.unnormalize(canon_corrs, res, res)
        canon_vis = all_kps[..., -1]
        num_kp = canon_vis.size(-1)
        N = canon_vis.size(0)
        colors = kps_cols.permute(1, 0, 2).expand(-1, N, -1).to(device)
        heatmaps = splat_points(
            torch.ones(num_kp, 3, res, res, device=device) * -1,
            canon_corrs.permute(1, 0, 2), sigma=6., opacity=1.,
            colors=colors, alpha_channel=canon_vis.permute(1, 0).unsqueeze(-1))
        writer.log_image_grid(heatmaps, 'kp_heatmaps', train_idx,
                                num_kp, padding=2, pad_value=1.)

    # Log parts from Image space to canonical space
    # Splat one part at a time to canonical
    # TODO: splat all at once
    num_parts = dset.num_parts
    part_kp_canons = []
    part_kp_vis = [] 
    for part in range(num_parts):
        part_masks = (parts == part).float().unsqueeze(1)
        kp, kp_vis, _ = load_fg_points(part_masks, resolution=vis_denseres)
        kp_canon = stn.transfer_forward(all_flows, kp[..., :2], res, is_flow=True)
        kp_canon = stn.unnormalize(kp_canon, res, res)
        part_kp_canons.append(kp_canon.reshape(-1, 2))
        part_kp_vis.append(kp_vis.reshape(-1))

    part_kp_canons = torch.stack(part_kp_canons)
    part_kp_vis = torch.stack(part_kp_vis)
    colors = parts_cols[:-1].unsqueeze(1).expand(-1, part_kp_vis.size(1), -1)
    heatmaps = splat_points(
        torch.ones(num_parts, 3, res, res, device=device) * -1,
        part_kp_canons, sigma=2., opacity=1.,
        colors=colors, alpha_channel=part_kp_vis.unsqueeze(-1))
    writer.log_image_grid(heatmaps, 'part_heatmaps', train_idx,
                          num_parts, padding=2, pad_value=1.)

    # Compute PCKs
    N = all_imgs.size(0)
    transfer_fn = stn.transfer_points
    pck_pairs = None
    if has_gt_kp:
        # First compute PCK for all 2-pairs
        if has_fixed_pairs:
            tuples = dset.fixed_pairs
            if dset.thresholds is not None:
                thresholds = [torch.from_numpy(dset.thresholds)[tuples[:, 1]]]
            else:
                thresholds = None
        else:
            tuples = sample_tuples(N)
            thresholds = None
        print(f"First computing 2-point PCK for {len(tuples)} pairs")
        gt_corrs, pred_corrs, vis = pck_loop(
            tuples, all_kps, transfer_fn, all_flows, all_masks, res,
            return_canon=False, is_flow=True)
        pck_pairs = compute_pck(pred_corrs, gt_corrs, vis, thresholds,
                                img_size=res)

    # Compute k-cycle PCK
    pck_cycles = []
    if not has_gt_kp:
        kp, kp_vis, kp_col_dense = load_fg_points(all_masks,
                                            resolution=vis_denseres)
        ignore_idx = kp_vis.sum(dim=0) == 0
        all_kps = torch.cat([kp[:, ~ignore_idx], kp_vis[:, ~ignore_idx].unsqueeze(-1)], dim=2)
        ignore_interim = True
    else:
        ignore_interim = False

    for k in [2, 3, 4]:
        tuples = sample_tuples(N, k=k, count=200)
        if has_fixed_pairs and dset.thresholds is not None:
            thresholds = torch.from_numpy(dset.thresholds[tuples[:, 1:]])
            thresholds = thresholds.reshape(-1)
        else:
            thresholds = None
        print(f"Next computing {k}-cycle PCK for {len(tuples)} tuples")
        gt_corrs, pred_corrs, vis = pck_loop(
            tuples, all_kps, transfer_fn, all_flows, all_masks, res,
            return_canon=False, is_flow=True, ignore_interim=ignore_interim)
        pck = compute_pck(pred_corrs, gt_corrs, vis, thresholds, img_size=res)
        pck_cycles.append(pck)

    return pck_pairs, pck_cycles


class Logger(SummaryWriter):

    def __init__(self, results_path, log_to_tb=False, log_to_wandb=True):
        super().__init__(results_path)
        self.results_path = results_path
        self.log_to_tb = log_to_tb
        self.log_to_wandb = log_to_wandb

    def _log_image_grid(self, images, logging_name, prefix, itr, range=(-1, 1),
                        scale_each=False, nrow=None, **kwargs):
        nrow = max(1, int(len(images) ** 0.5+0.5)) if nrow is None else nrow
        if type(images[0]) is torch.Tensor:
            ndarr = images2grid(images, return_as_PIL=True, nrow=nrow,
                                normalize=True, value_range=range,
                                scale_each=scale_each, **kwargs)
            grid = Image.fromarray(ndarr)
            grid.save(f"{self.results_path}/{logging_name}_{str(itr).zfill(7)}.png")
            if self.log_to_wandb:
                wandb.log({logging_name: wandb.Image(grid)}, step=itr)
        else:
            grid = concat_v(*images)
            grid.save(f"{self.results_path}/{logging_name}_{str(itr).zfill(7)}.png")
            if self.log_to_wandb:
                wandb.log({logging_name: [wandb.Image(im) for im in images]}, step=itr)

        if self.log_to_tb:
            self.add_image(f"{prefix}/{logging_name}", ndarr, itr,
                           dataformats='HWC')

    def log_image_grid(self, images, logging_name, itr, imgs_to_show,
                       log_mean_img=True, mean_range=None, range=(-1, 1),
                       scale_each=False, num_heads=1, nrow=None, **kwargs):
        self._log_image_grid(images[:imgs_to_show], logging_name, "grids", itr,
                             range=range, scale_each=scale_each, nrow=nrow, **kwargs)
        if log_mean_img:  # Log average images:
            images = images.reshape(images.size(0) // num_heads, num_heads,
                                    *images.size()[1:])
            self._log_image_grid(images.mean(dim=0), f'mean_{logging_name}',
                                 "means", itr, range=mean_range,
                                 scale_each=True, nrow=nrow)

    def add_scalar(self, tag, scalar_value, global_step=None, **kwargs):
        if self.log_to_wandb:
            wandb.log({tag: scalar_value}, step=global_step)
        return super().add_scalar(tag, scalar_value, global_step, **kwargs)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, **kwargs):
        if self.log_to_wandb:
            wandb.log(tag_scalar_dict, step=global_step)
        return super().add_scalars(main_tag, tag_scalar_dict, global_step, **kwargs)