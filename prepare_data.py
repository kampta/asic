import argparse
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets.utils import download_file_from_google_drive
from torchvision.utils import save_image

from datasets.cub import CUBDataset
from datasets.spair import SpairDataset
from models.utils import sample_from_reverse_flow
from commons.utils import str2bool

from commons.draw import draw_kps, get_dense_colors, splat_points, load_fg_points
from thirdparty.MLS.mls import mls_rigid_deformation
from commons.utils import map_minmax
from thirdparty.dino_vit_features.cosegmentation import coseg_from_feat
from thirdparty.dino_vit_features.extractor import ViTExtractor
from thirdparty.dino_vit_features.correspondences import corrs_from_feat
from thirdparty.dino_vit_features.part_cosegmentation import parts_from_feat
from thirdparty.DIS.isnet import ISNetDIS


@torch.no_grad()
def extract_features_and_saliency_maps(
        extractor, img_size, layer, facet, bin, transform, dset, out_dir,
        device):

    images_list = []
    descriptors_list = []
    saliency_maps_list = []

    num_patches = extractor.get_num_patches(img_size, img_size)[0]
    image_paths = dset.files
    # Extract features and saliency maps
    print("Extracting features and saliency maps")
    feat_dir = out_dir / f'feat_l{layer}_f{facet}_b{bin:1d}'
    feat_dir.mkdir(exist_ok=True, parents=True)

    saliency_map_dir = out_dir / 'saliency'
    saliency_map_dir.mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(len(dset))):
        img = dset.imgs[i].to(device)
        img_unnorm = img * 0.5 + 0.5
        img_np = ((img_unnorm) * 255).permute(1, 2, 0).cpu().numpy()
        images_list.append(img_np.astype(np.uint8))
        img_norm = transform(img_unnorm).unsqueeze(0)

        fname = Path(image_paths[i]).stem
        # Extract and save features
        feat_fname = feat_dir / f'{fname}.npy'
        if feat_fname.is_file():
            feat = np.load(feat_fname)
        else:
            feat = extractor.extract_descriptors(img_norm, layer, facet, bin)
            feat = feat.cpu().squeeze().numpy()
            np.save(feat_fname, feat)
        descriptors_list.append(feat)
        
        sal_fname = saliency_map_dir / f'{fname}.png'
        if sal_fname.is_file():
            saliency_map = Image.open(sal_fname).convert('L')
            saliency_map = np.array(saliency_map).astype(np.float32) / 255
            saliency_map = saliency_map.reshape(-1)
        else:
            saliency_map = extractor.extract_saliency_maps(img_norm)
            saliency_map = saliency_map.squeeze().cpu().numpy()
            saliency_map = saliency_map.reshape(num_patches, num_patches)
            saliency_map = Image.fromarray((saliency_map * 255).astype(np.uint8))
            saliency_map.save(sal_fname)
            saliency_map = np.array(saliency_map).astype(np.float32) / 255
            saliency_map = saliency_map.reshape(-1)
        saliency_maps_list.append(saliency_map)

    return images_list, descriptors_list, saliency_maps_list


def save_cosegmentations(extractor, dset, out_dir, img_size, transform, device,
        layer=11, facet='key', bin=False, thresh=0.065, elbow=0.975,
        votes_percentage=75, sample_interval=100):
    print("Running co-segmentation on collection of images")

    images_list, descriptors_list, saliency_maps_list = \
        extract_features_and_saliency_maps(
            extractor, img_size, layer, facet, bin, transform, dset, out_dir,
            device)
    image_paths = dset.files

    # Run cosegmentation
    print("Computing masks")
    segmentation_masks = coseg_from_feat(
        images_list, descriptors_list, saliency_maps_list,
        img_size, extractor.get_num_patches(img_size, img_size)[0],
        elbow, thresh, votes_percentage, sample_interval)
    masks_dir = out_dir / 'masks_coseg'
    masks_dir.mkdir(exist_ok=True, parents=True)
    for i in tqdm(range(len(dset))):
        fname = Path(image_paths[i]).stem
        mask_fname = masks_dir / f'{fname}.png'
        segmentation_masks[i].save(mask_fname)


@torch.no_grad()
def save_bg(model_path, dset, out_dir, in_size, device):
    net=ISNetDIS()

    model_path = Path(model_path)
    if not model_path.exists():
        model_id = "1nV57qKuy--d5u1yvkng9aXW1KS4sOpOi"
        download_file_from_google_drive(model_id, model_path.parent, filename=model_path.name)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net = net.to(device)
    net.eval()
    image_paths = dset.files

    out_dir = out_dir / 'masks'
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Computing masks")
    for i in tqdm(range(len(dset))):
        img = dset.imgs[i].to(device)
        # From [-1, 1] to [-0.5, 0.5]
        img = img / 2.0
        img = F.upsample(img.unsqueeze(0), in_size, mode='bilinear')
        mask = net(img)
        mask = torch.squeeze(F.upsample(mask[0][0], dset.img_size, mode='bilinear'), 0)
        ma = torch.max(mask)
        mi = torch.min(mask)
        mask = (mask-mi)/(ma-mi)

        fname = Path(image_paths[i]).stem
        mask_fname = out_dir / f'{fname}.png'
        mask = (mask.squeeze() * 255).cpu().numpy()
        Image.fromarray(mask.astype(np.uint8)).save(mask_fname)


@torch.no_grad()
def save_correspondences(extractor, dset, out_dir, img_size, transform, device,
        layer=9, facet='key', bin=False):
    print("Saving NBB for all pairs of images")

    _, descriptors_list, saliency_maps_list = \
        extract_features_and_saliency_maps(
            extractor, img_size, layer, facet, bin, transform, dset, out_dir, device)
    image_paths = dset.files
    num_patches = extractor.get_num_patches(img_size, img_size)[0]
    masks_dir = out_dir / 'masks'
    masks_list = []
    for i in tqdm(range(len(dset))):
        fname = Path(image_paths[i]).stem
        mask_fname = masks_dir / f'{fname}.png'
        mask = Image.open(mask_fname).convert('L')
        masks_list.append(mask)

    matches_dir = out_dir / f'nbb'
    matches_dir.mkdir(exist_ok=True, parents=True)

    descriptors_list = torch.stack([
        torch.from_numpy(descriptor).to(device)
        for descriptor in descriptors_list
    ])
    for i in tqdm(range(len(dset)-1)):
        img1 = dset.imgs[i].to(device)
        fname1 = Path(image_paths[i]).stem
        feat1 = descriptors_list[i]
        mask1 = masks_list[i]
        saliency_map1 = saliency_maps_list[i]

        for j in range(i+1, len(dset)):
            img2 = dset.imgs[j].to(device)
            fname2 = Path(image_paths[j]).stem
            feat2 = descriptors_list[j]
            mask2 = masks_list[j]
            saliency_map2 = saliency_maps_list[j]

            fname = matches_dir / f'{fname1}_{fname2}.npy'
            if fname.exists():
                continue

            pt1, pt2, pt1_idx, pt2_idx, ranks_sal, ranks_sim = corrs_from_feat(
                feat1, feat2, saliency_map1, saliency_map2,
                num_patches, extractor.stride[0], extractor.p, device,
                mask1, mask2)

            # Save the output
            fname = matches_dir / f'{fname1}_{fname2}.npy'
            d = {
                'kp1': pt1.cpu().numpy().astype(np.int32),
                'kp2': pt2.cpu().numpy().astype(np.int32),
                'kp1_idx': pt1_idx,
                'kp2_idx': pt2_idx,
                'ranks_attn': ranks_sal,
                'ranks_sim': ranks_sim.cpu().numpy(),
            }
            np.save(fname, d)

            # Save sparse correspondences
            colors = get_dense_colors(pt1, img_size)
            colors = colors.to(device).unsqueeze(0).expand(2, -1, -1)
            sparse_corrs = splat_points(
                torch.stack([img1, img2], dim=0),
                torch.stack([pt1, pt2]).float().to(device),
                sigma=2., opacity=1.0, colors=map_minmax(colors, 0, 1, -1, 1))
            fname = matches_dir / f'{fname1}_{fname2}.jpg'
            save_image(sparse_corrs, fname, normalize=True, padding=2, pad_value=1)


@torch.no_grad()
def save_mls(extractor, dset, out_dir, img_size, transform, device,
        layer=9, facet='key', bin=False, mls_num=None, mls_alpha=1.):
    print("Converting NBB to MLS for all pairs of images")
    _, descriptors_list, saliency_maps_list = \
        extract_features_and_saliency_maps(
            extractor, img_size, layer, facet, bin, transform, dset, out_dir, device)
    image_paths = dset.files
    num_patches = extractor.get_num_patches(img_size, img_size)[0]
    masks_dir = out_dir / 'masks'
    masks_list = []

    for i in tqdm(range(len(dset))):
        fname = Path(image_paths[i]).stem
        mask_fname = masks_dir / f'{fname}.png'
        mask = Image.open(mask_fname).convert('L')
        masks_list.append(mask)

    matches_dir = out_dir / f'nbb'
    matches_dir.mkdir(exist_ok=True, parents=True)

    descriptors_list = torch.stack([
        torch.from_numpy(descriptor).to(device)
        for descriptor in descriptors_list
    ])

    if mls_num is not None:
        flow_dir = out_dir / f'mls_num{mls_num}_alpha{mls_alpha}'
    else:
        flow_dir = out_dir / f'mls_alpha{mls_alpha}'
    flow_dir.mkdir(exist_ok=True, parents=True)

    for i in tqdm(range(len(dset)-1)):
        img1 = dset.imgs[i].to(device)
        fname1 = Path(image_paths[i]).stem
        mask1 = masks_list[i]
        mask1 = torch.from_numpy(np.array(mask1)>0).to(device)

        for j in range(i+1, len(dset)):
            torch.cuda.empty_cache()
            img2 = dset.imgs[j].to(device)
            fname2 = Path(image_paths[j]).stem
            mask2 = masks_list[j]
            mask2 = torch.from_numpy(np.array(mask2)>0).to(device)

            fname = matches_dir / f'{fname1}_{fname2}.npy'
            d = np.load(fname, allow_pickle=True).item()
            kp1 = d['kp1']
            kp1_idx = d['kp1_idx']
            kp2 = d['kp2']
            kp2_idx = d['kp2_idx']
            ranks_attn = d['ranks_attn']

            # Run kmeans to get a few well distributed keypoints
            # if mls_num is not None:
            #     use_indices = kmeans_correspondences(
            #         feat1[kp1_idx], feat2[kp2_idx], ranks_attn, mls_num)
            #     use_indices = use_indices.astype(np.int32)
            # else:
            use_indices = np.arange(len(kp1_idx))

            # Save sparse correspondences (from kmeans)
            sparse_corrs = draw_kps(
                img1, img2, kp1[use_indices], kp2[use_indices], lines=False)
            fname = flow_dir / f'sparse_{fname1}_{fname2}.jpg'
            sparse_corrs.save(fname)

            # Reverse flow from correspondences (MLS)
            flow21 = mls_rigid_deformation(
                torch.from_numpy(kp1[use_indices]).to(device),
                torch.from_numpy(kp2[use_indices]).to(device),
                alpha=mls_alpha, resolution=img_size)
            flow21 = flow21.permute(1, 2, 0)
            flow12 = mls_rigid_deformation(
                torch.from_numpy(kp2[use_indices]).to(device),
                torch.from_numpy(kp1[use_indices]).to(device),
                alpha=mls_alpha, resolution=img_size)
            flow12 = flow12.permute(1, 2, 0)

            fname = flow_dir / f'{fname1}_{fname2}.npy'
            np.save(fname, flow12.cpu().numpy())
            fname = flow_dir / f'{fname2}_{fname1}.npy'
            np.save(fname, flow21.cpu().numpy())

            # Dense correspondence (1 to 2) from MLS
            pt1_fg, pt1_fg_alpha, colors = load_fg_points(
                mask1.unsqueeze(0), resolution=img_size // 2)
            pt1_fg_to_2 = sample_from_reverse_flow(
                flow21.unsqueeze(0).float(), pt1_fg)
            colors = colors.to(device).expand(2, -1, -1)
            dense_corrs = splat_points(
                torch.stack([img1, img2], dim=0),
                torch.cat([pt1_fg, pt1_fg_to_2]).float(),
                sigma=1.3, opacity=0.75,
                colors=map_minmax(colors, 0, 1, -1, 1),
                alpha_channel=pt1_fg_alpha.unsqueeze(-1).expand(2, -1, -1)
            )
            img1_warped = F.grid_sample(
                img1.unsqueeze(0),
                map_minmax(flow21, 0, img_size, -1, 1).unsqueeze(0),
                align_corners=True)

            fname = flow_dir / f'dense_{fname1}_{fname2}.jpg'
            save_image(torch.cat([dense_corrs, img1_warped]), fname,
                       normalize=True, padding=2, pad_value=1)

            # Dense correspondence (2 to 1) from MLS
            pt2_fg, pt2_fg_alpha, colors = load_fg_points(
                mask2.unsqueeze(0), resolution=img_size // 2)
            pt2_fg_to_1 = sample_from_reverse_flow(
                flow12.unsqueeze(0).float(), pt2_fg)
            colors = colors.to(device).expand(2, -1, -1)
            dense_corrs = splat_points(
                torch.stack([img2, img1], dim=0),
                torch.cat([pt2_fg, pt2_fg_to_1]).float(),
                sigma=1.3, opacity=0.75,
                colors=map_minmax(colors, 0, 1, -1, 1),
                alpha_channel=pt2_fg_alpha.unsqueeze(-1).expand(2, -1, -1)
            )
            img2_warped = F.grid_sample(
                img2.unsqueeze(0),
                map_minmax(flow12, 0, img_size, -1, 1).unsqueeze(0),
                align_corners=True)

            fname = flow_dir / f'dense_{fname2}_{fname1}.jpg'
            save_image(torch.cat([dense_corrs, img2_warped]), fname,
                       normalize=True, padding=2, pad_value=1)


@torch.no_grad()
def save_part_cosegmentations(
        extractor, dset, out_dir, num_parts, img_size, transform, device,
        layer=11, facet='key', bin=False, thresh=0.065, elbow=0.975,
        votes_percentage=75, sample_interval=100, num_crop_augmentations=5,
        three_stages=False, elbow_second_stage=0.94):

    images_list, descriptors_list, _ = \
        extract_features_and_saliency_maps(
            extractor, img_size, layer, facet, bin, transform, dset, out_dir,
            device)

    image_paths = dset.files
    num_patches = extractor.get_num_patches(img_size, img_size)[0]
    masks_dir = out_dir / 'masks'
    masks_list = []
    for i in tqdm(range(len(dset))):
        fname = Path(image_paths[i]).stem
        mask_fname = masks_dir / f'{fname}.png'
        mask = Image.open(mask_fname).convert('L')
        mask = mask.resize((num_patches, num_patches), resample=Image.LANCZOS)
        mask = torch.from_numpy(np.array(mask).reshape(-1)).to(device) / 255.
        masks_list.append(mask)

    if num_parts > 0:
        parts_dir = out_dir / f'parts_num{num_parts}'
    else:
        parts_dir = out_dir / f'parts'
    parts_dir.mkdir(exist_ok=True, parents=True)

    parts_from_feat(
        extractor, layer, facet, bin, transform, descriptors_list,
        masks_list, images_list, dset.files, img_size, device, elbow,
        thresh, votes_percentage, sample_interval, num_parts,
        num_crop_augmentations, three_stages, elbow_second_stage, parts_dir)


def main():
    parser = argparse.ArgumentParser(description='Preprocess images')
    # Input
    parser.add_argument("--dset", type=str, default='cub',
                        choices=['cub', 'spair'],
                        help="data type")
    parser.add_argument("--img_dir", type=str, required= True, help="Path to images")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image size")

    # Output
    parser.add_argument("--out_dir", type=str, default='data',
                        help="Output path")

    # Cub and spair dataset arguments
    parser.add_argument("--cub_idx", type=int, default=1, help="cub category")
    parser.add_argument("--spair_cat", default='cat', help="spair category")
    parser.add_argument("--split", default='test', help="split")

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

    # Tasks
    parser.add_argument("--run", default=None,
                        choices=['coseg', 'bg', 'corrs', 'parts', 'mls', None],
                        help="To run")

    # Co-segmentation hyperparams
    parser.add_argument('--bg_layer', default=11, type=int)
    parser.add_argument('--bg_facet', default='key')
    parser.add_argument('--bg_bin', default=False, type=str2bool)
    parser.add_argument('--bg_thresh', default=0.065, type=float)
    parser.add_argument('--bg_elbow', default=0.975, type=float)
    parser.add_argument('--bg_votes_percentage', default=75, type=float)
    parser.add_argument('--bg_sample_interval', default=100, type=int)

    # BG-segmentation hyperparams
    parser.add_argument("--bg_raw_size", type=int, default=1024, help="Image size")
    parser.add_argument("--bg_model_path", default='thirdparty/DIS/isnet-general-use.pth',
                        help="Path to pretrained weights")

    # Correspondence Hyperparameters
    parser.add_argument('--nbb_layer', default=9, type=int)
    parser.add_argument('--nbb_facet', default='key')
    parser.add_argument('--nbb_bin', default=False, type=str2bool)
    parser.add_argument('--mls_num', default=None, type=int,
                        help="number of points for MLS")
    parser.add_argument('--mls_alpha', default=1., type=float,
                        help="rigidity coefficient")

    # Parts Hyperparameters
    parser.add_argument('--num_parts', default=4, type=int, help="number of parts")
    # Co-segmentation hyperparams
    parser.add_argument('--parts_layer', default=11, type=int)
    parser.add_argument('--parts_facet', default='key')
    parser.add_argument('--parts_bin', default=False, type=str2bool)
    parser.add_argument('--parts_thresh', default=0.065, type=float)
    parser.add_argument('--parts_elbow', default=0.975, type=float)
    parser.add_argument('--parts_votes_percentage', default=20, type=float)
    parser.add_argument('--parts_sample_interval', default=100, type=int)
    parser.add_argument('--parts_num_crop_augmentations', default=1, type=int)
    parser.add_argument('--parts_three_stages', default=False, type=str2bool)
    parser.add_argument('--parts_elbow_second_stage', default=0.94, type=float)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dset.lower() == 'cub':
        dset = CUBDataset(
            args.img_dir, split=args.split, img_size=args.img_size,
            cls_idx=args.cub_idx)
        base_dir = f'{args.dset.lower()}/{args.split}/{args.cub_idx:03d}'
    elif args.dset.lower() == 'spair':
        dset = SpairDataset(
            args.img_dir, split=args.split, img_size=args.img_size,
            spair_cat=args.spair_cat)
        base_dir = f'{args.dset.lower()}/{args.split}/{args.spair_cat}'
    else:
        raise NotImplementedError

    out_dir = Path(args.out_dir) / base_dir / f'{args.model_type}_s{args.stride}'
    out_dir.mkdir(exist_ok=True, parents=True)

    # Background subtraction
    if args.run is None or args.run == 'bg':
        # save_cosegmentations(
        #     extractor, dset, out_dir, args.img_size, transform, device,
        #     layer=args.bg_layer, facet=args.bg_facet, bin=args.bg_bin,
        #     thresh=args.bg_thresh, elbow=args.bg_elbow,
        #     votes_percentage=args.bg_votes_percentage,
        #     sample_interval=args.bg_sample_interval)
        save_bg(args.bg_model_path, dset, out_dir, args.bg_raw_size, device)

    # Neural Best Buddies
    transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    extractor = ViTExtractor(args.model_type, args.stride, device=device)
    if args.run is None or args.run == 'corrs':
        save_correspondences(
            extractor, dset, out_dir, args.img_size, transform, device,
            layer=args.nbb_layer, facet=args.nbb_facet, bin=args.nbb_bin)

    # Dense correspondence from best buddies
    if args.run == 'mls':
        save_mls(extractor, dset, out_dir, args.img_size, transform, device,
                 args.nbb_layer, args.nbb_facet, args.nbb_bin, args.mls_num,
                 args.mls_alpha)

    # Part Co-Segmentation
    if args.run is None or args.run == 'parts':
        save_part_cosegmentations(
            extractor, dset, out_dir, args.num_parts, args.img_size, transform,
            device, layer=args.parts_layer, facet=args.parts_facet,
            bin=args.parts_bin, thresh=args.parts_thresh, elbow=args.parts_elbow,
            votes_percentage=args.parts_votes_percentage,
            sample_interval=args.parts_sample_interval,
            num_crop_augmentations=args.parts_num_crop_augmentations,
            three_stages=args.parts_three_stages,
            elbow_second_stage=args.parts_elbow_second_stage)


if __name__ == '__main__':
    main()
