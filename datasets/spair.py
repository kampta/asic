import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from glob import glob
import json
from torchvision import transforms
from pathlib import Path
from datasets.utils import load_nbb, preprocess_kps_pad, SquarePad
from torchvision.datasets.utils import download_and_extract_archive


def load_spair_data(path, size=256, category='cat', split='test',
                    subsample=-1, seed=42):
    pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*{category}.json'))
    assert len(pairs) > 0, '# of groundtruth image pairs must be > 0'
    if subsample > 0:
        np.random.seed(seed)
        pairs = [pairs[ix] for ix in np.random.choice(len(pairs), subsample)]
    print(f'Number of SPairs for {category} = {len(pairs)}')
    category_anno = list(glob(f'{path}/ImageAnnotation/{category}/*.json'))[0]
    with open(category_anno) as f:
        num_kps = len(json.load(f)['kps'])
    print(f'Number of SPair key points for {category} <= {num_kps}')
    files = []
    kps = []
    thresholds = []
    blank_kps = torch.zeros(num_kps, 3)

    fixed_pairs = []
    for pair in pairs:
        with open(pair) as f:
            data = json.load(f)
        assert category == data["category"]
        assert data["mirror"] == 0
        source_fn = f'{path}/JPEGImages/{category}/{data["src_imname"]}'
        target_fn = f'{path}/JPEGImages/{category}/{data["trg_imname"]}'
        source_bbox = np.asarray(data["src_bndbox"])
        target_bbox = np.asarray(data["trg_bndbox"])

        source_size = data["src_imsize"][:2]  # (W, H)
        target_size = data["trg_imsize"][:2]  # (W, H)

        if source_fn not in files:
            files.append(source_fn)
            kps.append(torch.zeros(num_kps, 3))
            thresholds.append(0)
            source_idx = len(files) - 1
        else:
            source_idx = files.index(source_fn)
        if target_fn not in files:
            files.append(target_fn)
            kps.append(torch.zeros(num_kps, 3))
            thresholds.append(0)
            target_idx = len(files) - 1
        else:
            target_idx = files.index(target_fn)

        kp_ixs = [int(id) for id in data["kps_ids"]]
        kp_ixs = torch.tensor(kp_ixs).view(-1, 1).repeat(1, 3)
        source_raw_kps = torch.cat([torch.tensor(data["src_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        source_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=source_raw_kps)
        source_kps, *_, scale_src = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)
        kps[source_idx] = source_kps
        target_raw_kps = torch.cat([torch.tensor(data["trg_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        target_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=target_raw_kps)
        target_kps, *_, scale_trg = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)
        kps[target_idx] = target_kps
        fixed_pairs.append([source_idx, target_idx])
        threshold_src = max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0])
        threshold_trg = max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0])
        thresholds[source_idx] = threshold_src*scale_src
        thresholds[target_idx] = threshold_trg*scale_trg

    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    print(f'Final number of used key points: {kps.size(1)}')
    return files, kps, fixed_pairs, thresholds


def download_spair(to_path):
    # Downloads and extracts the SPair-71K dataset
    spair_dir = f'{to_path}/SPair-71k'
    if not os.path.isdir(spair_dir):
        print(f'Downloading SPair-71k to {to_path}')
        spair_url = 'http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz'
        download_and_extract_archive(spair_url, to_path, remove_finished=True)
    else:
        print('Found pre-existing SPair-71K directory')
    return spair_dir


class SpairDataset(Dataset):
    def __init__(self, data_dir, split='test', img_size=256, spair_cat='cat',
                 flow_dir=None, padding_mode='edge',  num_parts=0,
                 mask_threshold=1, use_coseg_masks=False):
        super().__init__()
        self.img_size = img_size
        self.split = split
        self.cat = spair_cat
        self.padding_mode = padding_mode
        self.flow_dir = flow_dir
        self.num_parts = num_parts
        self.mask_threshold = mask_threshold

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            SquarePad(padding_mode),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize,
        ])

        os.makedirs(data_dir, exist_ok=True)
        spair_dir = download_spair(data_dir)

        self.files, self.kps, fixed_pairs, thresholds = load_spair_data(
            spair_dir, size=img_size, split=split, category=spair_cat)
        imgs = [transform(Image.open(self.files[i]).convert('RGB'))
                for i in range(len(self))]
        self.imgs = torch.stack(imgs)
        self.fixed_pairs = np.array(fixed_pairs)
        self.thresholds = np.array(thresholds)

        self.masks = torch.ones(len(self), 1, img_size, img_size)
        self.pseudo_kps = None
        self.parts = None

        # Load masks
        if flow_dir is not None:
            if use_coseg_masks:
                mask_dir = Path(flow_dir) / 'masks_coseg'
            else:
                mask_dir = Path(flow_dir) / 'masks'
            assert mask_dir.exists(), f"{mask_dir} doesn't exist"
            masks = []
            for i in range(0, len(self)):
                fname = mask_dir / f'{Path(self.files[i]).stem}.png'
                mask = np.array(Image.open(fname).convert('L'))
                masks.append(mask)
            self.masks = torch.from_numpy(np.stack(masks) >= mask_threshold).float()

        # Load parts
        if flow_dir is not None:
            parts_str = 'parts' if num_parts <=0 else f'parts_num{num_parts}'
            parts_dir = Path(flow_dir) / f'{parts_str}'
            if parts_dir.exists():
                parts = []
                for i in range(0, len(self)):
                    fname = parts_dir / f'parts_s2_{Path(self.files[i]).stem}.npy'
                    part = np.load(fname)
                    parts.append(part)
                parts = np.stack(parts)
                num_parts = int(np.max(parts[~np.isnan(parts)])) + 1
                parts[np.isnan(parts)] = num_parts

                self.parts = torch.from_numpy(parts.astype(np.int64))
            else:
                print(f"{parts_dir} doesn't exist. Parts won't load.")
            self.num_parts = num_parts
            # self.parts = F.one_hot(parts, num_classes=num_parts+1).bool()
        
        # Load pseudo keypoints
        if flow_dir is not None:
            nbb_dir = Path(flow_dir) / 'nbb'
            if nbb_dir.exists():
                self.pseudo_kps = load_nbb(nbb_dir, self.files, self.parts)
                max_matches = self.pseudo_kps.shape[2]
                print(f'Max #matches between an image pair: {max_matches}')
            else:
                print(f"{nbb_dir} doesn't exist. Pseudo kps won't load.")

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    from torchvision.utils import save_image
    from datasets.utils import Augmentor
    from commons.draw import splat_points

    dset = train_dset = SpairDataset(
        'data/spair', split='test', img_size=256,
        spair_cat='cat', flow_dir='data/processed/spair/test/cat/dino_vits8_s2',
        num_parts=4, mask_threshold=1)

    device = 'cuda'
    all_imgs = dset.imgs = train_dset.imgs.to(device)
    all_imgs = all_imgs[:8]
    all_masks = dset.masks = train_dset.masks.unsqueeze(1).to(device)
    all_masks = all_masks[:8]
    all_kps = dset.kps = train_dset.kps.to(device)
    all_kps = all_kps[:8]

    # Check images, masks and keypoints
    save_image(all_masks, 'test_mask.png', normalize=True)
    out = splat_points(
        all_imgs, all_kps[..., :2], sigma=3., opacity=1.,
        alpha_channel=all_kps[..., 2].unsqueeze(-1))
    save_image(out, 'test.png', normalize=True)

    # Initialize augmentor
    aug = Augmentor()

    # Check perspective transforms (images, masks and keypoints)
    all_imgs_transform = aug.forward_perspective(all_imgs)

    all_masks_transform = aug.forward_perspective(all_masks, fixed=True)
    save_image(all_masks_transform, 'test_mask_perspective.png', normalize=True)

    all_kps_transform = aug.forward_perspective_kp(all_kps[..., :2])
    out = splat_points(
        all_imgs_transform, all_kps_transform, sigma=3., opacity=1.,
        alpha_channel=all_kps[..., 2].unsqueeze(-1))
    save_image(out, 'test_perspective.png', normalize=True)

    # Check TPS transforms (images and masks, no keypoints)
    all_imgs_transform = aug.forward_tps(all_imgs)
    save_image(all_imgs_transform, 'test_tps.png', normalize=True)

    all_masks_transform = aug.forward_tps(all_masks, fixed=True)
    save_image(all_masks_transform, 'test_mask_tps.png', normalize=True)

if __name__ == "__main__":
    dset = SpairDataset(
        'raw_data', split='test', img_size=256)
