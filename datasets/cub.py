import numpy as np
import os
import pandas as pd
import shutil
import torch

from pathlib import Path
from PIL import Image
from scipy.io import loadmat
import tarfile
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_file_from_google_drive, \
    download_url, extract_archive
from torchvision.utils import save_image

from datasets.utils import load_nbb, Augmentor


def python2_round(n):
    # ACSM is Python2; for parity, we use Python2 rounding in these utils.
    # https://stackoverflow.com/a/33019948
    from decimal import localcontext, Decimal, ROUND_HALF_UP
    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_UP
        rounded = Decimal(n).to_integral_value()
    return rounded


def perturb_bbox(bbox, pf=0., jf=0):
    '''
    Jitters and pads the input bbox.
    Args:
        bbox: Zero-indexed tight bbox.
        pf: padding fraction.
        jf: jittering fraction.
    Returns:
        pet_bbox: Jittered and padded box. Might have -ve or out-of-image coordinates
    '''
    pet_bbox = [coord for coord in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    pet_bbox[0] -= (pf*bwidth) + (1-2*np.random.random())*jf*bwidth
    pet_bbox[1] -= (pf*bheight) + (1-2*np.random.random())*jf*bheight
    pet_bbox[2] += (pf*bwidth) + (1-2*np.random.random())*jf*bwidth
    pet_bbox[3] += (pf*bheight) + (1-2*np.random.random())*jf*bheight

    return pet_bbox


def square_bbox(bbox, py2_round=True):
    '''
    Converts a bbox to have a square shape by increasing size along non-max dimension.
    '''
    round_fn = python2_round if py2_round else round
    sq_bbox = [int(round_fn(coord)) for coord in bbox]
    bwidth = sq_bbox[2] - sq_bbox[0] + 1
    bheight = sq_bbox[3] - sq_bbox[1] + 1
    maxdim = float(max(bwidth, bheight))

    dw_b_2 = int(round_fn((maxdim - bwidth) / 2.0))
    dh_b_2 = int(round_fn((maxdim - bheight) / 2.0))

    sq_bbox[0] -= dw_b_2
    sq_bbox[1] -= dh_b_2
    sq_bbox[2] = sq_bbox[0] + maxdim - 1
    sq_bbox[3] = sq_bbox[1] + maxdim - 1

    return sq_bbox


def load_CUB_keypoints(path):
    names = ['img_index', 'kp_index', 'x', 'y', 'visible']
    landmarks = pd.read_table(path, header=None, names=names,
                              delim_whitespace=True, engine='python')
    # (num_images, num_kps, 3)
    landmarks = landmarks.to_numpy().reshape((11788, 15, 5))[..., [2, 3, 4]]
    landmarks = torch.from_numpy(landmarks).float()
    return landmarks


def acsm_crop(img, bbox, bgval=0, border=True, py2_round=True):
    '''
    Crops a region from the image corresponding to the bbox.
    If some regions specified go outside the image boundaries, the pixel values are set to bgval.
    Args:
        img: image to crop
        bbox: bounding box to crop
        bgval: default background for regions outside image
    '''
    round_fn = python2_round if py2_round else round
    bbox = [int(round_fn(c)) for c in bbox]
    bwidth = bbox[2] - bbox[0] + 1
    bheight = bbox[3] - bbox[1] + 1

    im_shape = np.shape(img)
    im_h, im_w = im_shape[0], im_shape[1]

    nc = 1 if len(im_shape) < 3 else im_shape[2]

    img_out = np.ones((bheight, bwidth, nc), dtype=np.uint8) * bgval
    x_min_src = max(0, bbox[0])
    x_max_src = min(im_w, bbox[2] + 1)
    y_min_src = max(0, bbox[1])
    y_max_src = min(im_h, bbox[3] + 1)

    x_min_trg = x_min_src - bbox[0]
    x_max_trg = x_max_src - x_min_src + x_min_trg
    y_min_trg = y_min_src - bbox[1]
    y_max_trg = y_max_src - y_min_src + y_min_trg
    if border:
        img_in = img_out = img[y_min_src:y_max_src, x_min_src:x_max_src, :]
        left_pad = x_min_trg
        right_pad = bwidth - x_max_trg
        up_pad = y_min_trg
        down_pad = bheight - y_max_trg
        try:
            img_out = np.pad(
                img_out, mode='edge', pad_width=[
                    (up_pad, down_pad), (left_pad, right_pad), (0, 0)])
            assert ((img_out[y_min_trg:y_max_trg, x_min_trg:x_max_trg, :]
                     - img_in) ** 2).sum() == 0
            assert img_out.shape[0] == img_out.shape[1]
        except ValueError:
            print(f"""crop_shape: {img_out.shape},
                pad: {(up_pad, down_pad, left_pad, right_pad)},
                trg: {(y_min_trg, y_max_trg, x_min_trg, x_max_trg)},
                box: {(bheight, bwidth)}, img_shape: {im_shape}""")
            exit()
    else:
        img_out[y_min_trg:y_max_trg, x_min_trg:x_max_trg, :] = \
            img[y_min_src:y_max_src, x_min_src:x_max_src, :]
    return img_out


def preprocess_kps_box_crop(kps, bbox, size):
    # Once an image has been pre-processed via a box crop,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the cropped image.
    kps = kps.clone()
    kps[:, 0] -= bbox[0] + 1
    kps[:, 1] -= bbox[1] + 1
    w = 1 + bbox[2] - bbox[0]
    h = 1 + bbox[3] - bbox[1]
    assert w == h
    kps[:, [0, 1]] *= size / float(w)
    return kps


def cub_crop(img, target_res, bbox, border=True):
    # This function mimics ACSM's pre-processing used for the CUB dataset (up to image resampling and padding color)
    img = np.asarray(img)
    img = acsm_crop(img, bbox, 0, border=border)
    return Image.fromarray(np.squeeze(img)).resize(
        (target_res, target_res), Image.Resampling.LANCZOS)


def load_acsm_data(path, size=256, split='test', cls_idx=None):
    mat = loadmat(os.path.join(path, f'CUB_200_2011/cachedir/cub/data/{split}_cub_cleaned.mat'))
    files = [os.path.join(path, f'CUB_200_2011/images/{file[0]}')
             for file in mat['images']['rel_path'][0]]
    labels = [int(file[0].split('.')[0]) for file in mat['images']['rel_path'][0]]

    # These are the indices retained by ACSM (others are filtered):
    indices = [i[0, 0] - 1 for i in mat['images']['id'][0]]
    kps = load_CUB_keypoints(os.path.join(path, 'CUB_200_2011/parts/part_locs.txt'))[indices]
    b = mat['images']['bbox'][0]
    m = mat['images']['mask'][0]
    if cls_idx is not None:
        cls_ind = [i for i, c in enumerate(labels) if c==cls_idx]
        files = [files[i] for i in cls_ind]
        b = b[cls_ind]
        kps = kps[cls_ind]
        m = m[cls_ind]

    bboxes = []
    kps_out = []
    masks = []
    for ix, row in enumerate(b):
        x1, y1, x2, y2 = row[0, 0]
        bbox = np.array([x1[0, 0], y1[0, 0], x2[0, 0], y2[0, 0]]) - 1
        bbox = perturb_bbox(bbox, 0.05, 0)
        bbox = square_bbox(bbox)
        bboxes.append(bbox)
        kps_out.append(preprocess_kps_box_crop(kps[ix], bbox, size))
        masks.append(torch.from_numpy(np.array(cub_crop(np.expand_dims(m[ix], -1), size, bbox))))
    bboxes = np.stack(bboxes)
    kps_out = torch.stack(kps_out).float()
    masks = torch.stack(masks).float()
    assert bboxes.shape[0] == len(files)

    return files, bboxes, kps_out, masks


def download_cub(to_path):
    # Downloads the CUB-200-2011 dataset
    cub_dir = f'{to_path}/CUB_200_2011'
    if not os.path.isdir(cub_dir):
        tgz_path = f'{cub_dir}.tgz'
        print(f'Downloading CUB_200_2011 to {to_path}')
        cub_file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
        download_file_from_google_drive(cub_file_id, to_path)
        shutil.move(f'{to_path}/{cub_file_id}', tgz_path)
        extract_archive(tgz_path, remove_finished=True)
    else:
        print('Found pre-existing CUB directory')


def download_cub_metadata(to_path):
    # Downloads some metadata so we can use image pre-processing consistent with ACSM for CUB
    # Recommended by https://github.com/akanazawa/cmr/issues/3#issuecomment-451757610
    cache_dir = f'{to_path}/CUB_200_2011'
    cache_path = f'{cache_dir}/cachedir.tar.gz'

    if not os.path.isfile(cache_path):
        cache_url = f'https://www.dropbox.com/sh/ea3yprgrcjuzse5/AACgvZyPDkR9nWpPk-LJc9ATa/cachedir.tar.gz?dl=1'
        print('Downloading metadata used to form ACSM\'s CUB validation set')
        download_url(cache_url, cache_dir, filename="cachedir.tar.gz", md5="89842f8937a136bfdd7106e80f88d30f")
        with tarfile.open(cache_path) as f:
            # extracting file
            f.extractall(cache_dir)
    else:
        print('Found pre-existing meta data')


class CUBDataset(Dataset):
    def __init__(self, data_dir, split='test', img_size=256, cls_idx=1,
                 flow_dir=None, num_parts=0,
                 mask_threshold=1, use_coseg_masks=False, padding_mode='border'):
        super().__init__()
        self.img_size = img_size
        self.split = split
        self.cls_idx = cls_idx
        self.flow_dir = flow_dir
        self.num_parts = num_parts
        self.mask_threshold = mask_threshold
        self.fixed_pairs = None
        self.thresholds = None
        self.border = True if padding_mode=='border' else False

        os.makedirs(data_dir, exist_ok=True)
        download_cub(data_dir)
        download_cub_metadata(data_dir)

        self.files, self.bboxes, self.kps, self.masks = load_acsm_data(
            data_dir, size=img_size, split=split, cls_idx=cls_idx)

        imgs = []
        for i in range(len(self.files)):
            img = Image.open(self.files[i]).convert('RGB')
            img = cub_crop(img, self.img_size, self.bboxes[i], border=self.border)
            imgs.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))
        self.imgs = torch.stack(imgs) / 127.5 - 1.0  # normalize (-1, 1)

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
            self.masks = torch.from_numpy(np.stack(masks) > mask_threshold).float()

        self.parts = None
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
        self.pseudo_kps = None
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
    dset = CUBDataset(
        'raw_data', split='test', img_size=256)
