import os
import numpy as np
from pathlib import Path
import cv2
import torch
from PIL import Image
import kornia
import kornia.augmentation as K
import torch.nn as nn
from torchvision import transforms


PADDING_MODES = {
    'border': cv2.BORDER_REPLICATE,
    'zeros': cv2.BORDER_CONSTANT,
    'reflection': cv2.BORDER_REFLECT_101,
}


def load_nbb(nbb_dir, img_paths, parts=None):
    img_paths = [Path(f) for f in img_paths]
    num_images = len(img_paths)

    matches = [[[] for _ in range(num_images)] for _ in range(num_images)]
    max_kps = 0
    for i in range(0, num_images-1):
        fname_i = img_paths[i].stem
        for j in range(i+1, num_images):
            fname_j = img_paths[j].stem
            fname = os.path.join(nbb_dir, f'{fname_i}_{fname_j}.npy')
            d = np.load(fname, allow_pickle=True).item()
            kp1 = d['kp1']
            kp2 = d['kp2']
            sim = d['ranks_sim']
            # ranks = np.argsort(sim)[::-1]
            order = np.random.permutation(len(kp1))
            max_kps = max(max_kps, len(kp1))
            if parts is not None:
                kp1_parts = parts[i][kp1[order][:, 1], kp1[order][:, 0]]
                kp2_parts = parts[j][kp2[order][:, 1], kp2[order][:, 0]]
                matches[i][j] = np.concatenate([
                    kp1[order], sim[order][:, None], kp1_parts[:, None]], axis=1)
                matches[j][i] = np.concatenate([
                    kp2[order], sim[order][:, None], kp2_parts[:, None]], axis=1)
            else:
                matches[i][j] = np.concatenate([kp1[order], sim[order][:, None]],
                                               axis=1)
                matches[j][i] = np.concatenate([kp2[order], sim[order][:, None]],
                                               axis=1)

    # x, y, feature_sim, part_sim
    dim = 3 if parts is None else 4
    kps = np.zeros((num_images, num_images, max_kps, dim), dtype=np.float32)
    for i in range(0, num_images-1):
        for j in range(i+1, num_images):
            num_kps = len(matches[i][j])
            kps[i][j][:num_kps] = matches[i][j]
            # kps[i][j][:num_kps, 2] = 1

            kps[j][i][:num_kps] = matches[j][i]
            # kps[j][i][:num_kps, 2] = 1

    return kps


def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    kps *= kps[:, 2:3]  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale


class SquarePad:
    def __init__(self, padding_mode='border'):
        self.padding_mode = padding_mode

    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return transforms.functional.pad(image, padding, 0, self.padding_mode)


class Augmentor(nn.Module):
    def __init__(self, jitter=[0.4, 0.4, 0.2, 0.1], jitter_prob=0.8,
                 gray_prob=0.2, solar_prob=0.2, tps_scale=0.4):
        super().__init__()
        self.color_transform = K.AugmentationSequential(
            # https://github.com/facebookresearch/dino/blob/main/main_dino.py#L424
            K.ColorJitter(brightness=jitter[0], contrast=jitter[1],
                          saturation=jitter[2], hue=jitter[3], p=jitter_prob),
            K.RandomGrayscale(p=gray_prob),
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.1),
            K.RandomSolarize(0.1, 0.1, p=solar_prob),
        )

        self.perspective_transform = K.RandomPerspective(0.5, p=1.)
        self.affine_transform = K.RandomAffine(30, scale=(0.7, 1.1),
                                               padding_mode='border', p=1.0)
        self.elastic_transform = K.RandomElasticTransform(
            p=1.0, sigma=(16., 16.), alpha=(3, 3), padding_mode='border')

        # TPS doesn't support transforming points
        # Using it only for dense equivariance loss
        self.tps_transform = K.RandomThinPlateSpline(scale=tps_scale, p=1.)

    def forward(self, x):
        pass

    @torch.no_grad()
    def forward_color(self, img):
        return self.color_transform(img)

    @torch.no_grad()
    def forward_tps(self, img, fixed=False):
        if fixed:
            img_t = self.tps_transform(img, params=self.tps_transform._params)
        else:
            img_t = self.tps_transform(img)
        return img_t
    
    @torch.no_grad()
    def forward_geom(self, img, fixed=False):
        if fixed:
            img_t = self.elastic_transform(
                self.affine_transform(img, params=self.affine_transform._params),
                params=self.elastic_transform._params)
        else:
            img_t = self.elastic_transform(self.affine_transform(img))
        return img_t


    @torch.no_grad()
    def forward_perspective(self, img, fixed=False):
        if fixed:
            img_t = self.perspective_transform(img, params=self.perspective_transform._params)
        else:
            img_t = self.perspective_transform(img)
        return img_t

    @torch.no_grad()
    def forward_perspective_kp(self, kp):
        return kornia.geometry.transform_points(
            self.perspective_transform.transform_matrix, kp)