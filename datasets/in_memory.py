import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms

from datasets.utils import load_nbb


class InMemoryDataset(Dataset):
    def __init__(self, data_dir, img_size=256, flow_dir=None,
                 num_parts=0,  mask_threshold=1, use_coseg_masks=False,
                 every_k=1):

        self.img_size = img_size
        self.flow_dir = flow_dir
        self.num_parts = num_parts
        self.mask_threshold = mask_threshold

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])

        files = []
        imgs = []
        for base_dir, dirnames, filenames in os.walk(data_dir):
            if len(dirnames) > 0:
                continue
            for f in sorted(filenames):
                if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                filename = Path(base_dir) / f
                files.append(filename)
                img = Image.open(filename).convert('RGB')
                imgs.append(transform(img))
        
        self.files = files[::every_k]
        self.imgs = torch.stack(imgs[::every_k])

        self.kps = None
        self.fixed_pairs = None
        self.thresholds = None
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
                fname = mask_dir / f'{self.files[i].stem}.png'
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
                    fname = parts_dir / f'parts_s2_{self.files[i].stem}.npy'
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