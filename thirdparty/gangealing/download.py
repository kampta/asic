from torchvision.datasets.utils import download_url
import os
import shutil
import torch
from commons.distributed import primary, synchronize


# These are the pre-trained GANgealing checkpoints we currently have available for download (and the SSL VGG network)
VALID_MODELS = {'simclr_vgg_phase150'}


def find_model(model_name):
    if model_name in VALID_MODELS:
        using_pretrained_model = True
        return download_model(model_name), using_pretrained_model
    else:
        using_pretrained_model = False
        return torch.load(model_name, map_location=lambda storage, loc: storage), using_pretrained_model


def download_model(model_name, online_prefix='pretrained'):
    assert model_name in VALID_MODELS
    model_name = f'{model_name}.pt'  # add extension
    local_path = f'pretrained/{model_name}'
    if not os.path.isfile(local_path) and primary():  # download (only on primary process)
        web_path = f'http://efrosgans.eecs.berkeley.edu/gangealing/{online_prefix}/{model_name}'
        download_url(web_path, 'pretrained')
        local_path = f'pretrained/{model_name}'
    synchronize()  # Wait for the primary process to finish downloading the checkpoint
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


def download_lpips():
    local_path = f'pretrained/lpips_vgg_v0.1.pt'
    if not os.path.isfile(local_path) and primary():  # download (only on primary process)
        web_path = 'https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/vgg.pth'
        download_url(web_path, 'pretrained')
        shutil.move('pretrained/vgg.pth', local_path)
    synchronize()  # Wait for the primary process to finish downloading