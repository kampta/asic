from PIL import Image, ImageDraw, ImageColor, ImageFont
import seaborn as sns
import numpy as np
import torch
import plotly.colors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from math import pi as PI
import torch.nn.functional as F

from thirdparty.colormap.colormap_flow import color_wheel_fast_smooth
from commons.utils import map_minmax
from thirdparty.gangealing.laplacian_blending import LaplacianBlender
from thirdparty.gangealing.splat2d_cuda import splat2d

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True


def get_colors(N):
    # colors = torch.tensor(sns.color_palette(n_colors=N))
    if N > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap([
            "red", "yellow", "blue", "lime", "magenta", "indigo", "orange",
            "cyan", "darkgreen", "maroon", "black", "white", "chocolate",
            "gray", "blueviolet"])
    colors = np.array([cmap(x)[:3] for x in range(N)])

    return colors

def draw_kps(src_img, trg_img, src_kps, trg_kps, kps_colors=None, lines=True):
    # Expects kps in (x, y) order to make it compatible with splat_points
    if type(src_img) is torch.Tensor:
        src_img = (src_img.permute(1, 2, 0) + 1)*127.5
        src_img = Image.fromarray(src_img.cpu().numpy().astype(np.uint8))
    else:
        src_img = src_img.copy()
    if type(trg_img) is torch.Tensor:
        trg_img = (trg_img.permute(1, 2, 0) + 1)*127.5
        trg_img = Image.fromarray(trg_img.cpu().numpy().astype(np.uint8))
    else:
        trg_img = trg_img.copy()

    if type(src_kps) is torch.Tensor:
        src_kps = src_kps.cpu().numpy()

    if type(trg_kps) is torch.Tensor:
        trg_kps = trg_kps.cpu().numpy()

    if kps_colors is None:
        # kps_colors = ['black'] * len(src_kps)
        # kps_colors = np.array(sns.color_palette(n_colors=len(src_kps)))
        kps_colors = get_colors(len(src_kps))
        kps_colors = (kps_colors * 255).astype(np.uint8)
        kps_colors = [tuple(col) for col in kps_colors]
    elif type(kps_colors) is torch.Tensor:
        kps_colors = (kps_colors * 255).cpu().numpy().astype(np.uint8)
        kps_colors = [tuple(col) for col in kps_colors]

    src_imsize = src_img.size
    trg_imsize = trg_img.size

    assert len(src_kps) == len(trg_kps), \
        'The number of matching key-points NOT same'

    src_draw = ImageDraw.Draw(src_img)
    trg_draw = ImageDraw.Draw(trg_img)

    kps_radius = 4   # if lines else 1.5

    for kp_id, (src_kp, trg_kp) in enumerate(zip(src_kps, trg_kps)):         
        src_draw.ellipse((src_kp[0] - kps_radius, src_kp[1] - kps_radius,
                          src_kp[0] + kps_radius, src_kp[1] + kps_radius),
                         fill=kps_colors[kp_id], outline='white')
        trg_draw.ellipse((trg_kp[0] - kps_radius, trg_kp[1] - kps_radius,
                          trg_kp[0] + kps_radius, trg_kp[1] + kps_radius),
                         fill=kps_colors[kp_id], outline='white')

    total_width = src_imsize[0] + trg_imsize[0]
    total_height = max(src_imsize[1], trg_imsize[1])
    des_img = Image.new("RGB", (total_width, total_height), color='black')

    new_im = Image.new('RGB', (total_width, total_height))
    new_im.paste(src_img, (0, 0))
    new_im.paste(trg_img, (src_imsize[0], 0))
    new_im.paste(des_img, (0, max(src_imsize[1], trg_imsize[1])))
    new_im_draw = ImageDraw.Draw(new_im)

    if lines:
        for kp_id, (src_kp, trg_kp) in enumerate(zip(src_kps, trg_kps)):
            new_im_draw.line(
                (src_kp[0], src_kp[1], trg_kp[0] + src_imsize[1], trg_kp[1]),
                fill=kps_colors[int(kp_id)], width=2)
    return new_im


def get_dense_colors(points, resolution=256):
    colors = color_wheel_fast_smooth(resolution)
    if len(points.shape) == 2:
        return colors[points[:, 0], points[:, 1]]
    else:
        device = points.device
        N = len(points)
        colors = colors.permute(2, 0, 1).unsqueeze(0).expand(N, -1, -1, -1)
        points = map_minmax(points, 0, resolution-1, -1, 1).unsqueeze(-2)
        colors = F.grid_sample(colors.to(device), points, align_corners=False)
        return colors.squeeze(-1).permute(0, 2, 1)


def concat_h(*argv, pad=0):
    width = 0
    height = 0
    count = len(argv)

    for img in argv:
        width += img.width
        height = max(height, img.height)

    dst = Image.new('RGB', (width + (count-1)*pad, height))
    start = 0
    for i, img in enumerate(argv):
        dst.paste(img, (start, 0))
        start += img.width + pad
    return dst


def concat_v(*argv, pad=0):
    width = 0
    height = 0
    count = len(argv)

    for img in argv:
        height += img.height
        width = max(width, img.width)

    dst = Image.new('RGB', (width, height + (count-1)*pad))
    start = 0
    for i, img in enumerate(argv):
        dst.paste(img, (0, start))
        start += img.height + pad
    return dst


def concat_th(*argv, pad=0):
    width = 0
    count = 0
    height = 0
    for img in argv:
        width += img.squeeze().shape[-1]
        height = max(height, img.shape[-2])
        count += 1

    dst = torch.zeros(3, height, width + (count-1)*pad)
    start = 0
    for i, img in enumerate(argv):
        dst[:, :img.shape[-2], start:start+img.shape[-1]] = img.squeeze()
        start += img.shape[-1] + pad
    return dst


def concat_np(*argv, pad=0):
    fig = plt.figure(figsize=(5*len(argv), 4))
    for i, img in enumerate(argv):
        if type(img) is torch.Tensor:
            img = img.cpu().numpy()
        ax = fig.add_subplot(1, len(argv), i+1)
        ax.axis('off')
        if img.shape[0] != 1 and len(img.shape) == 2:
            img = img[None]

        if img.shape[0] == 1:
            ax.imshow(img[0], vmin=0, vmax=img.max(), cmap='turbo')
        else:
            ax.imshow(img)

    plt.tight_layout()
    return fig


def make_composite(img, mask, mask_color=[173, 216, 230]):
    rgb = np.copy(img)
    rgb[mask] = (rgb[mask] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)

    return Image.fromarray(rgb)


def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale_name)

    if hasattr(loc, "__iter__"):
        intermediate_colors = [get_continuous_color(colorscale, x)
                               for x in loc]
        return intermediate_colors
    return get_continuous_color(colorscale, loc)


def get_plotly_colors(num_points, colorscale):
    color_steps = torch.linspace(start=0, end=1, steps=num_points).tolist()
    colors = get_color(colorscale, color_steps)
    colors = [plotly.colors.unlabel_rgb(color) for color in colors]
    colors = torch.tensor(
        colors, dtype=torch.float, device='cuda').view(1, num_points, 3)
    # Map [0, 255] RGB colors to [-1, 1]
    colors = colors.div(255.0).add(-0.5).mul(2)
    return colors  # (1, P, 3)


@torch.inference_mode()
def splat_points_large(images, points, sigma, opacity, colors=None,
                       alpha_channel=None, max_batch_size=12, **kwargs):
    # max_batch_size should be something that fits on a single gpu
    # currently tested with img_size of (256, 256) and 16G gpu
    results = []
    for start_idx in range(0, len(images), max_batch_size):
        end_idx = min(start_idx+max_batch_size, len(images))

        sigma = sigma if type(sigma) == float else sigma[start_idx:end_idx]
        batch_colors = colors if colors is None else colors[start_idx:end_idx]
        batch_alpha_channel = alpha_channel if alpha_channel is None \
            else alpha_channel[start_idx:end_idx]

        out = splat_points(images[start_idx:end_idx], points[start_idx:end_idx],
                 sigma, opacity, colors=batch_colors,
                 alpha_channel=batch_alpha_channel, **kwargs)
        results.append(out)

    return torch.cat(results, dim=0)


@torch.inference_mode()
def splat_points(images, points, sigma, opacity, colorscale='turbo',
                 colors=None, alpha_channel=None, blend_alg='alpha'):
    """
    Highly efficient GPU-based splatting algorithm. This function is a wrapper
    for Splat2D to overlay points on images. For highest performance, use the
    colors argument directly instead of colorscale.
    images: (N, C, H, W) tensor in [-1, +1]
    points: (N, P, 2) tensor with values in [0, resolution - 1]
            (can be sub-pixel/non-integer coordinates)
            Can also be (N, K, P, 2) tensor, in which case points[:, i]
            gets a unique colorscale
            Expects points in (x, y) order.
    sigma: either float or (N,) tensor with values > 0
           controls the size of the splatted points
    opacity: float in [0, 1], controls the opacity of the splatted points
    colorscale: [Optional] str (or length-K list of str if points is size
                (N, K, P, 2)) indicating the Plotly colorscale to visualize
                points with
    colors: [Optional] (N, P, 3) tensor (or (N, K*P, 3)). If specified,
            colorscale will be ignored. Computing the colorscale
            often takes several orders of magnitude longer than the GPU-based
            splatting, so pre-computing the colors and passing them here
            instead of using the colorscale argument can provide a significant
            speed-up.
    alpha_channel: [Optional] (N, P, 1) tensor (or (N, K*P, 1)). If specified,
                    colors will be blended into the output image based on the
                    opacity values in alpha_channel (between 0 and 1).
    blend_alg: [Optiona] str. Specifies the blending algorithm to use when
               merging points into images. Can use alpha compositing ('alpha'),
               Laplacian Pyramid Blending ('laplacian') or a more conservative
               version of Laplacian Blending ('laplacian_light')
    :return (N, C, H, W) tensor in [-1, +1] with points splatted onto images
    """
    assert images.dim() == 4  # (N, C, H, W)
    assert points.dim() == 3 or points.dim() == 4  # (N, P, 2) or (N, K, P, 2)
    batch_size = images.size(0)
    # each index in the second dimension gets a unique colorscale
    if points.dim() == 4:
        num_points = points.size(2)
        points = points.reshape(
            points.size(0), points.size(1) * points.size(2), 2)  # (N, K*P, 2)
        if colors is None:
            if isinstance(colorscale, str):
                colorscale = [colorscale]
            assert len(colorscale) == points.size(1)
            # (1, K*P, 3)
            colors = torch.cat([
                get_plotly_colors(num_points, c) for c in colorscale], 1)
            colors = colors.repeat(batch_size, 1, 1)  # (N, K*P, 3)
    elif colors is None:
        num_points = points.size(1)
        # All batch elements use the same colorscale
        if isinstance(colorscale, str):
            # (N, P, 3)
            colors = get_plotly_colors(
                points.size(1), colorscale).repeat(batch_size, 1, 1)
        else:  # Each batch element uses its own colorscale
            assert len(colorscale) == batch_size
            colors = torch.cat([get_plotly_colors(num_points, c)
                                for c in colorscale], 0)
    if alpha_channel is None:
        alpha_channel = torch.ones(
            batch_size, points.size(1), 1, device='cuda')
    if isinstance(sigma, (float, int)):
        sigma = torch.tensor(
            sigma, device='cuda', dtype=torch.float).view(1).repeat(batch_size)
    blank_img = torch.zeros(batch_size, images.size(1), images.size(2),
                            images.size(3), device='cuda')
    blank_mask = torch.zeros(batch_size, 1, images.size(2), images.size(3),
                             device='cuda')
    # (N, C, H, W)
    prop_obj_img = splat2d(blank_img, points, colors, sigma, False)
    # (N, 1, H, W)
    prop_mask_img = splat2d(blank_mask, points, alpha_channel, sigma, True)
    prop_mask_img *= opacity
    if blend_alg == 'alpha':
        # basic alpha-composite
        out = prop_mask_img * prop_obj_img + (1 - prop_mask_img) * images
    elif blend_alg == 'laplacian':
        blender = LaplacianBlender().to(images.device)
        out = blender(images, prop_obj_img, prop_mask_img)
    elif blend_alg == 'laplacian_light':
        blender = LaplacianBlender(levels=3, gaussian_kernel_size=11,
                                   gaussian_sigma=0.5).to(images.device)
        out = blender(images, prop_obj_img, prop_mask_img)
    return out


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1].
    This function computes the intermediate
    color for any value in that range.
    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]
    Others are just swatches that need to be constructed into a colorscale:
        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(
            plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)
    :param colorscale: A plotly continuous colorscale defined with RGB string
                       colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    def hex_to_rgb(c): return "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    intermediate_color = plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )
    return intermediate_color


def square_to_sphere(resolution=512):
    us = torch.linspace(0, 1, steps=resolution+2)[1:-1]
    vs = torch.linspace(0, 1, steps=resolution+2)[1:-1]
    us, vs = torch.meshgrid(us, vs, indexing='xy')

    # Given two random uniform samplers u and v
    # theta = arccos(2 * u - 1) - pi/2
    # phi = 2 * pi * v
    # x = cos(theta) * cos(phi)
    # y = cos(theta) * sin(phi)
    # z = sin(theta)

    theta = torch.acos(2 * us.reshape(-1) - 1) - PI/2
    phi = 2 * PI * vs.reshape(-1)
    cos_theta = torch.cos(theta)
    X = cos_theta * torch.cos(phi)
    Y = cos_theta * torch.sin(phi)
    Z = torch.sin(theta)
    return torch.stack([X, Y, Z]).permute(1, 0)


def square_to_grid(resolution=512):
    us = torch.linspace(0, 1, steps=resolution+2)[1:-1]
    vs = torch.linspace(0, 1, steps=resolution+2)[1:-1]
    us, vs = torch.meshgrid(us, vs, indexing='xy')
    return torch.stack([us.reshape(-1), vs.reshape(-1)]).permute(1, 0)


def plot_3d(points, colors=None):
    if len(points.shape) == 2:
        points = points.reshape(1, *points.shape)
        colors = None if colors is None else colors.reshape(1, *colors.shape)
    elif len(points.shape) != 3:
        raise(f"Expected tensor of 2 or 3 dims. Got {points.shape} dims")

    points = points.cpu()
    batch_size = points.shape[0]
    fig = plt.figure(figsize=(5*batch_size, 4))
    for i in range(batch_size):
        ax = fig.add_subplot(1, batch_size, i+1, projection='3d')
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2],
                   c=colors[i], cmap="rgb")
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_zlabel('w')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
    plt.tight_layout()

    return fig


def plot_2d(points, colors=None, base_img=None, splat=False, size=512):
    device = points.device
    if len(points.shape) == 2:
        points = points.reshape(1, *points.shape)
        colors = None if colors is None else colors.reshape(1, *colors.shape)
    elif len(points.shape) != 3:
        raise(f"Expected tensor of 2 or 3 dims. Got {points.shape} dims")

    if base_img is None:
        base_img = torch.zeros(points.size(0), 3, size, size).to(device)

    if not splat:
        base_img = base_img.permute(0, 2, 3, 1)
        points = map_minmax(points, 0, 1, 0, size)
        points = torch.clamp(points.long(), 0, size-1)
        base_img[np.arange(points.size(0)).repeat(points.size(1)),
                 points[:, :, 1].reshape(-1), points[:, :, 0].reshape(-1)] = \
            colors.reshape(-1, 3)
        base_img = base_img.permute(0, 3, 1, 2)
    else:
        size = base_img.size(-1)
        points = map_minmax(points, 0, 1, 0, size-1)
        blend = splat_points(
            base_img, points[:, :, [1, 0]],
            sigma=2., opacity=0.9,
            colors=map_minmax(colors, 0, 1, -1, 1).to(device))
        base_img = map_minmax(blend, -1, 1, 0, 1)

    return base_img


def load_fg_points(img_mask, resolution=None, normalize=False, device='cuda'):
    # returns points in XY format
    if resolution is None:
        resolution = img_mask.size(-1)
    us = vs = torch.arange(resolution)
    us, vs = torch.meshgrid(us, vs, indexing='xy')
    points = torch.stack([us.reshape(-1), vs.reshape(-1)]).permute(1, 0)
    points = points.unsqueeze(0).expand(img_mask.size(0), -1, -1)
    points = points.to(device)

    img_mask = img_mask.float()
    if len(img_mask.shape) == 3:
        img_mask = img_mask.unsqueeze(1)
    scale_factor = resolution / img_mask.size(2)
    if resolution != img_mask.size(2):  # resize the mask:
        img_mask = F.interpolate(img_mask, scale_factor=scale_factor,
                                 mode='bilinear')

    img_mask = img_mask.squeeze(1)
    points_alpha = img_mask.reshape(img_mask.size(0), -1)
    points = points / (resolution-1)
    if not normalize:
        points *= (img_mask.size(2)/scale_factor-1)

    colors = color_wheel_fast_smooth(resolution).to(device)
    colors = colors.reshape(1, -1, 3).expand(img_mask.size(0), -1, -1)

    return points, points_alpha, colors


def load_text_points(text, pos=None, size=20, rot=0, img_size=256, colorscale='turbo'):
    # Measure the text area
    # font = ImageFont.truetype (r'Roboto-Bold.ttf', size)
    font = ImageFont.load_default()
    wi, hi = font.getbbox(text)[2:]

    # Create a dummy source image
    into = Image.new('1', (img_size, img_size), 0)
    # Copy the relevant area from the source image
    if pos is None:
        pos = (img_size // 2 - wi // 2, img_size // 2 - hi // 2)
    img = into.crop((pos[0], pos[1], pos[0] + wi, pos[1] + hi))

    # Print into the rotated area
    d = ImageDraw.Draw(img)
    d.text((0, 0), text, font=font, fill = (1))

    # Rotate it forward again
    img = img.rotate(rot, expand=1)

    # Insert it back into the source image
    into.paste(img, pos)
    text_points = np.where(np.array(into)>0)
    text_points = np.stack(text_points).transpose(1, 0)[:, [1, 0]]
    text_points = torch.from_numpy(np.ascontiguousarray(text_points)).float()
    text_colors = get_plotly_colors(len(text_points), colorscale).squeeze()
    return text_points, text_colors