# Modified from
# https://github.com/ChristophReich1996/Optical-Flow-Visualization-PyTorch

from typing import Optional, Union
import numpy as np
import torch
from math import pi as PI, sqrt
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def get_color_wheel() -> torch.Tensor:
    """
    Generates the color wheel.
    :return: (torch.Tensor) Color wheel tensor of the shape [55, 3]
    """
    # Set constants
    RY: int = 15
    YG: int = 6
    GC: int = 4
    CB: int = 11
    BM: int = 13
    MR: int = 6
    # Init color wheel
    color_wheel: torch.Tensor = torch.zeros((RY + YG + GC + CB + BM + MR, 3),
                                            dtype=torch.float32)
    # Init counter
    counter: int = 0
    # RY
    color_wheel[0:RY, 0] = 255
    color_wheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
    counter: int = counter + RY
    # YG
    color_wheel[counter:counter + YG, 0] = \
        255 - torch.floor(255 * torch.arange(0, YG) / YG)
    color_wheel[counter:counter + YG, 1] = 255
    counter: int = counter + YG
    # GC
    color_wheel[counter:counter + GC, 1] = 255
    color_wheel[counter:counter + GC, 2] = \
        torch.floor(255 * torch.arange(0, GC) / GC)
    counter: int = counter + GC
    # CB
    color_wheel[counter:counter + CB, 1] = \
        255 - torch.floor(255 * torch.arange(CB) / CB)
    color_wheel[counter:counter + CB, 2] = 255
    counter: int = counter + CB
    # BM
    color_wheel[counter:counter + BM, 2] = 255
    color_wheel[counter:counter + BM, 0] = \
        torch.floor(255 * torch.arange(0, BM) / BM)
    counter: int = counter + BM
    # MR
    color_wheel[counter:counter + MR, 2] = \
        255 - torch.floor(255 * torch.arange(MR) / MR)
    color_wheel[counter:counter + MR, 0] = 255
    # To device
    return color_wheel / 255


def _flow_hw_to_color(
        flow_vertical: torch.Tensor, flow_horizontal: torch.Tensor,
        color_wheel: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Private function applies the flow color wheel to flow components
    (vertical and horizontal).
    :param flow_vertical: (torch.Tensor) Vertical flow [height, width]
    :param flow_horizontal: (torch.Tensor) Horizontal flow [height, width]
    :param color_wheel: (torch.Tensor) Color wheel tensor of the shape [55, 3]
    :param: device: (torch.device) Device to be used
    :return: (torch.Tensor) Visualized flow of the shape [3, height, width]
    """
    # Get shapes
    _, height, width = flow_vertical.shape
    # Init flow image
    flow_image = torch.zeros(3, height, width,
                             dtype=torch.float32, device=device)
    # Get number of colors
    number_of_colors = color_wheel.shape[0]
    # Compute norm, angle and factors
    flow_norm = (flow_vertical ** 2 + flow_horizontal ** 2).sqrt()
    angle = torch.atan2(- flow_vertical, - flow_horizontal) / PI
    fk = (angle + 1.) / 2. * (number_of_colors - 1.)
    k0 = torch.floor(fk).long()
    k1 = k0 + 1
    k1[k1 == number_of_colors] = 0
    f = fk - k0
    # Iterate over color components
    for index in range(color_wheel.shape[1]):
        # Get component of all colors
        tmp = color_wheel[:, index]
        # Get colors
        color_0 = tmp[k0] / 255.
        color_1 = tmp[k1] / 255.
        # Compute color
        color = (1. - f) * color_0 + f * color_1
        # Get color index
        color_index = flow_norm <= 1
        # Set color saturation
        color[color_index] = 1 - flow_norm[color_index] * \
            (1. - color[color_index])
        color[~color_index] = color[~color_index] * 0.75
        # Set color in image
        flow_image[index] = torch.floor(255 * color)
    return flow_image


def flow_to_color(
        flow: torch.Tensor,
        clip_flow: Optional[Union[float, torch.Tensor]] = None) -> \
        torch.Tensor:
    """
    Function converts a given optical flow map into the classical color schema.
    :param flow: (torch.Tensor) Optical flow tensor
                 [batch size (optional), 2, height, width].
    :param clip_flow: (Optional[Union[float, torch.Tensor]])
                      Max value of flow values for clipping (default None).
    :return: (torch.Tensor) Flow visualization (float tensor) with the shape
             [batch size (if used), 3, height, width].
    """
    # Check parameter types
    assert torch.is_tensor(flow), \
        f"Given flow map must be a torch.Tensor, {type(flow)} given"
    assert torch.is_tensor(clip_flow) or isinstance(clip_flow, float) or \
        clip_flow is None, f"""Given clip_flow parameter must be a float,
        a torch.Tensor, or None, {type(clip_flow)} given"""
    # Check shapes
    assert flow.ndimension() in [3, 4], \
        f"Given flow must be a 3D/4D tensor, given tensor shape {flow.shape}."
    if torch.is_tensor(clip_flow):
        assert clip_flow.ndimension() == 0, f"""Given clip_flow tensor must be
        a scalar, given tensor shape {clip_flow.shape}."""
    # Manage batch dimension
    batch_dimension: bool = True
    if flow.ndimension() == 3:
        flow = flow[None]
        batch_dimension: bool = False
    # Save shape
    batch_size, _, height, width = flow.shape
    # Check flow dimension
    assert flow.shape[1] == 2, f"""Flow dimension must have the shape 2
        but tensor with {flow.shape[1]} given"""
    # Save device
    device: torch.device = flow.device
    # Clip flow if utilized
    if clip_flow is not None:
        flow = flow.clip(max=clip_flow)
    # Get horizontal and vertical flow
    flow_vertical: torch.Tensor = flow[:, 0:1]
    flow_horizontal: torch.Tensor = flow[:, 1:2]
    # Get max norm of flow
    flow_max_norm: torch.Tensor = \
        (flow_vertical ** 2 + flow_horizontal ** 2).sqrt()\
        .view(batch_size, -1).max(dim=-1)[0]
    flow_max_norm: torch.Tensor = flow_max_norm.view(batch_size, 1, 1, 1)
    # Normalize flow
    flow_vertical: torch.Tensor = flow_vertical / (flow_max_norm + 1e-05)
    flow_horizontal: torch.Tensor = flow_horizontal / (flow_max_norm + 1e-05)
    # Get color wheel
    color_wheel: torch.Tensor = get_color_wheel().to(device)
    # Init flow image
    flow_image = torch.zeros(batch_size, 3, height, width, device=device)
    # Iterate over batch dimension
    for index in range(batch_size):
        flow_image[index] = _flow_hw_to_color(
            flow_vertical=flow_vertical[index],
            flow_horizontal=flow_horizontal[index],
            color_wheel=color_wheel, device=device)
    return flow_image if batch_dimension else flow_image[0]


def radial_gradient(color, radii):
    colors = []
    for r in radii:
        colorr = r*color+(1-r)*np.array([1, 1, 1])
        colors.append(colorr)
    return colors


def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def color_wheel_slow():
    colorwheel = get_color_wheel()
    steps = 10
    lim = sqrt(2)/2
    theta = (2*PI) / colorwheel.shape[0]
    x = torch.linspace(0, 1, 50)
    N = colorwheel.shape[0]
    for i in range(N):
        angles = torch.linspace(i * theta, (i+1) * theta, steps)
        colors = tensor_linspace(colorwheel[i], colorwheel[(i+1) % N], steps)
        for j in range(steps):
            color = colors[:, j]
            line = radial_gradient(color, x)
            for k in range(len(x)):
                plt.scatter((x[k] * torch.cos(angles[j])).numpy(),
                            (-x[k] * torch.sin(angles[j])).numpy(),
                            color=line[k].numpy().clip(0, 1))
    plt.axis('off')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.savefig('colorwheel_slow.png')


def color_wheel_fast():
    colorwheel = get_color_wheel()
    N = colorwheel.shape[0]
    steps = 512
    lim = sqrt(2)
    xs = torch.linspace(-1, 1, steps=steps)
    ys = torch.linspace(-1, 1, steps=steps)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    r = torch.sqrt(x*x + y*y)  # (0, sqrt(2)]
    # https://math.stackexchange.com/questions/1327253/how-do-we-find-out-angle-from-x-y-coordinates
    theta = 2 * torch.arctan(-y / (-x+r)) + PI  # [0, 2*PI]

    # Not the greatest way but vectorized
    # Interpolate theta
    theta_ind = theta / (2*PI) * N  # [0, 55]
    theta_ind_floor = torch.floor(theta_ind).long()
    theta_ind_ceil = (torch.ceil(theta_ind) % N).long()
    residual = (theta_ind - theta_ind_floor).unsqueeze(-1)
    color = colorwheel[theta_ind_floor] * residual + \
        colorwheel[theta_ind_ceil] * (1 - residual)

    # Interpolate radius
    r = (r / lim).unsqueeze(-1)
    color = color * r + torch.ones(steps, steps, 3) * (1-r)
    # color = (color.numpy() * 255).astype(np.uint8)
    return color  # HWC


def expand_color_wheel(N=1):
    """
    N = number of points you want to create between two bins. Must be an
        integer >= 1
    """
    colorwheel = get_color_wheel()
    colorwheel_shifted = torch.roll(colorwheel, 1, dims=0)
    colorwheel_subdivided = tensor_linspace(
        colorwheel_shifted, colorwheel, steps=N+2)
    colorwheel_subdivided = colorwheel_subdivided[:, :, :N+1]
    colorwheel_subdivided = colorwheel_subdivided.permute(0, 2, 1).reshape(
        -1, colorwheel.size(-1))

    return torch.roll(colorwheel_subdivided, -(N+1), dims=0)


def color_wheel_fast_smooth(resolution=512, subdivision=16):
    lim = sqrt(2)
    colorwheel = expand_color_wheel(subdivision)
    N = colorwheel.shape[0]
    xs = torch.linspace(-1, 1, steps=resolution)
    ys = torch.linspace(-1, 1, steps=resolution)
    x, y = torch.meshgrid(xs, ys, indexing='xy')
    r = torch.sqrt(x*x + y*y)  # (0, sqrt(2)]
    # https://math.stackexchange.com/questions/1327253/how-do-we-find-out-angle-from-x-y-coordinates
    theta = 2 * torch.arctan(-y / (-x+r)) + PI  # [0, 2*PI]

    # Already got interpolated theta
    # Interpolate theta
    theta_ind = theta / (2*PI) * (N-1)  # [0, N-1]
    theta_ind = torch.round(theta_ind).long()
    color = colorwheel[theta_ind]

    # Interpolate radius
    r = (r / lim).unsqueeze(-1)
    color = color * r + torch.ones(resolution, resolution, 3) * (1-r)
    # color = (color.numpy() * 255).astype(np.uint8)
    return color  # HWC


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], u.shape[2], 3), np.uint8)
    colorwheel = get_color_wheel().numpy()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[..., ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.dim() == 4, 'input flow must have four dimensions'
    assert flow_uv.size(3) == 2, 'input flow must have shape [N,H,W,2]'
    flow_uv = (flow_uv.cpu().numpy() * (flow_uv.size(1) - 1))
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[..., 0]
    v = flow_uv[..., 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    colors = flow_uv_to_colors(u, v, convert_to_bgr)
    colors = torch.from_numpy(colors).float().div(255.0).permute(0, 3, 1, 2)
    return colors


if __name__ == "__main__":
    colormap = color_wheel_fast()
    save_image([colormap.permute(2, 0, 1)], 'colorwheel_fast.png')
    colormap = color_wheel_fast_smooth()
    save_image([colormap.permute(2, 0, 1)], 'colorwheel_fast_fixed.png')