import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from commons.utils import map_minmax


class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=-1, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class Normalize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        input = F.normalize(input, dim=1, p=2)
        return input.clamp(min=-1, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class Canonical(nn.Module):
    def __init__(self, size, std=0.1, clamp=True):
        super().__init__()
        mean = torch.zeros(size)
        std = torch.ones(size) * std
        self.grid = nn.Parameter(torch.normal(mean=mean, std=std),
                                 requires_grad=True)
        norm_class = Normalize()
        norm_class.apply(self.grid)
        if clamp:
            clamp_class = Clamp()
            clamp_class.apply(self.grid)

    def get_grid(self, N):
        return self.grid.expand(N, -1, -1, -1)

    def unwarp(self, flow, sample_res=256):
        N = flow.size(0)
        if sample_res is not None and sample_res != flow.size(1):
            scale_factor = sample_res / flow.size(1)
            sample_flow = F.interpolate(
                flow.permute(0, 3, 1, 2), scale_factor=scale_factor,
                mode='bilinear').permute(0, 2, 3, 1)
        else:
            sample_flow = flow
        warped_img = F.grid_sample(
            self.get_grid(N), sample_flow,
            padding_mode='border', align_corners=True)
        return warped_img

    def forward(self, x):
        return x


"""
https://github.com/ykasten/layered-neural-atlases/blob/19aa32dd0cf0de7e92d279fea82844f28a15d4a0/implicit_neural_networks.py#L19
"""
def positionalEncoding_vec(in_tensor, b):
    proj = torch.einsum('ij, k -> ijk', in_tensor, b)  # shape (batch, in_tensor.size(1), freqNum)
    mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)  # shape (batch, 2*in_tensor.size(1), freqNum)
    output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)
    return output


class CanonicalMLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=3, hidden_dim=256,
                 use_positional=True, positional_dim=10,
                 skip_layers=[4, 7], num_layers=8, resolution=256,
                 use_tanh=True, apply_softmax=False):
        super().__init__()
        self.use_tanh = use_tanh
        self.resolution = resolution
        self.apply_softmax = apply_softmax
        self.output_dim = output_dim
        if apply_softmax:
            self.softmax= nn.Softmax()
        if use_positional:
            encoding_dimensions = 2 * input_dim * positional_dim
            self.b = nn.Parameter(
                torch.tensor([(2 ** j) * np.pi
                for j in range(positional_dim)], requires_grad = False))
        else:
            encoding_dimensions = input_dim

        self.hidden = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                input_dims = hidden_dim

            if i == num_layers - 1:
                # last layer
                self.hidden.append(nn.Linear(input_dims, output_dim, bias=True))
            else:
                self.hidden.append(nn.Linear(input_dims, hidden_dim, bias=True))

        self.skip_layers = skip_layers
        self.num_layers = num_layers

        self.positional_dim = positional_dim
        self.use_positional = use_positional

    def get_grid(self, N, device='cuda'):
        resolution = self.resolution
        indsy = torch.linspace(0, resolution-1, resolution, device=device)
        indsx = torch.linspace(0, resolution-1, resolution, device=device)

        # Keep (x, y) indexing to make it consistent with the flow
        points = torch.stack(
            torch.meshgrid(indsx, indsy, indexing='xy'), dim=-1).reshape(-1, 2)

        with torch.no_grad():
            grid = self(points)

        grid = grid.reshape(1, resolution, resolution, self.output_dim)
        grid = grid.permute(0, 3, 1, 2)
        return grid.expand(N, -1, -1, -1)

    def unwarp(self, flow, sample_res=256):
        N = flow.size(0)
        # Output of flow model is usually normalized between -1 and 1
        # So we need to first scale it up to self.resolution
        flow = map_minmax(flow, -1, 1, 0, self.resolution-1)

        # Resize flow if computed at a lower resolution
        if sample_res is not None and sample_res != flow.size(1):
            scale_factor = sample_res / flow.size(1)
            sample_flow = F.interpolate(
                flow.permute(0, 3, 1, 2), scale_factor=scale_factor,
                mode='bilinear').permute(0, 2, 3, 1)
        else:
            sample_flow = flow

        # Unwarp
        warped_img = self(sample_flow.reshape(-1, 2))
        warped_img = warped_img.reshape(N, sample_res, sample_res, -1)
        warped_img = warped_img.permute(0, 3, 1, 2)
        return warped_img

    def forward(self, x):
        if self.use_positional:
            if self.b.device != x.device:
                self.b = self.b.to(x.device)
            pos = positionalEncoding_vec(x, self.b)
            x = pos

        input = x.detach().clone()
        for i, layer in enumerate(self.hidden):
            if i > 0:
                x = F.relu(x)
            if i in self.skip_layers:
                x = torch.cat((x, input), 1)
            x = layer(x)

        if self.use_tanh:
            x = torch.tanh(x)

        if self.apply_softmax:
            x = self.softmax(x)
        return x