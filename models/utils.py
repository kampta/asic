import torch


def unravel_index(indices, shape):
    # https://stackoverflow.com/a/65168284
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord, dim=-1)

    return coord


def sample_from_reverse_flow(flow, points):
    # Points are of size B x N x 2 in YX format
    B, N, _ = points.shape

    # Reshape flow from  (B, H, W, 2) to (B, H, W, 1, 1, 2)
    flow_reshaped = flow.unsqueeze(-2).unsqueeze(-2)

    # Reshape points from (B, N, 2) to (B, 1, 1, N, 2, 1)
    points = points.unsqueeze(1).unsqueeze(1).unsqueeze(-1)

    # (B, H, W, N)
    similarities = (flow_reshaped @ points)[..., 0, 0]
    distances = points.pow(2).squeeze(-1).sum(dim=-1) + \
        flow_reshaped.pow(2).sum(dim=-1).squeeze(-1) - 2 * similarities

    nearest_neighbors = distances.reshape(
        B, flow_reshaped.size(1) * flow_reshaped.size(2), N).argmin(dim=1)
    points_transfered = unravel_index(
        nearest_neighbors, (flow_reshaped.size(1), flow_reshaped.size(2)))
    return points_transfered


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)