import torch
import torch.nn as nn
import torch.nn.functional as F

from commons.utils import map_minmax


class LossCorrsSparse(nn.Module):
    def __init__(self, extractor=None, flow_size=256, T=1.0):
        super().__init__()
        self.extractor = extractor
        self.flow_size = flow_size
        self.T = T
        self.dist_fn = nn.PairwiseDistance(p=2)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, src_flow, trg_flow, src_kp, trg_kp, kp_vis, kp_wt):
        N = src_flow.size(0)
        res = src_flow.size(1)
        top_k = kp_vis.shape[1]
        # bb1_canon - N x 2 x top_k x 1
        # bb2_canon - N x 2 x 1 x top_k
        # Sample flow values using the pseudo GT from the flow_grid
        src_kp_canon = F.grid_sample(
            src_flow.permute(0, 3, 1, 2),
            map_minmax(src_kp.unsqueeze(2), 0, res-1, -1, 1), mode='bilinear',
            padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)
        trg_kp_canon = F.grid_sample(
            trg_flow.permute(0, 3, 1, 2),
            map_minmax(trg_kp.unsqueeze(1), 0, res-1, -1, 1), mode='bilinear',
            padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)

        # dists - N x top_k x top_k
        dists1 = self.dist_fn(src_kp_canon, trg_kp_canon.detach()) * (-1/self.T)
        dists2 = self.dist_fn(src_kp_canon.detach(), trg_kp_canon) * (-1/self.T)
        labels = torch.arange(top_k, dtype=torch.long, device='cuda')
        labels = labels.unsqueeze(0).repeat(N, 1)
        labels[~kp_vis] = -100
        
        loss = self.loss_fn(dists1, labels) + self.loss_fn(dists2, labels)
        loss *= kp_wt
        return loss.sum() / kp_vis.sum()

    def forward_eq(self, src_flow, trg_flow, src_kp, trg_kp, kp_vis):
        N = src_flow.size(0)
        res = src_flow.size(1)
        top_k = kp_vis.shape[1]
        # bb1_canon - N x 2 x top_k x 1
        # bb2_canon - N x 2 x 1 x top_k
        # Sample flow values using the pseudo GT from the flow_grid
        src_kp_canon = F.grid_sample(
            src_flow.permute(0, 3, 1, 2),
            map_minmax(src_kp.unsqueeze(2), 0, res-1, -1, 1), mode='bilinear',
            padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)
        trg_kp_canon = F.grid_sample(
            trg_flow.permute(0, 3, 1, 2),
            map_minmax(trg_kp.unsqueeze(1), 0, res-1, -1, 1), mode='bilinear',
            padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)

        # dists - N x top_k x top_k
        dists1 = self.dist_fn(src_kp_canon, trg_kp_canon.detach()) * (-1/self.T)
        dists2 = self.dist_fn(src_kp_canon.detach(), trg_kp_canon) * (-1/self.T)
        labels = torch.arange(top_k, dtype=torch.long, device='cuda')
        labels = labels.unsqueeze(0).repeat(N, 1)
        labels[~kp_vis] = -100
        return self.loss_fn(dists1, labels).mean() + self.loss_fn(dists2, labels).mean()