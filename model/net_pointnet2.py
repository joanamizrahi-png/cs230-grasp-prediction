"""
PointNet++ model for grasp success prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """Squared distance between two point sets."""
    return torch.cdist(src, dst, p=2.0).pow(2)


def index_points(points, idx):
    """Gather points by indices."""
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """Sample points that are far apart from each other."""
    B, N, _ = xyz.shape
    device = xyz.device

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest, :].unsqueeze(1)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance, dim=-1)

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Find all points within radius of each query point."""
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device

    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N

    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


class SetAbstraction(nn.Module):
    """Set Abstraction layer: samples, groups, and applies mini-PointNet."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, 3, device=xyz.device)
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)

        if self.group_all:
            grouped = torch.cat([xyz, points], dim=2) if points is not None else xyz
            grouped = grouped.unsqueeze(1)
        else:
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz -= new_xyz.unsqueeze(2)

            if points is not None:
                grouped_points = index_points(points, idx)
                grouped = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped = grouped_xyz

        grouped = grouped.permute(0, 3, 1, 2)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped = F.relu(bn(conv(grouped)))

        new_points = torch.max(grouped, dim=3)[0]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points


class PointNet2Grasp(nn.Module):
    """PointNet++ encoder + grasp params -> binary classification."""

    def __init__(self, normal_channel=False, grasp_dim=13):
        super(PointNet2Grasp, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel

        self.sa1 = SetAbstraction(512, 0.2, 32, in_channel, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = SetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], group_all=True)

        self.fc1 = nn.Linear(1024 + grasp_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)

    def forward(self, points, grasp):
        B = points.shape[0]
        xyz = points.transpose(2, 1)

        if self.normal_channel:
            normals = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            normals = None

        l1_xyz, l1_points = self.sa1(xyz, normals)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        point_features = l3_points.view(B, 1024)

        x = torch.cat([point_features, grasp], dim=1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x


def loss_fn(outputs, labels, pos_weight):
    """Weighted BCE loss to handle class imbalance."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion(outputs.squeeze(), labels)


def accuracy(outputs, labels):
    """Compute accuracy at threshold 0.5."""
    preds = torch.sigmoid(outputs.squeeze()) > 0.5
    return (preds == labels).float().mean().item()


metrics = {'accuracy': accuracy}


if __name__ == '__main__':
    model = PointNet2Grasp()
    points = torch.randn(4, 1024, 3)
    grasp = torch.randn(4, 13)
    out = model(points, grasp)
    print(f"Input: points {points.shape}, grasp {grasp.shape}")
    print(f"Output: {out.shape}")
