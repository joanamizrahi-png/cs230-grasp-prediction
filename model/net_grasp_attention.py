"""
PointNet with Gaussian attention for grasp success prediction.

Points are weighted by distance to the grasp center using a Gaussian kernel.
This encodes our intuition that nearby geometry matters most for grasp success.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """
    PointNet encoder with optional attention-weighted pooling.

    If weights are provided, combines max pooling with weighted average
    using pooling_ratio to balance global vs local features.
    """

    def __init__(self, input_dim=3, output_dim=1024, pooling_ratio=0.5):
        super(PointNetEncoder, self).__init__()
        self.pooling_ratio = pooling_ratio

        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x, weights=None):
        """
        Args:
            x: (B, N, 3) point cloud
            weights: (B, N) optional per-point weights
        Returns:
            (B, 1024) global feature
        """
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x_max = torch.max(x, dim=2)[0]

        if weights is not None:
            w = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
            w = w.unsqueeze(1)
            x_weighted = (x * w).sum(dim=2)
            x = (1 - self.pooling_ratio) * x_max + self.pooling_ratio * x_weighted
        else:
            x = x_max

        return x


class GraspSuccessPredictor(nn.Module):
    """
    PointNet with Gaussian attention centered on grasp position.

    The attention weights are computed as:
        w = exp(-dist^2 / (2 * sigma^2))
    where dist is distance from each point to the grasp center.
    """

    def __init__(self, point_dim=3, grasp_dim=13, hidden_dim=512,
                 use_grasp_attention=True, attention_sigma=1.0, pooling_ratio=0.5):
        super(GraspSuccessPredictor, self).__init__()

        self.point_encoder = PointNetEncoder(
            input_dim=point_dim,
            output_dim=1024,
            pooling_ratio=pooling_ratio
        )

        self.use_grasp_attention = use_grasp_attention
        self.attention_sigma = attention_sigma

        self.fc1 = nn.Linear(1024 + grasp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def forward(self, points, grasp):
        """
        Args:
            points: (B, N, 3) point cloud
            grasp: (B, 13) grasp parameters [position(3), rotation(9), width(1)]
        Returns:
            (B, 1) logits for grasp success
        """
        if self.use_grasp_attention:
            grasp_center = grasp[:, :3]
            dists = torch.norm(points - grasp_center.unsqueeze(1), dim=2)
            weights = torch.exp(-(dists ** 2) / (2 * self.attention_sigma ** 2))
            point_features = self.point_encoder(points, weights=weights)
        else:
            point_features = self.point_encoder(points, weights=None)

        x = torch.cat([point_features, grasp], dim=1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
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
