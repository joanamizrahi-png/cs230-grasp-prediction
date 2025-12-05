"""
PointNet with learned attention for grasp success prediction.

Instead of fixed Gaussian weights, this model learns which points to attend to
using a cross-attention mechanism between point features and grasp position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedAttentionEncoder(nn.Module):
    """
    PointNet encoder with learned cross-attention.

    Uses dot-product attention between point features (keys) and
    grasp position (query) to learn which points matter.
    """

    def __init__(self, input_dim=3, output_dim=1024, attention_dim=64, pooling_ratio=0.5):
        super(LearnedAttentionEncoder, self).__init__()
        self.pooling_ratio = pooling_ratio
        self.attention_dim = attention_dim

        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)

        # Project point features to keys
        self.point_attention = nn.Conv1d(output_dim, attention_dim, 1)

        # Project grasp center to query
        self.grasp_attention = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, attention_dim)
        )

    def forward(self, x, grasp_center):
        """
        Args:
            x: (B, N, 3) point cloud
            grasp_center: (B, 3) grasp position
        Returns:
            global_feature: (B, 1024)
            attention_weights: (B, N) for visualization
        """
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x_max = torch.max(x, dim=2)[0]

        # Compute attention
        keys = self.point_attention(x)  # (B, attention_dim, N)
        query = self.grasp_attention(grasp_center).unsqueeze(2)  # (B, attention_dim, 1)

        scores = (keys * query).sum(dim=1)  # (B, N)
        scores = scores / (self.attention_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=1)

        # Weighted pooling
        w = attention_weights.unsqueeze(1)
        x_weighted = (x * w).sum(dim=2)

        global_feature = (1 - self.pooling_ratio) * x_max + self.pooling_ratio * x_weighted

        return global_feature, attention_weights


class LearnedAttentionPredictor(nn.Module):
    """PointNet with learned attention for grasp success prediction."""

    def __init__(self, point_dim=3, grasp_dim=13, hidden_dim=512,
                 attention_dim=64, pooling_ratio=0.5, dropout=0.3):
        super(LearnedAttentionPredictor, self).__init__()

        self.point_encoder = LearnedAttentionEncoder(
            input_dim=point_dim,
            output_dim=1024,
            attention_dim=attention_dim,
            pooling_ratio=pooling_ratio
        )

        self.fc1 = nn.Linear(1024 + grasp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, points, grasp, return_attention=False):
        """
        Args:
            points: (B, N, 3) point cloud
            grasp: (B, 13) grasp parameters
            return_attention: if True, also return attention weights
        Returns:
            logits: (B, 1)
            attention_weights: (B, N) if return_attention=True
        """
        grasp_center = grasp[:, :3]
        point_features, attention_weights = self.point_encoder(points, grasp_center)

        x = torch.cat([point_features, grasp], dim=1)

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        if return_attention:
            return x, attention_weights
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
