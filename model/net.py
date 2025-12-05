"""
PointNet baseline for grasp success prediction.

This is our simplest model - it uses the original PointNet architecture
to encode the point cloud into a global feature, then concatenates with
grasp parameters for binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """
    PointNet encoder: extracts a global feature from a point cloud.

    Architecture:
        - Shared MLPs (64 -> 128 -> 1024) applied to each point
        - Max pooling across points for permutation invariance
    """

    def __init__(self, input_dim=3, output_dim=1024):
        super(PointNetEncoder, self).__init__()

        # Shared MLP layers (1x1 convolutions)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) point cloud
        Returns:
            (B, 1024) global feature vector
        """
        # Transpose for conv1d: (B, 3, N)
        x = x.transpose(2, 1)

        # Shared MLPs with batch norm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Max pooling across points
        x = torch.max(x, dim=2)[0]

        return x


class GraspSuccessPredictor(nn.Module):
    """
    Full model: PointNet encoder + grasp parameters -> binary classification.

    We concatenate the 1024-dim point cloud feature with the 13-dim grasp
    parameters, then pass through FC layers to predict success/failure.
    """

    def __init__(self, point_dim=3, grasp_dim=13, hidden_dim=512):
        super(GraspSuccessPredictor, self).__init__()

        self.point_encoder = PointNetEncoder(input_dim=point_dim, output_dim=1024)

        # FC layers: 1024+13 -> 512 -> 256 -> 1
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
        # Encode point cloud
        point_features = self.point_encoder(points)

        # Concatenate with grasp params
        x = torch.cat([point_features, grasp], dim=1)

        # FC layers
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
