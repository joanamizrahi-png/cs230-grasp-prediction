"""
Learned attention variant of PointNet for grasp success prediction.

Instead of fixed Gaussian attention (distance-based), this model learns
attention weights using a cross-attention mechanism between point features
and grasp features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedAttentionEncoder(nn.Module):
    """
    PointNet encoder with learned cross-attention between points and grasp.

    The attention mechanism:
    1. Encodes each point to get point features
    2. Encodes grasp position to get grasp query
    3. Computes attention as dot product between point features and grasp query
    4. Uses attention weights to create weighted pooling
    """

    def __init__(self, input_dim=3, output_dim=1024, attention_dim=64, pooling_ratio=0.5):
        super(LearnedAttentionEncoder, self).__init__()

        self.pooling_ratio = pooling_ratio
        self.attention_dim = attention_dim

        # Shared MLP layers for point features
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)

        # Attention layers
        # Project point features to attention space (keys)
        self.point_attention = nn.Conv1d(output_dim, attention_dim, 1)
        # Project grasp position to attention space (query)
        self.grasp_attention = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, attention_dim)
        )

    def forward(self, x, grasp_center):
        """
        Args:
            x: (batch_size, num_points, input_dim) point cloud
            grasp_center: (batch_size, 3) grasp center position
        Returns:
            global_feature: (batch_size, output_dim) global feature
            attention_weights: (batch_size, num_points) learned attention weights
        """
        batch_size = x.shape[0]

        # Transpose for conv1d: (batch, channels, num_points)
        x = x.transpose(2, 1)

        # Shared MLPs to get point features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x: (B, output_dim, N)

        # Global max pooling (always computed)
        x_max = torch.max(x, 2)[0]  # (B, output_dim)

        # Learned attention
        # Keys: project point features to attention space
        keys = self.point_attention(x)  # (B, attention_dim, N)

        # Query: project grasp center to attention space
        query = self.grasp_attention(grasp_center)  # (B, attention_dim)
        query = query.unsqueeze(2)  # (B, attention_dim, 1)

        # Attention scores: dot product between query and keys
        # (B, attention_dim, N) * (B, attention_dim, 1) -> sum over attention_dim
        scores = (keys * query).sum(dim=1)  # (B, N)
        scores = scores / (self.attention_dim ** 0.5)  # Scale by sqrt(d)

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # (B, N)

        # Weighted pooling using learned attention
        w = attention_weights.unsqueeze(1)  # (B, 1, N)
        x_weighted = (x * w).sum(dim=2)  # (B, output_dim)

        # Combine max pooling and attention pooling
        global_feature = (1 - self.pooling_ratio) * x_max + self.pooling_ratio * x_weighted

        return global_feature, attention_weights


class LearnedAttentionPredictor(nn.Module):
    """
    Grasp success predictor with learned attention mechanism.
    """

    def __init__(
        self,
        point_dim=3,
        grasp_dim=13,
        hidden_dim=512,
        attention_dim=64,
        pooling_ratio=0.5,
    ):
        super(LearnedAttentionPredictor, self).__init__()

        self.pooling_ratio = pooling_ratio

        # Point cloud encoder with learned attention
        self.point_encoder = LearnedAttentionEncoder(
            input_dim=point_dim,
            output_dim=1024,
            attention_dim=attention_dim,
            pooling_ratio=pooling_ratio
        )

        # Fully connected layers for combining features
        self.fc1 = nn.Linear(1024 + grasp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, 1)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(0.3)

    def forward(self, points, grasp, return_attention=False):
        """
        Args:
            points: (batch_size, num_points, 3) point cloud
            grasp: (batch_size, 13) grasp parameters
            return_attention: if True, also return attention weights for visualization

        Returns:
            logits: (batch_size, 1) grasp success logits
            attention_weights: (optional) (batch_size, num_points) attention weights
        """
        # Extract grasp center
        grasp_center = grasp[:, :3]

        # Encode point cloud with learned attention
        point_features, attention_weights = self.point_encoder(points, grasp_center)

        # Concatenate with grasp parameters
        x = torch.cat([point_features, grasp], dim=1)

        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        if return_attention:
            return x, attention_weights
        return x


def loss_fn(outputs, labels, pos_weight):
    """
    Compute the weighted binary cross entropy loss.
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = criterion(outputs.squeeze(), labels)
    return loss


def accuracy(outputs, labels):
    """
    Compute the accuracy.
    """
    predictions = torch.sigmoid(outputs.squeeze()) > 0.5
    correct = (predictions == labels).sum().item()
    return correct / len(labels)


metrics = {
    'accuracy': accuracy,
}
