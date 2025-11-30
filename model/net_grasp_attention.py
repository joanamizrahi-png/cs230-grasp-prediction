"""
Grasp-aware variant of the PointNet-based grasp success model.

This file mirrors net.py but adds optional grasp-aware attention:
points are weighted by distance to the grasp center before pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PointNetEncoder(nn.Module):
    """
    PointNet encoder that processes point cloud to global feature.
    Uses shared MLPs and pooling for permutation invariance.

    Supports optional per-point weights (e.g. grasp-aware attention):
    - Always computes a global max-pooled feature.
    - If weights are provided, also computes a weighted average feature
      and adds it to the max feature.
    """
    
    def __init__(self, input_dim=3, output_dim=1024):
        super(PointNetEncoder, self).__init__()
        
        # Shared MLP layers (applied to each point independently)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)
    
    def forward(self, x, weights=None):
        """
        Args:
            x: (batch_size, num_points, input_dim) point cloud
            weights: optional (batch_size, num_points) per-point weights.
                     If None, only max pooling is used (original PointNet).
        Returns:
            global_feature: (batch_size, output_dim) global feature
        """
        # Transpose for conv1d: (batch, channels, num_points)
        x = x.transpose(2, 1)
        
        # Shared MLPs
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling (original PointNet)
        x_max = torch.max(x, 2)[0]
        
        if weights is not None:
            # Normalize weights across points
            # weights: (B, N)
            w = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
            w = w.unsqueeze(1)  # (B, 1, N)

            # Weighted average pooling (grasp-aware local feature)
            x_weighted = (x * w).sum(dim=2)  # (B, output_dim)

            # Combine global and local features
            x = x_max + x_weighted
        else:
            x = x_max
        
        return x


class GraspSuccessPredictor(nn.Module):
    """
    Full model that combines PointNet encoder with grasp parameters
    for binary classification.

    Optionally uses grasp-aware attention:
    - Expresses points in a grasp-centered frame.
    - Weights points by a Gaussian of their distance to the grasp center.
    """
    
    def __init__(
        self,
        point_dim=3,
        grasp_dim=13,
        hidden_dim=512,
        use_grasp_attention=True,
        attention_sigma=0.05,
    ):
        super(GraspSuccessPredictor, self).__init__()
        
        # Point cloud encoder
        self.point_encoder = PointNetEncoder(input_dim=point_dim, output_dim=1024)
        
        # Grasp-aware attention settings
        self.use_grasp_attention = use_grasp_attention
        # bandwidth of distance weighting (same units as your point cloud)
        self.attention_sigma = attention_sigma
        
        # Fully connected layers for combining features
        self.fc1 = nn.Linear(1024 + grasp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, points, grasp):
        """
        Args:
            points: (batch_size, num_points, 3) point cloud
            grasp: (batch_size, 13) grasp parameters

                Assumes:
                    grasp[:, :3] is the grasp center position in the same
                    coordinate frame as `points`. If your encoding is different,
                    change this slice accordingly.

        Returns:
            logits: (batch_size, 1) grasp success logits
        """
        if self.use_grasp_attention:
            # ----- Grasp-aware preprocessing -----
            # Grasp center
            grasp_center = grasp[:, :3]
            
            # Express points in a grasp-centered frame: (B, N, 3)
            points_rel = points - grasp_center.unsqueeze(1)
            
            # Distance from each point to grasp center: (B, N)
            dists = torch.norm(points_rel, dim=2)
            
            # Grasp-aware radial weights (Gaussian in distance)
            sigma = self.attention_sigma
            weights = torch.exp(- (dists ** 2) / (2 * sigma ** 2))  # (B, N)
            
            # Encode point cloud with grasp-aware weights
            point_features = self.point_encoder(points_rel, weights=weights)
        else:
            # Original behavior: no grasp-aware weighting, points in original frame
            point_features = self.point_encoder(points, weights=None)
        
        # Concatenate with grasp parameters
        x = torch.cat([point_features, grasp], dim=1)
        
        # Fully connected layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def loss_fn(outputs, labels, pos_weight):
    """
    Compute the weighted binary cross entropy loss.
    
    Args:
        outputs: (batch_size, 1) predicted logits
        labels: (batch_size,) ground truth labels (0 or 1)
        pos_weight: weight for positive class
        
    Returns:
        loss: scalar
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = criterion(outputs.squeeze(), labels)
    return loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all samples.
    
    Args:
        outputs: (batch_size, 1) predicted logits
        labels: (batch_size,) ground truth labels
        
    Returns:
        accuracy: scalar in [0, 1]
    """
    predictions = torch.sigmoid(outputs.squeeze()) > 0.5
    correct = (predictions == labels).sum().item()
    return correct / len(labels)


# Metrics dictionary
metrics = {
    'accuracy': accuracy,
}
