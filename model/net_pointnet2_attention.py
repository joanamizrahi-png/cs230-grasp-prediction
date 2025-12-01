import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Utility Functions (Sampling & Grouping) - Same as base PointNet++
# --------------------------------------------------------------------------- #

def square_distance(src, dst):
    """
    Optimized version using torch.cdist
    src: [B, N, C]
    dst: [B, M, C]
    """
    return torch.cdist(src, dst, p=2.0).pow(2)

def index_points(points, idx):
    """
    Gathers points based on indices.
    points: [B, N, C]
    idx: [B, S]
    Returns: [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    FPS: Selects the farthest points to ensure good coverage.
    xyz: [B, N, 3]
    npoint: int, number of points to sample
    Returns: [B, npoint] indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Finds neighbors within a given radius (Ball Query).
    xyz: [B, N, 3] all points
    new_xyz: [B, S, 3] centroids
    Returns: [B, S, nsample] indices of neighbors
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

# --------------------------------------------------------------------------- #
# 2. Grasp-Aware Set Abstraction Module
# --------------------------------------------------------------------------- #

class GraspAwareSetAbstraction(nn.Module):
    """
    Set Abstraction with grasp-aware attention weighting.
    Similar to PointNetSetAbstraction but with weighted pooling.
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(GraspAwareSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        # Build mini-PointNet (MLP)
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points, weights=None):
        """
        Args:
            xyz: [B, C, N] Coordinates (C=3)
            points: [B, D, N] Additional features or None
            weights: [B, N] Optional grasp-aware attention weights
        Returns:
            new_xyz: [B, 3, npoint] or [B, 3, 1]
            new_points: [B, mlp[-1], npoint] or [B, mlp[-1], 1]
        """
        # Convert to standard format [B, N, C]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        # 1. Sampling (FPS)
        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, xyz.shape[2]).to(xyz.device)
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)

        # 2. Grouping (Ball Query)
        if self.group_all:
            new_points = torch.cat([xyz, points], dim=2) if points is not None else xyz
            new_points = new_points.unsqueeze(1)  # [B, 1, N, C+D]

            # If weights provided, also group them
            if weights is not None:
                grouped_weights = weights.unsqueeze(1)  # [B, 1, N]
        else:
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
            grouped_xyz -= new_xyz.view(-1, self.npoint, 1, 3)  # Local normalization

            if points is not None:
                grouped_points = index_points(points, idx)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz

            # Group the attention weights if provided
            if weights is not None:
                grouped_weights = index_points(weights.unsqueeze(-1), idx).squeeze(-1)  # [B, npoint, nsample]

        # Preparation for MLP [B, C_in, npoint, nsample]
        new_points = new_points.permute(0, 3, 1, 2)

        # 3. PointNet Layer (MLP)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # 4. Weighted Pooling if weights provided, otherwise max pooling
        if weights is not None:
            # Max pooling for global features
            new_points_max = torch.max(new_points, 3)[0]  # [B, C, npoint]

            # Weighted average pooling for grasp-aware local features
            # Normalize weights across neighborhood
            if self.group_all:
                w = grouped_weights / (grouped_weights.sum(dim=2, keepdim=True) + 1e-6)  # [B, 1, N]
                w = w.unsqueeze(1)  # [B, 1, 1, N]
            else:
                w = grouped_weights / (grouped_weights.sum(dim=2, keepdim=True) + 1e-6)  # [B, npoint, nsample]
                w = w.unsqueeze(1)  # [B, 1, npoint, nsample]

            # Weighted sum
            new_points_weighted = (new_points * w).sum(dim=3)  # [B, C, npoint]

            # Combine global and local features (like in grasp_attention)
            new_points = 0.5 * new_points_max + 0.5 * new_points_weighted
        else:
            # Standard max pooling
            new_points = torch.max(new_points, 3)[0]

        # Return to format [B, C_out, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


# --------------------------------------------------------------------------- #
# 3. PointNet++ with Grasp Attention Model
# --------------------------------------------------------------------------- #

class PointNet2GraspAttention(nn.Module):
    """
    PointNet++ with grasp-aware attention mechanism.
    Combines hierarchical local geometry learning with spatial attention to grasp location.
    """
    def __init__(
        self,
        normal_channel=False,
        grasp_dim=13,
        use_grasp_attention=True,
        attention_sigma=0.15,
        use_grasp_centered_coords=False
    ):
        super(PointNet2GraspAttention, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.grasp_dim = grasp_dim

        # Grasp-aware attention settings
        self.use_grasp_attention = use_grasp_attention
        self.attention_sigma = attention_sigma
        self.use_grasp_centered_coords = use_grasp_centered_coords

        # SA1: Set Abstraction Level 1 (Local)
        self.sa1 = GraspAwareSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel, mlp=[64, 64, 128], group_all=False
        )

        # SA2: Set Abstraction Level 2 (Regional)
        self.sa2 = GraspAwareSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )

        # SA3: Set Abstraction Level 3 (Global)
        self.sa3 = GraspAwareSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )

        # Classification Head (Fully Connected)
        self.fc1 = nn.Linear(1024 + grasp_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 1)

    def forward(self, points, grasp):
        """
        Args:
            points: (batch_size, num_points, 3) point cloud
            grasp: (batch_size, 13) grasp parameters
        Returns:
            logits: (batch_size, 1) grasp success logits
        """
        B = points.shape[0]

        # Compute grasp-aware attention weights if enabled
        if self.use_grasp_attention:
            # Grasp center (first 3 dimensions of grasp)
            grasp_center = grasp[:, :3]  # [B, 3]

            # Compute distance from each point to grasp center
            points_rel = points - grasp_center.unsqueeze(1)  # [B, N, 3]
            dists = torch.norm(points_rel, dim=2)  # [B, N]

            # Gaussian attention weights
            sigma = self.attention_sigma
            weights = torch.exp(-(dists ** 2) / (2 * sigma ** 2))  # [B, N]

            # Choose coordinate frame
            if self.use_grasp_centered_coords:
                input_points = points_rel
            else:
                input_points = points
        else:
            weights = None
            input_points = points

        # Transpose to [B, 3, N] format expected by PointNet++
        xyz = input_points.transpose(2, 1)

        # Separate features (normals) if present
        if self.normal_channel:
            normals = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            normals = None

        # Pass through Set Abstraction layers
        # Only use attention weights in the first layer (input level)
        # Subsequent layers use standard max pooling
        l1_xyz, l1_points = self.sa1(xyz, normals, weights=weights)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, weights=None)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points, weights=None)

        # Flatten the global vector [B, 1024, 1] -> [B, 1024]
        point_features = l3_points.view(B, 1024)

        # Concatenate point features with grasp parameters
        x = torch.cat([point_features, grasp], dim=1)

        # Fully Connected Layers
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

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
    Compute the accuracy, given the outputs and labels for all samples.
    """
    predictions = torch.sigmoid(outputs.squeeze()) > 0.5
    correct = (predictions == labels).sum().item()
    return correct / len(labels)


# Metrics dictionary
metrics = {
    'accuracy': accuracy,
}


# --------------------------------------------------------------------------- #
# 4. Test Block
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    # Simulate a batch
    dummy_points = torch.randn(8, 1024, 3)  # [B, N, 3]
    dummy_grasp = torch.randn(8, 13)  # [B, 13]

    # Instantiate model with grasp attention
    model = PointNet2GraspAttention(
        normal_channel=False,
        use_grasp_attention=True,
        attention_sigma=0.15
    )

    # Forward pass
    output = model(dummy_points, dummy_grasp)

    print("PointNet++ with Grasp Attention model loaded successfully.")
    print(f"Points shape: {dummy_points.shape}")
    print(f"Grasp shape: {dummy_grasp.shape}")
    print(f"Output shape: {output.shape}")  # Should be [8, 1]
    print(f"Using grasp attention: {model.use_grasp_attention}")
    print(f"Attention sigma: {model.attention_sigma}")
