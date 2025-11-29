import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Utility Functions (Sampling & Grouping)
# --------------------------------------------------------------------------- #

def square_distance(src, dst):
    """
    Calculates the squared Euclidean distance between two point clouds.
    src: [B, N, C]
    dst: [B, M, C]
    Returns: [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

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
    group_idx[sqrdists > radius ** 2] = N # Index out of bounds for points too far away
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask] # Padding: replace empty spots with the first found point
    return group_idx

# --------------------------------------------------------------------------- #
# 2. Set Abstraction Module (The building block)
# --------------------------------------------------------------------------- #

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
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

    def forward(self, xyz, points):
        """
        xyz: [B, C, N] Coordinates (C=3)
        points: [B, D, N] Additional features (optional, e.g., normals) or None
        """
        # Convert to standard format [B, N, C] for geometric calculations
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        # 1. Sampling (FPS)
        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, xyz.shape[2]).to(xyz.device) # Dummy
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, fps_idx)
        
        # 2. Grouping (Ball Query)
        if self.group_all:
            new_points = torch.cat([xyz, points], dim=2) if points is not None else xyz
            new_points = new_points.unsqueeze(1) # [B, 1, N, C+D]
        else:
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
            grouped_xyz -= new_xyz.view(-1, self.npoint, 1, 3) # Local normalization (relative coordinates)
            
            if points is not None:
                grouped_points = index_points(points, idx)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz

        # Preparation for MLP [B, C_in, npoint, nsample]
        new_points = new_points.permute(0, 3, 1, 2)

        # 3. PointNet Layer (MLP + MaxPool)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Max Pooling over the neighborhood (dim 3 = nsample)
        new_points = torch.max(new_points, 3)[0]
        
        # Return to format [B, C_out, npoint]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

# --------------------------------------------------------------------------- #
# 3. Full PointNet++ Model for Grasp Prediction
# --------------------------------------------------------------------------- #

class PointNet2Grasp(nn.Module):
    def __init__(self, normal_channel=False):
        super(PointNet2Grasp, self).__init__()
        in_channel = 6 if normal_channel else 3 # 3 for XYZ, 6 if XYZ + Normals
        self.normal_channel = normal_channel

        # SA1: Set Abstraction Level 1 (Local)
        # 512 centroids, radius 0.2, 32 neighbors. MLP increases features to 64, 64, 128
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, 
                                          in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        
        # SA2: Set Abstraction Level 2 (Regional)
        # 128 centroids, radius 0.4, 64 neighbors. Input 128 (from SA1) + 3 (coord) -> Output 256
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, 
                                          in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        
        # SA3: Set Abstraction Level 3 (Global)
        # Group all remaining points into a single global vector. Input 256 + 3 -> Output 1024
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, 
                                          in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        # Classification Head (Fully Connected)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, 1) # Single output (Logit) for binary classification

    def forward(self, xyz):
        """
        Input: xyz [B, C, N] (usually C=3 for x,y,z)
        """
        B, _, _ = xyz.shape
        
        # Separate features (normals) if present
        if self.normal_channel:
            normals = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            normals = None

        # Pass through Set Abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, normals)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Flatten the global vector [B, 1024, 1] -> [B, 1024]
        x = l3_points.view(B, 1024)
        
        # Fully Connected Layers
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x) # Returns raw logits
        
        return x

# --------------------------------------------------------------------------- #
# 4. Test Block (To verify dimensions)
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    # Simulate a batch of 8 point clouds with 1024 points each
    # Use CPU by default for testing to avoid CUDA errors if not available
    dummy_input = torch.randn(8, 3, 1024)
    
    # Instantiate model
    model = PointNet2Grasp(normal_channel=False)
    
    # Forward pass
    output = model(dummy_input)
    
    print("PointNet++ model loaded successfully.")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be [8, 1]