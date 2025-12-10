"""
Diffusion model for grasp generation.

Uses 6D rotation (always valid) + cross-attention conditioning on point cloud.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from model.net import PointNetEncoder


class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep encoding (same as transformers)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def rotation_6d_to_matrix(d6):
    """
    6D rotation -> 3x3 matrix via Gram-Schmidt.
    From "On the Continuity of Rotation Representations" (Zhou et al.)
    """
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def rotation_matrix_to_6d(matrix):
    """3x3 matrix -> 6D (just take first two columns)."""
    return matrix[..., :2].flatten(start_dim=-2)


class CrossAttention(nn.Module):
    """Grasp attends to point cloud features."""
    def __init__(self, query_dim, key_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query = nn.Linear(query_dim, hidden_dim)
        self.key = nn.Linear(key_dim, hidden_dim)
        self.value = nn.Linear(key_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, query_dim)

    def forward(self, x, context):
        batch_size = x.shape[0]

        q = self.query(x).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1)

        return self.out(out)


class ImprovedGraspDenoiser(nn.Module):
    """
    Denoiser network. Takes noisy grasp + timestep + point cloud, predicts noise.
    Uses 6D rotation and cross-attention.
    """
    def __init__(self, point_feat_dim=512, time_dim=128, hidden_dim=512):
        super().__init__()

        # grasp = pos(3) + rot_6d(6) + width(1) = 10D
        self.grasp_dim = 10

        # timestep embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        # point cloud encoder
        self.point_encoder = PointNetEncoder(input_dim=3, output_dim=point_feat_dim)

        # local point features for cross-attention
        self.point_conv1 = nn.Conv1d(3, 64, 1)
        self.point_conv2 = nn.Conv1d(64, 128, 1)
        self.point_bn1 = nn.BatchNorm1d(64)
        self.point_bn2 = nn.BatchNorm1d(128)

        # separate embeddings for pos/rot/width
        self.pos_embed = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.rot_embed = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.width_embed = nn.Linear(1, hidden_dim // 4)

        # cross-attention
        self.cross_attn = CrossAttention(
            query_dim=hidden_dim + hidden_dim // 4,
            key_dim=128,
            hidden_dim=256,
            num_heads=4
        )

        # main denoising network
        fusion_dim = hidden_dim + hidden_dim // 4 + point_feat_dim + time_dim + (hidden_dim + hidden_dim // 4)
        self.denoise_net = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # separate output heads
        self.pos_head = nn.Linear(hidden_dim, 3)
        self.rot_head = nn.Linear(hidden_dim, 6)
        self.width_head = nn.Linear(hidden_dim, 1)

    def forward(self, noisy_grasp, timestep, points):
        batch_size = noisy_grasp.shape[0]

        # split grasp
        pos = noisy_grasp[:, :3]
        rot_6d = noisy_grasp[:, 3:9]
        width = noisy_grasp[:, 9:10]

        # encode timestep
        time_emb = self.time_mlp(timestep)

        # encode point cloud
        point_feat_global = self.point_encoder(points)

        # local point features for cross-attention
        points_t = points.transpose(2, 1)
        point_feat_local = F.relu(self.point_bn1(self.point_conv1(points_t)))
        point_feat_local = F.relu(self.point_bn2(self.point_conv2(point_feat_local)))
        point_feat_local = point_feat_local.transpose(2, 1)

        # embed grasp components
        pos_emb = self.pos_embed(pos)
        rot_emb = self.rot_embed(rot_6d)
        width_emb = self.width_embed(width)
        grasp_emb = torch.cat([pos_emb, rot_emb, width_emb], dim=-1)

        # cross-attention
        attn_feat = self.cross_attn(grasp_emb, point_feat_local)

        # concat everything and denoise
        x = torch.cat([grasp_emb, point_feat_global, time_emb, attn_feat], dim=-1)
        x = self.denoise_net(x)

        # predict noise
        noise_pos = self.pos_head(x)
        noise_rot = self.rot_head(x)
        noise_width = self.width_head(x)

        return torch.cat([noise_pos, noise_rot, noise_width], dim=-1)


class ImprovedGraspDiffusion(nn.Module):
    """DDPM for grasp generation with 6D rotation."""

    def __init__(self, denoiser, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()

        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        self.grasp_dim = 10

        # beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def grasp_13d_to_10d(self, grasp_13d):
        """13D (pos + rot_9d + width) -> 10D (pos + rot_6d + width)"""
        pos = grasp_13d[:, :3]
        rot_9d = grasp_13d[:, 3:12]
        width = grasp_13d[:, 12:13]
        rot_matrix = rot_9d.view(-1, 3, 3)
        rot_6d = rotation_matrix_to_6d(rot_matrix)
        return torch.cat([pos, rot_6d, width], dim=-1)

    def grasp_10d_to_13d(self, grasp_10d):
        """10D -> 13D"""
        pos = grasp_10d[:, :3]
        rot_6d = grasp_10d[:, 3:9]
        width = grasp_10d[:, 9:10]
        rot_matrix = rotation_6d_to_matrix(rot_6d)
        rot_9d = rot_matrix.flatten(start_dim=-2)
        return torch.cat([pos, rot_9d, width], dim=-1)

    def forward_diffusion(self, x_0, t, noise=None):
        """Add noise (forward process)."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_t = self.sqrt_alphas_cumprod[t.cpu()].to(x_0.device).view(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t.cpu()].to(x_0.device).view(-1, 1)

        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        return x_t, noise

    def forward(self, grasp_13d, points):
        """Training: add noise, predict it."""
        batch_size = grasp_13d.shape[0]
        device = grasp_13d.device

        grasp_10d = self.grasp_13d_to_10d(grasp_13d)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

        noise = torch.randn_like(grasp_10d)
        noisy_grasp, _ = self.forward_diffusion(grasp_10d, t, noise)

        predicted_noise = self.denoiser(noisy_grasp, t, points)
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    def forward_with_position_weight(self, grasp_13d, points, position_weight=2.0):
        """Training with weighted loss (position needs more supervision)."""
        batch_size = grasp_13d.shape[0]
        device = grasp_13d.device

        grasp_10d = self.grasp_13d_to_10d(grasp_13d)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

        noise = torch.randn_like(grasp_10d)
        noisy_grasp, _ = self.forward_diffusion(grasp_10d, t, noise)

        predicted_noise = self.denoiser(noisy_grasp, t, points)

        # weighted loss: position matters more
        pos_loss = F.mse_loss(predicted_noise[:, :3], noise[:, :3])
        rot_loss = F.mse_loss(predicted_noise[:, 3:9], noise[:, 3:9])
        width_loss = F.mse_loss(predicted_noise[:, 9:], noise[:, 9:])

        return position_weight * pos_loss + rot_loss + width_loss

    @torch.no_grad()
    def sample(self, points, num_samples=1, guidance_scale=0.0, classifier=None):
        """Generate grasps via reverse diffusion."""
        batch_size = points.shape[0]
        device = points.device

        if num_samples > 1:
            points = points.repeat_interleave(num_samples, dim=0)

        # start from noise
        x = torch.randn(batch_size * num_samples, self.grasp_dim, device=device)

        alphas = self.alphas.to(device)
        alphas_cumprod = self.alphas_cumprod.to(device)
        betas = self.betas.to(device)
        posterior_variance = self.posterior_variance.to(device)

        # reverse process
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size * num_samples,), t, device=device, dtype=torch.long)

            predicted_noise = self.denoiser(x, t_batch, points)

            # classifier guidance (optional)
            if guidance_scale > 0 and classifier is not None:
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    x_13d = self.grasp_10d_to_13d(x_in)
                    logits = classifier(points, x_13d)
                    grad = torch.autograd.grad(logits.sum(), x_in)[0]
                predicted_noise = predicted_noise - guidance_scale * grad

            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            beta_t = betas[t]

            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            )

            if t > 0:
                x = x + torch.sqrt(posterior_variance[t]) * torch.randn_like(x)

        return self.grasp_10d_to_13d(x)


def create_improved_diffusion_model(num_timesteps=1000):
    """Create the diffusion model."""
    denoiser = ImprovedGraspDenoiser(
        point_feat_dim=512,
        time_dim=128,
        hidden_dim=512
    )
    return ImprovedGraspDiffusion(
        denoiser=denoiser,
        num_timesteps=num_timesteps,
        beta_start=1e-4,
        beta_end=0.02
    )
