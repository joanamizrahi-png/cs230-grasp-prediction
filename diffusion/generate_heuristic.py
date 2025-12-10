"""
Heuristic grasp generation: sample near surface + rank with classifier.
No diffusion - just sample plausible grasps and pick the best ones.
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import utils
import model.net as net
import model.net_grasp_attention as net_attention
import model.data_loader as data_loader


def estimate_surface_normals(points, k=10):
    """
    Estimate surface normals using PCA on local neighborhoods.
    Normal points outward (away from centroid).

    Args:
        points: (N, 3) point cloud
        k: number of neighbors for normal estimation

    Returns:
        normals: (N, 3) estimated normals (pointing outward)
    """
    from scipy.spatial import cKDTree

    points_np = points.cpu().numpy() if isinstance(points, torch.Tensor) else points
    tree = cKDTree(points_np)

    normals = np.zeros_like(points_np)
    centroid = points_np.mean(axis=0)

    for i in range(len(points_np)):
        # Find k nearest neighbors
        _, idx = tree.query(points_np[i], k=k)
        neighbors = points_np[idx]

        # PCA to find normal
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Normal is eigenvector with smallest eigenvalue
        normal = eigenvectors[:, 0]

        # Orient normal to point away from centroid
        to_centroid = centroid - points_np[i]
        if np.dot(normal, to_centroid) > 0:
            normal = -normal

        normals[i] = normal

    return normals


def check_grasp_collision(position, rotation, points, gripper_depth=0.10, gripper_width=0.08):
    """
    Check if gripper collides with point cloud (points between fingers).

    Args:
        position: (3,) gripper position
        rotation: (3, 3) rotation matrix
        points: (N, 3) point cloud
        gripper_depth: how far fingers extend
        gripper_width: distance between fingers

    Returns:
        True if collision (bad grasp), False if clear (good grasp)
    """
    # Transform points to gripper frame
    points_centered = points - position
    points_gripper = points_centered @ rotation  # Now in gripper frame

    # Gripper frame: Z = approach, X = finger spread direction
    # Check points that are:
    # 1. In front of gripper (Z > 0) and within finger reach (Z < gripper_depth)
    # 2. Between fingers (|X| < gripper_width/2)
    # 3. Not too far in Y direction (|Y| < some tolerance)

    z = points_gripper[:, 2]
    x = points_gripper[:, 0]
    y = points_gripper[:, 1]

    in_z = (z > 0) & (z < gripper_depth)
    in_x = np.abs(x) < gripper_width / 2
    in_y = np.abs(y) < gripper_width / 2  # Use same tolerance

    collision_points = in_z & in_x & in_y

    return np.sum(collision_points) > 5  # Allow a few points (noise tolerance)


def sample_plausible_grasps(points, num_samples, normals=None, max_attempts_per_grasp=10):
    """
    Sample plausible grasp poses near object surface with collision checking.

    Args:
        points: (N, 3) point cloud (normalized to ~[-1, 1])
        num_samples: number of grasps to generate
        normals: (N, 3) precomputed normals, or None to compute
        max_attempts_per_grasp: max tries to find collision-free grasp

    Returns:
        grasps: (num_samples, 13) grasp parameters
    """
    points_np = points.cpu().numpy() if isinstance(points, torch.Tensor) else points

    # Compute normals if not provided
    if normals is None:
        normals = estimate_surface_normals(points_np)

    grasps = []
    collision_rejected = 0

    for _ in range(num_samples):
        grasp_found = False

        for attempt in range(max_attempts_per_grasp):
            # 1. Pick random surface point
            idx = np.random.randint(len(points_np))
            surface_pt = points_np[idx]
            normal = normals[idx]  # Points outward from surface

            # 2. Position grasp along normal, offset from surface
            offset = np.random.uniform(0.6, 1.4)
            position = surface_pt + normal * offset

            # 3. Create rotation matrix where Z-axis points TOWARD surface
            z_axis = -normal
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)

            # Create orthogonal x and y axes
            if abs(z_axis[0]) < 0.9:
                arbitrary = np.array([1, 0, 0])
            else:
                arbitrary = np.array([0, 1, 0])

            x_axis = np.cross(arbitrary, z_axis)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)

            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)

            # Add random rotation around approach axis
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = cos_a * x_axis + sin_a * y_axis
            y_rot = -sin_a * x_axis + cos_a * y_axis

            rotation = np.column_stack([x_rot, y_rot, z_axis])

            # 4. Check for collision
            if not check_grasp_collision(position, rotation, points_np):
                # No collision - accept this grasp
                width = 0.08
                grasp = np.concatenate([position, rotation.flatten(), [width]])
                grasps.append(grasp)
                grasp_found = True
                break
            else:
                collision_rejected += 1

        # If no collision-free grasp found, use last attempt anyway
        if not grasp_found:
            width = 0.08
            grasp = np.concatenate([position, rotation.flatten(), [width]])
            grasps.append(grasp)

    if collision_rejected > 0:
        print(f"  Collision check rejected {collision_rejected} grasp attempts")

    return np.array(grasps, dtype=np.float32)


def generate_and_rank_grasps(classifier, dataloader, params, num_candidates=100, num_keep=10, max_batches=None):
    """
    Generate random plausible grasps and rank with classifier.
    """
    classifier.eval()

    all_grasps = []
    all_success_probs = []
    all_points = []
    all_max_dists = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Generating grasps')):
            if max_batches is not None and batch_idx >= max_batches:
                break

            points = batch['points'].to(params.device)
            max_dists = batch['max_dist'].to(params.device)
            batch_size = points.shape[0]

            for i in range(batch_size):
                pts = points[i].cpu().numpy()

                # Precompute normals once per object
                normals = estimate_surface_normals(pts)

                # Generate candidate grasps
                candidates = sample_plausible_grasps(pts, num_candidates, normals)
                candidates_tensor = torch.from_numpy(candidates).to(params.device)

                # Repeat points for classifier
                pts_repeated = points[i:i+1].repeat(num_candidates, 1, 1)

                # Score with classifier
                logits = classifier(pts_repeated, candidates_tensor)
                probs = torch.sigmoid(logits.squeeze())

                # Keep top-k
                top_indices = torch.argsort(probs, descending=True)[:num_keep]

                best_grasps = candidates_tensor[top_indices].cpu().numpy()
                best_probs = probs[top_indices].cpu().numpy()

                all_grasps.append(best_grasps)
                all_success_probs.append(best_probs)
                all_points.append(np.tile(pts, (num_keep, 1, 1)))
                all_max_dists.append(np.full(num_keep, max_dists[i].cpu().numpy()))

    # Concatenate
    all_grasps = np.concatenate(all_grasps, axis=0)
    all_success_probs = np.concatenate(all_success_probs, axis=0)
    all_points = np.concatenate(all_points, axis=0)
    all_max_dists = np.concatenate(all_max_dists, axis=0)

    return {
        'grasps': all_grasps,
        'success_probs': all_success_probs,
        'points': all_points,
        'max_dists': all_max_dists,
        'num_samples': num_keep,
        'num_candidates': num_candidates
    }


def analyze_results(results):
    """Print analysis of generated grasps."""
    success_probs = results['success_probs']

    print("\n=== Generation Results (Random Sampling + Classifier Ranking) ===")
    print(f"Total grasps generated: {len(success_probs)}")
    print(f"Candidates per object: {results['num_candidates']}")
    print(f"Kept per object: {results['num_samples']}")

    print(f"\nSuccess probability statistics:")
    print(f"  Mean: {success_probs.mean():.4f}")
    print(f"  Std:  {success_probs.std():.4f}")
    print(f"  Min:  {success_probs.min():.4f}")
    print(f"  Max:  {success_probs.max():.4f}")
    print(f"  Median: {np.median(success_probs):.4f}")

    thresholds = [0.5, 0.7, 0.8, 0.9]
    print(f"\nGrasps above threshold:")
    for thresh in thresholds:
        count = (success_probs > thresh).sum()
        pct = 100 * count / len(success_probs)
        print(f"  > {thresh}: {count} ({pct:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--classifier_dir', default='experiments/ATTENTION/grasp_attention')
    parser.add_argument('--num_candidates', type=int, default=100,
                        help="Number of random grasps to sample per object")
    parser.add_argument('--num_keep', type=int, default=10,
                        help="Number of best grasps to keep per object")
    parser.add_argument('--num_objects', type=int, default=None)
    parser.add_argument('--output_file', default='generated_grasps_random.npz')
    parser.add_argument('--output_dir', default='experiments/DIFFUSION/random_sampling')

    args = parser.parse_args()

    # Load classifier params
    classifier_params_path = os.path.join(args.classifier_dir, 'params.json')
    assert os.path.isfile(classifier_params_path), f"No config at {classifier_params_path}"
    classifier_params = utils.Params(classifier_params_path)

    # Setup
    params = classifier_params
    params.cuda = torch.cuda.is_available()
    params.device = torch.device('cuda' if params.cuda else 'cpu')

    print("Loading dataset...")
    dataloaders = data_loader.fetch_dataloader(['val'], args.data_dir, params)
    val_dl = dataloaders['val']
    print("- done.")

    # Load classifier
    print("Loading classifier...")
    use_attention = hasattr(classifier_params, 'attention_sigma')

    if use_attention:
        print(f"  Using grasp-attention classifier (sigma={classifier_params.attention_sigma})")
        classifier = net_attention.GraspSuccessPredictor(
            point_dim=3, grasp_dim=13, hidden_dim=512,
            use_grasp_attention=True,
            attention_sigma=classifier_params.attention_sigma
        )
    else:
        classifier = net.GraspSuccessPredictor(point_dim=3, grasp_dim=13, hidden_dim=512)

    classifier = classifier.to(params.device)

    checkpoint = os.path.join(args.classifier_dir, 'best.pth.tar')
    utils.load_checkpoint(checkpoint, classifier)
    print("- done.")

    # Generate
    max_batches = None
    if args.num_objects is not None:
        max_batches = (args.num_objects + params.batch_size - 1) // params.batch_size

    print(f"\nGenerating {args.num_candidates} candidates per object, keeping top {args.num_keep}...")
    results = generate_and_rank_grasps(
        classifier, val_dl, params,
        num_candidates=args.num_candidates,
        num_keep=args.num_keep,
        max_batches=max_batches
    )

    analyze_results(results)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    print(f"\nSaving to {output_path}")
    np.savez(output_path, **results)
    print("- done.")
