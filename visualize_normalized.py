#!/usr/bin/env python3
"""
Visualize what the dataloader outputs (what the model sees during training).

This shows normalized point clouds with normalized grasps, exactly as they
are fed to the model. Use this to verify data pipeline is working correctly.

Usage:
    python visualize_normalized.py --num_samples 5 --num_points 1024
    python visualize_normalized.py --split val --idx 0 1 2
"""

import argparse
import numpy as np
import trimesh
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

import model.data_loader as data_loader
from acronym_tools import create_gripper_marker


def grasp_to_transform(grasp):
    """Convert grasp parameters (13D) to 4x4 transformation matrix."""
    position = grasp[:3]
    rotation_matrix = grasp[3:12].reshape(3, 3)

    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = position

    return T


def visualize_dataloader_samples(data_dir='data', num_samples=5, num_points=1024,
                                  split='train', max_grasps_per_object=50, indices=None):
    """
    Visualize samples from the dataloader.

    Args:
        data_dir: Path to data directory
        num_samples: Number of samples to visualize (ignored if indices provided)
        num_points: Number of points in point cloud
        split: 'train', 'val', or 'test'
        max_grasps_per_object: Max grasps per object (for dataset loading)
        indices: Specific sample indices to visualize (optional)
    """
    print(f"Loading {split} dataset with {num_points} points...")

    # Create dataset WITHOUT augmentation to see raw normalized data
    dataset = data_loader.GraspDataset(
        data_path=data_dir,
        num_points=num_points,
        augment=False,  # No augmentation - see raw normalized data
        split=split,
        split_by='object',
        max_grasps_per_object=max_grasps_per_object,
        use_precomputed=True
    )

    print(f"Dataset has {len(dataset)} samples")

    # Determine which indices to visualize
    if indices is not None:
        sample_indices = [i for i in indices if i < len(dataset)]
        if len(sample_indices) < len(indices):
            print(f"Warning: Some indices out of range, using {len(sample_indices)} valid indices")
    else:
        sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    print(f"Visualizing {len(sample_indices)} samples: {list(sample_indices)}\n")

    for i, idx in enumerate(sample_indices):
        sample = dataset[idx]

        points = sample['points'].numpy()
        grasp = sample['grasp'].numpy()
        label = sample['label'].item()
        max_dist = sample['max_dist'].item()  # Exactly what dataloader outputs

        # Get grasp components
        position = grasp[:3]
        rotation = grasp[3:12].reshape(3, 3)
        width = grasp[12]

        # Get object info
        mesh_path, _, _ = dataset.samples[idx]
        obj_name = Path(mesh_path).parent.name + "/" + Path(mesh_path).stem

        print(f"{'='*60}")
        print(f"Sample {i+1}/{len(sample_indices)} (index {idx})")
        print(f"Object: {obj_name}")
        print(f"{'='*60}")
        print(f"Label: {'SUCCESS' if label == 1 else 'FAILURE'}")
        print(f"Max dist (for gripper scaling): {max_dist:.4f}")
        print(f"\nPoint cloud stats:")
        print(f"  Shape: {points.shape}")
        print(f"  Range: [{points.min():.3f}, {points.max():.3f}]")
        print(f"  Mean: {points.mean(axis=0)}")
        print(f"\nGrasp (normalized):")
        print(f"  Position: {position}")
        print(f"  Width: {width:.4f}m")

        # Check rotation validity
        det = np.linalg.det(rotation)
        ortho_error = np.linalg.norm(rotation @ rotation.T - np.eye(3))
        print(f"  Rotation det: {det:.4f} (should be ~1)")
        print(f"  Rotation ortho error: {ortho_error:.6f} (should be ~0)")

        # Distance from grasp to nearest point
        dists = np.linalg.norm(points - position, axis=1)
        min_dist_to_surface = dists.min()
        print(f"  Distance to nearest point: {min_dist_to_surface:.4f}")

        # Create visualization
        scene = trimesh.Scene()

        # Add point cloud (blue)
        pc_colors = np.tile([100, 100, 255, 255], (len(points), 1))
        pcd = trimesh.PointCloud(points, colors=pc_colors)
        scene.add_geometry(pcd, node_name='point_cloud')

        # Add gripper (green=success, red=failure)
        color = [0, 255, 0] if label == 1 else [255, 0, 0]
        gripper = create_gripper_marker(color=color)

        # Scale gripper to match normalized space (same as visualize_rotation_augmentation)
        gripper_scale = 1.0 / max_dist
        print(f"Gripper scale: {gripper_scale:.2f}")
        gripper.apply_scale(gripper_scale)

        # Apply grasp transform
        T = grasp_to_transform(grasp)
        gripper.apply_transform(T)
        scene.add_geometry(gripper, node_name='gripper')

        # Add coordinate axes at origin
        axes = trimesh.creation.axis(origin_size=0.02, axis_length=0.5)
        scene.add_geometry(axes, node_name='axes')

        print(f"\nVisualization:")
        print(f"  Blue points: Normalized point cloud")
        print(f"  {'Green' if label == 1 else 'Red'} gripper: {'Successful' if label == 1 else 'Failed'} grasp")
        print(f"  RGB axes: Origin")
        print(f"\nClose window to see next sample...")

        scene.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize dataloader output (what the model sees).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of random samples to visualize (ignored if --idx provided)"
    )
    parser.add_argument(
        "--num_points", type=int, default=1024,
        help="Number of points in point cloud (512, 1024, 2048)"
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
        help="Dataset split to visualize"
    )
    parser.add_argument(
        "--max_grasps", type=int, default=50,
        help="Max grasps per object"
    )
    parser.add_argument(
        "--idx", type=int, nargs="+", default=None,
        help="Specific sample indices to visualize (e.g., --idx 0 5 10)"
    )

    args = parser.parse_args()

    visualize_dataloader_samples(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        num_points=args.num_points,
        split=args.split,
        max_grasps_per_object=args.max_grasps,
        indices=args.idx
    )


if __name__ == "__main__":
    main()
