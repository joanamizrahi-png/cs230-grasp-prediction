"""
Visualize generated grasps.
Green = high confidence, Red = low confidence.
"""

import sys
import argparse
import numpy as np
import trimesh
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from acronym_tools import create_gripper_marker


def grasp_params_to_transform(grasp):
    """
    Convert 13D grasp parameters to 4x4 transform matrix.

    Args:
        grasp: (13,) array [x, y, z, rot_matrix_flat(9), width]
               where rot_matrix_flat is a flattened 3x3 rotation matrix

    Returns:
        T: 4x4 transform matrix
        width: grasp width
    """
    # Extract position (3), rotation matrix (9), and width (1)
    position = grasp[:3]
    rotation_matrix = grasp[3:12].reshape(3, 3)
    width = grasp[12]

    # Build transform matrix
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = position

    return T, width


def visualize_generated_grasps(npz_file, num_grasps=20, success_threshold=0.5):
    """
    Visualize generated grasps from .npz file.

    Args:
        npz_file: Path to .npz file with generated grasps
        num_grasps: Number of grasps to visualize per object
        success_threshold: Threshold for "successful" grasp (based on classifier prediction)
    """
    print(f"Loading generated grasps from {npz_file}...")
    data = np.load(npz_file, allow_pickle=True)

    grasps = data['grasps']  # (N, 13)
    success_probs = data['success_probs']  # (N,)
    points = data['points']  # (N, num_points, 3)
    max_dists = data['max_dists']  # (N,) - max_dist for each sample

    print(f"Loaded {len(grasps)} generated grasps")
    print(f"Success probability range: [{success_probs.min():.3f}, {success_probs.max():.3f}]")
    print(f"Mean success probability: {success_probs.mean():.3f}")

    # Count how many objects we have (assuming multiple grasps per object)
    num_samples = data['num_samples'] if 'num_samples' in data else 10
    num_objects = len(grasps) // num_samples

    print(f"\nAssuming {num_samples} grasps per object → {num_objects} objects")

    # Visualize each object
    for obj_idx in range(min(num_objects, 10)):  # Max 10 objects to avoid overwhelming
        start_idx = obj_idx * num_samples
        end_idx = start_idx + num_samples

        obj_grasps = grasps[start_idx:end_idx]
        obj_probs = success_probs[start_idx:end_idx]
        obj_points = points[start_idx]  # All grasps share same point cloud
        obj_max_dist = max_dists[start_idx]  # Get max_dist for this object

        print(f"\n{'='*60}")
        print(f"Object {obj_idx + 1}/{num_objects}")
        print(f"{'='*60}")

        # Create point cloud
        point_cloud = trimesh.points.PointCloud(obj_points)

        # NEW: Both point clouds AND grasps are normalized to [-1, 1]
        # They use the same centroid/max_dist normalization, so they're already aligned!
        # No scaling needed - we can visualize directly

        max_normalized_dist = np.max(np.abs(obj_points))
        grasp_positions = obj_grasps[:, :3]
        grasp_extent = np.max(np.abs(grasp_positions))

        print(f"Point cloud extent: {max_normalized_dist:.4f} (normalized space)")
        print(f"Grasp extent: {grasp_extent:.4f} (normalized space)")
        print(f"Both should be similar since they share the same normalization!")

        # Calculate gripper scale from max_dist (same as visualize_rotation_augmentation.py)
        gripper_scale = 1.0 / obj_max_dist
        print(f"Max dist: {obj_max_dist:.4f}m")
        print(f"Gripper scale: {gripper_scale:.2f}")

        # Separate grasps by success probability
        high_prob_indices = np.where(obj_probs >= success_threshold)[0]
        low_prob_indices = np.where(obj_probs < success_threshold)[0]

        print(f"High confidence (≥{success_threshold}): {len(high_prob_indices)}")
        print(f"Low confidence (<{success_threshold}): {len(low_prob_indices)}")

        # Limit number of grasps
        if len(high_prob_indices) > num_grasps:
            high_prob_indices = np.random.choice(high_prob_indices, num_grasps, replace=False)
        if len(low_prob_indices) > num_grasps:
            low_prob_indices = np.random.choice(low_prob_indices, num_grasps, replace=False)

        # Create gripper markers for high confidence grasps (green)
        high_confidence_grippers = []
        for idx in high_prob_indices:
            T, width = grasp_params_to_transform(obj_grasps[idx])

            gripper = create_gripper_marker(color=[0, 255, 0])  # Green
            gripper.apply_scale(gripper_scale)  # Use actual max_dist-based scale
            gripper.apply_transform(T)
            high_confidence_grippers.append(gripper)

        # Create gripper markers for low confidence grasps (red)
        low_confidence_grippers = []
        for idx in low_prob_indices:
            T, width = grasp_params_to_transform(obj_grasps[idx])

            gripper = create_gripper_marker(color=[255, 0, 0])  # Red
            gripper.apply_scale(gripper_scale)  # Use actual max_dist-based scale
            gripper.apply_transform(T)
            low_confidence_grippers.append(gripper)

        print(f"\nShowing visualization...")
        print(f"  Green: high confidence (≥{success_threshold}): {len(high_confidence_grippers)} grasps")
        print(f"  Red: low confidence (<{success_threshold}): {len(low_confidence_grippers)} grasps")

        # Show scene
        scene = trimesh.Scene([point_cloud] + high_confidence_grippers + low_confidence_grippers)
        scene.show()


def visualize_comparison(npz_files, num_grasps=10):
    """
    Visualize and compare grasps from multiple .npz files (e.g., different guidance scales).

    Args:
        npz_files: List of paths to .npz files
        num_grasps: Number of grasps to show per file
    """
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
    ]

    all_data = []
    for npz_file in npz_files:
        print(f"Loading {npz_file}...")
        data = np.load(npz_file, allow_pickle=True)
        all_data.append(data)
        print(f"  {len(data['grasps'])} grasps, mean success: {data['success_probs'].mean():.3f}")

    # Visualize first object from each file
    num_samples = all_data[0]['num_samples'] if 'num_samples' in all_data[0] else 10

    # Get point cloud (should be same across files)
    point_cloud = trimesh.points.PointCloud(all_data[0]['points'][0])

    grippers = []
    for file_idx, data in enumerate(all_data):
        color = colors[file_idx % len(colors)]
        obj_grasps = data['grasps'][:num_samples]

        # Sample a few grasps
        indices = np.random.choice(len(obj_grasps), min(num_grasps, len(obj_grasps)), replace=False)

        for idx in indices:
            T, width = grasp_params_to_transform(obj_grasps[idx])
            gripper = create_gripper_marker(color=color)
            gripper.apply_transform(T)
            grippers.append(gripper)

    print(f"\nShowing comparison visualization...")
    for i, npz_file in enumerate(npz_files):
        color = colors[i % len(colors)]
        print(f"  Color {color}: {Path(npz_file).name}")

    scene = trimesh.Scene([point_cloud] + grippers)
    scene.show()


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Visualize generated grasps from diffusion model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", nargs="+", help="NPZ file(s) with generated grasps.")
    parser.add_argument(
        "--num_grasps", type=int, default=20,
        help="Number of grasps to show (per confidence level)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Success probability threshold for high/low confidence."
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare multiple files side-by-side (different colors)."
    )

    args = parser.parse_args(argv)

    if args.compare and len(args.input) > 1:
        visualize_comparison(args.input, num_grasps=args.num_grasps)
    else:
        for f in args.input:
            visualize_generated_grasps(
                f,
                num_grasps=args.num_grasps,
                success_threshold=args.threshold
            )


if __name__ == "__main__":
    main()
