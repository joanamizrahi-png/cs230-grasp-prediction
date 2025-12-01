"""
Visualize rotation augmentation to verify point clouds and grasps rotate together.
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

import model.data_loader as data_loader
import utils
import trimesh
from acronym_tools import create_gripper_marker

def grasp_to_transform(grasp):
    """Convert grasp parameters to 4x4 transformation matrix."""
    position = grasp[:3]
    rotation_matrix = grasp[3:12].reshape(3, 3)

    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = position

    return T

if __name__ == '__main__':
    print("Visualizing rotation augmentation...")
    print("="*60)

    # Load params
    params = utils.Params('experiments/NEW/normalized_grasps_512pts_50epochs/params.json')
    params.cuda = False
    params.device = 'cpu'
    params.num_workers = 0

    # Create dataset with augmentation ENABLED
    print("\nCreating dataset with augmentation enabled...")
    dataset = data_loader.GraspDataset(
        data_path='data',
        num_points=params.num_points,
        augment=True,  # Enable augmentation
        split='train',
        split_by=params.split_by,
        max_grasps_per_object=params.max_grasps_per_object,
        use_precomputed=True
    )

    # Create dataset with augmentation DISABLED (for comparison)
    print("Creating dataset with augmentation disabled...")
    dataset_no_aug = data_loader.GraspDataset(
        data_path='data',
        num_points=params.num_points,
        augment=False,  # Disable augmentation
        split='train',
        split_by=params.split_by,
        max_grasps_per_object=params.max_grasps_per_object,
        use_precomputed=True
    )

    # Get the same sample from both datasets
    idx = 0
    print(f"\nLoading sample {idx}...")

    # Original (no augmentation)
    sample_orig = dataset_no_aug[idx]
    points_orig = sample_orig['points'].numpy()
    grasp_orig = sample_orig['grasp'].numpy()

    # Augmented
    sample_aug = dataset[idx]
    points_aug = sample_aug['points'].numpy()
    grasp_aug = sample_aug['grasp'].numpy()

    # Load max_dist from the npz file to get proper gripper scale
    # Get the mesh path for this sample
    mesh_path, _, _ = dataset.samples[idx]
    from pathlib import Path
    mesh_path_obj = Path(mesh_path)
    rel_path = mesh_path_obj.relative_to('data')
    pc_path = Path('data/point_clouds/512pts') / rel_path.with_suffix('.npz')

    # Load max_dist
    if pc_path.exists():
        pc_data = np.load(pc_path)
        max_dist = float(pc_data['max_dist'])
        print(f"Loaded max_dist from {pc_path.name}: {max_dist:.4f}")
    else:
        print(f"Warning: Could not find {pc_path}, using default scale")
        max_dist = 0.05  # Default assumption

    print("\n" + "="*60)
    print("COMPARISON:")
    print("="*60)
    print(f"\nOriginal point cloud range: [{points_orig.min():.3f}, {points_orig.max():.3f}]")
    print(f"Augmented point cloud range: [{points_aug.min():.3f}, {points_aug.max():.3f}]")
    print(f"\nOriginal grasp position: {grasp_orig[:3]}")
    print(f"Augmented grasp position: {grasp_aug[:3]}")

    # Check if rotation was applied (positions should be different)
    pos_diff = np.linalg.norm(grasp_orig[:3] - grasp_aug[:3])
    print(f"\nPosition difference: {pos_diff:.4f}")
    if pos_diff > 0.01:
        print("✓ Rotation was applied!")
    else:
        print("⚠ No rotation detected (positions are very similar)")

    # Visualize both
    print("\n" + "="*60)
    print("VISUALIZATION:")
    print("="*60)
    print("Creating visualization...")
    print("  - BLUE point cloud & gripper = Original (no augmentation)")
    print("  - RED point cloud & gripper = Augmented (with rotation)")
    print("\nBoth should show the same object, just rotated differently!")

    # Create scene
    scene = trimesh.Scene()

    # Create point clouds
    pcd_orig = trimesh.PointCloud(points_orig, colors=[100, 100, 255, 255])  # Blue
    pcd_aug = trimesh.PointCloud(points_aug, colors=[255, 100, 100, 255])  # Red

    scene.add_geometry(pcd_orig, node_name='points_original')
    scene.add_geometry(pcd_aug, node_name='points_augmented')

    # Create grippers
    # Grasps are already normalized by data_loader
    # Use same scale as visualize_normalized_alignment: 1.0 / max_dist
    gripper_scale = 1.0 / max_dist
    print(f"Gripper scale: {gripper_scale:.2f}")

    T_orig = grasp_to_transform(grasp_orig)
    gripper_orig = create_gripper_marker(color=[0, 0, 255, 200])  # Blue
    gripper_orig.apply_scale(gripper_scale)
    gripper_orig.apply_transform(T_orig)
    scene.add_geometry(gripper_orig, node_name='gripper_original')

    T_aug = grasp_to_transform(grasp_aug)
    gripper_aug = create_gripper_marker(color=[255, 0, 0, 200])  # Red
    gripper_aug.apply_scale(gripper_scale)
    gripper_aug.apply_transform(T_aug)
    scene.add_geometry(gripper_aug, node_name='gripper_augmented')

    # Add coordinate axes
    axes = trimesh.creation.axis(origin_size=0.02, axis_length=0.5)
    scene.add_geometry(axes)

    # Visualize
    print("\nOpening visualization window...")
    print("(Close the window to exit)")
    scene.show()

    print("\n" + "="*60)
    print("Visualization closed.")
    print("="*60)
