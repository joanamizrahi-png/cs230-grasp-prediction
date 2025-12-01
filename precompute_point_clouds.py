#!/usr/bin/env python3
"""
Pre-compute point clouds from meshes to speed up training.

This script:
1. Loads all mesh files
2. Applies the scale factor from ACRONYM dataset
3. Samples point clouds from each scaled mesh
4. Normalizes to [-1, 1] for neural network training
5. Saves them as .npy files for fast loading during training

Run this once before training (future training runs will be much faster).
"""

import argparse
import numpy as np
import trimesh
from pathlib import Path
from tqdm import tqdm
import h5py


def sample_point_cloud(mesh, num_points):
    """Sample points uniformly from mesh surface.

    Note: Mesh should already be scaled before calling this function.
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def normalize_point_cloud(points):
    """Normalize point cloud to [-1, 1] based on bounding box.

    This normalization is important for neural network training stability.
    Centers the point cloud at origin and scales to fit in unit cube.

    Returns:
        normalized_points: Normalized point cloud
        centroid: Center of point cloud (for denormalization)
        max_dist: Maximum distance from center (for denormalization)
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid

    max_dist = np.max(np.abs(points))
    if max_dist > 0:
        points = points / max_dist

    return points, centroid, max_dist


def precompute_point_clouds(data_path, num_points, output_dir, no_normalize=False):
    """
    Pre-compute point clouds for all meshes.

    Args:
        data_path: Path to ACRONYM data directory
        num_points: Number of points to sample per mesh
        output_dir: Directory to save pre-computed point clouds
        no_normalize: If True, skip normalization (for joint normalization in data loader)
    """
    data_path = Path(data_path)
    # Add num_points to the directory structure
    output_dir = Path(output_dir) / f'{num_points}pts'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all grasp files
    grasp_dir = data_path / 'grasps'
    h5_files = sorted(list(grasp_dir.glob('*.h5')))

    print(f"Found {len(h5_files)} grasp files")
    print(f"Sampling {num_points} points per mesh")
    print(f"Saving to: {output_dir}")
    print()

    # Track unique meshes to avoid reprocessing
    processed_meshes = set()
    skipped = 0
    errors = 0

    for h5_path in tqdm(h5_files, desc='Processing meshes'):
        try:
            with h5py.File(h5_path, 'r') as f:
                # Get mesh path and scale
                if 'object/file' not in f:
                    continue

                mesh_rel_path = f['object/file'][()].decode('utf-8')
                mesh_path = data_path / mesh_rel_path

                # Get scale factor - CRITICAL for alignment with grasps!
                scale = f['object/scale'][()]

                if not mesh_path.exists():
                    errors += 1
                    continue

                # Skip if already processed
                if str(mesh_path) in processed_meshes:
                    skipped += 1
                    continue

                # Create output path (maintain directory structure)
                rel_path = mesh_path.relative_to(data_path)
                if no_normalize:
                    output_path = output_dir / rel_path.with_suffix('.npy')
                else:
                    output_path = output_dir / rel_path.with_suffix('.npz')

                # Skip if output already exists
                if output_path.exists():
                    processed_meshes.add(str(mesh_path))
                    skipped += 1
                    continue

                # Load mesh and apply scale (to match grasp coordinate system!)
                mesh = trimesh.load(mesh_path, force='mesh')
                mesh.apply_scale(scale)
                points = sample_point_cloud(mesh, num_points)

                # Optionally normalize for neural network training
                if not no_normalize:
                    points, centroid, max_dist = normalize_point_cloud(points)

                    # Save point cloud AND normalization parameters
                    # This allows us to normalize grasps consistently
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(output_path.with_suffix('.npz'),
                             points=points.astype(np.float32),
                             centroid=centroid.astype(np.float32),
                             max_dist=np.float32(max_dist))
                else:
                    # Save point cloud only (no normalization)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(output_path, points.astype(np.float32))

                processed_meshes.add(str(mesh_path))

        except Exception as e:
            errors += 1
            tqdm.write(f"Error processing {h5_path}: {e}")

    print()
    print("="*60)
    print("Pre-computation complete")
    print("="*60)
    print(f"Processed {len(processed_meshes)} unique meshes")
    print(f"Skipped {skipped} (already existed)")
    print(f"Errors: {errors}")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-compute point clouds from meshes')
    parser.add_argument('--data_dir', default='data',
                        help='Path to ACRONYM data directory')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Number of points to sample per mesh')
    parser.add_argument('--output_dir', default='data/point_clouds',
                        help='Directory to save pre-computed point clouds')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip normalization (for joint normalization in data loader)')

    args = parser.parse_args()

    precompute_point_clouds(args.data_dir, args.num_points, args.output_dir, args.no_normalize)
