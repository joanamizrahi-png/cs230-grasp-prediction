"""
Script to check the status of the ACRONYM dataset and verify setup.
"""

import os
from pathlib import Path
import h5py

def check_dataset(data_dir='data'):
    """Check the current state of the dataset."""

    print("=" * 60)
    print("ACRONYM Dataset Status Check")
    print("=" * 60)

    data_path = Path(data_dir)

    # Check grasps directory
    grasp_dir = data_path / 'grasps'
    if not grasp_dir.exists():
        print(f"\nGrasp directory not found: {grasp_dir}")
        print("   Create it with: mkdir -p data/grasps")
        return

    h5_files = list(grasp_dir.glob('*.h5'))
    print(f"\nGrasp files (.h5): {len(h5_files)}")

    if len(h5_files) == 0:
        print("   No grasp files found")
        print("   Download with:")
        print("   wget https://github.com/NVlabs/acronym/releases/download/v0.1.0/acronym.tar.gz")
        return
    elif len(h5_files) < 100:
        print(f"   Only {len(h5_files)} files (full dataset has 8,872)")
    else:
        print(f"   Good ({len(h5_files)} files)")

    # Check meshes directory
    mesh_dir = data_path / 'meshes'
    if not mesh_dir.exists():
        print(f"\nMesh directory not found: {mesh_dir}")
        print("   Create it with: mkdir -p data/meshes")
        return

    categories = [d for d in mesh_dir.iterdir() if d.is_dir()]
    print(f"\nMesh categories: {len(categories)}")

    if len(categories) == 0:
        print("   No mesh categories found")
        print("   You need to download ShapeNet meshes from https://shapenet.org/")
        return
    elif len(categories) < 50:
        print(f"   Only {len(categories)} categories (full dataset has 262)")
    else:
        print(f"   Good ({len(categories)} categories)")

    # Count total mesh files
    total_meshes = 0
    for cat_dir in categories:
        meshes = list(cat_dir.glob('*.obj'))
        total_meshes += len(meshes)

    print(f"\nTotal mesh files (.obj): {total_meshes}")
    if total_meshes < 100:
        print(f"   Only {total_meshes} meshes (full dataset has 8,872)")
    else:
        print(f"   Good ({total_meshes} meshes)")

    # Check a sample file to verify mesh paths
    print(f"\nChecking mesh path consistency...")
    if len(h5_files) > 0:
        sample_h5 = h5_files[0]
        try:
            with h5py.File(sample_h5, 'r') as f:
                if 'object/file' in f:
                    mesh_rel_path = f['object/file'][()].decode('utf-8')
                    mesh_full_path = data_path / mesh_rel_path

                    print(f"   Sample grasp file: {sample_h5.name}")
                    print(f"   Expected mesh path: {mesh_rel_path}")

                    if mesh_full_path.exists():
                        print(f"   Mesh found at expected location")
                    else:
                        print(f"   Mesh NOT found at: {mesh_full_path}")
                        print(f"   Check that your mesh directory structure matches!")

                # Check grasp data
                if 'grasps/transforms' in f:
                    n_grasps = len(f['grasps/transforms'])
                    print(f"   Number of grasps in this file: {n_grasps}")

                    if 'grasps/qualities/flex/object_in_gripper' in f:
                        qualities = f['grasps/qualities/flex/object_in_gripper'][:]
                        success_rate = (qualities > 0).mean()
                        print(f"   Success rate: {success_rate:.1%}")
        except Exception as e:
            print(f"   Error reading sample file: {e}")

    # Dataset splits estimation
    print(f"\nEstimated dataset splits (70/15/15):")
    if len(h5_files) >= 3:
        n_train = int(0.7 * len(h5_files))
        n_val = int(0.15 * len(h5_files))
        n_test = len(h5_files) - n_train - n_val

        print(f"   Train: {n_train} objects")
        print(f"   Val:   {n_val} objects")
        print(f"   Test:  {n_test} objects")

        if n_val == 0 or n_test == 0:
            print(f"   WARNING: Some splits will be empty")
            print(f"   You need at least 7 objects for non-empty splits")
    else:
        print(f"   Need at least 3 objects for proper splits")

    # Summary
    print("\n" + "=" * 60)
    if len(h5_files) >= 8000 and total_meshes >= 8000:
        print("Full dataset appears to be properly set up")
    elif len(h5_files) >= 100 and total_meshes >= 100:
        print("Partial dataset detected. Good for testing")
    else:
        print("Dataset incomplete. See messages above")
    print("=" * 60)

    return {
        'n_grasp_files': len(h5_files),
        'n_categories': len(categories),
        'n_meshes': total_meshes
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data',
                        help='Path to data directory')
    args = parser.parse_args()

    check_dataset(args.data_dir)
