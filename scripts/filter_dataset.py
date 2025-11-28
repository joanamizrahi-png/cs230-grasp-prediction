"""
Filter dataset to only include objects that have corresponding meshes.
Useful when you have grasp files but not all meshes downloaded.
"""

import h5py
from pathlib import Path
import shutil
from tqdm import tqdm

def filter_dataset(data_dir='data', output_dir='data_filtered'):
    """
    Create a filtered dataset with only grasp files that have corresponding meshes.

    Args:
        data_dir: Original data directory
        output_dir: Where to save filtered dataset
    """

    data_path = Path(data_dir)
    output_path = Path(output_dir)

    grasp_dir = data_path / 'grasps'
    mesh_dir = data_path / 'meshes'

    output_grasp_dir = output_path / 'grasps'
    output_mesh_dir = output_path / 'meshes'

    # Create output directories
    output_grasp_dir.mkdir(parents=True, exist_ok=True)
    output_mesh_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Filtering Dataset to Available Meshes")
    print("=" * 60)
    print()

    # Get all h5 files
    h5_files = list(grasp_dir.glob('*.h5'))
    print(f"Found {len(h5_files)} grasp files")

    # Get available mesh categories
    mesh_categories = [d.name for d in mesh_dir.iterdir() if d.is_dir()]
    print(f"Found {len(mesh_categories)} mesh categories: {mesh_categories}")
    print()

    # Filter grasp files
    valid_grasps = []
    missing_meshes = []

    print("Checking which grasp files have corresponding meshes...")
    for h5_file in tqdm(h5_files):
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'object/file' not in f:
                    continue

                mesh_rel_path = f['object/file'][()].decode('utf-8')
                mesh_full_path = data_path / mesh_rel_path

                if mesh_full_path.exists():
                    valid_grasps.append((h5_file, mesh_full_path))
                else:
                    missing_meshes.append(h5_file.name)
        except Exception as e:
            print(f"Error reading {h5_file.name}: {e}")
            continue

    print()
    print(f"Found {len(valid_grasps)} grasp files with meshes")
    print(f"Missing meshes for {len(missing_meshes)} grasp files")

    if len(valid_grasps) == 0:
        print("\nNo valid grasp-mesh pairs found")
        return

    # Copy valid files
    print()
    print("Copying valid files to filtered dataset...")

    copied_meshes = set()

    for h5_file, mesh_file in tqdm(valid_grasps):
        # Copy grasp file
        dest_grasp = output_grasp_dir / h5_file.name
        if not dest_grasp.exists():
            shutil.copy2(h5_file, dest_grasp)

        # Copy mesh file (only once per mesh)
        if mesh_file not in copied_meshes:
            # Get category and create directory
            category = mesh_file.parent.name
            category_dir = output_mesh_dir / category
            category_dir.mkdir(exist_ok=True)

            dest_mesh = category_dir / mesh_file.name
            if not dest_mesh.exists():
                shutil.copy2(mesh_file, dest_mesh)

            copied_meshes.add(mesh_file)

    print()
    print("=" * 60)
    print("Filtered dataset created")
    print("=" * 60)
    print(f"Location: {output_path}")
    print(f"Grasp files: {len(valid_grasps)}")
    print(f"Mesh files: {len(copied_meshes)}")
    print()
    print("Use this filtered dataset with:")
    print(f"  python train.py --data_dir {output_dir} --model_dir experiments/test_pipeline")
    print()


def show_stats(data_dir='data'):
    """Show statistics about available data."""

    data_path = Path(data_dir)
    grasp_dir = data_path / 'grasps'
    mesh_dir = data_path / 'meshes'

    h5_files = list(grasp_dir.glob('*.h5'))

    valid_count = 0
    missing_count = 0

    print("Checking dataset...")
    for h5_file in tqdm(h5_files[:100]):  # Sample first 100 for speed
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'object/file' not in f:
                    continue

                mesh_rel_path = f['object/file'][()].decode('utf-8')
                mesh_full_path = data_path / mesh_rel_path

                if mesh_full_path.exists():
                    valid_count += 1
                else:
                    missing_count += 1
        except:
            continue

    print(f"\nSample of 100 files:")
    print(f"  With meshes: {valid_count}")
    print(f"  Missing meshes: {missing_count}")
    print(f"  Estimated total with meshes: {int(valid_count * len(h5_files) / 100)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Filter dataset to available meshes')
    parser.add_argument('--data_dir', default='data',
                        help='Original data directory')
    parser.add_argument('--output_dir', default='data_filtered',
                        help='Output directory for filtered dataset')
    parser.add_argument('--stats', action='store_true',
                        help='Just show statistics, don\'t create filtered dataset')

    args = parser.parse_args()

    if args.stats:
        show_stats(args.data_dir)
    else:
        filter_dataset(args.data_dir, args.output_dir)
