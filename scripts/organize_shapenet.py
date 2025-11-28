"""
Organize ShapeNet meshes into category folders expected by ACRONYM.

The ShapeNetSem download has a flat structure (models-OBJ/models/*.obj),
but ACRONYM expects meshes/CategoryName/hash.obj. This script reads the
ACRONYM grasp files to find the expected paths and organizes the meshes accordingly.
"""

import h5py
from pathlib import Path
import shutil
from tqdm import tqdm
from collections import defaultdict


def organize_meshes(grasp_dir, shapenet_obj_dir, output_mesh_dir):
    """
    Organize ShapeNet OBJ files into category folders.

    Args:
        grasp_dir: Path to ACRONYM grasps directory (contains .h5 files)
        shapenet_obj_dir: Path to ShapeNetSem models-OBJ/models directory
        output_mesh_dir: Where to create organized meshes directory
    """

    grasp_path = Path(grasp_dir)
    shapenet_path = Path(shapenet_obj_dir)
    output_path = Path(output_mesh_dir)

    print("=" * 60)
    print("Organizing ShapeNet Meshes for ACRONYM")
    print("=" * 60)
    print()

    # Get all h5 files
    h5_files = list(grasp_path.glob('*.h5'))
    print(f"Found {len(h5_files)} grasp files")

    # Build a mapping of expected paths from h5 files
    expected_paths = {}  # {obj_filename: full_expected_path}
    categories_needed = set()

    print("Reading expected mesh paths from grasp files...")
    for h5_file in tqdm(h5_files):
        try:
            with h5py.File(h5_file, 'r') as f:
                if 'object/file' not in f:
                    continue

                # Expected path like: meshes/Mug/10f6e09036350e92b3f21f1137c3c347.obj
                mesh_rel_path = f['object/file'][()].decode('utf-8')

                # Extract filename and category
                parts = Path(mesh_rel_path).parts
                if len(parts) >= 3 and parts[0] == 'meshes':
                    category = parts[1]
                    obj_filename = parts[2]

                    expected_paths[obj_filename] = mesh_rel_path
                    categories_needed.add(category)
        except Exception as e:
            print(f"Error reading {h5_file.name}: {e}")
            continue

    print(f"\nFound {len(expected_paths)} unique meshes needed")
    print(f"Categories needed: {len(categories_needed)}")
    print()

    # Get available OBJ files from ShapeNet
    available_objs = {}
    print("Scanning ShapeNet OBJ files...")
    for obj_file in tqdm(list(shapenet_path.glob('*.obj'))):
        available_objs[obj_file.name] = obj_file

    print(f"Found {len(available_objs)} OBJ files in ShapeNet")
    print()

    # Copy files to organized structure
    output_path.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = []

    print("Organizing meshes into category folders...")
    for obj_filename, expected_path in tqdm(expected_paths.items()):
        if obj_filename in available_objs:
            # Create category directory
            dest_file = output_path.parent / expected_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            if not dest_file.exists():
                shutil.copy2(available_objs[obj_filename], dest_file)
            copied += 1
        else:
            missing.append(obj_filename)

    # Also copy .mtl files if they exist
    print("\nCopying material files (.mtl)...")
    for obj_filename in tqdm(expected_paths.keys()):
        mtl_filename = obj_filename.replace('.obj', '.mtl')
        mtl_src = shapenet_path / mtl_filename

        if mtl_src.exists() and obj_filename in available_objs:
            expected_path = expected_paths[obj_filename]
            mtl_dest = output_path.parent / expected_path.replace('.obj', '.mtl')

            if not mtl_dest.exists():
                shutil.copy2(mtl_src, mtl_dest)

    print()
    print("=" * 60)
    print("Organization Complete")
    print("=" * 60)
    print(f"Meshes copied: {copied}")
    print(f"Missing meshes: {len(missing)}")
    print(f"Success rate: {100 * copied / len(expected_paths):.1f}%")
    print()

    if len(missing) > 0 and len(missing) <= 10:
        print(f"Missing files: {missing}")
    elif len(missing) > 10:
        print(f"First 10 missing: {missing[:10]}")

    print()
    print(f"Organized meshes location: {output_path}")
    print()

    return copied, len(missing)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Organize ShapeNet meshes for ACRONYM')
    parser.add_argument('--grasp_dir', required=True,
                        help='Path to ACRONYM grasps directory')
    parser.add_argument('--shapenet_dir', required=True,
                        help='Path to ShapeNetSem models-OBJ/models directory')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for organized meshes')

    args = parser.parse_args()

    organize_meshes(args.grasp_dir, args.shapenet_dir, args.output_dir)
