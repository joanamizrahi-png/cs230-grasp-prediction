"""
Data loading utilities for ACRONYM grasp dataset.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import trimesh
from tqdm import tqdm


class GraspDataset(Dataset):
    """
    Dataset for loading point clouds and grasp labels from ACRONYM.
    
    Each sample consists of:
    - Point cloud: (N, 3) array of 3D points sampled from object mesh
    - Grasp pose: position (3,) + rotation matrix (3, 3) + width (1,)
    - Label: binary success/failure (1 or 0)
    """
    
    def __init__(self, data_path, num_points=2048, augment=True, split='train',
                 split_by='object', max_grasps_per_object=None, use_precomputed=True):
        """
        Args:
            data_path: Path to ACRONYM data (should contain 'meshes/' and 'grasps/' subdirs)
            num_points: Number of points to sample from mesh
            augment: Whether to apply data augmentation
            split: 'train', 'val', or 'test'
            split_by: 'object' (split objects) or 'grasp' (split grasps within objects)
            max_grasps_per_object: Limit grasps per object (None = use all)
            use_precomputed: Whether to use pre-computed point clouds (much faster!)
        """
        self.data_path = data_path
        self.num_points = num_points
        self.augment = augment
        self.split = split
        self.split_by = split_by
        self.max_grasps_per_object = max_grasps_per_object
        self.use_precomputed = use_precomputed

        # Check if pre-computed point clouds are available
        # Point clouds are organized by num_points: data/point_clouds/512pts/, data/point_clouds/2048pts/, etc.
        self.precomputed_dir = Path(data_path) / 'point_clouds' / f'{num_points}pts'
        if self.use_precomputed and not self.precomputed_dir.exists():
            print(f"Warning: Pre-computed point clouds not found at {self.precomputed_dir}")
            print("Falling back to loading meshes on-the-fly (slower)")
            print(f"To speed up training, run: python precompute_point_clouds.py --num_points {num_points}")
            self.use_precomputed = False

        # Load dataset samples
        self.samples = self._load_samples()

        print(f'{split} dataset: {len(self.samples)} samples')
        
    def _load_samples(self):
        """Load all samples for this split."""
        if self.split_by == 'object':
            return self._load_samples_split_by_object()
        else:
            return self._load_samples_split_by_grasp()
    
    def _load_samples_split_by_object(self):
        """
        Split dataset by OBJECTS: some objects in train, others in val/test.
        Better for testing cross-category generalization.
        """
        samples = []
        
        grasp_dir = Path(self.data_path) / 'grasps'
        if not grasp_dir.exists():
            print(f"Warning: Grasp directory not found: {grasp_dir}")
            return samples
        
        h5_files = sorted(list(grasp_dir.glob('*.h5')))
        print(f"Found {len(h5_files)} object files")
        
        # Split objects: 70% train, 15% val, 15% test
        np.random.seed(42)
        indices = np.arange(len(h5_files))
        np.random.shuffle(indices)
        
        n_train = int(0.7 * len(h5_files))
        n_val = int(0.15 * len(h5_files))
        
        if self.split == 'train':
            selected_indices = indices[:n_train]
        elif self.split == 'val':
            selected_indices = indices[n_train:n_train+n_val]
        else:  # test
            selected_indices = indices[n_train+n_val:]
        
        h5_files = [h5_files[i] for i in selected_indices]
        print(f"Using {len(h5_files)} objects for {self.split} split")
        
        # Load each object file
        for h5_path in tqdm(h5_files, desc=f'Loading {self.split} data'):
            samples.extend(self._load_grasps_from_file(h5_path))
        
        self._print_dataset_stats(samples)
        return samples
    
    def _load_samples_split_by_grasp(self):
        """
        Split dataset by GRASPS: all objects used, but their grasps split into train/val/test.
        Better for having balanced object distribution across splits.
        """
        samples = []
        
        grasp_dir = Path(self.data_path) / 'grasps'
        if not grasp_dir.exists():
            print(f"Warning: Grasp directory not found: {grasp_dir}")
            return samples
        
        h5_files = sorted(list(grasp_dir.glob('*.h5')))
        print(f"Found {len(h5_files)} object files")
        
        # Load all objects but split their grasps
        for h5_path in tqdm(h5_files, desc=f'Loading {self.split} data'):
            object_samples = self._load_grasps_from_file(h5_path, split_grasps=True)
            samples.extend(object_samples)
        
        self._print_dataset_stats(samples)
        return samples
    
    def _load_grasps_from_file(self, h5_path, split_grasps=False):
        """
        Load grasps from a single .h5 file.
        
        Args:
            h5_path: Path to .h5 file
            split_grasps: If True, only return grasps for current split
        """
        samples = []
        
        try:
            with h5py.File(h5_path, 'r') as f:
                # Get mesh path
                if 'object/file' in f:
                    mesh_rel_path = f['object/file'][()].decode('utf-8')
                    mesh_path = Path(self.data_path) / mesh_rel_path
                    
                    if not mesh_path.exists():
                        print(f"Warning: Mesh not found: {mesh_path}")
                        return samples
                else:
                    return samples
                
                # Get grasp data
                if 'grasps/transforms' not in f:
                    return samples
                
                transforms = f['grasps/transforms'][:]
                qualities = f['grasps/qualities/flex/object_in_gripper'][:]
                labels = (qualities > 0).astype(np.float32)
                
                # Get grasp widths if available
                if 'grasps/widths' in f:
                    widths = f['grasps/widths'][:]
                else:
                    widths = np.full(len(transforms), 0.08)
                
                # Determine which grasps to use
                n_grasps = len(transforms)
                if split_grasps:
                    # Split grasps for this object
                    indices = np.arange(n_grasps)
                    np.random.seed(42 + hash(str(h5_path)) % 1000)
                    np.random.shuffle(indices)
                    
                    n_train = int(0.7 * n_grasps)
                    n_val = int(0.15 * n_grasps)
                    
                    if self.split == 'train':
                        indices = indices[:n_train]
                    elif self.split == 'val':
                        indices = indices[n_train:n_train+n_val]
                    else:  # test
                        indices = indices[n_train+n_val:]
                else:
                    # Use all grasps
                    indices = np.arange(n_grasps)
                
                # Limit number of grasps if specified
                if self.max_grasps_per_object is not None:
                    indices = indices[:self.max_grasps_per_object]
                
                # Create samples
                for i in indices:
                    transform = transforms[i]
                    position = transform[:3, 3]
                    rotation = transform[:3, :3]
                    
                    grasp_data = {
                        'position': position,
                        'rotation': rotation,
                        'width': widths[i]
                    }
                    
                    samples.append((str(mesh_path), grasp_data, labels[i]))
        
        except Exception as e:
            print(f"Error loading {h5_path}: {e}")
        
        return samples
    
    def _print_dataset_stats(self, samples):
        """Print statistics about the dataset."""
        print(f"Loaded {len(samples)} grasp samples for {self.split}")
        
        if len(samples) > 0:
            labels = [s[2] for s in samples]
            success_rate = np.mean(labels)
            n_success = int(np.sum(labels))
            n_failure = len(labels) - n_success
            
            print(f"  Success: {n_success} ({success_rate:.1%})")
            print(f"  Failure: {n_failure} ({1-success_rate:.1%})")
            
            # Count unique objects
            unique_meshes = len(set([s[0] for s in samples]))
            print(f"  Unique objects: {unique_meshes}")
    
    def _sample_point_cloud(self, mesh):
        """Sample points uniformly from mesh surface."""
        points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
        return points
    
    def _normalize_point_cloud(self, points):
        """Normalize point cloud to [-1, 1] based on bounding box."""
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        max_dist = np.max(np.abs(points))
        if max_dist > 0:
            points = points / max_dist
        
        return points
    
    def _augment_point_cloud(self, points):
        """
        Apply data augmentation:
        - Random rotation around Z-axis
        - Random jittering (Gaussian noise)
        - Random point dropout

        Returns:
            points: augmented point cloud
            rotation_matrix: 3x3 rotation matrix applied (or identity if no augmentation)
        """
        # Generate random rotation around Z-axis (up axis)
        angle = np.random.uniform(0, 2 * np.pi)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ], dtype=np.float32)

        if not self.augment:
            return points, np.eye(3, dtype=np.float32)

        # Apply rotation
        points = points @ rotation_matrix.T

        # Add Gaussian noise
        noise = np.random.normal(0, 0.02, points.shape)
        points = points + noise

        # Random dropout
        dropout_rate = np.random.uniform(0.05, 0.10)
        keep_indices = np.random.choice(
            len(points),
            int(len(points) * (1 - dropout_rate)),
            replace=False
        )
        points = points[keep_indices]

        if len(points) < self.num_points:
            padding_size = self.num_points - len(points)
            padding = points[np.random.choice(len(points), padding_size)]
            points = np.vstack([points, padding])

        return points, rotation_matrix
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Load a single sample.

        Returns:
            dict with:
                - points: (num_points, 3) point cloud (normalized to [-1,1])
                - grasp: (13,) grasp parameters [position(3), rotation_flat(9), width(1)]
                         Position normalized to [-1,1], width kept in meters
                - label: scalar binary label
        """
        mesh_path, grasp_data, label = self.samples[idx]

        # Variables to store normalization parameters
        centroid = None
        max_dist = None

        # Load point cloud (either pre-computed or from mesh)
        try:
            if self.use_precomputed:
                # Load pre-computed point cloud with normalization params (FAST!)
                mesh_path_obj = Path(mesh_path)
                rel_path = mesh_path_obj.relative_to(self.data_path)
                pc_path = self.precomputed_dir / rel_path.with_suffix('.npz')

                if pc_path.exists():
                    # Load npz file with points, centroid, and max_dist
                    data = np.load(pc_path)
                    points = data['points']
                    centroid = data['centroid']
                    max_dist = data['max_dist']
                else:
                    # Check if old .npy format exists
                    pc_path_old = self.precomputed_dir / rel_path.with_suffix('.npy')
                    if pc_path_old.exists():
                        raise RuntimeError(
                            f"Found old .npy format without normalization params: {pc_path_old}\n"
                            f"Please re-run: python precompute_point_clouds.py --num_points {self.num_points}\n"
                            f"This will generate .npz files with centroid/max_dist for proper grasp normalization."
                        )
                    else:
                        raise FileNotFoundError(
                            f"No precomputed point cloud found at {pc_path}\n"
                            f"Please run: python precompute_point_clouds.py --num_points {self.num_points}"
                        )
            else:
                # Load from mesh (SLOW - only for backward compatibility)
                mesh = trimesh.load(mesh_path, force='mesh')
                sampled_points = self._sample_point_cloud(mesh)
                centroid = np.mean(sampled_points, axis=0)
                centered = sampled_points - centroid
                max_dist = np.max(np.abs(centered))
                if max_dist > 0:
                    points = centered / max_dist
                else:
                    points = centered

        except Exception as e:
            print(f"Error loading point cloud from {mesh_path}: {e}")
            points = np.zeros((self.num_points, 3), dtype=np.float32)
            grasp = np.zeros(13, dtype=np.float32)
            return {
                'points': torch.from_numpy(points),
                'grasp': torch.from_numpy(grasp),
                'label': torch.tensor(label, dtype=torch.float32)
            }

        # Normalize grasp position using same centroid and max_dist as point cloud
        position = grasp_data['position'].astype(np.float32)
        if centroid is not None and max_dist is not None and max_dist > 0:
            # Normalize position to [-1, 1] using same params as point cloud
            position = (position - centroid) / max_dist

        rotation = grasp_data['rotation'].astype(np.float32)
        width = np.array([grasp_data['width']], dtype=np.float32)  # Keep width in meters

        # Apply augmentation (rotation/noise) and get rotation matrix
        points, rotation_matrix = self._augment_point_cloud(points)

        # Apply same rotation to grasp position and orientation
        position = rotation_matrix @ position
        rotation = rotation_matrix @ rotation  # Rotate the 3x3 rotation matrix

        grasp = np.concatenate([position, rotation.flatten(), width])

        return {
            'points': torch.from_numpy(points.astype(np.float32)),
            'grasp': torch.from_numpy(grasp),
            'label': torch.tensor(label, dtype=torch.float32),
            'max_dist': torch.tensor(max_dist if max_dist is not None else 0.05, dtype=torch.float32)  # For visualization
        }


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    
    Args:
        types: (list) has one or more of 'train', 'val', 'test'
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    
    for split in types:
        # Determine if we should use augmentation
        if split == 'train':
            augment = True
            shuffle = True
        else:
            augment = False
            shuffle = False
        
        # Create dataset
        dataset = GraspDataset(
            data_path=data_dir,
            num_points=params.num_points,
            augment=augment,
            split=split,
            split_by=params.split_by,
            max_grasps_per_object=params.max_grasps_per_object if hasattr(params, 'max_grasps_per_object') else None,
            use_precomputed=params.use_precomputed if hasattr(params, 'use_precomputed') else True
        )
        
        # Create dataloader
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=params.batch_size,
            shuffle=shuffle,
            num_workers=params.num_workers,
            pin_memory=params.cuda
        )
    
    return dataloaders
