"""
Utilities for fixing/validating generated grasps:
- orthonormalize_rotation: fix rotation matrices via SVD
- fix_grasp_approach: make gripper point toward object
- project_grasp_to_surface: move grasp closer to object
"""

import torch
import numpy as np


def orthonormalize_rotation(rotation_flat):
    """
    Orthonormalize a 3x3 rotation matrix using SVD to ensure it's a valid rotation.

    Diffusion models can produce slightly distorted rotation matrices that don't
    satisfy R @ R.T = I and det(R) = 1. This function projects the matrix to the
    closest valid rotation matrix.

    Args:
        rotation_flat: (9,) or (batch, 9) flattened rotation matrix

    Returns:
        orthonormalized_flat: (9,) or (batch, 9) valid rotation matrix
    """
    is_batched = len(rotation_flat.shape) == 2
    is_torch = isinstance(rotation_flat, torch.Tensor)

    if not is_batched:
        R = rotation_flat.reshape(3, 3)

        if is_torch:
            # SVD decomposition: R = U @ S @ V^T
            # Closest rotation is: R_orth = U @ V^T
            U, S, Vh = torch.linalg.svd(R)
            R_orth = U @ Vh

            # Ensure det(R) = 1 (not -1, which would be a reflection)
            if torch.det(R_orth) < 0:
                U[:, -1] = -U[:, -1]
                R_orth = U @ Vh
        else:
            U, S, Vh = np.linalg.svd(R)
            R_orth = U @ Vh

            if np.linalg.det(R_orth) < 0:
                U[:, -1] = -U[:, -1]
                R_orth = U @ Vh

        return R_orth.flatten()

    else:
        # Batched version
        batch_size = rotation_flat.shape[0]

        if is_torch:
            result = torch.zeros_like(rotation_flat)
        else:
            result = np.zeros_like(rotation_flat)

        for i in range(batch_size):
            result[i] = orthonormalize_rotation(rotation_flat[i])

        return result


def orthonormalize_grasps(grasps):
    """
    Orthonormalize the rotation matrices in grasp parameters.

    Args:
        grasps: (13,) or (batch, 13) grasp parameters [pos(3), rot(9), width(1)]

    Returns:
        grasps_fixed: grasps with orthonormalized rotation matrices
    """
    is_batched = len(grasps.shape) == 2
    is_torch = isinstance(grasps, torch.Tensor)

    if is_torch:
        grasps_fixed = grasps.clone()
    else:
        grasps_fixed = grasps.copy()

    if not is_batched:
        grasps_fixed[3:12] = orthonormalize_rotation(grasps[3:12])
    else:
        grasps_fixed[:, 3:12] = orthonormalize_rotation(grasps[:, 3:12])

    return grasps_fixed


def fix_grasp_approach(grasps, points, debug=False):
    """
    Fix grasp orientation so gripper fingers are closer to object than handle.

    The gripper extends along +Z from origin (handle at Z=0) to fingers (Z≈0.11).
    For a correct grasp, the finger tips should be closer to the object center
    than the gripper origin (handle). If not, we flip the grasp 180° around Y.

    Args:
        grasps: (batch, 13) grasp parameters [pos(3), rot(9), width(1)]
        points: (batch, N, 3) point clouds
        debug: if True, print statistics

    Returns:
        grasps_fixed: grasps with corrected approach directions
    """
    is_torch = isinstance(grasps, torch.Tensor)
    GRIPPER_LENGTH = 0.10  # Approximate distance from origin to finger tips

    if is_torch:
        grasps_fixed = grasps.clone()
    else:
        grasps_fixed = grasps.copy()

    batch_size = grasps.shape[0]
    num_flipped = 0

    for i in range(batch_size):
        pos = grasps[i, :3]  # Gripper origin (handle end)
        rot = grasps[i, 3:12].reshape(3, 3)

        # Get approach direction (Z-axis, third column)
        approach = rot[:, 2]

        # Compute where the finger tips would be
        # In normalized space, we need to scale GRIPPER_LENGTH appropriately
        # But since both pos and points are normalized, we can use a reasonable offset
        finger_tip = pos + approach * GRIPPER_LENGTH

        if is_torch:
            centroid = points[i].mean(dim=0)

            # Check if fingers are closer to object center than handle
            dist_origin_to_center = torch.norm(pos - centroid)
            dist_tip_to_center = torch.norm(finger_tip - centroid)

            # If handle is closer than fingers, flip the grasp
            if dist_tip_to_center > dist_origin_to_center:
                # R_flip = diag(-1, 1, -1) flips X and Z, keeping Y
                rot_fixed = rot.clone()
                rot_fixed[:, 0] = -rot[:, 0]  # Flip X-axis
                rot_fixed[:, 2] = -rot[:, 2]  # Flip Z-axis (approach)
                grasps_fixed[i, 3:12] = rot_fixed.flatten()
                num_flipped += 1
        else:
            centroid = points[i].mean(axis=0)

            dist_origin_to_center = np.linalg.norm(pos - centroid)
            dist_tip_to_center = np.linalg.norm(finger_tip - centroid)

            if dist_tip_to_center > dist_origin_to_center:
                rot_fixed = rot.copy()
                rot_fixed[:, 0] = -rot[:, 0]
                rot_fixed[:, 2] = -rot[:, 2]
                grasps_fixed[i, 3:12] = rot_fixed.flatten()
                num_flipped += 1

    if debug:
        print(f"  fix_grasp_approach: flipped {num_flipped}/{batch_size} grasps ({100*num_flipped/batch_size:.1f}%)")

    return grasps_fixed


def project_grasp_to_surface(grasp, points, max_distance=0.05, fix_width=True,
                            fixed_width_value=0.08, align_orientation=True, k_neighbors=10):
    """
    Project grasp center to nearest point on object surface and optionally fix constraints.

    This ensures generated grasps are close to the object, fixing the issue
    where diffusion outputs grasps far from the surface.

    Args:
        grasp: (13,) or (batch, 13) grasp parameters [position(3), rotation(9), width(1)]
        points: (N, 3) or (batch, N, 3) point cloud
        max_distance: max allowed distance in meters (default 5cm)
        fix_width: if True, set width to fixed_width_value (default True)
        fixed_width_value: gripper width in meters (default 0.08m = 8cm)
        align_orientation: if True, align grasp approach with surface normal (default True)
        k_neighbors: number of neighbors for normal estimation (default 10)

    Returns:
        projected_grasp: grasp with corrected position, width, and optionally orientation
    """
    is_batched = len(grasp.shape) == 2

    if not is_batched:
        # Single grasp
        grasp_center = grasp[:3]
        rotation_flat = grasp[3:12]
        width = grasp[12]

        is_torch = isinstance(grasp, torch.Tensor)

        # Find nearest point on surface
        if is_torch:
            distances = torch.norm(points - grasp_center, dim=1)
            nearest_idx = torch.argmin(distances)
            nearest_point = points[nearest_idx]
            min_distance = distances[nearest_idx]
        else:
            distances = np.linalg.norm(points - grasp_center, axis=1)
            nearest_idx = np.argmin(distances)
            nearest_point = points[nearest_idx]
            min_distance = distances[nearest_idx]

        # Create modified grasp
        projected_grasp = grasp.clone() if is_torch else grasp.copy()

        # Fix position if too far
        if min_distance > max_distance:
            projected_grasp[:3] = nearest_point

        # Fix width to constant value
        if fix_width:
            if is_torch:
                projected_grasp[12] = torch.tensor(fixed_width_value, dtype=grasp.dtype, device=grasp.device)
            else:
                projected_grasp[12] = fixed_width_value

        # Align orientation with surface normal
        if align_orientation:
            # Estimate surface normal from k nearest neighbors
            if is_torch:
                _, k_nearest_indices = torch.topk(distances, k_neighbors, largest=False)
                k_nearest_points = points[k_nearest_indices]

                # Compute covariance and normal via PCA
                centered = k_nearest_points - k_nearest_points.mean(dim=0)
                cov = (centered.T @ centered) / (k_neighbors - 1)
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                normal = eigenvectors[:, 0]  # smallest eigenvalue = normal direction

                # Ensure normal points outward (away from object center)
                object_center = points.mean(dim=0)
                to_surface = nearest_point - object_center
                if torch.dot(normal, to_surface) < 0:
                    normal = -normal

                # Reconstruct rotation matrix with approach aligned to normal
                # Original rotation matrix (3x3)
                R_original = rotation_flat.reshape(3, 3)

                # New approach vector = NEGATIVE surface normal (approach from outside)
                # The gripper should move inward toward the object, so we flip the normal
                approach = -normal / torch.norm(normal)

                # Keep original binormal direction (closing direction), orthogonalize
                binormal = R_original[:, 1]
                binormal = binormal - torch.dot(binormal, approach) * approach
                binormal = binormal / (torch.norm(binormal) + 1e-6)

                # Third axis = cross product
                third_axis = torch.cross(approach, binormal)

                # New rotation matrix
                R_new = torch.stack([approach, binormal, third_axis], dim=1)

                projected_grasp[3:12] = R_new.flatten()

            else:
                k_nearest_indices = np.argpartition(distances, k_neighbors)[:k_neighbors]
                k_nearest_points = points[k_nearest_indices]

                # Compute covariance and normal via PCA
                centered = k_nearest_points - k_nearest_points.mean(axis=0)
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                normal = eigenvectors[:, 0]  # smallest eigenvalue = normal direction

                # Ensure normal points outward
                object_center = points.mean(axis=0)
                to_surface = nearest_point - object_center
                if np.dot(normal, to_surface) < 0:
                    normal = -normal

                # Reconstruct rotation matrix
                R_original = rotation_flat.reshape(3, 3)
                # New approach vector = NEGATIVE surface normal (approach from outside)
                approach = -normal / np.linalg.norm(normal)
                binormal = R_original[:, 1]
                binormal = binormal - np.dot(binormal, approach) * approach
                binormal = binormal / (np.linalg.norm(binormal) + 1e-6)
                third_axis = np.cross(approach, binormal)
                R_new = np.stack([approach, binormal, third_axis], axis=1)

                projected_grasp[3:12] = R_new.flatten()

        return projected_grasp

    else:
        # Batched grasps
        batch_size = grasp.shape[0]
        projected_grasps = grasp.clone() if isinstance(grasp, torch.Tensor) else grasp.copy()

        for i in range(batch_size):
            projected_grasps[i] = project_grasp_to_surface(
                grasp[i],
                points[i] if len(points.shape) == 3 else points,
                max_distance=max_distance,
                fix_width=fix_width,
                fixed_width_value=fixed_width_value,
                align_orientation=align_orientation,
                k_neighbors=k_neighbors
            )

        return projected_grasps


def compute_distance_to_surface(grasp, points):
    """
    Compute minimum distance from grasp center to object surface.

    Args:
        grasp: (13,) or (batch, 13) grasp parameters
        points: (N, 3) or (batch, N, 3) point cloud

    Returns:
        distance: scalar or (batch,) minimum distance
    """
    is_batched = len(grasp.shape) == 2

    if not is_batched:
        grasp_center = grasp[:3]

        if isinstance(grasp, torch.Tensor):
            distances = torch.norm(points - grasp_center, dim=1)
            return torch.min(distances)
        else:
            distances = np.linalg.norm(points - grasp_center, axis=1)
            return np.min(distances)
    else:
        batch_size = grasp.shape[0]
        min_distances = []

        for i in range(batch_size):
            dist = compute_distance_to_surface(
                grasp[i],
                points[i] if len(points.shape) == 3 else points
            )
            min_distances.append(dist)

        if isinstance(grasp, torch.Tensor):
            return torch.stack(min_distances)
        else:
            return np.array(min_distances)


def compute_classifier_gradient_with_distance(grasp, points, classifier, distance_weight=0.5):
    """
    Compute classifier guidance gradient with distance penalty.

    This improves upon standard classifier guidance by explicitly penalizing
    grasps that are far from the object surface.

    Args:
        grasp: (batch, 13) grasp parameters (requires_grad=True)
        points: (batch, N, 3) point cloud
        classifier: GraspSuccessPredictor model
        distance_weight: weight for distance penalty (default 0.5)

    Returns:
        gradient: (batch, 13) gradient for guidance
    """
    # Classifier score (success probability)
    logits = classifier(points, grasp)
    success_prob = torch.sigmoid(logits.squeeze())

    # Distance penalty (exponential to strongly discourage far grasps)
    grasp_center = grasp[:, :3]  # (batch, 3)

    # Compute minimum distance to surface for each grasp
    min_distances = []
    for i in range(grasp.shape[0]):
        dists = torch.norm(points[i] - grasp_center[i], dim=1)
        min_distances.append(torch.min(dists))
    min_distance = torch.stack(min_distances)

    # Exponential penalty: exp(d / 0.02)
    # At d=0: penalty=1, at d=0.02: penalty=e≈2.7, at d=0.1: penalty≈148
    distance_penalty = torch.exp(min_distance / 0.02)

    # Combined objective: maximize success prob, minimize distance
    objective = success_prob - distance_weight * distance_penalty

    # Compute gradient
    objective_sum = objective.sum()
    gradient = torch.autograd.grad(objective_sum, grasp, create_graph=False)[0]

    return gradient


def filter_valid_grasps(grasps, points, max_distance=0.05):
    """
    Filter out grasps that are too far from the object surface.

    Args:
        grasps: (N, 13) generated grasps
        points: (M, 3) point cloud
        max_distance: maximum allowed distance in meters

    Returns:
        valid_grasps: (K, 13) filtered grasps where K <= N
        valid_indices: (K,) indices of valid grasps
    """
    distances = compute_distance_to_surface(grasps, points)

    if isinstance(grasps, torch.Tensor):
        valid_mask = distances <= max_distance
        valid_indices = torch.where(valid_mask)[0]
        valid_grasps = grasps[valid_mask]
    else:
        valid_mask = distances <= max_distance
        valid_indices = np.where(valid_mask)[0]
        valid_grasps = grasps[valid_mask]

    return valid_grasps, valid_indices


# Example usage functions

def sample_with_surface_projection(diffusion_model, points, num_samples=100,
                                   projection_interval=10, max_distance=0.05,
                                   guidance_scale=0.0, classifier=None,
                                   use_distance_guidance=True):
    """
    Sample grasps with periodic surface projection during diffusion.

    This is a wrapper around diffusion_model.sample() that adds surface
    projection every N steps to keep grasps close to the object.

    Args:
        diffusion_model: GraspDiffusion model
        points: (batch, N, 3) point cloud
        num_samples: number of grasps to generate
        projection_interval: project every N timesteps (default 10)
        max_distance: max distance for projection
        guidance_scale: classifier guidance scale
        classifier: optional classifier for guidance
        use_distance_guidance: use distance-aware gradient (vs standard)

    Returns:
        grasps: (batch * num_samples, 13) generated grasps
    """
    # This would require modifying the sample() method
    # For now, we'll do post-processing
    grasps = diffusion_model.sample(
        points,
        num_samples=num_samples,
        guidance_scale=guidance_scale,
        classifier=classifier
    )

    # Post-process: project all grasps to surface
    grasps_projected = project_grasp_to_surface(grasps, points, max_distance)

    return grasps_projected


def score_and_filter_grasps(grasps, points, classifier, max_distance=0.05, top_k=10):
    """
    Score generated grasps with classifier and filter by distance and quality.

    Args:
        grasps: (N, 13) generated grasps
        points: (M, 3) point cloud
        classifier: GraspSuccessPredictor model
        max_distance: max distance from surface
        top_k: return top K grasps by score

    Returns:
        best_grasps: (top_k, 13) best valid grasps
        scores: (top_k,) corresponding success probabilities
    """
    # Filter by distance first
    valid_grasps, valid_indices = filter_valid_grasps(grasps, points, max_distance)

    if len(valid_grasps) == 0:
        print(f"Warning: No valid grasps within {max_distance}m of surface. Using projection.")
        # Project all to surface
        valid_grasps = project_grasp_to_surface(grasps, points, max_distance)

    # Score with classifier
    with torch.no_grad():
        if not isinstance(valid_grasps, torch.Tensor):
            valid_grasps = torch.tensor(valid_grasps, dtype=torch.float32)
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)

        # Expand points to match batch size
        points_batch = points.unsqueeze(0).repeat(len(valid_grasps), 1, 1)

        logits = classifier(points_batch, valid_grasps)
        scores = torch.sigmoid(logits.squeeze())

    # Get top K
    top_k = min(top_k, len(valid_grasps))
    top_scores, top_indices = torch.topk(scores, top_k)
    best_grasps = valid_grasps[top_indices]

    return best_grasps, top_scores
