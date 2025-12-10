"""
Generate grasps using diffusion model + classifier guidance.
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
from diffusion.diffusion_model import create_improved_diffusion_model
from diffusion.grasp_constraints import orthonormalize_grasps, fix_grasp_approach, compute_distance_to_surface
import model.net as net
import model.net_grasp_attention as net_attention
import model.data_loader as data_loader


def check_collision(position, rotation, points, gripper_depth=0.10, gripper_width=0.08):
    """Check if gripper collides with point cloud."""
    points_centered = points - position
    points_gripper = points_centered @ rotation

    z = points_gripper[:, 2]
    x = points_gripper[:, 0]
    y = points_gripper[:, 1]

    in_z = (z > 0) & (z < gripper_depth)
    in_x = np.abs(x) < gripper_width / 2
    in_y = np.abs(y) < gripper_width / 2

    return np.sum(in_z & in_x & in_y) > 5


def generate_grasps(diffusion_model, classifier, dataloader, params,
                    num_samples=10, guidance_scale=1.0, max_batches=None):
    """Generate grasps with classifier guidance."""
    diffusion_model.eval()
    classifier.eval()

    all_grasps = []
    all_probs = []
    all_points = []
    all_max_dists = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Generating')):
            if max_batches and batch_idx >= max_batches:
                break

            points = batch['points'].to(params.device)
            max_dists = batch['max_dist'].to(params.device)

            # generate grasps
            generated = diffusion_model.sample(
                points, num_samples=num_samples,
                guidance_scale=guidance_scale, classifier=classifier
            )

            points_repeated = points.repeat_interleave(num_samples, dim=0)

            # fix rotations and approach direction
            generated = orthonormalize_grasps(generated)
            generated = fix_grasp_approach(generated, points_repeated)
            generated[:, 12] = 0.08  # fix width

            # filter collisions
            grasps_np = generated.cpu().numpy()
            points_np = points_repeated.cpu().numpy()

            collision_mask = []
            for i in range(len(grasps_np)):
                pos = grasps_np[i, :3]
                rot = grasps_np[i, 3:12].reshape(3, 3)
                collision_mask.append(not check_collision(pos, rot, points_np[i]))
            collision_mask = np.array(collision_mask)

            # score with classifier
            logits = classifier(points_repeated, generated)
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            probs[~collision_mask] = 0.0  # mark collisions as bad

            max_dists_repeated = max_dists.repeat_interleave(num_samples)

            all_grasps.append(grasps_np)
            all_probs.append(probs)
            all_points.append(points_np)
            all_max_dists.append(max_dists_repeated.cpu().numpy())

    return {
        'grasps': np.concatenate(all_grasps),
        'success_probs': np.concatenate(all_probs),
        'points': np.concatenate(all_points),
        'max_dists': np.concatenate(all_max_dists),
        'num_samples': num_samples,
        'guidance_scale': guidance_scale
    }


def print_stats(results):
    """Print generation statistics."""
    probs = results['success_probs']
    print(f"\n=== Results ===")
    print(f"Generated {len(probs)} grasps")
    print(f"Success prob: {probs.mean():.3f} +/- {probs.std():.3f}")
    print(f"Range: [{probs.min():.3f}, {probs.max():.3f}]")

    for thresh in [0.5, 0.7, 0.8, 0.9]:
        count = (probs > thresh).sum()
        print(f"  > {thresh}: {count} ({100*count/len(probs):.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--diffusion_dir', default='experiments/DIFFUSION/diffusion_model')
    parser.add_argument('--classifier_dir', default='experiments/ATTENTION/grasp_attention')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--num_objects', type=int, default=None)
    parser.add_argument('--output_file', default='generated_grasps.npz')
    args = parser.parse_args()

    # load params
    diffusion_params = utils.Params(os.path.join(args.diffusion_dir, 'params.json'))
    classifier_params = utils.Params(os.path.join(args.classifier_dir, 'params.json'))

    params = diffusion_params
    params.cuda = torch.cuda.is_available()
    params.device = torch.device('cuda' if params.cuda else 'cpu')
    params.num_points = classifier_params.num_points

    print("Loading data...")
    dataloaders = data_loader.fetch_dataloader(['val'], args.data_dir, params)
    val_dl = dataloaders['val']

    # load diffusion model
    print("Loading diffusion model...")
    diffusion = create_improved_diffusion_model(
        num_timesteps=getattr(params, 'num_diffusion_steps', 500)
    )
    diffusion = diffusion.to(params.device)
    utils.load_checkpoint(os.path.join(args.diffusion_dir, 'best.pth.tar'), diffusion)

    # load classifier
    print("Loading classifier...")
    if hasattr(classifier_params, 'attention_sigma'):
        classifier = net_attention.GraspSuccessPredictor(
            point_dim=3, grasp_dim=13, hidden_dim=512,
            use_grasp_attention=True,
            attention_sigma=classifier_params.attention_sigma
        )
    else:
        classifier = net.GraspSuccessPredictor(point_dim=3, grasp_dim=13, hidden_dim=512)
    classifier = classifier.to(params.device)
    utils.load_checkpoint(os.path.join(args.classifier_dir, 'best.pth.tar'), classifier)

    # generate
    max_batches = None
    if args.num_objects:
        max_batches = (args.num_objects + params.batch_size - 1) // params.batch_size

    print(f"Generating {args.num_samples} grasps/object, guidance={args.guidance_scale}")
    results = generate_grasps(
        diffusion, classifier, val_dl, params,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        max_batches=max_batches
    )

    print_stats(results)

    output_path = os.path.join(args.diffusion_dir, args.output_file)
    print(f"Saving to {output_path}")
    np.savez(output_path, **results)
