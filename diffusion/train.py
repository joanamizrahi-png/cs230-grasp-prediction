"""
Train the diffusion model for grasp generation.
"""

import argparse
import logging
import os
import json
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import utils
from diffusion.diffusion_model import create_improved_diffusion_model
import model.data_loader as data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data')
parser.add_argument('--model_dir', default='experiments/DIFFUSION/diffusion_model')
parser.add_argument('--restore_file', default=None)
parser.add_argument('--position_weight', type=float, default=2.0,
                    help="weight for position loss (position is harder to learn)")


def train_epoch(model, optimizer, dataloader, params, position_weight=1.0):
    """Train for one epoch."""
    model.train()
    losses = []

    with tqdm(total=len(dataloader)) as t:
        for batch in dataloader:
            points = batch['points'].to(params.device)
            grasp = batch['grasp'].to(params.device)

            loss = model.forward_with_position_weight(grasp, points, position_weight=position_weight)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())
            t.set_postfix(loss=f'{loss.item():.4f}')
            t.update()

    avg_loss = sum(losses) / len(losses)
    logging.info(f"- Train loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, dataloader, params):
    """Evaluate on val set."""
    model.eval()
    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            points = batch['points'].to(params.device)
            grasp = batch['grasp'].to(params.device)
            loss = model(grasp, points)
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    logging.info(f"- Val loss: {avg_loss:.4f}")
    return avg_loss


def check_rotation_quality(model, dataloader, params, epoch):
    """Check if generated rotations are valid."""
    model.eval()
    batch = next(iter(dataloader))
    points = batch['points'][:8].to(params.device)

    with torch.no_grad():
        generated = model.sample(points, num_samples=1)

    rot_matrices = generated[:, 3:12].reshape(-1, 3, 3).cpu().numpy()
    dets = [np.linalg.det(r) for r in rot_matrices]
    ortho_errors = [np.linalg.norm(r @ r.T - np.eye(3)) for r in rot_matrices]

    logging.info(f"Rotation quality (epoch {epoch}):")
    logging.info(f"  det: {np.mean(dets):.4f} +/- {np.std(dets):.4f}")
    logging.info(f"  ortho error: {np.mean(ortho_errors):.4f}")


def train_and_evaluate(model, train_dl, val_dl, optimizer, params, model_dir, position_weight=2.0):
    """Main training loop."""
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params.num_epochs, eta_min=1e-6
    )

    for epoch in range(params.num_epochs):
        logging.info(f"Epoch {epoch + 1}/{params.num_epochs}")

        train_loss = train_epoch(model, optimizer, train_dl, params, position_weight)
        history['train_loss'].append(train_loss)

        val_loss = evaluate(model, val_dl, params)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 10 == 0:
            check_rotation_quality(model, val_dl, params, epoch + 1)

        is_best = val_loss <= best_val_loss
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict()
        }, is_best=is_best, checkpoint=model_dir)

        if is_best:
            logging.info("- New best model!")
            best_val_loss = val_loss

        scheduler.step()

        # save training curves
        plt.figure(figsize=(10, 4))
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(model_dir, 'training_curves.png'), dpi=150)
        plt.close()

    return history


if __name__ == '__main__':
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # create default params if needed
    json_path = os.path.join(args.model_dir, 'params.json')
    if not os.path.isfile(json_path):
        default_params = {
            "learning_rate": 0.0001,
            "batch_size": 64,
            "num_epochs": 100,
            "num_points": 512,
            "num_diffusion_steps": 500,
            "num_workers": 4,
            "split_by": "object",
            "max_grasps_per_object": 50
        }
        with open(json_path, 'w') as f:
            json.dump(default_params, f, indent=4)

    params = utils.Params(json_path)
    params.cuda = torch.cuda.is_available()
    params.device = torch.device('cuda' if params.cuda else 'cpu')

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logging.info("Loading data...")
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("Creating diffusion model...")
    model = create_improved_diffusion_model(
        num_timesteps=getattr(params, 'num_diffusion_steps', 500)
    )
    model = model.to(params.device)
    logging.info(f"- {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=0.01)

    if args.restore_file:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info(f"Restoring from {restore_path}")
        utils.load_checkpoint(restore_path, model, optimizer)

    logging.info(f"Training with position_weight={args.position_weight}")
    train_and_evaluate(model, train_dl, val_dl, optimizer, params, args.model_dir,
                       position_weight=args.position_weight)
