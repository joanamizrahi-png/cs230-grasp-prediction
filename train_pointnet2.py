"""
Training script for Grasp Prediction with PointNet++
"""

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import utils
import torch.nn as nn

# --- MODIFIED IMPORTS ---
from model.data_loader import GraspDataset
from model.net_pointnet2 import PointNet2Grasp  # Import PointNet++ model
from evaluate_pointnet2 import evaluate 

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """
    Train the model on one epoch.
    """
    # Set model to training mode
    model.train()

    # Initialize averages for metrics
    summ = []
    loss_avg = utils.RunningAverage()

    # Progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (data_batch) in enumerate(dataloader):
            
            # 1. Prepare data
            # DataLoader returns a dictionary
            points = data_batch['points'].float()
            labels = data_batch['label'].float()

            # Move to GPU if available
            if params.cuda:
                points, labels = points.cuda(), labels.cuda()

            # --- CRITICAL MODIFICATION FOR POINTNET++ ---
            # PointNet++ expects [Batch, Channels, N_Points] -> [B, 3, N]
            # The dataset usually provides [B, N, 3]
            points = points.permute(0, 2, 1) 
            # ---------------------------------------------

            # 2. Forward pass
            # Model output: Logits of shape [Batch, 1]
            output = model(points)
            
            # Squeeze to go from [B, 1] to [B] to match labels shape
            output = output.squeeze()

            # 3. Compute loss
            # BCEWithLogitsLoss takes raw logits and labels
            loss = loss_fn(output, labels)

            # 4. Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 5. Evaluate metrics (for tqdm display only)
            if i % params.save_summary_steps == 0:
                # move to cpu, and convert to numpy
                # --- CONVERSION LOGITS -> PROBS ---
                # Apply Sigmoid here only for display metrics
                probs = torch.sigmoid(output)
                preds = (probs > 0.5).float() # Threshold at 0.5
                
                output_batch = preds.detach().cpu().numpy()
                labels_batch = labels.detach().cpu().numpy()

                # Compute metrics defined in params.json (e.g., accuracy)
                summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # Update average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # Compute mean metrics for the entire epoch
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """
    Train the model and evaluate every epoch.
    """
    # Reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # --- TRAIN ---
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # --- EVALUATE ---
        # Note: Ensure your evaluate.py also handles Sigmoid conversion!
        # If evaluate.py uses the same model, it will receive Logits.
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, save best metrics
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Load data
    train_dl = GraspDataset(args.data_dir, num_points=params.num_points, split='train')
    val_dl = GraspDataset(args.data_dir, num_points=params.num_points, split='val')

    # Data Loaders
    train_loader = torch.utils.data.DataLoader(train_dl, batch_size=params.batch_size, shuffle=True,
                                               num_workers=params.num_workers, pin_memory=params.cuda)
    val_loader = torch.utils.data.DataLoader(val_dl, batch_size=params.batch_size, shuffle=False,
                                             num_workers=params.num_workers, pin_memory=params.cuda)

    logging.info("- done.")

    # Define the model and optimizer
    # --- INSTANTIATE POINTNET++ ---
    # normal_channel=False because ACRONYM/ShapeNet use XYZ by default
    model = PointNet2Grasp(normal_channel=False) 
    
    if params.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # --- LOSS FUNCTION ---
    # Use BCEWithLogitsLoss for numerical stability with logits
    loss_fn = nn.BCEWithLogitsLoss()

    # fetch loss function and metrics
    metrics = GraspDataset.metrics # Uses metrics defined in dataset or utils

    # Train the model
    logging.info("Starting training for {} epochs".format(params.num_epochs))
    train_and_evaluate(model, train_loader, val_loader, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)