"""
Evaluation script for Grasp Prediction with PointNet++
Adapted for CS230
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import utils
from model.data_loader import GraspDataset

# --- MODIFIED IMPORT ---
from model.net_pointnet2 import PointNet2Grasp # Import PointNet++

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir containing weights to load")

def evaluate(model, loss_fn, dataloader, metrics, params):
    """
    Evaluate the model on `dataloader`.
    Returns:
        metrics_mean: (dict) average of all metrics on the set
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize summary accumulator
    summ = []

    # To store all predictions and labels for ROC/PR curves
    all_outputs = []
    all_labels = []

    # Iterate over batches
    # Use torch.no_grad() to save memory since we don't need gradients for evaluation
    with torch.no_grad():
        for data_batch in dataloader:
            
            # 1. Prepare Data
            points = data_batch['points'].float()
            labels = data_batch['label'].float()

            if params.cuda:
                points, labels = points.cuda(), labels.cuda()

            # --- CRITICAL FOR POINTNET++ ---
            # Permute dimensions: [Batch, N_Points, Channels] -> [Batch, Channels, N_Points]
            points = points.permute(0, 2, 1)
            # -------------------------------

            # 2. Forward Pass
            output = model(points)
            output = output.squeeze() # [Batch, 1] -> [Batch]
            
            # Compute Loss
            # loss_fn is BCEWithLogitsLoss, so it takes logits
            loss = loss_fn(output, labels)

            # 3. Process Outputs for Metrics
            # Convert Logits -> Probabilities [0, 1]
            probs = torch.sigmoid(output)
            
            # Convert Probabilities -> Binary Predictions (0 or 1)
            preds = (probs > 0.5).float()

            # Move to CPU for metric calculation
            output_batch = preds.cpu().numpy() # Hard predictions for accuracy
            probs_batch = probs.cpu().numpy()  # Probabilities for ROC/AUC
            labels_batch = labels.cpu().numpy()

            # 4. Compute Metrics
            # For accuracy, we use hard predictions (0 or 1)
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
            
            # Accumulate for plotting
            all_outputs.append(probs_batch) # Store probabilities!
            all_labels.append(labels_batch)

    # Compute mean of metrics
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    
    # 5. Plot Curves (ROC and PR)
    # Flatten lists
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    
    # Save curves to the model directory
    # Ensure utils.plot_curves exists and handles probabilities
    try:
        utils.plot_curves(all_labels, all_outputs, params.model_dir)
    except AttributeError:
        logging.warning("utils.plot_curves not found. Skipping curve generation.")

    return metrics_mean


if __name__ == '__main__':
    """
    Main function to run evaluation independently
    """
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # Load data
    test_dl = GraspDataset(args.data_dir, num_points=params.num_points, split='test')

    test_loader = torch.utils.data.DataLoader(test_dl, batch_size=params.batch_size, shuffle=False,
                                              num_workers=params.num_workers, pin_memory=params.cuda)

    logging.info("- done.")

    # Define the model
    # --- INSTANTIATE POINTNET++ ---
    model = PointNet2Grasp(normal_channel=False) 
    
    if params.cuda:
        model = model.cuda()
        
    # Define Loss function (BCEWithLogitsLoss for stability)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Get metrics
    metrics = GraspDataset.metrics

    # Load weights
    logging.info("Starting evaluation")
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_loader, metrics, params)
    
    # Save results
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)