"""
Train the model
"""

import argparse
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import utils
import model.net as net
import model.data_loader as data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")


def train(model, optimizer, loss_fn, dataloader, metrics, params, pos_weight):
    """Train the model on `num_steps` batches
    
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        pos_weight: (torch.Tensor) weight for positive class in BCE loss
    """

    # Set model to training mode
    model.train()

    # Summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, batch in enumerate(dataloader):
            # Move to GPU if available
            points = batch['points'].to(params.device)
            grasp = batch['grasp'].to(params.device)
            labels = batch['label'].to(params.device)

            # Compute model output and loss
            output = model(points, grasp)
            loss = loss_fn(output, labels, pos_weight)

            # Clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # Performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # Compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output, labels)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # Update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # Compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) 
                    for metric in summ[0]}
    metrics_string = " ; ".join(f"{k}: {v:05.3f}" for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    
    return metrics_mean


def evaluate(model, loss_fn, dataloader, metrics, params, pos_weight):
    """Evaluate the model on `num_steps` batches.
    
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        pos_weight: (torch.Tensor) weight for positive class in BCE loss
    """

    # Set model to evaluation mode
    model.eval()

    # Summary for current eval loop
    summ = []
    
    # Collect all predictions and labels for computing metrics
    all_predictions = []
    all_labels = []
    all_logits = []

    # Compute metrics over the dataset
    with torch.no_grad():
        for batch in dataloader:
            # Move to GPU if available
            points = batch['points'].to(params.device)
            grasp = batch['grasp'].to(params.device)
            labels = batch['label'].to(params.device)

            # Compute model output
            output = model(points, grasp)
            loss = loss_fn(output, labels, pos_weight)

            # Compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output, labels)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
            
            # Store predictions for later metric computation
            probs = torch.sigmoid(output.squeeze())
            predictions = probs > 0.5
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(probs.cpu().numpy())

    # Compute mean of all metrics in summary
    if len(summ) == 0:
        logging.warning("- Eval: No data in dataloader, skipping evaluation")
        return {'loss': 0.0, 'accuracy': 0.0, 'roc_auc': 0.0, 'avg_precision': 0.0}

    metrics_mean = {metric: np.mean([x[metric] for x in summ])
                    for metric in summ[0]}

    # Compute additional metrics using sklearn
    from sklearn.metrics import roc_auc_score, average_precision_score

    metrics_mean['roc_auc'] = roc_auc_score(all_labels, all_logits)
    metrics_mean['avg_precision'] = average_precision_score(all_labels, all_logits)
    
    metrics_string = " ; ".join(f"{k}: {v:05.3f}" for k, v in metrics_mean.items())
    logging.info("- Eval metrics: " + metrics_string)
    
    return metrics_mean


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.
    
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # Reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info(f"Restoring parameters from {restore_path}")
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_ap = 0.0

    # Compute pos_weight for handling class imbalance
    # Assuming ~15% success rate as mentioned in proposal
    pos_weight = torch.tensor([85.0 / 15.0]).to(params.device)

    # Track metrics for plotting
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_roc_auc': [],
        'val_avg_precision': [],
        'learning_rate': []
    }

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info(f"Epoch {epoch + 1}/{params.num_epochs}")

        # Compute number of batches in one epoch (one full pass over the training set)
        train_metrics = train(model, optimizer, loss_fn, train_dataloader, metrics, params, pos_weight)
        
        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params, pos_weight)

        val_ap = val_metrics['avg_precision']
        is_best = val_ap >= best_val_ap

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best Average Precision")
            best_val_ap = val_ap

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        # Save metrics to history for plotting
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        history['val_avg_precision'].append(val_metrics['avg_precision'])
        history['learning_rate'].append(scheduler.get_last_lr()[0])

        # Update learning rate
        scheduler.step()

    # Save training history
    history_path = os.path.join(model_dir, 'training_history.json')
    utils.save_dict_to_json(history, history_path)
    logging.info(f"Saved training history to {history_path}")

    # Plot training curves
    plot_training_curves(history, model_dir)
    logging.info(f"Saved training curves to {model_dir}")


def plot_training_curves(history, model_dir):
    """Plot training curves and save as PNG files."""

    epochs = range(1, len(history['train_loss']) + 1)

    # Combined plot: Loss and Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax1.plot(epochs, history['train_loss'], linewidth=2, label='Train', marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], linewidth=2, label='Validation', marker='o', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy subplot
    ax2.plot(epochs, history['train_acc'], linewidth=2, label='Train', marker='o', markersize=4)
    ax2.plot(epochs, history['val_acc'], linewidth=2, label='Validation', marker='o', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Training Accuracy', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Metrics plot: ROC-AUC and Average Precision
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC-AUC
    ax1.plot(epochs, history['val_roc_auc'], linewidth=2, marker='o', markersize=4, color='green')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('ROC-AUC', fontsize=11)
    ax1.set_title('Validation ROC-AUC', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Average Precision
    ax2.plot(epochs, history['val_avg_precision'], linewidth=2, marker='o', markersize=4, color='purple')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Average Precision', fontsize=11)
    ax2.set_title('Validation Average Precision', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'metrics_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
    params = utils.Params(json_path)

    # Use GPU if available
    params.cuda = torch.cuda.is_available()
    params.device = torch.device('cuda' if params.cuda else 'cpu')

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    
    # Fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.GraspSuccessPredictor().to(params.device)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # Fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params,
                       args.model_dir, args.restore_file)
