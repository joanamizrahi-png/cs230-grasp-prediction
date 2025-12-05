"""
Train the learned attention model
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
import model.net_learned_attention as net
import model.data_loader as data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/learned_attention',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")


def train(model, optimizer, loss_fn, dataloader, metrics, params, pos_weight):
    """Train the model on `num_steps` batches"""
    model.train()
    summ = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for i, batch in enumerate(dataloader):
            points = batch['points'].to(params.device)
            grasp = batch['grasp'].to(params.device)
            labels = batch['label'].to(params.device)

            output = model(points, grasp)
            loss = loss_fn(output, labels, pos_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % params.save_summary_steps == 0:
                summary_batch = {metric: metrics[metric](output, labels)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    metrics_mean = {metric: np.mean([x[metric] for x in summ])
                    for metric in summ[0]}
    metrics_string = " ; ".join(f"{k}: {v:05.3f}" for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    return metrics_mean


def evaluate(model, loss_fn, dataloader, metrics, params, pos_weight):
    """Evaluate the model"""
    model.eval()
    summ = []
    all_predictions = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            points = batch['points'].to(params.device)
            grasp = batch['grasp'].to(params.device)
            labels = batch['label'].to(params.device)

            output = model(points, grasp)
            loss = loss_fn(output, labels, pos_weight)

            summary_batch = {metric: metrics[metric](output, labels)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

            probs = torch.sigmoid(output.squeeze())
            predictions = probs > 0.5

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(probs.cpu().numpy())

    if len(summ) == 0:
        logging.warning("- Eval: No data in dataloader, skipping evaluation")
        return {'loss': 0.0, 'accuracy': 0.0, 'roc_auc': 0.0, 'avg_precision': 0.0}

    metrics_mean = {metric: np.mean([x[metric] for x in summ])
                    for metric in summ[0]}

    from sklearn.metrics import roc_auc_score, average_precision_score

    metrics_mean['roc_auc'] = roc_auc_score(all_labels, all_logits)
    metrics_mean['avg_precision'] = average_precision_score(all_labels, all_logits)

    metrics_string = " ; ".join(f"{k}: {v:05.3f}" for k, v in metrics_mean.items())
    logging.info("- Eval metrics: " + metrics_string)

    return metrics_mean


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch."""
    start_epoch = 0
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info(f"Restoring parameters from {restore_path}")
        checkpoint = utils.load_checkpoint(restore_path, model, optimizer)
        start_epoch = checkpoint.get('epoch', 0)
        logging.info(f"Resuming from epoch {start_epoch + 1}")

    best_val_ap = 0.0

    pos_weight_value = 34.4 / 65.6
    pos_weight = torch.tensor([pos_weight_value]).to(params.device)

    logging.info(f"Using pos_weight = {pos_weight_value:.3f}")

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_roc_auc': [],
        'val_avg_precision': [],
        'learning_rate': []
    }

    history_path = os.path.join(model_dir, 'training_history.json')
    if restore_file is not None and os.path.exists(history_path):
        import json
        with open(history_path, 'r') as f:
            existing_history = json.load(f)
        for key in history.keys():
            if key in existing_history:
                history[key] = existing_history[key][:start_epoch]
        logging.info(f"Loaded existing training history ({len(history['train_loss'])} epochs)")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(start_epoch, params.num_epochs):
        logging.info(f"Epoch {epoch + 1}/{params.num_epochs}")

        train_metrics = train(model, optimizer, loss_fn, train_dataloader, metrics, params, pos_weight)
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params, pos_weight)

        val_ap = val_metrics['avg_precision']
        is_best = val_ap >= best_val_ap

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)

        if is_best:
            logging.info("- Found new best Average Precision")
            best_val_ap = val_ap
            best_json_path = os.path.join(model_dir, "metrics_val_best.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        last_json_path = os.path.join(model_dir, "metrics_val_last.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        history['val_avg_precision'].append(val_metrics['avg_precision'])
        history['learning_rate'].append(scheduler.get_last_lr()[0])

        utils.save_dict_to_json(history, history_path)
        scheduler.step()

    logging.info(f"Saved training history to {history_path}")
    plot_training_curves(history, model_dir)


def plot_training_curves(history, model_dir):
    """Plot training curves and save as PNG files."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], linewidth=2, label='Train', marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], linewidth=2, label='Validation', marker='o', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training Loss', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['val_roc_auc'], linewidth=2, marker='o', markersize=4, color='green')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('ROC-AUC', fontsize=11)
    ax1.set_title('Validation ROC-AUC', fontsize=13)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['val_avg_precision'], linewidth=2, marker='o', markersize=4, color='purple')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Average Precision', fontsize=11)
    ax2.set_title('Validation Average Precision', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'metrics_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), f"No json configuration file found at {json_path}"
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()
    params.device = torch.device('cuda' if params.cuda else 'cpu')

    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    logging.info("Loading the datasets...")

    dataloaders = data_loader.fetch_dataloader(['train', 'val'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Learned attention settings
    attention_dim = getattr(params, 'attention_dim', 64)
    pooling_ratio = getattr(params, 'pooling_ratio', 0.5)
    dropout = getattr(params, 'dropout', 0.3)
    weight_decay = getattr(params, 'weight_decay', 0.0)

    model = net.LearnedAttentionPredictor(
        attention_dim=attention_dim,
        pooling_ratio=pooling_ratio,
        dropout=dropout,
    ).to(params.device)

    logging.info(f"Using LEARNED attention with attention_dim={attention_dim}")
    logging.info(f"Using pooling_ratio={pooling_ratio}, dropout={dropout}, weight_decay={weight_decay}")

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=weight_decay)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params,
                       args.model_dir, args.restore_file)
