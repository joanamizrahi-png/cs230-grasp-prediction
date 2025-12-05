"""
Evaluate the Gaussian attention model on the test set.
"""

import argparse
import logging
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix
from tqdm import tqdm

import utils
import model.net_grasp_attention as net
import model.data_loader as data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/grasp_attention',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Name of checkpoint file to load")


def evaluate(model, loss_fn, dataloader, metrics, params, pos_weight):
    """Evaluate the model and return metrics and predictions."""
    model.eval()
    summ = []
    all_predictions = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
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

    metrics_mean = {metric: np.mean([x[metric] for x in summ])
                    for metric in summ[0]}

    metrics_mean['roc_auc'] = roc_auc_score(all_labels, all_logits)
    metrics_mean['avg_precision'] = average_precision_score(all_labels, all_logits)

    metrics_string = " ; ".join(f"{k}: {v:05.3f}" for k, v in metrics_mean.items())
    logging.info("- Test metrics: " + metrics_string)

    return metrics_mean, all_predictions, all_labels, all_logits


def plot_curves(all_labels, all_logits, metrics, model_dir):
    """Plot and save ROC and PR curves."""
    fpr, tpr, _ = roc_curve(all_labels, all_logits)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'roc_curve.png'), dpi=300)
    plt.close()

    precision, recall, _ = precision_recall_curve(all_labels, all_logits)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f'PR (AP = {metrics["avg_precision"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'pr_curve.png'), dpi=300)
    plt.close()

    logging.info(f"- Saved curves to {model_dir}")


if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), f"No configuration file found at {json_path}"
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()
    params.device = torch.device('cuda' if params.cuda else 'cpu')

    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    logging.info("Creating the dataset...")
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']
    logging.info("- done.")

    attention_sigma = getattr(params, 'attention_sigma', 0.05)
    model = net.GraspSuccessPredictor(attention_sigma=attention_sigma).to(params.device)
    logging.info(f"Using Gaussian attention with sigma={attention_sigma}")

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info(f"Starting evaluation from {args.restore_file}")
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Same pos_weight as training for consistent loss
    pos_weight_value = 34.4 / 65.6
    pos_weight = torch.tensor([pos_weight_value]).to(params.device)

    test_metrics, predictions, labels, logits = evaluate(
        model, loss_fn, test_dl, metrics, params, pos_weight
    )

    save_path = os.path.join(args.model_dir, f"metrics_test_{args.restore_file}.json")
    utils.save_dict_to_json(test_metrics, save_path)

    plot_curves(labels, logits, test_metrics, args.model_dir)

    # Print confusion matrix
    predictions_binary = np.array(predictions).astype(int)
    labels_binary = np.array(labels).astype(int)
    cm = confusion_matrix(labels_binary, predictions_binary)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(f"True Negatives (correct failures):  {tn}")
    print(f"False Positives (wrong successes):  {fp}")
    print(f"False Negatives (wrong failures):   {fn}")
    print(f"True Positives (correct successes): {tp}")
    print(f"\nActual - Success: {np.sum(labels_binary)} ({np.mean(labels_binary)*100:.1f}%), "
          f"Failure: {len(labels_binary)-np.sum(labels_binary)} ({(1-np.mean(labels_binary))*100:.1f}%)")
    print(f"Predicted - Success: {np.sum(predictions_binary)} ({np.mean(predictions_binary)*100:.1f}%), "
          f"Failure: {len(predictions_binary)-np.sum(predictions_binary)} ({(1-np.mean(predictions_binary))*100:.1f}%)")
    print("="*60)
