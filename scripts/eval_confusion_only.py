"""Minimal evaluation script - just confusion matrix, faster imports"""
import sys
import os
import argparse

# Minimal imports
import torch
import numpy as np

# Only import what we absolutely need
sys.path.insert(0, os.path.dirname(__file__))
import utils
import model.net as net
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', required=True)
parser.add_argument('--restore_file', default='best')
args = parser.parse_args()

print(f"Loading model from {args.model_dir}...")

# Load params
params = utils.Params(os.path.join(args.model_dir, 'params.json'))
params.cuda = torch.cuda.is_available()
params.device = torch.device('cuda' if params.cuda else 'cpu')

# Load model
model = net.GraspSuccessPredictor().to(params.device)
utils.load_checkpoint(os.path.join(args.model_dir, f'{args.restore_file}.pth.tar'), model)
model.eval()

# Load test data
print("Loading test data...")
dataloaders = data_loader.fetch_dataloader(['test'], 'data', params)
test_dl = dataloaders['test']

# Get predictions
print("Running predictions...")
all_predictions = []
all_labels = []

with torch.no_grad():
    for i, batch in enumerate(test_dl):
        points = batch['points'].to(params.device)
        grasp = batch['grasp'].to(params.device)
        labels = batch['label'].to(params.device)

        output = model(points, grasp)
        probs = torch.sigmoid(output.squeeze())
        predictions = (probs > 0.5).float()

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(test_dl)} batches...")

# Compute confusion matrix manually (no sklearn needed)
predictions = np.array(all_predictions).astype(int)
labels = np.array(all_labels).astype(int)

tp = np.sum((predictions == 1) & (labels == 1))
tn = np.sum((predictions == 0) & (labels == 0))
fp = np.sum((predictions == 1) & (labels == 0))
fn = np.sum((predictions == 0) & (labels == 1))

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)
print(f"\n                    Predicted")
print(f"                Failure  Success")
print(f"Actual Failure    {tn:5d}    {fp:5d}")
print(f"       Success    {fn:5d}    {tp:5d}")

print(f"\nDetailed Counts:")
print(f"  True Negatives (correct failures):   {tn}")
print(f"  False Positives (wrong successes):   {fp}")
print(f"  False Negatives (wrong failures):    {fn}")
print(f"  True Positives (correct successes):  {tp}")

total = len(labels)
actual_success = np.sum(labels)
actual_failure = total - actual_success
pred_success = np.sum(predictions)
pred_failure = total - pred_success

print(f"\nActual Distribution:")
print(f"  Successes: {actual_success} ({actual_success/total*100:.1f}%)")
print(f"  Failures:  {actual_failure} ({actual_failure/total*100:.1f}%)")

print(f"\nPredicted Distribution:")
print(f"  Successes: {pred_success} ({pred_success/total*100:.1f}%)")
print(f"  Failures:  {pred_failure} ({pred_failure/total*100:.1f}%)")

print(f"\nMetrics:")
accuracy = (tp + tn) / total
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if tp + fn > 0:
    recall_success = tp / (tp + fn)
    print(f"  Success Recall: {recall_success:.4f} ({recall_success*100:.1f}%)")

if tn + fp > 0:
    recall_failure = tn / (tn + fp)
    print(f"  Failure Recall: {recall_failure:.4f} ({recall_failure*100:.1f}%)")

print(f"\nUnique predictions: {len(np.unique(predictions))}")
if len(np.unique(predictions)) == 1:
    pred_class = "SUCCESS" if predictions[0] == 1 else "FAILURE"
    print(f"  WARNING: Model predicts ONLY {pred_class}!")
    print("   This confirms the model has collapsed to predicting one class.")
else:
    print(f"Model predicts both classes")

print("="*60)
