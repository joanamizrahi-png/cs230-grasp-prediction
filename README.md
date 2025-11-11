# Grasp Success Prediction from 3D Point Clouds

CS230 Deep Learning Project - Joana Mizrahi, Loïc Poisson, Abigail Aleshire

Predicting grasp success directly from 3D point clouds using PointNet architecture.

## Project Structure

```
grasp-prediction/
├── data/                          # ACRONYM dataset (not in repo)
│   ├── meshes/                    # 3D object meshes
│   └── grasps/                    # Grasp annotations (.h5 files)
├── data_filtered/                 # Subset for testing (not in repo)
├── model/
│   ├── net.py                     # PointNet model architecture
│   ├── data_loader.py             # GraspDataset class
├── experiments/
│   └── base_model/
│       └── params.json            # Hyperparameters
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── utils.py                       # Helper functions
├── check_dataset.py               # Dataset verification
├── filter_dataset.py              # Create filtered subset
├── requirements.txt               # Dependencies
├── README.md                      # This file
└── DATASET.md                     # Dataset download instructions
```

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd grasp-prediction
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download ACRONYM dataset

**Option A: Sample data (for quick testing)**
```bash
git clone https://github.com/NVlabs/acronym.git temp_acronym
cp -r temp_acronym/data/examples/* data/
rm -rf temp_acronym
```

**Option B: Full dataset**

1. Download the full dataset from [ACRONYM GitHub](https://github.com/NVlabs/acronym#downloads):
   - `acronym.tar.gz` (1.6 GB) - grasp annotations
   - ShapeNet meshes (51 GB) - requires registration at shapenet.org

2. Extract to the `data/` directory:
```bash
tar -xzf acronym.tar.gz
# Organize so you have:
# data/grasps/*.h5
# data/meshes/**/*.obj
```

## Usage

### Training

Train the baseline PointNet model:

```bash
python train.py --data_dir data --model_dir experiments/base_model
```

Training logs and checkpoints will be saved to `experiments/base_model/`.

Training generates:
- `best.pth.tar` - Best model checkpoint
- `training_curves.png` - Loss and accuracy plots
- `metrics_curves.png` - ROC-AUC and AP plots
- `train.log` - Full training log

### Evaluation

Evaluate on test set:

```bash
python evaluate.py --data_dir data --model_dir experiments/base_model --restore_file best
```

This will:
- Load the best model checkpoint
- Evaluate on test set
- Generate ROC and PR curves
- Save metrics to JSON

### Hyperparameter Tuning

Edit `experiments/base_model/params.json`:

```json
{
    "learning_rate": 0.001,
    "batch_size": 128,
    "num_epochs": 50,
    "num_points": 2048,
    "num_workers": 4,
    "save_summary_steps": 10,
    "split_by": "object"
}
```

Key parameters:
- `num_points`: Number of points sampled from each mesh (512, 1024, 2048)
- `batch_size`: Adjust based on GPU memory
- `split_by`: "object" for cross-category testing, "grasp" for balanced splits
- `num_workers`: Set to 0 on Colab, 4+ on clusters

## AWS Deployment

See [DATASET.md](DATASET.md) for dataset download instructions

**Quick start on AWS:**
```bash
# SSH into AWS
ssh -i ~/.ssh/cs230-final-key.pem ec2-user@<your-aws-ip>

# Setup (first time only)
git clone <your-repo-url> grasp-prediction
cd grasp-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run training
python train.py --data_dir data --model_dir experiments/my_model
```

## Dataset Statistics (Full ACRONYM)

- **Objects**: 8,872 meshes from ShapeNet
- **Categories**: 262 object categories
- **Grasps**: ~17.7 million labeled grasp attempts
- **Success rate**: ~15% (class imbalance)
- **Split**: 183 train, 39 val, 40 test categories (for cross-category generalization)

## Model Architecture

### PointNet Baseline

```
To complete
```

### PointNet++

```
To add
```

### Attention mechanism

```
To add
```

### Ablation studies

```
To add
```

### Grasp planning

```
To add
```

## Metrics

- **Average Precision (AP)**: Primary metric, accounts for class imbalance
- **ROC-AUC**: Discriminative ability across all thresholds
- **Accuracy**: Simple percentage correct (can be misleading with imbalance)



## Team Members

- **Joana Mizrahi** - jmizrahi@stanford.edu
- **Loïc Poisson** - lpoisson@stanford.edu  
- **Abigail Aleshire** - abbya@stanford.edu

## References

1. [PointNet: Deep Learning on Point Sets](https://arxiv.org/abs/1612.00593)
2. [ACRONYM: A Large-Scale Grasp Dataset](https://arxiv.org/abs/2011.09584)
3. [CS230 Code Examples](https://github.com/cs230-stanford/cs230-code-examples)
