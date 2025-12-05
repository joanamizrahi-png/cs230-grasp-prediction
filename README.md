# Grasp Success Prediction from 3D Point Clouds

CS230 Deep Learning Project - Joana Mizrahi, Loïc Poisson, Abigail Aleshire

Predicting grasp success directly from 3D point clouds using deep learning architectures including PointNet, PointNet++, and attention-enhanced variants.

## Project Structure

```
grasp-prediction/
├── model/
│   ├── net.py                     # PointNet baseline model
│   ├── net_pointnet2.py           # PointNet++ model
│   ├── net_grasp_attention.py     # Gaussian attention model
│   ├── net_learned_attention.py   # Learned attention model
│   └── data_loader.py             # Dataset and data loading
├── scripts/
│   ├── check_dataset.py           # Dataset verification
│   ├── filter_dataset.py          # Create filtered subset
│   └── organize_shapenet.py       # ShapeNet data organization
├── train.py                       # Train PointNet baseline
├── train_pointnet2.py             # Train PointNet++
├── train_attention.py             # Train Gaussian attention model
├── train_learned_attention.py     # Train learned attention model
├── evaluate.py                    # Evaluate PointNet baseline
├── evaluate_pointnet2.py          # Evaluate PointNet++
├── evaluate_attention.py          # Evaluate attention models
├── precompute_point_clouds.py     # Precompute point clouds from meshes
├── utils.py                       # Helper functions
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/joanamizrahi-png/cs230-grasp-prediction.git
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

Download the full dataset from [ACRONYM GitHub](https://github.com/NVlabs/acronym#downloads):
- `acronym.tar.gz` (1.6 GB) - grasp annotations
- ShapeNet meshes (51 GB) - requires registration at shapenet.org

Extract grasps and organize meshes:
```bash
# Extract ACRONYM grasps
tar -xzf acronym.tar.gz -C data/

# ShapeNet downloads as flat files, but ACRONYM expects meshes/Category/hash.obj
# Use the organize script to restructure:
python scripts/organize_shapenet.py \
    --grasp_dir data/grasps \
    --shapenet_dir /path/to/ShapeNetSem/models-OBJ/models \
    --output_dir data/meshes

# (Optional) If you don't have all meshes, filter to matched pairs:
python scripts/filter_dataset.py --data_dir data --output_dir data_filtered
```

### 5. Precompute point clouds

```bash
python precompute_point_clouds.py --data_dir data --num_points 1024
```

## Usage

### Training

**PointNet baseline:**
```bash
python train.py --data_dir data --model_dir experiments/pointnet
```

**PointNet++:**
```bash
python train_pointnet2.py --data_dir data --model_dir experiments/pointnet2
```

**Gaussian attention:**
```bash
python train_attention.py --data_dir data --model_dir experiments/attention
```

**Learned attention:**
```bash
python train_learned_attention.py --data_dir data --model_dir experiments/learned_attention
```

### Evaluation

```bash
python evaluate.py --data_dir data --model_dir experiments/pointnet --restore_file best
python evaluate_pointnet2.py --data_dir data --model_dir experiments/pointnet2 --restore_file best
python evaluate_attention.py --data_dir data --model_dir experiments/attention --restore_file best
```

### Hyperparameters

Edit `experiments/<model>/params.json`:

```json
{
    "learning_rate": 0.001,
    "batch_size": 128,
    "num_epochs": 50,
    "num_points": 1024,
    "num_workers": 4,
    "split_by": "object",
    "max_grasps_per_object": 50,
    "attention_sigma": 1.0
}
```

## Model Architectures

### PointNet Baseline
Global feature extraction from point clouds using shared MLPs and max pooling, concatenated with grasp parameters for binary classification.

### PointNet++
Hierarchical point cloud processing with set abstraction layers for multi-scale feature learning.

### Gaussian Attention
Spatial attention mechanism using Gaussian distance weighting centered on grasp position with learnable σ parameter.

### Learned Attention
Query-key attention mechanism that learns to focus on task-relevant point cloud regions.

## Dataset Statistics

- **Objects**: 8,872 meshes from ShapeNet
- **Categories**: 262 object categories
- **Grasps**: ~17.7 million labeled grasp attempts
- **Success rate**: ~35% (handled with weighted BCE loss)
- **Split**: 183 train, 39 val, 40 test categories

## Results

| Model | Accuracy | ROC-AUC | Avg Precision |
|-------|----------|---------|---------------|
| PointNet (baseline) | 72.9% | 0.795 | 0.864 |
| PointNet++ | 72.4% | 0.768 | 0.841 |
| Gaussian Attention (σ=1.0) | 73.2% | 0.803 | 0.871 |
| Learned Attention | 73.4% | 0.804 | 0.871 |

*Results on test set with 512 points per object.*

**Best model** (learned attention, 2048 points, regularized): **74.1% accuracy**, **0.813 ROC-AUC**, **0.875 AP**

## Team

- Joana Mizrahi - jmizrahi@stanford.edu
- Loïc Poisson - lpoisson@stanford.edu
- Abigail Aleshire - abbya@stanford.edu

## References

1. [PointNet: Deep Learning on Point Sets](https://arxiv.org/abs/1612.00593)
2. [PointNet++: Deep Hierarchical Feature Learning](https://arxiv.org/abs/1706.02413)
3. [ACRONYM: A Large-Scale Grasp Dataset](https://arxiv.org/abs/2011.09584)
