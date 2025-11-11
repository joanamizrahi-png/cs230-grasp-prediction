# AWS Setup Guide

Instructions for running training on AWS EC2.

## Table of Contents
1. [First-Time Setup](#first-time-setup)
2. [Running Training](#running-training)
3. [Downloading Results](#downloading-results)
4. [Common Issues](#common-issues)

## First-Time Setup

### Connect to AWS

```bash
ssh -i ~/.ssh/cs230-final-key.pem ec2-user@<your-instance-ip>
```

### Clone Repository

```bash
cd ~
git clone <your-repo-url> grasp-prediction
cd grasp-prediction
```

### Setup Python Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Note: Always activate the virtual environment before running Python scripts:
```bash
source .venv/bin/activate
```

### Download Dataset

The ACRONYM dataset has two parts: grasp annotations and 3D meshes.

**Download grasp annotations:**
```bash
cd ~/grasp-prediction
source .venv/bin/activate

mkdir -p data/grasps
pip install gdown
gdown https://drive.google.com/uc?id=1OjykLD9YmnFdfYpH2qO8yBo-I-22vKwu -O acronym.tar.gz
tar -xzf acronym.tar.gz
mv grasps/*.h5 data/grasps/
rm acronym.tar.gz
rmdir grasps
```

**Download meshes:**

ShapeNet meshes are available on HuggingFace (https://huggingface.co/datasets/ShapeNet/ShapeNetSem). After getting access, download and extract:

```bash
mkdir -p data/meshes
# Download ShapeNetSem from HuggingFace
wget <download-link> -O shapenet.zip
unzip shapenet.zip
mv ShapeNetSem/models/* data/meshes/
```

**Verify setup:**
```bash
python check_dataset.py --data_dir data
```

## Running Training

### Basic Training

```bash
cd ~/grasp-prediction
source .venv/bin/activate

# Create experiment directory
mkdir -p experiments/my_experiment
cp experiments/test_pipeline/params.json experiments/my_experiment/

# Run training in background
nohup python train.py --data_dir data --model_dir experiments/my_experiment > train.log 2>&1 &
```

### Monitor Progress

```bash
# Watch training log
tail -f experiments/my_experiment/train.log

# Check if still running
ps aux | grep train.py
```

### Evaluation

After training finishes:

```bash
python evaluate.py --data_dir data --model_dir experiments/my_experiment --restore_file best
```

## Downloading Results

From your local machine:

```bash
# Download entire experiment folder
scp -i ~/.ssh/cs230-final-key.pem -r ec2-user@<ip>:~/grasp-prediction/experiments/my_experiment ~/Downloads/
```

Results include:
- `best.pth.tar` - best model checkpoint
- `training_curves.png` - training progress plots
- `roc_curve.png` and `pr_curve.png` - evaluation curves
- `train.log` - full training log

## Common Issues

**"No module named 'torch'"**
- Solution: Activate virtual environment with `source .venv/bin/activate`

**Out of memory error**
- Solution: Reduce `batch_size` in `params.json`

**Can't find meshes**
- Solution: Verify data structure with `python check_dataset.py --data_dir data`

## Updating Code

When you make changes on your local machine:

```bash
# On local machine
git add .
git commit -m "description"
git push

# On AWS
cd ~/grasp-prediction
git pull
source .venv/bin/activate
# Continue with training
```

## Virtual Environments

Virtual environments keep project dependencies isolated. Create a separate `.venv` on each machine (Mac and AWS) - don't upload the Mac version to AWS.

**Why?**
- Different operating systems (macOS vs Linux)
- Platform-specific compiled packages
- Large size (~1-2 GB)

**Workflow:**
- Mac: Create `.venv`, develop code, push to GitHub
- AWS: Create new `.venv`, pull code, run training
