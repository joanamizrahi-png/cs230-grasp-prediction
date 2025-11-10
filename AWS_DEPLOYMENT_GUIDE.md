# AWS Deployment Guide - Grasp Prediction Project

Complete guide for deploying and running your PointNet grasp prediction model on AWS.

---

## 📋 Table of Contents
1. [Project Structure Overview](#project-structure-overview)
2. [GitHub Setup](#github-setup)
3. [Understanding Virtual Environments (.venv)](#-understanding-virtual-environments-venv)
4. [Virtual Environments: Mac vs AWS](#-virtual-environments-mac-vs-aws)
5. [AWS Instance Connection](#aws-instance-connection)
6. [First-Time Setup on AWS](#first-time-setup-on-aws)
7. [Running Training](#running-training)
8. [Running Evaluation](#running-evaluation)
9. [Managing Multiple Experiments](#managing-multiple-experiments)
10. [Downloading Results](#downloading-results)
11. [Troubleshooting](#troubleshooting)

---

## 📁 Project Structure Overview

```
grasp-prediction/
├── 📄 Core Scripts
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluation script
│   ├── utils.py              # Helper functions
│   ├── check_dataset.py      # Verify dataset
│   └── filter_dataset.py     # Create dataset subset
│
├── 📦 model/                 # Neural network code
│   ├── net.py                # PointNet architecture
│   └── data_loader.py        # Dataset loading
│
├── 🧪 experiments/           # Training runs (each is independent)
│   ├── base_model/           # Example experiment
│   │   ├── params.json       # Hyperparameters ⭐
│   │   ├── best.pth.tar      # Best model weights ⭐
│   │   ├── training_curves.png  # Training plots ⭐
│   │   └── train.log         # Training logs
│   └── my_experiment/        # You can create multiple!
│
├── 💾 data/                  # Full ACRONYM dataset (~53 GB)
│   ├── grasps/               # 8,836 .h5 files
│   └── meshes/               # 8,872 .obj files
│
├── 💾 data_filtered/         # Subset for testing (smaller)
│   ├── grasps/               # 3 .h5 files
│   └── meshes/               # 2 .obj files
│
├── 📚 Documentation
│   ├── README.md
│   ├── DATASET.md
│   └── AWS_DEPLOYMENT_GUIDE.md  # This file!
│
└── ⚙️ Config
    ├── requirements.txt      # Python dependencies
    ├── .gitignore           # Git ignore rules
    └── .venv/               # Virtual env (DON'T upload to AWS!)
```

### What Each Experiment Folder Contains:
```
experiments/my_experiment/
├── params.json               # Hyperparameters (learning rate, batch size, etc.)
├── best.pth.tar             # Best model (highest validation AP)
├── last.pth.tar             # Most recent model
├── train.log                # Text log of training
├── training_history.json    # Metrics for each epoch
├── training_curves.png      # Loss & Accuracy plots
├── metrics_curves.png       # ROC-AUC & AP plots
├── metrics_val_best.json    # Best validation metrics
├── roc_curve.png           # ROC curve (after evaluation)
└── pr_curve.png            # Precision-Recall curve (after evaluation)
```

---

## 🐙 GitHub Setup

### What to Commit to GitHub

**✅ DO Commit:**
- All `.py` files
- `model/` directory
- `requirements.txt`
- `README.md`, `DATASET.md`
- `experiments/*/params.json` (config files only!)
- `.gitignore`

**❌ DON'T Commit:**
- `.venv/` (virtual environment)
- `data/` or `data_filtered/` (too large)
- `__pycache__/` (auto-generated)
- `*.pth.tar` (model weights)
- `*.png`, `*.log` (can regenerate)
- `tensorboard/` directories

### Your `.gitignore` File

Make sure your `.gitignore` includes:
```
.venv/
__pycache__/
*.pyc
data/
data_filtered/
*.pth.tar
*.png
*.log
tensorboard/
```

### Push to GitHub
```bash
cd "/Users/joanamizrahi/.../grasp-prediction"
git add *.py model/ requirements.txt README.md
git commit -m "Add training pipeline"
git push origin main
```

---

## 🐍 Understanding Virtual Environments (.venv)

### What is a Virtual Environment?

A **virtual environment** (.venv) is an isolated Python environment that keeps your project's dependencies separate from your system Python. Think of it as a separate folder containing:
- A specific Python version
- All packages installed with `pip install` (torch, numpy, etc.)
- Package versions specific to your project

### Why Use Virtual Environments?

1. **Isolation**: Different projects can use different package versions
2. **Clean installs**: No conflicts with system packages
3. **Reproducibility**: Same environment on Mac, AWS, etc.
4. **Easy cleanup**: Delete the folder and start fresh

### How to Know if venv is Active

When activated, you'll see `(.venv)` at the start of your terminal prompt:

```bash
# NOT activated:
joanamizrahi@mac grasp-prediction $

# Activated:
(.venv) joanamizrahi@mac grasp-prediction $
```

---

## 🖥️ Virtual Environments: Mac vs AWS

### Important: Create Separate venvs on Each Machine

**DO NOT upload your Mac's .venv to AWS!** Here's why:
- Mac and AWS use different operating systems (macOS vs Linux)
- Python packages contain OS-specific compiled code
- Your Mac's .venv is **~1-2 GB** - too large to upload
- It's easier to recreate from `requirements.txt`

### Workflow Overview

**On Mac (Development):**
```bash
# 1. Create venv (ONCE, first time only)
cd "/Users/joanamizrahi/.../grasp-prediction"
python3 -m venv .venv

# 2. Activate it (EVERY TIME you open a new terminal)
source .venv/bin/activate

# 3. Install packages (ONCE, or when requirements.txt changes)
pip install -r requirements.txt

# 4. Work on your code...
# Edit train.py, model/net.py, etc.

# 5. Test locally (optional)
python train.py --data_dir data_filtered --model_dir experiments/test

# 6. Push changes to GitHub
git add train.py model/ requirements.txt
git commit -m "Update training code"
git push origin main

# 7. Deactivate when done (optional)
deactivate
```

**On AWS (Training):**
```bash
# 1. SSH into AWS
ssh -i ~/.ssh/cs230-final-key.pem ec2-user@<ip>

# 2. FIRST TIME: Clone repo and create NEW venv
cd ~
git clone <your-repo-url> grasp-prediction
cd grasp-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. SUBSEQUENT TIMES: Pull changes and activate existing venv
cd ~/grasp-prediction
git pull origin main  # Get your latest code changes
source .venv/bin/activate  # Activate existing venv

# 4. Run training
python train.py --data_dir data --model_dir experiments/run1
```

### Step-by-Step: What Happens When

| Step | Mac | AWS |
|------|-----|-----|
| **First time setup** | `python3 -m venv .venv` creates folder | `python3 -m venv .venv` creates NEW folder |
| **Install packages** | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| **Every terminal session** | `source .venv/bin/activate` | `source .venv/bin/activate` |
| **Update code** | Edit files → `git push` | `git pull` to get changes |
| **Run Python** | `python train.py` (uses .venv Python) | `python train.py` (uses .venv Python) |

### Common Virtual Environment Commands

```bash
# Create venv (do once per machine)
python3 -m venv .venv

# Activate venv (do every terminal session)
source .venv/bin/activate

# Check which Python you're using (should show .venv path)
which python
# Output: /Users/.../grasp-prediction/.venv/bin/python

# Install packages
pip install -r requirements.txt

# Check installed packages
pip list

# Deactivate venv (optional, can just close terminal)
deactivate
```

### Troubleshooting Virtual Environments

**"No module named 'torch'" error**
- **Cause**: venv not activated
- **Solution**: `source .venv/bin/activate` then try again

**"Command not found: python"**
- **Cause**: Wrong Python command
- **Solution**: Use `python3` instead of `python` on Mac/Linux

**How to start fresh with venv**
```bash
# Remove old venv
rm -rf .venv

# Create new one
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Accidentally uploaded .venv to GitHub?**
- Make sure `.gitignore` contains `.venv/`
- Remove from git: `git rm -r --cached .venv`
- Commit: `git commit -m "Remove .venv from tracking"`

---

## 🔌 AWS Instance Connection

### Your Connection Details

**Private Key**: `~/.ssh/cs230-final-key.pem`
**Username**: `ec2-user` (NOT `root`!)
**Instance IP**: Your AWS instance public IP (changes if you stop/start)

### Connect to AWS

```bash
# Make sure key has correct permissions (do this once)
chmod 400 ~/.ssh/cs230-final-key.pem

# Connect to instance
ssh -i ~/.ssh/cs230-final-key.pem ec2-user@<your-instance-ip>

# Example:
ssh -i ~/.ssh/cs230-final-key.pem ec2-user@35.167.65.187
```

**To find your instance IP:**
1. Go to AWS Console → EC2 → Instances
2. Click your instance
3. Copy "Public IPv4 address" or "Public IPv4 DNS"

### Get Root Privileges (if needed)
```bash
sudo -i
```

---

## 🚀 First-Time Setup on AWS

### Step 1: Clone Your Code

```bash
# After SSH into AWS
cd ~
git clone <your-github-repo-url> grasp-prediction
cd grasp-prediction
```

### Step 2: Create Virtual Environment

```bash
# Create venv
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# You should see (.venv) in your prompt
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs: `torch`, `trimesh`, `h5py`, `scikit-learn`, `matplotlib`, etc.

### Step 4: Upload Data

**Option A: Upload from your Mac (for small dataset)**
```bash
# On your Mac (open new terminal, don't close AWS session)
cd "/Users/joanamizrahi/.../grasp-prediction"

# Create tarball
tar -czf data-filtered.tar.gz data_filtered/

# Upload to AWS
scp -i ~/.ssh/cs230-final-key.pem data-filtered.tar.gz ec2-user@<aws-ip>:~/grasp-prediction/

# Back on AWS terminal:
cd ~/grasp-prediction
tar -xzf data-filtered.tar.gz
rm data-filtered.tar.gz  # Clean up
```

**Option B: Download Full Dataset on AWS (Recommended)**

The ACRONYM dataset requires **two components**:
1. **Grasp annotations** (8,836 .h5 files, ~1.6 GB) - from ACRONYM dataset
2. **3D meshes** (8,872 .obj files, ~51 GB) - from ShapeNetSem dataset

**⚠️ Important**: For detailed ShapeNet download instructions, see [DATASET.md](DATASET.md). Below is a quick overview.

```bash
# On AWS terminal
cd ~/grasp-prediction
source .venv/bin/activate  # MUST activate venv first!

# Create data directory structure
mkdir -p data/grasps
mkdir -p data/meshes
```

**Part 1: Download ACRONYM Grasp Annotations**
```bash
# Install gdown (needs venv activated!)
pip install gdown

# Download from Google Drive
gdown https://drive.google.com/uc?id=1OjykLD9YmnFdfYpH2qO8yBo-I-22vKwu -O acronym.tar.gz

# Extract
tar -xzf acronym.tar.gz

# Move .h5 files to data/grasps/
# (Adjust path based on extracted folder structure)
mv acronym/grasps/*.h5 data/grasps/
rm acronym.tar.gz
```

**Part 2: Download ShapeNetSem Meshes**

ShapeNet requires registration:

1. **Register at ShapeNet**: Go to https://shapenet.org/
2. **Request ShapeNetSem**: Sign up and accept terms
3. **Download**: You'll receive a download link via email (~51 GB)
4. **Transfer to AWS**:

```bash
# Option A: Download on Mac, then upload to AWS
# (On Mac terminal)
scp -i ~/.ssh/cs230-final-key.pem ShapeNetSem.v0.zip ec2-user@<aws-ip>:~/grasp-prediction/

# (On AWS terminal)
cd ~/grasp-prediction
unzip ShapeNetSem.v0.zip

# Option B: Download directly on AWS (if you have the link)
wget "<shapenet-download-link>" -O shapenet.zip
unzip shapenet.zip
```

**Part 3: Organize Meshes into Correct Structure**

The extracted ShapeNet has category folders with numeric IDs. You need to move the `.obj` files:

```bash
# After extracting ShapeNet, you'll have folders like:
# ShapeNetSem/models/02691156/ (airplane category)
# ShapeNetSem/models/02773838/ (bag category)
# etc.

# Move these category folders to data/meshes/
mv ShapeNetSem/models/* data/meshes/

# Or copy to preserve original:
cp -r ShapeNetSem/models/* data/meshes/
```

**Final directory structure:**
```
data/
├── grasps/
│   ├── Bottle_12345.h5
│   ├── Mug_67890.h5
│   └── ... (8,836 .h5 files)
└── meshes/
    ├── 02691156/        # Airplane category
    │   ├── 1a2b3c.obj
    │   └── ...
    ├── 02773838/        # Bag category
    │   ├── 4d5e6f.obj
    │   └── ...
    └── ... (262 category folders)
```

**Why do I need to activate venv?**

When you run `source .venv/bin/activate`, you're telling your terminal:
- Use Python from `.venv/bin/python` (not system Python)
- Use packages installed in `.venv/lib/` (torch, trimesh, h5py, etc.)
- Use `pip` that installs to `.venv/` (not system)

**Without activating venv:**
```bash
pip install gdown        # ❌ Installs to system Python
python check_dataset.py  # ❌ May fail - "No module named 'h5py'"
```

**With venv activated:**
```bash
source .venv/bin/activate
pip install gdown        # ✅ Installs to .venv
python check_dataset.py  # ✅ Works - uses packages from .venv
```

**Visual indicator**: You'll see `(.venv)` at the start of your prompt:
```
(.venv) [ec2-user@ip-172-31-45-123 grasp-prediction]$
```

**Rule of thumb**: ALWAYS activate venv before:
- Running Python scripts
- Installing packages with pip
- Running any command that uses Python packages

### Step 5: Verify Setup

```bash
# Check dataset
python check_dataset.py --data_dir data_filtered
# or
python check_dataset.py --data_dir data

# You should see:
# ✓ Found X grasp files
# ✓ Found Y mesh files
# ✓ Mesh paths are consistent
```

---

## 🎯 Running Training

### Basic Training

```bash
# Activate venv (if not already)
source .venv/bin/activate

# Run training
python train.py --data_dir data_filtered --model_dir experiments/test_run
```

### Background Training (Recommended for Long Runs)

If training takes hours, run in background so you can disconnect:

```bash
# Run in background with logging
nohup python train.py --data_dir data_filtered --model_dir experiments/test_run > train_output.log 2>&1 &

# Check it's running
jobs

# View live progress
tail -f train_output.log

# Or view experiment log
tail -f experiments/test_run/train.log

# Stop watching (doesn't stop training)
# Press Ctrl+C
```

### Monitor Training

```bash
# See last 50 lines of log
tail -50 experiments/test_run/train.log

# Watch training live
tail -f experiments/test_run/train.log

# Check if training is still running
ps aux | grep train.py
```

### Training Output

After training completes, you'll find in `experiments/test_run/`:
- `best.pth.tar` - Best model weights
- `training_curves.png` - Loss & Accuracy plots
- `metrics_curves.png` - ROC-AUC & AP plots
- `train.log` - Full training log

---

## 📊 Running Evaluation

### Evaluate on Test Set

```bash
# After training completes
python evaluate.py --data_dir data_filtered --model_dir experiments/test_run --restore_file best
```

### Evaluation Output

This creates in `experiments/test_run/`:
- `roc_curve.png` - ROC curve
- `pr_curve.png` - Precision-Recall curve
- `metrics_test_best.json` - Test metrics

### View Metrics

```bash
# View test metrics
cat experiments/test_run/metrics_test_best.json

# Example output:
# {
#   "loss": 1.559,
#   "accuracy": 0.754,
#   "roc_auc": 0.645,
#   "avg_precision": 0.841
# }
```

---

## 🔬 Managing Multiple Experiments

### Create New Experiment

Each experiment is a separate folder. Just create a new `params.json`:

```bash
# Create new experiment directory
mkdir -p experiments/high_learning_rate

# Copy and modify params
cp experiments/test_run/params.json experiments/high_learning_rate/
nano experiments/high_learning_rate/params.json

# Modify learning_rate, batch_size, etc.
# Save and exit (Ctrl+X, Y, Enter)

# Run training with new config
python train.py --data_dir data_filtered --model_dir experiments/high_learning_rate
```

### Compare Experiments

```bash
# Compare validation metrics across experiments
cat experiments/test_run/metrics_val_best.json
cat experiments/high_learning_rate/metrics_val_best.json

# List all experiments
ls -lh experiments/

# See training curves
# (Download PNGs to your Mac to view)
```

### Experiment Naming Conventions

Use descriptive names:
- `baseline_pointnet` - Your baseline
- `lr_0.01` - High learning rate test
- `batch_256` - Large batch size
- `2048_points` - Different point density
- `split_by_object` - Different data split
- `final_model` - Your best model

---

## 💾 Downloading Results

### Download Specific Files

```bash
# On your Mac (new terminal)
# Download a single file
scp -i ~/.ssh/cs230-final-key.pem ec2-user@<aws-ip>:~/grasp-prediction/experiments/test_run/training_curves.png ~/Downloads/

# Download best model
scp -i ~/.ssh/cs230-final-key.pem ec2-user@<aws-ip>:~/grasp-prediction/experiments/test_run/best.pth.tar ~/Downloads/
```

### Download Entire Experiment

```bash
# On your Mac
# Download entire experiment folder
scp -i ~/.ssh/cs230-final-key.pem -r ec2-user@<aws-ip>:~/grasp-prediction/experiments/test_run ~/Downloads/

# Now you have locally:
# ~/Downloads/test_run/
#   ├── training_curves.png
#   ├── roc_curve.png
#   ├── best.pth.tar
#   └── ...
```

### Download All Experiments

```bash
# Download all experiments at once
scp -i ~/.ssh/cs230-final-key.pem -r ec2-user@<aws-ip>:~/grasp-prediction/experiments ~/Downloads/
```

---

## 🔧 Common Tasks

### Check GPU Usage

```bash
# See if GPU is being used
nvidia-smi

# Watch GPU usage live
watch -n 1 nvidia-smi
```

### Stop Training

```bash
# Find process ID
ps aux | grep train.py

# Kill process (replace <PID> with actual process ID)
kill <PID>

# Or force kill if stuck
kill -9 <PID>
```

### Resume Training

If training stopped, resume from last checkpoint:

```bash
python train.py --data_dir data_filtered --model_dir experiments/test_run --restore_file last
```

### Clean Up Old Experiments

```bash
# Delete experiment you don't need
rm -rf experiments/old_test

# Keep only essential files (delete big files)
cd experiments/test_run
rm tensorboard/*  # Delete TensorBoard logs
```

---

## 🚨 Troubleshooting

### "No module named 'torch'"

**Problem**: Virtual environment not activated

**Solution**:
```bash
source .venv/bin/activate
```

### "CUDA out of memory"

**Problem**: Batch size too large for GPU

**Solution**: Reduce `batch_size` in `params.json`:
```json
{
  "batch_size": 32  // Try 32, 16, or 8
}
```

### "Mesh not found" errors

**Problem**: Grasp files reference meshes you don't have

**Solution**: Use filtered dataset:
```bash
python filter_dataset.py
python train.py --data_dir data_filtered --model_dir experiments/test_run
```

### Training is very slow

**Possible causes**:
1. **No GPU**: Check with `nvidia-smi`, instance may not have GPU
2. **Too many points**: Reduce `num_points` in params.json
3. **Too many workers**: Reduce `num_workers` in params.json

### Can't connect to AWS

**Problem**: Instance stopped or IP changed

**Solution**:
1. Go to AWS Console
2. Check instance is "Running"
3. Get new public IP if instance was stopped
4. Update your SSH command

### "Permission denied" for SSH

**Problem**: Wrong key permissions

**Solution**:
```bash
chmod 400 ~/.ssh/cs230-final-key.pem
```

---

## 📖 Quick Reference

### Essential Commands

```bash
# Connect to AWS
ssh -i ~/.ssh/cs230-final-key.pem ec2-user@<ip>

# Activate venv
source .venv/bin/activate

# Train
python train.py --data_dir data --model_dir experiments/my_exp

# Train in background
nohup python train.py --data_dir data --model_dir experiments/my_exp > train.log 2>&1 &

# Evaluate
python evaluate.py --data_dir data --model_dir experiments/my_exp --restore_file best

# Monitor training
tail -f experiments/my_exp/train.log

# Download results
scp -i ~/.ssh/cs230-final-key.pem -r ec2-user@<ip>:~/grasp-prediction/experiments/my_exp ~/Downloads/
```

### File Locations

- **Code**: `~/grasp-prediction/`
- **Data**: `~/grasp-prediction/data/` or `data_filtered/`
- **Experiments**: `~/grasp-prediction/experiments/`
- **Logs**: `experiments/<name>/train.log`
- **Models**: `experiments/<name>/best.pth.tar`
- **Plots**: `experiments/<name>/*.png`

---

## 🎯 Typical Workflow

### First Time Setup (One-Time Only)

**On Mac:**
1. Create virtual environment: `python3 -m venv .venv`
2. Activate it: `source .venv/bin/activate`
3. Install packages: `pip install -r requirements.txt`
4. Push code to GitHub: `git push origin main`

**On AWS:**
1. SSH into AWS
2. Clone code from GitHub: `git clone <repo-url> grasp-prediction`
3. Create NEW venv: `python3 -m venv .venv`
4. Activate it: `source .venv/bin/activate`
5. Install packages: `pip install -r requirements.txt`
6. Upload or download data
7. Verify with `check_dataset.py`

### Every Time You Make Code Changes

**On Mac (Development):**
1. Activate venv: `source .venv/bin/activate`
2. Edit your code (train.py, model/net.py, etc.)
3. Test locally (optional): `python train.py --data_dir data_filtered ...`
4. Push to GitHub: `git add ... && git commit -m "..." && git push`

**On AWS (Training):**
1. SSH into AWS
2. Navigate to project: `cd ~/grasp-prediction`
3. Pull latest changes: `git pull origin main`
4. Activate venv: `source .venv/bin/activate`
5. Create experiment: `mkdir -p experiments/my_exp && cp experiments/test_pipeline/params.json experiments/my_exp/`
6. Edit config if needed: `nano experiments/my_exp/params.json`
7. Run training: `nohup python train.py --data_dir data --model_dir experiments/my_exp > train.log 2>&1 &`
8. Monitor: `tail -f experiments/my_exp/train.log`
9. Evaluate: `python evaluate.py --data_dir data --model_dir experiments/my_exp --restore_file best`
10. Download results: `scp -i ~/.ssh/cs230-final-key.pem -r ec2-user@<ip>:~/grasp-prediction/experiments/my_exp ~/Downloads/`

### When Experimenting with Hyperparameters

1. Create new experiment folder: `mkdir experiments/new_exp`
2. Copy and modify `params.json` (change ONE thing at a time!)
3. Run training with new config
4. Compare metrics with previous experiments
5. Keep best model, delete failed experiments

---

## 📬 Contact & Help

If you get stuck:
1. Check this guide first
2. Look at `train.log` for error messages
3. Try `python check_dataset.py` to verify data
4. Check AWS console that instance is running
5. Google the error message

**Good luck with your training!** 🚀
