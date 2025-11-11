# Quick Reference Card

One-page cheat sheet for running your grasp prediction model.

---

## 🐍 Virtual Environment Quick Guide

**What is .venv?**
- Isolated Python environment with your project's packages
- Create separate .venv on Mac AND AWS (don't upload Mac's .venv!)

**Setup (First Time Only):**
```bash
# On Mac OR AWS (do separately on each)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Every Terminal Session:**
```bash
source .venv/bin/activate  # See (.venv) in prompt when active
```

**Workflow:**
- **Mac**: Code → `git push`
- **AWS**: `git pull` → activate venv → train

---

## 🔌 Connect to AWS

```bash
ssh -i ~/.ssh/cs230-final-key.pem ec2-user@<your-aws-ip>
```

---

## 🚀 Run Training

```bash
# Activate venv
source .venv/bin/activate

# Quick test (see output)
python train.py --data_dir data_filtered --model_dir experiments/my_test

# Production run (background)
nohup python train.py --data_dir data --model_dir experiments/my_model > train.log 2>&1 &
```

---

## 📊 Run Evaluation

```bash
python evaluate.py --data_dir data --model_dir experiments/my_model --restore_file best
```

---

## 👀 Monitor Progress

```bash
# Watch training live
tail -f experiments/my_model/train.log

# Check if still running
ps aux | grep train.py

# View metrics
cat experiments/my_model/metrics_val_best.json
```

---

## 💾 Download Results

```bash
# On your Mac (new terminal)
scp -i ~/.ssh/cs230-final-key.pem -r ec2-user@<aws-ip>:~/grasp-prediction/experiments/my_model ~/Downloads/
```

---

## 🆕 New Experiment

```bash
# Create folder
mkdir -p experiments/new_exp

# Copy config
cp experiments/test_pipeline/params.json experiments/new_exp/

# Edit config
nano experiments/new_exp/params.json
# Change: learning_rate, batch_size, num_epochs, split_by, etc.
# Save: Ctrl+X, Y, Enter

# Run training
python train.py --data_dir data --model_dir experiments/new_exp
```

---

## 📝 What You Get After Training

```
experiments/my_model/
├── best.pth.tar           # Best model ⭐
├── training_curves.png    # Loss & Accuracy plots ⭐
├── metrics_curves.png     # ROC-AUC & AP plots ⭐
├── train.log              # Full training log
└── params.json            # Your hyperparameters
```

---

## 📈 What You Get After Evaluation

```
experiments/my_model/
├── roc_curve.png          # ROC curve ⭐
├── pr_curve.png           # Precision-Recall curve ⭐
└── metrics_test_best.json # Test metrics ⭐
```

---

## ⚙️ Key Configuration Options

Edit `experiments/my_model/params.json`:

```json
{
    "learning_rate": 0.001,    // Lower = slower but more stable
    "batch_size": 128,          // Lower if out of memory
    "num_epochs": 50,           // How long to train
    "num_points": 2048,         // Points per mesh (512, 1024, 2048)
    "num_workers": 4,           // CPU workers (2-8)
    "split_by": "object"        // "object" or "grasp"
}
```

**split_by**:
- `"object"` = Different objects in train/val/test (harder, tests generalization)
- `"grasp"` = Same objects, split their grasps (easier, tests interpolation)

---

## 🚨 Quick Fixes

### Out of memory?
```json
{"batch_size": 32}  // or 16
```

### Training too slow?
```json
{"num_points": 512, "num_workers": 2}
```

### Empty validation set?
```json
{"split_by": "grasp"}
```

### Can't find meshes?
```bash
python filter_dataset.py
python train.py --data_dir data_filtered ...
```

---

## 📱 AWS Instance IPs

Keep track of your AWS instance details:

```
Instance Name: _______________
Public IP: ___________________
Key: ~/.ssh/cs230-final-key.pem
Username: ec2-user
```

*Note: IP changes if you stop/start the instance!*

---

## 🎯 Typical Session

```bash
# 1. Connect
ssh -i ~/.ssh/cs230-final-key.pem ec2-user@<ip>

# 2. Navigate & Update
cd ~/grasp-prediction
git pull origin main  # Get latest code from Mac
source .venv/bin/activate

# 3. Train
nohup python train.py --data_dir data --model_dir experiments/run1 > log.txt 2>&1 &

# 4. Monitor
tail -f log.txt  # Ctrl+C to stop watching

# 5. Evaluate
python evaluate.py --data_dir data --model_dir experiments/run1 --restore_file best

# 6. Download (from Mac terminal)
scp -i ~/.ssh/cs230-final-key.pem -r ec2-user@<ip>:~/grasp-prediction/experiments/run1 ~/Downloads/

# 7. Disconnect
exit
```

---

---

## 📦 Setting Up Full Dataset on AWS

**Quick Overview** (see [AWS_DEPLOYMENT_GUIDE.md](AWS_DEPLOYMENT_GUIDE.md) for details):

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Download ACRONYM grasps
pip install gdown
gdown https://drive.google.com/uc?id=1OjykLD9YmnFdfYpH2qO8yBo-I-22vKwu -O acronym.tar.gz
tar -xzf acronym.tar.gz
mv grasps/*.h5 data/grasps/

# 3. Get ShapeNet meshes
# Download from HuggingFace: https://huggingface.co/datasets/ShapeNet/ShapeNetSem
# ShapeNetSem v0 (~12.2 GB)
# Extract and move: mv ShapeNetSem/models/* data/meshes/

# 4. Verify
python check_dataset.py --data_dir data
```

**Why activate venv?**
- Without it: `pip install` goes to system Python ❌
- With it: `pip install` goes to project .venv ✅
- Look for `(.venv)` in your prompt!

---

## 📚 Full Documentation

See [AWS_DEPLOYMENT_GUIDE.md](AWS_DEPLOYMENT_GUIDE.md) for detailed explanations.
See [DATASET.md](DATASET.md) for complete dataset download instructions.
