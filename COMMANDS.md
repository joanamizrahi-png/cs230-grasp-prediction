# Quick Command Reference

## Connect to AWS

```bash
ssh -i ~/.ssh/cs230-final-key.pem ec2-user@<your-aws-ip>
```

## Training

```bash
# Activate environment
source .venv/bin/activate

# Run training
python train.py --data_dir data --model_dir experiments/my_model

# Run in background
nohup python train.py --data_dir data --model_dir experiments/my_model > train.log 2>&1 &

# Monitor
tail -f experiments/my_model/train.log
```

## Evaluation

```bash
python evaluate.py --data_dir data --model_dir experiments/my_model --restore_file best
```

## Download Results

From local machine:

```bash
scp -i ~/.ssh/cs230-final-key.pem -r ec2-user@<ip>:~/grasp-prediction/experiments/my_model ~/Downloads/
```

## Configuration

Edit `experiments/my_model/params.json`:

```json
{
    "learning_rate": 0.001,
    "batch_size": 128,
    "num_epochs": 50,
    "num_points": 2048,
    "num_workers": 4,
    "split_by": "object"
}
```

Key parameters:
- `split_by`: "object" for different objects in train/val/test, "grasp" for split grasps per object
- `num_points`: points sampled per mesh (512, 1024, or 2048)
- `batch_size`: reduce if out of memory

## Workflow

1. Develop code locally
2. Push to GitHub: `git push`
3. On AWS: `git pull`
4. Activate venv: `source .venv/bin/activate`
5. Run training
6. Download results
