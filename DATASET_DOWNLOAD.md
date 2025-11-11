# Dataset Download Instructions

The ACRONYM dataset consists of grasp annotations and 3D meshes from ShapeNet.

## Download Grasp Annotations

```bash
# Install download tool
pip install gdown

# Download from Google Drive
gdown https://drive.google.com/uc?id=1OjykLD9YmnFdfYpH2qO8yBo-I-22vKwu -O acronym.tar.gz

# Extract
tar -xzf acronym.tar.gz

# Move to data directory
mkdir -p data/grasps
mv grasps/*.h5 data/grasps/

# Clean up
rm acronym.tar.gz
rmdir grasps
```

## Download ShapeNet Meshes

ShapeNet requires registration. Two options:

**Option 1: HuggingFace (Recommended)**
- Visit: https://huggingface.co/datasets/ShapeNet/ShapeNetSem
- Request access (requires approval)
- Download ShapeNetSem v0 (~12.2 GB)
- Extract and organize:

```bash
mkdir -p data/meshes
unzip shapenet_sem.zip
mv ShapeNetSem/models/* data/meshes/
```

**Option 2: ShapeNet.org**
- Register at: https://www.shapenet.org/
- Request ShapeNetSem dataset
- Download when approved
- Follow same extraction steps

## Verify Dataset

```bash
python check_dataset.py --data_dir data
```

Expected structure:
```
data/
├── grasps/
│   └── *.h5 (8,836 files)
└── meshes/
    └── */ (category folders with .obj files)
```

## Dataset Statistics

- Grasp annotations: 8,836 files (~1.6 GB)
- Meshes: 12,000 models across 270 categories (~12.2 GB)
- Total grasps: ~17.7 million
- Success rate: ~15%

## Notes

- ShapeNetSem is a curated subset of ShapeNet with semantic annotations
- ACRONYM was designed to work with ShapeNetSem
- The original GitHub release link no longer works, use Google Drive instead
