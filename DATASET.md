# Downloading the Full ACRONYM Dataset

This guide helps you download and organize the complete ACRONYM dataset for training on the full 8,872 objects.

## Dataset Components

1. **Grasp annotations** (1.6 GB) - Required
2. **ShapeNet meshes** (51 GB) - Required
3. **Watertight meshes** (optional) - Better for some applications

## Step 1: Download Grasp Annotations

**Note**: The original GitHub release link is no longer available. Use Google Drive instead.

```bash
# Install gdown for Google Drive downloads
pip install gdown

# Download from Google Drive
gdown https://drive.google.com/uc?id=1OjykLD9YmnFdfYpH2qO8yBo-I-22vKwu -O acronym.tar.gz

# Extract (creates grasps/ folder directly)
tar -xzf acronym.tar.gz

# Move to data directory
mkdir -p data/grasps
mv grasps/*.h5 data/grasps/

# Cleanup
rm acronym.tar.gz
rmdir grasps
```

**Alternative**: If the Google Drive link stops working, check the [ACRONYM GitHub repository](https://github.com/NVlabs/acronym) for updated download links.

## Step 2: Download ShapeNet Meshes

ShapeNet is now available on HuggingFace.

**Download from HuggingFace**

1. **Visit HuggingFace**: https://huggingface.co/datasets/ShapeNet/ShapeNetSem
2. **Download the archive**: ShapeNetSem v0 release (~12.2 GB)
3. **Extract and organize**:
   ```bash
   # Download (replace URL with actual HuggingFace download link)
   wget <huggingface-download-url> -O shapenet_sem.zip

   # Extract
   unzip shapenet_sem.zip

   # Organize meshes
   mkdir -p data/meshes

   # The extracted ShapeNetSem has category folders with numeric IDs
   # Move them to data/meshes/
   mv ShapeNetSem/models/* data/meshes/
   ```


## Step 3: Verify Dataset Structure

The `data/` directory should look like:

```
data/
├── grasps/
│   ├── Bottle_xxxxx.h5
│   ├── Mug_xxxxx.h5
│   ├── Laptop_xxxxx.h5
│   └── ... (8,872 files)
└── meshes/
    ├── Bottle/
    │   └── xxxxx.obj
    ├── Mug/
    │   └── xxxxx.obj
    └── ... (262 categories)
```

## Step 4: Verify Data Loading

Test that data loads correctly:

```bash
python -c "
from model.data_loader import GraspDataset
dataset = GraspDataset('data', num_points=512, split='train')
print(f'Dataset size: {len(dataset)}')
sample = dataset[0]
print(f'Point cloud shape: {sample[\"points\"].shape}')
print(f'Grasp shape: {sample[\"grasp\"].shape}')
print('Data loading successful!')
"
```

## Storage Requirements

- **Grasp annotations**: 1.6 GB
- **ShapeNet meshes**: ~51 GB  
- **Total**: ~53 GB


## Alternative: Download Subset

For faster experimentation, download only specific categories:

```bash
# Choose categories (example: household items)
CATEGORIES="Mug Bottle Bowl Laptop Chair Table"

# Only copy those .h5 files and meshes
for cat in $CATEGORIES; do
    cp acronym/grasps/${cat}_*.h5 data/grasps/
    cp -r ShapeNet/meshes/$cat data/meshes/
done
```

Popular categories for robotics:
- Bottles, Mugs, Bowls (kitchen)
- Laptops, Keyboards, Mice (office)
- Tools, Scissors (manipulation)
- Toys, Blocks (simple shapes)

## Processing Meshes (Advanced)

ACRONYM paper mentions creating watertight meshes for better simulation:

```bash
# Clone Manifold tool
git clone https://github.com/hjwdzh/Manifold
cd Manifold && make

# Process a mesh
./manifold input.obj temp.watertight.obj -s
./simplify -i temp.watertight.obj -o output.obj -m -r 0.02
```

This is **optional** - the original meshes work fine for our point cloud sampling.

## Troubleshooting

**"Mesh not found" errors:**
- Check that mesh paths in .h5 files match directory structure
- Use: `h5dump -d "object/file" data/grasps/Mug_xxx.h5` to see expected path

**Out of disk space:**
- Use symbolic links: `ln -s $GROUP_HOME/acronym_data data`
- Store on scratch space during training
- Use fewer categories for prototyping

**Slow data loading:**
- Increase `num_workers` in params.json
- Preprocess: save point clouds instead of meshes
- Use SSD storage if available

## Questions?

Contact the team or check:
- [ACRONYM GitHub](https://github.com/NVlabs/acronym)
- [ACRONYM Paper](https://arxiv.org/abs/2011.09584)
- [ShapeNet](https://www.shapenet.org/)
