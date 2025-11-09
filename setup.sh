#!/bin/bash
# Quick setup script for getting started with sample data

echo "Setting up Grasp Prediction Project..."
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download sample data
echo "Downloading ACRONYM sample data..."
if [ ! -d "data/grasps" ]; then
    git clone https://github.com/NVlabs/acronym.git temp_acronym
    mkdir -p data
    cp -r temp_acronym/data/examples/* data/
    rm -rf temp_acronym
    echo "Sample data downloaded to data/"
else
    echo "Data directory already exists, skipping download."
fi

echo ""
echo "Setup complete!"
echo ""
echo "To train the model:"
echo "  source .venv/bin/activate"
echo "  python train.py"
echo ""
echo "To evaluate:"
echo "  python evaluate.py"
echo ""
echo "For full dataset, see README.md"
