#!/bin/bash
# Script to download and setup FSC147 dataset
# Dataset source: https://github.com/cvlab-stonybrook/LearningToCountEverything

echo "=========================================="
echo "FSC147 Dataset Download Script"
echo "=========================================="

# Create data directory
DATA_DIR="${1:-./FSC147}"
echo "Data will be downloaded to: $DATA_DIR"
mkdir -p $DATA_DIR
cd $DATA_DIR

# Download main dataset from Google Drive
echo ""
echo "Downloading FSC147 dataset (this may take a while)..."
echo "Note: This is a 3GB+ download"

# Using gdown for Google Drive downloads (install with: pip install gdown)
# Alternative: use wget with direct links if available

# Main dataset images (384x384 version)
echo "Downloading images..."
if command -v gdown &> /dev/null; then
    # FSC147 images from Google Drive
    gdown --fuzzy "https://drive.google.com/file/d/1ByFmTPPyqpUx9T2-vQK2X6wJ6pMky0zj/view?usp=sharing" -O FSC147_384_V2.zip
else
    echo "gdown not found. Installing..."
    pip install gdown
    gdown --fuzzy "https://drive.google.com/file/d/1ByFmTPPyqpUx9T2-vQK2X6wJ6pMky0zj/view?usp=sharing" -O FSC147_384_V2.zip
fi

# Alternative method using wget (if gdown fails)
if [ ! -f FSC147_384_V2.zip ]; then
    echo "Trying alternative download method..."
    # You can use this direct link approach
    wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ByFmTPPyqpUx9T2-vQK2X6wJ6pMky0zj' -O FSC147_384_V2.zip
fi

# Download annotations
echo "Downloading annotations..."
wget https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data/annotation_FSC147_384.json
wget https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data/Train_Test_Val_FSC_147.json

# Alternative: Clone the entire annotations repo (smaller)
# git clone --depth 1 https://github.com/cvlab-stonybrook/LearningToCountEverything.git temp_repo
# cp temp_repo/data/*.json .
# rm -rf temp_repo

# Extract images
echo "Extracting images..."
unzip -q FSC147_384_V2.zip

# The extracted folder might have a different name, so let's standardize it
if [ -d "FSC-147" ]; then
    mv FSC-147/images_384_VarV2 .
    rmdir FSC-147
elif [ -d "images_384_VarV2" ]; then
    echo "Images folder already in correct location"
fi

# Clean up zip file to save space
echo "Cleaning up..."
rm -f FSC147_384_V2.zip

# Verify the dataset structure
echo ""
echo "Verifying dataset structure..."
if [ -f "annotation_FSC147_384.json" ] && [ -f "Train_Test_Val_FSC_147.json" ] && [ -d "images_384_VarV2" ]; then
    echo "✓ Dataset downloaded successfully!"
    echo ""
    echo "Dataset structure:"
    echo "  $DATA_DIR/"
    echo "  ├── annotation_FSC147_384.json"
    echo "  ├── Train_Test_Val_FSC_147.json"
    echo "  └── images_384_VarV2/"
    echo "      └── [6135 images]"

    # Count images
    IMG_COUNT=$(ls images_384_VarV2/*.jpg 2>/dev/null | wc -l)
    echo ""
    echo "Total images found: $IMG_COUNT"

    # Check splits
    echo ""
    echo "Dataset splits:"
    python3 -c "
import json
data = json.load(open('Train_Test_Val_FSC_147.json'))
for split in data:
    print(f'  {split}: {len(data[split])} images')
"
else
    echo "⚠ Error: Dataset structure incomplete. Please check the download."
    echo "Missing files:"
    [ ! -f "annotation_FSC147_384.json" ] && echo "  - annotation_FSC147_384.json"
    [ ! -f "Train_Test_Val_FSC_147.json" ] && echo "  - Train_Test_Val_FSC_147.json"
    [ ! -d "images_384_VarV2" ] && echo "  - images_384_VarV2/"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "Use --data_dir $DATA_DIR when running training scripts"
echo "=========================================="