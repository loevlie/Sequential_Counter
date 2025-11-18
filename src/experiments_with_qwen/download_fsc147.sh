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

# Method 1: Direct download from alternative sources
echo "Attempting download from alternative sources..."

# Try downloading from HuggingFace mirror (if available)
# echo "Method 1: Trying HuggingFace mirror..."
# wget -c https://huggingface.co/datasets/FSC147/resolve/main/FSC147_384_V2.zip -O FSC147_384_V2.zip

# Method 2: Using gdown with proper file ID
if [ ! -f FSC147_384_V2.zip ] || [ ! -s FSC147_384_V2.zip ]; then
    echo "Method 2: Trying gdown with correct file ID..."
    if command -v gdown &> /dev/null; then
        gdown --id 1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S -O FSC147_384_V2.zip
    else
        echo "gdown not found. Installing..."
        pip install gdown
        gdown --id 1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S -O FSC147_384_V2.zip
    fi
fi

# Method 3: Manual download instructions
if [ ! -f FSC147_384_V2.zip ] || [ ! -s FSC147_384_V2.zip ]; then
    echo ""
    echo "================================================"
    echo "MANUAL DOWNLOAD REQUIRED"
    echo "================================================"
    echo "The automatic download failed. Please download manually:"
    echo ""
    echo "1. Visit: https://drive.google.com/file/d/1ByFmTPPyqpUx9T2-vQK2X6wJ6pMky0zj/view"
    echo "2. Click 'Download' button"
    echo "3. Save the file as: $DATA_DIR/FSC147_384_V2.zip"
    echo ""
    echo "Alternative source:"
    echo "GitHub repo: https://github.com/cvlab-stonybrook/LearningToCountEverything"
    echo ""
    echo "Once downloaded, place the file here and re-run this script."
    echo "================================================"

    # Still download the annotations
    echo ""
    echo "Downloading annotations (these should work)..."
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