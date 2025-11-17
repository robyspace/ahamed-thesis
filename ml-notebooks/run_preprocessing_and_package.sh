#!/bin/bash

# ============================================================================
# Preprocessing and Packaging Script for Google Colab
# Hybrid Serverless-Container Thesis - Phase 3: ML Model Development
# ============================================================================

set -e  # Exit on error

echo "================================================================================"
echo "HYBRID THESIS - ML DATA PREPROCESSING AND PACKAGING"
echo "================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -d "data-output" ]; then
    echo "❌ Error: data-output directory not found!"
    echo "Please run this script from the thesis root directory:"
    echo "  cd /path/to/ahamed-thesis"
    echo "  bash ml-notebooks/run_preprocessing_and_package.sh"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed!"
    exit 1
fi

echo -e "${BLUE}📊 Step 1: Checking collected data...${NC}"
echo ""

# Count JSONL files
total_files=$(find data-output -name "*.jsonl" -type f | wc -l)
nov16_files=$(find data-output -name "*2025-11-16.jsonl" -type f | wc -l)
nov17_files=$(find data-output -name "*2025-11-17.jsonl" -type f | wc -l)

echo "Found data files:"
echo "  Total JSONL files: $total_files"
echo "  Nov 16 files: $nov16_files"
echo "  Nov 17 files: $nov17_files"
echo ""

if [ "$nov16_files" -eq 0 ] && [ "$nov17_files" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Warning: No data files from Nov 16-17 found!${NC}"
    echo "The preprocessing script is configured to use Nov 16-17 data."
    echo "Please update DATE_FILTER in 01_data_preprocessing.py if needed."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for required Python packages
echo -e "${BLUE}📦 Step 2: Checking Python dependencies...${NC}"
echo ""

python3 -c "import pandas" 2>/dev/null || {
    echo "Installing pandas..."
    pip install pandas --quiet
}

python3 -c "import numpy" 2>/dev/null || {
    echo "Installing numpy..."
    pip install numpy --quiet
}

echo -e "${GREEN}✅ Dependencies ready${NC}"
echo ""

# Run preprocessing
echo "================================================================================"
echo -e "${BLUE}🔄 Step 3: Running data preprocessing...${NC}"
echo "================================================================================"
echo ""

python3 ml-notebooks/01_data_preprocessing.py

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}❌ Preprocessing failed! Please check the error messages above.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Preprocessing complete!${NC}"
echo ""

# Check output files
echo "================================================================================"
echo -e "${BLUE}📁 Step 4: Verifying output files...${NC}"
echo "================================================================================"
echo ""

OUTPUT_DIR="ml-notebooks/processed-data"

if [ ! -f "$OUTPUT_DIR/ml_training_data.csv" ]; then
    echo -e "${YELLOW}❌ Error: ml_training_data.csv not found!${NC}"
    exit 1
fi

# Show file sizes
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.json 2>/dev/null || true
echo ""

# Get file sizes
training_size=$(du -h "$OUTPUT_DIR/ml_training_data.csv" | cut -f1)
full_size=$(du -h "$OUTPUT_DIR/full_processed_data.csv" | cut -f1)

echo "File sizes:"
echo "  ml_training_data.csv: $training_size"
echo "  full_processed_data.csv: $full_size"
echo ""

# Show summary statistics
if [ -f "$OUTPUT_DIR/preprocessing_summary.json" ]; then
    echo "Summary statistics:"
    python3 << EOF
import json
with open('$OUTPUT_DIR/preprocessing_summary.json', 'r') as f:
    summary = json.load(f)
    print(f"  Total requests: {summary.get('total_requests', 0):,}")
    print(f"  Training samples: {summary.get('total_training_samples', 0):,}")
    print(f"  Date range: {summary.get('date_range', [])}")
    print(f"  Total Lambda cost: \${summary.get('total_cost_lambda', 0):.4f}")
    print(f"  Total ECS cost: \${summary.get('total_cost_ecs', 0):.4f}")
EOF
    echo ""
fi

# Create package for Colab
echo "================================================================================"
echo -e "${BLUE}📦 Step 5: Creating package for Google Colab...${NC}"
echo "================================================================================"
echo ""

PACKAGE_DIR="ml-notebooks/colab-upload"
mkdir -p "$PACKAGE_DIR"

# Copy essential files
cp "$OUTPUT_DIR/ml_training_data.csv" "$PACKAGE_DIR/"
cp "ml-notebooks/02_ml_model_training_colab.ipynb" "$PACKAGE_DIR/"
cp "ml-notebooks/README.md" "$PACKAGE_DIR/"

echo "Package created in: $PACKAGE_DIR"
echo ""
echo "Contents:"
ls -lh "$PACKAGE_DIR"
echo ""

# Create instructions file
cat > "$PACKAGE_DIR/UPLOAD_INSTRUCTIONS.txt" << 'INSTRUCTIONS'
================================================================================
GOOGLE COLAB UPLOAD INSTRUCTIONS
================================================================================

Step 1: Open Google Colab
--------------------------
Visit: https://colab.research.google.com/

Step 2: Upload Notebook
------------------------
1. Click "File" → "Upload notebook"
2. Upload: 02_ml_model_training_colab.ipynb
3. The notebook will open automatically

Step 3: Upload Training Data
-----------------------------
When the notebook prompts you (in the "Upload Data Files" cell):
1. Click the "Choose Files" button
2. Select: ml_training_data.csv
3. Wait for upload to complete

Step 4: Run All Cells
----------------------
1. Click "Runtime" → "Run all"
2. Or run cells one by one with Shift+Enter
3. Training will take 10-30 minutes depending on data size

Step 5: Download Trained Model
-------------------------------
At the end of the notebook, download these files:
- best_model_*.pkl (or .h5 for neural network)
- scaler.pkl
- feature_columns.json
- model_metadata.json

Step 6: Upload to Your Repository
----------------------------------
Save the downloaded model files to:
  ahamed-thesis/ml-models/trained/

Expected Performance
--------------------
- Target Accuracy: 85%+
- Training samples: ~212,000
- Features: 5 (workload_type, payload_size, hour, day, weekend)
- Models: Random Forest, XGBoost, Neural Network

Troubleshooting
---------------
- If out of memory: Use Colab Pro or reduce data size
- If low accuracy: Check preprocessing_summary.json for data quality
- If errors: Read ml-notebooks/README.md for detailed help

================================================================================
Good luck with your ML training! 🚀
================================================================================
INSTRUCTIONS

echo -e "${GREEN}✅ Upload instructions created${NC}"
echo ""

# Final summary
echo "================================================================================"
echo -e "${GREEN}✅ ALL DONE! READY FOR GOOGLE COLAB${NC}"
echo "================================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. 📂 Navigate to the upload package:"
echo "   cd $PACKAGE_DIR"
echo ""
echo "2. 📤 Files to upload to Google Colab:"
echo "   • 02_ml_model_training_colab.ipynb (upload first)"
echo "   • ml_training_data.csv (upload when prompted)"
echo ""
echo "3. 📖 Read UPLOAD_INSTRUCTIONS.txt for detailed steps"
echo ""
echo "4. 🚀 Open Google Colab:"
echo "   https://colab.research.google.com/"
echo ""
echo "5. 📊 Expected training time: 10-30 minutes"
echo ""
echo "6. 🎯 Target accuracy: 85%+"
echo ""
echo "================================================================================"
echo ""

# Offer to open README
read -p "Would you like to view the detailed README now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat ml-notebooks/README.md | less
fi

echo ""
echo -e "${GREEN}Happy training! 🎉${NC}"
