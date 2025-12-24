#!/bin/bash
# Install dependencies for RF-DETR project

set -e

echo "=========================================="
echo "Installing RF-DETR Dependencies"
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install core dependencies for verification
echo ""
echo "Installing core dependencies (pydantic, numpy, torch)..."
pip install pydantic numpy torch torchvision

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import pydantic
import numpy
import torch
print('✓ All core dependencies installed successfully')
print(f'  - pydantic: {pydantic.__version__}')
print(f'  - numpy: {numpy.__version__}')
print(f'  - torch: {torch.__version__}')
"

echo ""
echo "=========================================="
echo "✓ Dependencies installed successfully!"
echo "=========================================="
echo ""
echo "To use the virtual environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To run verification:"
echo "  python3 verify_pydantic_fixes.py"
echo ""

