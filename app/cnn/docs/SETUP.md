# Setup Guide

## Python Version Requirements

**TensorFlow requires Python 3.9 - 3.12**

Your system has Python 3.14, which is not yet supported by TensorFlow.

## Solution Options

### Option 1: Install Python 3.12 (Recommended)

```bash
# Using Homebrew (macOS)
brew install python@3.12

# Create venv with Python 3.12
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 2: Use Conda

```bash
# Create conda environment with Python 3.12
conda create -n som_analyzer python=3.12
conda activate som_analyzer
pip install -r requirements.txt
```

### Option 3: Use pyenv

```bash
# Install pyenv if not already installed
brew install pyenv

# Install Python 3.12
pyenv install 3.12.0

# Set local Python version for this project
pyenv local 3.12.0

# Create virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 4: Install Without TensorFlow (Testing Only)

If you just want to test the structure without TensorFlow:

```bash
# Install minimal dependencies
pip install pandas numpy scikit-learn pillow matplotlib seaborn tqdm
```

This lets you run the data preparation and validation scripts, but training won't work.

## Quick Setup Commands

Once you have Python 3.9-3.12:

```bash
# Automatic setup (creates venv and installs dependencies)
./run.sh setup

# Manual setup
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Verify Installation

```bash
# Activate environment
source venv/bin/activate

# Check versions
python --version        # Should be 3.9-3.12
pip list | grep tensorflow
pip list | grep keras

# Test imports
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import keras; print(f'Keras {keras.__version__}')"
```

## Running the Project

```bash
# Always activate the environment first
source venv/bin/activate

# Then run commands
python src/prepare_data.py
python src/train.py
python src/predict.py --model models/model.keras --image data/raw_maps/test.png
```

Or use the convenience script (automatically activates venv):

```bash
./run.sh validate
./run.sh prepare
./run.sh train
./run.sh predict data/raw_maps/test.png
```

## Troubleshooting

### "No module named tensorflow"
- Make sure virtual environment is activated: `source venv/bin/activate`
- Check Python version: `python --version` (must be 3.9-3.12)
- Reinstall: `pip install tensorflow>=2.14.0`

### "ERROR: Could not find a version that satisfies the requirement tensorflow"
- Your Python version is too new (3.13+)
- Install Python 3.12 using one of the options above

### GPU Support
If you have an NVIDIA GPU and want to use it:

```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]>=2.14.0
```

## Dependencies Summary

Required:
- Python 3.9 - 3.12
- TensorFlow 2.14+
- Keras 2.14+
- pandas, numpy, scikit-learn
- Pillow, OpenCV
- matplotlib, seaborn

Optional:
- CUDA & cuDNN (for GPU acceleration)
