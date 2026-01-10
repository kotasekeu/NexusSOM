# SOM Quality Analyzer

A deep learning system that uses Convolutional Neural Networks (CNN) to evaluate the quality of Self-Organizing Map (SOM) visualizations. The system predicts a quality score (0-1) for SOM map images based on learned patterns from training data.

## Project Overview

This proof-of-concept project demonstrates how deep learning can be applied to automatically assess SOM visualization quality. The CNN is trained on SOM map images (e.g., U-Matrix visualizations) paired with quality scores derived from quantitative metrics (Quantization Error, Topographic Error, and Inactive Neuron Ratio).

### Key Features

- **CNN-based Regression**: Predicts continuous quality scores (0-1 range)
- **Automated Data Pipeline**: Scripts for data preparation, training, and prediction
- **Python Virtual Environment**: Clean, isolated dependency management
- **Model Variants**: Standard and lightweight CNN architectures
- **Comprehensive Logging**: TensorBoard integration and CSV logs
- **Sample Data Included**: Test the pipeline immediately with provided examples

## Project Structure

```
MapAnalyser/
├── data/
│   ├── raw_maps/              # Input: PNG images of SOM maps
│   │   ├── a506531841bdb53cd900ffd66c6987bf.png  # Sample image 1
│   │   └── b607642952cec64de011ffe77d7a98cg.png  # Sample image 2
│   ├── results.csv            # Input: SOM metrics file (sample included)
│   └── processed/
│       └── dataset.csv        # Generated: Processed dataset for training
├── src/
│   ├── validate_data.py       # Data validation script
│   ├── prepare_data.py        # Data preparation & quality calculation
│   ├── model.py               # CNN model definitions
│   ├── train.py               # Training script
│   ├── predict.py             # Prediction script
│   └── evaluate.py            # Model evaluation & visualization
├── models/                    # Saved trained models
├── logs/                      # Training logs and TensorBoard data
├── docs/                      # Additional documentation
├── requirements.txt           # Python dependencies
├── run.sh                     # Convenience script
└── README.md                  # This file
```

## Prerequisites

### System Requirements

- **Python 3.9 - 3.12** (TensorFlow limitation)
- **pip** (Python package manager)
- **5+ GB free disk space** (for models and logs)

### Python Version Check

```bash
python3 --version  # Should show 3.9.x - 3.12.x
```

**⚠️ Important**: TensorFlow does not yet support Python 3.13+. If you have Python 3.13 or newer, you'll need to install Python 3.12. See [docs/SETUP.md](docs/SETUP.md) for instructions.

### Your Data

- **`data/raw_maps/`**: Directory containing PNG images of SOM maps (filenames should be UIDs like `a506531841bdb53cd900ffd66c6987bf.png`)
- **`data/results.csv`**: CSV file with columns: `uid`, `best_mqe`, `topographic_error`, `inactive_neuron_ratio`

**Note**: Sample data is included so you can test the pipeline immediately!

## Quick Start Guide

### Step 1: Setup Python Environment

```bash
# Option A: Using the convenience script (recommended)
./run.sh setup

# Option B: Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The setup command will:
- Create a virtual environment (`venv/`)
- Install TensorFlow, Keras, and all dependencies
- Verify the installation

### Step 2: Prepare Your Data

The project includes sample data for testing. To use your own data:

```bash
# Remove sample images (optional)
rm data/raw_maps/*.png

# Copy your SOM map images
cp /path/to/your/maps/*.png data/raw_maps/

# Copy your metrics file
cp /path/to/your/results.csv data/
```

Verify your `results.csv` has the required columns:
- `uid`: Unique identifier matching the image filenames (without .png extension)
- `best_mqe`: Mean Quantization Error (lower is better)
- `topographic_error`: Topographic Error (lower is better)
- `inactive_neuron_ratio`: Ratio of inactive neurons (lower is better)

### Step 3: Validate Your Data (Optional)

Check if your data is properly formatted:

```bash
./run.sh validate

# Or manually:
source venv/bin/activate
python src/validate_data.py
```

This will check for:
- Missing required columns
- Corrupt image files
- Mismatched UIDs between CSV and images
- Data quality issues

### Step 4: Prepare the Dataset

Calculate quality scores and create the training dataset:

```bash
./run.sh prepare

# Or manually:
source venv/bin/activate
python src/prepare_data.py
```

This script will:
- Load the `results.csv` file
- Normalize the metrics (MQE, TE, inactive ratio)
- Calculate quality scores using the formula:
  ```
  score = 0.5 × (1 - norm_mqe) + 0.3 × (1 - norm_te) + 0.2 × (1 - norm_inactive)
  ```
- Verify that image files exist for each UID
- Generate `data/processed/dataset.csv` with `filepath` and `quality_score` columns

### Step 5: Train the Model

Train the CNN model on your prepared dataset:

```bash
# Train with default settings (standard model, 50 epochs)
./run.sh train

# Or train the lightweight model (faster)
./run.sh train-lite

# Manual with custom parameters:
source venv/bin/activate
python src/train.py --model standard --epochs 50 --batch-size 32
```

#### Training Options:

- `--model`: Choose architecture (`standard` or `lite`)
  - `standard`: Full CNN with 4 convolutional blocks (~5M parameters)
  - `lite`: Lightweight CNN for faster training (~500K parameters)
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)
- `--image-size`: Input image size in pixels (default: 224)
- `--learning-rate`: Learning rate for Adam optimizer (default: 0.001)

**Example with custom parameters:**
```bash
python src/train.py --model lite --epochs 100 --batch-size 16 --image-size 224
```

The training script will:
- Split data into train/validation/test sets (70%/12.75%/17.25%)
- Apply data augmentation to training images
- Save the best model based on validation loss
- Log metrics to TensorBoard
- Evaluate the model on the test set

### Step 6: Monitor Training (Optional)

View training progress in real-time with TensorBoard:

```bash
./run.sh tensorboard

# Or manually:
source venv/bin/activate
tensorboard --logdir=logs/som_quality_standard_TIMESTAMP --port=6006
```

Then open your browser to `http://localhost:6006`

### Step 7: Make Predictions

Use the trained model to predict quality scores for new SOM maps:

#### Single Image Prediction

```bash
./run.sh predict data/raw_maps/example.png

# Or manually:
source venv/bin/activate
python src/predict.py \
    --model models/som_quality_standard_TIMESTAMP_best.keras \
    --image data/raw_maps/example.png
```

Output:
```
================================================================================
PREDICTED QUALITY SCORE: 0.847532
================================================================================
Quality Level: Excellent
================================================================================
```

#### Batch Prediction on Directory

```bash
source venv/bin/activate
python src/predict.py \
    --model models/som_quality_standard_TIMESTAMP_best.keras \
    --image-dir data/raw_maps/ \
    --output predictions.csv
```

This will:
- Process all images in the directory
- Generate a CSV file with predictions
- Display summary statistics and top/bottom quality maps

### Step 8: Evaluate Model Performance

Generate detailed evaluation metrics and visualizations:

```bash
./run.sh evaluate

# Or manually:
source venv/bin/activate
python src/evaluate.py \
    --model models/som_quality_standard_TIMESTAMP_best.keras \
    --test-csv models/som_quality_standard_TIMESTAMP_test_set.csv
```

This creates:
- `evaluation_results/metrics.txt` - Performance metrics (MSE, MAE, R², RMSE)
- `evaluation_results/predictions_vs_truth.png` - Scatter plot
- `evaluation_results/error_distribution.png` - Error histogram
- `evaluation_results/residual_plot.png` - Residual analysis
- `evaluation_results/detailed_predictions.csv` - Per-image results

## Model Architecture

### Standard Model

A deep CNN with 4 convolutional blocks:
- **Block 1**: 2×Conv(32) → MaxPool → Dropout(0.25)
- **Block 2**: 2×Conv(64) → MaxPool → Dropout(0.25)
- **Block 3**: 2×Conv(128) → MaxPool → Dropout(0.3)
- **Block 4**: 2×Conv(256) → MaxPool → Dropout(0.3)
- **Dense Layers**: GlobalAvgPool → Dense(256) → Dense(128) → Dense(1, sigmoid)
- **Regularization**: Batch normalization, L2 regularization, dropout
- **Parameters**: ~5M

### Lightweight Model

A faster alternative with 3 convolutional blocks:
- **Block 1**: Conv(32) → MaxPool → Dropout(0.25)
- **Block 2**: Conv(64) → MaxPool → Dropout(0.25)
- **Block 3**: Conv(128) → MaxPool → Dropout(0.3)
- **Dense Layers**: GlobalAvgPool → Dense(64) → Dense(1, sigmoid)
- **Parameters**: ~500K

Both models use:
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Output**: Single neuron with sigmoid activation (range 0-1)

## Quality Score Formula

Quality scores are calculated from normalized SOM metrics:

```python
# Normalize metrics to [0, 1] range
norm_mqe = (mqe - min_mqe) / (max_mqe - min_mqe)
norm_te = (te - min_te) / (max_te - min_te)
norm_inactive = (inactive - min_inactive) / (max_inactive - min_inactive)

# Calculate quality (invert since lower errors = higher quality)
quality_score = 0.5 × (1 - norm_mqe) +
                0.3 × (1 - norm_te) +
                0.2 × (1 - norm_inactive)
```

**Weights rationale**:
- 50% Mean Quantization Error (most important for data representation)
- 30% Topographic Error (topology preservation)
- 20% Inactive Neuron Ratio (map utilization)

You can modify these weights in [src/prepare_data.py](src/prepare_data.py).

## Command Reference

### Using run.sh (Convenience Script)

```bash
./run.sh setup         # Create venv and install dependencies
./run.sh validate      # Validate your data
./run.sh prepare       # Prepare dataset from results.csv
./run.sh train         # Train standard model
./run.sh train-lite    # Train lightweight model
./run.sh predict <img> # Predict quality for single image
./run.sh evaluate      # Evaluate model on test set
./run.sh tensorboard   # Launch TensorBoard
./run.sh clean         # Clean models and logs
./run.sh clean-all     # Clean everything including venv
```

### Manual Commands

```bash
# Activate environment first
source venv/bin/activate

# Data preparation
python src/validate_data.py
python src/prepare_data.py

# Training
python src/train.py --model standard --epochs 50 --batch-size 32
python src/train.py --model lite --epochs 100 --batch-size 16

# Prediction
python src/predict.py --model models/MODEL.keras --image data/raw_maps/test.png
python src/predict.py --model models/MODEL.keras --image-dir data/raw_maps/ --output predictions.csv

# Evaluation
python src/evaluate.py --model models/MODEL.keras --test-csv models/TEST_SET.csv

# TensorBoard
tensorboard --logdir=logs/som_quality_TIMESTAMP --port=6006
```

## Tips and Best Practices

### Data Preparation
- Ensure image filenames match UIDs in `results.csv`
- Use consistent image format (PNG recommended)
- Minimum dataset size: 500-1000 samples for reasonable results
- Remove corrupt or invalid images before training

### Training
- Start with the lightweight model to verify the pipeline works
- Use the standard model for better accuracy (if resources allow)
- Monitor validation loss to detect overfitting
- Early stopping will restore the best weights automatically
- Training on GPU is recommended but CPU works for small datasets

### Prediction
- Use the `_best.keras` model file (best validation performance)
- Images are automatically resized to match training size
- Batch prediction is faster than processing images one-by-one

### Expected Training Time

With a dataset of ~1000 samples:
- **Lightweight model**: ~10-30 min (CPU) / ~5-10 min (GPU)
- **Standard model**: ~30-60 min (CPU) / ~10-20 min (GPU)

*Times vary based on dataset size and hardware*

## Troubleshooting

### Issue: "No module named tensorflow"
**Solution**: Make sure virtual environment is activated:
```bash
source venv/bin/activate
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Issue: "Could not find a version that satisfies the requirement tensorflow"
**Solution**: Your Python version is too new (3.13+). Install Python 3.12:
```bash
brew install python@3.12
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

See [docs/SETUP.md](docs/SETUP.md) for detailed Python installation instructions.

### Issue: "No images found" error during data preparation
**Solution**: Verify that:
- Images are in `data/raw_maps/` directory
- Filenames match the `uid` column in `results.csv`
- Image files have `.png` extension

### Issue: Out of memory during training
**Solutions**:
- Reduce batch size: `python src/train.py --batch-size 8`
- Use lightweight model: `./run.sh train-lite`
- Reduce image size: `python src/train.py --image-size 128`

### Issue: Training loss not decreasing
**Solutions**:
- Check if dataset is properly prepared
- Verify images load correctly (no all-black images)
- Try adjusting learning rate: `--learning-rate 0.01` or `--learning-rate 0.0001`
- Ensure quality scores have variance (not all the same value)

## File Formats

### Input: `data/results.csv`
```csv
uid,best_mqe,topographic_error,inactive_neuron_ratio
a506531841bdb53cd900ffd66c6987bf,0.523,0.012,0.05
b607642952cec64de011ffe77d7a98cg,0.381,0.008,0.03
...
```

### Generated: `data/processed/dataset.csv`
```csv
filepath,quality_score
data/raw_maps/a506531841bdb53cd900ffd66c6987bf.png,0.7234
data/raw_maps/b607642952cec64de011ffe77d7a98cg.png,0.8456
...
```

### Output: `predictions.csv`
```csv
filename,filepath,predicted_quality_score
example1.png,data/raw_maps/example1.png,0.8234
example2.png,data/raw_maps/example2.png,0.6123
...
```

## Python Dependencies

All dependencies are specified in [requirements.txt](requirements.txt):

- **Deep Learning**: TensorFlow 2.14+, Keras 2.14+
- **Data Processing**: pandas, numpy, scikit-learn
- **Image Processing**: Pillow, OpenCV
- **Visualization**: matplotlib, seaborn
- **Utilities**: tqdm

## Advanced Usage

### Custom Quality Score Formula

Edit [src/prepare_data.py](src/prepare_data.py) to modify the quality score calculation:

```python
# In calculate_quality_score() function, change weights:
data['quality_score'] = (
    0.6 * (1 - data['norm_mqe']) +      # Increase MQE weight
    0.3 * (1 - data['norm_te']) +
    0.1 * (1 - data['norm_inactive'])   # Decrease inactive weight
)
```

### Fine-tuning Existing Model

Continue training from a saved model:

```python
# Load existing model
model = keras.models.load_model('models/som_quality_standard_best.keras')

# Continue training
history = model.fit(
    train_gen(),
    steps_per_epoch=len(train_gen),
    validation_data=val_gen(),
    validation_steps=len(val_gen),
    epochs=50,  # Additional epochs
    initial_epoch=50,  # Starting epoch
    callbacks=callbacks
)
```

### Using GPU Acceleration

If you have an NVIDIA GPU with CUDA support:

```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]>=2.14.0

# Verify GPU is detected
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Additional Documentation

- **[docs/SETUP.md](docs/SETUP.md)** - Detailed Python version setup guide
- **[docs/INSTALL.txt](docs/INSTALL.txt)** - Quick installation reference
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - 5-minute quick start
- **[docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md)** - Project overview

## Sample Data

This project includes sample data for immediate testing:
- `data/results.csv` - 2 sample SOM metric entries
- `data/raw_maps/*.png` - 2 sample SOM visualization images (224x224)

These samples let you test the entire pipeline without preparing your own data first. For real training, you'll need 500+ samples.

## For Your Deep Learning Course

This project demonstrates:
- ✅ CNN for regression (not classification)
- ✅ Custom data pipeline
- ✅ Image preprocessing and augmentation
- ✅ Model checkpointing and early stopping
- ✅ Learning rate scheduling
- ✅ TensorBoard integration
- ✅ Model evaluation metrics
- ✅ Batch normalization and dropout regularization
- ✅ Transfer of domain knowledge (SOM metrics → quality score)

## Citation

If you use this code for research, please cite:

```
@misc{som_quality_analyzer_2026,
  title={SOM Quality Analyzer: CNN-based Quality Assessment for Self-Organizing Maps},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourusername/MapAnalyser}}
}
```

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or issues, please open an issue on the project repository or refer to the troubleshooting section above.

---

**Ready to get started?** Run `./run.sh setup` and follow the Quick Start Guide above!
