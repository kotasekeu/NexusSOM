# Testing "The Eye" CNN in Google Colab

This guide explains how to test your trained SOM quality prediction model in Google Colab.

## Quick Start

### 1. Upload Notebook to Colab
- Go to https://colab.research.google.com/
- Click "Upload" and select `test_the_eye_colab.ipynb`
- Or use: File → Upload notebook

### 2. Enable GPU (Recommended)
- Runtime → Change runtime type → Hardware accelerator → GPU
- Click Save

### 3. Run the Setup Cells
Execute the first 3 cells in order:
1. **Install dependencies** - Upgrades TensorFlow to 2.18+
2. **Import libraries** - Loads required packages
3. **Verify versions** - Checks TensorFlow compatibility

**Important:** After running cell 1 (installation), you may need to restart the runtime:
- Runtime → Restart runtime
- Then re-run cells 2-3

---

## TensorFlow Version Compatibility

### Problem
The model was trained with TensorFlow 2.18+ which includes newer parameters like `quantization_config`. Older TensorFlow versions (like 2.13) in Colab can't load these models.

### Solutions

#### Option A: Upgrade TensorFlow in Colab (Recommended)
The notebook automatically upgrades TensorFlow to 2.18+. After installation:
1. Restart the runtime
2. Re-run the import cell
3. Verify TensorFlow >= 2.18

#### Option B: Export Model in Compatible Format
If you have trouble with Option A, export your model locally first:

```bash
# On your local machine (in venv_tf)
cd /Users/tomas/OSU/Python/NexusSom/app
source venv_tf/bin/activate

# Export to SavedModel format (most compatible)
python3 cnn/export_model_for_colab.py \
    --model models/som_quality_standard_20260112_210424_best.keras \
    --format savedmodel

# Creates: models/som_quality_standard_20260112_210424_best_savedmodel/
```

Then in Colab, upload the entire SavedModel directory and load with:
```python
model = keras.models.load_model('som_quality_standard_20260112_210424_best_savedmodel')
```

#### Option C: Weights Only Export
Export just architecture + weights (maximum compatibility):

```bash
python3 cnn/export_model_for_colab.py \
    --model models/som_quality_standard_20260112_210424_best.keras \
    --format weights

# Creates directory with:
# - architecture.json
# - weights.h5
# - metadata.json
# - load_model.py
```

---

## Using the Notebook

The notebook provides 3 testing modes:

### Option A: Single Image Test
1. Upload one RGB SOM image
2. Get instant prediction with visualization
3. Shows quality score, label, and confidence

**Use case:** Quick test of individual maps

### Option B: Batch Prediction (ZIP)
1. Create a ZIP file with multiple RGB images
2. Upload the ZIP
3. Get predictions for all images
4. Download results as CSV
5. View top 10 best/worst quality maps

**Use case:** Analyze results from an EA run

**How to create ZIP:**
```bash
# On your local machine
cd /Users/tomas/OSU/Python/NexusSom/app/test/results/20260112_140511/maps_dataset/rgb
zip -r rgb_maps.zip *.png

# Upload rgb_maps.zip to Colab
```

### Option C: Test Set Evaluation
1. Upload test set CSV (from training)
2. Upload ZIP with corresponding images
3. Get full evaluation metrics:
   - MAE, RMSE
   - Accuracy, Precision, Recall, F1
   - Confusion matrix
   - Prediction distribution plots

**Use case:** Validate model performance on held-out test data

**Files needed:**
- Test CSV: `models/som_quality_standard_20260112_210424_test_set.csv`
- Images: Create ZIP from filepaths in the CSV

---

## Model Performance Expectations

Based on local evaluation:

- **Accuracy:** ~99.33%
- **Precision:** ~100% (no false positives)
- **Recall:** ~95% (catches most good maps)
- **MAE:** ~0.03
- **RMSE:** ~0.08

Quality Score Interpretation:
- **0.0 - 0.3:** Very bad quality (high confidence)
- **0.3 - 0.5:** Bad quality
- **0.5 - 0.7:** Good quality
- **0.7 - 1.0:** Excellent quality (high confidence)

---

## Troubleshooting

### "TypeError: Unrecognized keyword arguments"
**Cause:** TensorFlow version too old

**Fix:**
1. Run installation cell (upgrades TensorFlow)
2. Runtime → Restart runtime
3. Re-run import cells
4. Verify TensorFlow >= 2.18

### "Model loading failed"
**Cause:** Version incompatibility persists

**Fix:**
1. Use `export_model_for_colab.py` to export in SavedModel format
2. Upload SavedModel directory to Colab
3. Load with `keras.models.load_model()`

### "Could not load image"
**Cause:** Filepath mismatch in test CSV

**Fix:**
- Ensure images are in ZIP root (not nested folders)
- Check that filenames in CSV match actual files

### "Out of memory"
**Cause:** Too many images loaded at once

**Fix:**
- Use GPU runtime (more memory)
- Process in smaller batches
- Reduce image_size parameter

---

## RGB Channel Interpretation

"The Eye" analyzes RGB composite images where:

- **Red Channel:** U-Matrix
  - Dark blue = similar neighbors (clusters)
  - Yellow = large distances (boundaries)

- **Green Channel:** Distance Map
  - Dark blue = low quantization error
  - Yellow = high quantization error

- **Blue Channel:** Dead Neurons Map
  - Dark = active neurons
  - Bright = dead neurons

### What Makes a "Good" SOM?
- Clear cluster boundaries (U-Matrix shows distinct regions)
- Low quantization error (Distance Map mostly dark)
- Few dead neurons (Dead Neurons Map mostly dark)
- Well-organized topology (smooth transitions)

---

## Advanced Usage

### Adjust Quality Threshold
Default threshold is 0.5. To use a different threshold:

```python
# In prediction cells, change:
result = predict_single_image(model, img_path, threshold=0.7)  # More strict
result = predict_single_image(model, img_path, threshold=0.3)  # More lenient
```

### Batch Process Large Datasets
For very large datasets, process in chunks:

```python
batch_size = 100
for i in range(0, len(image_files), batch_size):
    batch = image_files[i:i+batch_size]
    # Process batch
    # Save results
    # Clear memory
```

### Export Predictions with Visualizations
Save prediction visualizations:

```python
for idx, row in results_df.head(10).iterrows():
    plt.figure(figsize=(10, 8))
    img = Image.open(row['filepath'])
    plt.imshow(img)
    plt.title(f"Score: {row['quality_score']:.4f} ({row['quality_label']})")
    plt.savefig(f"prediction_{row['filename']}", dpi=150, bbox_inches='tight')
    plt.close()

# Zip and download
!zip predictions.zip prediction_*.png
files.download('predictions.zip')
```

---

## Files Overview

- **test_the_eye_colab.ipynb** - Main notebook for Colab testing
- **export_model_for_colab.py** - Convert model to compatible formats
- **COLAB_README.md** - This file
- **evaluate_model.py** - Local evaluation script (reference)

---

## Next Steps

After validating model performance in Colab:

1. **Integrate into EA** - Use "The Eye" for real-time quality prediction during evolution
2. **Generate More Training Data** - Run EA with diverse datasets to improve model
3. **Fine-tune Model** - Retrain on new labeled data for better performance
4. **Deploy** - Use model to filter/rank SOM configurations automatically

---

**Questions or Issues?**
- Check notebook cell outputs for error messages
- Verify TensorFlow version >= 2.18
- Try exporting model in different format
- Review local evaluation results for comparison
