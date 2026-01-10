# Quick Start Guide

Get your SOM Quality Analyzer up and running in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- Your SOM data ready:
  - PNG images in `data/raw_maps/`
  - `data/results.csv` with metrics

## 5-Step Quick Start

### Step 1: Build the Container

```bash
./run.sh build
# OR
docker-compose build
```

### Step 2: Prepare Your Data

Make sure your data is in place:
- `data/raw_maps/` contains your SOM map PNG images
- `data/results.csv` contains your metrics

Then run:

```bash
./run.sh prepare
# OR
docker-compose run --rm som-analyzer python src/prepare_data.py
```

### Step 3: Train the Model

```bash
./run.sh train
# OR for lightweight model
./run.sh train-lite
```

This will:
- Train for 50 epochs
- Save the best model to `models/`
- Log metrics to `logs/`

### Step 4: Make Predictions

```bash
./run.sh predict data/raw_maps/your_image.png
```

### Step 5: Evaluate Performance

```bash
./run.sh evaluate
```

This creates plots and metrics in `evaluation_results/`

## Quick Commands Reference

| Task | Command |
|------|---------|
| Build container | `./run.sh build` |
| Prepare data | `./run.sh prepare` |
| Train model | `./run.sh train` |
| Train lightweight | `./run.sh train-lite` |
| Predict single image | `./run.sh predict <path>` |
| Evaluate model | `./run.sh evaluate` |
| Open shell | `./run.sh shell` |
| View TensorBoard | `./run.sh tensorboard` |
| Clean files | `./run.sh clean` |

## Manual Docker Commands

If you prefer not to use `run.sh`:

```bash
# Prepare data
docker-compose run --rm som-analyzer python src/prepare_data.py

# Train
docker-compose run --rm som-analyzer python src/train.py --epochs 50 --batch-size 32

# Predict
docker-compose run --rm som-analyzer python src/predict.py \
    --model models/som_quality_standard_TIMESTAMP_best.keras \
    --image data/raw_maps/example.png

# Batch predict
docker-compose run --rm som-analyzer python src/predict.py \
    --model models/som_quality_standard_TIMESTAMP_best.keras \
    --image-dir data/raw_maps/ \
    --output predictions.csv

# Evaluate
docker-compose run --rm som-analyzer python src/evaluate.py \
    --model models/som_quality_standard_TIMESTAMP_best.keras \
    --test-csv models/som_quality_standard_TIMESTAMP_test_set.csv
```

## Training Options

Customize training with these flags:

```bash
docker-compose run --rm som-analyzer python src/train.py \
    --model [standard|lite] \
    --epochs 100 \
    --batch-size 16 \
    --image-size 224 \
    --learning-rate 0.001
```

## Expected Training Time

- **Lightweight model**: ~10-30 min (CPU) / ~5-10 min (GPU)
- **Standard model**: ~30-60 min (CPU) / ~10-20 min (GPU)

*Times vary based on dataset size and hardware*

## Troubleshooting

### "No images found" error
- Check that images are in `data/raw_maps/`
- Verify filenames match UIDs in `results.csv`

### Out of memory
```bash
# Reduce batch size
docker-compose run --rm som-analyzer python src/train.py --batch-size 8

# Or use lightweight model
./run.sh train-lite
```

### GPU not working
Edit `docker-compose.yml` and remove the GPU section:
```yaml
# Remove these lines:
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

## File Locations

After training, you'll find:

- **Best model**: `models/*_best.keras`
- **Training logs**: `logs/som_quality_*/`
- **Test set**: `models/*_test_set.csv`
- **Predictions**: `predictions.csv` (if using batch predict)
- **Evaluation**: `evaluation_results/`

## Next Steps

1. Check the [README.md](README.md) for detailed documentation
2. Visualize training: `./run.sh tensorboard`
3. Experiment with different architectures and hyperparameters
4. Use predictions to filter your SOM maps

## Support

For issues or questions, refer to the main [README.md](README.md) documentation.

---

**Happy training!**
