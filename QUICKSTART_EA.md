# Quick Start: Running EA with Continuous Optimization

## Test the Updated EA (Small Run)

```bash
cd /Users/tomas/OSU/Python/NexusSom/app

# Quick test (50 pop × 10 gen = 500 evaluations, ~5-10 minutes)
python3 ea/ea.py -i test/iris.csv -c test/ea-iris-config-small.json
```

## Run Large-Scale Campaign (for CNN Training)

**Step 1**: Create large-scale config:
```json
// test/ea-iris-config-large.json
{
  "EA_SETTINGS": {
    "population_size": 100,
    "generations": 200
  },
  // ... rest same as ea-iris-config.json
}
```

**Step 2**: Run campaign:
```bash
python3 ea/ea.py -i test/iris.csv -c test/ea-iris-config-large.json
```

**Expected**:
- Runtime: 4-8 hours (100 × 200 = 20,000 SOM evaluations)
- Output: `test/results/YYYYMMDD_HHMMSS/`
- RGB maps: `maps_dataset/rgb/` (20,000 images)
- Results CSV: `results.csv` (20,000 rows)

## Check Results

```bash
# Check Pareto front evolution
cat test/results/YYYYMMDD_HHMMSS/pareto_front_log.txt

# Check results CSV
head -20 test/results/YYYYMMDD_HHMMSS/results.csv

# Count RGB maps
ls test/results/YYYYMMDD_HHMMSS/maps_dataset/rgb/ | wc -l

# Check parameter ranges (continuous values)
python3 -c "
import pandas as pd
df = pd.read_csv('test/results/YYYYMMDD_HHMMSS/results.csv')
print('Start LR range:', df['start_learning_rate'].min(), '-', df['start_learning_rate'].max())
print('End LR range:', df['end_learning_rate'].min(), '-', df['end_learning_rate'].max())
print('Unique values:', df['start_learning_rate'].nunique())
"
```

## Parameter Intervals Reference

| Parameter | Min | Max | Type |
|-----------|-----|-----|------|
| start_learning_rate | 0.0 | 1.0 | float |
| end_learning_rate | 0.0 | 1.0 | float |
| start_radius_init_ratio | 0.0 | 1.0 | float |
| start_batch_percent | 0.0 | 15.0 | float |
| end_batch_percent | 0.0 | 15.0 | float |
| epoch_multiplier | 0.0 | 25.0 | float |
| growth_g | 1.0 | 50.0 | float |
| num_batches | 1 | 30 | int |
| map_size | 5 | 20 | int pair |

## Troubleshooting

**Error: "Unknown parameter type"**
- Check config format matches new interval-based format
- Ensure all parameters have `"type"` field

**Error: "Bounds required for polynomial mutation"**
- Ensure all float/int parameters have `"min"` and `"max"` fields

**Slow execution**
- Reduce `population_size` and `generations`
- Use smaller dataset for testing
- Check CPU usage (should use all cores)

## Next: CNN Training

Once EA campaign completes with 5,000+ configurations:

```bash
# Prepare CNN dataset
python3 cnn/prepare_data.py --ea_run test/results/YYYYMMDD_HHMMSS/

# Start CNN training
python3 cnn/train.py --dataset cnn_dataset.csv
```
