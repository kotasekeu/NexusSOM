# Fixed [0, 1.0] Normalization with Quality Threshold

**Date**: 2026-01-12
**Status**: ✅ IMPLEMENTED

---

## Overview

Implemented **fixed [0, 1.0] normalization** for U-Matrix and Distance Map visualizations with automatic quality detection. This combines visualization consistency with quality enforcement:

1. **Fixed colormap range**: All maps use [0, 1.0] normalization (dark blue = 0, yellow = 1.0)
2. **Quality threshold**: Maps with max > 1.0 are automatically penalized by EA
3. **Visual consistency**: Same colors mean same quality across ALL maps

---

## The Problem

Previously, we had two competing approaches:

### Approach 1: Auto-normalization
- Each map normalized to its own [min, max]
- **Problem**: Same colors mean different values → confusing for CNN and humans

### Approach 2: Dynamic global max
- Calculate global max across all maps, then normalize to [0, global_max]
- **Problem**: Requires two-pass processing, complex implementation

---

## The Solution: Fixed [0, 1.0] + Quality Penalty

**Key insight**: Well-organized SOMs (with normalized input data) should naturally have U-Matrix and Distance Map values in [0, 1] range. If values exceed 1.0, the SOM has poor organization.

This turns normalization from a visualization problem into a **quality metric**:

- **Good SOMs**: max ≤ 1.0 → visualize normally, no penalty
- **Bad SOMs**: max > 1.0 → visualize with clipping, apply heavy EA penalty

---

## Implementation

### 1. Track Maximum Values

**File**: [app/som/som.py:157-161](app/som/som.py#L157-L161)

Added `u_matrix_max` to metrics:

```python
return {
    'u_matrix_mean': np.mean(u_matrix),
    'u_matrix_std': np.std(u_matrix),
    'u_matrix_max': np.max(u_matrix)  # NEW
}
```

### 2. EA Quality Check and Penalty

**File**: [app/ea/ea.py:781-813](app/ea/ea.py#L781-L813)

Calculate distance map max and apply penalty if either exceeds 1.0:

```python
# Calculate distance map max for quality check
distance_map, _ = som.compute_quantization_error(data, mask=ignore_mask)
distance_map_max = np.max(distance_map) if distance_map is not None else 0.0
training_results['distance_map_max'] = distance_map_max

# Apply penalty for poor organization (U-Matrix or Distance Map max > 1.0)
# Good SOMs should have values in [0, 1] range
u_matrix_max = training_results.get('u_matrix_max', 0.0)
ORGANIZATION_THRESHOLD = 1.0
penalty_factor = 1.0

if u_matrix_max > ORGANIZATION_THRESHOLD or distance_map_max > ORGANIZATION_THRESHOLD:
    # Heavy penalty for poor organization: multiply by (1 + excess * 10)
    excess_u = max(0, u_matrix_max - ORGANIZATION_THRESHOLD)
    excess_d = max(0, distance_map_max - ORGANIZATION_THRESHOLD)
    max_excess = max(excess_u, excess_d)
    penalty_factor = 1.0 + (max_excess * 10.0)  # 10x multiplier for severe penalty
    training_results['best_mqe'] *= penalty_factor
    training_results['topographic_error'] = topographic_error * penalty_factor
    log_message(uid, f"Applied {(penalty_factor - 1.0) * 100:.1f}% penalty for poor organization (U-Matrix max: {u_matrix_max:.3f}, Distance max: {distance_map_max:.3f})", working_dir)
else:
    training_results['topographic_error'] = topographic_error

# Apply penalty for excessive dead neurons (>10%)
DEAD_NEURON_THRESHOLD = 0.10
if dead_ratio > DEAD_NEURON_THRESHOLD:
    dead_penalty = 1.0 + (dead_ratio - DEAD_NEURON_THRESHOLD)
    training_results['best_mqe'] *= dead_penalty
    training_results['topographic_error'] *= dead_penalty
    penalty_factor *= dead_penalty
    log_message(uid, f"Applied {(dead_penalty - 1.0) * 100:.1f}% penalty for {dead_ratio:.1%} dead neurons", working_dir)
```

**Penalty calculation**:
- If `u_matrix_max = 1.5`, excess = 0.5, penalty = 1 + (0.5 × 10) = 6.0 (500% penalty!)
- If `distance_max = 2.0`, excess = 1.0, penalty = 1 + (1.0 × 10) = 11.0 (1000% penalty!)
- Penalties stack with dead neuron penalties

### 3. Fixed Visualization Range

**File**: [app/som/visualization.py](app/som/visualization.py)

#### Updated `_create_map` function (lines 13-23)

Added `vmax` parameter for fixed maximum:

```python
def _create_map(som: KohonenSOM, values: np.ndarray, title: str, output_file: str,
                cmap: str, cbar_label: str = None, show_text: list = None, show_title: bool = True,
                fixed_norm: bool = False, vmax: float = None):
    """
    Universal function for rendering any SOM map (U-Matrix, Hitmap, etc.).

    Args:
        show_title: If True, display title above the map. Set to False for CNN training data.
        fixed_norm: If True, normalize to [0, vmax] for consistent visualization across all maps.
        vmax: Maximum value for fixed normalization. If None, uses values.max().
    """
```

#### Apply fixed normalization (lines 53-56)

```python
# Apply fixed normalization if requested (black=0, white=vmax)
if fixed_norm:
    max_val = vmax if vmax is not None else values.max()
    collection.set_clim(vmin=0, vmax=max_val)
```

#### Updated legend (lines 83-87)

```python
# Use fixed normalization if requested, otherwise use min/max
if fixed_norm:
    max_val = vmax if vmax is not None else values.max()
    norm = Normalize(vmin=0, vmax=max_val)
else:
    norm = Normalize(vmin=values.min(), vmax=values.max())
```

#### U-Matrix generation (lines 145-150)

```python
# Use viridis colormap with fixed [0, 1.0] normalization
# Good SOMs should have U-Matrix values in [0, 1] range
# Values > 1.0 indicate poor organization and trigger EA penalty
_create_map(som, u_matrix, "U-Matrix", output_file, cmap='viridis',
            cbar_label="Average distance to neighbors", show_title=show_title,
            fixed_norm=True, vmax=1.0)
```

#### Distance Map generation (lines 342-347)

```python
# Use viridis colormap with fixed [0, 1.0] normalization
# Good SOMs should have distance map values in [0, 1] range
# Values > 1.0 indicate poor organization and trigger EA penalty
_create_map(som, neuron_error_map, "Distance Map (Neuron QE)", output_file,
            cmap='viridis', cbar_label="Quantization Error", show_title=show_title,
            fixed_norm=True, vmax=1.0)
```

---

## How It Works

### Example 1: Good SOM
```
U-Matrix max: 0.42
Distance map max: 0.38
```

**Visualization**:
- U-Matrix: Dark blue → Green (only uses 0-0.42 of the [0, 1.0] color range)
- Distance map: Dark blue → Green (only uses 0-0.38 of the [0, 1.0] color range)
- **Visual appearance**: Mostly dark/blue colors (good!)

**EA Penalty**: None (both ≤ 1.0)

### Example 2: Moderate SOM
```
U-Matrix max: 0.85
Distance map max: 0.92
```

**Visualization**:
- U-Matrix: Dark blue → Yellow (uses most of the [0, 1.0] color range)
- Distance map: Dark blue → Yellow (uses most of the [0, 1.0] color range)
- **Visual appearance**: Full color gradient (acceptable)

**EA Penalty**: None (both ≤ 1.0)

### Example 3: Poor SOM
```
U-Matrix max: 1.6
Distance map max: 2.3
```

**Visualization**:
- U-Matrix: Dark blue → Yellow (clipped at 1.0, values > 1.0 appear as yellow)
- Distance map: Dark blue → Yellow (clipped at 1.0, values > 1.0 appear as yellow)
- **Visual appearance**: Saturated yellow (bad!)

**EA Penalty**:
- Excess = max(1.6 - 1.0, 2.3 - 1.0) = 1.3
- Penalty factor = 1.0 + (1.3 × 10.0) = 14.0
- **Result**: MQE and topographic error multiplied by 14 (1300% penalty!)

This SOM will be heavily dominated in Pareto ranking and unlikely to make the final Pareto front.

---

## Benefits

### 1. Visual Consistency
- **Same colors = same quality** across all maps
- Dark blue always means 0 (perfect)
- Yellow always means 1.0 (threshold)
- Saturated yellow means > 1.0 (poor organization)

### 2. Automatic Quality Detection
- No need for manual inspection of max values
- EA automatically penalizes poor organization
- Pareto front will naturally exclude badly organized SOMs

### 3. CNN Training
- **Consistent color-to-quality mapping** across entire dataset
- CNN learns that dark = good, yellow = bad
- No need for the CNN to learn different scales

### 4. Human Labeling
- **Easier to judge quality** at a glance
- Mostly dark/blue → probably good
- Lots of yellow → probably bad
- Saturated yellow → definitely bad

### 5. Simplicity
- **No two-pass processing** required
- **No dynamic global max** calculation
- Just use fixed [0, 1.0] and let EA handle the rest

---

## Impact on EA Search

### Before
Pareto front could include:
```
Config A: u_max=0.5, d_max=0.4, mqe=0.001 → In Pareto front
Config B: u_max=2.5, d_max=3.2, mqe=0.001 → In Pareto front (same MQE!)
```

### After
```
Config A: u_max=0.5, d_max=0.4, mqe=0.001 → In Pareto front ✓
Config B: u_max=2.5, d_max=3.2, mqe=0.001 × 24 = 0.024 → Dominated, removed ✗
```

Config B gets penalty factor = 1 + (2.2 × 10) = 23.0, making it uncompetitive.

---

## Tuning the Penalty

The penalty multiplier can be adjusted in [app/ea/ea.py:797](app/ea/ea.py#L797):

```python
penalty_factor = 1.0 + (max_excess * MULTIPLIER)
```

**Current**: `MULTIPLIER = 10.0` (very aggressive)

**Alternatives**:
- `MULTIPLIER = 5.0`: Moderate penalty (e.g., excess=0.5 → 3.5x penalty)
- `MULTIPLIER = 20.0`: Extreme penalty (e.g., excess=0.5 → 11x penalty)
- `MULTIPLIER = 1.0`: Linear penalty (e.g., excess=0.5 → 1.5x penalty)

**Recommendation**: Keep 10.0 for now. This strongly discourages poor organization while still allowing EA to evaluate such individuals (useful for understanding the search space).

---

## Labeling Tool Integration

The labeling tool already displays U-Matrix statistics including min/max/mean/std. With this change:

- **max ≤ 1.0**: Normal appearance, likely good quality
- **max > 1.0**: Saturated yellow areas, likely bad quality, will be auto-labeled as bad

This aligns the visual appearance with the EA's quality assessment.

---

## Testing

### Verify Penalty Application

Run EA and check logs for penalty messages:

```bash
cd /Users/tomas/OSU/Python/NexusSom/app
python3 run_ea.py --c ./test/ea-iris-config-small.json --i ./test/iris.csv
grep "poor organization" test/results/YYYYMMDD_HHMMSS/evolution.log
```

**Expected output**:
```
[UID abc123...] Applied 600.0% penalty for poor organization (U-Matrix max: 1.600, Distance max: 0.800)
```

### Verify Visualization

Check generated maps:

```bash
ls test/results/YYYYMMDD_HHMMSS/maps_dataset/*.png
```

All U-Matrix and Distance maps should:
- Use viridis colormap
- Range from dark blue (0) to yellow (1.0)
- Values > 1.0 appear as saturated yellow

### Verify Pareto Front Quality

```bash
python3 << EOF
import pandas as pd
df = pd.read_csv('test/results/YYYYMMDD_HHMMSS/pareto_front.csv')
print("U-Matrix max range:", df['u_matrix_max'].min(), "-", df['u_matrix_max'].max())
print("Distance max range:", df['distance_map_max'].min(), "-", df['distance_map_max'].max())
print("Maps with max > 1.0:", len(df[(df['u_matrix_max'] > 1.0) | (df['distance_map_max'] > 1.0)]))
EOF
```

**Expected**: Few (if any) Pareto front solutions with max > 1.0

---

## Status: ✅ PRODUCTION READY

All components are fully implemented and tested:

1. ✅ U-Matrix max tracking
2. ✅ Distance map max tracking
3. ✅ Organization quality check
4. ✅ Heavy penalty for max > 1.0
5. ✅ Fixed [0, 1.0] visualization normalization
6. ✅ Viridis colormap for both maps
7. ✅ CSV export of u_matrix_max and distance_map_max columns
8. ✅ Deduplication stats calculated correctly (multiprocessing-safe)

No configuration changes required - everything works automatically!

---

## Bug Fixes (2026-01-12)

### Fix 1: Missing CSV Columns

**Problem**: `u_matrix_max` and `distance_map_max` were calculated but not exported to results.csv

**Fix**: Added columns to `base_fields` in `log_result_to_csv()` ([ea/ea.py:560](app/ea/ea.py#L560))

```python
base_fields = ['uid', 'best_mqe', 'duration', 'topographic_error',
               'u_matrix_mean', 'u_matrix_std', 'u_matrix_max', 'distance_map_max',
               'total_weight_updates', 'epochs_ran', 'dead_neuron_count', 'dead_neuron_ratio']
```

### Fix 2: Deduplication Stats Show 0/0

**Problem**: Multiprocessing Pool causes global variables (EVALUATION_STATS) to not be shared between processes. Child processes update their local copy, but main process never sees the changes.

**Fix**: Calculate stats from results.csv instead of relying on globals ([ea/ea.py:518-537](app/ea/ea.py#L518-L537))

```python
# Calculate deduplication statistics from results.csv
# (Can't use EVALUATION_STATS due to multiprocessing - globals not shared between processes)
csv_path = os.path.join(WORKING_DIR, "results.csv")
try:
    import pandas as pd
    df = pd.read_csv(csv_path)
    new_evaluations = len(df)  # Unique UIDs in results.csv
    total_requested = population_size * generations  # Total individual evaluations requested
    cache_hits = total_requested - new_evaluations  # Duplicates skipped
except Exception as e:
    # Fallback to empty stats if file doesn't exist
    total_requested = 0
    new_evaluations = 0
    cache_hits = 0
```

### Log File Location

**Note**: Evolution logs are written to `log.txt`, NOT `evolution.log`

**Location**: `test/results/YYYYMMDD_HHMMSS/log.txt`

Example log entries:
```
[2026-01-12 13:21:46] [a59442d1207c28ac] Applied 25.5% penalty for poor organization (U-Matrix max: 0.461, Distance max: 1.025)
[2026-01-12 13:21:46] [a59442d1207c28ac] Applied 57.0% penalty for 67.0% dead neurons
```

---

## Related Documentation

- [CNN Visualization and EA Penalty](CNN_VISUALIZATION_AND_EA_PENALTY.md)
- [EA Deduplication](EA_DEDUPLICATION.md)
- [EA Float Precision](EA_FLOAT_PRECISION_AND_VALIDATION.md)
- [CNN U-Matrix Color Interpretation](CNN_UMATRIX_COLOR_INTERPRETATION.md)
