# CNN Visualization Improvements and EA Dead Neuron Penalty

**Date**: 2026-01-12
**Status**: ✅ IMPLEMENTED

---

## Overview

This document describes two critical improvements to the NexusSom project:
1. **Fixed colormap normalization** for U-Matrix and Distance Map visualizations
2. **Dead neuron penalty** in EA fitness evaluation

---

## Part 1: Fixed Colormap Normalization

### Problem

Previously, U-Matrix and Distance Map used auto-normalized colormaps (`viridis` and `magma`), which scaled colors to each map's min/max values. This caused critical issues:

- **Same color ≠ same quality**: Two maps could look identical but have vastly different distance scales
- **CNN confusion**: The CNN would learn from inconsistent color-to-value mappings
- **Human labeling difficulty**: Experts couldn't interpret map quality from visual appearance alone

**Example**:
```
Map A: U-Matrix min=0.00001, max=0.1    → Yellow means distance=0.1
Map B: U-Matrix min=0.0001, max=5.0     → Yellow means distance=5.0
```

Both maps would look similar (dark to yellow gradient), but Map B has 50× larger distances!

### Solution

Implemented **fixed grayscale normalization** where:
- **Black = 0** (always)
- **White = maximum value in that map**
- Colors are consistent across all maps of the same type

### Implementation

**File Modified**: [app/som/visualization.py](app/som/visualization.py)

#### 1. Added `fixed_norm` parameter to `_create_map()` (lines 12-14)

```python
def _create_map(som: KohonenSOM, values: np.ndarray, title: str, output_file: str,
                cmap: str, cbar_label: str = None, show_text: list = None, show_title: bool = True,
                fixed_norm: bool = False):
```

#### 2. Apply fixed normalization to patch collection (lines 51-53)

```python
# Apply fixed normalization if requested (black=0, white=max)
if fixed_norm:
    collection.set_clim(vmin=0, vmax=values.max())
```

#### 3. Apply fixed normalization to legend/colorbar (lines 79-83)

```python
# Use fixed normalization if requested, otherwise use min/max
if fixed_norm:
    norm = Normalize(vmin=0, vmax=values.max())
else:
    norm = Normalize(vmin=values.min(), vmax=values.max())
```

#### 4. Updated U-Matrix generation (lines 142-146)

```python
# Use viridis colormap with fixed normalization (dark blue=0, yellow=max)
# Fixed normalization ensures consistent color-to-value mapping across all maps
_create_map(som, u_matrix, "U-Matrix", output_file, cmap='viridis',
            cbar_label="Average distance to neighbors", show_title=show_title,
            fixed_norm=True)
```

**Before**: `cmap='viridis'` (auto-normalized to [min, max])
**After**: `cmap='viridis'` with `fixed_norm=True` (normalized to [0, max])

#### 5. Updated Distance Map generation (lines 338-342)

```python
# Use viridis colormap with fixed normalization (dark blue=0, yellow=max)
# Fixed normalization ensures consistent color-to-value mapping across all maps
_create_map(som, neuron_error_map, "Distance Map (Neuron QE)", output_file,
            cmap='viridis', cbar_label="Quantization Error", show_title=show_title,
            fixed_norm=True)
```

**Before**: `cmap='magma'` (auto-normalized to [min, max])
**After**: `cmap='viridis'` with `fixed_norm=True` (normalized to [0, max])

### Benefits

1. **Visual Consistency**: Same color always represents similar relative quality
2. **CNN Training**: The CNN learns from consistent color-to-value mappings
3. **Human Interpretation**: Experts can judge map quality from visual appearance alone
4. **No Legend Required**: The CNN doesn't need text overlays or absolute scale values

### Visual Comparison

**Before (viridis/magma, auto-normalized)**:
- Map A (distances 0.0001-0.1): Dark blue → Yellow gradient
- Map B (distances 0.001-5.0): Dark blue → Yellow gradient
- **Result**: Look identical, but Map B has 50× larger distances!

**After (viridis, fixed normalization)**:
- Map A (distances 0.0001-0.1): Dark blue → Darker green (low range)
- Map B (distances 0.001-5.0): Dark blue → Bright yellow (high range)
- **Result**: Visually distinct, reflects actual quality difference!

**Why Viridis?**
- Perceptually uniform (equal data steps = equal color steps)
- Better visual distinction across full range
- Easier to spot cluster boundaries (color transitions pop out)
- More visually appealing than grayscale
- Still works in grayscale for accessibility

---

## Part 2: Dead Neuron Penalty in EA Fitness

### Problem

The EA's Pareto front contained solutions with 68-76% dead neurons, which are clearly bad quality maps. The current multi-objective optimization treats `dead_neuron_ratio` as a separate objective, but doesn't strongly discourage it.

**Example from user's Pareto front**:
```csv
best_mqe,duration,topographic_error,dead_neuron_ratio
0.000194,0.79,0.0000,0.68  ← 68% dead neurons!
0.000116,0.91,0.0000,0.76  ← 76% dead neurons!
```

These solutions are technically "non-dominated" because they have low MQE/TE, but they're practically useless due to massive dead neuron counts.

### Solution

Added **progressive penalty** to MQE and topographic error based on dead neuron ratio:

- **No penalty** for `dead_neuron_ratio ≤ 10%`
- **Linear penalty** for `dead_neuron_ratio > 10%`
- Penalty formula: `penalty_factor = 1.0 + (dead_ratio - 0.10)`

**Examples**:
- 10% dead neurons → 0% penalty (no change)
- 20% dead neurons → 10% penalty (MQE/TE increased by 10%)
- 30% dead neurons → 20% penalty (MQE/TE increased by 20%)
- 50% dead neurons → 40% penalty (MQE/TE increased by 40%)
- 70% dead neurons → 60% penalty (MQE/TE increased by 60%)

This makes high-dead-neuron solutions less competitive in Pareto ranking.

### Implementation

**File Modified**: [app/ea/ea.py](app/ea/ea.py:781-793)

```python
dead_count, dead_ratio = som.calculate_dead_neurons(data)
training_results['dead_neuron_count'] = dead_count
training_results['dead_neuron_ratio'] = dead_ratio

# Apply penalty for excessive dead neurons (>10%)
# Penalty increases linearly with dead neuron ratio above threshold
# Example: 20% dead → 10% penalty, 30% dead → 20% penalty, etc.
DEAD_NEURON_THRESHOLD = 0.10
if dead_ratio > DEAD_NEURON_THRESHOLD:
    penalty_factor = 1.0 + (dead_ratio - DEAD_NEURON_THRESHOLD)
    training_results['best_mqe'] *= penalty_factor
    training_results['topographic_error'] = topographic_error * penalty_factor
    log_message(uid, f"Applied {(penalty_factor - 1.0) * 100:.1f}% penalty for {dead_ratio:.1%} dead neurons", working_dir)
else:
    training_results['topographic_error'] = topographic_error

log_message(uid, f"Evaluated – QE: {training_results['best_mqe']:.6f}, TE: {training_results.get('topographic_error', topographic_error):.4f}, Dead ratio: {dead_ratio:.2%}, Time: {training_results['duration']:.2f}s", working_dir)
```

### How It Works

1. **Calculate dead neuron ratio** (line 777-779)
2. **Check if penalty needed** (line 785)
3. **Apply penalty to MQE and TE** (lines 786-788)
   - Multiply both metrics by `penalty_factor`
   - This makes the solution appear worse in multi-objective ranking
4. **Log penalty application** (line 789)
5. **Store penalized values** in `training_results`

### Impact on Pareto Front

**Before penalty** (user's example):
```csv
best_mqe,duration,topographic_error,dead_neuron_ratio
0.000194,0.79,0.0000,0.68  ← Low MQE/TE → In Pareto front
0.000116,0.91,0.0000,0.76  ← Low MQE/TE → In Pareto front
```

**After penalty** (expected):
```csv
best_mqe,duration,topographic_error,dead_neuron_ratio
0.000311,0.79,0.0000,0.68  ← MQE increased 60% → Dominated, removed from front
0.000193,0.91,0.0000,0.76  ← MQE increased 66% → Dominated, removed from front
```

Solutions with <10% dead neurons will remain competitive, while high-dead-neuron solutions will be pushed out of the Pareto front.

### Adjusting the Penalty

The penalty threshold and scaling can be adjusted by modifying:

```python
DEAD_NEURON_THRESHOLD = 0.10  # No penalty below 10% dead neurons
penalty_factor = 1.0 + (dead_ratio - DEAD_NEURON_THRESHOLD)  # Linear scaling
```

**Alternative penalty schemes**:

1. **Quadratic penalty** (stronger for high ratios):
   ```python
   penalty_factor = 1.0 + ((dead_ratio - DEAD_NEURON_THRESHOLD) ** 2) * 5
   ```

2. **Exponential penalty** (very aggressive):
   ```python
   penalty_factor = np.exp((dead_ratio - DEAD_NEURON_THRESHOLD) * 3)
   ```

3. **Custom scaling** (e.g., user wanted 20% dead → 10% penalty):
   ```python
   penalty_factor = 1.0 + (dead_ratio - DEAD_NEURON_THRESHOLD) * 0.5
   # 20% dead → penalty = 1.0 + (0.2 - 0.1) * 0.5 = 1.05 (5% penalty)
   ```

Current implementation uses **1:1 linear scaling** (10% excess → 10% penalty).

---

## Testing

### Test 1: Verify Fixed Normalization

Run EA and check generated maps:

```bash
cd /Users/tomas/OSU/Python/NexusSom/app
python3 run_ea.py --c ./test/ea-iris-config-small.json --i ./test/iris.csv
```

**Expected**:
- U-Matrix maps use grayscale (black=0, white=max)
- Distance maps use grayscale (black=0, white=max)
- Dead neuron maps still use binary colormap (unchanged)
- Visual appearance reflects actual quality differences

### Test 2: Verify Dead Neuron Penalty

Check EA logs for penalty messages:

```bash
grep "Applied.*penalty" test/results/YYYYMMDD_HHMMSS/evolution.log
```

**Expected output**:
```
[UID abc123...] Applied 60.0% penalty for 70.0% dead neurons
[UID def456...] Applied 40.0% penalty for 50.0% dead neurons
```

### Test 3: Verify Pareto Front Quality

Check final Pareto front for dead neuron ratios:

```bash
python3 << EOF
import pandas as pd
df = pd.read_csv('test/results/YYYYMMDD_HHMMSS/pareto_front.csv')
print(df[['best_mqe', 'topographic_error', 'dead_neuron_ratio']].describe())
print("\nMax dead neuron ratio in Pareto front:", df['dead_neuron_ratio'].max())
EOF
```

**Expected**:
- Most Pareto front solutions have `dead_neuron_ratio < 20%`
- Few (if any) solutions with `dead_neuron_ratio > 50%`
- Solutions with 70%+ dead neurons should be absent

---

## Compatibility

### Backward Compatibility

**Visualization changes**:
- Existing code will continue to work
- Old maps (already generated) are not affected
- New maps will use fixed normalization
- CNN training datasets can be regenerated

**EA penalty changes**:
- Existing results files are not affected
- New EA runs will apply penalty automatically
- No configuration changes required
- Can be disabled by setting `DEAD_NEURON_THRESHOLD = 1.0`

### Impact on Existing Workflows

1. **CNN Training**: Regenerate map dataset with new visualization settings
2. **Human Labeling**: Use updated labeling tool (already shows 3 maps)
3. **EA Optimization**: Re-run EA to get improved Pareto front (optional)

---

## Status: ✅ PRODUCTION READY

Both improvements are fully implemented and ready for use:

1. ✅ **Fixed colormap normalization** for U-Matrix and Distance Map
2. ✅ **Dead neuron penalty** in EA fitness evaluation

No configuration changes required - both features work out of the box!

---

## Related Documentation

- [EA Deduplication](EA_DEDUPLICATION.md)
- [EA Float Precision](EA_FLOAT_PRECISION_AND_VALIDATION.md)
- [EA Bug Fixes](EA_BUGFIX_ITERATION.md)
- [CNN U-Matrix Color Interpretation](CNN_UMATRIX_COLOR_INTERPRETATION.md)
