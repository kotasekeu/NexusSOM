# EA Float Precision and Enhanced Validation

**Date**: 2026-01-11
**Status**: ✅ IMPLEMENTED

---

## Changes Implemented

### 1. Float Precision Limiting (2 Decimal Places)

**Problem**: Floating-point parameters had excessive precision (e.g., `48.16083009614849`), making them:
- Hard to read
- Unlikely to produce duplicates for deduplication cache
- More precision than needed for SOM training

**Solution**: Round all float parameters to 2 decimal places

**Files Modified**: `app/ea/genetic_operators.py`

#### Random Generation (line 145)
```python
# Before
config[key] = random.uniform(spec['min'], spec['max'])

# After
config[key] = round(random.uniform(spec['min'], spec['max']), 2)
```

#### Crossover (lines 197-198)
```python
# Before
child1[key] = c1
child2[key] = c2

# After
child1[key] = round(c1, 2)
child2[key] = round(c2, 2)
```

#### Mutation (lines 255-256)
```python
# Before
mutated[key] = polynomial_mutation(config[key], eta=eta, bounds=bounds, mutation_prob=mutation_prob)

# After
mutated[key] = round(polynomial_mutation(config[key], eta=eta, bounds=bounds, mutation_prob=mutation_prob), 2)
```

---

### 2. Enhanced Parameter Ordering Validation

**Problem**: Some parameter pairs have ordering requirements:
- Learning rate should DECREASE: `start_lr >= end_lr`
- Batch size should INCREASE: `start_batch <= end_batch`
- Radius should DECREASE: `start_radius >= end_radius`

Without validation, invalid combinations could be generated.

**Solution**: Enhanced `validate_and_repair()` to swap values if ordering is wrong

**File Modified**: `app/ea/ea.py` (lines 59-75)

```python
# Ensure start_learning_rate >= end_learning_rate (LR should decrease during training)
if 'start_learning_rate' in repaired_config and 'end_learning_rate' in repaired_config:
    if repaired_config['start_learning_rate'] < repaired_config['end_learning_rate']:
        repaired_config['start_learning_rate'], repaired_config['end_learning_rate'] = \
            repaired_config['end_learning_rate'], repaired_config['start_learning_rate']

# Ensure end_batch_percent >= start_batch_percent (batch size should grow during training)
if 'start_batch_percent' in repaired_config and 'end_batch_percent' in repaired_config:
    if repaired_config['start_batch_percent'] > repaired_config['end_batch_percent']:
        repaired_config['start_batch_percent'], repaired_config['end_batch_percent'] = \
            repaired_config['end_batch_percent'], repaired_config['start_batch_percent']

# Ensure start_radius >= end_radius (radius should decrease during training)
if 'start_radius' in repaired_config and 'end_radius' in repaired_config:
    if repaired_config['start_radius'] < repaired_config['end_radius']:
        repaired_config['start_radius'], repaired_config['end_radius'] = \
            repaired_config['end_radius'], repaired_config['start_radius']
```

---

### 3. Smart growth_g Handling for Linear Decay/Growth

**Problem**: The `growth_g` parameter is only used for exponential/logarithmic decay/growth curves. For linear curves, it's ignored. This means:

```python
Config A: {lr_decay='linear-drop', growth_g=15.3}  # growth_g ignored
Config B: {lr_decay='linear-drop', growth_g=42.7}  # growth_g ignored
```

These are **functionally identical** but have different UIDs, wasting evaluations.

**Solution**: Set `growth_g = 0` when all decay/growth types are linear

**File Modified**: `app/ea/ea.py` (lines 81-95)

```python
# Set growth_g = 0 for linear decay/growth types (where it's not used)
# This prevents different individuals that are functionally identical
if 'lr_decay_type' in repaired_config and repaired_config['lr_decay_type'] == 'linear-drop':
    repaired_config['growth_g'] = 0
elif 'radius_decay_type' in repaired_config and repaired_config['radius_decay_type'] == 'linear-drop':
    if 'lr_decay_type' not in repaired_config or repaired_config['lr_decay_type'] == 'linear-drop':
        repaired_config['growth_g'] = 0
elif 'batch_growth_type' in repaired_config and repaired_config['batch_growth_type'] == 'linear-growth':
    if ('lr_decay_type' not in repaired_config or repaired_config['lr_decay_type'] == 'linear-drop') and \
       ('radius_decay_type' not in repaired_config or repaired_config['radius_decay_type'] == 'linear-drop'):
        repaired_config['growth_g'] = 0

# Otherwise ensure growth_g is at least 1.0 when it's actually used
if 'growth_g' in repaired_config and repaired_config['growth_g'] != 0:
    repaired_config['growth_g'] = max(1.0, repaired_config['growth_g'])
```

**Logic**:
- If ALL curves are linear → `growth_g = 0`
- If ANY curve is non-linear → `growth_g >= 1.0`

---

## Benefits

### 1. Improved Readability
**Before**:
```
start_learning_rate: 0.48160830096148486
end_learning_rate: 0.7234512389674123
```

**After**:
```
start_learning_rate: 0.48
end_learning_rate: 0.72
```

### 2. Better Deduplication
With 2 decimal places, there are only 100 possible values in [0, 1] instead of infinite. This increases the likelihood of duplicate detection, improving cache hit rate.

**Example**:
```
Generation 1: {start_lr: 0.48160830..., ...}
Generation 5: {start_lr: 0.48241932..., ...}
```

**Before**: Different UIDs (different evaluations)
**After**: Both round to 0.48 → same UID → cache hit!

### 3. Proper Constraint Enforcement
All parameter pairs now respect their ordering requirements:
- ✓ LR always decreases
- ✓ Batch size always increases
- ✓ Radius always decreases

### 4. Elimination of Duplicate Functionality
Configurations with linear decay no longer differ only by unused `growth_g` values:

**Before**: These would be evaluated separately:
```
{lr_decay: 'linear-drop', growth_g: 15.3}
{lr_decay: 'linear-drop', growth_g: 42.7}
```

**After**: Both become:
```
{lr_decay: 'linear-drop', growth_g: 0}
```
→ Same UID → Evaluated once!

---

## Test Results

```
Config 1:
  LR: 0.60 -> 0.30 (start >= end: True)
  Batch: 4.84 -> 5.81 (start <= end: True)
  Decay: exp-drop, growth_g: 19.94 (0 if linear: N/A)

Config 2:
  LR: 0.93 -> 0.49 (start >= end: True)
  Batch: 6.21 -> 7.79 (start <= end: True)
  Decay: exp-drop, growth_g: 4.19 (0 if linear: N/A)

Config 5:
  LR: 0.35 -> 0.15 (start >= end: True)
  Batch: 10.50 -> 14.36 (start <= end: True)
  Decay: linear-drop, growth_g: 0 (0 if linear: True)
```

✓ All floats have 2 decimal places
✓ All ordering constraints satisfied
✓ growth_g = 0 when decay type is linear

---

## Impact on Search Space

### Before
- Float precision: ~15 decimal places
- Possible values in [0, 1]: ∞ (continuous)
- Duplicate probability: ~0%

### After
- Float precision: 2 decimal places
- Possible values in [0, 1]: 101 (0.00, 0.01, ..., 1.00)
- Duplicate probability: Significantly higher
- Functional duplicates: Eliminated (via growth_g=0)

---

## Compatibility

These changes are **fully backward compatible**:
- Existing configurations still work
- Old results files are not affected
- Validation is automatic and transparent

---

## Status: ✅ PRODUCTION READY

All improvements are implemented and tested. The EA will now:
- Generate cleaner, more readable parameters
- Respect all ordering constraints
- Eliminate functionally duplicate configurations
- Improve deduplication cache hit rate

---

## Bug Fix: growth_g Division by Zero

**Date**: 2026-01-11

### Problem
Initial implementation set `growth_g = 0` too aggressively, causing division by zero in exponential/logarithmic decay functions:

```
/Users/tomas/OSU/Python/NexusSom/app/som/som.py:187: RuntimeWarning: invalid value encountered in scalar divide
  return start + (end - start) * (np.log(self.growth_g * t + 1) / np.log(self.growth_g * N + 1))

[ERROR] Individual failed: cannot convert float NaN to integer
```

### Root Cause
The validation logic was setting `growth_g = 0` when only **some** curves were linear, but if **any** curve was non-linear (exp/log), it would try to use `growth_g` in the formula, resulting in division by zero.

### Fix
Updated logic to only set `growth_g = 0` when **ALL** curves are linear:

```python
# Old (buggy)
if 'lr_decay_type' in repaired_config and repaired_config['lr_decay_type'] == 'linear-drop':
    repaired_config['growth_g'] = 0  # BUG: What if radius is exp-drop?

# New (correct)
all_linear = True
if 'lr_decay_type' in repaired_config and repaired_config['lr_decay_type'] != 'linear-drop':
    all_linear = False
if 'radius_decay_type' in repaired_config and repaired_config['radius_decay_type'] != 'linear-drop':
    all_linear = False
if 'batch_growth_type' in repaired_config and repaired_config['batch_growth_type'] != 'linear-growth':
    all_linear = False

if all_linear:
    repaired_config['growth_g'] = 0
else:
    repaired_config['growth_g'] = max(1.0, repaired_config['growth_g'])
```

### Test Results
```
All linear: growth_g = 0 (safe - not used anywhere)
LR exp: growth_g = 1.0 (required for exp-drop formula)
Batch exp: growth_g = 1.0 (required for exp-growth formula)
All exp: growth_g = 1.0 (required for all formulas)
```

✓ No more division by zero errors
✓ growth_g = 0 only when truly unused
✓ growth_g >= 1.0 when needed by any non-linear curve

---

## Status: ✅ FULLY FIXED

All validation issues resolved. The EA now correctly handles all parameter combinations without errors.
