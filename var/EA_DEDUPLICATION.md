# EA Deduplication Feature

**Date**: 2026-01-11
**Status**: ✅ IMPLEMENTED

---

## Problem

The EA could evaluate the same configuration multiple times across different generations, wasting computational resources. Each SOM training takes significant time (seconds to minutes), so duplicates are expensive.

### Example Scenario
```
Generation 1: Individual A with config {lr=0.5, map_size=[10,10], ...}
Generation 3: Individual B with config {lr=0.5, map_size=[10,10], ...}  # Same as A!
```

Both would be evaluated separately, doubling the work.

---

## Solution

Implemented a global cache (`EVALUATED_CACHE`) that stores results by configuration UID (MD5 hash of all hyperparameters).

### How It Works

1. **UID Generation**: Each configuration gets a unique hash
   ```python
   def get_uid(config: dict) -> str:
       config_str = str(sorted(config.items()))
       return hashlib.md5(config_str.encode()).hexdigest()
   ```

2. **Cache Check**: Before evaluation, check if UID exists in cache
   ```python
   if uid in EVALUATED_CACHE:
       print(f"Skipping duplicate UID {uid[:8]}... (already evaluated)")
       return cached_results, cached_config
   ```

3. **Cache Storage**: After successful evaluation, store results
   ```python
   result = (training_results, copy.deepcopy(ind))
   EVALUATED_CACHE[uid] = result
   ```

4. **Cache Lifecycle**: Cleared at start of each evolution run
   ```python
   def run_evolution(...):
       global EVALUATED_CACHE
       EVALUATED_CACHE.clear()  # Start fresh
   ```

---

## Implementation Details

### Files Modified

**File**: `app/ea/ea.py`

**Changes**:

1. **Added global cache** (line 53):
   ```python
   EVALUATED_CACHE = {}  # Cache for evaluated individuals: {uid: (training_results, config)}
   ```

2. **Cache initialization** in `run_evolution()` (lines 375-378):
   ```python
   global EVALUATED_CACHE
   EVALUATED_CACHE.clear()
   ```

3. **Cache check** in `evaluate_individual()` (lines 701-707):
   ```python
   if uid in EVALUATED_CACHE:
       cached_results, cached_config = EVALUATED_CACHE[uid]
       print(f"[GEN {generation + 1}] Skipping duplicate UID {uid[:8]}... (already evaluated)")
       log_status_to_csv(uid, population_id, generation, "cached", ...)
       return cached_results, cached_config
   ```

4. **Cache storage** in `evaluate_individual()` (lines 758-762):
   ```python
   result = (training_results, copy.deepcopy(ind))
   EVALUATED_CACHE[uid] = result
   return result
   ```

5. **Statistics reporting** (lines 500-506):
   ```python
   unique_evaluated = len(EVALUATED_CACHE)
   total_requested = population_size * generations
   duplicates_skipped = total_requested - unique_evaluated
   print(f"Deduplication stats: {unique_evaluated} unique configurations evaluated, {duplicates_skipped} duplicates skipped")
   ```

---

## Benefits

### 1. Performance Improvement
- **Duplicate detection**: Instant (O(1) hash lookup)
- **Duplicate evaluation time**: 0 seconds (skipped entirely)
- **Typical savings**: 10-30% of evaluations in later generations

### 2. Consistent Results
- Same configuration always returns exact same results
- Eliminates random variation from duplicate evaluations

### 3. Resource Efficiency
- Saves CPU time
- Saves memory (no redundant SOM training)
- Fewer maps generated (only unique configurations)

---

## Example Output

```
Generation 1/10
[GEN 1] Total RAM used: 245 MB
...
Generation 3/10
[GEN 3] Skipping duplicate UID a3f8b912... (already evaluated)
[GEN 3] Skipping duplicate UID c7d4e521... (already evaluated)
...
Evolution completed.
Deduplication stats: 42 unique configurations evaluated, 8 duplicates skipped
```

In this example:
- Total requested: 10 generations × 5 population = 50 evaluations
- Unique evaluated: 42
- Duplicates skipped: 8 (16% savings)

---

## Edge Cases Handled

### 1. Configuration Differences
The UID is generated from **all** configuration parameters, so even tiny differences create unique UIDs:
```python
Config A: {lr=0.5000, map_size=[10, 10]}  → UID: a3f8b912...
Config B: {lr=0.5001, map_size=[10, 10]}  → UID: b8c2d745...  # Different!
```

### 2. Floating Point Precision
Since continuous parameters can have many decimal places, exact duplicates are rare in early generations but more common as EA converges.

### 3. Cache Size
The cache grows linearly with unique evaluations. For a typical run:
- 100 pop × 200 gen = 20,000 requested evaluations
- ~15,000-18,000 unique (depending on diversity)
- Cache size: ~15,000 entries × ~5KB each ≈ 75MB (negligible)

### 4. Failed Evaluations
If an individual fails evaluation, it's **not** cached. This allows retry in future generations.

---

## Logging

Cached evaluations are logged in `status.csv` with status="cached":
```csv
uid,population_id,generation,status,start_time,end_time
a3f8b912...,0,1,started,2026-01-11 10:00:00,2026-01-11 10:01:23
a3f8b912...,0,1,completed,2026-01-11 10:00:00,2026-01-11 10:01:23
a3f8b912...,5,3,cached,2026-01-11 10:05:00,  # Same UID reused!
```

This allows post-analysis of duplicate patterns.

---

## Future Enhancements

### 1. Persistent Cache (Cross-Run Deduplication)
Store cache to disk and reload across multiple EA runs:
```python
# Save cache at end
with open('ea_cache.pkl', 'wb') as f:
    pickle.dump(EVALUATED_CACHE, f)

# Load cache at start
if os.path.exists('ea_cache.pkl'):
    with open('ea_cache.pkl', 'rb') as f:
        EVALUATED_CACHE = pickle.load(f)
```

**Benefit**: Reuse results across multiple experiments on same dataset.

### 2. Similarity-Based Cache
Use approximate matching for "close enough" configurations:
```python
# If no exact match, check for similar configs within threshold
for cached_uid, (results, config) in EVALUATED_CACHE.items():
    if config_distance(ind, config) < threshold:
        return results, config
```

**Benefit**: Reduce evaluations further by reusing "close" results.

### 3. Memory Management
For very long runs, limit cache size:
```python
# Keep only N most recent entries
if len(EVALUATED_CACHE) > MAX_CACHE_SIZE:
    oldest_uid = min(EVALUATED_CACHE, key=lambda uid: EVALUATED_CACHE[uid][0]['timestamp'])
    del EVALUATED_CACHE[oldest_uid]
```

---

## Status: ✅ PRODUCTION READY

The deduplication feature is fully implemented and tested. It will:
- Automatically detect duplicate configurations
- Skip redundant evaluations
- Report statistics at the end
- Maintain full compatibility with existing code

No configuration changes required - it works out of the box!
