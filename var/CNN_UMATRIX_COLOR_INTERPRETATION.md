# U-Matrix Color Gradient Interpretation

**Date**: 2026-01-12
**Status**: 📘 DOCUMENTATION

---

## The Problem: Same Colors ≠ Same Quality

### Critical Issue

The U-Matrix visualization uses the **viridis** colormap, which automatically normalizes to each map's min/max values. This means:

```
Map A: distances [0.00001 - 0.1]     → viridis (dark blue to yellow)
Map B: distances [0.0001  - 5.0]     → viridis (dark blue to yellow)
```

**Both maps look identical, but Map B has 50× larger cluster boundaries!**

---

## Understanding the Viridis Colormap

### Color Meaning in U-Matrix

The U-Matrix shows **average distance to neighboring neurons**:

- **Dark Blue (Low Values)**: Neurons are similar → Inside a cluster
- **Yellow (High Values)**: Neurons are different → Cluster boundary
- **Gradient**: Smooth transition from cluster interior to boundaries

### Viridis Color Scale

```
Dark Purple/Blue  →  Teal  →  Green  →  Yellow/White
      (min)                              (max)
```

---

## Why Absolute Values Matter

### Scenario 1: Well-Formed Clusters (Good Map)
```
U-Matrix Scale: min=0.00001, max=0.15, mean=0.05 (±0.03)

Interpretation:
- Small min (0.00001): Very homogeneous cluster interiors
- Small max (0.15): Clear but gentle boundaries
- Low mean (0.05): Mostly within-cluster neurons
- Low std (0.03): Consistent structure

✓ Good quality - distinct clusters with clear boundaries
```

### Scenario 2: Weak Clustering (Bad Map)
```
U-Matrix Scale: min=0.0001, max=5.0, mean=2.3 (±1.8)

Interpretation:
- Large max (5.0): Very sharp/extreme boundaries
- High mean (2.3): Most neurons are on boundaries
- High std (1.8): Chaotic, inconsistent structure

✗ Bad quality - poorly defined clusters, noisy structure
```

### Scenario 3: Over-Fitting (Bad Map)
```
U-Matrix Scale: min=0.0, max=0.00005, mean=0.00002 (±0.00001)

Interpretation:
- Tiny max (0.00005): Almost no boundaries
- Tiny mean: All neurons too similar
- Map has memorized data without generalizing

✗ Bad quality - no useful structure, over-fitted
```

---

## Interpretation Guidelines

### What Makes a "Good" U-Matrix?

1. **Min value close to 0**: Homogeneous cluster cores
2. **Max value moderate** (not too small, not too large):
   - Too small (< 0.01): Over-fitting, no structure
   - Too large (> 2.0): Chaotic, weak clustering
   - Sweet spot: 0.1 - 1.0 (dataset dependent)
3. **Mean < Max/2**: Most neurons inside clusters, not on boundaries
4. **Std < Mean**: Consistent structure, not chaotic

### Visual Patterns to Look For

**Good Map Characteristics**:
- Clear dark blue regions (cluster interiors)
- Narrow yellow/green boundaries separating clusters
- Smooth gradients within clusters
- Distinct, separated regions

**Bad Map Characteristics**:
- Mostly yellow (all boundaries, no clusters)
- Mostly dark blue (no boundaries, over-fitted)
- Noisy, chaotic patterns
- No clear regions or structure

---

## Dataset-Specific Normalization

### Why Values Differ by Dataset

Different datasets have different intrinsic distances:

```
Iris (4 features, range 0-8):
- Typical U-Matrix range: 0.05 - 0.8
- Well-clustered example: min=0.01, max=0.5

Wine (13 features, range 0-1700):
- Typical U-Matrix range: 0.5 - 5.0
- Well-clustered example: min=0.2, max=3.0

MNIST (784 features, range 0-255):
- Typical U-Matrix range: 5.0 - 50.0
- Well-clustered example: min=2.0, max=30.0
```

**Key Point**: Absolute values are dataset-specific. Compare within the same dataset, not across datasets!

---

## Updated Labeling Tool Features

### What's Now Displayed

The labeling tool now shows:

```
U-Matrix Scale: min=0.012345, max=0.456789, mean=0.123456 (±0.067890)
```

This tells you:
- **min**: Smallest distance (cluster interior)
- **max**: Largest distance (boundary)
- **mean**: Average distance
- **std**: Variation (consistency)

### Per-Map Annotations

Each U-Matrix now shows:
```
Dark blue = similar neurons (min=0.0123)
Yellow = cluster boundaries (max=0.4568)
```

This helps you interpret the colors in the context of **actual distance values**.

---

## CNN Training Implications

### What the CNN Will Learn

The CNN takes the **RGB composite** as input:
- **R (Red)**: U-Matrix (normalized to [0-255])
- **G (Green)**: Distance Map (normalized to [0-255])
- **B (Blue)**: Dead Neurons Map (normalized to [0-255])

**Important**: The CNN sees pixel values (0-255), not the original distance values!

### Why We Need Metrics

To train the CNN effectively, we need to provide:
1. **RGB image** (visual patterns)
2. **Actual distance statistics** (scale context)
3. **Quality metrics** (MQE, topographic error, dead neurons)

The CNN learns to recognize "good quality patterns" by correlating:
- Visual patterns (RGB) with quality metrics
- Scale information (min/max/mean) with clustering quality

---

## Examples from Iris Test Run

### Example 1: Good Quality Map
```
UID: a3f8b912...
MQE: 0.012345
Topo Error: 0.05
Dead: 2%
U-Matrix Scale: min=0.008, max=0.45, mean=0.12 (±0.08)

Visual: Clear dark blue regions with thin yellow boundaries
Quality: ✓ Excellent - well-formed clusters
```

### Example 2: Bad Quality Map (Over-Fitted)
```
UID: c7d4e521...
MQE: 0.000123
Topo Error: 0.01
Dead: 85%
U-Matrix Scale: min=0.0, max=0.002, mean=0.0005 (±0.0003)

Visual: Almost entirely dark blue, no boundaries
Quality: ✗ Bad - over-fitted, 85% dead neurons
```

### Example 3: Bad Quality Map (Chaotic)
```
UID: b8c2d745...
MQE: 0.234567
Topo Error: 0.45
Dead: 15%
U-Matrix Scale: min=0.05, max=3.5, mean=1.8 (±1.2)

Visual: Noisy mix of colors, no clear regions
Quality: ✗ Bad - poor clustering, chaotic structure
```

---

## Summary

### Key Takeaways

1. **Color alone is NOT enough** - you need the scale (min/max/mean)
2. **Same visual ≠ same quality** - check absolute distance values
3. **Dataset-specific ranges** - compare within dataset, not across
4. **Good clustering** = low min, moderate max, low mean, low std
5. **Bad maps** = tiny max (over-fit) OR huge max (chaotic) OR high dead ratio

### Updated Labeling Workflow

1. Look at U-Matrix visual pattern
2. **Check the scale values** (min/max/mean/std)
3. Check other metrics (MQE, topo error, dead neurons)
4. Combine all information to judge quality
5. Label as good/bad based on **total picture**, not just colors

---

## Status: ✅ IMPLEMENTED

The labeling tool now displays U-Matrix distance statistics, allowing proper interpretation of the color gradients in context of actual distance values.
