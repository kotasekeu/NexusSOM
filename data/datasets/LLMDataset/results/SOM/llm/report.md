**Dataset Summary**

This dataset contains physicochemical measurements and sensory evaluation of 30 wine samples from a regional winery. The dataset includes various characteristics such as alcohol content, acidity, sugar levels, tannin concentration, color intensity, aroma score, body/mouthfeel evaluation, and overall quality classification.

**Map Quality**

The SOM map has a size of 3x3 hex with no dead neurons (0%). The Mean Quantization Error (MQE) is 0.4215, indicating good mapping quality. The topographic error is 0.1, suggesting that the map preserves some of the neighborhood relationships between samples.

**Key Findings**

The SOM analysis reveals clear separation by wine type, with distinct clusters for red wines, white wines, and a transition zone in the middle row containing high-quality whites, rosés, and a mixed low/medium cluster.

**Cluster Analysis**

### Cluster 1: High-Quality Red Wines (Neuron 0_1)

- Contains: High-quality red wine samples with balanced characteristics.
- Key characteristics:
  - Alcohol content: 13.00% (high)
  - Acidity: 5.00 g/L (typical for reds)
  - Sugar levels: 2.28 g/L (dry)
  - Tannin concentration: 7.63 (balanced, typical for high-quality reds)
- Differs from other clusters by having balanced tannin and color intensity.

### Cluster 2: Medium-Quality Red Wines (Neuron 0_2)

- Contains: Medium-quality red wine samples with moderate characteristics.
- Key characteristics:
  - Alcohol content: 11.80% (moderate)
  - Acidity: 4.30 g/L (typical for reds)
  - Sugar levels: 1.87 g/L (dry)
  - Tannin concentration: 5.87 (moderate, typical for medium-quality reds)
- Differs from other clusters by having moderate tannin and color intensity.

### Cluster 3: High-Quality White Wines (Neuron 1_0)

- Contains: High-quality white wine samples with high alcohol content and aroma score.
- Key characteristics:
  - Alcohol content: 13.13% (high)
  - Acidity: 5.50 g/L (typical for whites)
  - Sugar levels: 4.53 g/L (moderate)
  - Tannin concentration: 1.80 (low, typical for high-quality whites)
- Differs from other clusters by having high alcohol content and aroma score.

### Cluster 4: Low-Quality White Wines (Neuron 2_2)

- Contains: Low-quality white wine samples with very sweet, acidic, and thin body.
- Key characteristics:
  - Alcohol content: 9.85% (low)
  - Acidity: 7.55 g/L (high)
  - Sugar levels: 8.65 g/L (very high)
  - Tannin concentration: 0.45 (very low, typical for poor-quality whites)
- Differs from other clusters by having very sweet and acidic characteristics.

### Cluster 5: Mixed Low/Medium Quality (Neuron 1_2)

- Contains: A mixed cluster of low/medium quality reds and rosés with moderate values.
- Key characteristics:
  - Alcohol content: 11.30% (moderate)
  - Acidity: 4.79 g/L (typical for reds/rosés)
  - Sugar levels: 2.83 g/L (dry)
  - Tannin concentration: 3.86 (moderate, typical for mixed cluster)
- Differs from other clusters by having moderate values and a mix of red and rosé characteristics.

### Cluster 6: Premium High-Alcohol Red Wines (Neuron 0_0)

- Contains: Premium high-alcohol red wine samples with extreme tannin and color intensity.
- Key characteristics:
  - Alcohol content: 14.65% (very high)
  - Acidity: 5.95 g/L (typical for reds)
  - Sugar levels: 3.00 g/L (dry)
  - Tannin concentration: 9.20 (extreme, typical for premium high-alcohol reds)
- Differs from other clusters by having extreme tannin and color intensity.

### Cluster 7: High-Quality Rosé Wines (Neuron 1_1)

- Contains: High-quality rosé wine samples with balanced characteristics.
- Key characteristics:
  - Alcohol content: 12.90% (moderate)
  - Acidity: 4.90 g/L (typical for rosés)
  - Sugar levels: 3.35 g/L (dry)
  - Tannin concentration: 3.95 (balanced, typical for high-quality rosés)
- Differs from other clusters by having balanced tannin and color intensity.

### Cluster 8: Low-Quality White Wines (Neuron 2_1)

- Contains: Low-quality white wine samples with very sweet, acidic, and thin body.
- Key characteristics:
  - Alcohol content: 10.90% (low)
  - Acidity: 6.93 g/L (high)
  - Sugar levels: 7.20 g/L (very high)
  - Tannin concentration: 0.90 (very low, typical for poor-quality whites)
- Differs from other clusters by having very sweet and acidic characteristics.

### Cluster 9: Transition Zone (Neuron 1_2)

- Contains: A mixed cluster of high-quality whites, rosés, and a low/medium quality red/rosé mix.
- Key characteristics:
  - Alcohol content: 11.30% (moderate)
  - Acidity: 4.79 g/L (typical for reds/rosés)
  - Sugar levels: 2.83 g/L (dry)
  - Tannin concentration: 3.86 (moderate, typical for mixed cluster)
- Differs from other clusters by having moderate values and a mix of high-quality whites, rosés, and low/medium quality reds/rosés.

**Anomalies**

Four global outliers were detected:

1. Sample W026 (Neuron 0_0): Has extreme tannin concentration (9.50), color intensity (11.80), alcohol content (15.20), and body/mouthfeel evaluation (9.80).
2. Sample W027 (Neuron 2_2): Has very low alcohol content (9.50), high acidity (7.80), very high sugar levels (9.20), and very low tannin concentration (0.30).

Two local outliers were detected:

1. Sample W006 (Neuron 0_0): Has high alcohol content (14.10) and extreme tannin concentration (8.90).
2. Sample W017 (Neuron 2_2): Has very sweet sugar levels (8.10), low tannin concentration (0.60), and thin body/mouthfeel evaluation (2.80).

**Conclusions**

This SOM analysis reveals clear separation by wine type, with distinct clusters for red wines, white wines, and a transition zone in the middle row containing high-quality whites, rosés, and a mixed low/medium cluster. The analysis also detects anomalies in terms of extreme tannin concentration, color intensity, alcohol content, acidity, sugar levels, and body/mouthfeel evaluation.

The results suggest that:

* High-quality red wines have balanced tannin and color intensity.
* High-quality white wines have high alcohol content and aroma score.
* Low-quality white wines have very sweet, acidic, and thin body characteristics.
* The transition zone contains a mix of high-quality whites, rosés, and low/medium quality red/rosé mixes.

These findings can be used to inform wine production decisions, such as selecting grape varieties for specific wine types or adjusting winemaking techniques to improve quality.