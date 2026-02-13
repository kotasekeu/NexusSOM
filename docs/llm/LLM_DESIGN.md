# LLM Module — The Voice — Design

---

## 1. How It Works

The Voice is not an AI that analyses raw data. It is a translation layer — it takes structured SOM analysis results and converts them into natural language that a domain expert can understand without knowing SOM internals.

### 1.1 Two-Context Architecture

The LLM receives two separate context documents:

**Context A — Dataset Description** (user-provided, static per dataset)
```
This is the Breast Cancer Wisconsin Diagnostic Dataset.
Each row is one tumor biopsy sample.

Columns:
- id: unique patient identifier
- diagnosis: B = Benign, M = Malignant
- radius_mean: mean distance from center to perimeter of the cell nucleus
- texture_mean: standard deviation of gray-scale values in the cell image
- ...

Domain knowledge:
- Malignant tumors tend to have larger radius, irregular shape (higher concavity)
- Benign tumors cluster together with lower metric values
```

**Context B — SOM Analysis Summary** (auto-generated from SOM outputs)
```
MAP OVERVIEW:
- Size: 5x5 hexagonal, 569 samples, 25 neurons
- Quality: MQE 0.569, topographic error 0.12, 5 dead neurons (20%)

CLUSTERS:
- Neuron 1_0: 242 samples (largest), 100% Benign, QE 0.45
  Avg radius_mean: 12.1, avg area_mean: 420, avg smoothness: 0.09
- Neuron 0_4: 47 samples, 100% Malignant, QE 0.66
  Avg radius_mean: 18.5, avg area_mean: 1050, avg smoothness: 0.11
- ...

ANOMALIES:
- Sample 911296202: area_mean=2501 (global max, 2.5σ above mean)
  radius_worst=36.04 (global max), perimeter_worst=251.20 (global max)
  Located in neuron 0_4 (Malignant cluster) — extreme case
- ...
```

### 1.2 System Prompt

The system prompt constrains the LLM to:
1. Only discuss the provided dataset and SOM analysis
2. Ground every claim in specific data from Context B
3. Use terminology from Context A (domain language, not SOM jargon)
4. When asked about something not in the data, say "This information is not available in the current analysis"

### 1.3 Processing Pipeline

```
Step 1: Load SOM outputs (JSON/CSV)
Step 2: Load dataset context file (user-provided TXT)
Step 3: Build Context B — summarize clusters, anomalies, distributions
Step 4: Compose full prompt = System Prompt + Context A + Context B + User Query
Step 5: Send to LLM → receive response
Step 6: Save response (+ prompt for traceability)
```

---

## 2. Technology Choices

### 2.1 LLM Provider

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| **OpenAI API (GPT-4)** | Best reasoning, large context (128k) | Paid, data leaves local machine | Good for production |
| **Anthropic API (Claude)** | Strong reasoning, 200k context | Paid, data leaves local machine | Good for production |
| **Local model (Ollama/llama.cpp)** | Free, data stays local, offline | Weaker reasoning, slower, smaller context | Good for development/privacy |
| **Configurable** | Support all via adapter pattern | More code | **Chosen approach** |

**Decision**: Configurable provider with adapter pattern. Default to local model for development, cloud API for production/thesis evaluation.

### 2.2 Context Strategy

For a 5×5 map (25 neurons, ~500 samples): full per-neuron summary fits easily in any context window (~2-3k tokens).

For a 30×30 map (900 neurons, ~10k+ samples): per-neuron detail would exceed useful context. Strategy:
- **Group neurons into regions** (e.g., quadrants or U-matrix-based clusters)
- **Report only non-empty neurons** (skip dead neurons)
- **Top-N anomalies** instead of full extremes list
- **Summarize dimension stats** instead of per-neuron-per-dimension tables

### 2.3 Prompt Engineering vs Fine-Tuning

| Approach | Effort | Quality | Flexibility |
|----------|--------|---------|-------------|
| **Prompt engineering** | Low | Good enough | High — works with any LLM |
| Fine-tuning | High | Potentially better | Low — locked to one model |
| RAG | Medium | Good for large datasets | Medium |

**Decision**: Prompt engineering only. The SOM summary is structured enough that a well-crafted system prompt + context is sufficient. Fine-tuning would require many example reports we don't have. RAG adds complexity without clear benefit when the full context fits in the window.

---

## 3. Context Builder — What Must Be Computed

The SOM module currently outputs raw JSON files. The Context Builder transforms these into LLM-ready summaries.

### 3.1 From `clusters.json` + `original_input.csv`
- Per-neuron sample count
- Per-neuron dimension averages (mean of original values, not normalized)
- Cluster size distribution (largest, smallest, empty)

### 3.2 From `quantization_errors.json`
- Total map quality (MQE)
- Worst neurons (highest QE) — potential problem areas
- Dead neurons (QE = 0 with no samples)

### 3.3 From `extremes.json`
- Group extremes by type: global outliers vs neuron-local outliers
- Summarize: "5 samples are global outliers, 12 samples are local outliers"
- Top-N most extreme samples with full detail

### 3.4 From `pie_data_{col}.json`
- Per-neuron dominant category
- Purity: is each neuron 100% one category or mixed?
- Map regions: which part of the map is Benign vs Malignant?

### 3.5 From `preprocessing_info.json`
- Which columns were used, which were ignored (and why)
- Column types (numeric vs categorical)
- Missing value information

### 3.6 Computed (new)
- Inter-neuron distances from `weights_readable.csv` — cluster separation
- Overall map interpretation: "Left side = Benign, Right side = Malignant"

---

## 4. Example Output

Given the BreastCancer dataset, the generated report should read something like:

> **Dataset Summary**
>
> The analysis covers 569 breast cancer biopsy samples mapped onto a 5×5 hexagonal SOM.
> The map achieved an MQE of 0.569 with 5 dead neurons (20%).
>
> **Key Findings**
>
> The map shows clear separation between benign and malignant tumors.
> Neuron 1_0 contains 242 samples (43% of all data), all benign, forming the
> largest cluster. Malignant samples concentrate in neurons 0_3, 0_4, 1_3, 1_4,
> 2_3, 2_4 (upper-right region).
>
> **Anomalies**
>
> Sample 911296202 is the most extreme case — it holds the global maximum for
> area_mean (2501), radius_worst (36.04), perimeter_worst (251.20), and
> area_worst (4254). This sample is located in the malignant cluster (neuron 0_4)
> and represents an unusually large tumor.
>
> **Cluster Characteristics**
>
> Benign clusters (neurons 1_0, 2_1, 3_0): lower radius (avg 12.1), smaller area
> (avg 420), smoother boundaries (smoothness avg 0.09).
> Malignant clusters (neurons 0_3, 0_4, 1_4): higher radius (avg 18.5), larger
> area (avg 1050), more irregular shape (higher concavity).
