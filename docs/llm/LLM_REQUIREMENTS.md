# LLM Module — The Voice — Requirements

**Verze**: 1.1  
**Aktualizováno**: 2026-05-11

---

## 1. Purpose

The Voice translates SOM analysis results into natural language. The LLM does not analyse raw data — it receives pre-processed structured summaries from the SOM/Data Mining pipeline and generates human-readable insights. The LLM must be constrained to talk only about the provided dataset and analysis results.

---

## 2. Functional Requirements

### FR-LLM-1: Input Data Preparation

#### FR-LLM-1.1 SOM Analysis Summary
Generate a structured text/JSON file from SOM results that the LLM can consume. Must include:
- [x] Map overview: map size, topology, total samples, training quality (MQE, topographic error, dead ratio)
- [x] Cluster summary: for each neuron — sample count, dominant category (from pie data), quantization error
- [x] Cluster purity per categorical column
- [x] Cluster dimension means (original unnormalized values) per numeric column
- [x] Extremes summary: outlier samples with human-readable explanations (from extremes.json)
- [x] Dimension statistics: per-column global min, max, mean, std (from preprocessing_info.json + original data)
- [x] Category distributions: per-neuron category counts (from pie_data_*.json)

**Implementace**: `app/analysis/` — viz `docs/llm/RESULT_ANALYZER.md`

#### FR-LLM-1.2 Dataset Context File
User-provided text file describing the dataset in natural language. Must include:
- [ ] Dataset name and domain (e.g., "Breast Cancer Wisconsin Diagnostic Dataset")
- [ ] What each row represents (e.g., "one tumor biopsy sample")
- [ ] Description of each dimension/column — name, what it measures, units, value range, domain meaning
- [ ] Description of categorical columns — what each category means (e.g., "B = Benign, M = Malignant")
- [ ] Any domain-specific context the LLM needs to generate meaningful insights (e.g., "higher radius_mean typically indicates malignant tumors")

#### FR-LLM-1.3 Analysis Context File
Auto-generated file combining FR-LLM-1.1 and FR-LLM-1.2 into a single LLM-ready context document.
- [x] Merge SOM summary + dataset context into one structured prompt-ready text
- [x] Include section markers so LLM can reference specific parts
- [x] Keep total size within LLM context window limits (~30 KB / ~7 400 tokenů pro 18×18, 3 000 vzorků)

### FR-LLM-2: LLM Constraints

#### FR-LLM-2.1 Dataset Scope Lock
- [x] LLM must only answer questions about the provided dataset and its SOM analysis
- [x] LLM must refuse or redirect questions outside the dataset scope
- [x] System prompt must enforce this boundary

#### FR-LLM-2.2 Grounded Responses
- [x] All claims must reference specific data from the SOM analysis (neuron IDs, sample IDs, metric values)
- [x] LLM must not invent data points or statistics
- [x] When uncertain, LLM must state uncertainty explicitly

#### FR-LLM-2.3 Report Generation
- [x] Generate structured report: summary → clusters → anomalies → patterns → recommendations
- [ ] Report must be reproducible — same input produces consistent structure (závisí na LLM temperature)
- [ ] Support different detail levels (brief summary vs full report)

### FR-LLM-3: Interaction Modes

#### FR-LLM-3.1 Report Mode
- [x] One-shot: provide context, get full analysis report
- [x] No conversation needed — suitable for automated pipeline output

#### FR-LLM-3.2 Conversational Mode
- [x] User asks questions about the dataset and SOM results
- [x] LLM answers using the provided context
- [x] Follow-up questions possible within the same session

### FR-LLM-4: Output Format

#### FR-LLM-4.1 Text Report
- [ ] Markdown formatted report with sections and subsections
- [ ] Tables for cluster comparisons and statistics
- [ ] Clear language accessible to non-technical users

#### FR-LLM-4.2 Structured Data
- [ ] JSON output option for programmatic consumption
- [ ] Key findings as structured objects (cluster descriptions, anomaly flags, patterns)

---

## 3. Non-Functional Requirements

### NFR-LLM-1: LLM Provider
- [ ] Must work with multiple LLM backends (OpenAI API, local models, Anthropic API)
- [ ] Configuration-based provider selection
- [ ] API key management via environment variables

### NFR-LLM-2: Context Window
- [ ] SOM summary must fit within LLM context window
- [ ] For large maps (30×30 = 900 neurons), summarize by regions rather than per-neuron
- [ ] Fallback: chunked processing for datasets exceeding context limits

### NFR-LLM-3: Reproducibility
- [ ] Temperature = 0 for report generation (deterministic output)
- [ ] System prompt and context saved alongside output for traceability

---

## 4. Data Flow

```
SOM Results (JSON/CSV)  ──┐
                          ├──→ Context Builder ──→ LLM Prompt ──→ LLM ──→ Report
Dataset Context (TXT)   ──┘
```

### 4.1 Available SOM Outputs (Current)

| File | Content | LLM Use |
|------|---------|---------|
| `clusters.json` | neuron_key → [sample_ids] | Cluster composition, sizes |
| `quantization_errors.json` | total QE + per-neuron QE | Map quality, problem areas |
| `extremes.json` | sample_id → [reasons] | Anomaly explanations |
| `pie_data_{col}.json` | per-neuron category counts | Category distribution per cluster |
| `preprocessing_info.json` | column metadata (type, nunique, status) | Dimension descriptions |
| `original_input.csv` | raw data | Reference for specific sample values |
| `training_data_readable.csv` | normalized data | Not needed for LLM |

### 4.2 Implementovaná data (app/analysis/)

| Data | Zdroj | Stav |
|------|--------|------|
| Per-neuron dimension means + median + std + min + max | original_input.csv + clusters.json | ✅ |
| Global dimension statistics (incl. percentiles p25/p75/p90/p95) | original_input.csv | ✅ |
| Global category distributions | pie_data_*.json | ✅ |
| Cluster size distribution | clusters.json | ✅ |
| Cluster purity per categorical column | pie_data_*.json | ✅ |
| Cluster category counts per value | pie_data_*.json | ✅ |
| Cluster Z-score deviation from global mean | original_input.csv | ✅ |
| Map topology metrics (Gini, coverage ratio, dead neurons) | clusters.json + weights.npy | ✅ |
| Local numeric outliers (z-score >2.5σ per cluster) | original_input.csv | ✅ |
| Multi-dimensional outliers (outlier on ≥2 dims) | original_input.csv | ✅ |
| "1 of N" isolated outlier per cluster | original_input.csv | ✅ |
| Global extremes enrichment | extremes.json | ✅ |
| Inter-cluster distances | weights.npy | ❌ (budoucí) |
| Map region spatial summary | cluster positions + categories | ❌ (budoucí) |
| Dataset context file | User-provided | — |

---

## 5. Acceptance Criteria

- [ ] AC-1: Given SOM results + dataset context → LLM generates correct, grounded report
- [ ] AC-2: LLM refuses to answer questions unrelated to the provided dataset
- [ ] AC-3: Report contains no invented data — every claim traceable to input files
- [ ] AC-4: Context builder handles maps from 5×5 to 30×30 within context limits
- [ ] AC-5: Report is understandable by non-technical domain expert
- [ ] AC-6: System works with at least 2 different LLM providers
