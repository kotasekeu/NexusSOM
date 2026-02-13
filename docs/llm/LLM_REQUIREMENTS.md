# LLM Module — The Voice — Requirements

Version: 1.0

---

## 1. Purpose

The Voice translates SOM analysis results into natural language. The LLM does not analyse raw data — it receives pre-processed structured summaries from the SOM/Data Mining pipeline and generates human-readable insights. The LLM must be constrained to talk only about the provided dataset and analysis results.

---

## 2. Functional Requirements

### FR-LLM-1: Input Data Preparation

#### FR-LLM-1.1 SOM Analysis Summary
Generate a structured text/JSON file from SOM results that the LLM can consume. Must include:
- [ ] Map overview: map size, topology, total samples, training quality (MQE, topographic error, dead ratio)
- [ ] Cluster summary: for each neuron — sample count, dominant category (from pie data), quantization error
- [ ] Extremes summary: outlier samples with human-readable explanations (from extremes.json)
- [ ] Dimension statistics: per-column global min, max, mean, std (from preprocessing_info.json + original data)
- [ ] Category distributions: per-neuron category counts (from pie_data_*.json)

#### FR-LLM-1.2 Dataset Context File
User-provided text file describing the dataset in natural language. Must include:
- [ ] Dataset name and domain (e.g., "Breast Cancer Wisconsin Diagnostic Dataset")
- [ ] What each row represents (e.g., "one tumor biopsy sample")
- [ ] Description of each dimension/column — name, what it measures, units, value range, domain meaning
- [ ] Description of categorical columns — what each category means (e.g., "B = Benign, M = Malignant")
- [ ] Any domain-specific context the LLM needs to generate meaningful insights (e.g., "higher radius_mean typically indicates malignant tumors")

#### FR-LLM-1.3 Analysis Context File
Auto-generated file combining FR-LLM-1.1 and FR-LLM-1.2 into a single LLM-ready context document.
- [ ] Merge SOM summary + dataset context into one structured prompt-ready text
- [ ] Include section markers so LLM can reference specific parts
- [ ] Keep total size within LLM context window limits

### FR-LLM-2: LLM Constraints

#### FR-LLM-2.1 Dataset Scope Lock
- [ ] LLM must only answer questions about the provided dataset and its SOM analysis
- [ ] LLM must refuse or redirect questions outside the dataset scope
- [ ] System prompt must enforce this boundary

#### FR-LLM-2.2 Grounded Responses
- [ ] All claims must reference specific data from the SOM analysis (neuron IDs, sample IDs, metric values)
- [ ] LLM must not invent data points or statistics
- [ ] When uncertain, LLM must state uncertainty explicitly

#### FR-LLM-2.3 Report Generation
- [ ] Generate structured report: summary → clusters → anomalies → patterns → recommendations
- [ ] Report must be reproducible — same input produces consistent structure
- [ ] Support different detail levels (brief summary vs full report)

### FR-LLM-3: Interaction Modes

#### FR-LLM-3.1 Report Mode
- [ ] One-shot: provide context, get full analysis report
- [ ] No conversation needed — suitable for automated pipeline output

#### FR-LLM-3.2 Conversational Mode
- [ ] User asks questions about the dataset and SOM results
- [ ] LLM answers using the provided context
- [ ] Follow-up questions possible within the same session

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

### 4.2 Missing Data (Must Be Created)

| Data | Source | Purpose |
|------|--------|---------|
| Per-neuron dimension means | Compute from original_input.csv + clusters.json | Cluster characterization |
| Cluster size distribution | Compute from clusters.json | Balance analysis |
| Inter-cluster distances | Compute from weights.csv | Cluster separation quality |
| Dataset context file | User-provided | Domain knowledge for interpretation |

---

## 5. Acceptance Criteria

- [ ] AC-1: Given SOM results + dataset context → LLM generates correct, grounded report
- [ ] AC-2: LLM refuses to answer questions unrelated to the provided dataset
- [ ] AC-3: Report contains no invented data — every claim traceable to input files
- [ ] AC-4: Context builder handles maps from 5×5 to 30×30 within context limits
- [ ] AC-5: Report is understandable by non-technical domain expert
- [ ] AC-6: System works with at least 2 different LLM providers
