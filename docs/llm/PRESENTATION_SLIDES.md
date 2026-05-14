# NexusSOM — The Voice: LLM Analysis Layer

---

## What Is "The Voice"

The LLM layer is a **translation layer**, not an AI analyst.

- Takes structured SOM output → produces natural language insight
- LLM never sees raw data — only pre-computed statistical summaries
- Constrained by system prompt to stay grounded in the provided data
- Every claim must reference a specific neuron, sample ID, or metric value

> Goal: a domain expert with no SOM knowledge reads the report and gains actionable insight.

---

## Pipeline Overview

```
dataset.csv
    ↓
SOM Training (run_som.py)
    ↓  weights.npy, clusters.json, quantization_errors.json, extremes.json
Statistical Analysis (app/analysis/)
    ↓  llm_context.json  ~450 KB, ~7 400 tokens
Context Builder (context_builder.py)
    ↓  structured text prompt (SOM stats + domain description)
LLM Inference (Ollama / API)
    ↓  streaming response
report.md  +  report.pdf
```

---

## Two-Context Architecture

**Context A — Dataset Description** *(user-provided, static)*
- Plain text file: `dataset_context.txt`
- What each column means, units, value ranges
- Domain knowledge (e.g. "SMOKING=2 means symptom present")
- Categorical label decoding (binary 1/2 → absent/present)

**Context B — SOM Analysis Summary** *(auto-generated)*
- Map quality: MQE, topographic error, dead neuron ratio
- Per-cluster: dominant category, purity, dimension means/std, Z-score deviation
- Top-20 anomalies with full row values + delta from mean
- Global dimension statistics: min, max, mean, std, p25–p95

---

## What Is Computed — `app/analysis/`

| Area | Implemented |
|---|---|
| Global stats: min, max, mean, std, median, p25–p95 | ✅ |
| Global category distributions | ✅ |
| Per-cluster: sample count, QE, dominant category, purity | ✅ |
| Per-cluster: mean, median, std, min, max per dimension | ✅ |
| Per-cluster Z-score deviation from global mean | ✅ |
| Map topology: coverage ratio, Gini coefficient, dead neurons | ✅ |
| Anomaly type: local numeric outlier (>2.5σ from cluster mean) | ✅ |
| Anomaly type: multi-dimensional outlier (≥2 dimensions) | ✅ |
| Anomaly type: isolated "1-of-N" (>3× median distance from centroid) | ✅ |
| Anomaly records: full row + Δ from mean + severity flags | ✅ |
| Distribution skewness, kurtosis | ❌ |
| Feature correlations | ❌ |
| Boundary samples (near-equal distance to two neurons) | ❌ |
| Cluster health score (compactness × separation) | ❌ |
| Spatial map regions (flood-fill by dominant category) | ❌ |

---

## Anomaly Records — Full Row Detail

Each top anomaly includes the complete sample row with annotations:

```
AGE=30.0 [-25.17 / -45.6%] [GLOBAL_MIN]
SMOKING=2 [cluster_dominant=1]
CHEST_PAIN=2 [+0.42 / +26.3%] [MODERATE_DEVIATION]
LUNG_CANCER=YES [cluster_dominant=NO]
```

Severity flags:
- **GLOBAL_MIN / GLOBAL_MAX** — sample holds global extreme value
- **HIGH_DEVIATION** — >50% delta from global mean
- **MODERATE_DEVIATION** — >20% delta from global mean
- **differs_from_cluster_dominant** — categorical value differs from neuron's majority

---

## `llm_context.json` — Size & Scope

Target: fit within local 8B model context window (≤16 384 tokens)

| Section | Tokens (est.) |
|---|---|
| Map overview | ~50 |
| Clusters (324 neurons × summary) | ~4 200 |
| Anomalies (top-20 + full records) | ~2 100 |
| Global dimension stats | ~200 |
| Category distributions | ~100 |
| **Total** | **~6 650** |

Optimizations applied: removed `category_counts` per-neuron bloat, removed `cluster_local_outliers` list → saved ~730 KB (1 181 KB → 454 KB).

---

## Output Modes

### Report Mode
```bash
python3 app/run_llm.py -i results/20260512_133509 -m report
```
- One-shot: full analysis streamed to terminal
- Saved: `llm/report.md` + `llm/prompt_log.json` (full prompt for traceability)

### Chat Mode
```bash
python3 app/run_llm.py -i results/20260512_133509 -m chat
```
- Interactive Q&A with full SOM context loaded
- LLM constrained to provided dataset — refuses off-topic questions
- Session memory: follow-up questions build on previous answers

### PDF Mode
```bash
python3 app/run_llm.py -i results/20260512_133509 -m pdf
```
- Combines `report.md` + all SOM visualizations into a single PDF
- Includes: cover page, report text, dimension stats table, cluster table, anomaly list
- Visual pages: U-Matrix, distance map, hit map, component planes, category pie maps
- File size: ~7.6 MB (25 pages), images downscaled to 700 px via PIL before embedding

---

## LLM Providers

| Provider | Context | Privacy | Quality |
|---|---|---|---|
| **Ollama (local)** — `llama3.1:8b` | 16 k tokens | Full — no data leaves machine | ⭐⭐⭐ |
| **Ollama (remote GPU)** — `qwen2.5:32b` | 32 k tokens | Local network only | ⭐⭐⭐⭐⭐ |
| **Ollama (remote GPU)** — `llama3.1:70b` | 128 k tokens | Local network only | ⭐⭐⭐⭐⭐ |
| **Anthropic API (Claude)** | 200 k tokens | Cloud | ⭐⭐⭐⭐⭐ |
| **OpenAI API (GPT-4o)** | 128 k tokens | Cloud | ⭐⭐⭐⭐⭐ |

Current default: `llama3.1:8b` via Ollama on localhost.  
Custom model `nexusom-analyst` — llama3.1:8b base with baked-in system prompt, `temperature=0`.

---

## What You Need to Prepare

Before running LLM analysis on a new dataset:

1. **`dataset_context.txt`** in the dataset directory
   - Column descriptions, units, value ranges
   - Category label decoding
   - Domain knowledge hints
   - Without this: LLM reports raw codes (`SMOKING=1`) without domain meaning

2. **SOM run** with `run_som.py`
   - Automatically generates `llm_context.json` since v1.0
   - For older runs: `python3 app/run_analysis.py -i results/<timestamp>`

3. **Ollama running** (or API key in env)
   - `ollama serve` + `ollama pull llama3.1:8b`

---

## What Is Done ✅

- `app/analysis/` — independent statistical post-processing module
- `llm_context.json` — structured context, ~450 KB, fits 8B context window
- Anomaly detection: numeric outliers, multi-dim outliers, 1-of-N isolated
- Full anomaly records with delta annotations (PHP-style severity flags)
- `run_llm.py` — unified CLI for report / chat / PDF modes
- `pdf_builder.py` — 25-page PDF with all visualizations + stat tables
- `nexusom-analyst` custom Ollama model (baked system prompt)
- Remote GPU inference via `--url` flag
- `dataset_context.txt` auto-discovery (4 levels up + ABOUT.md fallback)

---

## What We Can Do Next

### Priority 1 — Better LLM Grounding
- **Feature correlations (A4)** — "AGE and SMOKING are 0.71 correlated" enables richer cluster narratives
- **Distribution shape (A3)** — skewness + kurtosis flags non-normal dimensions for the LLM
- **Cluster entropy (B6)** — categorical homogeneity score alongside purity
- **IQR-based outliers (C3)** — Tukey method, robust for skewed distributions

### Priority 2 — Spatial Awareness
- **Spatial map regions (D5)** — flood-fill neighboring neurons by dominant category → "left half = NO cancer"
- **Neighbor consistency (D3)** — U-matrix numerically → quantify cluster boundary sharpness
- **Boundary samples (C8)** — samples equidistant to two neurons → ambiguous classification signal

### Priority 3 — Output Quality
- **Multi-provider support** — Anthropic / OpenAI API adapter alongside Ollama
- **Chunked processing** — for maps >30×30 where full per-neuron detail exceeds context
- **Structured JSON output** — machine-readable findings alongside narrative report
- **Temperature=0 enforcement** — reproducible report generation

---

## Architecture Summary

```
app/
├── analysis/          ← statistics only, no SOM dependency
│   └── src/
│       ├── loader.py      IO — reads result files
│       ├── stats.py       global stats, cluster stats, topology
│       ├── anomalies.py   outlier detection (numeric, multi-dim, 1-of-N)
│       └── context.py     assembles llm_context.json
├── llm/
│   └── src/
│       ├── context_builder.py   prompt assembly (Context A + B)
│       ├── llm_client.py        Ollama HTTP adapter
│       ├── report_generator.py  report + chat + PDF orchestration
│       └── pdf_builder.py       fpdf2 PDF with visualizations
├── run_llm.py         CLI entry point
└── run_analysis.py    standalone llm_context.json regeneration
```
