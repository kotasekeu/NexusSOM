# NexusSOM - System Components

**Component Architecture and Responsibilities**

---

## Core Components (Priority 1 - Current Implementation)

### 1. Self-Organizing Map (SOM)
**Role**: Primary analytical engine for data exploration and visualization

#### Phase 1: Statically-Configured Hybrid SOM ✅
**Status**: Implemented

**Capabilities**:
- Multiple training modes: `stochastic`, `deterministic`, `hybrid`
- Flexible topology support: hexagonal, square grids
- Ignore mask for selective feature processing
- Early stopping with configurable patience
- Comprehensive metrics: MQE, Topographic Error, Dead Neuron Ratio
- History logging for all training steps
- Multi-format output: `.npy`, `.csv`, visualization maps

**Key Features**:
- **Hybrid Training**: Progressively larger batches from distinct dataset sections
- **Adaptive Scheduling**: Dynamic learning rate, radius, and batch size decay
- **Reproducibility**: Fixed random seed support for deterministic results

#### Phase 2: Dynamically-Controlled SOM 🔜
**Status**: Planned (Fáze 5)

**Capabilities**:
- Real-time parameter adjustment via external controller ("The Brain")
- Dynamic parameter multipliers (learning rate, radius, batch size)
- Feedback loop with metric recording
- Delegated stopping decisions to controller
- Baseline schedule with dynamic adjustments

---

### 2. Evolutionary Algorithm (EA)
**Role**: Autonomous hyperparameter optimization and data generation

#### Phase 1: Hyperparameter Optimizer ✅
**Status**: Implemented

**Capabilities**:
- **Multi-objective optimization** (NSGA-II):
  - Minimize: `best_mqe`, `duration`, `topographic_error`, `dead_neuron_ratio`
- **Mixed search space**:
  - Categorical parameters (from list)
  - Continuous float parameters (min-max range)
  - Integer parameters (min-max range)
- **Advanced genetic operators**:
  - Simulated Binary Crossover (SBX)
  - Polynomial Mutation
- **Parallel evaluation**: Multi-core processing
- **Comprehensive output**:
  - Individual directories with UID
  - Training graphs, SOM maps (U-Matrix, Distance, Dead Neurons)
  - Centralized `results.csv` with all hyperparameters and metrics
  - Pareto front logging

**Data Generation**:
- Creates diverse dataset of SOM configurations
- Generates RGB multi-channel maps for CNN training
- Produces quality metrics for supervised learning

#### Phase 2: CNN-Guided EA 🔄
**Status**: In Progress (Fáze 2-3)

**Capabilities**:
- **CNN Integration**:
  - Generate 3-channel RGB maps
  - Query CNNQualityEvaluator for visual quality score
  - Augmented fitness with `cnn_quality_score`
- **Oracle Integration**:
  - Query MLP for optimal starting configuration
  - Inject recommended config into initial population
  - Accelerate convergence with smart initialization
- **UID Deduplication**:
  - Skip re-evaluation of identical configurations
  - Check existing `uid` directories and `results.csv` entries

---

### 3. CNN "The Eye" 👁️
**Role**: Visual quality assessment of SOM maps

#### Current Status: In Development (Fáze 2) 🔄

**Capabilities**:
- **Multi-channel input**: RGB images (224x224x3)
  - R channel: U-Matrix (topological structure)
  - G channel: Distance Map (quantization error)
  - B channel: Dead Neurons Map (neuron activity)
- **Quality score prediction**: Regression output (0-1 range)
- **Training data**: EA-generated maps with calculated quality scores
- **Quality formula**:
  ```
  quality_score = 0.5 × (1 - norm_mqe) +
                  0.3 × (1 - norm_te) +
                  0.2 × (1 - norm_dead_ratio)
  ```

**Architecture**:
- Standard CNN: 4 conv blocks (~5M parameters)
- Lightweight CNN: 3 conv blocks (~500K parameters)
- Output: Single regression neuron with sigmoid activation

**Training Strategy**:
1. **Phase 1**: Small dataset (30-50 maps) - proof of concept
2. **Phase 2**: Large dataset (10,000 maps from 20 datasets)
3. **Phase 3**: Continuous improvement with human feedback

**Integration Points**:
- Input: RGB PNG from `maps_dataset/rgb/`
- Output: Quality score for EA fitness augmentation
- Feedback: Guide EA evolution towards visually better maps

---

## AI Control Layer (Priority 2 - Future Development)

### 4. LSTM "The Brain" 🧠
**Role**: Real-time dynamic control of SOM training

**Status**: Planned (Fáze 5) 🔜

**Capabilities**:
- **Real-time monitoring**: Track MQE, TE, dead neuron ratio during training
- **Dynamic parameter adjustment**: Compute multipliers for:
  - Learning rate factor
  - Radius factor
  - Batch size factor
- **Adaptive stopping**: Decide when training has converged
- **State-based decisions**: [current_state] → [optimal_action]

**Training Data**:
- Historical training runs from EA
- Sequences: `[metrics_history] → [parameter_adjustments]`
- Labels: Final quality scores, convergence speed

**Integration**:
- Injected into SOM via `controller` parameter
- Methods: `get_lr_factor()`, `get_radius_factor()`, `should_stop()`
- Feedback: `record_metrics(iteration, mqe, te, etc.)`

---

### 5. MLP "The Oracle" 🔮
**Role**: Intelligent initialization based on dataset characteristics

**Status**: Planned (Fáze 5) 🔜

**Capabilities**:
- **Dataset analysis**: Extract meta-features
  - Number of samples, dimensions
  - Data distribution (mean, std, skewness)
  - Correlation structure, PCA variance
- **Configuration prediction**: Recommend optimal starting hyperparameters
  - Map size (m × n)
  - Learning rate schedule
  - Radius schedule
  - Training mode preference

**Training Data**:
- Pairs: `[dataset_features] → [best_configuration]`
- Source: Successful EA runs across diverse datasets
- Labels: Pareto-optimal configurations

**Integration**:
- Called before EA initialization
- Injects recommended config into initial population
- Accelerates convergence by skipping random exploration phase

---

## Application Layer (Priority 3 - Productization)

### 6. Analysis Module 🔍
**Role**: Automated statistics and anomaly detection on SOM results

**Status**: Implemented ✅

**Capabilities**:
- **Global statistics**: min, max, mean, std, percentiles (p25/p75/p90/p95) per numeric column
- **Category distributions**: global class balance per categorical column
- **Cluster statistics**: mean, median, std, min, max + purity + category counts per neuron
- **Cluster Z-score deviation**: how far each cluster's mean is from the global mean
- **Map topology**: active/dead neurons, Gini coefficient, coverage ratio
- **Local outliers**: Z-score-based (>2.5σ from cluster mean) + multi-dimensional outlier detection
- **"1 of N" pattern**: isolated sample significantly farther from cluster centroid than peers
- **Global extremes enrichment**: links extremes.json entries to cluster context

**Structure**:
```
app/analysis/
├── src/loader.py      ← IO only, no computation
├── src/stats.py       ← pure statistics
├── src/anomalies.py   ← anomaly detection
└── src/context.py     ← assembles llm_context.json
```

**Integration**:
- Called automatically by `run.py` after `perform_analysis()`
- Can be run standalone: `python3 app/run_analysis.py -i <results_dir>`
- `context_builder.py` calls it as fallback when `llm_context.json` is missing
- Output: `<results_dir>/json/llm_context.json`

---

### 7. LLM "The Voice" 🗣️
**Role**: Natural language interpretation and reporting

**Status**: Implemented ✅

**Capabilities**:
- **Translation**: Convert technical SOM outputs to human language
- **Report mode**: One-shot full analysis report (streamed to terminal + saved as `report.md`)
- **Chat mode**: Interactive Q&A about the dataset and SOM results
- **Context integration**: Uses `dataset_context.txt` (or `ABOUT.MD` fallback) for domain knowledge
- **Remote model support**: Works with local Ollama or remote GPU server via `--url`

**Integration**:
- Input: `llm_context.json` (from Analysis Module) + `dataset_context.txt`
- Output: `<results_dir>/llm/report.md` + `prompt_log.json`
- Entry point: `python3 app/run_llm.py -i <results_dir> -m report|chat --model llama3.1:8b`
- Docs: `docs/llm/RUN.md`

---

### 8. User Interface 💻
**Role**: Interactive visualization and control

**Status**: Future Development 🔜

**Capabilities**:
- **Data upload**: Drag-and-drop CSV/Excel support
- **Context input**: Simple text description of data
- **One-click analysis**: Automated end-to-end processing
- **Interactive maps**: Zoom, pan, hover tooltips
- **Report viewing**: Integrated LLM-generated reports
- **Export**: Download maps, data, reports

**Technology Stack**:
- Frontend: React/Vue.js
- Backend: FastAPI/Flask
- Visualization: D3.js, Plotly

---

### 9. Database Layer 🗄️
**Role**: Persistent storage for production deployment

**Status**: Future Development 🔜

**Capabilities**:
- **Metadata storage**: PostgreSQL for configurations, metrics
- **Run history**: Track all analyses, configurations, results
- **Vector storage**: Efficient similarity search (pgvector, Pinecone)
- **File storage**: S3/MinIO for maps and artifacts

**Schema**:
- `datasets`: Uploaded data metadata
- `ea_runs`: EA execution history
- `som_configurations`: Hyperparameter sets
- `evaluations`: Individual SOM training results
- `maps`: Generated visualization artifacts

---

### 10. Growing SOM 📈
**Role**: Autonomous map size optimization

**Status**: Future Research 🔜

**Capabilities**:
- **Dynamic topology**: Start small, grow as needed
- **Node insertion**: Add neurons in high-error regions
- **Node deletion**: Remove underutilized neurons
- **Automatic sizing**: Eliminate manual `map_size` tuning

**Integration**:
- Extends Phase 2 Dynamic SOM
- Works with "The Brain" for growth decisions
- Further reduces manual hyperparameter selection

---

## System Integration Flow

### Current State (Phase 1-2):
```
User → CSV Data
  ↓
EA (Static Hyperparameter Search)
  ↓
Parallel SOM Training
  ↓
Maps + Metrics → CNN Training
  ↓
Results.csv + RGB Maps
```

### Target State (All Components Integrated):
```
User → UI: Upload Data + Context
  ↓
Oracle: Analyze Dataset → Recommend Config
  ↓
EA: Initialize with Smart Config
  ↓
Brain: Real-time SOM Control
  ↓
Eye: Visual Quality Assessment
  ↓
Data Mining: Find Patterns & Anomalies
  ↓
LLM: Generate Human Report
  ↓
UI: Interactive Visualization + Report
  ↓
Database: Store Results
```

---

## Component Status Summary

| Component | Phase | Status | Priority |
|-----------|-------|--------|----------|
| **SOM (Static)** | 1 | ✅ Implemented | P1 |
| **EA (Optimizer)** | 1 | ✅ Implemented | P1 |
| **CNN (Eye)** | 1-2 | 🔄 In Progress | P1 |
| **EA + CNN** | 2 | 🔄 In Progress | P1 |
| **Analysis Module** | 2 | ✅ Implemented | P2 |
| **LLM (Voice)** | 2 | ✅ Implemented | P2 |
| **SOM (Dynamic)** | 2 | 🔜 Planned | P2 |
| **LSTM (Brain)** | 2 | 🔜 Planned | P2 |
| **MLP (Oracle)** | 2 | 🔜 Planned | P2 |
| **UI** | 3 | 🔜 Future | P3 |
| **Database** | 3 | 🔜 Future | P3 |

---

**Document Version**: 1.0
**Last Updated**: January 2026
**Project**: NexusSOM Platform