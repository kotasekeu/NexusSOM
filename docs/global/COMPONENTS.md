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

### 6. Data Mining Module 🔍
**Role**: Automated anomaly and pattern detection

**Status**: Future Development 🔜

**Capabilities**:
- **Cluster analysis**: Identify distinct groups in data
- **Anomaly detection**: Find outliers and unusual patterns
- **Pattern discovery**: Detect trends, correlations, dependencies
- **Automated insights**: Generate findings without human guidance

**Integration**:
- Operates on trained SOM maps
- Uses cluster boundaries from U-Matrix
- Identifies anomalies from distance map
- Reports findings to LLM for interpretation

---

### 7. LLM "The Voice" 🗣️
**Role**: Natural language interpretation and reporting

**Status**: Future Development 🔜

**Capabilities**:
- **Translation**: Convert technical outputs to human language
- **Report generation**: Create understandable summaries
  - "The analysis found 5 distinct clusters..."
  - "Anomalies detected in region X represent 3% of data..."
  - "The map quality is excellent with low quantization error..."
- **Context integration**: Use user-provided dataset description
- **Recommendation**: Suggest next steps based on findings

**Integration**:
- Input: SOM metrics, Data Mining findings, visual map quality
- Output: Natural language report (markdown, PDF, HTML)
- Template-based generation with dynamic content

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
| **SOM (Dynamic)** | 2 | 🔜 Planned | P2 |
| **LSTM (Brain)** | 2 | 🔜 Planned | P2 |
| **MLP (Oracle)** | 2 | 🔜 Planned | P2 |
| **Data Mining** | 3 | 🔜 Future | P3 |
| **LLM (Voice)** | 3 | 🔜 Future | P3 |
| **UI** | 3 | 🔜 Future | P3 |
| **Database** | 3 | 🔜 Future | P3 |

---

**Document Version**: 1.0
**Last Updated**: January 2026
**Project**: NexusSOM Platform