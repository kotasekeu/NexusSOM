# NexusSOM Platform
## Intelligent Self-Organizing Maps with AI-Driven Optimization

**Version**: 1.0
**Date**: February 2, 2026
**Status**: Phase 1 Complete, Phase 2 In Progress

---

## Project Vision

**Goal**: Fully automated SOM hyperparameter optimization using multi-agent AI system

**Problem Solved**:
- Manual SOM hyperparameter tuning is time-consuming
- Quality assessment requires expert knowledge
- No real-time adaptation during training

**Solution**: AI agents working together to optimize, evaluate, and control SOM training

---

## System Architecture

### Multi-Agent AI System

```
Data Input → Oracle (MLP) → EA Optimizer ↔ The Eye (CNN)
                               ↓
                          SOM Training ↔ The Brain (LSTM)
                               ↓
                          Results → Data Mining → The Voice (LLM) → User Report
```

---

## Core Components

### 1. Self-Organizing Map (SOM) - **DONE** ✅

**Purpose**: Core unsupervised learning algorithm for data visualization and clustering

**Status**: Phase 1 Complete (100%) | **Future Focus**: Hybrid mode with hexagonal topology

#### How It Works

**Training Algorithm**:
1. **Initialization**: Random weight vectors for each neuron in grid
2. **Sample Selection**: Pick data sample(s) from dataset
3. **Best Matching Unit (BMU)**: Find neuron with closest weight vector (Euclidean distance)
4. **Neighborhood Update**: Update BMU and neighbors based on:
   - Learning rate (α): Controls update magnitude
   - Neighborhood radius (σ): Defines influence area
   - Distance from BMU: Gaussian decay function
5. **Weight Update Formula**: `w_new = w_old + α × h(d) × (x - w_old)`
   - `h(d)`: Neighborhood function (Gaussian)
   - `d`: Distance from BMU
6. **Decay Schedules**: Learning rate and radius decrease over epochs
7. **Convergence**: Stop when MQE improvement < threshold for N epochs

**Training Modes**:
- **Stochastic**: Update per single sample (noisy but explores more)
- **Deterministic**: Update per full dataset pass (stable but slower)
- **Hybrid** ⭐ **(Future Focus)**: Progressive batch size growth (best of both worlds)
  - Starts with small batches (exploration phase)
  - Grows to larger batches (convergence phase)
  - Combines benefits of both modes

**Topology**:
- **Square**: 4 neighbors (Manhattan distance)
- **Hexagonal** ⭐ **(Future Focus)**: 6 neighbors, better circular symmetry
  - More natural neighborhood structure
  - Better topology preservation
  - Uses axial coordinates for hex geometry

**Quality Metrics**:
- **MQE** (Mean Quantization Error): Average distance from samples to BMU
- **Topographic Error**: % samples with non-adjacent 1st and 2nd BMU
- **Dead Neurons**: % neurons never activated (indicates poor map utilization)

**Adaptive Scheduling**:
- Learning rate: Exponential/linear/step decay
- Radius: Exponential/linear/step decay
- Batch size: Linear/exponential growth in hybrid mode

---

### 2. Evolutionary Algorithm (EA) - **DONE** ✅

**Purpose**: Multi-objective hyperparameter optimization using NSGA-II

**Status**: Phase 1 Complete (100%)

#### How It Works

**NSGA-II Algorithm** (Non-dominated Sorting Genetic Algorithm):

1. **Initialization**:
   - Generate random population (N individuals)
   - Each individual = complete SOM configuration
   - Configurable via JSON search space

2. **Evaluation**:
   - Train SOM with each configuration
   - Measure 4 objectives:
     - **Minimize MQE**: Quality of data representation
     - **Minimize Duration**: Training time efficiency
     - **Minimize Topographic Error**: Topology preservation
     - **Minimize Dead Neuron Ratio**: Map utilization
   - Parallel processing across CPU cores

3. **Non-dominated Sorting**:
   - Rank individuals into fronts (F1, F2, F3...)
   - F1 = Pareto front (no individual dominates these)
   - Individual A dominates B if: A is better in ≥1 objective AND not worse in any

4. **Crowding Distance**:
   - Calculate diversity metric within each front
   - Prefer individuals in less crowded regions
   - Maintains solution diversity

5. **Selection**:
   - Tournament selection based on:
     - Rank (lower front = better)
     - Crowding distance (higher = better if same front)

6. **Genetic Operators**:
   - **Simulated Binary Crossover (SBX)**:
     - Blends two parents to create offspring
     - Distribution index (η) controls spread
     - Respects variable bounds
   - **Polynomial Mutation**:
     - Small random perturbations
     - Mutation probability per gene
     - Maintains feasibility

7. **Survival Selection**:
   - Combine parents + offspring
   - Select best N individuals for next generation
   - Elitism: F1 always survives

**Search Space Configuration**:
- Categorical parameters: Random selection from list
- Continuous parameters: Bounded float ranges
- Integer parameters: Bounded integer ranges
- Example: `learning_rate: [0.01, 1.0]`, `map_size: [[5,5], [10,10], [20,20]]`

**Output Generation**:
- Each individual gets unique UID (MD5 hash of config)
- Directory per individual: visualizations, weights, history
- Centralized `results.csv`: All metrics + hyperparameters
- RGB maps for CNN training (3 channels: U-matrix, Distance, Dead neurons)

---

### 3. CNN "The Eye" 👁️ - **77% COMPLETE** 🔄

**Purpose**: Visual quality assessment of SOM maps through deep learning

**Status**: Phase 1 - 77% complete (10/13 requirements)

#### How It Works

**Problem**: Assess SOM quality visually without computing expensive metrics

**Solution**: Train CNN to predict quality score from SOM visualization

**Input Processing**:
- **3-channel RGB image** (variable size: 5×5 to 30×30 pixels)
  - **R channel**: U-Matrix (neuron distances, topology quality)
  - **G channel**: Distance Map (quantization error per neuron)
  - **B channel**: Dead Neurons Map (activation frequency)
- **1 neuron = 1 pixel** (native resolution, no interpolation)
- Normalize to [0, 1] range

**Architecture** (Global Average Pooling Design):

```
Input (H, W, 3) - variable size
    ↓
Conv Block 1: 2×Conv(32) + BN + ReLU + Dropout(0.25)
    ↓ (padding='same', no size change)
Conv Block 2: 2×Conv(64) + BN + ReLU + Dropout(0.25)
    ↓ (padding='same', no size change)
Conv Block 3: 2×Conv(128) + BN + ReLU + Dropout(0.3)
    ↓ (padding='same', no size change)
Conv Block 4: 2×Conv(256) + BN + ReLU + Dropout(0.3)
    ↓ (any size → 256 features per pixel)
GlobalAveragePooling2D: (H, W, 256) → (256,)
    ↓ (size-independent feature vector)
Dense(256) + BN + ReLU + Dropout(0.5)
    ↓
Dense(128) + BN + ReLU + Dropout(0.5)
    ↓
Dense(1) + Sigmoid → Quality Score [0, 1]
```

**Key Design Decisions**:
- **No MaxPooling**: Preserves spatial information for small maps (5×5)
- **padding='same'**: Maintains dimensions through convolutions
- **GAP (Global Average Pooling)**: Handles any input size → fixed 256 features
- **Regression output**: Single neuron with sigmoid (0=bad, 1=good)
- **Loss**: MSE (Mean Squared Error) for continuous quality prediction

**Training Pipeline**:

1. **Auto-Labeling** (Threshold-based):
   ```python
   if dead_neuron_ratio > 0.30:     label = 0.1  # BAD
   if topographic_error > 0.50:     label = 0.1  # BAD
   if dead < 0.05 AND topo < 0.10:  label = 0.9  # GOOD
   else:                             label = None # UNCERTAIN
   ```

2. **GroupedSizeDataLoader**:
   - Groups images by size (5×5, 10×10, 15×15, etc.)
   - Batches contain only same-sized images
   - Avoids interpolation artifacts
   - Efficient GPU utilization

3. **Data Augmentation** (minimal):
   - Horizontal flip (50% probability)
   - Vertical flip (50% probability)
   - NO rotation/zoom/color (would distort topology)

4. **Training**:
   - Optimizer: Adam (lr=0.001)
   - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
   - Metrics: MAE, MSE, RMSE
   - Validation split: 20%

5. **Pseudo-Labeling** (Iterative improvement):
   ```
   Train on auto-labeled → Predict on uncertain cases
   → Add confident predictions (>0.65 or <0.35)
   → Retrain with expanded dataset → Repeat
   ```

**What's Next**:
- ⏸️ First training run (884 labeled samples ready)
- ⏸️ Inference script for prediction
- ⏸️ EA integration (add CNN score to fitness objectives)

---

### 4. LSTM "The Brain" 🧠 - **TBA** 🔜

**Purpose**: Real-time SOM training controller

**Status**: Phase 2 - To Be Announced

**Planned Features**:
- Real-time monitoring of training metrics
- Dynamic parameter adjustment (learning rate, radius, batch size)
- Adaptive stopping decisions
- State-based optimization using LSTM memory

**Integration**: Controller interface for SOM Phase 2

**Training Data**: Historical EA training sequences

---

### 5. MLP "The Oracle" 🔮 - **TBA** 🔜

**Purpose**: Intelligent initial configuration prediction

**Status**: Phase 2 - To Be Announced

**Planned Features**:
- Dataset analysis (samples, dimensions, correlations)
- Optimal map size prediction
- Learning rate schedule recommendation
- Training mode preference

**Integration**: Pre-EA initialization with smart starting point

**Training Data**: Successful EA run configurations

---

### 6. Data Mining Module 🔍 - **TBA** 🔜

**Purpose**: Automated insight discovery from SOM results

**Status**: Future Development

**Planned Features**:
- Cluster identification and analysis
- Anomaly detection
- Pattern discovery
- Automated findings ranking

---

### 7. LLM "The Voice" 🗣️ - **TBA** 🔜

**Purpose**: Natural language report generation

**Status**: Future Development

**Planned Features**:
- Technical output → Natural language translation
- Context-aware insights
- Domain-specific recommendations
- Understandable summaries for non-experts

---

### 8. User Interface 💻 - **TBA** 🔜

**Purpose**: One-click data analysis

**Status**: Future Development

**Planned Features**:
- Drag-and-drop data upload (CSV, Excel)
- Interactive map visualization
- Progress tracking with time estimates
- Export capabilities (PNG, SVG, PDF, JSON)

---

## Current Project Status

### Phase 1: Core Infrastructure ✅
- **SOM Training**: 100% Complete
- **EA Optimization**: 100% Complete
- **CNN Data Pipeline**: 100% Complete
- **CNN Model**: 100% Complete

### Phase 2: AI Integration 🔄
- **CNN Training**: Ready (0% trained)
- **EA-CNN Integration**: Not Started
- **LSTM Controller**: Not Started
- **MLP Oracle**: Not Started

### Phase 3: User Experience 🔜
- **Data Mining**: Not Started
- **LLM Report Generation**: Not Started
- **Web UI**: Not Started

**Overall Progress**: ~40% Complete

---

## System Flow (Target)

### Complete AI-Driven Pipeline 🔜

```
1. Load Dataset
   ↓
2. Oracle (MLP) Analysis
   - Analyze dataset statistics (samples, dims, correlation)
   - Predict optimal starting config
   - Initialize EA with smart population
   ↓
3. EA with CNN Guidance
   - Generate SOM configs
   - CNN predicts visual quality (fast)
   - Add CNN score to fitness objectives
   - Faster convergence to good solutions
   ↓
4. SOM Training with LSTM Control
   - LSTM monitors metrics in real-time
   - Adjusts learning rate dynamically
   - Adjusts radius dynamically
   - Decides optimal stopping point
   ↓
5. Data Mining
   - Identify clusters automatically
   - Detect anomalies
   - Find patterns and correlations
   ↓
6. LLM Report Generation
   - Translate technical metrics to natural language
   - Context-aware insights
   - Actionable recommendations
   ↓
7. Interactive UI
   - User views results
   - Explores maps interactively
   - Exports findings
```

---

## Key Innovations

### 1. Variable Input Size CNN (GAP Architecture)
**Problem**: SOMs can be 5×5 to 30×30 neurons - how to train one CNN for all sizes?

**Traditional Approach**: Resize all images to fixed size (e.g., 224×224)
- Creates interpolation artifacts
- Loses spatial information for small maps
- CNN might learn size instead of quality

**Our Solution**: Global Average Pooling
- No MaxPooling → preserves small map details
- All Conv layers use padding='same' → maintains dimensions
- GAP at end: (H, W, 256) → (256) regardless of H, W
- Single model handles all sizes natively

### 2. Grouped Batching by Size
**Problem**: Can't batch different-sized images in TensorFlow

**Traditional Approach**: Resize to common size
- 17×17 → 32×32 creates half-pixel artifacts
- Distorts small maps

**Our Solution**: GroupedSizeDataLoader
- Groups images by size before batching
- Batch contains only same-sized images (all 10×10, etc.)
- No interpolation needed
- Efficient GPU utilization maintained

### 3. Threshold + Pseudo-Labeling Pipeline
**Problem**: Manual labeling of 10,000+ SOM maps is infeasible

**Our Solution**: Two-stage automated labeling
1. **Auto-Labeling** (threshold-based):
   - Label extreme cases (clearly bad: dead>30%, topo>50%)
   - Label excellent cases (dead<5% AND topo<10%)
   - Leave uncertain cases unlabeled
2. **Pseudo-Labeling** (iterative):
   - Train CNN on auto-labeled data
   - Predict on uncertain cases
   - Add confident predictions (>0.65 or <0.35)
   - Retrain → Repeat → Converge

### 4. Multi-Objective Hyperparameter Optimization
**Problem**: SOM has 15+ hyperparameters, conflicting objectives

**Our Solution**: NSGA-II with 4 objectives
- Finds Pareto-optimal trade-offs
- No single "best" solution - user chooses preference
- Fast (minimize MQE, TE, dead ratio)
- Efficient (minimize duration)

### 5. Multi-Agent AI System (Future)
**Vision**: Specialized AI agents cooperating
- **Oracle (MLP)**: Predicts good starting point
- **The Eye (CNN)**: Fast visual quality assessment
- **The Brain (LSTM)**: Real-time training control
- **The Voice (LLM)**: Natural language insights
- Each agent specializes → better than single monolithic model

---

## Success Metrics

### Phase 1 (Current)
- ✅ 1000+ SOM configurations generated
- ✅ Multi-objective optimization working
- ✅ CNN dataset prepared

### Phase 2 (In Progress)
- ⏸️ CNN accuracy > 80% on quality prediction
- ⏸️ EA converges 30% faster with CNN guidance
- ⏸️ LSTM reduces training time by 20%

### Phase 3 (Future)
- 🔜 End-to-end automation (data → report)
- 🔜 Non-expert users can analyze data successfully
- 🔜 System outperforms manual tuning

---

## Research Contributions

1. **Novel CNN Architecture**: Variable-size SOM quality assessment
2. **Multi-Agent Optimization**: Coordinated AI agents for hyperparameter tuning
3. **Automated Labeling**: Threshold + pseudo-labeling for training data
4. **Real-Time Control**: LSTM-based dynamic parameter adjustment

---

## Project Timeline

**Q4 2025**: Phase 1 Complete
- SOM core implementation
- EA optimizer
- CNN architecture design

**Q1 2026**: CNN Training & Integration
- Auto-labeling complete ✅
- First training (in progress)
- EA-CNN integration

**Q2 2026**: Advanced AI Components
- LSTM controller development
- MLP oracle development
- Growing SOM research

**Q3 2026**: User Experience
- Data mining module
- LLM report generation
- Web UI development

**Q4 2026**: Production Release
- End-to-end system testing
- Documentation
- Public release

---

## Summary

**NexusSOM** is building the future of automated machine learning:

✅ **Done**: Core SOM + EA + CNN data pipeline
🔄 **In Progress**: CNN training + EA integration
🔜 **Next**: LSTM control + MLP prediction + User interface

**Vision**: Upload data → Get insights → No manual tuning required

**Impact**: Democratizing unsupervised learning for non-experts

---

**End of Presentation**
