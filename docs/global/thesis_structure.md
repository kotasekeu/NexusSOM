# Thesis Structure — AI-Driven Data Analysis with Self-Organizing Maps

## Abstract

Self-Organizing Maps are a powerful unsupervised learning method for data visualization and clustering, but their practical use is limited by sensitivity to hyperparameters and the need for expert tuning. This thesis explores whether neural networks can automate this process — from initial configuration through training control to output quality assessment. A modular system (NexusSOM) is developed: not as a finished product, but as a tool to map the process of achieving AI-driven SOM analysis, document challenges at every stage, and honestly evaluate the real benefit of each component. Validation uses a clean real-world dataset to demonstrate functionality and a dataset with injected errors to prove anomaly detection capability.

## Pipeline

```
Dataset → Oracle (MLP) → EA + Eye (CNN) → SOM + Brain (LSTM) → Data Mining → Voice (LLM) → UI
```

---

## Chapters

### 1. Introduction

Short introduction to the problem, research goals, thesis scope and structure.

#### 1.1 Problem Statement
Why manual SOM tuning is impractical for non-experts and why automation is needed.

#### 1.2 Research Goals
Can neural networks drive the entire SOM analysis pipeline?

#### 1.3 Thesis Scope
Working prototype — not a commercial product. Focus on process, challenges and real benefit evaluation.

#### 1.4 Thesis Structure
Overview of chapters.

---

### 2. Theoretical Background

Theoretical foundations for all methods and technologies used in the system.

#### 2.1 Self-Organizing Maps
Algorithm, topologies (square, hexagonal), training modes (stochastic, deterministic, hybrid).

#### 2.2 Evolutionary Algorithms
Multi-objective optimization, NSGA-II.

#### 2.3 Convolutional Neural Networks
Image classification, feature extraction, pooling strategies.

#### 2.4 Recurrent Neural Networks
LSTM architecture, time-series prediction.

#### 2.5 Multilayer Perceptron
Regression, function approximation.

#### 2.6 Large Language Models
Text generation, structured summarization.

#### 2.7 Related Work
Existing SOM tools, AutoML approaches, neural-guided optimization.

---

### 3. System Architecture

Overview of the NexusSOM modular design, component roles and inter-module data flow.

#### 3.1 Modular Design
Each module is independently usable, AI components are optional enhancements.

#### 3.2 System Pipeline
Dataset → Oracle (MLP) → EA + Eye (CNN) → SOM + Brain (LSTM) → Data Mining → Voice (LLM) → UI

#### 3.3 Inter-Module Data Flow
Data formats, dependencies and communication between modules.

---

### 4. Self-Organizing Map

Core algorithm implementation — the central component of the system.

#### 4.1 Kohonen Algorithm Implementation
Weight initialization, BMU selection, neighborhood function, weight update rule.

#### 4.2 Training Modes
Stochastic, deterministic, hybrid with dynamic batch scheduling.

#### 4.3 Topology
Square vs hexagonal grid, cube coordinate system for hex distance.

#### 4.4 Decay Scheduling
Linear, exponential, logarithmic, step-down functions for learning rate, radius and batch size.

#### 4.5 Quality Metrics
MQE, topographic error, dead neuron ratio, U-matrix statistics.

#### 4.6 Data Preprocessing
Validation, missing value detection, normalization, categorical encoding.

#### 4.7 Issues and Challenges
##### 4.7.1 Logging Performance
File I/O overhead from per-call log writes slowing down the pipeline.
##### 4.7.2 Early Stopping Sensitivity
Moving average window can misread normal MQE fluctuations as convergence.
##### 4.7.3 Missing Data and the Ignore Mask
Sensor malfunction or extreme values create NaN entries — without masking, fill values distort the map.
##### 4.7.4 Mask Changing Data Semantics
With masking, dimension basis becomes non-uniform across samples — system-level consequence for interpretation.

#### 4.8 Evaluation
What works, what remains fragile, lessons learned.

---

### 5. Evolutionary Hyperparameter Optimization

NSGA-II based multi-objective search for optimal SOM configurations.

#### 5.1 NSGA-II Algorithm
Non-dominated sorting, crowding distance, SBX crossover, polynomial mutation.

#### 5.2 Search Space
12+ hyperparameters — learning rate, radius, decay types, batch config, map size, topology.

#### 5.3 Multi-Objective Fitness
MQE, duration, topographic error, dead neuron ratio.

#### 5.4 EA vs Hybrid SOM
Do we need EA when hybrid SOM already adapts dynamically? Hybrid adapts within a run; EA optimizes across runs. EA also generates training data for all neural network modules.

#### 5.5 Output
Pareto front of configurations, checkpoint generation for downstream modules.

#### 5.6 Issues and Challenges
Problems encountered, lessons learned.

#### 5.7 Evaluation
Real benefit of EA-driven optimization vs manual tuning.

---

### 6. The Eye — CNN Visual Quality Assessment

Convolutional neural network for judging SOM map quality from visualization.

#### 6.1 Motivation
Fast quality screening without computing full metrics.

#### 6.2 Architecture
Conv2D blocks, no MaxPooling, GlobalAveragePooling2D for size-invariant input (5×5 to 30×30).

#### 6.3 Data Pipeline
##### 6.3.1 Auto-Labeling
Threshold-based rules from EA metrics (dead ratio, topographic error).
##### 6.3.2 Pseudo-Labeling
Iterative CNN prediction on uncertain samples to expand labeled dataset.
##### 6.3.3 GroupedSizeDataLoader
Batching by map dimensions to avoid interpolation artifacts.

#### 6.4 EA Integration
CNN score as additional fitness objective for faster EA convergence.

#### 6.5 Issues and Challenges
Arbitrary label thresholds, pseudo-label noise amplification, CNN learning rendering artifacts alongside quality signals.

#### 6.6 Evaluation
Honest cost-benefit — what the CNN adds vs its complexity cost.

---

### 7. The Brain — LSTM Training Controller

Recurrent neural network for real-time adaptive control of SOM training.

#### 7.1 Motivation
Replace static decay schedules with learned dynamic control.

#### 7.2 Architecture
Sequence model over training checkpoints (MQE, learning rate, radius, dead ratio over time).

#### 7.3 Training Data
Checkpoint time-series generated from EA runs.

#### 7.4 Control Tasks
##### 7.4.1 Optimal Stopping Prediction
Predict when further training yields no improvement.
##### 7.4.2 Dynamic Parameter Adjustment
Adjust learning rate and radius mid-training based on observed trajectory.

#### 7.5 Issues and Challenges
Small number of sequences per EA run, variable-length training, generalization across datasets and map sizes.

#### 7.6 Evaluation
Real benefit vs static scheduling.

---

### 8. The Oracle — MLP Configuration Predictor

Multilayer perceptron for predicting initial SOM configuration from dataset statistics.

#### 8.1 Motivation
Skip EA cold-start by initializing with an informed population.

#### 8.2 Input Features
Dimensionality, sample count, correlations, missing value ratio, categorical ratio.

#### 8.3 Output
Predicted SOM configuration — map size, learning rate, radius, decay types.

#### 8.4 Training Data
Dataset–configuration–quality triplets from EA history.

#### 8.5 Issues and Challenges
Requires large number of diverse datasets, overfitting risk to specific data domains.

#### 8.6 Evaluation
Does Oracle initialization actually reduce EA generations needed?

---

### 9. Data Mining

Automatic analysis of trained SOM results — cluster identification and anomaly detection.

#### 9.1 Cluster Identification
Automatic cluster detection from SOM topology and U-matrix.

#### 9.2 Anomaly Detection
Outlier samples, unusual neuron activations, high quantization error regions.

#### 9.3 Pattern Discovery
Correlation analysis within and between clusters.

#### 9.4 Mask Interpretation
Interpreting clusters when samples contributed different dimension subsets (connection to SOM issue 4.7.4).

#### 9.5 Issues and Challenges
Cluster validity metrics, handling non-uniform feature basis, scalability.

#### 9.6 Evaluation
Quality and usefulness of automated analysis vs manual inspection.

---

### 10. The Voice — LLM Report Generation

Large language model for translating SOM analysis results into natural language.

#### 10.1 Motivation
Make SOM analysis accessible to non-technical users.

#### 10.2 Input
Structured analysis results — clusters, anomalies, metrics, patterns.

#### 10.3 Output
Natural language report with insights and actionable recommendations.

#### 10.4 Issues and Challenges
Hallucination risk, domain-specific terminology, reproducibility of generated text.

#### 10.5 Evaluation
Usefulness and accuracy of generated reports.

---

### 11. Experimental Validation

Two-part validation of the complete system.

#### 11.1 Clean Dataset
Real-world dataset, full pipeline execution, demonstrate valid interpretable results, compare AI-guided vs manual configuration.

#### 11.2 Error-Injected Dataset
Same dataset with deliberately placed errors (missing values, outliers, sensor drift). Prove the system detects injected anomalies, test mask behavior and cluster robustness.

---

### 12. Discussion

Critical evaluation of the system as a whole.

#### 12.1 Per-Component Evaluation
Actual contribution of each module vs its development cost.

#### 12.2 Essential vs Optional Components
Which modules are necessary, which are enhancements.

#### 12.3 Open Issues
Unresolved challenges and technical debt.

#### 12.4 Comparison
AI-driven approach vs simpler alternatives (manual tuning, grid search).

---

### 13. Conclusion and Future Work

Summary of contributions and direction for continued research.

#### 13.1 Contributions and Lessons Learned
What was achieved, what was learned in the process.

#### 13.2 Future Work — PhD Scope
Large-scale multi-dataset validation, model training across diverse domains, before/after comparison of AI-driven vs traditional analysis, resolution of issues found in this thesis.

---

## Appendices
- A: SOM Configuration Parameter Reference
- B: EA Search Space and NSGA-II Settings
- C: CNN Architecture Details
- D: Full Experimental Results and Metrics
