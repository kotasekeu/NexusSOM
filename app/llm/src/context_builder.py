import json
import os
import csv


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def find_som_results(dataset_path):
    """Find SOM results directory within a dataset path."""
    som_dir = os.path.join(dataset_path, "results", "SOM")
    if os.path.isdir(som_dir):
        return som_dir

    # If dataset_path itself is a SOM results dir
    if os.path.isfile(os.path.join(dataset_path, "json", "clusters.json")):
        return dataset_path

    raise FileNotFoundError(f"No SOM results found in '{dataset_path}'")


def build_context(dataset_path, max_clusters=50, max_anomalies=10):
    """
    Build LLM context from SOM results and dataset description.

    Returns (system_prompt, user_context) tuple.
    """
    som_dir = find_som_results(dataset_path)
    json_dir = os.path.join(som_dir, "json")

    # Check for pre-built llm_context.json first
    llm_context_path = os.path.join(json_dir, "llm_context.json")
    if os.path.isfile(llm_context_path):
        context_data = load_json(llm_context_path)
    else:
        context_data = _build_context_from_raw(json_dir, som_dir)

    # Load dataset description
    dataset_context_path = os.path.join(dataset_path, "dataset_context.txt")
    if os.path.isfile(dataset_context_path):
        dataset_description = load_text(dataset_context_path)
    else:
        dataset_description = "No dataset description provided."

    system_prompt = _build_system_prompt()
    user_context = _format_context(context_data, dataset_description,
                                   max_clusters, max_anomalies)

    return system_prompt, user_context


def _build_context_from_raw(json_dir, som_dir):
    """Build context from raw SOM output files when llm_context.json doesn't exist."""
    context = {}

    # Load clusters
    clusters_path = os.path.join(json_dir, "clusters.json")
    if os.path.isfile(clusters_path):
        clusters = load_json(clusters_path)
        context["clusters_raw"] = clusters

    # Load quantization errors
    qe_path = os.path.join(json_dir, "quantization_errors.json")
    if os.path.isfile(qe_path):
        qe_data = load_json(qe_path)
        context["mqe"] = qe_data.get("total_quantization_error")
        context["neuron_qe"] = qe_data.get("neuron_quantization_errors", {})

    # Load extremes
    extremes_path = os.path.join(json_dir, "extremes.json")
    if os.path.isfile(extremes_path):
        context["extremes"] = load_json(extremes_path)

    # Load pie data (all categorical columns)
    pie_files = [f for f in os.listdir(json_dir) if f.startswith("pie_data_")]
    context["categories"] = {}
    for pf in pie_files:
        col_name = pf.replace("pie_data_", "").replace(".json", "")
        context["categories"][col_name] = load_json(os.path.join(json_dir, pf))

    return context


def _build_system_prompt():
    return """You are a data analysis assistant specialized in interpreting Self-Organizing Map (SOM) results.

RULES:
1. You ONLY discuss the provided dataset and its SOM analysis results.
2. Every claim you make must be grounded in specific data from the analysis (neuron IDs, sample IDs, metric values).
3. You NEVER invent or fabricate data points, statistics, or sample IDs.
4. If asked about something not covered by the provided data, say: "This information is not available in the current analysis."
5. Use the domain terminology from the dataset description, not SOM technical jargon, when explaining findings to the user.
6. When describing clusters, explain what the samples in each cluster have in common using the original dimension names and their domain meanings.
7. When reporting anomalies, explain why they are unusual in domain terms."""


def _format_context(context_data, dataset_description, max_clusters, max_anomalies):
    """Format context data into a structured text prompt."""
    sections = []

    # Section 1: Dataset description
    sections.append("=== DATASET DESCRIPTION ===")
    sections.append(dataset_description)

    # Section 2: Map overview
    if "map" in context_data:
        m = context_data["map"]
        sections.append("\n=== MAP OVERVIEW ===")
        sections.append(f"Size: {m['size'][0]}x{m['size'][1]} {m.get('topology', 'unknown')}")
        sections.append(f"Total samples: {m['total_samples']}")
        sections.append(f"Total neurons: {m['total_neurons']}, Dead: {m['dead_neurons']} ({m['dead_ratio']:.0%})")
        sections.append(f"Mean Quantization Error (MQE): {m['mqe']:.4f}")
        sections.append(f"Topographic Error: {m.get('topographic_error', 'N/A')}")

    # Section 3: Clusters
    if "clusters" in context_data:
        sections.append("\n=== CLUSTERS ===")
        clusters = context_data["clusters"][:max_clusters]
        for c in sorted(clusters, key=lambda x: x["sample_count"], reverse=True):
            line = f"\nNeuron {c['neuron']}: {c['sample_count']} samples, QE={c['quantization_error']:.4f}"
            if "dominant_category" in c:
                cats = ", ".join(f"{k}={v}" for k, v in c["dominant_category"].items())
                line += f", dominant: {cats}"
            if "purity" in c:
                purities = ", ".join(f"{k}={v:.0%}" for k, v in c["purity"].items())
                line += f", purity: {purities}"
            sections.append(line)

            if "dimension_means" in c:
                dims = ", ".join(f"{k}={v:.2f}" for k, v in c["dimension_means"].items())
                sections.append(f"  Averages: {dims}")

            if "description" in c:
                sections.append(f"  Description: {c['description']}")

    # Section 4: Anomalies
    if "anomalies" in context_data:
        a = context_data["anomalies"]
        sections.append(f"\n=== ANOMALIES ===")
        sections.append(f"Global outliers: {a['global_outlier_count']}, Local outliers: {a['local_outlier_count']}")

        for anomaly in a.get("top_anomalies", [])[:max_anomalies]:
            sections.append(f"\nSample {anomaly['sample_id']} (neuron {anomaly['neuron']}):")
            for reason in anomaly["reasons"]:
                sections.append(f"  - {reason}")

    # Section 5: Dimension statistics
    if "dimension_stats" in context_data:
        sections.append("\n=== DIMENSION STATISTICS ===")
        for dim, stats in context_data["dimension_stats"].items():
            sections.append(
                f"{dim}: min={stats['min']}, max={stats['max']}, "
                f"mean={stats['mean']:.2f}, std={stats['std']:.2f}"
            )

    # Section 6: Map regions
    if "map_regions" in context_data:
        sections.append("\n=== MAP SPATIAL PATTERNS ===")
        sections.append(context_data["map_regions"].get("summary", ""))

    return "\n".join(sections)
