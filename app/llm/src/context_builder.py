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


def _find_dataset_context(start_path: str, max_levels: int = 4) -> str | None:
    """Search for dataset_context.txt (or ABOUT.MD fallback) walking up parent dirs."""
    current = os.path.abspath(start_path)
    about_fallback = None
    for _ in range(max_levels):
        candidate = os.path.join(current, "dataset_context.txt")
        if os.path.isfile(candidate):
            return candidate
        if about_fallback is None:
            for name in ("ABOUT.MD", "ABOUT.md", "about.md"):
                about = os.path.join(current, name)
                if os.path.isfile(about):
                    about_fallback = about
                    break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return about_fallback


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

    # Search for dataset_context.txt upward from dataset_path (also covers parent dataset dirs)
    ctx_path = _find_dataset_context(dataset_path)
    dataset_description = load_text(ctx_path) if ctx_path else "No dataset description provided."

    system_prompt = _build_system_prompt()
    user_context = _format_context(context_data, dataset_description,
                                   max_clusters, max_anomalies)

    return system_prompt, user_context


def _build_context_from_raw(json_dir, som_dir):
    """Build context from raw SOM output files when llm_context.json doesn't exist."""
    try:
        import sys as _sys
        _app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if _app_dir not in _sys.path:
            _sys.path.insert(0, _app_dir)
        from analysis.src.context import build_llm_context
        return build_llm_context(som_dir)
    except Exception:
        pass

    # Minimal fallback when analysis module is unavailable
    context = {}
    clusters_path = os.path.join(json_dir, "clusters.json")
    if os.path.isfile(clusters_path):
        context["clusters_raw"] = load_json(clusters_path)
    qe_path = os.path.join(json_dir, "quantization_errors.json")
    if os.path.isfile(qe_path):
        qe_data = load_json(qe_path)
        context["mqe"] = qe_data.get("total_quantization_error")
    extremes_path = os.path.join(json_dir, "extremes.json")
    if os.path.isfile(extremes_path):
        context["extremes"] = load_json(extremes_path)
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
        size = m.get('size', [])
        size_str = f"{size[0]}x{size[1]}" if len(size) == 2 else "unknown"
        sections.append(f"Size: {size_str} {m.get('topology', 'hex')}")
        sections.append(f"Total samples: {m['total_samples']}")
        sections.append(f"Total neurons: {m['total_neurons']}, Dead: {m['dead_neurons']} ({m['dead_ratio']:.0%})")
        sections.append(f"Mean Quantization Error (MQE): {m['mqe']:.4f}")
        te = m.get('topographic_error')
        sections.append(f"Topographic Error: {te:.4f}" if te is not None else "Topographic Error: N/A")
        sil = m.get('silhouette')
        if sil is not None:
            sections.append(f"Silhouette (cluster separation): {sil:.4f}  [>0.5 good, ~0 overlapping, <0 misassigned]")
        tc = m.get('trustworthiness_continuity', {})
        if tc:
            for k_str in ('5', '10', '20'):
                kv = tc.get(k_str) or tc.get(int(k_str))
                if kv:
                    sections.append(
                        f"T&C k={k_str}: Trustworthiness={kv['trustworthiness']:.4f}  "
                        f"Continuity={kv['continuity']:.4f}"
                    )

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
            if "silhouette" in c:
                line += f", sil={c['silhouette']:.3f}"
            sections.append(line)

            if "top_features" in c and c["top_features"]:
                dim_stats = c.get("dimension_stats", {})
                parts = []
                for f in c["top_features"]:
                    feat = f["feature"]
                    z    = f["z_score"]
                    mean = (dim_stats.get(feat) or {}).get("mean")
                    mean_str = f"={mean:.2f}" if mean is not None else ""
                    parts.append(f"{feat}{mean_str}(z={z:+.2f})")
                sections.append(f"  Key features: {', '.join(parts)}")
            elif "dimension_means" in c:
                dims = ", ".join(f"{k}={v:.2f}" for k, v in c["dimension_means"].items())
                sections.append(f"  Averages: {dims}")

            if "description" in c:
                sections.append(f"  Description: {c['description']}")

    # Section 4: Anomalies summary
    if "anomalies" in context_data:
        a = context_data["anomalies"]
        sections.append(f"\n=== ANOMALIES ===")
        sections.append(
            f"Global extremes: {a['global_outlier_count']}, "
            f"Local outliers: {a['local_outlier_count']}"
        )
        for anomaly in a.get("top_anomalies", [])[:max_anomalies]:
            atype = anomaly.get('type', '')
            line  = f"\nSample {anomaly['sample_id']} (neuron {anomaly['neuron']}, type={atype})"
            if anomaly.get('distance_ratio'):
                line += f", distance_ratio={anomaly['distance_ratio']:.2f}x"
            sections.append(line)
            for reason in anomaly.get("reasons", []):
                sections.append(f"  - {reason}")

    # Section 5: Anomaly records — full row values with delta annotations
    if "anomaly_records" in context_data and context_data["anomaly_records"]:
        sections.append("\n=== ANOMALY RECORDS (full row data) ===")
        sections.append(
            "Format: col=value [delta / delta_pct%] [FLAG]  "
            "FLAGS: global_min, global_max, high_deviation(>50%), moderate_deviation(>20%)"
        )
        for rec in context_data["anomaly_records"][:max_anomalies]:
            sections.append(
                f"\nSample {rec['sample_id']} | neuron {rec['neuron']} | "
                f"type={rec.get('type','')} "
                + (f"| ratio={rec['distance_ratio']:.2f}x" if rec.get('distance_ratio') else "")
            )
            col_parts = []
            for col, info in rec.get("columns", {}).items():
                val   = info.get("value", "")
                delta = info.get("delta")
                dpct  = info.get("delta_pct")
                flag  = info.get("flag", "")
                differs = info.get("differs_from_cluster_dominant")
                part  = f"{col}={val}"
                if delta is not None:
                    part += f" [{delta:+.2f} / {dpct:+.1f}%]"
                if flag:
                    part += f" [{flag.upper()}]"
                if differs:
                    part += f" [cluster_dom={differs}]"
                col_parts.append(part)
            sections.append("  " + "  |  ".join(col_parts))

    # Section 6: Dimension statistics
    if "dimension_stats" in context_data:
        sections.append("\n=== DIMENSION STATISTICS ===")
        for dim, stats in context_data["dimension_stats"].items():
            sections.append(
                f"{dim}: min={stats['min']}, max={stats['max']}, "
                f"mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                f"median={stats.get('median', stats['mean']):.2f}, "
                f"p25={stats.get('p25', '')}, p75={stats.get('p75', '')}"
            )

    # Section 7: Category distributions
    if "category_distributions" in context_data:
        sections.append("\n=== CATEGORY DISTRIBUTIONS ===")
        for col, dist in context_data["category_distributions"].items():
            parts = ", ".join(f"{v}={r:.0%}" for v, r in dist.items())
            sections.append(f"{col}: {parts}")

    # Section 8: Map regions
    if "map_regions" in context_data:
        sections.append("\n=== MAP SPATIAL PATTERNS ===")
        sections.append(context_data["map_regions"].get("summary", ""))

    return "\n".join(sections)
