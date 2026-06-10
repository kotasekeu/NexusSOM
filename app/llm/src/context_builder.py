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


def _compute_mask_summary(som_dir: str, preprocessing_info: dict) -> dict | None:
    """
    Read ignore_mask.csv and compute missing/masked data statistics.
    Excludes always-masked columns (primary ID etc.).
    Returns summary dict or None if mask file not found.
    """
    import numpy as np
    mask_path = os.path.join(som_dir, 'csv', 'ignore_mask.csv')
    if not os.path.isfile(mask_path):
        return None

    mask = []
    with open(mask_path, encoding='utf-8') as f:
        for line in f:
            mask.append([v.strip().lower() == 'true' for v in line.split(',')])
    if not mask:
        return None

    mask = np.array(mask, dtype=bool)
    n_samples, n_dims = mask.shape

    # Column names from preprocessing_info (same order as training data)
    col_names = list(preprocessing_info.keys()) if preprocessing_info else \
                [f'dim_{i}' for i in range(n_dims)]

    # Always-masked = structural ignore (ID etc.) — all rows True
    always_idx  = [i for i in range(n_dims) if mask[:, i].all()]
    always_names = [col_names[i] for i in always_idx if i < len(col_names)]

    # Active columns = not always masked
    active = [(i, col_names[i] if i < len(col_names) else f'dim_{i}')
              for i in range(n_dims) if i not in always_idx]

    # Per-column missing count
    col_missing = {name: int(mask[:, idx].sum())
                   for idx, name in active if mask[:, idx].any()}

    # Samples with any missing active-column value
    active_idx  = [i for i, _ in active]
    active_mask = mask[:, active_idx] if active_idx else np.zeros((n_samples, 0), dtype=bool)
    affected    = int(active_mask.any(axis=1).sum())

    # Top 5 most incomplete samples
    missing_per = active_mask.sum(axis=1)
    top_si = np.argsort(missing_per)[::-1][:5]
    top_samples = []
    for si in top_si:
        if missing_per[si] == 0:
            break
        dims = [name for idx, name in active if mask[si, idx]]
        top_samples.append({'sample_index': int(si), 'missing_dims': dims})

    return {
        'always_masked':          always_names,
        'n_affected_samples':     affected,
        'n_total_samples':        n_samples,
        'col_missing':            col_missing,
        'top_incomplete_samples': top_samples,
    }


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

    # Mask summary — read directly from ignore_mask.csv (always fresh, not cached in json)
    preprocessing_info = load_json(os.path.join(json_dir, "preprocessing_info.json")) or {}
    mask_summary = _compute_mask_summary(som_dir, preprocessing_info)

    ctx_path = _find_dataset_context(dataset_path)
    dataset_description = load_text(ctx_path) if ctx_path else "No dataset description provided."

    system_prompt = _build_system_prompt()
    user_context  = _format_context(context_data, dataset_description,
                                    max_clusters, max_anomalies,
                                    mask_summary=mask_summary)

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
    return """Jsi datový analytik specializovaný na interpretaci výsledků samoorganizující se mapy (SOM).
Uživatel ti poskytl kontext s výsledky analýzy. Odpovídáš výhradně na základě tohoto kontextu.

STYL ODPOVĚDÍ:
- Odpověz PŘÍMO a STRUČNĚ. Začni okamžitě odpovědí, ne popisem co hodláš udělat.
- NIKDY nezobrazuj svůj myšlenkový proces ani průběžné úvahy.
- NIKDY neopakuj otázku ani neshrnuj co ses dozvěděl před odpovědí.
- Na faktické dotazy (největší cluster, počet vzorků apod.) odpověz jednou větou s číslem.
- Na analytické dotazy použij 3–5 vět nebo bullet pointy. Cituj konkrétní neurony a hodnoty.
- Pokud je vhodná tabulka, použij markdown tabulku.
- VÝCHOZÍ DÉLKA: max. 10 řádků. Pokud otázka explicitně žádá detaily, můžeš být delší.
- Při výpisu položek (anomálie, clustery): max. 5 položek. Na každou položku max. 2 řádky.
- Po stručném výpisu vždy nabídni: "Chceš detail k některé položce?"

PRAVIDLA GROUNDINGU:
1. Odpovídáš POUZE na základě poskytnutého kontextu analýzy — žádná obecná znalost.
2. Každé tvrzení musí být doloženo konkrétními daty (ID neuronu, hodnota dimenze, počet vzorků).
3. NIKDY nevymýšlíš data, statistiky ani ID vzorků.
4. Pokud informace v kontextu není, řekni: "Tato informace není v aktuální analýze k dispozici."
5. Používej terminologii domény datasetu, ne SOM technický žargon.
6. Dotazy mimo dataset (recepty, obecné znalosti apod.) odmítni zdvořile jednou větou.

PŘÍKLAD správné odpovědi na "Který cluster obsahuje nejvíce vzorků?":
Neuron 21_1 obsahuje 93 vzorků — nejvíce ze všech neuronů v mapě.

PŘÍKLAD špatné odpovědi (NEDĚLEJ TOTO):
Podle údajů v části MAP OVERVIEW... Nejbližší cluster který obsahuje... [opakování dat bez odpovědi]"""


def _format_context(context_data, dataset_description, max_clusters, max_anomalies,
                    mask_summary=None):
    """Format context data into a structured text prompt."""
    sections = []

    # Section 1: Dataset description
    sections.append("=== POPIS DATASETU ===")
    if dataset_description and dataset_description.strip() != "No dataset description provided.":
        sections.append(
            "Následující text popisuje dataset, jeho doménový kontext a jednotlivé sloupce/dimenze. "
            "Použij tyto informace pro interpretaci výsledků SOM analýzy. "
            "Odpovídej v terminologii této domény — nepoužívej technický žargon SOM pokud to není nutné.\n"
        )
        sections.append(dataset_description)
    else:
        sections.append("Popis datasetu není k dispozici. Odpovídej na základě názvů sloupců.")

    # Section 2: Map overview
    if "map" in context_data:
        m = context_data["map"]
        sections.append("\n=== MAP OVERVIEW ===")
        size = m.get('size', [])
        size_str = f"{size[0]}x{size[1]}" if len(size) == 2 else "unknown"
        sections.append(f"Size: {size_str} {m.get('topology', 'hex')}")
        sections.append(f"Total samples: {m['total_samples']}")
        dead_count = m['dead_neurons']
        dead_info  = f"Dead: {dead_count} ({m['dead_ratio']:.0%})"
        # For small dead counts list which neurons; for large counts just the number
        if dead_count > 0 and "clusters" in context_data:
            active_keys = {c['neuron'] for c in context_data["clusters"]}
            size_val = m.get('size', [0, 0])
            if len(size_val) == 2:
                all_keys = {f"{i}_{j}"
                            for i in range(size_val[0])
                            for j in range(size_val[1])}
                dead_keys = sorted(all_keys - active_keys)
                if 0 < len(dead_keys) <= 15:
                    dead_info += f" — {', '.join(dead_keys)}"
        sections.append(f"Total neurons: {m['total_neurons']}, {dead_info}")
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

    # Section 9: Masked / missing data (from ignore_mask.csv)
    if mask_summary is not None:
        sections.append("\n=== MASKOVANÁ / CHYBĚJÍCÍ DATA ===")
        if mask_summary['always_masked']:
            sections.append(f"Strukturálně ignorované sloupce: {', '.join(mask_summary['always_masked'])}")
        affected = mask_summary['n_affected_samples']
        total    = mask_summary['n_total_samples']
        if affected == 0:
            sections.append("Žádná chybějící data (mimo strukturálně ignorované sloupce).")
        else:
            sections.append(f"Vzorky s chybějícími hodnotami: {affected} / {total} "
                            f"({affected / total * 100:.1f} %)")
            for col, cnt in sorted(mask_summary.get('col_missing', {}).items(),
                                   key=lambda x: -x[1]):
                sections.append(f"  - {col}: {cnt} vzorků")
            for s in mask_summary.get('top_incomplete_samples', []):
                dims = ', '.join(s['missing_dims'])
                sections.append(f"  vzorek #{s['sample_index']}: chybí {dims}")

    return "\n".join(sections)
