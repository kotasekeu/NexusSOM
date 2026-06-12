"""
verify_ea_run.py — diagnostic tool for EA result directories.

Usage:
  python app/tools/verify_ea_run.py <results_dir>
  python app/tools/verify_ea_run.py data/datasets/BreastCancer/results/EA-30x50

Checks:
  1. Overall run statistics and penalty distribution
  2. Map-size correlation with penalties
  3. Generation-by-generation Pareto archive evolution
  4. Elitism verification: are best solutions retained across generations?
  5. Final archive Pareto dominance check (detects internal dominance violations)
  6. Crowding-distance ejection: good solutions lost to cap vs. true dominance
  7. Parameter correlation with penalization
  8. Recommendations
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def load_results(results_dir: str) -> list[dict]:
    """Load results.csv. For duplicate UIDs (re-evaluations), keep only the first row."""
    path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: results.csv not found in {results_dir}")
    seen: set[str] = set()
    rows: list[dict] = []
    duplicates = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row.get('uid', '')
            if uid in seen:
                duplicates += 1
                continue
            seen.add(uid)
            rows.append(row)
    if duplicates:
        print(f"  NOTE: {duplicates} duplicate UID rows skipped (non-deterministic re-evaluations)")
    return rows


# ---------------------------------------------------------------------------
# Pareto front CSV parsing (new format: one row per generation/solution)
# Falls back to legacy pareto_front_log.txt if CSV not found.
# ---------------------------------------------------------------------------

def _fv(v, default=None):
    """Safe float parse, returns default on empty/None."""
    if v is None or v == '':
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def parse_pareto_log(results_dir: str) -> list[dict]:
    """
    Return list of {gen, count, solutions:[{uid, uid8, ratio, te, time, dead, map, is_penalized, pen_factor}]}.
    Reads pareto_front.csv (new) or falls back to pareto_front_log.txt (legacy).
    """
    csv_path = os.path.join(results_dir, "pareto_front.csv")
    if os.path.exists(csv_path):
        return _parse_pareto_csv(csv_path)

    txt_path = os.path.join(results_dir, "pareto_front_log.txt")
    if os.path.exists(txt_path):
        return _parse_pareto_txt(txt_path)

    return []


def _parse_pareto_csv(path: str) -> list[dict]:
    """Parse new CSV format into the same structure as the legacy parser."""
    gen_map: dict[int, list] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gn   = int(row.get('generation', 0))
            uid  = row.get('uid', '')
            ms_raw = row.get('map_size', '?')
            try:
                ms = json.loads(ms_raw.replace("'", '"'))
                map_str = f"{ms[0]}x{ms[1]}"
            except Exception:
                map_str = ms_raw
            tc = _fv(row.get('raw_topo_corr'))
            gen_map[gn].append({
                'uid':        uid,
                'uid8':       uid[:8],
                'ratio':      _fv(row.get('raw_mqe_ratio')),
                'te':         _fv(row.get('raw_te')),
                'topo_corr':  tc,
                'one_minus_rho': (1.0 - tc) if tc is not None else None,
                'time':       _fv(row.get('duration')),
                'dead':       _fv(row.get('dead_ratio')),
                'map':        map_str,
                'is_penalized': str(row.get('is_penalized', '')).lower() == 'true',
                'pen_factor': _fv(row.get('penalty_factor'), 1.0),
            })

    return [
        {'gen': gn, 'count': len(sols), 'solutions': sols}
        for gn, sols in sorted(gen_map.items())
    ]


def _parse_pareto_txt(path: str) -> list[dict]:
    """Legacy parser for pareto_front_log.txt (pre-CSV format)."""
    log = open(path, encoding="utf-8").read()
    parts = re.split(r'--- Generation (\d+) \| Number of solutions: (\d+) ---', log)

    def _parse_solutions(content: str) -> list[dict]:
        solutions = []
        entries = re.split(r'UID: ([a-f0-9]{32})', content)
        for i in range(1, len(entries), 2):
            uid  = entries[i]
            body = entries[i + 1]
            ratio_m = re.search(r'(?:raw_ratio|MQE_ratio)=([\d.]+)', body)
            te_m    = re.search(r'(?:raw_TE|TE)=([\d.]+)', body)
            time_m  = re.search(r'Time=([\d.]+)', body)
            dead_m  = re.search(r'Dead=([\d.]+)', body)
            ms_m    = re.search(r'map_size: \[(\d+), (\d+)\]', body)
            solutions.append({
                'uid':         uid,
                'uid8':        uid[:8],
                'ratio':       _fv(ratio_m.group(1)) if ratio_m else None,
                'te':          _fv(te_m.group(1))    if te_m    else None,
                'time':        _fv(time_m.group(1))  if time_m  else None,
                'dead':        _fv(dead_m.group(1))  if dead_m  else None,
                'map':         f"{ms_m.group(1)}x{ms_m.group(2)}" if ms_m else "?",
                'is_penalized': False,
                'pen_factor':  1.0,
            })
        return solutions

    gens = []
    for i in range(1, len(parts), 3):
        gens.append({
            'gen':       int(parts[i]),
            'count':     int(parts[i + 1]),
            'solutions': _parse_solutions(parts[i + 2]),
        })
    return gens


# ---------------------------------------------------------------------------
# Pareto dominance helpers
# ---------------------------------------------------------------------------

def _cv_dominates(a: dict, b: dict) -> bool:
    """
    Constrained dominance (Deb 2002) on the 3 raw NSGA-II objectives:
    raw_mqe_ratio, raw_te, 1 - topological_correlation (dead_ratio is a
    constraint, not an objective — legacy ISSUES.md #85).
    'constraint_violation' key: 0.0 = feasible, >0 = infeasible.
    Missing objective values are skipped (legacy runs without topo_corr
    degrade to a 2-objective check).
    """
    cv_a = a.get('constraint_violation', 0.0) or 0.0
    cv_b = b.get('constraint_violation', 0.0) or 0.0
    feasible_a = cv_a < 1e-9
    feasible_b = cv_b < 1e-9

    if feasible_a and not feasible_b:
        return True
    if not feasible_a and feasible_b:
        return False
    if not feasible_a and not feasible_b:
        return cv_a < cv_b

    # Both feasible: standard Pareto on 3 raw objectives
    objs = ['ratio', 'te', 'one_minus_rho']
    better = False
    for o in objs:
        av, bv = a.get(o), b.get(o)
        if av is None or bv is None:
            continue
        if av > bv:
            return False
        if av < bv:
            better = True
    return better


def find_dominated_in_archive(solutions: list[dict]) -> list[str]:
    """Return UIDs of solutions dominated by another using constrained dominance."""
    dominated = []
    for i, s in enumerate(solutions):
        for j, other in enumerate(solutions):
            if i == j:
                continue
            if _cv_dominates(other, s):
                dominated.append(s['uid'])
                break
    return dominated


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------

SEP = "=" * 72
SEP2 = "-" * 72

def _ratio_class(r, is_penalized=False):
    if r is None:
        return "unknown"
    label = "GOOD" if r < 1.0 else ("ok" if r < 2.0 else "bad")
    return f"{label}+PEN" if is_penalized else label


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------

def report_overview(rows: list[dict]):
    section("1. OVERVIEW")
    n = len(rows)

    # Use raw_mqe_improvement_ratio if available, fall back to mqe_improvement_ratio
    def _raw_ratio(r):
        v = r.get('raw_mqe_improvement_ratio') or r.get('mqe_improvement_ratio')
        return float(v) if v not in ('', None) else None

    ratios = [_raw_ratio(r) for r in rows]
    ratios = [r for r in ratios if r is not None]
    penalized = [r for r in rows if str(r.get('is_penalized', '')).lower() == 'true']

    if not ratios:
        print("  No ratio values found.")
        return

    good = sum(1 for r in ratios if r < 1.0)
    ok   = sum(1 for r in ratios if 1.0 <= r < 2.0)
    bad  = sum(1 for r in ratios if r >= 2.0)

    print(f"  Total evaluated runs  : {n}")
    print(f"  raw_ratio < 1  (good) : {good:5d}  ({good/n*100:.1f}%)")
    print(f"  raw_ratio 1-2  (ok)   : {ok:5d}  ({ok/n*100:.1f}%)")
    print(f"  raw_ratio >= 2 (bad)  : {bad:5d}  ({bad/n*100:.1f}%)")
    print(f"  Is penalized (any)    : {len(penalized):5d}  ({len(penalized)/n*100:.1f}%)")
    print(f"  Best raw_ratio        : {min(ratios):.4f}")
    print(f"  Worst raw_ratio       : {max(ratios):.4f}")
    print(f"  Unique UIDs           : {n}")


# ---------------------------------------------------------------------------
# 2. Map size vs penalty
# ---------------------------------------------------------------------------

def report_map_penalty(rows: list[dict]):
    section("2. MAP SIZE vs PENALTY RATE")
    buckets: dict[int, list] = defaultdict(list)
    for r in rows:
        try:
            ms = json.loads(r['map_size'].replace("'", '"'))
            area = ms[0] * ms[1]
        except Exception:
            area = 0
        raw_ratio = float(r.get('raw_mqe_improvement_ratio') or r.get('mqe_improvement_ratio') or 0)
        is_pen = str(r.get('is_penalized', '')).lower() == 'true'
        buckets[area].append((raw_ratio, is_pen))

    print(f"  {'Area':>6}  {'N':>5}  {'%penalized':>11}  {'avg_raw_ratio':>14}  {'best_raw_ratio':>15}")
    print(f"  {SEP2}")
    for area in sorted(buckets):
        entries = buckets[area]
        ratios  = [e[0] for e in entries]
        pen_pct = sum(1 for e in entries if e[1]) / len(entries) * 100
        print(f"  {area:6d}  {len(entries):5d}  {pen_pct:10.1f}%  {sum(ratios)/len(ratios):14.3f}  {min(ratios):15.4f}")


# ---------------------------------------------------------------------------
# 3. Archive evolution per generation
# ---------------------------------------------------------------------------

def report_archive_evolution(gen_data: list[dict]):
    section("3. ARCHIVE EVOLUTION PER GENERATION")
    if not gen_data:
        print("  No pareto_front_log.txt found.")
        return

    print(f"  {'Gen':>4}  {'Size':>5}  {'GOOD':>6}  {'ok':>4}  {'PEN':>5}  best_ratio")
    print(f"  {SEP2}")
    for g in gen_data:
        sols = g['solutions']
        good = sum(1 for s in sols if s['ratio'] is not None and s['ratio'] < 1.0)
        ok   = sum(1 for s in sols if s['ratio'] is not None and 1.0 <= s['ratio'] < 2.0)
        pen  = sum(1 for s in sols if s['ratio'] is not None and s['ratio'] >= 2.0)
        best = min((s['ratio'] for s in sols if s['ratio'] is not None), default=None)
        best_str = f"{best:.4f}" if best is not None else "?"
        print(f"  {g['gen']:4d}  {g['count']:5d}  {good:6d}  {ok:4d}  {pen:5d}  {best_str}")


# ---------------------------------------------------------------------------
# 4. Elitism: track best UIDs across generations
# ---------------------------------------------------------------------------

def report_elitism(gen_data: list[dict]):
    section("4. ELITISM CHECK — BEST SOLUTION TRACKING")
    if len(gen_data) < 2:
        return

    # Collect all UIDs that were ever the best
    all_best_uids = set()
    for g in gen_data:
        sols_with_ratio = [s for s in g['solutions'] if s['ratio'] is not None]
        if sols_with_ratio:
            best = min(sols_with_ratio, key=lambda s: s['ratio'])
            all_best_uids.add(best['uid'])

    # Track per-generation presence of each unique best UID
    ever_good = set()
    for g in gen_data:
        for s in g['solutions']:
            if s['ratio'] is not None and s['ratio'] < 1.0:
                ever_good.add(s['uid'])

    print(f"  Tracking {len(ever_good)} UIDs that ever appeared as 'good' (ratio < 1):\n")
    print(f"  {'UID':10}  {'first':>6}  {'last':>6}  {'gaps':>5}  best_ratio  note")
    print(f"  {SEP2}")

    uid_gens = defaultdict(list)
    uid_ratio = {}
    for g in gen_data:
        for s in g['solutions']:
            if s['uid'] in ever_good:
                uid_gens[s['uid']].append(g['gen'])
                if s['uid'] not in uid_ratio and s['ratio'] is not None:
                    uid_ratio[s['uid']] = s['ratio']

    for uid, present_gens in sorted(uid_gens.items(), key=lambda x: uid_ratio.get(x[0], 99)):
        first, last = min(present_gens), max(present_gens)
        # gaps: generations where it should be in archive but isn't
        expected = set(range(first, last + 1))
        gaps = len(expected - set(present_gens))
        note = ""
        if gaps > 0:
            missing_gens = sorted(expected - set(present_gens))
            note = f"MISSING in gens {missing_gens[:5]}{'...' if len(missing_gens) > 5 else ''}"
        ratio_str = f"{uid_ratio.get(uid, '?'):.4f}" if uid_ratio.get(uid) is not None else "?"
        print(f"  {uid[:10]:10}  {first:6d}  {last:6d}  {gaps:5d}  {ratio_str:10}  {note}")


# ---------------------------------------------------------------------------
# 5. Final archive Pareto dominance check
# ---------------------------------------------------------------------------

def report_final_archive(gen_data: list[dict], rows: list[dict]):
    section("5. FINAL ARCHIVE ANALYSIS")
    if not gen_data:
        return

    final = gen_data[-1]
    sols = final['solutions']
    gen_num = final['gen']

    uid_to_row = {r['uid']: r for r in rows}
    print(f"  Final generation: {gen_num}, archive size: {len(sols)}\n")
    print(f"  {'UID':10}  {'status':12}  {'raw_ratio':10}  {'raw_TE':8}  {'rho':6}  {'dead':6}  {'pen_x':7}  {'time':7}  map")
    print(f"  {SEP2}")
    def _fval(row_val, fallback, default=999.0):
        """Safe float: prefer row_val, then fallback, handle 0.0 correctly."""
        for v in (row_val, fallback):
            if v is not None and v != '':
                return float(v)
        return default

    for s in sorted(sols, key=lambda x: x['ratio'] if x['ratio'] is not None else 999):
        row = uid_to_row.get(s['uid'], {})
        raw_r  = _fval(row.get('raw_mqe_improvement_ratio'), s['ratio'])
        raw_te = _fval(row.get('raw_topographic_error'), s['te'])
        dead   = _fval(row.get('dead_neuron_ratio'), s['dead'])
        pen_f  = _fval(row.get('penalty_factor'), 1.0, default=1.0)
        is_pen = str(row.get('is_penalized', '')).lower() == 'true'
        time_v = _fval(row.get('duration'), s['time'], default=0.0)
        status = _ratio_class(raw_r, is_pen)
        pen_str = f"x{pen_f:.1f}" if is_pen else "—"
        rho_v = s.get('topo_corr')
        rho_str = f"{rho_v:6.3f}" if rho_v is not None else "     ?"
        print(f"  {s['uid8']:10}  {status:12}  {raw_r:10.4f}  {raw_te:8.4f}  {rho_str}  {dead:6.3f}  {pen_str:7}  {time_v:7.1f}  {s['map']}")

    # Dominance check using results.csv values (3 raw objectives, first evaluation per UID).
    # topo_corr: prefer results.csv raw_topological_correlation (present since
    # cleanup F17), fall back to pareto_front.csv raw_topo_corr; None for legacy runs.
    uid_to_row = {r['uid']: r for r in rows}
    enriched: list[dict] = []
    for s in sols:
        row = uid_to_row.get(s['uid'], {})
        cv = _fv(row.get('constraint_violation'), default=0.0)
        rho = _fv(row.get('raw_topological_correlation'))
        if rho is None:
            rho = s.get('topo_corr')
        enriched.append({
            'uid':                s['uid'],
            'ratio':              _fval(row.get('raw_mqe_improvement_ratio'), s['ratio']),
            'te':                 _fval(row.get('raw_topographic_error'), s['te']),
            'one_minus_rho':      (1.0 - rho) if rho is not None else None,
            'constraint_violation': cv,
        })

    dominated = find_dominated_in_archive(enriched)
    if dominated:
        print(f"\n  BUG: {len(dominated)} solutions in archive are dominated by another (constrained dominance check)!")
        for uid in dominated:
            print(f"    UID {uid[:8]}")
    else:
        print(f"\n  OK: No internal dominance in final archive (constrained dominance, raw objectives).")

    print(f"\n  Constraint violation breakdown from results.csv:")
    print(f"  {'UID':10}  {'cv':8}  {'u_mat_max':10}  {'dist_max':10}  {'dead_ratio':12}  {'map':8}  penalty_reason")
    print(f"  {SEP2}")
    for s in sorted(sols, key=lambda x: x['ratio'] if x['ratio'] is not None else 999):
        row = uid_to_row.get(s['uid'])
        if not row:
            continue
        cv  = float(row.get('constraint_violation', 0) or 0)
        um  = float(row.get('u_matrix_max', 0) or 0)
        dm  = float(row.get('distance_map_max', 0) or 0)
        dr  = float(row.get('dead_neuron_ratio', 0) or 0)
        m_m = row.get('map_m', '?')
        m_n = row.get('map_n', '?')
        reason = row.get('penalty_reason', '') or ('none (feasible)' if cv < 1e-9 else '')
        print(f"  {s['uid8']:10}  {cv:8.4f}  {um:10.3f}  {dm:10.3f}  {dr:12.4f}  {m_m}x{m_n:8}  {reason}")


# ---------------------------------------------------------------------------
# 6. Crowding ejection vs true dominance
# ---------------------------------------------------------------------------

def report_crowding_ejection(gen_data: list[dict]):
    section("6. CROWDING EJECTION — GOOD SOLUTIONS DROPPED")
    if len(gen_data) < 2:
        return

    all_good_uids: dict[str, float] = {}
    for g in gen_data:
        for s in g['solutions']:
            if s['ratio'] is not None and s['ratio'] < 1.0:
                if s['uid'] not in all_good_uids:
                    all_good_uids[s['uid']] = s['ratio']

    final_uids = {s['uid'] for s in gen_data[-1]['solutions']}

    dropped = {uid: r for uid, r in all_good_uids.items() if uid not in final_uids}
    retained = {uid: r for uid, r in all_good_uids.items() if uid in final_uids}

    print(f"  Good solutions (ratio<1) ever seen : {len(all_good_uids)}")
    print(f"  Retained in final archive          : {len(retained)}")
    print(f"  NOT in final archive               : {len(dropped)}\n")

    if not dropped:
        print("  All good solutions survived to the final archive.")
        return

    print(f"  Dropped good solutions:")
    print(f"  {'UID':10}  {'ratio':8}  {'last_seen_gen':14}  reason")
    print(f"  {SEP2}")
    for uid, ratio in sorted(dropped.items(), key=lambda x: x[1]):
        # Find last generation where this UID appeared
        last_gen = max(g['gen'] for g in gen_data if any(s['uid'] == uid for s in g['solutions']))
        # Find what next gen looked like
        next_gen_data = next((g for g in gen_data if g['gen'] == last_gen + 1), None)
        reason = "unknown"
        if next_gen_data:
            next_sols = next_gen_data['solutions']
            # Is it dominated by anything in next gen?
            this_sol = next(s for g in gen_data for s in g['solutions'] if s['uid'] == uid)
            dominators = [s for s in next_sols if _cv_dominates(s, this_sol)]
            if dominators:
                reason = f"dominated by {dominators[0]['uid8']} (ratio={dominators[0]['ratio']:.3f})"
            else:
                reason = f"CROWDING CAP (not dominated; archive size={len(next_sols)})"
        print(f"  {uid[:10]:10}  {ratio:8.4f}  {last_gen:14d}  {reason}")


# ---------------------------------------------------------------------------
# 7. Parameter correlation with penalty
# ---------------------------------------------------------------------------

def report_param_penalty(rows: list[dict]):
    section("7. PARAMETER CORRELATION WITH PENALTY")
    cat_params = ['lr_decay_type', 'radius_decay_type', 'batch_growth_type', 'normalize_weights_flag']
    print(f"  Categorical parameters — constraint violation rate by value:\n")
    for param in cat_params:
        buckets: dict[str, list] = defaultdict(list)
        for r in rows:
            val = r.get(param) or '?'
            raw_ratio = float(r.get('raw_mqe_improvement_ratio') or r.get('mqe_improvement_ratio') or 0)
            is_pen = str(r.get('is_penalized', '')).lower() == 'true'
            buckets[val].append((raw_ratio, is_pen))
        if not buckets:
            continue
        print(f"  {param}:")
        for val, entries in sorted(buckets.items()):
            pen_pct = sum(1 for _, p in entries if p) / len(entries) * 100
            avg_ratio = sum(r for r, _ in entries) / len(entries)
            print(f"    {val:30s} n={len(entries):5d}  %infeasible={pen_pct:5.1f}%  avg_raw_ratio={avg_ratio:.3f}")
        print()


# ---------------------------------------------------------------------------
# 8. Recommendations
# ---------------------------------------------------------------------------

def report_recommendations(rows: list[dict], gen_data: list[dict], results_dir: str = ""):
    section("8. RECOMMENDATIONS")

    ratios = [float(r['mqe_improvement_ratio']) for r in rows
              if r.get('mqe_improvement_ratio') not in ('', None)]
    pen_pct = sum(1 for r in ratios if r >= 2) / len(ratios) * 100 if ratios else 0

    recs = []

    # Check normalize_weights_flag penalty correlation
    norm_true  = [r for r in rows if str(r.get('normalize_weights_flag', '')).lower() in ('true', '1')]
    norm_false = [r for r in rows if str(r.get('normalize_weights_flag', '')).lower() in ('false', '0')]
    if norm_true:
        pen_true = sum(1 for r in norm_true
                       if str(r.get('is_penalized', '')).lower() == 'true') / len(norm_true) * 100
        if pen_true >= 95:
            recs.append(
                f"NORMALIZE_WEIGHTS=True ALWAYS PENALIZED ({pen_true:.0f}% of {len(norm_true)} runs).\n"
                f"    normalize_weights_flag=True consistently produces poor organization metrics.\n"
                f"    Recommended: remove True from search space or set to always False."
            )

    if pen_pct > 50:
        recs.append(
            f"HIGH PENALTY RATE ({pen_pct:.0f}%): Map size min is too small.\n"
            f"    Many small maps (5x5, 6x6) train fast but fail U-Matrix/Distance thresholds.\n"
            f"    Recommended: raise map_size min from 5 to ~10 in config."
        )

    # Count how many good solutions exist at peak vs final
    if gen_data:
        peak_good = max(
            sum(1 for s in g['solutions'] if s['ratio'] is not None and s['ratio'] < 1.0)
            for g in gen_data
        )
        final_good = sum(1 for s in gen_data[-1]['solutions']
                         if s['ratio'] is not None and s['ratio'] < 1.0)
        archive_size = gen_data[-1]['count']
        if peak_good > final_good:
            recs.append(
                f"ARCHIVE REGRESSION: peak good solutions={peak_good}, final={final_good}.\n"
                f"    Good solutions were later crowded out of the archive (size cap={archive_size}).\n"
                f"    With constrained dominance, infeasible solutions should rank behind feasible ones;\n"
                f"    regression may indicate the archive is still too small or constraint thresholds\n"
                f"    are too strict (check constraint_violation column in pareto_front.csv).\n"
                f"    Recommended: increase max_archive_size or review ORGANIZATION_THRESHOLD."
            )

    # Check for non-deterministic re-evaluations (same UID, different results)
    uid_counts: dict[str, int] = defaultdict(int)
    with open(os.path.join(
        # re-read raw CSV to count duplicates before dedup
        results_dir if os.path.isabs(results_dir) else os.path.join(os.getcwd(), results_dir),
        "results.csv"
    ), encoding="utf-8") as _f:
        for _row in csv.DictReader(_f):
            uid_counts[_row.get('uid', '')] += 1
    duplicates = sum(c - 1 for c in uid_counts.values() if c > 1)
    if duplicates > 0:
        # Check if duplicates are truly non-deterministic (different values) or just parallel re-evaluations
        try:
            import pandas as _pd
            _df = _pd.read_csv(os.path.join(results_dir, "results.csv"))
            _dup_uids = _df[_df.duplicated('uid', keep=False)]['uid'].unique()
            _nondeterministic = 0
            for _u in _dup_uids:
                _rows = _df[_df['uid'] == _u][['raw_mqe_improvement_ratio', 'raw_topographic_error', 'dead_neuron_ratio']]
                if _rows.nunique().max() > 1:
                    _nondeterministic += 1
        except Exception:
            _nondeterministic = duplicates

        if _nondeterministic > 0:
            recs.append(
                f"NON-DETERMINISTIC RE-EVALUATION: {_nondeterministic} UIDs have different values across duplicate rows.\n"
                f"    Root cause: np.random.seed() applied AFTER weight init in som.py — verify the fix is in place."
            )
        else:
            recs.append(
                f"DUPLICATE EVALUATIONS: {duplicates} duplicate UID rows (identical values — seed is deterministic).\n"
                f"    Runs since the 2026-06-11 cleanup (issues.md #89) dedup offspring by gene-only UID\n"
                f"    before evaluation — duplicates indicate a pre-cleanup run or a regression."
            )

    has_dead = any(
        s['dead'] is not None
        for g in gen_data for s in g['solutions']
    ) if gen_data else False
    if not has_dead:
        recs.append(
            "LEGACY LOG FORMAT: dead_ratio not in pareto_front_log.txt.\n"
            "    Dominance check falls back to results.csv values (correct).\n"
            "    Note: new runs include Dead= in the log format."
        )

    for i, rec in enumerate(recs, 1):
        print(f"  [{i}] {rec}\n")

    if not recs:
        print("  No issues detected.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify EA run results.")
    parser.add_argument("results_dir", help="Path to EA results directory")
    parser.add_argument("--sections", nargs="+", type=int,
                        help="Run only specific sections (1-8)")
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        sys.exit(f"ERROR: directory not found: {results_dir}")

    print(f"\nEA RUN VERIFICATION")
    print(f"Directory: {os.path.abspath(results_dir)}")

    rows = load_results(results_dir)
    gen_data = parse_pareto_log(results_dir)

    run = args.sections or list(range(1, 9))

    if 1 in run: report_overview(rows)
    if 2 in run: report_map_penalty(rows)
    if 3 in run: report_archive_evolution(gen_data)
    if 4 in run: report_elitism(gen_data)
    if 5 in run: report_final_archive(gen_data, rows)
    if 6 in run: report_crowding_ejection(gen_data)
    if 7 in run: report_param_penalty(rows)
    if 8 in run: report_recommendations(rows, gen_data, results_dir)

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
