#!/usr/bin/env python3
"""
coverage_sim.py — simulate the SOM batch-sampling regime without training.

Replicates the exact sample-selection code path of KohonenSOM.train()
(app/som/som.py: permutation -> array_split into num_batches sections ->
per-iteration batch-percent decay -> ceil -> np.random.choice per section,
without replacement) and measures dataset coverage. No weights, no BMU
search — a simulated run costs seconds instead of minutes.

Purpose (docs/ea/SEARCH_SPACE.md step 1): decide whether dataset splitting
(num_batches > 1) measurably improves coverage and therefore whether the
EA should keep it, fix it, or drop it.

Null model: sections are a *random* index partition, so per-sample hit
counts should follow ~Poisson(lambda), lambda = total_updates / n_samples,
regardless of num_batches — splitting is predicted to change nothing at
equal budget. The simulation tests this prediction (and the ceil/array_split
edge effects the analytic model ignores).

IMPORTANT confound: samples_per_section is computed from TOTAL dataset
size (som.py train(): `total_samples * batch_percent / 100`), so raising
num_batches multiplies the per-iteration sample throughput by num_batches.
Use --normalize-budget in sweeps to compare configurations at equal budget.

Sampling methods:
  random    — as implemented in som.py today: np.random.choice from each
              section every iteration (with replacement *across*
              iterations).
  reshuffle — proposed alternative: walk a shuffled permutation of each
              section with a pointer, RE-SHUFFLE when exhausted (random
              reshuffling / epoch shuffling, standard in SGD). Guarantees
              per-sample hit counts equal +-1, independent of seed; the
              presentation ORDER stays random every epoch (no fixed-cycle
              periodicity). 'cycle' is accepted as a deprecated alias —
              the name was dropped because it wrongly evokes a fixed
              deterministic ring buffer (itertools.cycle).

Subcommands:
  verify  — replay a real run (fixed seed) and compare against its
            sample_coverage.json; validates that the simulator and the
            in-training tracking agree exactly.
  run     — one simulated configuration, stats to stdout.
  sweep   — grid of configurations x repeats with random seeds, CSV out.
  compare — sampling methods x epoch multipliers head-to-head on one
            dataset; reports hit-count distributions and the minimum
            multiplier reaching a target coverage per method.

Examples:
  .venv/bin/python3 app/tools/coverage_sim.py verify \
      data/datasets/WineQuality/results/20260531_153004 \
      --config data/datasets/WineQuality/config-som-coverage.json

  .venv/bin/python3 app/tools/coverage_sim.py run --size 6497 \
      --num-batches 5 --percents 1:5:linear-growth --epoch-multiplier 5

  .venv/bin/python3 app/tools/coverage_sim.py sweep \
      --csv data/datasets/WineQuality/wine.csv \
      --batches 1,2,5,10 --percents 0.5:2:linear-growth,1:5:linear-growth \
      --epoch-multiplier 5 --repeats 25 --normalize-budget \
      --out coverage_runs.csv

  .venv/bin/python3 app/tools/coverage_sim.py compare \
      --csv data/datasets/WineQuality/wine.csv \
      --methods random,reshuffle --multipliers 1,2,3,5,7,10 \
      --repeats 25 --coverage-target 0.999 --out compare_runs.csv
"""
import argparse
import json
import math
import os
import sys

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None


# --- sampling kernel (mirrors app/som/som.py — keep in sync) ---------------

def get_decay_value(t: int, N: int, start: float, end: float,
                    decay_type: str, growth_g: float) -> float:
    """Verbatim copy of KohonenSOM.get_decay_value (growth_g passed in)."""
    if N <= 1:
        return start
    if decay_type == 'static':
        return start
    elif decay_type == 'linear-drop':
        return start - (t / (N - 1)) * (start - end)
    elif decay_type == 'linear-growth':
        return start + (t / (N - 1)) * (end - start)
    elif decay_type == 'exp-drop':
        norm = (1 - np.exp(-growth_g * t / N)) / (1 - np.exp(-growth_g))
        return start - norm * (start - end)
    elif decay_type == 'exp-growth':
        return start + (end - start) * (np.exp(growth_g * t / N) - 1) / (np.exp(growth_g) - 1)
    elif decay_type == 'log-drop':
        norm = np.log(growth_g * t + 1) / np.log(growth_g * N + 1)
        return start - norm * (start - end)
    elif decay_type == 'log-growth':
        return start + (end - start) * (np.log(growth_g * t + 1) / np.log(growth_g * N + 1))
    elif decay_type == 'step-down':
        step_count = 10
        step_size = N // step_count
        current_step = min(t // step_size, step_count - 1)
        factor = 0.7 ** current_step
        return max(end, start * factor)
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}")


def simulate_run(n_samples: int, num_batches: int,
                 start_pct: float, end_pct: float, growth_type: str,
                 growth_g: float, epoch_multiplier: float,
                 seed: int | None = None,
                 replicate_init: tuple | None = None,
                 method: str = 'random') -> dict:
    """
    Run the sampling loop of KohonenSOM.train() without any SOM math.

    method='random' mirrors som.py exactly. Uses the legacy global
    np.random API on purpose: with `seed` and `replicate_init=(m, n, dim)`
    (consumes the same rand() draw as weight initialization) the random
    stream is identical to a real run, so the resulting counts must match
    sample_coverage.json exactly.

    method='reshuffle' is the proposed without-replacement variant (random
    reshuffling / epoch shuffling): each section is walked through a
    shuffled order with a pointer and RE-shuffled when exhausted, so hit
    counts are equal +-1 by construction while the presentation order
    stays random every epoch.

    MQE early stopping is ignored — it is effectively disabled in the SOM
    (docs/som/issues.md #2); a stopped run only truncates the budget.
    """
    if method == 'cycle':  # deprecated legacy name
        method = 'reshuffle'
    if method not in ('random', 'reshuffle'):
        raise ValueError(f"Unknown sampling method: {method}")
    if seed is not None:
        np.random.seed(seed)
    if replicate_init is not None:
        m, n, dim = replicate_init
        np.random.rand(m, n, dim)  # weight init draw precedes the permutation

    total_iterations = int(n_samples * epoch_multiplier)
    shuffled_indices = np.random.permutation(n_samples)
    section_indices = np.array_split(shuffled_indices, num_batches)

    if method == 'reshuffle':
        orders = [np.random.permutation(sec) for sec in section_indices]
        positions = [0] * num_batches

    counts = np.zeros(n_samples, dtype=np.int64)
    first_visit = np.full(n_samples, -1, dtype=np.int64)
    visited = 0
    iters_to_99 = -1
    threshold_99 = math.ceil(0.99 * n_samples)

    for t in range(total_iterations):
        pct = get_decay_value(t, total_iterations, start_pct, end_pct,
                              growth_type, growth_g)
        samples_per_section = max(1, math.ceil(n_samples * pct / 100.0))

        selected = []
        if method == 'random':
            for section in section_indices:
                num_to_take = min(samples_per_section, len(section))
                chosen = np.random.choice(section, num_to_take, replace=False)
                selected.extend(chosen)
        else:
            for s, section in enumerate(section_indices):
                sec_len = len(section)
                if sec_len == 0:
                    continue
                remaining = min(samples_per_section, sec_len)
                while remaining > 0:
                    take_now = min(remaining, sec_len - positions[s])
                    selected.extend(orders[s][positions[s]:positions[s] + take_now])
                    positions[s] += take_now
                    remaining -= take_now
                    if positions[s] >= sec_len:
                        orders[s] = np.random.permutation(section)
                        positions[s] = 0

        sel = np.asarray(selected)
        # reshuffle mode can wrap an epoch boundary within one iteration ->
        # sel may contain duplicates; bookkeeping must tolerate them
        newly = np.unique(sel[counts[sel] == 0])
        if len(newly):
            first_visit[newly] = t
            visited += len(newly)
        np.add.at(counts, sel, 1)
        if iters_to_99 < 0 and visited >= threshold_99:
            iters_to_99 = t + 1

    t_third = max(1, total_iterations // 3)
    unseen_after_third = float(np.mean((first_visit < 0) | (first_visit >= t_third)))

    return {
        'counts': counts,
        'first_visit': first_visit,
        'total_iterations': total_iterations,
        'iters_to_99': iters_to_99,
        # fraction of samples not yet seen during the first third of
        # training — proxy for "missed the global organization phase"
        'unseen_after_third': round(unseen_after_third, 5),
    }


# --- metrics ----------------------------------------------------------------

def gini(counts: np.ndarray) -> float:
    """Gini coefficient of per-sample hit counts (0 = perfectly even)."""
    x = np.sort(counts.astype(np.float64))
    n = len(x)
    total = x.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * index - n - 1) @ x / (n * total))


def coverage_metrics(counts: np.ndarray) -> dict:
    n = len(counts)
    total_updates = int(counts.sum())
    lam = total_updates / n
    never = int(np.sum(counts == 0))
    return {
        'n_samples': n,
        'total_updates': total_updates,
        'lambda': round(lam, 3),
        'mean': round(float(counts.mean()), 3),
        'std': round(float(counts.std()), 3),
        'min': int(counts.min()),
        'p25': float(np.percentile(counts, 25)),
        'median': float(np.median(counts)),
        'p75': float(np.percentile(counts, 75)),
        'max': int(counts.max()),
        'never': never,
        'never_ratio': round(never / n, 5),
        'poisson_never_ratio': round(float(np.exp(-lam)), 5),
        'gini': round(gini(counts), 4),
        'poisson_gini': round(_poisson_gini(lam), 4),
    }


def _poisson_gini(lam: float, sims: int = 20000) -> float:
    """Reference Gini of iid Poisson(lambda) counts (Monte Carlo)."""
    if lam <= 0:
        return 0.0
    rng = np.random.default_rng(0)
    return gini(rng.poisson(lam, size=sims))


def dimension_metrics(counts: np.ndarray, data: "pd.DataFrame",
                      bins: int = 10) -> dict:
    """
    Data-space coverage: are hits / never-visited samples concentrated in
    some region of any dimension? Index-uniform sampling predicts a flat
    profile; large deviations would support the "splitting covers dataset
    parts" claim. Reported as worst-case over numeric dimensions.
    """
    overall_never = float(np.mean(counts == 0))
    mean_hits = float(counts.mean())
    worst_spread = 0.0
    worst_never = 0.0
    for col in data.select_dtypes(include=[np.number]).columns:
        vals = data[col].to_numpy()
        try:
            edges = np.unique(np.quantile(vals, np.linspace(0, 1, bins + 1)))
            if len(edges) < 3:
                continue  # near-constant dimension
            bin_idx = np.clip(np.searchsorted(edges, vals, side='right') - 1,
                              0, len(edges) - 2)
        except (ValueError, TypeError):
            continue
        bin_means = np.array([counts[bin_idx == b].mean()
                              for b in range(len(edges) - 1)])
        spread = (bin_means.max() - bin_means.min()) / mean_hits if mean_hits else 0.0
        worst_spread = max(worst_spread, float(spread))
        if overall_never > 0:
            bin_never = np.array([np.mean(counts[bin_idx == b] == 0)
                                  for b in range(len(edges) - 1)])
            worst_never = max(worst_never, float(bin_never.max() / overall_never))
    return {
        'dim_hits_spread_max': round(worst_spread, 4),
        'dim_never_ratio_max': round(worst_never, 3),
    }


# --- summary reports ---------------------------------------------------------

def _fmt_pct(x, std=None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x * 100:.2f} %"
    if std is not None and not (isinstance(std, float) and np.isnan(std)) and std > 0:
        s += f" ± {std * 100:.2f}"
    return s


def build_compare_summary(df: "pd.DataFrame", target: float) -> str:
    """Readable markdown report for compare-style results (experiment B)."""
    df = df.copy()
    # CSVs written before the rename recorded reshuffle as 'cycle'
    df['method'] = df['method'].replace({'cycle': 'reshuffle'})
    df['met'] = (1.0 - df['never_ratio']) >= target
    lines = [
        "# Coverage comparison — methods × epoch multipliers (experiment B)",
        "",
        f"Coverage target: ≥{target:.1%} of samples visited at least once. "
        f"Total simulated runs: {len(df)}. "
        "Cell values are means across random-seed repeats.",
        "",
    ]
    conclusions = []
    for ds, dg in df.groupby('dataset'):
        n = int(dg['n_samples'].iloc[0])
        ems = sorted(dg['em'].unique())
        lines.append(f"## {ds} (N = {n})")
        lines.append("")
        lines.append("| metric | " + " | ".join(f"em={em:g}" for em in ems) + " |")
        lines.append("|---" * (len(ems) + 1) + "|")
        ds_min = {}
        for method in sorted(dg['method'].unique()):
            agg = dg[dg['method'] == method].groupby('em').agg(
                never=('never_ratio', 'mean'),
                never_std=('never_ratio', 'std'),
                hmin=('min', 'mean'), hmed=('median', 'mean'), hmax=('max', 'mean'),
                gini=('gini', 'mean'),
                unseen=('unseen_after_third', 'mean'),
                met=('met', 'mean'), reps=('met', 'size'))

            def cells(fmt):
                return " | ".join(fmt(agg.loc[em]) if em in agg.index else "—"
                                  for em in ems)

            lines.append(f"| **{method}** — never visited | "
                         + cells(lambda r: _fmt_pct(r['never'], r['never_std'])) + " |")
            lines.append(f"| {method} — hits min / median / max | "
                         + cells(lambda r: f"{r['hmin']:.1f} / {r['hmed']:.1f} / {r['hmax']:.1f}") + " |")
            lines.append(f"| {method} — gini | "
                         + cells(lambda r: f"{r['gini']:.3f}") + " |")
            lines.append(f"| {method} — unseen after first third | "
                         + cells(lambda r: _fmt_pct(r['unseen'])) + " |")
            lines.append(f"| {method} — repeats meeting target | "
                         + cells(lambda r: f"{int(round(r['met'] * r['reps']))}/{int(r['reps'])}") + " |")
            full = agg[agg['met'] >= 1.0]
            ds_min[method] = float(full.index.min()) if len(full) else None
        concl = ", ".join(f"`{m}` = {v:g}" if v is not None else f"`{m}` = not reached"
                          for m, v in ds_min.items())
        lines.append("")
        lines.append(f"**Minimum epoch_multiplier meeting the target in every repeat:** {concl}.")
        lines.append("")
        conclusions.append((ds, n, ds_min))

    methods = sorted(df['method'].unique())
    lines.append("## Conclusion — minimum epoch_multiplier per dataset")
    lines.append("")
    lines.append("| dataset (N) | " + " | ".join(f"`{m}`" for m in methods) + " |")
    lines.append("|---" * (len(methods) + 1) + "|")
    for ds, n, ds_min in conclusions:
        row = " | ".join(f"{ds_min[m]:g}" if ds_min.get(m) is not None else "not reached"
                         for m in methods)
        lines.append(f"| {ds} ({n}) | {row} |")
    lines.append("")
    lines.append(f"Analytic reference for `random` (1 sample/iteration): "
                 f"never-visited ≈ e^(−em), dataset-size independent. "
                 f"`reshuffle` guarantees full coverage from em = 1 by "
                 f"construction.")
    return "\n".join(lines)


def build_sweep_summary(df: "pd.DataFrame") -> str:
    """Readable markdown report for sweep-style results (experiment A)."""
    df = df.copy()
    # columns absent in CSVs written by older tool versions
    if 'method' not in df.columns:
        df['method'] = 'random'
    df['method'] = df['method'].replace({'cycle': 'reshuffle'})
    if 'unseen_after_third' not in df.columns:
        df['unseen_after_third'] = np.nan
    df['em'] = (df['total_iterations'] / df['n_samples']).round(2)
    lines = [
        "# Coverage sweep — splitting / method grid (experiment A)",
        "",
        f"Total simulated runs: {len(df)}. Cell values are means across "
        "random-seed repeats.",
        "",
    ]
    for (ds, norm), dg in df.groupby(['dataset', 'normalized']):
        n = int(dg['n_samples'].iloc[0])
        mode = "budget-normalized" if norm else "raw semantics"
        lines.append(f"## {ds} (N = {n}) — {mode}")
        lines.append("")
        lines.append("| method | batches | profile | em | λ | never | Poisson never "
                     "| gini | Poisson gini | unseen ⅓ |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        agg = dg.groupby(['method', 'num_batches', 'profile', 'em']).agg(
            lam=('lambda', 'mean'),
            never=('never_ratio', 'mean'),
            pnever=('poisson_never_ratio', 'mean'),
            gini=('gini', 'mean'),
            pgini=('poisson_gini', 'mean'),
            unseen=('unseen_after_third', 'mean'))
        for (method, nb, prof, em), r in agg.iterrows():
            lines.append(f"| {method} | {nb} | {prof} | {em:g} | {r['lam']:.1f} "
                         f"| {_fmt_pct(r['never'])} | {_fmt_pct(r['pnever'])} "
                         f"| {r['gini']:.3f} | {r['pgini']:.3f} "
                         f"| {_fmt_pct(r['unseen'])} |")
        lines.append("")
    lines.append("Reading: if `random` matches the Poisson reference (never and "
                 "gini) at every `num_batches`, splitting adds nothing beyond "
                 "budget (λ). `reshuffle` should show gini ≈ 0 and never = 0 "
                 "throughout.")
    return "\n".join(lines)


def write_summary(md: str, csv_path: str, md_path: str | None = None) -> str:
    path = md_path or os.path.splitext(csv_path)[0] + '_summary.md'
    with open(path, 'w') as f:
        f.write(md + '\n')
    return path


# --- CLI helpers ------------------------------------------------------------

def parse_methods(spec: str) -> list[str]:
    """Split a comma-separated method list, mapping the legacy 'cycle' name."""
    return ['reshuffle' if m.strip() == 'cycle' else m.strip()
            for m in spec.split(',')]


def parse_percent_profiles(spec: str) -> list[tuple[float, float, str]]:
    """Parse 'start:end:growth_type[,start:end:growth_type...]'."""
    profiles = []
    for part in spec.split(','):
        fields = part.split(':')
        if len(fields) != 3:
            raise argparse.ArgumentTypeError(
                f"percent profile must be start:end:growth_type, got '{part}'")
        profiles.append((float(fields[0]), float(fields[1]), fields[2]))
    return profiles


def load_dataset(args) -> tuple[int, "pd.DataFrame | None", str]:
    if args.csv:
        if pd is None:
            sys.exit("pandas is required for --csv")
        df = pd.read_csv(args.csv, delimiter=args.delimiter)
        name = os.path.splitext(os.path.basename(args.csv))[0]
        return len(df), df, name
    if args.size:
        return args.size, None, f"synthetic_{args.size}"
    sys.exit("either --csv or --size is required")


# --- subcommands ------------------------------------------------------------

def cmd_verify(args):
    with open(args.config) as f:
        cfg = json.load(f)
    cov_path = os.path.join(args.results_dir, 'csv', 'sample_coverage.json')
    with open(cov_path) as f:
        recorded = np.array(json.load(f)['counts'], dtype=np.int64)
    td = np.load(os.path.join(args.results_dir, 'csv', 'training_data.npy'),
                 mmap_mode='r')
    n_samples, dim = td.shape
    m, n = cfg['map_size']
    seed = cfg.get('random_seed')
    if seed is None:
        sys.exit("verify requires a run with a fixed random_seed in the config")
    if n_samples != len(recorded):
        sys.exit(f"size mismatch: training_data {n_samples} vs counts {len(recorded)}")

    # default must mirror KohonenSOM.__init__ (flipped to reshuffle
    # 2026-06-12); configs of older runs carry an explicit 'random'
    method, = parse_methods(cfg.get('sampling_method', 'reshuffle'))
    print(f"Replaying: N={n_samples} dim={dim} map={m}x{n} seed={seed} "
          f"method={method} num_batches={cfg['num_batches']} "
          f"pct={cfg['start_batch_percent']}->{cfg['end_batch_percent']} "
          f"({cfg['batch_growth_type']}) em={cfg['epoch_multiplier']}")
    sim = simulate_run(n_samples, cfg['num_batches'],
                       cfg['start_batch_percent'], cfg['end_batch_percent'],
                       cfg['batch_growth_type'], cfg.get('growth_g', 1.0),
                       cfg['epoch_multiplier'], seed=seed,
                       replicate_init=(m, n, dim), method=method)
    if np.array_equal(sim['counts'], recorded):
        print(f"MATCH: simulated counts identical to recorded "
              f"sample_coverage.json ({len(recorded)} samples, "
              f"{int(recorded.sum())} updates).")
        print("In-training tracking and simulator agree exactly.")
    else:
        diff = int(np.sum(sim['counts'] != recorded))
        print(f"MISMATCH: {diff}/{len(recorded)} samples differ "
              f"(sim sum={int(sim['counts'].sum())}, recorded sum={int(recorded.sum())}).")
        print("Either the run used different parameters/seed, stopped early, "
              "or the sampling code has drifted from the simulator.")
        sys.exit(1)


def cmd_run(args):
    n_samples, df, name = load_dataset(args)
    args.method, = parse_methods(args.method)
    (start_pct, end_pct, growth_type), = parse_percent_profiles(args.percents)
    sim = simulate_run(n_samples, args.num_batches, start_pct, end_pct,
                       growth_type, args.growth_g, args.epoch_multiplier,
                       seed=args.seed, method=args.method)
    stats = coverage_metrics(sim['counts'])
    stats['iters_to_99'] = sim['iters_to_99']
    stats['total_iterations'] = sim['total_iterations']
    stats['unseen_after_third'] = sim['unseen_after_third']
    if df is not None:
        stats.update(dimension_metrics(sim['counts'], df))
    print(f"dataset={name} method={args.method} "
          f"num_batches={args.num_batches} "
          f"pct={start_pct}->{end_pct} ({growth_type}) "
          f"em={args.epoch_multiplier} seed={args.seed}")
    for k, v in stats.items():
        print(f"  {k:>22}: {v}")


def cmd_sweep(args):
    if pd is None:
        sys.exit("pandas is required for sweep")
    n_samples, df, name = load_dataset(args)
    batches = [int(b) for b in args.batches.split(',')]
    profiles = parse_percent_profiles(args.percents)

    methods = parse_methods(args.methods)
    rows = []
    n_configs = len(batches) * len(profiles) * len(methods)
    print(f"Sweep: {name} (N={n_samples}), {n_configs} configurations "
          f"x {args.repeats} repeats = {n_configs * args.repeats} runs"
          f"{' [budget-normalized]' if args.normalize_budget else ''}")
    for method in methods:
        for nb in batches:
            for start_pct, end_pct, growth_type in profiles:
                s_pct, e_pct = start_pct, end_pct
                if args.normalize_budget:
                    # samples_per_section is derived from TOTAL size, so divide
                    # by num_batches to keep per-iteration throughput constant
                    s_pct, e_pct = start_pct / nb, end_pct / nb
                for rep in range(args.repeats):
                    seed = int(np.random.SeedSequence().entropy % (2**31))
                    np.random.seed(seed)  # random but recorded -> reproducible
                    sim = simulate_run(n_samples, nb, s_pct, e_pct, growth_type,
                                       args.growth_g, args.epoch_multiplier,
                                       method=method)
                    row = {
                        'dataset': name,
                        'method': method,
                        'num_batches': nb,
                        'profile': f"{start_pct}:{end_pct}:{growth_type}",
                        'normalized': args.normalize_budget,
                        'repeat': rep,
                        'seed': seed,
                        'total_iterations': sim['total_iterations'],
                        'iters_to_99': sim['iters_to_99'],
                        'unseen_after_third': sim['unseen_after_third'],
                    }
                    row.update(coverage_metrics(sim['counts']))
                    if df is not None:
                        row.update(dimension_metrics(sim['counts'], df))
                    rows.append(row)
                done = len(rows)
                print(f"  method={method} num_batches={nb} "
                      f"pct={start_pct}->{end_pct} "
                      f"({growth_type}): {args.repeats} repeats done "
                      f"[{done}/{n_configs * args.repeats}]")

    out = pd.DataFrame(rows)
    header = not os.path.exists(args.out)
    out.to_csv(args.out, mode='a', header=header, index=False)
    print(f"\nAppended {len(out)} rows to {args.out}")

    summary = out.groupby(['method', 'num_batches', 'profile']).agg(
        updates=('total_updates', 'mean'),
        never_mean=('never_ratio', 'mean'),
        never_std=('never_ratio', 'std'),
        gini_mean=('gini', 'mean'),
        gini_std=('gini', 'std'),
        poisson_never=('poisson_never_ratio', 'mean'),
        poisson_gini=('poisson_gini', 'mean'),
    ).round(4)
    print("\nSummary (compare never/gini against the Poisson reference — "
          "matching values mean splitting adds nothing):")
    print(summary.to_string())

    # regenerate the readable report from everything accumulated in the CSV
    full = pd.read_csv(args.out)
    path = write_summary(build_sweep_summary(full), args.out)
    print(f"Markdown summary (all {len(full)} accumulated rows): {path}")


def cmd_compare(args):
    """
    Sampling methods x epoch multipliers head-to-head: "we ran this dataset
    with these methods and coverage is X" + the minimum multiplier per
    method that reaches the target coverage.
    """
    if pd is None:
        sys.exit("pandas is required for compare")
    n_samples, df, name = load_dataset(args)
    methods = parse_methods(args.methods)
    multipliers = [float(m) for m in args.multipliers.split(',')]
    (start_pct, end_pct, growth_type), = parse_percent_profiles(args.percents)

    rows = []
    total = len(methods) * len(multipliers) * args.repeats
    print(f"Compare: {name} (N={n_samples}), num_batches={args.num_batches}, "
          f"pct={start_pct}->{end_pct} ({growth_type}), "
          f"{len(methods)} methods x {len(multipliers)} multipliers "
          f"x {args.repeats} repeats = {total} runs, "
          f"coverage target {args.coverage_target:.1%}")
    for method in methods:
        for em in multipliers:
            for rep in range(args.repeats):
                seed = int(np.random.SeedSequence().entropy % (2**31))
                np.random.seed(seed)
                sim = simulate_run(n_samples, args.num_batches, start_pct,
                                   end_pct, growth_type, args.growth_g, em,
                                   method=method)
                row = {
                    'dataset': name,
                    'method': method,
                    'em': em,
                    'repeat': rep,
                    'seed': seed,
                    'unseen_after_third': sim['unseen_after_third'],
                    'iters_to_99': sim['iters_to_99'],
                }
                row.update(coverage_metrics(sim['counts']))
                row['target_met'] = (1.0 - row['never_ratio']) >= args.coverage_target
                rows.append(row)
            print(f"  method={method} em={em}: {args.repeats} repeats done "
                  f"[{len(rows)}/{total}]")

    out = pd.DataFrame(rows)
    if args.out:
        header = not os.path.exists(args.out)
        out.to_csv(args.out, mode='a', header=header, index=False)
        print(f"\nAppended {len(out)} rows to {args.out}")

    summary = out.groupby(['method', 'em']).agg(
        updates=('total_updates', 'mean'),
        lam=('lambda', 'mean'),
        hits_min=('min', 'mean'),
        hits_p25=('p25', 'mean'),
        hits_med=('median', 'mean'),
        hits_p75=('p75', 'mean'),
        hits_max=('max', 'mean'),
        never_mean=('never_ratio', 'mean'),
        never_std=('never_ratio', 'std'),
        gini=('gini', 'mean'),
        unseen_3rd=('unseen_after_third', 'mean'),
        target_rate=('target_met', 'mean'),
    ).round(4)
    print(f"\nPer-method coverage vs epoch multiplier "
          f"(hits_* = mean across repeats of per-run hit-count stats):")
    print(summary.to_string())

    print(f"\nMinimum epoch_multiplier reaching >={args.coverage_target:.1%} "
          f"coverage in every repeat:")
    for method in methods:
        ok = summary.loc[method][summary.loc[method]['target_rate'] >= 1.0]
        if len(ok):
            print(f"  {method}: em = {ok.index.min():g}")
        else:
            print(f"  {method}: not reached with tested multipliers "
                  f"(max tested {max(multipliers):g})")

    # regenerate the readable report from everything accumulated in the CSV
    # (covers all datasets appended so far, not just this invocation)
    report_src = pd.read_csv(args.out) if args.out else out
    path = write_summary(build_compare_summary(report_src, args.coverage_target),
                         args.out or 'coverage_compare.csv')
    print(f"\nMarkdown summary (all {len(report_src)} accumulated rows): {path}")


def cmd_summarize(args):
    if pd is None:
        sys.exit("pandas is required for summarize")
    df = pd.read_csv(args.results_csv)
    if 'em' in df.columns:
        md = build_compare_summary(df, args.coverage_target)
    else:
        md = build_sweep_summary(df)
    print(md)
    path = write_summary(md, args.results_csv, args.md)
    print(f"\nSummary written to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = parser.add_subparsers(dest='command', required=True)

    p_verify = sub.add_parser('verify', help='replay a real fixed-seed run '
                              'and compare with its sample_coverage.json')
    p_verify.add_argument('results_dir', help='SOM results dir '
                          '(contains csv/sample_coverage.json)')
    p_verify.add_argument('--config', required=True,
                          help='the config-som JSON the run was started with')
    p_verify.set_defaults(func=cmd_verify)

    def add_common_no_em(p):
        p.add_argument('--csv', help='dataset CSV (enables data-space metrics)')
        p.add_argument('--size', type=int, help='dataset size without a CSV')
        p.add_argument('--delimiter', default=',')
        p.add_argument('--growth-g', type=float, default=1.0)

    def add_common(p):
        add_common_no_em(p)
        p.add_argument('--epoch-multiplier', type=float, required=True)

    p_run = sub.add_parser('run', help='single simulated configuration')
    add_common(p_run)
    p_run.add_argument('--num-batches', type=int, required=True)
    p_run.add_argument('--percents', required=True,
                       help='start:end:growth_type (single profile)')
    p_run.add_argument('--method', default='random',
                       choices=['random', 'reshuffle', 'cycle'],
                       help="'cycle' is a deprecated alias for 'reshuffle'")
    p_run.add_argument('--seed', type=int, default=None)
    p_run.set_defaults(func=cmd_run)

    p_sweep = sub.add_parser('sweep', help='grid x repeats with random seeds')
    add_common(p_sweep)
    p_sweep.add_argument('--batches', required=True,
                         help='comma-separated num_batches values, e.g. 1,2,5,10')
    p_sweep.add_argument('--percents', required=True,
                         help='comma-separated profiles start:end:growth_type')
    p_sweep.add_argument('--methods', default='random',
                         help='comma-separated sampling methods '
                              '(random,reshuffle)')
    p_sweep.add_argument('--repeats', type=int, default=25)
    p_sweep.add_argument('--normalize-budget', action='store_true',
                         help='divide percents by num_batches so all '
                              'configurations process the same number of '
                              'samples per iteration')
    p_sweep.add_argument('--out', default='coverage_runs.csv',
                         help='output CSV (appended across invocations)')
    p_sweep.set_defaults(func=cmd_sweep)

    p_cmp = sub.add_parser('compare', help='sampling methods x epoch '
                           'multipliers head-to-head on one dataset')
    add_common_no_em(p_cmp)
    p_cmp.add_argument('--methods', default='random,reshuffle',
                       help='comma-separated sampling methods '
                            '(random,reshuffle)')
    p_cmp.add_argument('--multipliers', required=True,
                       help='comma-separated epoch_multiplier values, '
                            'e.g. 1,2,3,5,7,10')
    p_cmp.add_argument('--num-batches', type=int, default=1)
    p_cmp.add_argument('--percents', default='0.01:0.01:static',
                       help='single profile start:end:growth_type; the '
                            'default is stochastic-like (1 sample/iteration '
                            'for datasets up to 10 000 rows)')
    p_cmp.add_argument('--repeats', type=int, default=25)
    p_cmp.add_argument('--coverage-target', type=float, default=0.999,
                       help='fraction of samples that must be visited at '
                            'least once (default 0.999)')
    p_cmp.add_argument('--out', default=None,
                       help='optional output CSV (appended)')
    p_cmp.set_defaults(func=cmd_compare)

    p_sum = sub.add_parser('summarize', help='readable markdown report from '
                           'an accumulated sweep/compare CSV')
    p_sum.add_argument('results_csv', help='CSV produced by sweep or compare '
                       '(type auto-detected)')
    p_sum.add_argument('--coverage-target', type=float, default=0.999,
                       help='target used to recompute the minimum-multiplier '
                            'conclusion (compare CSVs)')
    p_sum.add_argument('--md', default=None,
                       help='output path (default: <csv>_summary.md)')
    p_sum.set_defaults(func=cmd_summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
