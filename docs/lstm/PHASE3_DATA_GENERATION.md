# Phase 3 — Generování perturbačních dat pro LSTM controller

**Verze**: 1.0  
**Aktualizováno**: 2026-05-16  
**Skript**: `app/lstm/generate_phase3_data.py`

---

## Proč tato data existují

LSTM Phase 3 controller musí vědět, **co se stane, když uprostřed SOM tréninku zvýším learning rate o 10 % nebo snížím radius o 15 %**. Bez příkladů "v bodě X jsem změnil parametr, výsledek byl Y" se LSTM controller nemůže naučit nic smysluplného — má data pouze z tréninků s fixním schedlem, kde se parametry nikdy dynamicky nezměnily.

Řešení: vzít nejlepší hotové SOM konfigurace (Pareto individua z EA) a znovu je spustit — jednou čistě jako referenci, a pak N-krát s náhodnou perturbací parametrů v každém checkpointu. Tím vzniknou **trajektorie** s měřitelným dopadem dynamické změny na finální kvalitu.

---

## Konceptuální analogie k EA

Celý proces je myšlením podobný jedné generaci EA — ale místo prohledávání prostoru hyperparametrů prohledáváme **prostor dynamických zásahů** do parametrů během tréninku jednoho konkrétního individua.

```
EA prohledává:            Perturbace prohledává:
─────────────────         ──────────────────────────────────────
Jaká konfigurace?         Jaký SOM? → již víme (Pareto individuum)
  (map_size, lr, …)       Co se změní? → dynamické faktory lr_f, radius_f
                          Kdy? → náhodně v každém MQE checkpointu (prob=0.4)
                          Jak moc? → ±25 % multiplikativně
                          Výsledek? → delta finální kvality vs baseline
```

Klíčový rozdíl od EA: tady neoptimalizujeme konfiguraci — konfiguraci již máme (top Pareto). Zkoušíme, **jak reaguje tato konkrétní konfigurace na dynamické úpravy v průběhu tréninku**.

### Příklad

Vezměme Pareto individuum `30e740aa` z LungCancer datasetu:
- Baseline (čistý run, statický schedule): `MQE = 1.6492`
- Variant 1 (lr+radius perturbace, seed=17283): `MQE = 1.4764` → `Δ = +0.1728` (lepší)
- Variant 2 (pouze radius, seed=44912): `MQE = 1.7103` → `Δ = -0.0611` (horší)
- ...

LSTM controller se naučí z těchto příkladů: **jaké zásahy v jakém momentu tréninku vedly ke zlepšení**.

---

## Struktura jednoho běhu

### Baseline run

Spustí SOM s konfigurací Pareto individua — **bez jakékoliv perturbace**, čistý deterministický schedule. Výsledkem je referenční `final_mqe`, vůči které se porovnávají všechny varianty.

### Variant run (perturbovaný)

Stejná konfigurace, ale při každém MQE checkpointu (cca každých 300 SOM iterací) se s pravděpodobností **PERTURB_PROB = 0.4** (40 %) aplikuje náhodný multiplikátor:

```
lr_factor     = U(0.75, 1.25)   # ±25 % learning rate
radius_factor = U(0.75, 1.25)   # ±25 % radius
```

Výsledný parametr se nenahrazuje — **násobí** aktuální hodnotu ze statického schedlu:

```
actual_lr     = static_schedule_lr(t)     * lr_factor
actual_radius = static_schedule_radius(t) * radius_factor
```

Tím perturbace nedestruuje průběh — jen ho mírně posouvá nahoru nebo dolů v každém okamžiku.

### Tři typy perturbace

| Typ | Co se mění | Použití |
|---|---|---|
| `lr+radius` | oba parametry současně | testuje interakci |
| `lr_only` | pouze learning rate | izoluje efekt LR |
| `radius_only` | pouze radius | izoluje efekt radius |

Varianty se střídají cyklicky (`v_idx % 3`), takže při 8 variantách na individuum vzniknou 3× lr+radius, 3× lr_only, 2× radius_only.

---

## Delta MQE — co znamená

```
delta_mqe = baseline_mqe - perturbed_mqe
```

| delta_mqe | Interpretace |
|---|---|
| **> 0** | perturbovaný běh skončil s **nižším** (lepším) MQE — dynamická změna pomohla |
| **= 0** | stejný výsledek jako baseline |
| **< 0** | perturbace zhoršila výsledek — tento typ zásahu byl nevhodný |

Příklad z LungCancer, seed 42 (5 individuí, 8 variant, 45 celkem):
- 29/40 perturbovaných běhů bylo lepší než baseline (72 %)
- `delta_mqe` se pohybovalo přibližně v rozsahu −0.08 až +0.23

Vysoký podíl "lepších" (>50 %) je žádoucí — znamená, že dynamické úpravy mají potenciál zlepšit výsledek. Pokud by bylo jen 10 % lepších, data by obsahovala převážně záporné příklady a model by se naučil neprovádět žádné zásahy.

---

## Výstupní formát — trajektorie

Každá trajektorie odpovídá jednomu SOM běhu (baseline nebo variant). Soubor `app/lstm/data/phase3/trajectories.json` je seznam objektů:

```json
{
  "uid":                  "30e740aa",
  "variant":              "lr+radius",
  "variant_seed":         17283,
  "final_mqe":            1.4764,
  "delta_mqe":            0.1728,
  "better_than_baseline": true,
  "checkpoints": [
    {
      "iteration":         300,
      "progress":          0.074,
      "mqe":               1.9213,
      "topographic_error": 0.12,
      "dead_neuron_ratio": 0.23,
      "learning_rate":     0.81,
      "radius":            14.2,
      "lr_factor":         1.18,
      "radius_factor":     0.94
    },
    ...
  ]
}
```

| Pole | Typ | Popis |
|---|---|---|
| `uid` | string | ID Pareto individua z EA |
| `variant` | string | `"baseline"`, `"lr+radius"`, `"lr_only"`, `"radius_only"` |
| `variant_seed` | int | seed použitý pro RNG perturbace |
| `final_mqe` | float | výsledné MQE tohoto běhu |
| `delta_mqe` | float | `baseline_mqe − final_mqe` (kladné = lepší) |
| `better_than_baseline` | bool | `delta_mqe > 0` |
| `checkpoints` | list | sekvence MQE checkpointů (typicky 25 na run) |
| `checkpoints[i].lr_factor` | float | aplikovaný LR multiplikátor v tomto bodě |
| `checkpoints[i].radius_factor` | float | aplikovaný radius multiplikátor v tomto bodě |

Trajektorie baselines mají `lr_factor = 1.0` a `radius_factor = 1.0` vždy — slouží jako referenční bod, ale nejsou zahrnuty do trénovacích příkladů (nemají delta k učení).

---

## Paralelizace — jak to funguje

Každý SOM trénink je nezávislý (žádné sdílené mutable state). Celá sada 45 runů se proto rozdělí do poolu procesů:

```
5 Pareto individuí × (1 baseline + 8 variant) = 45 úloh
Pool(processes=11)  →  ~4–5 běhů paralelně
Celkový čas: ~400 s místo ~2 000 s sekvenčně
```

Každý worker dostane `task` dict se vším potřebným:

```python
task = {
    'uid':           '30e740aa',
    'row_dict':      { ... },        # hyperparametry z results.csv
    'data':          np.ndarray,     # normalizovaná trénovací data
    'ignore_mask':   np.ndarray,
    'perturb_config': {'type': 'lr+radius', 'seed': 17283},  # nebo None pro baseline
    'som_seed':      17283,
    'variant':       'lr+radius',
    'variant_seed':  17283,
}
```

**Proč `perturb_config` jako dict a ne funkce**: Python multiprocessing serializuje úlohy přes pickle. Lambda funkce a closures nelze pickle-ovat. Worker proto dostane serializovatelný dict a perturbační funkci si **zkonstruuje sám** uvnitř procesu.

Po dokončení všech workerů se výsledky sesumírují a delta se spočítá porovnáním s baselinemi (které jsou v kolekci výsledků jako ostatní záznamy).

### Příkaz

```bash
# Všechny datasety + všechny seedy najednou (doporučeno)
.venv/bin/python3 app/lstm/generate_phase3_data.py \
    --results_root data/datasets \
    --n_pareto 5 --n_variants 8 \
    --n_workers 0 \
    --output app/lstm/data/phase3

# Jeden seed (debug / ověření)
.venv/bin/python3 app/lstm/generate_phase3_data.py \
    --seed_dir data/datasets/LungCancerDataset/results/20260515_111812/seed_42 \
    --dataset  data/datasets/LungCancerDataset/dataset.csv \
    --n_pareto 5 --n_variants 8
```

`--n_workers 0` (výchozí) použije `cpu_count − 1` automaticky.

### Výpis při běhu

Progress bary se promíchají (paralelní stdout), ale po dokončení poolu se vytiskne čistý souhrn za každé individuum:

```
Running 45 SOM trains across 11 workers...
SOM Training: 100%|...| 4020/4020 [00:29<...] best_mqe=1.704192
SOM Training: 100%|...| 4020/4020 [00:30<...] best_mqe=1.692170
...
All runs done in 401.2s  (8.9s/run avg)

  ── Individual 30e740aa ── baseline MQE=1.6492  best Δ=+0.1728  better: 5/8
  ── Individual a1b2c3d4 ── baseline MQE=1.8821  best Δ=+0.0341  better: 3/8
  → 45 trajectories, 29/40 perturbed better than baseline (72%)
```

---

## Co se děje při --results_root (více seedů)

Skript projde celou adresářovou strukturu:

```
data/datasets/
  LungCancerDataset/
    dataset.csv          ← auto-detekováno
    results/
      20260515_111812/
        seed_42/         ← zpracováno
        seed_101/        ← zpracováno
        ...
  BreastCancer/
    dataset.csv
    results/
      .../seed_*/        ← zpracováno
```

Trajektorie z každého seedu se akumulují v paměti. Na konci se všechny zapíšou do **jediného** `trajectories.json` (přepíše předchozí). Seed za seedem se zpracovávají sekvenčně (dataset se načítá vždy jen jeden), ale uvnitř každého seedu SOM tréninky běží paralelně.

---

## Co LSTM controller z těchto dat učí

Po přípravě přes `prepare_phase3_dataset.py` vznikne trénovací dataset:

```
X:   (N, T, 6)   — sekvence checkpointů (progress, mqe_rel, te, dead, lr_rel, radius_rel)
y:   (N, T, 2)   — aplikované faktory (lr_factor, radius_factor)
ctx: (N, 4)      — kontext datasetu (n_samples, n_active_dim, n_numeric, n_categorical)
adv: (N,)        — normalized advantage = max(0, delta_mqe) / max_delta ∈ [0, 1]
```

Advantage-weighted loss: příklady s vysokým `delta_mqe` (perturbace hodně pomohla) dostávají vyšší váhu. Příklady, kde perturbace nezměnila výsledek nebo zhoršila, dostávají váhu 0 — model se z nich neučí "co nedělat", jen ignoruje šum.

Cílem tréninku není naučit se konkrétní faktory (které byly náhodné) — ale naučit se, **jaký stav konvergence** (progress, MQE trend, topographic error) koreluje s výhodnou intervencí, a predikovat faktory, které tento stav zlepší.

---

## Parametry perturbace

| Konstanta | Hodnota | Popis |
|---|---|---|
| `LR_PERTURB` | 0.25 | Maximální odchylka LR faktoru od 1.0 |
| `RADIUS_PERTURB` | 0.25 | Maximální odchylka radius faktoru od 1.0 |
| `PERTURB_PROB` | 0.4 | Pravděpodobnost perturbace v každém checkpointu |

Při 25 checkpointech na run a pravděpodobnosti 0.4 je průměrně 10 perturbovaných bodů z 25. Kumulativní drift po 25 checkpointech s průměrnou odchylkou ±12.5 % na krok může dosáhnout ±30–40 % od statického schedlu — dostatečný signál pro trénink, ale ne tak velký, aby destabilizoval SOM.

---

## Návaznost — co spustit po vygenerování dat

```bash
# 1. Příprava datasetu pro trénink
.venv/bin/python3 app/lstm/prepare_phase3_dataset.py

# 2. Trénink LSTM Phase 3 controller
cd app/lstm && ../.venv/bin/python3 src/train_phase3.py

# 3. Volitelně: vizualizace
../.venv/bin/python3 visualize_model.py
```

Model se uloží jako `app/lstm/models/lstm_controller_latest.keras`.  
Aktivuje se v konfigu přes `"use_lstm_controller": true`.
