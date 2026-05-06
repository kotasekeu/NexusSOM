# Struktura výstupů EA

Popis adresářové struktury a obsahu souborů generovaných Evolučním Algoritmem (EA).

---

## Organizace dat

```
data/results/
├── 01-BreastCancer/
│   ├── breast-cancer.csv          # původní dataset
│   ├── config-ea.json             # konfigurace použitá pro tento běh
│   ├── config-som.json
│   └── results/
│       └── breasts-ea-6-50-5/     # {název}-ea-{generace}-{pop}-{seeds}
├── 02-IndianLiverPatientRecords/
├── 03-LungCancerDataset/
└── 04-PimaIndiansDiabetes/
```

Každý dataset má vlastní složku s prefixem pořadového čísla. Uvnitř `results/` může být více pojmenovaných běhů (jiné konfigurace nebo verze EA).

---

## Struktura běhu EA (`{název}-ea-{gen}-{pop}-{seeds}/`)

```
breasts-ea-6-50-5/
├── log.txt                        # kompletní log celého EA běhu
├── calibration_probe.csv          # výsledky kalibračních sond (org_threshold)
├── dataset_meta.json              # metadata datasetu (n_samples, dimenze, typy sloupců)
├── csv/
│   ├── training_data.npy          # normalizovaná trénovací data (numpy, shape: n×d)
│   ├── training_data_readable.csv # čitelná verze trénovacích dat
│   ├── ignore_mask.csv            # maska ignorovaných dimenzí (0/1)
│   └── original_input.csv        # původní data před normalizací
├── json/
│   └── preprocessing_info.json   # detaily předzpracování (scalery, encodery)
├── calibration_probes/
│   └── 0/ 1/ ... 14/             # 15 kalibračních SOM tréninků
│       └── log.txt
└── seed_42/ seed_7/ seed_101/ seed_1337/ seed_2026/
```

### `dataset_meta.json`

Metadata datasetu uložená při předzpracování:

| Pole | Popis |
|---|---|
| `ds_n_samples` | Počet vzorků |
| `ds_n_original_cols` | Původní počet sloupců |
| `ds_n_active_dimensions` | Počet dimenzí použitých pro trénink |
| `ds_n_numeric` / `ds_n_categorical` | Typy dimenzí |
| `ds_n_ignored` | Ignorované sloupce |
| `ds_missing_ratio` | Podíl chybějících hodnot |
| `ds_has_primary_id` | Zda dataset obsahuje primární ID sloupec |

---

## Struktura seedy (`seed_{X}/`)

```
seed_42/
├── results.csv        # všichni vyhodnocení jedinci (hyperparametry + metriky)
├── pareto_front.csv   # Pareto-optimální jedinci na konci běhu
├── status.csv         # technický log start/konec/chyba každého vyhodnocení
├── log.txt            # log EA procesu pro tento seed
└── individuals/
    └── {md5_hash}/    # jeden jedinec = jedna SOM konfigurace
```

### `results.csv`

Hlavní soubor pro analýzu. Každý řádek = jeden vyhodnocený jedinec.

**Výkonnostní metriky:**

| Sloupec | Popis |
|---|---|
| `uid` | Hash konfigurace (= název složky v `individuals/`) |
| `raw_best_mqe` | Nejlepší MQE dosažené během tréninku |
| `raw_topographic_error` | Topografická chyba (0–1) |
| `raw_mqe_improvement_ratio` | Relativní zlepšení MQE (1 - final/initial) |
| `best_mqe` | MQE po aplikaci penalizace |
| `topographic_error` | TE po aplikaci penalizace |
| `duration` | Doba tréninku v sekundách |
| `dead_neuron_ratio` | Podíl mrtvých neuronů (0–1) |
| `constraint_violation` | Součet porušení omezení |
| `is_penalized` | Zda byl jedinec penalizován |
| `penalty_reason` | Důvod penalizace (dead neurons, org threshold) |
| `u_matrix_mean/std/max` | Statistiky U-matice |
| `distance_map_max` | Max vzdálenost v distance mapě |
| `total_weight_updates` | Celkový počet aktualizací vah |
| `epochs_ran` | Skutečný počet epoch (po early stopping) |
| `initial_mqe` | MQE na začátku tréninku |

**Hyperparametry:** `map_size`, `start_learning_rate`, `end_learning_rate`, `lr_decay_type`, `start_radius_init_ratio`, `radius_decay_type`, `start_batch_percent`, `end_batch_percent`, `batch_growth_type`, `epoch_multiplier`, `growth_g`, `num_batches`

**Metadata datasetu:** všechna pole z `dataset_meta.json` jsou zkopírována do každého řádku (prefix `ds_`)

### `pareto_front.csv`

Podmnožina `results.csv` obsahující pouze Pareto-optimální jedince z pohledu cílů EA (typicky MQE kvalita × čas). Struktura sloupců je stejná.

### `status.csv`

Technický log sledující průběh vyhodnocení:

| Sloupec | Popis |
|---|---|
| `uid` | Hash jedince |
| `population_id` | Pořadí v populaci |
| `generation` | Generace EA |
| `status` | `started` / `completed` / `failed` |
| `start_time` / `end_time` | Časy vyhodnocení |

---

## Struktura jedince (`individuals/{uid}/`)

```
0059094ec068aa3be7ec0189e4f53da4/
├── log.txt                        # log SOM tréninku tohoto jedince
├── csv/
│   ├── training_checkpoints.json  # časová řada průběhu tréninku
│   ├── weights.npy                # natrénované váhy SOM (shape: m×n×d)
│   └── weights_readable.csv      # čitelná verze vah
└── visualizations/
    ├── mqe_evolution.png          # průběh MQE v čase
    ├── learning_rate_decay.png    # průběh learning rate
    ├── radius_decay.png           # průběh sousedství
    ├── batch_size_growth.png      # průběh velikosti dávky
    ├── u_matrix.png               # U-matice natrénované mapy
    ├── distance_map.png           # distance mapa
    ├── dead_neurons_map.png       # mapa mrtvých neuronů
    └── legends/                  # samostatné soubory legend pro výše uvedené mapy
```

### `training_checkpoints.json`

Časová řada průběhu tréninku. Seznam ~500 záznamů (dle `mqe_evaluations_per_run`):

```json
[
  {
    "iteration": 0,
    "progress": 0.0,
    "mqe": 0.659,
    "topographic_error": 0.146,
    "dead_neuron_ratio": 0.847,
    "learning_rate": 0.648,
    "radius": 8.61
  },
  ...
]
```

`progress` je normalizovaný čas (0.0–1.0) nezávislý na absolutním počtu iterací — umožňuje porovnávat křivky různě dlouhých tréninků.

### `weights.npy`

Natrénované váhy SOM jako numpy array tvaru `(m, n, d)`:
- `m × n` = rozměry mapy (např. 14×14)
- `d` = počet dimenzí vstupu

---

## Škálování dat

Přibližný objem dat na dataset (5 seedů, ~290 jedinců/seed):

| Typ dat | Počet |
|---|---|
| Jedinci celkem | ~1 450 |
| Checkpointů celkem | ~725 000 |
| Weight souborů | ~1 450 `.npy` |
| Vizualizací | ~10 150 `.png` |
