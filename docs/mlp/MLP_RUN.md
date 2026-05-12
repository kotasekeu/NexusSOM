# MLP — Postup spuštění od EA výsledků po nasazení

Kompletní průchod: sběr dat z EA → příprava datasetu → trénink → aktivace v EA.

---

## Přehled

MLP predikuje kvalitu SOM konfigurace (MQE improvement ratio, topographic error, dead neuron ratio) **bez spuštění SOM tréninku**. EA ho používá jako pre-screen filtr — konfigurace s předpokládanou nízkou kvalitou jsou přeskočeny a ušetří se výpočetní čas.

```
data/results/           →  prepare_dataset.py  →  all_combined_mlp.csv
                                                         ↓
                                                    train.py
                                                         ↓
                                              mlp_latest.keras
                                              mlp_scaler_latest.pkl
                                                         ↓
                                              config-ea.json → EA run
```

---

## Předpoklady

- Alespoň 3–4 dokončené EA běhy v `data/results/` (různé datasety)
- Každý běh musí mít `seed_*/results.csv` se sloupcem `raw_mqe_improvement_ratio`
- Python prostředí s TensorFlow, scikit-learn, pandas, joblib

---

## Krok 1 — Příprava trénovacího datasetu

Spustit z kořene projektu:

```bash
python3 app/mlp/prepare_dataset.py --results_root data/results
```

**Co skript udělá:**
1. Projde všechny `data/results/*/results/*/seed_*/results.csv`
2. Sloučí data ze všech datasetů a seedů
3. Vyloučí penalizované jedince (`is_penalized=True`)
4. Vyloučí konfigurace s `batch_growth_type=log-growth` (odstraněno ze search space)
5. Aplikuje fixní one-hot schéma pro kategorické parametry
6. Uloží výsledek

**Výstup:**
- `app/mlp/data/all_combined_mlp.csv` — trénovací data
- `app/mlp/data/all_combined_mlp_metadata.json` — schéma features a targets

**Volitelné flagy:**

| Flag | Popis |
|---|---|
| `--include_penalized` | Zahrne i penalizované jedince |
| `--include_legacy_types` | Zahrne konfigurace s `log-growth` (starší data) |
| `--output cesta.csv` | Vlastní výstupní cesta |

**Kontrola výstupu:**
```
Samples:  3586
Features: 25
Targets:  ['raw_mqe_improvement_ratio', 'raw_topographic_error', 'dead_neuron_ratio']
```

---

## Krok 2 — Trénink MLP

```bash
cd app/mlp
python3 src/train.py
```

Nebo s vlastním datasetem a parametry:
```bash
python3 src/train.py --dataset data/all_combined_mlp.csv --epochs 300 --batch-size 32
```

**Co skript udělá:**
1. Načte dataset a metadata
2. Stratifikovaný split podle datasetu (70 % train / 15 % val / 15 % test)
3. StandardScaler fit na trénovacích datech
4. Trénink s early stopping (patience=30), ReduceLROnPlateau (patience=15)
5. Uloží model a scaler do dvou sad cest:
   - **Timestampované** (`models/mlp_standard_YYYYMMDD_HHMMSS_best.keras`) — archiv
   - **Stabilní** (`models/mlp_latest.keras`, `models/mlp_scaler_latest.pkl`) — pro EA config

**Volitelné flagy:**

| Flag | Default | Popis |
|---|---|---|
| `--model` | `standard` | Architektura: `standard` (256→128→64→32) nebo `lite` (128→64→32) |
| `--epochs` | `300` | Max. počet epoch (early stopping typicky zastaví dříve) |
| `--batch-size` | `32` | Velikost dávky |
| `--learning-rate` | `0.001` | Learning rate Adamu |

**Orientační výsledky na 4 datasetech (~3 500 vzorků):**

| Target | MAE | Popis |
|---|---|---|
| `mqe_improvement_ratio` | ~0.05 | Chyba ~5 p.b. na škále 0–1 |
| `topographic_error` | ~0.019 | Chyba ~1.9 p.b. |
| `dead_neuron_ratio` | ~0.018 | Chyba ~1.8 p.b. |

---

## Krok 3 — Vyhodnocení modelu (volitelné)

```bash
cd app/mlp
python3 evaluate_predictions.py
```

Vypíše MAE a RMSE per target a ukázkové predikce vs. skutečné hodnoty na testovacím setu.

---

## Krok 4 — Konfigurace EA

Aktualizuj `NEURAL_NETWORKS` sekci v `config-ea.json` konkrétního datasetu:

```json
"NEURAL_NETWORKS": {
  "use_mlp": true,
  "mlp_model_path": "C:/cesta/k/projektu/app/mlp/models/mlp_latest.keras",
  "mlp_scaler_path": "C:/cesta/k/projektu/app/mlp/models/mlp_scaler_latest.pkl",
  "mlp_filter_bad_configs": true,
  "mlp_bad_quality_threshold": 0.5,
  "verbose": true
}
```

**Parametry:**

| Parametr | Popis |
|---|---|
| `use_mlp` | Zapne načtení modelu. Samotné načtení nezpomaluje EA — predikce proběhne pouze když je zapnuto filtrování |
| `mlp_filter_bad_configs` | Zapne pre-screen filtr. Konfigurace s predikovaným `mqe_improvement_ratio < threshold` jsou přeskočeny (nízké predikované zlepšení = špatná konfigurace) |
| `mlp_bad_quality_threshold` | Práh pro přeskočení. Hodnota 0.5 = přeskoč konfigurace kde model predikuje zlepšení MQE < 50 %. Doporučené rozmezí: 0.4–0.6 |
| `verbose` | Loguje každé přeskočení do logu jedince |

**Poznámka k cestám:** Cesty musí být absolutní nebo relativní k pracovnímu adresáři při spuštění EA (`run_ea.py`). Na Windows používej zpětná lomítka nebo raw stringy.

---

## Krok 5 — Spuštění EA s MLP

```bash
python3 app/run_ea.py --config data/datasets/BreastCancer/config-ea.json
```

V logu EA uvidíš potvrzení načtení:
```
INFO: NN integration enabled — MLP=True, LSTM=False, CNN=False
✓ MLP model and scaler loaded
```

A při filtrování:
```
MLP pre-screen: predicted MQE=0.623 > threshold=0.500, skipping SOM training
```

---

## Feature vektor (25 dimenzí)

Co MLP dostane jako vstup při predikci v EA — musí přesně odpovídat schématu z tréninku:

**Numerické (10):**
`start_learning_rate`, `end_learning_rate`, `start_radius_init_ratio`, `start_batch_percent`, `end_batch_percent`, `epoch_multiplier`, `growth_g`, `num_batches`, `map_m`, `map_n`

**Dataset kontext (5):**
`ds_n_samples`, `ds_n_active_dimensions`, `ds_n_numeric`, `ds_n_categorical`, `ds_missing_ratio`

**One-hot kategorické (10):**
`lr_decay_type_{exp-drop,linear-drop,log-drop,step-down}`,
`radius_decay_type_{exp-drop,linear-drop,log-drop,step-down}`,
`batch_growth_type_{exp-growth,linear-growth}`

---

## Opakované přetrénování po nových EA datech

Workflow po doběhnutí dalšího datasetu:

```bash
# 1. Aktualizuj dataset
python3 app/mlp/prepare_dataset.py --results_root data/results

# 2. Přetrénuj model (přepíše mlp_latest.keras)
cd app/mlp && python3 src/train.py

# 3. Zkontroluj kvalitu
python3 evaluate_predictions.py
```

Config-ea.json **není potřeba měnit** — cesty ke stabilním souborům (`mlp_latest.keras`, `mlp_scaler_latest.pkl`) zůstávají stejné. Stačí spustit EA znovu.

Timestampované zálohy v `app/mlp/models/` zůstávají — je možné se vrátit ke staršímu modelu přejmenováním souboru na `mlp_latest.keras`.

---

## Časté problémy

**`MLP scaler not found. MLP disabled.`**
Cesta v `mlp_scaler_path` neexistuje nebo je špatná. Ověř absolutní cestu.

**Predikce jsou konstantní / nesmyslné**
Dataset kontext (`ds_n_samples` atd.) není předán do `encode_config_for_mlp`. Ověř, že `nn_integration.py` čte tyto hodnoty z `dataset_meta.json`, ne z konfigurace jedince.

**`KeyError: 'raw_mqe_improvement_ratio'`**
Starší výsledky EA nemají tento sloupec. Přesuň je do archivu a nech jen validní běhy s novou verzí EA.

**MAE pro IndianLiverPatientRecords výrazně horší**
Normální při tréninku na malém počtu datasetů. Přidáním více různorodých datasetů se zobecnění zlepší.
