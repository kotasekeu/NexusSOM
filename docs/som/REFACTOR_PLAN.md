# SOM modul — plán vyčištění a rozdělení (branch `4-som-cleanup`)

**Datum**: 2026-06-10
**Baseline**: testy `tests/unit/test_som_*.py` + `tests/integration/test_som_*.py`
zachycují chování a kontrakty modulu — průvodce testy viz `TESTING.md`.
Po každé fázi refaktoru musí celá suite projít.

---

## 1. Inventura — co v modulu je

| Soubor | Řádky | Stav |
|---|---|---|
| `som.py` (KohonenSOM) | 581 | jádro, zdravé; obsahuje file IO které tam nepatří |
| `preprocess.py` | 233 | funkční; vrací cestu k .npy místo dat, mutuje config |
| `analysis.py` | 218 | BMU přiřazení + extrémy + pie data; OK |
| `visualization.py` | 404 | funkční; vyžaduje živý KohonenSOM objekt |
| `graphs.py` | 82 | funkční; 1 bug |
| `run.py` | 238 | orchestrace CLI; importy mimo modul |
| `utils.py` | 40 | OK |
| `result_analyzer.py` | 302 | **mrtvý kód** — plně nahrazen `app/analysis/src/`, nikdo neimportuje |

### Mrtvý kód k odstranění

- `result_analyzer.py` — celý soubor (nahrazen `app/analysis/src/context.py`)
- `KohonenSOM._get_neighbors()` — 0 použití (nahrazeno vektorizovaným TE)
- `KohonenSOM.grid_distance()` — 0 použití; navíc hex větev nepoužívá
  row-offset konzistentně s `cube_coords` (další důvod smazat, ne opravit)

### Nalezené bugy (opravit při refaktoru)

| # | Bug | Kde |
|---|---|---|
| B1 | `generate_all_maps` hledá `clusters.json` v rootu výsledků, ale `perform_analysis` ho ukládá do `json/` → **cluster_map.png se nikdy negeneruje** | `visualization.py:396` |
| B2 | `graphs.py` čte `training_results.get('best MQE')`, klíč je `best_mqe` → bod nejlepšího MQE se nikdy nevykreslí | `graphs.py:51` |
| B3 | `run_metrics.json` má `topographic_error: null`, pokud neběží checkpointy (TE se bere jen z posledního checkpointu místo přímého výpočtu) | `run.py:185` |
| B4 | primární ID sloupec se klasifikuje do `numerical_column` → `_detect_extremes` hlásí ID jako outlier (downstream se filtruje textově) | `preprocess.py` |
| B5 | `_get_bmu_assignments` anotace `-> pd.DataFrame`, vrací tuple | `analysis.py:60` |

### Vazby ven z modulu (blokují samostatnost)

- `run.py` → `analysis.src.context` (llm_context) a `ea.nn_integration` (LSTM hooky)
- `som.py train()` → zápisy na disk (weights.npy, checkpoints, coverage) + `log_message`

### Vazby dovnitř (kdo modul používá — nesmí se rozbít)

- `ea/ea.py`: `preprocess`, `KohonenSOM` (+ metriky), `graphs`, `generate_individual_maps`
- `lstm/generate_phase3_data.py`: `KohonenSOM`, `preprocess_data`
- UI spouští `run_som.py` jako subprocess

---

## 2. Cílová architektura

Princip: **SOM core = čistá funkce nad poli.** Vstup `(data, mask, hyperparametry)`,
výstup `(weights, metriky, historie)`. Vše ostatní jsou vyměnitelné vrstvy okolo
— přesně to, co ablation study potřebuje zapínat/vypínat a UI volat samostatně.

```
app/som/
├── __init__.py        # veřejné API modulu (jediný kontrakt pro EA/LSTM/UI)
├── som.py             # KohonenSOM — POUZE algoritmus + metriky, žádný disk IO
├── preprocess.py      # vstupní fáze; vrací (data, mask, info, stats) — ne cesty
├── analysis.py        # BMU přiřazení, extrémy, pie data (beze změny role)
├── persistence.py     # NOVÉ: ukládání results dir (weights, checkpoints,
│                      #   coverage, run_metrics, exporty) — vytaženo z train()
├── visualization.py   # rendering z (weights, geometry) NEBO z results dir
├── graphs.py          # tréninkové grafy z historie
├── utils.py           # log, config
└── run.py             # CLI orchestrátor; jediné místo s vazbami ven
                       #   (analysis.src llm_context, ea.nn_integration hooky)
```

### Klíčová rozhodnutí

1. **`train()` přestane sahat na disk.** Ukládání weights/checkpoints/coverage
   se přesune do `persistence.py`, volá ho `run.py`. EA a multi-seed tool pak
   můžou trénovat in-memory bez working_dir (dnes EA platí IO daň za každého
   jedince).
2. **Preprocess jako vyměnitelná strategie.** Kontrakt:
   `preprocess(df, config) → PreprocessResult(data, mask, info, stats)`.
   Implementace: `nexus` (současná), `scale-only` (jen MinMax, bez masky
   a klasifikace), `none` (raw hodnoty). Volba `preprocess_strategy` v configu.
   → ablace A1.1 (rozpad bez preprocessingu) je pak jeden přepínač, a stejným
   mechanismem lze porovnat náš preprocess s cizí přípravou dat.
3. **Vizualizace bez živého SOM objektu.** Mapové funkce přijmou
   `(weights, map_type)` místo `KohonenSOM` instance (geometrie se odvodí
   z `weights.shape`). Tím lze renderovat **kterýkoliv uložený běh** z
   `weights.npy` — přesně to potřebuje Streamlit UI pro interaktivní
   porovnávání ablation výsledků bez re-tréninku.
4. **NN hooky zůstávají injektované callbacky** (`lstm_early_stop_fn`,
   `dynamic_schedule_fn`) — import `ea.nn_integration` se přesune výhradně do
   `run.py`. `som.py` samo o žádné NN neví → modul je samostatný.
5. **Žádný `processing_type`** — hybrid s parametry batchování pokrývá
   stochastic i deterministic (issues.md #6); ablace A1.2 = tři configy.

### Kontrakt pro ablation study a UI

- Jeden běh = jeden results dir se stabilním layoutem (vynucuje e2e test
  `test_som_pipeline.py::test_output_layout`).
- Programové API: `run.py` vystaví `run_pipeline(input_csv, config, output_dir,
  seed=None) → results_dir` — multi-seed tool (article_implementation bod 1)
  i Streamlit „spustit organizaci" volají tohle, ne subprocess parsing.
- UI porovnávání: čte N results dirů (nebo DB přes API) + renderuje mapy
  z uložených weights přes bod 3.

---

## 3. Postup (fáze, po každé zelená suite)

| Fáze | Obsah | Testy | Stav |
|---|---|---|---|
| **F1 úklid** | smazat `result_analyzer.py`, `_get_neighbors`, `grid_distance`; opravit B1–B5; doplnit anglické docstringy kde chybí | stávající baseline musí projít (B1/B2 testy rozšířit o cluster_map + best-MQE bod) | ✅ 2026-06-10 |
| **F2 čisté jádro** | vytáhnout disk IO z `train()` do `persistence.py`; `preprocess_data` vrací data místo cesty (+ přestane mutovat config — vrací `PreprocessResult`) | upravit baseline na nový kontrakt, e2e beze změny | ✅ 2026-06-10 |
| **F3 nezávislost** | vizualizace z `(weights, map_type)`; importy ven z modulu jen v `run.py`; `__init__.py` definuje veřejné API; aktualizovat EA/LSTM importy | nové unit testy vizualizace z .npy | ✅ 2026-06-10 |
| **F4 ablace/UI API** | `preprocess_strategy` přepínač; `run_pipeline()` funkce; multi-seed tool nad ní | nové testy strategie + run_pipeline | ✅ 2026-06-10 |

### Poznámky k F2 (provedeno)

- `train(data, ignore_mask, lstm_early_stop_fn, dynamic_schedule_fn, log_fn)` —
  žádné `working_dir`, žádné zápisy; coverage counts jsou v návratové hodnotě.
- `preprocess_data(df, config, log_fn)` → `PreprocessResult`; artefakty ukládá
  `persistence.save_preprocess_artifacts`. Config se nemutuje — `run.py` skládá
  `runtime_config` explicitně.
- Nové volby: `show_progress` (SOM, default True; EA/batch nástroje vypínají),
  `save_training_plots` + `save_visualizations` (run.py, default True),
  `generate_training_plots` + `save_individual_weights` (EA FIXED_PARAMS;
  weights per individual se nově **defaultně neukládají** — od uzavření CNN je
  nikdo nečte, spatial analýza běží nad `som.weights` v paměti).
- NumPy optimalizace: U-Matrix vektorizovaná pro square i hex
  (`visualization.compute_u_matrix`, ekvivalence s původní smyčkou pokryta
  testem); odstraněn O(kategorie²) lookup v pie datech. Hlavní tréninková
  smyčka je záměrně sekvenční (online SOM updaty nelze dávkovat bez změny
  sémantiky algoritmu).
- ⚠ `KohonenSOM.calculate_u_matrix_metrics()` dál počítá 4-sousedovou
  aproximaci i pro hex mapy — ponecháno kvůli srovnatelnosti s 5 853
  existujícími EA individui (metriky krmí MLP). Přesný výpočet je
  `visualization.compute_u_matrix`; sjednotit až s vědomým rozhodnutím
  o přerušení kontinuity dat.

### Poznámky k F3 (provedeno)

- Celá `visualization.py` pracuje z `(weights, map_type)` — žádná funkce už
  nebere živý `KohonenSOM`. Vnitřní numpy helpery `_bmu_indices` a
  `_neuron_qe_map` nahrazují volání metod core objektu.
- **`render_results_dir(results_dir)`** — nový vstupní bod pro UI/ablaci:
  znovu vyrenderuje všechny mapy uloženého běhu čistě z artefaktů
  (weights.npy, training_data.npy, ignore_mask.csv, preprocessing_info.json);
  klasifikaci sloupců rekonstruuje z preprocessing_info.
- `som/__init__.py` definuje veřejné API (`KohonenSOM`, `preprocess_data`,
  `validate_input_data`, `PreprocessResult`, `persistence`); matplotlib vrstvy
  se záměrně nereexportují.
- Izolace ověřena: jediný import ven z balíčku je `analysis.src.context`
  v `run.py` (+ lazy `ea.nn_integration` tamtéž) — přesně dle plánu.

### Poznámky k F4 (provedeno)

- **`preprocess_strategy`** v configu: `nexus` (default, plná pipeline) |
  `scale-only` (bez noise exclusion a bez masky, s normalizací — izoluje
  přínos masky) | `none` (jen encoding, bez normalizace — očekávaný rozpad,
  ablace A1.1). Strategie se zapisuje do `dataset_meta.json`
  (`ds_preprocess_strategy`).
- **`som.run.run_pipeline(input_path, config, output_dir=None, seed=None)`** —
  programové API celé pipeline; nemutuje předaný config, seed override pro
  multi-seed běhy; CLI `run_som.py` je nad ním (+ nový `-s/--seed` argument).
  Reprodukovatelnost stejného seedu pokryta testem.
- **`app/tools/multi_seed_som.py`** — multi-seed nástroj (article_implementation
  bod 1): N běhů přes `run_pipeline`, porovnává finální metriky (mean ± std),
  překryv MQE křivek a **stabilitu clusterizace přes pairwise Adjusted Rand
  Index**; výstup `multi_seed_metrics.csv` + `multi_seed_summary.json` +
  `mqe_evolution_comparison.png`. Defaultně negeneruje mapy/grafy
  (`--with-maps` je zapne).

Dokumentace `CONFIG.md`, `RUN.md`, `RESULTS.md` přepsána podle finálního
stavu (2026-06-10) — refaktoring SOM modulu je tím kompletní.

Dokumentaci (`CONFIG.md`, `RUN.md`, `RESULTS.md`) přepsat po F4 podle finálního stavu.

---

## 4. Stav dokumentace docs/som (audit 2026-06-10)

| Soubor | Stav |
|---|---|
| `issues.md` | ✅ aktuální a hodnotný (mj. zdůvodnění ignore mask, vypnutého early stoppingu, zrušení processing_type) |
| `ANOMALY_VALIDATION.md` | ✅ aktuální — popis trips validace + per-dim QE; vstup pro článek/ablaci |
| `article_*.md` | 🗑 `article_response.md` a `article_analysis.md` smazány 2026-06-11 (obsah destilován do `article_implementation.md`, klíčový poznatek o topologickém kolapsu → issues #22); `article_implementation.md` zůstává, dokud nejsou body 2–8 implementovány |
| `SOM_REQUIREMENTS.md` | 🗑 smazáno 2026-06-11 — zastaralá FR specifikace (tři tréninkové módy, „hotový" early stopping, již odstraněné metody); platná fakta pokrývají CONFIG/TESTING/issues |
| `CONFIG.md` | ✅ přepsáno 2026-06-10 podle finálního stavu (vč. `preprocess_strategy`, output control, NN sekce) |
| `RUN.md` | ✅ přepsáno 2026-06-10 (CLI + `run_pipeline` API + multi-seed + `render_results_dir`) |
| `RESULTS.md` | ✅ přepsáno 2026-06-10 (kompletní layout = kontrakt vynucený e2e testem) |
| `EXAMPLES.md` | 🗑 nahrazeno `VISUALIZATIONS.md` (2026-06-11) — SOM mapy + tréninkové grafy + topo projekce s příklady z SCurve (`examples/maps|plots|topo`) |
