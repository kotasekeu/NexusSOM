# Fáze 2 — Databáze + API + LLM tool calling: specifikace požadavků

**Verze:** 1.1 · **Datum:** 2026-06-02  
**Autor:** NexusSom projekt  
**Stav:** Návrh

---

## 1. Kontext a motivace

### 1.1 Aktuální stav systému

Systém se skládá z několika oddělených vrstev bez společné datové vrstvy:

```
[SOM / EA skripty] → CSV + JSON + NPY + PNG soubory → [Streamlit čte přímo]
                                                     ↘ [LLM dostane dump 8K tokenů]
```

Streamlit UI v `app/ui/app.py` dnes:
- čte `run_metrics.json`, `clusters.json`, `sample_assignments.csv` přímo z disku
- zobrazuje PNG soubory z `visualizations/` a `maps_dataset/`
- předává celý `llm_context.json` (~8K tokenů) LLM při každém startu chatu
- nemá přehled nad více běhy nebo datasety najednou — vždy pracuje s jednou složkou

EA výsledky (`seed_*/individuals/`) jsou prakticky nedostupné z UI — UI pro EA neexistuje.

### 1.2 Cíl Fáze 2

**Komplexní upgrade celého systému** na třívrstvou architekturu:

```
[SOM / EA skripty] → SQLite DB  ←→  FastAPI  ←→  Streamlit UI (nový)
                                          ↑
                                    LLM tool calling (jeden z konzumentů API)
```

API je páteř systému. Streamlit UI i LLM jsou jeho konzumenti — ne privilegovaní uživatelé souborového systému.

### 1.3 Problémy aktuálního stavu

| Oblast | Problém | Dopad |
|--------|---------|-------|
| **Navigace** | UI pracuje vždy s jednou složkou, nelze porovnat běhy | Nelze provádět analýzu více běhů najednou |
| **EA výsledky** | Žádné UI pro EA — výsledky jsou jen PNG soubory | Pareto fronta, evoluce HV, porovnání jedinců nedostupné |
| **Vizualizace** | Streamlit zobrazuje statické PNG ze souboru, vždy celou sadu | Nelze filtrovat, zoom, interaktivita |
| **LLM kontext** | 8K tokenů bulk dump na začátku chatu | Při 50×50 mapě přeteče celé context window ještě před první otázkou |
| **Data duplikace** | Stejná čísla v CSV i JSON, `llm_context.json` je derivát ostatních | Žádný single source of truth, obtížná konzistence |
| **Škálovatelnost** | UI čte `.npy`, `.json`, `.csv` přímo, nefunguje přes síť | Nelze nasadit jinam než lokálně |

---

## 2. Typy běhů — analýza datové struktury

Systém má dva typy výstupů. Každý má jinou hierarchii souborů.

### 2.1 SOM běh (jednorázový)

```
results/{timestamp}/
├── run_metrics.json          ← map_size, mqe, te, duration
├── dataset_meta.json         ← n_samples, n_dims, n_categorical, ...
├── csv/
│   ├── sample_assignments.csv   ← sample_id, bmu_i, bmu_j, qe, qe_dim_*, is_outlier
│   ├── weights.npy
│   ├── original_input.csv
│   └── ignore_mask.csv
├── json/
│   ├── clusters.json            ← neuron_key → [sample_ids]
│   ├── quantization_errors.json ← total_qe, neuron_qe per neuron
│   ├── extremes.json            ← anomálie s důvody
│   ├── preprocessing_info.json  ← dim names, scaler params
│   └── llm_context.json         ← generovaný kontext pro LLM (nepřesouvá se do DB)
├── llm/
│   ├── report.md
│   └── prompt_log.json
└── visualizations/...
```

**Detekce:** absence `seed_*/` adresářů → SOM run.

### 2.2 EA běh (multi-seed, multi-generace)

```
results/{timestamp}/
├── dataset_meta.json
├── calibration_probe.csv        ← probe_idx, org_max (per probe)
├── seed_{N}/                    ← jeden nebo více seedů
│   ├── pareto_front.csv         ← generation, uid, raw_mqe_ratio, raw_te, dead_ratio, ...
│   ├── results.csv              ← všichni jedinci: uid, metrics, hyperparams, ds_meta
│   ├── pareto_metrics.csv       ← generation, front_size, hv, spacing, spread_*
│   ├── status.csv               ← uid, population_id, generation, status, start/end_time
│   └── individuals/{uid}/
│       ├── csv/weights.npy
│       ├── csv/weights_readable.csv
│       ├── csv/training_checkpoints.json
│       └── visualizations/...
├── pareto_plot/                 ← vizualizace per seed
└── json/ csv/ ...               ← sdílená data (preprocessing_info, atd.)
```

**Detekce:** přítomnost `seed_*/` adresářů → EA run.  
**Poznámka:** EA jedinci nemají `sample_assignments` ani `clusters` — ty vznikají až po výběru nejlepšího jedince a spuštění jako samostatný SOM běh.

---

## 3. Požadavky na nové UI — co musí API podporovat

Nový Streamlit UI bude mít 5 hlavních obrazovek. Každá závisí výhradně na API — žádné přímé čtení souborů.

### 3.1 Přehled datasetů a běhů (Dashboard)

```
┌─────────────────────────────────────────────────────┐
│  NexusSom Dashboard                                  │
│                                                       │
│  Dataset: [WineQuality ▾]     Typ: [SOM | EA | Vše]  │
│                                                       │
│  Běhy:                                                │
│  ┌──────────────┬──────┬──────┬───────┬────────────┐ │
│  │ ID           │ Mapa │ MQE  │ TE    │ Datum      │ │
│  ├──────────────┼──────┼──────┼───────┼────────────┤ │
│  │ 20260530_... │ 15×15│ 0.12 │ 0.023 │ 30.5.2026  │ │
│  │ 20260515_... │ EA   │ —    │ —     │ 15.5.2026  │ │
│  └──────────────┴──────┴──────┴───────┴────────────┘ │
└─────────────────────────────────────────────────────┘
```

API volání: `GET /datasets`, `GET /datasets/{name}/runs`

### 3.2 SOM výsledky (tab Výsledky)

Nahrazuje aktuální Streamlit tab "Výsledky" — ale data přicházejí z API, ne ze souborů.

**Metriky a přehled:**
- MQE, TE, dead neurons, trvání → `GET /runs/{id}/summary`
- Histogram QE vzorků → `GET /runs/{id}/samples/qe_distribution`

**Mapa vizualizace** — PNG soubory zůstávají na disku, API vrátí jejich seznam a cesty:
- `GET /runs/{id}/images` → `[{name: "u_matrix", path: "visualizations/u_matrix.png"}, ...]`
- Streamlit zobrazí `st.image()` přes relativní cestu nebo `/static/` mount

**Clustery:**
- Tabulka všech neuronů s hit count → `GET /runs/{id}/clusters`
- Klik na neuron → detail → `GET /runs/{id}/clusters/{key}`
- Dimenzionální statistiky per cluster

**Anomálie:**
- Tabulka outlierů s důvody → `GET /runs/{id}/anomalies`
- Detail vzorku → `GET /runs/{id}/samples/{id}`

### 3.3 EA výsledky (tab EA) — nová obrazovka, dnes neexistuje

```
┌─────────────────────────────────────────────────────┐
│  EA Analýza: LungCancerDataset / 20260515_111812     │
│                                                       │
│  Seedy: [101] [42] [1337] [2026] [7]                  │
│                                                       │
│  Pareto evoluce (HV):    Finální Pareto fronta:       │
│  [graf HV per generace]  [scatter MQE vs TE]          │
│                                                       │
│  Jedinci na Pareto frontě:                            │
│  ┌────────────┬──────┬──────┬───────────────────────┐ │
│  │ uid        │ MQE  │  TE  │ Hyperparametry        │ │
│  └────────────┴──────┴──────┴───────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

API volání:
- `GET /ea/{id}` → přehled EA runu
- `GET /ea/{id}/seeds/{sid}/pareto` → Pareto evoluce (data pro graf HV)
- `GET /ea/{id}/individuals/{uid}` → detail jedince
- `GET /ea/{id}/compare` → srovnání Pareto front přes seedy

### 3.4 Trénink SOM (tab Trénink)

Aktuální tab zůstává — spustí SOM trénink jako subprocess, zobrazí live log. Po dokončení zaregistruje nový run do DB.

API volání po dokončení tréninku:
- `POST /import/run` → zaregistruje nový run_dir do DB (async import)
- `GET /import/status/{job_id}` → stav importu (běží / hotovo / chyba)

### 3.5 Chat s LLM (tab Chat)

Místo bulk kontextu dostane LLM summary + tools.

API volání:
- `GET /runs/{id}/summary` → init kontext (~500 tokenů)
- Tools volaná LLM za chodu: `get_cluster_detail`, `get_top_anomalies`, `get_dimension_stats`, `search_clusters`, `get_sample_detail`

### 3.6 Porovnání běhů (tab Porovnat) — nová obrazovka

```
Vybrat běhy: [20260530_125716 ✓] [20260528_093012 ✓] [Přidat...]

MQE:  ████ 0.124  ░░░░ 0.098
TE:   ░░ 0.023    ███ 0.041
Dead: ██ 15%       ░ 3%
```

API volání: `GET /runs/compare?ids=id1,id2` → navrhnout jako nový endpoint P2.

---

## 4. Datový model (SQLite)

### 3.1 Schéma — přehled tabulek

```
datasets
  └── som_runs ─────── sample_assignments
        └── neuron_qe      └── (qe_dims jako JSON)
        └── clusters
        └── anomalies

ea_runs
  └── ea_seeds
        └── ea_individuals
        └── ea_pareto_metrics
        └── calibration_probes
```

### 3.2 DDL — tabulky

```sql
-- Datasety registrované v systému
CREATE TABLE datasets (
    id          INTEGER PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,     -- např. "WineQuality"
    path        TEXT NOT NULL,             -- abs. cesta k data/datasets/...
    description TEXT,                      -- obsah ABOUT.md pokud existuje
    n_samples   INTEGER,
    n_dims      INTEGER,
    n_categorical INTEGER,
    created_at  TEXT DEFAULT (datetime('now'))
);

-- Jednorázové SOM běhy (+ vybrané jedinci z EA po analýze)
CREATE TABLE som_runs (
    id           TEXT PRIMARY KEY,         -- timestamp, např. "20260530_125716"
    dataset_id   INTEGER REFERENCES datasets(id),
    ea_uid       TEXT,                     -- NULL pro standalone SOM; uid pro EA-selected
    map_m        INTEGER NOT NULL,
    map_n        INTEGER NOT NULL,
    mqe          REAL,
    topographic_error REAL,
    dead_neuron_ratio REAL,
    duration_s   REAL,
    run_path     TEXT NOT NULL,            -- abs. cesta k results/{id}/
    created_at   TEXT
);

-- Přiřazení vzorků k neuronům (per SOM run)
CREATE TABLE sample_assignments (
    id         INTEGER PRIMARY KEY,
    run_id     TEXT REFERENCES som_runs(id),
    sample_id  INTEGER NOT NULL,
    bmu_i      INTEGER NOT NULL,
    bmu_j      INTEGER NOT NULL,
    bmu_key    TEXT NOT NULL,              -- "i_j"
    qe         REAL,
    qe_dims    TEXT,                       -- JSON: {"x": 0.12, "y": 0.05, ...}
    is_outlier INTEGER DEFAULT 0
);
CREATE INDEX idx_sa_run_bmu ON sample_assignments(run_id, bmu_key);
CREATE INDEX idx_sa_run_sample ON sample_assignments(run_id, sample_id);

-- QE per neuron (agregace)
CREATE TABLE neuron_qe (
    run_id        TEXT REFERENCES som_runs(id),
    neuron_key    TEXT NOT NULL,
    qe_mean       REAL,
    qe_max        REAL,
    sample_count  INTEGER,
    PRIMARY KEY (run_id, neuron_key)
);

-- Clustery (neuron → seznam vzorků)
CREATE TABLE clusters (
    run_id       TEXT REFERENCES som_runs(id),
    neuron_key   TEXT NOT NULL,
    sample_ids   TEXT NOT NULL,            -- JSON array
    sample_count INTEGER,
    PRIMARY KEY (run_id, neuron_key)
);

-- Anomálie
CREATE TABLE anomalies (
    id         INTEGER PRIMARY KEY,
    run_id     TEXT REFERENCES som_runs(id),
    sample_id  INTEGER,
    reason     TEXT,                       -- JSON: [{"dim": "x", "type": "global_min", ...}]
    qe         REAL
);

-- EA runs (parent)
CREATE TABLE ea_runs (
    id         TEXT PRIMARY KEY,           -- timestamp
    dataset_id INTEGER REFERENCES datasets(id),
    run_path   TEXT NOT NULL,
    created_at TEXT
);

-- EA seeds
CREATE TABLE ea_seeds (
    id              INTEGER PRIMARY KEY,
    ea_run_id       TEXT REFERENCES ea_runs(id),
    seed_value      INTEGER,
    n_generations   INTEGER,
    final_hv        REAL,                  -- hypervolume finální fronty
    pareto_size     INTEGER
);

-- EA jedinci (všichni, ne jen Pareto)
CREATE TABLE ea_individuals (
    uid              TEXT NOT NULL,
    seed_id          INTEGER REFERENCES ea_seeds(id),
    generation       INTEGER,
    map_m            INTEGER,
    map_n            INTEGER,
    mqe              REAL,
    mqe_ratio        REAL,                 -- raw_mqe_ratio (normalizovaná)
    topographic_error REAL,
    dead_ratio       REAL,
    topo_corr        REAL,                 -- raw_topo_corr / Spearman ρ
    constraint_violation REAL,
    is_penalized     INTEGER DEFAULT 0,
    is_pareto_final  INTEGER DEFAULT 0,    -- na finální frontě
    hyperparams      TEXT,                 -- JSON: lr, radius, batch params, ...
    duration_s       REAL,
    PRIMARY KEY (uid, seed_id)
);
CREATE INDEX idx_ea_ind_seed ON ea_individuals(seed_id, generation);

-- Metriky Pareto fronty per generace
CREATE TABLE ea_pareto_metrics (
    seed_id     INTEGER REFERENCES ea_seeds(id),
    generation  INTEGER,
    front_size  INTEGER,
    hv          REAL,
    spacing     REAL,
    spread_mqe  REAL,
    spread_te   REAL,
    PRIMARY KEY (seed_id, generation)
);

-- Calibration probes
CREATE TABLE calibration_probes (
    ea_run_id  TEXT REFERENCES ea_runs(id),
    probe_idx  INTEGER,
    org_max    REAL,
    PRIMARY KEY (ea_run_id, probe_idx)
);
```

---

## 4. FastAPI — struktura a endpointy

### 4.1 Adresářová struktura

```
app/
  api/
    main.py           ← FastAPI app, CORS, static files mount, lifespan
    database.py       ← SQLAlchemy engine + get_db dependency
    models.py         ← ORM modely
    schemas.py        ← Pydantic response modely
    routers/
      datasets.py     ← /datasets
      runs.py         ← /runs, /runs/{id}/summary, /runs/compare
      clusters.py     ← /runs/{id}/clusters
      anomalies.py    ← /runs/{id}/anomalies
      samples.py      ← /runs/{id}/samples
      dimensions.py   ← /runs/{id}/dimensions
      images.py       ← /runs/{id}/images  (static soubory)
      ea.py           ← /ea
      import_api.py   ← /import (async trigger)
  tools/
    import_to_db.py   ← CLI skript pro batch import
```

**Static files:** `app/api/main.py` mountuje `data/` jako `/static/` — Streamlit pak načte mapu jako `<img src="http://localhost:8000/static/datasets/WineQuality/results/.../u_matrix.png">` bez kopírování souborů.

### 4.2 Datasety a běhy

| Method | Endpoint | Konzument | Priorita |
|--------|----------|-----------|----------|
| GET | `/datasets` | Dashboard — výběr datasetu | P1 |
| GET | `/datasets/{name}` | Dashboard — detail + seznam všech běhů | P1 |
| GET | `/datasets/{name}/runs` | Dashboard — tabulka SOM + EA běhů | P1 |
| GET | `/runs/{id}/summary` | Chat init kontext (LLM), Výsledky tab header | P1 |
| GET | `/runs/compare?ids=id1,id2,...` | Tab Porovnat | P2 |

### 4.3 SOM — výsledky a analýza

| Method | Endpoint | Konzument | Priorita |
|--------|----------|-----------|----------|
| GET | `/runs/{id}/clusters` | Tab Výsledky → tabulka neuronů | P1 |
| GET | `/runs/{id}/clusters/{neuron_key}` | Klik na neuron, LLM tool | P1 |
| GET | `/runs/{id}/anomalies` | Tab Výsledky → anomálie, LLM tool | P1 |
| GET | `/runs/{id}/dimensions` | Tab Výsledky → dim stats, LLM tool | P1 |
| GET | `/runs/{id}/dimensions/{name}` | LLM tool get_dimension_stats | P1 |
| GET | `/runs/{id}/neurons` | Hit mapa tabulka | P2 |
| GET | `/runs/{id}/samples/{id}` | Detail vzorku, LLM tool get_sample_detail | P2 |
| GET | `/runs/{id}/samples/qe_distribution` | Histogram QE v UI | P2 |

**Příklad response `/runs/{id}/summary`** (použití: LLM init kontext + UI header):
```json
{
  "run_id": "20260530_125716",
  "dataset": "SwissRoll",
  "map_size": [15, 15],
  "mqe": 0.1247,
  "topographic_error": 0.023,
  "n_samples": 2000,
  "n_dims": 3,
  "n_clusters_active": 142,
  "n_dead_neurons": 83,
  "n_anomalies": 37,
  "description": "Swiss Roll 3D manifold dataset..."
}
```

### 4.4 Vizualizace — statické soubory

| Method | Endpoint | Konzument | Priorita |
|--------|----------|-----------|----------|
| GET | `/runs/{id}/images` | Tab Výsledky → výběr mapy k zobrazení | P1 |
| GET | `/static/...` | Přímé URL pro `st.image()` v Streamlitu | P1 |

**Response `/runs/{id}/images`:**
```json
[
  {"name": "u_matrix",          "category": "map",      "path": "/static/.../u_matrix.png"},
  {"name": "hit_map",           "category": "map",      "path": "/static/.../hit_map.png"},
  {"name": "component_alcohol", "category": "dim",      "path": "/static/.../component_alcohol.png"},
  {"name": "dim_qe_dominant",   "category": "dim_qe",   "path": "/static/.../dim_qe_dominant.png"},
  {"name": "topology_2d_pca",   "category": "topology", "path": "/static/.../topology_2d_pca.png"}
]
```

UI si obrázky filtruje podle `category` — každá kategorie má vlastní sekci v tabs Výsledky.

### 4.5 EA endpointy

| Method | Endpoint | Konzument | Priorita |
|--------|----------|-----------|----------|
| GET | `/ea` | Dashboard — seznam EA běhů | P1 |
| GET | `/ea/{id}` | Tab EA → header (n_seeds, dataset, generace) | P1 |
| GET | `/ea/{id}/seeds/{sid}/pareto` | Tab EA → graf HV evoluce + scatter Pareto | P1 |
| GET | `/ea/{id}/images` | Tab EA → pareto_plot PNG soubory | P1 |
| GET | `/ea/{id}/individuals/{uid}` | Tab EA → detail jedince | P2 |
| GET | `/ea/{id}/compare` | Tab EA → srovnání seedů | P2 |

### 4.6 Import a správa

| Method | Endpoint | Konzument | Priorita |
|--------|----------|-----------|----------|
| GET | `/health` | Monitoring, testovací endpoint | P1 |
| POST | `/import/run` | Tab Trénink — po dokončení SOM/EA registruje nový run | P1 |
| GET | `/import/status/{job_id}` | Tab Trénink — progress importu | P1 |
| POST | `/import/scan` | Dev utility — bulk scan všech results/ | P2 |

---

## 5. Import skript

`app/tools/import_to_db.py` — jednorázový + přírůstkový import.

### Logika detekce typu běhu

```python
def detect_run_type(run_path: Path) -> str:
    """'ea' pokud existují seed_* adresáře, jinak 'som'"""
    seeds = list(run_path.glob("seed_*/"))
    return "ea" if seeds else "som"
```

### Import SOM běhu — pořadí

1. Přečíst `run_metrics.json` + `dataset_meta.json` → upsert `datasets` + insert `som_runs`
2. Přečíst `csv/sample_assignments.csv` → batch insert `sample_assignments`
3. Přečíst `json/clusters.json` → insert `clusters`
4. Přečíst `json/quantization_errors.json` → insert `neuron_qe`
5. Přečíst `json/extremes.json` → insert `anomalies`

### Import EA běhu — pořadí

1. Upsert `datasets` z `dataset_meta.json` → insert `ea_runs`
2. Pro každý `seed_*/`:
   - Insert `ea_seeds`
   - Přečíst `results.csv` → batch insert `ea_individuals`
   - Přečíst `pareto_metrics.csv` → insert `ea_pareto_metrics`
3. Přečíst `calibration_probe.csv` → insert `calibration_probes`
4. Označit finální Pareto jedince dle posledního `pareto_front.csv`

### Přírůstkový import

```python
# Přeskočit run_id které již existují v DB
existing = {r.id for r in db.query(SomRun.id)}
if run_id in existing:
    continue
```

---

## 6. LLM Tool calling (Fáze 2C)

### Náhrada bulk kontextu

| Aktuální (Fáze 1) | Fáze 2C |
|-------------------|---------|
| Celý `llm_context.json` (~8K tokenů) na startu | Kompaktní summary (~500 tokenů) + tools |
| LLM se ptá z paměti | LLM volá endpoint, dostane přesná data |

### Tool definice

```python
TOOLS = [
    {
        "name": "get_cluster_detail",
        "description": "Detail neuronu/clusteru: počet vzorků, dim stats, QE.",
        "parameters": {"neuron_key": "string (např. '3_4')"}
    },
    {
        "name": "get_top_anomalies",
        "description": "Top N nejanomaálnějších vzorků s důvody.",
        "parameters": {"n": "int (default 5)"}
    },
    {
        "name": "get_dimension_stats",
        "description": "Statistiky dimenze: min/max/mean/std, hodnoty per cluster.",
        "parameters": {"dimension_name": "string"}
    },
    {
        "name": "search_clusters",
        "description": "Clustery splňující podmínku (dominantní kategorie, min vzorků).",
        "parameters": {"category": "string (optional)", "min_samples": "int (optional)"}
    },
    {
        "name": "get_sample_detail",
        "description": "Detail konkrétního vzorku: hodnoty dimenzí, BMU, QE, anomálie.",
        "parameters": {"sample_id": "int"}
    },
]
```

### Agent loop (schéma)

```
user: "Popiš clustery kde dominuje vysoká kyselost"
  → LLM: volá search_clusters(category=None) → GET /runs/{id}/clusters
  → API vrátí: [{neuron_key, sample_count, top_dim_deviations}, ...]
  → LLM: volá get_dimension_stats("fixed_acidity")
  → API vrátí: {mean: 8.3, std: 1.7, per_cluster: {...}}
  → LLM: odpoví s konkrétními hodnotami
```

### Modely s nativní podporou tool calling v Ollama

- `qwen2.5:14b` ✓ (doporučený)
- `llama3.1:8b` ✓
- `phi4:14b` ✓
- `gemma2:27b` — omezená podpora

---

## 7. Pracovní postup — Scrum breakdown

Celková závislost sprintů:
```
Sprint 1 (DB)  →  Sprint 2 (API)  →  Sprint 3 (UI)  →  Sprint 4 (LLM)
```
Sprint 3 (UI) závisí na Sprint 2 (API musí existovat). Sprint 4 (LLM tool calling) závisí na obou.

---

### Sprint 1 — Datová vrstva · odhad: 2–3 dny

| # | User Story | Akceptační kritéria | Složitost |
|---|-----------|---------------------|-----------|
| 1.1 | Jako vývojář chci SQLAlchemy modely pro všechny tabulky | Modely, DB init bez chyb, `pytest tests/unit/test_db_models.py` zelený | M |
| 1.2 | Jako uživatel chci importovat SOM běh do DB | `import_to_db.py --path .../SwissRoll/results/20260530_125716` naplní tabulky, idempotentní | M |
| 1.3 | Jako uživatel chci importovat EA běh do DB | Import LungCancer EA, správná hierarchie, `is_pareto_final` označeno | L |
| 1.4 | Jako vývojář chci unit testy importu | `pytest tests/unit/test_importer.py` zelený (incl. edge cases chybějících souborů) | M |
| 1.5 | Jako uživatel chci bulk import všech výsledků | `import_to_db.py --scan data/datasets/` projde vše, přeskočí existující run_id | M |

**DoD Sprint 1:** `nexusom.db` obsahuje SwissRoll SOM + LungCancer EA, `pytest tests/unit/` zelený.

---

### Sprint 2 — API vrstva · odhad: 2–3 dny

| # | User Story | Akceptační kritéria | Složitost |
|---|-----------|---------------------|-----------|
| 2.1 | Jako vývojář chci FastAPI app se static files | `/health` OK, `/static/` servuje PNG soubory ze `data/`, CORS nakonfigurován | S |
| 2.2 | Jako UI vývojář chci datasety a běhy | `GET /datasets`, `/datasets/{name}`, `/datasets/{name}/runs` fungují | S |
| 2.3 | Jako UI vývojář chci images endpoint | `GET /runs/{id}/images` vrátí kategorizovaný seznam PNG s `/static/` URL | M |
| 2.4 | Jako UI+LLM vývojář chci SOM analýzové endpointy | clusters, anomálies, dimensions (P1) fungují, filtry OK | M |
| 2.5 | Jako LLM vývojář chci summary endpoint | `GET /runs/{id}/summary` ≤ 800 tokenů, pokrývá vše co LLM potřebuje na init | M |
| 2.6 | Jako výzkumník chci EA endpointy | EA overview + Pareto evoluce + images pro pareto_plot/ | M |
| 2.7 | Jako vývojář chci import API | `POST /import/run` + `GET /import/status/{id}` fungují async | M |
| 2.8 | Jako vývojář chci OpenAPI dokumentaci + integration testy | `/docs` přístupná, `pytest tests/integration/` zelený | M |

**DoD Sprint 2:** Všechny P1 endpointy fungují, `pytest tests/integration/` zelený, Streamlit se může napojit.

---

### Sprint 3 — Nový Streamlit UI · odhad: 3–4 dny

Streamlit UI se přepíše tak, aby veškerá data šla přes API. Žádné přímé čtení souborů.

| # | User Story | Akceptační kritéria | Složitost |
|---|-----------|---------------------|-----------|
| 3.1 | Jako uživatel chci Dashboard s přehledem datasetů a běhů | Dropdown dataset, tabulka SOM/EA běhů, klik → otevře detail | M |
| 3.2 | Jako uživatel chci tab Výsledky napojený na API | Metriky, mapy (přes `/static/`), clustery, anomálie — vše z API, ne ze souborů | L |
| 3.3 | Jako výzkumník chci tab EA | HV evoluce (line chart), Pareto scatter, tabulka jedinců, Pareto plot PNG | L |
| 3.4 | Jako uživatel chci tab Porovnat | Výběr 2+ běhů ze stejného datasetu, side-by-side metriky | M |
| 3.5 | Jako uživatel chci tab Trénink s auto-importem | Po dokončení tréninku `POST /import/run`, polling stavu, refresh seznamu běhů | M |
| 3.6 | Jako vývojář chci odstranit přímé čtení souborů z UI | Grep po `open(`, `.read()`, `pd.read_csv`, `json.load` v `app/ui/` → nula výsledků | L |

**DoD Sprint 3:** Uživatel může procházet všechny datasety a běhy bez znalosti adresářové struktury, UI funguje i kdyby `data/` bylo na jiném disku než UI.

---

### Sprint 4 — LLM tool calling · odhad: 3–4 dny

| # | User Story | Akceptační kritéria | Složitost |
|---|-----------|---------------------|-----------|
| 4.1 | Jako uživatel chci LLM s tool calling místo bulk kontextu | Chat init = `/summary` (~500 tok), LLM volá tools za chodu, odpovědi citují data | XL |
| 4.2 | Jako uživatel chci vidět jaké tools LLM zavolal | UI zobrazí tool calls jako collapsible `st.expander` pod odpovědí | M |
| 4.3 | Jako uživatel chci conversation summarization | Při > 80 % context window se automaticky zhustí starší zprávy | M |
| 4.4 | Jako výzkumník chci grounding check | Score 0–1 pod každou odpovědí, varování při < 0.3 | M |
| 4.5 | Jako vývojář chci LLM testy | `pytest tests/llm/` zelený (bez requires_ollama marku) | M |

**DoD Sprint 4:** Uživatel chatuje o WineQuality 30×30, context window < 60 % po 10 výměnách, odpovědi citují konkrétní neuron klíče a hodnoty.

---

## 8. TDD — strategie testování

Přístup: **test-first**. Každá story začíná napsáním testu, který selže. Implementace probíhá dokud test neprojde. Pokrytí edge cases je důležitější než pokrytí happy path — happy path je triviální, edge cases odhalují skutečné chyby.

### 8.1 Adresářová struktura testů

```
tests/
  conftest.py               ← globální fixtures: in-memory DB, test client, miniaturní data
  fixtures/
    som_minimal/            ← minimální SOM run (3×3 mapa, 10 vzorků)
    ea_minimal/             ← minimální EA run (1 seed, 3 generace, 5 jedinců)
    edge_cases/
      empty_clusters.json   ← SOM run bez žádného aktivního neuronu
      all_dead_weights.npy  ← všechny neurony mrtvé
      missing_dims.csv      ← sample_assignments s NaN QE hodnotami
  unit/
    test_db_models.py       ← validace ORM modelů
    test_importer.py        ← logika importu (bez FS, mockované soubory)
    test_schemas.py         ← Pydantic serializace/validace
  integration/
    test_api_som.py         ← SOM endpointy end-to-end (TestClient + in-memory DB)
    test_api_ea.py          ← EA endpointy end-to-end
  edge_cases/
    test_import_extremes.py ← extrémní vstupy při importu
    test_api_extremes.py    ← extrémní vstupy přes API
  llm/
    test_tools.py           ← tool calling logika
    test_grounding.py       ← grounding check
    test_summarization.py   ← conversation summarization
```

### 8.2 Hlavní fixtures (`conftest.py`)

```python
@pytest.fixture
def db():
    """In-memory SQLite DB, každý test dostane čistou instanci."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session

@pytest.fixture
def client(db):
    """FastAPI TestClient s injektovanou in-memory DB."""
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)

@pytest.fixture
def minimal_som_run(db):
    """3×3 mapa, 10 vzorků, 2 dimenze, 2 mrtvé neurony."""
    ...  # vloží data do db, vrátí run_id

@pytest.fixture
def minimal_ea_run(db):
    """1 seed, 3 generace, 5 jedinců, finální Pareto fronta o 2 členech."""
    ...
```

---

### 8.3 Unit testy — DB modely (`test_db_models.py`)

| Test | Co testuje | Očekávaný výsledek |
|------|-----------|-------------------|
| `test_som_run_requires_dataset` | Insert SOM run bez dataset_id | IntegrityError (FK) |
| `test_neuron_key_format` | `bmu_key` ve formátu "0_0", "14_14", "0_14" | Uloženo bez chyby |
| `test_qe_dims_json_roundtrip` | `qe_dims = {"x": 0.12, "y": None}` | Načteno zpět identické |
| `test_sample_ids_json_array` | `clusters.sample_ids` jako JSON list | `json.loads()` vrátí list int |
| `test_ea_individual_pk_composite` | Stejné `uid` pro dva různé seedy | Bez chyby (composite PK) |
| `test_ea_individual_pk_duplicate` | Stejné `uid` i `seed_id` dvakrát | IntegrityError |

---

### 8.4 Unit testy — Import logika (`test_importer.py`)

| Test | Co testuje | Očekávaný výsledek |
|------|-----------|-------------------|
| `test_detect_som_run` | Adresář bez `seed_*/` | `detect_run_type()` vrátí `"som"` |
| `test_detect_ea_run` | Adresář s `seed_101/` | `detect_run_type()` vrátí `"ea"` |
| `test_detect_empty_dir` | Prázdný adresář | Vrátí `None` nebo vyhodí `ValueError` |
| `test_import_som_idempotent` | Import stejného run_id dvakrát | Druhý import přeskočen, počet řádků v DB stejný |
| `test_import_ea_idempotent` | Import stejného ea_run_id dvakrát | Identické chování |
| `test_som_row_counts` | Import miniaturního SOM | Počet řádků `sample_assignments` = počet řádků v CSV |
| `test_ea_individuals_count` | Import EA run | Počet `ea_individuals` = počet řádků v `results.csv` |
| `test_pareto_final_flag` | Označení finálních Pareto jedinců | `is_pareto_final=1` jen pro uid v posledním `pareto_front.csv` |

---

### 8.5 Edge cases — import (`test_import_extremes.py`)

Tyto testy ověřují, že import nepadá a chová se deterministicky při neúplných nebo extrémních datech.

| Test | Scénář | Očekávaný výsledek |
|------|--------|-------------------|
| `test_missing_clusters_json` | `json/clusters.json` neexistuje | Import proběhne, tabulka `clusters` prázdná pro tento run |
| `test_missing_sample_assignments` | `csv/sample_assignments.csv` chybí | Import proběhne, `sample_assignments` prázdná, log varování |
| `test_nan_qe_values` | QE sloupce obsahují `NaN` | Uloženo jako `NULL`, ne jako chyba |
| `test_all_dead_neurons` | Všechny neurony mají 0 vzorků | `neuron_qe` má `sample_count=0`, `clusters` prázdná |
| `test_empty_clusters` | `clusters.json` = `{}` | Tabulka prázdná, žádná chyba |
| `test_zero_anomalies` | `extremes.json` = `{}` | Tabulka `anomalies` prázdná |
| `test_single_neuron_map` | 1×1 mapa (všechny vzorky do jednoho neuronu) | Funguje, `bmu_key="0_0"` pro všechny záznamy |
| `test_large_map_import` | 30×30 mapa, 5000 vzorků (fixture ze SwissRoll) | Import do 10 sekund, počty sedí |
| `test_ea_single_seed` | EA run s jediným seedem | Funguje, `ea_seeds` má 1 záznam |
| `test_ea_empty_pareto` | Finální Pareto fronta = 0 členů | `is_pareto_final=0` pro všechny, žádná chyba |
| `test_ea_missing_pareto_metrics` | `pareto_metrics.csv` chybí | `ea_pareto_metrics` prázdná, ostatní data importována |
| `test_dataset_name_special_chars` | Dataset název "Swiss-Roll_v2" | Uloženo bez chyby, GET `/datasets/Swiss-Roll_v2` funguje |

---

### 8.6 Integration testy — SOM API (`test_api_som.py`)

```python
# Příklad struktury testu — happy path
def test_get_clusters_returns_all_active(client, minimal_som_run):
    r = client.get(f"/runs/{minimal_som_run}/clusters")
    assert r.status_code == 200
    data = r.json()
    assert len(data) > 0
    assert all("neuron_key" in c and "sample_count" in c for c in data)

# Edge case
def test_get_cluster_nonexistent_key(client, minimal_som_run):
    r = client.get(f"/runs/{minimal_som_run}/clusters/99_99")
    assert r.status_code == 404
```

| Test | Scénář | Očekávaný výsledek |
|------|--------|-------------------|
| `test_health_ok` | `GET /health` | `200`, `status: "ok"` |
| `test_datasets_list_nonempty` | `GET /datasets` po importu | List ≥ 1, správné `name` a `n_samples` |
| `test_dataset_not_found` | `GET /datasets/neexistuje` | `404` |
| `test_run_summary_token_budget` | `GET /runs/{id}/summary` | Response JSON má ≤ 800 tokenů (měřeno `len(json_str)//4`) |
| `test_clusters_sorted_by_sample_count` | `GET /runs/{id}/clusters?sort_by=sample_count` | Sestupně seřazeno |
| `test_clusters_top_n` | `GET /runs/{id}/clusters?top=5` | Vráceno přesně 5 výsledků |
| `test_cluster_detail_correct_stats` | `GET /runs/{id}/clusters/0_0` | `sample_count` odpovídá počtu záznamů v `sample_assignments` |
| `test_cluster_nonexistent` | Neexistující `neuron_key` | `404` |
| `test_anomalies_limit` | `GET /runs/{id}/anomalies?limit=3` | Vráceno ≤ 3 záznamy |
| `test_anomalies_empty_run` | Run bez anomálií | `200`, prázdný list |
| `test_dimensions_all_present` | `GET /runs/{id}/dimensions` | Všechny dimenze z `preprocessing_info.json` přítomny |
| `test_run_nonexistent` | `GET /runs/0000_000000/clusters` | `404` |
| `test_invalid_sort_by` | `?sort_by=neexistujici_sloupec` | `422 Unprocessable Entity` |
| `test_negative_limit` | `?limit=-1` | `422` |

---

### 8.7 Integration testy — EA API (`test_api_ea.py`)

| Test | Scénář | Očekávaný výsledek |
|------|--------|-------------------|
| `test_ea_list` | `GET /ea` | List EA runů, správné dataset jméno |
| `test_ea_overview` | `GET /ea/{id}` | Správný `n_seeds`, `n_generations` |
| `test_pareto_evolution_monotonic_hv` | `GET /ea/{id}/seeds/{sid}/pareto` | HV neklesá mezi generacemi (NSGA-II garantuje) |
| `test_pareto_evolution_generation_count` | Počet generací v response | Odpovídá počtu řádků v `pareto_metrics.csv` |
| `test_ea_nonexistent` | `GET /ea/neexistuje` | `404` |
| `test_individual_detail` | `GET /ea/{id}/individuals/{uid}` | Hyperparams jako dict, metriky přítomny |
| `test_individual_nonexistent_uid` | Neexistující uid | `404` |

---

### 8.8 Edge cases — API (`test_api_extremes.py`)

| Test | Scénář | Očekávaný výsledek |
|------|--------|-------------------|
| `test_all_dead_neurons_summary` | Run kde všechny neurony mrtvé | `n_clusters_active=0`, `n_dead_neurons=map_m*map_n` |
| `test_single_neuron_map_clusters` | 1×1 mapa | Jeden cluster s klíčem `"0_0"`, `sample_count=n_samples` |
| `test_run_without_analysis` | SOM run bez `clusters.json` | `GET /clusters` vrátí `200` s prázdným listem, ne `500` |
| `test_summary_no_description` | Dataset bez `ABOUT.md` | `description: null`, žádná chyba |
| `test_large_cluster_response` | Neuron s 1000+ vzorky | Response do 2 sekund, `sample_ids` truncated nebo paginated |
| `test_concurrent_requests` | 10 paralelních `GET /clusters` | Všechny vrátí `200`, žádný deadlock |

---

### 8.9 LLM testy (`tests/llm/`)

**`test_tools.py`**

| Test | Scénář | Očekávaný výsledek |
|------|--------|-------------------|
| `test_tool_get_cluster_detail_valid` | Volání s existujícím `neuron_key` | Dict s `sample_count`, `dim_stats` |
| `test_tool_get_cluster_detail_invalid` | Neexistující `neuron_key` | Vrátí error dict `{"error": "not found"}`, nevyhodí exception |
| `test_tool_get_top_anomalies_default` | `n` nepředáno | Vrátí 5 výsledků (default) |
| `test_tool_unknown_name` | LLM zavolá neexistující tool | `{"error": "unknown tool"}` |
| `test_agent_loop_terminates` | LLM stále volá tools bez odpovědi | Max 5 iterací, pak přímá odpověď |
| `test_agent_loop_no_tools` | LLM odpoví bez tool call | Vrátí odpověď přímo, 1 iterace |

**`test_grounding.py`**

| Test | Scénář | Očekávaný výsledek |
|------|--------|-------------------|
| `test_grounded_neuron_key` | Odpověď obsahuje "neuron 3_4" | `grounded=True`, `score ≥ 0.33` |
| `test_grounded_numeric_value` | Odpověď obsahuje "14.12" = mean z kontextu | `grounded=True` |
| `test_not_grounded_generic` | "Tento dataset obsahuje různé vzorky." | `grounded=False`, `score=0` |
| `test_grounding_case_insensitive` | Klíč "3_4" vs "Neuron 3_4" | Detekováno správně |
| `test_grounding_empty_response` | LLM vrátí `""` | `grounded=False`, bez exception |

**`test_summarization.py`**

| Test | Scénář | Očekávaný výsledek |
|------|--------|-------------------|
| `test_summarize_short_history` | History kratší než práh | Vrácena beze změny |
| `test_summarize_preserves_system` | První 3 zprávy = system + kontext | Vždy zachovány |
| `test_summarize_preserves_recent` | Posledních 6 zpráv | Vždy zachovány |
| `test_summarize_reduces_length` | History 30 zpráv → summarized | Výsledek kratší než vstup |
| `test_summarize_output_is_valid_history` | Výstupní history | Všechny záznamy mají `role` + `content` |

---

### 8.10 TDD workflow per story

Každá story ze Scrum backlogu se implementuje takto:

```
1. Napsat testy z příslušné sekce výše (červené)
2. Spustit: pytest tests/ -x -v → všechny selžou
3. Implementovat minimální kód který testy uspokojí
4. Spustit: pytest tests/ -x -v → zelené
5. Refactor (pokud potřeba) → testy stále zelené
6. Commitovat: "feat: Story X.Y — <popis>" + testovací soubor
```

Pořadí priorit při psaní testů: **edge cases první**, happy path je bonus. Pokud test selže nečekaně, je to informace — upravit implementaci, ne test.

### 8.11 CI konfigurace (budoucí)

```yaml
# .github/workflows/test.yml (orientační)
- run: pytest tests/unit tests/integration tests/edge_cases -v --tb=short
- run: pytest tests/llm -v --tb=short -m "not requires_ollama"
```

Testy v `tests/llm/` označit `@pytest.mark.requires_ollama` — lokální, nepouštět v CI bez modelu.

---

## 9. Technické závislosti

```
fastapi>=0.115        ← async, Pydantic v2
sqlalchemy>=2.0       ← moderní async session
uvicorn               ← ASGI server
alembic               ← DB migrace
pydantic>=2.0         ← response modely
```

Přidání do existujícího `requirements.txt`.

### Konfigurace

```python
# app/api/config.py
DB_PATH    = Path("nexusom.db")          # nebo env DB_PATH
DATA_ROOT  = Path("data/datasets")       # nebo env DATA_ROOT
API_HOST   = "127.0.0.1"
API_PORT   = 8000
```

---

## 9. Migrace a zpětná kompatibilita

- **JSON/CSV soubory zůstávají** jako archiv — nic se nesmaže
- `llm_context.json` se do DB **nepřesouvá** — generuje se on-demand z DB
- Streamlit UI funguje v režimu souborů (Fáze 1) dokud Fáze 2C není dokončena
- Import je idempotentní — lze spustit opakovaně bez duplikátů

---

## 10. Co NENÍ součástí Fáze 2

- Multi-uživatelský přístup (→ PostgreSQL, autentizace) — mimo scope
- Persistentní ukládání LLM zpráv na serveru — zůstává v `chat_history.json`
- Realtime streaming odpovědí přes WebSocket — Ollama streaming zůstává přímý
- Frontend rewrite (React/Vue) — Streamlit zůstává
