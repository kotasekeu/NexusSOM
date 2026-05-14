# LLM Module — The Voice — Design

**Verze**: 2.0
**Aktualizováno**: 2026-05-13

---

## 1. Princip

The Voice je **translation layer** — neprovádí analýzu surových dat. Přebírá strukturované
výstupy z `app/analysis/` a převádí je do přirozeného jazyka srozumitelného doménovému expertovi.

Klíčové omezení: LLM nikdy nevidí raw CSV. Dostane `llm_context.json` (~450 KB, ~7 400 tokenů)
předpřipravený statistickým pipelinem. Každý claim musí být podložen konkrétním neuronem,
sample_id nebo číselnou hodnotou.

---

## 2. Architektura

### 2.1 Two-Context Model

**Context A — Dataset Description** (user-provided, statický per dataset)

```
data/datasets/<Dataset>/dataset_context.txt
```

Obsahuje:
- Popis datasetu a domény
- Popis každého sloupce (název, jednotky, rozsah hodnot)
- Dekódování kategorií (`1 = symptom absent, 2 = symptom present`)
- Doménové znalosti (`SMOKING a YELLOW_FINGERS jsou korelovány s vyšším rizikem`)

Fallback: pokud `dataset_context.txt` neexistuje, `context_builder.py` hledá `ABOUT.MD` nebo
`ABOUT.md` až 4 úrovně adresáře nahoru (typicky Kaggle datasety).

**Context B — SOM Analysis Summary** (auto-generated z `llm_context.json`)

Assemblováno `context_builder.py::_format_context()`:
- Map overview: size, topology, MQE, topographic error, dead ratio, Gini
- Per-cluster: dominant category, purity, dimension mean/std/Z-score
- Anomaly records: top-20 vzorků s plnými hodnotami řádku + delta anotace
- Global dimension statistics: min, max, mean, std, median, p25–p95
- Global category distributions

### 2.2 Prompt Architektura

```
System Prompt (constraints)
    ↓
Context A (dataset_context.txt)
    ↓
Context B (_format_context(llm_context.json))
    ↓
User Query / Report Instruction
```

System prompt zajišťuje:
1. LLM odpovídá pouze k poskytnutému datasetu a SOM analýze
2. Každé tvrzení musí odkazovat na konkrétní data z Context B
3. Při nejistotě LLM explicitně uvede "This information is not available in the current analysis"
4. Používá terminologii z Context A (doménový jazyk, ne SOM žargon)

### 2.3 Processing Pipeline

```
Step 1: run_som.py → SOM training → results/<timestamp>/
Step 2: app/analysis/ → compute llm_context.json  (automaticky po tréninku)
Step 3: run_llm.py → context_builder.py → načte llm_context.json + dataset_context.txt
Step 4: _format_context() → sestaví strukturovaný text prompt (~Context B)
Step 5: compose full prompt = system_prompt + Context A + Context B + user_query
Step 6: llm_client.py → Ollama HTTP API → streaming response
Step 7: uložit report.md + prompt_log.json (+ volitelně report.pdf)
```

---

## 3. Komponenty

### 3.1 `app/analysis/` — Statistical Pre-processor

Samostatný modul, nezávislý na `app/som/`. Čte pouze soubory z results adresáře.

```
app/analysis/src/
├── loader.py     → IO — načítá clusters.json, quantization_errors.json,
│                   extremes.json, pie_data_*.json, original_input.csv, weights.npy
├── stats.py      → globální statistiky, cluster statistiky, topologie mapy
├── anomalies.py  → detekce outlierů (numeric, multi-dim, 1-of-N)
└── context.py    → assembly llm_context.json + anomaly_records s delta anotacemi
```

Výstup: `results/<timestamp>/json/llm_context.json`

Podrobná dokumentace: `docs/llm/RESULT_ANALYZER.md`

### 3.2 `app/llm/src/context_builder.py`

`build_context(dataset_path)`:
1. Hledá `llm_context.json` → pokud existuje, načte přímo
2. Fallback: volá `analysis.src.context.build_llm_context()` (live výpočet)
3. Hledá `dataset_context.txt` (nebo `ABOUT.MD`) jako Context A
4. `_format_context()` → sestaví pipe-separated text prompt

Anomaly records formát v Context B:
```
AGE=30.0 [-25.17 / -45.6%] [GLOBAL_MIN] | SMOKING=2 [cluster_dom=1] | CHEST_PAIN=2 [+0.42 / +26.3%]
```

### 3.3 `app/llm/src/llm_client.py`

Ollama HTTP adapter. Volá `POST /api/chat` s `stream=True`.

Konfigurace:
- `base_url`: výchozí `http://localhost:11434`
- `model`: výchozí `llama3.1:8b`
- `num_ctx`: výchozí `16384` (pokrývá ~7 400 tokenů kontextu + odpověď)
- `temperature`: `0` pro deterministické výstupy

### 3.4 `app/llm/src/report_generator.py`

Orchestruje výstupní mód. Po dokončení reportu automaticky spustí `pdf_builder.py`.

Uložené soubory:
```
results/<timestamp>/llm/
├── report.md          ← výsledná zpráva
├── prompt_log.json    ← použitý prompt + model + timestamp (traceability)
└── report.pdf         ← volitelně, pokud spuštěno s -m pdf nebo po report módu
```

### 3.5 `app/llm/src/pdf_builder.py`

Generuje PDF z `report.md` + všech SOM vizualizací pomocí `fpdf2`.

Struktura PDF (25 stran):
1. Cover page
2. Report text (z report.md, sekce jako nadpisy)
3. Dimension statistics table
4. Cluster summary table
5. Anomaly list s barevnými bloky (červená = global_min/max, oranžová = high_deviation)
6. Key maps: U-Matrix, distance map, hit map, dead neurons
7. Component planes (grid, auto page break)
8. Category pie maps
9. Training plots (MQE + topographic error progression)

Optimalizace: obrázky downscalované přes PIL na max 700 px před vložením (3 240 px → 7.6 MB PDF).

---

## 4. LLM Provider

### Aktuální implementace

Ollama HTTP API. Jediný adaptér — `llm_client.py`.

| Model | VRAM | Inference | Kvalita |
|---|---|---|---|
| `llama3.1:8b` | 5 GB | lokálně / CPU | ⭐⭐⭐ |
| `nexusom-analyst` | 5 GB | lokálně | ⭐⭐⭐ + baked system prompt |
| `qwen2.5:32b` | 19 GB | vzdálený GPU | ⭐⭐⭐⭐⭐ |
| `llama3.1:70b` | 41 GB | vzdálený GPU | ⭐⭐⭐⭐⭐ |

Vzdálený GPU: `--url http://<ip>:11434` (Ollama spuštěno s `OLLAMA_HOST=0.0.0.0`).

### `nexusom-analyst` custom model

```
app/llm/Modelfile
```

- Base: `llama3.1:8b`
- Baked-in system prompt (eliminuje overhead per-request)
- `temperature 0`, `num_ctx 16384`

Rebuild po změně Modelfile:
```bash
ollama create nexusom-analyst -f app/llm/Modelfile
```

### Budoucí providery (neimlementováno)

Anthropic API a OpenAI API nejsou zatím přidány. Adapter pattern je záměr do budoucna —
aktuálně `llm_client.py` je Ollama-only.

---

## 5. Output Modes

### Report Mode
```bash
python3 app/run_llm.py -i results/<timestamp> -m report [--model llama3.1:8b] [--url ...]
```
One-shot: celá analýza streamována do terminálu → uložena jako `report.md` + `report.pdf`.

### Chat Mode
```bash
python3 app/run_llm.py -i results/<timestamp> -m chat
```
Interaktivní Q&A. Celý kontext načten jednorázově, follow-up dotazy ve stejné session.
Ukončení: `exit` / `quit` / `q`.

### PDF Mode
```bash
python3 app/run_llm.py -i results/<timestamp> -m pdf
```
Přeskočí LLM inferenci, jen sestaví PDF z existujícího `report.md` + vizualizací.

---

## 6. Rozhodnutí

### Prompt Engineering vs. Fine-tuning

Zvoleno: **prompt engineering only**.

- Kontext B je dostatečně strukturovaný — 7 400 tokenů pokryje celou mapu 18×18
- Fine-tuning by vyžadoval stovky příkladových reportů
- Výsledky jsou přenositelné na jakýkoliv Ollama model nebo cloud API

### Context Window Strategy

Pro mapy ≤ 20×20 (≤ 400 neuronů): full per-neuron detail bez omezení.

Pro mapy > 30×30 (> 900 neuronů): context by překročil 16k tokenů. Nutné:
- Agregace neuronů do regionů (D5 — zatím neimplementováno)
- Top-N neuronů místo všech
- Chunked processing (neimplementováno)

Aktuálně testováno na 18×18 (324 neuronů) = ~6 650 tokenů — v limitu.

### Determinismus

`temperature=0` v `Modelfile` a `llm_client.py`. Stejný vstup → konzistentní struktura reportu.
Plná reprodukovatelnost závisí na modelu — lokální modely jsou deterministické při `temperature=0`.

---

## 7. Kontext — Token Budget (18×18, 3 000 vzorků)

| Sekce | Tokeny (est.) |
|---|---|
| Map overview | ~50 |
| Clusters (324 neuronů) | ~4 200 |
| Anomaly records (top-20) | ~2 100 |
| Global dimension stats | ~200 |
| Category distributions | ~100 |
| Dataset context (Context A) | ~300 |
| **Celkem** | **~6 950** |

`num_ctx 16384` → ~9 400 tokenů volného prostoru pro odpověď.

Optimalizace provedené pro zmenšení `llm_context.json`:
- Odstraněno `category_counts` per neuron (→ pouze `purity` + `dominant_category`)
- Odstraněno `cluster_local_outliers` (→ pouze `top_anomalies` summary)
- Výsledek: 1 181 KB → 454 KB

---

## 8. Chybějící / Budoucí

| Funkce | Priorita | Poznámka |
|---|---|---|
| Anthropic / OpenAI API adapter | Střední | Adapter pattern připraven v designu |
| Skewness + kurtosis (A3) | Vysoká | Lepší popis distribucí pro LLM |
| Feature correlations (A4) | Vysoká | Kauzální vztahy v reportu |
| Spatial map regions (D5) | Vysoká | "Levá polovina = NO cancer" |
| Boundary samples (C8) | Střední | Vzorky na hranici dvou clusterů |
| Cluster health score (E1–E3) | Střední | Kompaktnost × separace |
| Chunked processing pro velké mapy | Nízká | Pro mapy > 30×30 |
| JSON structured output | Nízká | Strojově čitelné výsledky vedle MD reportu |
