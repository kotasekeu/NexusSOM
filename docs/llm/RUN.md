# LLM Analýza — Spuštění

Modul LLM ("The Voice") vezme výsledky SOM běhu a vygeneruje přirozenou analýzu nebo
umožní interaktivní dialog o datech. LLM nevidí surová data — dostane strukturovaný
kontext sestavený `result_analyzer.py` z výstupů SOM.

---

## Prerekvizity

### 1. Ollama server

```bash
ollama serve
```

### 2. Dostupný model

```bash
ollama list
```

Doporučené modely (seřazeny dle preference):

| Model | Velikost | Poznámka |
|---|---|---|
| `nexusom-analyst:latest` | 4.9 GB | Vlastní model s baked-in system promptem (llama3.1:8b základ) |
| `llama3.1:8b` | 4.9 GB | Výchozí — ověřeno funkční |
| `mistral:7b-instruct` | 4.4 GB | Alternativa |

Pokud model chybí:
```bash
ollama pull llama3.1:8b
```

### 3. Hotový SOM běh

LLM čte z výstupního adresáře SOM (`json/llm_context.json`). Ten se generuje
automaticky po každém `run_som.py` nebo EA běhu.

Pro regeneraci `llm_context.json` (např. po aktualizaci analytického modulu):
```bash
.venv/bin/python3 -c "
from app.analysis.src.context import save_llm_context
save_llm_context('data/datasets/<Dataset>/results/<timestamp>')
"
```

`llm_context.json` obsahuje (od verze 2.2):
- MQE, topographic error, dead neurons, Gini koeficient hustoty
- Silhouette score (globální + per cluster)
- Trustworthiness & Continuity pro k=5, 10, 20
- Per-cluster: dominantní kategorie, purity, feature importance (top 5 dle |Z-score|)
- Anomálie s plnými hodnotami dimenzí a delta anotacemi

---

## Report mode — jednorázová analýza

```bash
python3 app/run_llm.py \
  -i data/datasets/LungCancerDataset/results/20260511_172536 \
  -m report \
  --model llama3.1:8b
```

LLM vygeneruje kompletní zprávu streamovaně do terminálu a uloží:
- `results/<timestamp>/llm/report.md` — výsledná zpráva
- `results/<timestamp>/llm/prompt_log.json` — použitý prompt pro dohledatelnost

---

## Chat mode — interaktivní dialog

```bash
python3 app/run_llm.py \
  -i data/datasets/LungCancerDataset/results/20260511_172536 \
  -m chat \
  --model llama3.1:8b
```

LLM dostane celý kontext (mapu, clustery, anomálie, statistiky) a čeká na otázky:

```
Chat mode — ask questions about the dataset. Type 'exit' to quit.

You: Které neurony mají nejvyšší podíl pacientů s rakovinou plic?
Assistant: Neuron 3_17 obsahuje 27 vzorků, z nichž 93 % jsou diagnostikováni s
rakovinou plic (LUNG_CANCER=YES)...

You: exit
```

Ukončení: `exit`, `quit` nebo `q`.

---

## Argumenty

| Argument | Povinný | Výchozí | Popis |
|---|---|---|---|
| `-i`, `--input` | ✅ | — | Cesta k výstupnímu adresáři SOM běhu |
| `-m`, `--mode` | ne | `report` | `report` nebo `chat` |
| `--model` | ne | `llama3.1:8b` | Název Ollama modelu |
| `--url` | ne | `http://localhost:11434` | URL Ollama serveru |

---

## Velké modely — druhý počítač (16 GB GPU, 64 GB RAM)

### Doporučené modely

| Model | VRAM (Q4) | Kde běží | Kvalita | Poznámka |
|---|---|---|---|---|
| `qwen2.5:32b` | ~19 GB | 16 GB GPU + ~3 GB RAM offload | ⭐⭐⭐⭐⭐ | Nejlepší poměr kvality/velikosti pro tuto konfiguraci |
| `gemma2:27b` | ~16 GB | celý v GPU | ⭐⭐⭐⭐ | Těsně se vejde do 16 GB VRAM |
| `mistral-small:22b` | ~13 GB | celý v GPU | ⭐⭐⭐⭐ | Rychlý, spolehlivý |
| `llama3.1:70b` | ~41 GB | 16 GB GPU + ~25 GB RAM offload | ⭐⭐⭐⭐⭐ | Nejvyšší kvalita, ale pomalejší inference |
| `llama3.1:8b` | ~5 GB | celý v GPU | ⭐⭐⭐ | Ověřeno funkční, referenční |

```bash
# Na vzdáleném PC — spustit Ollama na všech rozhraních:
OLLAMA_HOST=0.0.0.0 ollama serve

# Stáhnout velký model:
ollama pull qwen2.5:32b

# Rebuild nexusom-analyst na větší základ (volitelné):
# Změnit v Modelfile: FROM qwen2.5:32b
ollama create nexusom-analyst -f app/llm/Modelfile
```

### Vzdálené spuštění z vývojového PC

`--url` argument předá adresu vzdáleného Ollama serveru:

```bash
python3 app/run_llm.py \
  -i data/datasets/LungCancerDataset/results/20260511_172536 \
  -m chat \
  --model qwen2.5:32b \
  --url http://192.168.1.xxx:11434
```

Report a chat mode fungují identicky — model běží na GPU PC, kontext se sestaví lokálně.

---

## dataset_context.txt — doménový kontext

Bez tohoto souboru LLM reportuje jen číselné hodnoty bez doménové interpretace.
Soubor se umístí do adresáře datasetu:

```
data/datasets/LungCancerDataset/dataset_context.txt
```

`build_context` ho hledá automaticky od adresáře výsledků výše až 4 úrovně.
Pokud `dataset_context.txt` neexistuje, použije se `ABOUT.MD` nebo `ABOUT.md` jako fallback
(Kaggle datasety ho typicky obsahují).

**Formát:**

```
DATASET: Lung Cancer Dataset
DOMAIN: Medical / Oncology
DESCRIPTION: Symptom-based lung cancer risk dataset with 3 000 patient records
SAMPLE: Each row represents one patient with a set of observed symptoms

COLUMNS:
- id: Unique patient identifier. Not a clinical measurement.
- GENDER: Patient sex. M = Male, F = Female.
- AGE: Age at time of recording. Unit: years.
- LUNG_CANCER: Diagnosis. YES = lung cancer confirmed, NO = not confirmed.

CATEGORIES:
- Binary symptom columns (SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE,
  CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING,
  SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN):
  1 = symptom absent, 2 = symptom present

DOMAIN KNOWLEDGE:
- SMOKING and YELLOW_FINGERS are correlated with higher cancer risk
- COUGHING, SHORTNESS_OF_BREATH, CHEST_PAIN are primary respiratory symptoms
- Clusters with LUNG_CANCER purity > 90 % represent high-risk patient profiles
```

Zdrojový text z `ABOUT.MD` datasetu slouží jako základ pro tento soubor.

---

## Kde se ukládají výstupy

```
data/datasets/LungCancerDataset/results/<timestamp>/
├── json/
│   └── llm_context.json          ← vstup pro LLM (generuje analysis/context.py)
├── csv/
│   └── sample_assignments.csv    ← sample_id → neuron, QE, is_outlier (per vzorek)
└── llm/
    ├── report.md                  ← výsledná zpráva (report mode)
    └── prompt_log.json            ← použitý prompt + metadata
```

---

## Přestavení nexusom-analyst modelu

Po aktualizaci `Modelfile` (např. změna `num_ctx`):

```bash
ollama create nexusom-analyst -f app/llm/Modelfile
```

Aktuální nastavení v `app/llm/Modelfile`:
- base model: `llama3.1:8b`
- `temperature 0` — deterministické výstupy
- `num_ctx 16384` — dostatečné pro SOM context ~7 400 tokenů + odpověď

---

## Řešení problémů

| Problém | Příčina | Řešení |
|---|---|---|
| `ConnectionError: Ollama is not running` | Server neběží | `ollama serve` |
| `Model 'X' not found` | Model není stažen | `ollama pull <model>` |
| `No SOM results found` | Špatná cesta `-i` | Předat timestamped results dir |
| `llm_context.json` chybí | Starý SOM běh | `python3 app/som/result_analyzer.py <dir>` |
| Odpovědi bez doménového kontextu | Chybí `dataset_context.txt` | Vytvořit soubor viz výše |
| Krátká/oříznutá odpověď | `num_ctx` příliš malé | Zvýšit v `llm_client.py` nebo `Modelfile` |
