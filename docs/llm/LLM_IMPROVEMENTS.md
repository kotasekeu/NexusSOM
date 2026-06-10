# LLM — plánovaná vylepšení

Vylepšení pro obhajobu předmětu Aplikované použití LLM.  
Základ funguje: grounding (odmítnutí off-topic), streaming, persistentní chat, context window konfigurace.

---

## ✅ 2. Context window monitor

**Stav:** implementováno (`app/ui/app.py`)

Zobrazuje nad historií chatu průběžné využití kontextového okna:

```
Context: 8,241 / 16,384 tokenů  [████████████░░░░░░░░]  50%
| kontext: 7,843 tok · konverzace: 398 tok
```

- Odhad: 1 token ≈ 4 znaky (funguje pro EN/CZ mix)
- Breakdown: fixní část (system prompt + SOM kontext) vs. rostoucí konverzace
- Barvy: zelená < 60 %, oranžová 60–85 %, červená > 85 %
- Při > 85 % zobrazí varování s doporučením smazat chat

**Akademická hodnota:** demonstruje znalost context window limitací LLM; viditelně ukazuje proč velké mapy (30×30 = ~8K tokenů) nechávají málo prostoru pro konverzaci.

---

## 1. Prompt engineering — system prompt + few-shot

**Stav:** neimplementováno  
**Soubor:** `app/llm/src/context_builder.py` → funkce `_build_system_prompt()`

### Co přidat

**A) Lepší role a formát:**
```
Jsi senior datový analytik specializovaný na interpretaci SOM výsledků.
Odpovědi formátuj markdown: bullet pointy pro výčty, tabulky pro srovnání.
Vždy cituj konkrétní data (ID neuronu, hodnoty dimenzí).
Přemýšlej krok za krokem než odpovíš (chain-of-thought).
```

**B) Few-shot příklady přímo v system promptu:**
```
Příklad správné odpovědi na "Popiš cluster 3_4":
Neuron 3_4 obsahuje 12 vzorků (průměrné QE 0.08).
Dominantní kategorie: Malignant (83 %).
Dimenze nad průměrem: radius_mean = 18.2 (globální průměr 14.1, +29 %).
Vzorky jsou typicky vzorky s velkým průměrem nádoru a výraznou texturou.

Příklad nesprávné odpovědi:
Cluster 3_4 je skupinou podobných vzorků. [❌ bez konkrétních dat]
```

**Postup:**
1. Rozšířit `_build_system_prompt()` v `context_builder.py`
2. Přidat sekci EXAMPLES s 2–3 vzory Q&A
3. Přidat formátovací instrukce
4. Porovnat odpovědi before/after — ukázat na obhajobě

**Akademická hodnota:** prompt engineering je klíčové téma; few-shot learning bez dotrénování modelu.

---

## ✅ 3. Quick action tlačítka

**Stav:** implementováno (`app/ui/app.py`)  
**Soubor:** `app/ui/app.py` → Chat tab, pod context monitorem

### Co přidat

Řada tlačítek nad chat inputem s předdefinovanými prompty:

| Tlačítko | Prompt |
|----------|--------|
| 📊 Popiš clustery | *"Shrň hlavní skupiny vzorků. Pro každou skupinu uveď dominantní kategorii, typické hodnoty dimenzí a počet vzorků."* |
| ⚠ Najdi anomálie | *"Jaké vzorky jsou anomální a proč? Vysvětli každou anomálii v kontextu domény."* |
| 📐 Srovnej dimenze | *"Které dimenze nejvíce odlišují jednotlivé clustery? Uveď konkrétní hodnoty."* |
| 🗺 Přehled mapy | *"Popiš celkovou kvalitu SOM: MQE, topografická chyba, mrtvé neurony. Co to znamená pro spolehlivost výsledků?"* |

**Postup:**
1. Přidat `st.columns` s tlačítky nad `st.chat_input`
2. Při kliknutí vložit prompt jako by ho napsal uživatel (přidat do history + spustit odpověď)

**Akademická hodnota:** prompt templates — strukturované přístupy k dotazování LLM.

---

## 4. Conversation summarization

**Stav:** neimplementováno  
**Soubory:** `app/ui/app.py` + nová funkce v `app/llm/src/`

### Problém

LLM history roste s každou zprávou. Po ~20 výměnách přesáhne context window a starší zprávy jsou tiše ignorovány. Grounding se zhoršuje.

### Řešení

Když konverzace překročí práh (např. 70 % context window), shrň starší zprávy přes LLM a nahraď je jednou summarizační zprávou:

```python
def summarize_history(client, history, keep_last_n=6):
    """
    Shrne starší zprávy do jednoho bloku a zachová posledních keep_last_n.
    Vrátí novou, kratší history se zachovaným system promptem a kontextem.
    """
    fixed  = history[:3]          # system + kontext + intro (nikdy neshrnutelné)
    old    = history[3:-keep_last_n]
    recent = history[-keep_last_n:]

    if not old:
        return history

    summary_prompt = (
        "Shrň následující konverzaci do max. 5 bullet pointů. "
        "Zachovej konkrétní fakta (hodnoty, ID neuronů). "
        "Odpověz pouze shrnutím, bez úvodu:\n\n"
        + "\n".join(f"{m['role']}: {m['content']}" for m in old)
    )
    summary = client.generate(summary_prompt)

    summary_msg = {
        "role": "assistant",
        "content": f"[Shrnutí předchozí konverzace]\n{summary}"
    }
    return fixed + [summary_msg] + recent
```

Trigger: automaticky při > 80 % využití okna, nebo tlačítko "Zhustit historii".

**Akademická hodnota:** memory management v LLM aplikacích — reálný inženýrský problém.

---

## 5. Grounding check

**Stav:** neimplementováno  
**Soubory:** `app/ui/app.py` + nová funkce

### Problém

Malé lokální modely (llama3.1:8b) mohou ignorovat kontext a odpovídat ze svého tréninku — halucinace. Bez mechanismu detekce to není vidět.

### Řešení

Po každé odpovědi zkontrolovat, zda LLM cituje konkrétní hodnoty/identifikátory z kontextu. Pokud ne, zobrazit varování.

```python
def check_grounding(response: str, llm_context: dict) -> dict:
    """
    Heuristická kontrola: hledá neuron klíče (např. "3_4"), 
    sample ID, nebo číselné hodnoty z kontextu v odpovědi.
    Vrátí {'grounded': bool, 'found': [str], 'score': float}
    """
    import re
    found = []

    # Neuron klíče (i_j formát)
    neuron_keys = set(llm_context.get('clusters', {}).keys())
    for key in neuron_keys:
        if key in response:
            found.append(f'neuron {key}')

    # Číselné hodnoty z dimension_stats (zaokrouhlené)
    for dim, stats in llm_context.get('dimension_stats', {}).items():
        mean_str = f"{stats.get('mean', 0):.2f}"
        if mean_str in response:
            found.append(f'{dim}={mean_str}')

    score = min(len(found) / 3, 1.0)   # 3+ reference = plně grounded
    return {'grounded': len(found) > 0, 'found': found, 'score': score}
```

UI: zobrazit pod každou odpovědí asistenta:
- ✅ Grounded — cituje: `neuron 3_4`, `radius_mean=14.12`
- ⚠ Ověř odpověď — bez konkrétních citací z dat

**Akademická hodnota:** přímo adresuje hallucination problem — jeden z klíčových témat oboru; ukazuje měřitelný přístup k evaluaci LLM výstupů.

---

## Priorita implementace — Fáze 1

| # | Název | Čas | Stav |
|---|-------|-----|------|
| 2 | Context window monitor | 30 min | ✅ hotovo |
| 1 | System prompt + few-shot | 1–2 hod | ✅ hotovo |
| 3 | Quick action tlačítka | 1 hod | ✅ hotovo |
| 4 | Conversation summarization | 3–4 hod | ⬜ TODO |
| 5 | Grounding check | 2–3 hod | ⬜ TODO |

---

## Fáze 2 — Databáze + API + Tool calling

### Kontext a motivace

Aktuální pipeline ukládá veškerá data do CSV a JSON souborů na disku. To funguje pro jednoho uživatele na lokálním stroji, ale má tři strukturální problémy:

1. **LLM kontext:** celý `llm_context.json` (až 31K+ znaků pro 30×30 mapu) se posílá modelu na začátku každého chatu. Zaplní ~50–80 % context window ještě před první otázkou. Pro mapy 50×50+ to přestane fungovat úplně.
2. **Frontend škálovatelnost:** přímá práce se soubory v UI (Streamlit čte `.npy`, `.json`, `.csv` přímo) nefunguje přes síť ani pro více uživatelů.
3. **Duplicitní data:** výsledky jsou v CSV i JSON, bez single source of truth.

Přechod na databázi + API řeší všechny tři najednou a LLM tool calling je přirozené rozšíření.

---

### 2.1 Databáze

**Doporučená strategie:** začít s SQLite (nulová závislost, jeden soubor), migrovat na PostgreSQL když bude potřeba víceuživatelský přístup nebo nasazení.

**Co přesunout z JSON/CSV do DB:**

| Aktuální soubor | Tabulka | Klíčové sloupce |
|-----------------|---------|-----------------|
| `csv/sample_assignments.csv` | `sample_assignments` | `run_id, sample_id, bmu_i, bmu_j, qe, qe_dim_*` |
| `json/clusters.json` | `clusters` | `run_id, neuron_key, sample_ids[]` |
| `json/quantization_errors.json` | `neuron_qe` | `run_id, neuron_key, qe, sample_count` |
| `json/extremes.json` | `anomalies` | `run_id, sample_id, reasons[]` |
| `run_metrics.json` | `runs` | `run_id, dataset, map_size, mqe, te, duration` |
| `pareto_front.csv` | `ea_archive` | `run_id, generation, uid, objectives` |

`llm_context.json` se **nepřesouvá** — generuje se on-demand z DB dotazů.

---

### 2.2 FastAPI backend

```
app/
  api/
    main.py           ← FastAPI app, CORS, lifespan
    routers/
      runs.py         ← GET /runs, GET /runs/{id}
      clusters.py     ← GET /runs/{id}/clusters, GET /runs/{id}/clusters/{neuron}
      anomalies.py    ← GET /runs/{id}/anomalies
      dimensions.py   ← GET /runs/{id}/dimensions/{name}/stats
      samples.py      ← GET /runs/{id}/samples/{sample_id}
    db.py             ← SQLAlchemy session, engine
    models.py         ← ORM modely
```

Klíčové endpointy pro LLM tool calling:

```
GET /runs/{id}/clusters?top=10&sort_by=sample_count
GET /runs/{id}/clusters/{neuron_key}
GET /runs/{id}/anomalies?limit=5
GET /runs/{id}/dimensions/{name}/stats
GET /runs/{id}/samples/{sample_id}/detail
GET /runs/{id}/summary          ← kompaktní přehled pro LLM init (~500 tokenů)
```

---

### 2.3 LLM Tool calling

Místo posílání celého kontextu na začátku dostane model:
- kompaktní summary (~500 tokenů): velikost mapy, základní metriky, počet clusterů
- definice nástrojů (tools) které může volat

```python
TOOLS = [
    {
        "name": "get_cluster_detail",
        "description": "Vrátí detaily konkrétního neuronu/clusteru: počet vzorků, dominantní kategorie, průměrné hodnoty dimenzí.",
        "parameters": {"neuron_key": "string (např. '3_4')"}
    },
    {
        "name": "get_top_anomalies",
        "description": "Vrátí top N nejanomaálnějších vzorků s vysvětlením.",
        "parameters": {"n": "int (default 5)"}
    },
    {
        "name": "get_dimension_stats",
        "description": "Vrátí statistiky konkrétní dimenze: min, max, mean, std, hodnoty per cluster.",
        "parameters": {"dimension_name": "string"}
    },
    {
        "name": "search_clusters",
        "description": "Najde clustery splňující podmínku (dominantní kategorie, min. počet vzorků).",
        "parameters": {"category": "string (volitelné)", "min_samples": "int (volitelné)"}
    },
]
```

**Agent loop** v `llm_client.py`:
```
uživatel: "Popiš clustery kde dominuje červené víno"
  → LLM: volá search_clusters(category="red", min_samples=10)
  → API vrátí: [neuron 3_4: 45 vzorků, neuron 7_2: 38 vzorků, ...]
  → LLM: odpoví na základě vrácených dat
```

Context window se nezaplní na startu — LLM si vyžádá jen co potřebuje.

**Modely s podporou tool calling v Ollama:**
- `qwen2.5:14b` ✓ (doporučený)
- `llama3.1:8b` ✓
- `phi4:14b` ✓
- `gemma2:27b` — omezená podpora

---

### 2.4 Priorita a návaznost

```
Fáze 1 (hotovo)     Fáze 2A              Fáze 2B              Fáze 2C
JSON/CSV soubory  →  SQLite + migrace  →  FastAPI endpointy →  Tool calling LLM
                      (1–2 dny)            (2–3 dny)            (2–3 dny)
```

Fáze 2A a 2B jsou užitečné nezávisle na LLM — FastAPI endpointy poslouží i novému frontendu (React, Vue, nebo lepší Streamlit).

**Poznámka k migraci dat:** aktuální JSON/CSV soubory zůstávají jako archiv. Import do DB se provede jednorázovým skriptem `app/tools/import_to_db.py` který projde všechny `results/` složky a naplní tabulky.
