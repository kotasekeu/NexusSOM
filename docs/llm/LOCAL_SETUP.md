# Lokální LLM — instalace a konfigurace

Ollama na Windows s GPU akcelerací.

---

## Hardware

| Komponenta | Specifikace |
|------------|-------------|
| CPU | Intel Core i5 14. gen |
| RAM | 64 GB |
| GPU | MSI RTX 5060 Ti 16 GB VRAM |

---

## 1. Instalace Ollama (Windows)

1. Stáhnout instalátor z **https://ollama.com/download/windows**
2. Spustit `.exe` — nainstaluje Ollama jako systémovou službu
3. Ollama se spustí automaticky při startu Windows a běží na pozadí

Ověření:
```powershell
ollama --version
# Ollama je dostupná na http://localhost:11434
```

---

## 2. Doporučené modely

### Primární — `qwen2.5:14b`

Nejlepší volba pro tento projekt. Silný na strukturovanou analýzu dat, tabulky, interpretaci čísel. Celý se vejde do 16 GB VRAM → rychlá inference bez CPU offloadingu.

```powershell
ollama pull qwen2.5:14b
# ~9 GB download, ~9 GB VRAM při běhu
```

### Alternativa — `gemma2:27b`

Silnější reasoning, větší model. Těsně se vejde do 16 GB VRAM (15 GB v Q4).

```powershell
ollama pull gemma2:27b
# ~15 GB download, ~15 GB VRAM při běhu
```

### Záloha — `phi4:14b`

Microsoft Phi-4, výborný na analytické úlohy, stejná velikost jako qwen2.5:14b.

```powershell
ollama pull phi4:14b
# ~9 GB download, ~9 GB VRAM při běhu
```

### Rychlý fallback — `llama3.1:8b`

Aktuální default v projektu, slabší na analýzu dat ale nejrychlejší.

```powershell
ollama pull llama3.1:8b
# ~5 GB download, ~5 GB VRAM
```

---

## 3. Ověření GPU akcelerace

Po instalaci zkontrolovat, že Ollama skutečně používá GPU:

```powershell
ollama run qwen2.5:14b "řekni ahoj"
# V jiném terminálu zároveň:
nvidia-smi
# VRAM usage by se měl zvýšit o ~9 GB
```

Pokud Ollama nepoužívá GPU, zkontrolovat:
- CUDA drivers jsou aktuální (GeForce Experience nebo ručně z nvidia.com)
- `ollama ps` zobrazuje model s GPU sloupcem

---

## 4. Nastavení v NexusSom

Parametry se nastavují v UI (záložka Chat → Nastavení LLM) nebo přímo v kódu:

```python
# app/llm/src/llm_client.py — výchozí hodnoty
OllamaClient(
    base_url="http://localhost:11434",
    model="qwen2.5:14b",
    num_ctx=16384,          # kontext okno v tokenech
    temperature_report=0,   # report: deterministický
    temperature_chat=0.3,   # chat: přirozená konverzace
    num_gpu=-1,             # -1 = všechny vrstvy na GPU
)
```

**Context window** (`num_ctx`):
| Hodnota | Použití |
|---------|---------|
| 4 096 | malé datasety, rychlé odpovědi |
| 16 384 | standard — vejde se většina llm_context.json |
| 32 768 | velké mapy (30×30+), bohatý kontext |

---

## 5. Spuštění UI

```powershell
# Z root adresáře projektu
.venv\Scripts\python.exe run_ui.py
# nebo
.venv\Scripts\streamlit.exe run app\ui\app.py
```

Otevře se na **http://localhost:8501**

---

## 6. Modely v paměti

Ollama drží model načtený v VRAM po dobu nečinnosti (default 5 minut). Pro uvolnění:

```powershell
ollama stop qwen2.5:14b
```

Pro změnu doby držení v paměti (např. 1 hodina):
```powershell
$env:OLLAMA_KEEP_ALIVE = "1h"
ollama serve
```
