# Prezentace: Evoluční optimalizace hyperparametrů SOM
## Biologií inspirované metody — NexusSom projekt

---

## Slide 1 — Titulní stránka

**Název:** Víceúčelová evoluční optimalizace hyperparametrů samoorganizující se mapy

**Podtitul:** NSGA-II pro hledání optimálního nastavení trénování SOM

**Obsah:**
- Jméno, předmět, datum
- Krátký popis: co projekt řeší (automatické hledání nejlepšího způsobu trénování SOM na reálných datech)

---

## Slide 2 — Motivace: Proč optimalizovat SOM?

**Obsah:**
- SOM (Kohonen) má ~15 hyperparametrů: velikost mapy, rychlost učení (start/end/typ poklesu), poloměr sousedství, počet epoch, velikost dávky atd.
- Ruční nastavení je nákladné a nespolehlivé — prostor je příliš velký na grid search
- Různé hyperparametry vedou k různým kompromisům: *rychlé trénování vs. kvalita mapy*
- **Cíl EA:** automaticky najít sadu Pareto-optimálních konfigurací

**Klíčová otázka:** "Jak natrénovat SOM co nejlépe, co nejrychleji, s minimem mrtvých neuronů?"

---

## Slide 3 — Proč víceúčelová optimalizace?

**Obsah:**
- SOM má 4 (resp. 5) navzájem si konkurující cíle — nelze je sloučit do jednoho skaláru bez ztráty informace:

| Cíl | Popis | Směr |
|-----|-------|------|
| MQE (Mean Quantization Error) | Přesnost mapování dat na neurony | minimalizovat |
| Topografická chyba | Zachování topologie vstupního prostoru | minimalizovat |
| Poměr mrtvých neuronů | Podíl neuronů bez přiřazených vzorků | minimalizovat |
| Doba trénování | Výpočetní náklady | minimalizovat |
| CNN skóre vizuální kvality | Kvalita vizualizace mapy (volitelné) | maximalizovat |

- Výstupem není jedno "nejlepší" řešení, ale **Pareto fronta** — sada kompromisních řešení
- Uživatel si pak vybírá dle priority (rychlost vs. kvalita)

---

## Slide 4 — NSGA-II: Proč tato varianta?

**Obsah:**
- **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) — Deb et al., 2002
- Klíčové důvody volby:

**Vhodné pro tento problém:**
1. Navržen přímo pro víceúčelovou optimalizaci → bez potřeby váhování cílů
2. Smíšený prohledávací prostor (spojité + kategorické parametry) → flexibilní genetické operátory
3. Elitismus přes Pareto archiv → dobré řešení se neztrácí
4. Aktivní udržování diverzity přes crowding distance → Pareto fronta je rovnoměrně pokrytá
5. Škáluje dobře s počtem cílů (4–5 cílů)

---

## Slide 5 — Proč NE jiné přístupy?

**Obsah:**

| Metoda | Problém |
|--------|---------|
| **Jednoobjektový GA** | Vyžaduje váhování cílů (subjektivní), ztrácí trade-off informaci |
| **Diferenciální evoluce (DE)** | Navržena pro ryze spojité prostory; kategorické parametry (typ poklesu LR, typ mapy) jsou neobratné |
| **PSO** | Částice nemají přirozený způsob pohybu přes kategorické hodnoty; žádný vestavěný mechanismus Pareto archivace |
| **Grid search** | Prostor 15 parametrů (část spojitá) → kombinatorická exploze; neefektivní |
| **Náhodné prohledávání** | Čisté náhodné — žádné využití nalezených dobrých řešení (exploit); vysoké plýtvání výpočty |
| **Tabu search / horolezec** | Navržen pro jednoobjektovou optimalizaci; uvíznutí v lokálním optimu |

**Závěr:** NSGA-II je přirozená volba pro smíšený prostor s více konkurujícími si cíly.

---

## Slide 6 — Architektura algoritmu

**Obsah (diagram / schéma):**

```
Inicializace populace (N jedinců, náhodné konfigurace)
         ↓
┌─── Generace 1..G ─────────────────────────────────────┐
│                                                         │
│  Paralelní vyhodnocení populace (multiprocessing Pool) │
│         ↓                                               │
│  Kombinace populace + archiv (elitismus)                │
│         ↓                                               │
│  Non-dominated sorting → Pareto fronty (ranky)         │
│         ↓                                               │
│  Crowding distance → diverzita v rámci fronty          │
│         ↓                                               │
│  Turnajová selekce (k=3)                               │
│         ↓                                               │
│  SBX křížení + polynomiální mutace → nová generace     │
│         ↓                                               │
│  Validace a oprava porušených omezení                  │
└────────────────────────────────────────────────────────┘
         ↓
Pareto fronta (archiv nejlepších řešení)
```

**Parametry:** Populace: 20, Generace: 30, Turnaj: k=3, SBX η=20, Mutace η=20, P(mutace)=0.1

---

## Slide 7 — Non-dominated sorting a Crowding Distance

**Obsah:**
- **Non-dominated sorting:** Jedinci jsou řazeni do front podle dominance
  - Jedinec A *dominuje* B, pokud je ve všech cílech ≤ B a alespoň v jednom < B
  - Front 0 = Pareto fronta (nikdo ho nedominuje) → nejlepší jedinci
  - Front 1 = nejlepší po odstranění fronty 0, atd.

- **Crowding distance:** Míra "prostoru" kolem jedince na frontě
  - Hraniční jedinci (extrémní hodnoty cíle) → ∞ vzdálenost (vždy zachováni)
  - Vyšší crowding distance = méně přeplněná oblast = prémiová diverzita
  - Při stejném ranku vítězí jedinec s větší crowding distance → rovnoměrné pokrytí Pareto fronty

**Kód:** `non_dominated_sort()` + `crowding_distance_assignment()` v `ea.py`

---

## Slide 8 — Genetické operátory

**Obsah:**

**SBX křížení (Simulated Binary Crossover):**
- Navržen pro spojité parametry — simuluje binární křížení v reálném prostoru
- Parametr η (distribuční index): vyšší η = potomci blíže rodičům (exploitace vs. explorace)
- Aplikován na: `start_learning_rate`, `end_learning_rate`, `start_radius`, `epoch_multiplier`, atd.
- Vzorec: β závisí na náhodném u ∈ [0,1]: `child = 0.5 * ((1+β)*p1 + (1-β)*p2)`

**Polynomiální mutace:**
- Malé perturbace kolem stávající hodnoty, s pravděpodobností 0.1 na gen
- Distribuční index η: vyšší = menší mutace (jemné doladění)
- Zachovává hranice prohledávacího prostoru

**Kategorické parametry** (typ poklesu LR, typ mapy, typ dávky):
- Křížení: uniformní výběr z rodiče 1 nebo 2
- Mutace: náhodná náhrada z přípustných hodnot

**Validace a oprava omezení** (`validate_and_repair()`):
- `start_lr ≥ end_lr` (LR musí klesat)
- `start_radius ≥ end_radius` (poloměr musí klesat)
- `start_batch ≤ end_batch` (dávka musí růst)
- `growth_g = 0` pokud všechny křivky jsou lineární (zamezení funkčně duplicitních jedinců)

---

## Slide 9 — Prohledávací prostor

**Obsah:**

| Parametr | Typ | Rozsah |
|----------|-----|--------|
| `map_size` | diskrétní pár int | 5×5 až 20×20 |
| `processing_type` | kategorický | stochastic / deterministic / hybrid |
| `start_learning_rate` | float | [0.01, 1.0] |
| `end_learning_rate` | float | [0.001, 0.5] |
| `lr_decay_type` | kategorický | linear / exp / log / step |
| `start_radius_init_ratio` | float | [0.05, 1.0] |
| `radius_decay_type` | kategorický | linear / exp / log / step |
| `start_batch_percent` | float | [0.5, 15.0] |
| `end_batch_percent` | float | [1.0, 30.0] |
| `batch_growth_type` | kategorický | linear / exp / log |
| `epoch_multiplier` | float | [1.0, 20.0] |
| `normalize_weights_flag` | bool | True / False |
| `growth_g` | float | [1.0, 50.0] |
| `num_batches` | int | [1, 20] |

**Celkem ~15 parametrů, část spojitá, část kategorická** → smíšený operátor nezbytný

---

## Slide 10 — Paralelní vyhodnocení a cache

**Obsah:**
- Každý jedinec = jedno spuštění SOM trénování → výpočetně nákladné (sekundy až minuty)
- **Paralelní Pool:** `multiprocessing.Pool(min(10, cpu_count))` — simultánní vyhodnocení celé populace
- **UID-based cache:** každá konfigurace má deterministický hash (UID) → duplicitní konfigurace v různých generacích se nevyhodnocují znovu
- **Elitismus:** archiv nejlepší Pareto fronty přežívá do další generace bez přehodnocování

```
UID = MD5(sorted(config_dict))
if UID in EVALUATED_CACHE:
    return cached_result   # bez spuštění SOM
```

**Výsledek:** řádově vyšší efektivita — zejm. v pozdějších generacích s konvergující populací

---

## Slide 11 — Přínosy: Data pro trénování neuronových sítí

**Obsah:**
- Každý vyhodnocený jedinec generuje **strukturovaná data** pro tři sítě:

| Co se uloží | Pro koho |
|-------------|----------|
| Konfigurace + výsledné metriky (MQE, TE, dead ratio, čas) | **MLP "Prorok"** — predikce kvality z hyperparametrů |
| 10 checkpointů průběhu trénování (MQE, TE v čase) | **LSTM "Věštec"** — predikce finální kvality z raného průběhu |
| RGB vizualizace mapy (U-matice + vzdálenostní mapa + mrtvé neurony) | **CNN "Oko"** — klasifikace vizuální kvality |

- Po prvním EA běhu (1 reálný dataset + 9 virtuálních): 1 500 řádků pro MLP, 1 500 sekvencí pro LSTM, 1 500 RGB obrázků pro CNN
- Sítě se natrénují a **zpětně integrují do EA** jako urychlovače

---

## Slide 12 — Uzavřená smyčka: EA + Neuronové sítě

**Obsah (diagram):**

```
                    ┌──────────────────────┐
                    │   EA (NSGA-II)        │
                    │                      │
  Konfigurace  ──→  │  MLP pre-screen      │ → přeskočit zjevně špatné konfigurace
                    │  (The Prophet)        │   (bez spuštění SOM)
                    │                      │
  SOM trénování ──→ │  LSTM early stop     │ → ukončit trénování předčasně
                    │  (The Oracle)         │   pokud průběh neperspektivní
                    │                      │
  Vizualizace   ──→ │  CNN kvalita         │ → 5. Pareto cíl
                    │  (The Eye)            │   (vizuální kvalita mapy)
                    └──────────────────────┘
                              ↓
                      Pareto fronta
                              ↓
                    Nová trénovací data
                              ↓
                    Přetrénování sítí
                              ↑─────────┘ (iterativní zlepšování)
```

- **MLP:** Predikuje MQE, TE, dead_ratio z hyperparametrů → filtruje bad kandidáty bez spuštění SOM
- **LSTM:** Sleduje prvních N checkpointů a předpovídá finální kvalitu → předčasné zastavení
- **CNN:** Hodnotí vizuální strukturu Pareto-optimální mapy → 5. cíl v NSGA-II

---

## Slide 13 — Proč NE neuroevoluce (přímá)?

**Obsah:**
- Neuroevoluce = evoluční optimalizace vah/architektury NN (NEAT, CMA-ES na vahách, atd.)
- Zde se optimalizují **hyperparametry SOM**, ne váhy NN → jiný problém
- Naše NNs (MLP, LSTM, CNN) jsou trénované gradientním sestupem (supervised/unsupervised)
- EA je použit jako **globální optimalizátor** pro prostor, kde gradient není dostupný
  - SOM trénování není diferencovatelné vzhledem k hyperparametrům
  - Různé konfigurace vedou k různým výstupům bez analytické závislosti

**Vazba na přednáškové téma č. 10 (neuroevoluce):**
- Zpětná smyčka EA → data → NN → EA je formou *koevoluce*: EA zlepšuje data pro NN, NN zlepšuje efektivitu EA

---

## Slide 14 — Výsledky a diskuse

**Obsah:**
- Ukázka Pareto fronty z reálného běhu (graf: MQE vs. čas, různé topografické chyby barevně)
- Porovnání konfigurací z fronty fronty 0: různé trade-off profily
- CNN skóre: dobré mapy mají skóre ~0.99, špatné ~0.001–0.005
- Typické výsledky: 5–15 Pareto-optimálních řešení po 30 generacích, populace 20

**Omezení a budoucí práce:**
- MLP predikuje záporné hodnoty pro TE a dead_ratio → nutné ořezání (clip/ReLU)
- LSTM trénován hlavně na simulovaných sekvencích (real checkpointů jen ~500)
- CNN trénován na omezeném labeled datasetu → přetrénování při binární klasifikaci

---

## Slide 15 — Shrnutí

**Obsah:**

**Co bylo implementováno:**
- NSGA-II s SBX křížením a polynomiální mutací pro smíšený prostor
- 4–5 Pareto cílů: MQE, topografická chyba, mrtvé neurony, čas, CNN vizuální skóre
- Paralelní vyhodnocení + UID cache pro efektivitu
- Validace a oprava omezení prohledávacího prostoru
- Integrační vrstva pro 3 neuronové sítě (MLP, LSTM, CNN)

**Proč NSGA-II:**
- Víceúčelový problém bez zřejmého váhování → Pareto přístup přirozený
- Smíšený prostor → SBX + polynomiální mutace vs. DE/PSO
- Elitismus přes archiv → žádná ztráta kvalitních řešení

**Hlavní přínos:**
- Automatická generace trénovacích dat pro NN
- Uzavřená smyčka EA ↔ NN pro iterativní zlepšování efektivity optimalizace
