# CNN-EA-SOM Integration Plan
**Projekt:** NexusSom - Inteligentní optimalizace SOM pomocí CNN a EA

**Cíl:** Vytvořit proof-of-concept systém, kde CNN analyzuje kvalitu SOM map a poskytuje feedback pro evoluční algoritmus.

---

## 🎯 Hlavní cíle projektu

1. **Proof of Concept**: CNN dokáže rozpoznat špatné mapy (např. vysoký poměr neaktivních neuronů, špatný rozměr mapy)
2. **Dynamická EA**: Hledání hyperparametrů z celého spektra (ne z diskrétní množiny)
   - Rozměr mapy: 3x3 až 300x300 (spojitý prostor)
   - Všechny parametry dynamicky
3. **CNN-řízená evoluce**: CNN sleduje průběh organizace a evoluce, dynamicky upravuje parametry EA a SOM
4. **Vícekanálové vstupy**: CNN analyzuje 3 mapy současně (U-Matrix, Distance Map, Dead Neurons Map)

---

## 📋 Implementační kroky

### **FÁZE 1: Příprava datového pipeline** ✅ HOTOVO

#### ✅ Krok 1.1: Generování mapy neaktivních neuronů - HOTOVO
- [x] Implementovat metodu v `app/som/visualization.py`
  - Metoda: `generate_dead_neurons_map(som, data, output_path)` ✅
  - Vstup: SOM objekt, trénovací data ✅
  - Výstup: PNG s vizualizací mrtvých neuronů (černá=mrtvé, bílá=aktivní) ✅
- [x] Integrovat do `generate_individual_maps()` v `visualization.py` ✅
- [x] Test: Vygenerováno úspěšně na testovacím SOM ✅
- [x] BONUS: Přidán parametr `show_title=False` pro mapy bez titulků (CNN kompatibilita) ✅

#### ✅ Krok 1.2: Centralizované ukládání map - HOTOVO
- [x] Upravit `app/ea/ea.py` - funkce `copy_maps_to_dataset()` ✅
  - Vytvořen sdílený adresář: `WORKING_DIR/maps_dataset/` ✅
  - Pro každý UID zkopírováno: `{uid}_u_matrix.png`, `{uid}_distance_map.png`, `{uid}_dead_neurons_map.png` ✅
- [x] Integrace do `evaluate_individual()` - automatické kopírování po generování map ✅
- [x] Test: EA vygenerovalo 35 jedinců, všechny mapy v `maps_dataset/` ✅
- [x] BONUS: Opravena sys.path manipulace pro EA ✅

#### ✅ Krok 1.3: Generování vícekanálových obrázků - HOTOVO
- [x] Vytvořena funkce `combine_maps_to_rgb()` v `app/ea/ea.py` ✅
  - Vstup: 3 PNG soubory (U-Matrix, Distance, Dead Neurons) ✅
  - Výstup: RGB PNG se 3 kanály ✅
    - R kanál: U-Matrix ✅
    - G kanál: Distance Map ✅
    - B kanál: Dead Neurons Map ✅
  - Zachovány původní rozměry (bez resize - resize bude v CNN prepare_data.py) ✅
- [x] Integrováno do EA po dokončení evoluce ✅
- [x] RGB obrázky uloženy v `maps_dataset/rgb/` ✅
- [x] Test: 35 RGB obrázků úspěšně vygenerováno ✅

#### ⏭️ Krok 1.4: Rozšíření results.csv - PŘESKOČENO
- [x] Results.csv již obsahuje všechny potřebné SOM hyperparametry ✅
  - UID, metriky (best_mqe, topographic_error, dead_neuron_ratio) ✅
  - Všechny hyperparametry (map_size, learning_rate, radius, batch, epoch_multiplier, atd.) ✅
- [x] CNN adaptováno pro čtení existujícího formátu ✅

---

### **FÁZE 2: Adaptace CNN modelu** 🔄 PROBÍHÁ

#### ✅ Krok 2.1: Úprava prepare_data.py - HOTOVO
- [x] Přepsán `app/cnn/src/prepare_data.py` pro práci s EA run directories ✅
  - Funkce `collect_ea_runs()`: skenování adresářů s EA běhy ✅
  - Funkce `prepare_dataset_from_runs()`: kombinování dat z více běhů ✅
  - Změna: `inactive_neuron_ratio` → `dead_neuron_ratio` ✅
  - Argparse pro flexibilní příkazovou řádku: `--runs-dir`, `--output` ✅
- [x] Dataset obsahuje: filepath (RGB PNG), quality_score, uid, run_dir ✅
- [x] Quality score vzorec:
  ```python
  # Váhy: 50% MQE, 30% TE, 20% Dead Neuron Ratio
  quality_score = 0.5 * (1 - norm_mqe) +
                  0.3 * (1 - norm_te) +
                  0.2 * (1 - norm_dead)
  ```
- [x] Test: Úspěšně vygenerováno 36 vzorků z 1 EA běhu ✅
- [x] Quality scores v rozsahu 0.12 - 0.90 (dobrá distribuce) ✅

#### 🔄 Krok 2.2: Model už podporuje RGB - K OVĚŘENÍ
- [ ] `app/cnn/src/model.py` už má input shape (224, 224, 3) ✓
- [ ] Ověřit kompatibilitu s RGB obrázky
- [ ] Přidat dokumentaci k modelu: význam kanálů (R=U-Matrix, G=Distance, B=Dead)

#### 🔄 Krok 2.3: Ověření CNN pipeline - K PROVEDENÍ
- [ ] Zkontrolovat `app/cnn/src/train.py` - ověřit že načítá RGB správně
- [ ] Zkontrolovat `app/cnn/src/predict.py` - přidat možnost analyzovat celý adresář map
- [ ] Test: Spustit trénování na malém datasetu (36 vzorků)

---

### **FÁZE 3: Generování testovacích dat** 📊

#### ✅ Krok 3.1: Malý testovací dataset (proof of concept)
- [ ] Vytvořit konfiguraci EA pro test: `app/test/ea-test-config.json`
  - Populace: 10 jedinců
  - Generace: 3
  - Rozměry map: různé (5x5, 10x10, 15x15, 20x20)
  - Různé parametry pro různorodost
- [ ] Spustit EA na malém datasetu (např. iris.csv)
- [ ] Zkontrolovat výstupy:
  - `maps/` obsahuje 30 vícekanálových obrázků (10 jedinců × 3 generace)
  - `results.csv` obsahuje 30 řádků s hyperparametry
- [ ] **Cílový počet:** 30-50 map pro první test CNN

#### ✅ Krok 3.2: CNN trénování na testovacích datech
- [ ] Zkopírovat data do `app/cnn/data/`
  - `raw_maps/` ← vícekanálové obrázky z EA
  - `results.csv` ← metriky a hyperparametry
- [ ] Spustit: `cd app/cnn && ./run.sh prepare`
- [ ] Spustit: `./run.sh train-lite` (rychlejší pro test)
- [ ] Ověřit, že CNN se naučí rozpoznat:
  - ✓ Špatné mapy (vysoký dead_neuron_ratio, špatný map_size)
  - ✓ Dobré mapy (nízké MQE, TE)

---

### **FÁZE 4: Velká testovací kampaň** 🚀

#### ✅ Krok 4.1: Příprava různorodých datasetů
- [ ] Připravit 10 reálných datasetů různých velikostí:
  - Malé (50-200 vzorků): Iris, Wine, Breast Cancer
  - Střední (200-1000): Digits, Fashion
  - Velké (1000+): vlastní data
- [ ] Vygenerovat 10 syntetických datasetů:
  - Pomocí `make_blobs`, `make_circles`, `make_moons`
  - Různé počty clusters (2-10)
  - Různé dimenze (2-50)

#### ✅ Krok 4.2: Spuštění EA na všech datasetech
- [ ] Vytvořit skript: `app/ea/run_campaign.py`
  - Pro každý dataset:
    - Populace: 50 jedinců
    - Generace: 10
    - Různé map_size (5x5 až 50x50)
  - Všechny výstupy do `results/campaign_TIMESTAMP/`
- [ ] **Cílový počet:** 10.000 map (20 datasetů × 50 jedinců × 10 generací)
- [ ] Spočítat dobu běhu, odhadnout potřebné zdroje

#### ✅ Krok 4.3: Trénování CNN na velkém datasetu
- [ ] Zkopírovat všechny mapy do `app/cnn/data/raw_maps/`
- [ ] Agregovat všechny `results.csv` do jednoho
- [ ] Spustit: `./run.sh prepare`
- [ ] Spustit: `./run.sh train` (standardní model, více epoch)
  - Parametry: `--epochs 100 --batch-size 32`
- [ ] Evaluace modelu: `./run.sh evaluate`
- [ ] Zkontrolovat metriky: MSE, MAE, R²

---

### **FÁZE 5: Dynamická EA s CNN feedbackem** 🔄

#### ✅ Krok 5.1: Integrace CNN do EA
- [ ] Vytvořit modul: `app/integration/cnn_evaluator.py`
  - Třída: `CNNQualityEvaluator`
  - Metody:
    - `__init__(model_path)`: Načte natrénovaný CNN model
    - `evaluate_map(multichannel_image_path)`: Vrátí CNN quality score
    - `evaluate_batch(image_paths)`: Batch evaluace pro rychlost
- [ ] Test: Načíst model, evaluovat testovací mapu

#### ✅ Krok 5.2: CNN-augmentovaná fitness funkce
- [ ] Upravit `app/ea/ea.py` - přidat hybridní fitness
  - Původní fitness: `best_mqe`, `topographic_error`, `dead_neuron_ratio`
  - CNN fitness: `cnn_quality_score`
  - Kombinovaný fitness:
    ```python
    combined_fitness = 0.6 * original_fitness + 0.4 * cnn_quality_score
    ```
- [ ] Přidat parametr do EA configu: `use_cnn_evaluation: true/false`

#### ✅ Krok 5.3: Dynamické vyhledávání hyperparametrů
- [ ] Upravit `app/ea/ea.py` - funkce `random_config()`
  - `map_size`: `(random.randint(3, 300), random.randint(3, 300))`
  - Spojité hodnoty pro všechny parametry (místo diskrétní množiny)
  - Příklad:
    ```python
    'start_learning_rate': random.uniform(0.01, 1.0)
    'start_radius_init_ratio': random.uniform(0.05, 1.5)
    'epoch_multiplier': random.uniform(1.0, 50.0)
    ```
- [ ] Upravit `crossover()` a `mutate()` pro spojité parametry
  - Crossover: průměr hodnot nebo uniform crossover
  - Mutace: Gaussian noise nebo uniform mutation

#### ✅ Krok 5.4: Adaptivní úprava parametrů během běhu
- [ ] Přidat do EA: CNN-based parameter adaptation
  - Po každé generaci:
    - CNN analyzuje nejlepší mapy
    - Pokud CNN detekuje špatné vzory (např. příliš mnoho mrtvých neuronů):
      - Zmenšit `map_size` v příštích generacích
      - Zvýšit `epoch_multiplier`
    - Pokud CNN vidí dobrou organizaci:
      - Zachovat současné parametry
      - Fine-tuning kolem dobrých hodnot
- [ ] Implementovat adaptivní `SEARCH_SPACE` během běhu

---

### **FÁZE 6: Testování a validace** ✅

#### ✅ Krok 6.1: Proof of Concept testy
- [ ] **Test 1**: CNN rozpozná špatné mapy
  - Vytvořit záměrně špatnou mapu (5x5 pro 1000 vzorků)
  - CNN by mělo dát nízký score (<0.3)
- [ ] **Test 2**: CNN rozpozná dobré mapy
  - Vytvořit optimální mapu (správný map_size)
  - CNN by mělo dát vysoký score (>0.7)
- [ ] **Test 3**: CNN-řízená EA konverguje rychleji
  - Spustit EA bez CNN: 20 generací
  - Spustit EA s CNN: 20 generací
  - Porovnat kvalitu Pareto fronty

#### ✅ Krok 6.2: Srovnání na reálných vs. generovaných datech
- [ ] Spustit EA+CNN na 10 reálných datasetech
- [ ] Spustit EA+CNN na 10 generovaných datasetech
- [ ] Analyzovat rozdíly:
  - Které parametry CNN preferuje?
  - Liší se optimální `map_size` pro reálná vs. generovaná data?
  - Je CNN bias vůči určitým typům dat?
- [ ] Vizualizovat výsledky (scatter plots, histogramy)

#### ✅ Krok 6.3: Dokumentace výsledků
- [ ] Vytvořit report: `results/PROOF_OF_CONCEPT_REPORT.md`
  - CNN metriky (MSE, MAE, R²)
  - Příklady špatných/dobrých map detekovaných CNN
  - Grafy: CNN score vs. original metrics
  - Pareto fronty: s CNN vs. bez CNN
- [ ] Připravit prezentaci s výsledky

---

## 🛠️ Technické detaily

### Struktura adresářů po implementaci

```
app/
├── ea/
│   ├── ea.py                      # ✅ Upraveno: RGB kombinování, copy_maps_to_dataset(), combine_maps_to_rgb()
│   └── run_campaign.py            # 🔜 Plánováno: hromadné spouštění EA
├── som/
│   ├── visualization.py           # ✅ Upraveno: generate_dead_neurons_map(), show_title parameter
│   └── multichannel.py            # ⏭️ Nepotřebné (RGB kombinování už v ea.py)
├── cnn/
│   ├── src/
│   │   ├── model.py               # ✅ Beze změn (už podporuje 3 kanály)
│   │   ├── prepare_data.py        # ✅ Upraveno: nový formát s EA runs, dead_neuron_ratio
│   │   ├── train.py               # 🔄 K ověření: RGB loading
│   │   └── predict.py             # 🔜 K úpravě: batch prediction
│   └── data/
│       └── processed/
│           └── dataset.csv        # ✅ Vygenerován s RGB filepaths a quality scores
├── integration/                   # 🔜 Plánováno (Fáze 5)
│   ├── __init__.py
│   ├── cnn_evaluator.py           # 🔜 CNN wrapper pro EA
│   └── adaptive_ea.py             # 🔜 Adaptivní EA logika
└── test/
    ├── results/                   # ✅ EA výsledky
    │   └── TIMESTAMP/
    │       ├── maps_dataset/      # ✅ Centrální adresář map
    │       │   ├── {uid}_u_matrix.png
    │       │   ├── {uid}_distance_map.png
    │       │   ├── {uid}_dead_neurons_map.png
    │       │   └── rgb/           # ✅ RGB kombinované mapy
    │       │       └── {uid}_rgb.png
    │       ├── individuals/       # ✅ Detaily jednotlivců
    │       └── results.csv        # ✅ S hyperparametry a metrikami
    └── ea-iris-config.json        # ✅ Testovací konfigurace

results/                           # 🔜 Pro velké kampaně
└── campaign_TIMESTAMP/
    ├── dataset_01/
    ├── dataset_02/
    └── ...
```

---

## 📊 Časový odhad

| Fáze | Kroky | Odhadovaný čas |
|------|-------|----------------|
| **FÁZE 1** | Datový pipeline | 4-6 hodin |
| **FÁZE 2** | Adaptace CNN | 2-3 hodiny |
| **FÁZE 3** | Malý test | 2 hodiny (včetně běhu EA) |
| **FÁZE 4** | Velká kampaň | 8-12 hodin (hlavně čekání na EA) |
| **FÁZE 5** | Dynamická EA | 6-8 hodin |
| **FÁZE 6** | Testování | 4-6 hodin |
| **CELKEM** | | **26-37 hodin** |

*Pozn.: Čas běhu EA závisí na hardwaru a velikosti datasetů*

---

## 🎓 Prezentace výsledků

### Klíčové ukazatele pro proof of concept:
1. ✅ **CNN accuracy**: MSE < 0.05, R² > 0.80
2. ✅ **Detekce špatných map**: Precision > 90% pro quality_score < 0.3
3. ✅ **EA konvergence**: S CNN o 30-50% rychlejší dosažení Pareto fronty
4. ✅ **Adaptivita**: Automatické zmenšení map_size při detekci mrtvých neuronů

### Demo scénář:
1. Ukázat špatnou mapu (5x5 pro velká data) → CNN dá nízký score
2. Ukázat dobrou mapu (optimální rozměr) → CNN dá vysoký score
3. Spustit EA s CNN → sledovat adaptaci parametrů
4. Porovnat výsledky: EA bez CNN vs. EA s CNN

---

## 📝 Poznámky

- **Priorita 1**: FÁZE 1-3 (základní funkčnost, malý test)
- **Priorita 2**: FÁZE 4 (velká kampaň pro robustní CNN)
- **Priorita 3**: FÁZE 5-6 (pokročilé funkce, validace)

- **Quick wins**: Krok 1.1, 1.3, 2.2 lze udělat rychle
- **Časově náročné**: Krok 4.2 (běh EA), 5.3-5.4 (implementace)
- **Kritické**: Krok 1.4, 2.2 (kompatibilita formátů)

---

**Další krok:** Začít s FÁZE 1, Krok 1.1 - Implementace mapy neaktivních neuronů.

Jste připraveni začít? 🚀
