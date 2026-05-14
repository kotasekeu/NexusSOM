# LSTM — Vizualizace natrénovaného modelu (Phase 2)

Popis generovaných grafů, jejich interpretace a příkazy pro spuštění.

---

## Generování grafů

Z adresáře `app/lstm/`:

```bash
# Aktuální model (lstm_latest.keras)
python3 visualize_model.py

# Konkrétní model s vlastním labelem
python3 visualize_model.py --model models/lstm_standard_20260506.keras --label v2

# Porovnání dvou modelů
python3 visualize_model.py --label v2 --compare models/lstm_standard_20260329.keras --label_b v1
```

**Výstup:** `app/lstm/visualizations/{label}/` — čtyři PNG soubory.

---

## Grafy

### 1. `scatter.png` — Actual vs. Predicted

Bodový graf skutečné vs. predikované hodnoty pro každý ze tří targetů. Přerušovaná čára = ideální predikce (y = x).

| Target | Co říká dobrý scatter |
|---|---|
| MQE improvement ratio | Body těsně kolem diagonály, rovnoměrně po celém rozsahu 0.2–0.9 |
| Topographic error | Body soustředěné blízko 0 — TE je přirozeně nízká u většiny konfigurací |
| Dead neuron ratio | Bimodální nebo velmi nízké hodnoty — většina map nemá mrtvé neurony |

Každý bod odpovídá jednomu testovacímu oknu (prefix délky K ze sekvence jednoho individua).
Nadpis grafu zobrazuje MAE, RMSE a Pearsonovo r.

**Aktuální model (`lstm_standard_20260506`):**

| Target | MAE | RMSE | r | Bias |
|---|---|---|---|---|
| MQE improvement ratio | 0.041 | 0.049 | −0.13 | −0.040 |
| Topographic error | 0.038 | 0.040 | −0.20 | −0.038 |
| Dead neuron ratio | 0.0035 | 0.0038 | +0.24 | +0.0035 |

**Pozorování:** Všechny tři targety vykazují systematický bias — model podceňuje MQE a TE, mírně přeceňuje Dead. Nízké Pearsonovo r naznačuje, že model zachycuje průměr populace spíše než individuální varianci. Testovací set obsahuje pouze "dobré" konfigurace (nepotrestané), takže rozsah targetů je úzký a model se může jevit jako konzervativní.

---

### 2. `residuals.png` — Distribuce chyb

Histogram residuálů `(predikce − skutečná hodnota)` pro každý target. Černá čára = 0, červená = průměr residuálu.

**Co sledovat:**
- **Centrace kolem nuly** — model není systematicky nad nebo pod (ideálně)
- **Symetrie** — asymetrický ocas = model přeceňuje/podceňuje část rozsahu
- **Šířka (std)** — měřítko průměrné chyby

**Aktuální model:**

| Target | std residuálů | Bias (průměr) |
|---|---|---|
| MQE improvement ratio | 0.027 | −0.040 |
| Topographic error | 0.013 | −0.038 |
| Dead neuron ratio | 0.0014 | +0.0035 |

MQE a TE mají záporný bias — distribuce residuálů je systematicky posunutá doleva. To odpovídá situaci, kdy model predikuje konzervativně (nižší než skutečné hodnoty). Pro opravu je potřeba buď víc dat pokrývající celý rozsah hodnot, nebo rekalibrace výstupu.

---

### 3. `prefix_accuracy.png` — Přesnost podle délky prefixu

Dva sloupcové grafy:
- **Vlevo:** MAE pro každý target při K ∈ {20, 30, 40, 50, 60, 70 %}
- **Vpravo:** MAE quality score při každém K

Ukazuje, zda model profituje z delšího prefixu — čím víc checkpointů vidí, tím přesnější predikce.

**Aktuální výsledky:**

| K | QS MAE | MQE MAE |
|---|---|---|
| 20 % (n=10) | 0.0293 | 0.0434 |
| 30 % (n=10) | 0.0266 | 0.0392 |
| 40 % (n=10) | 0.0260 | 0.0382 |
| 50 % (n=10) | 0.0263 | 0.0390 |
| 60 % (n=10) | 0.0271 | 0.0411 |

**Interpretace:**
Přesnost se mírně zlepšuje od K=20 % do K=40 % a poté se stabilizuje nebo mírně zhoršuje. Zlepšení je malé (0.003 u QS MAE) — model se naučil predikovat průměr populace, nikoliv individuální trajektorii. Jasná monotónní závislost by naznačovala, že LSTM skutečně extrahuje informaci z časové řady; slabá závislost znamená, že model spoléhá převážně na statický kontext (vlastnosti datasetu).

Po přetrénování na více datech by tato křivka měla mít výraznější klesající trend.

---

### 4. `early_stopping.png` — Analýza předčasného ukončení

Tři panely:
- **Vlevo:** Histogram quality score — predikovaný vs. skutečný, se svislou čarou prahu 0.75
- **Uprostřed:** Scatter QS skutečný vs. predikovaný. Červené body = konfigurace, které by měly být ukončeny (`QS > 0.75`), zelené = OK
- **Vpravo:** Konfuzní matice pro rozhodnutí stop/continue při prahu 0.75

Quality score:
```python
quality_score = (1 - mqe_ratio) + topographic_error + dead_neuron_ratio * 0.5
# nižší = lepší SOM; EA zastaví trénink pokud quality_score > 0.75
```

**Aktuální výsledky:**

| Metrika | Hodnota |
|---|---|
| QS MAE | 0.027 |
| True QS mean | 0.336 (std=0.030) |
| Pred QS mean | 0.340 (std=0.004) |
| True stops (QS > 0.75) | 0 / 60 |
| Pred stops (QS > 0.75) | 0 / 60 |
| Accuracy | 1.000 |

**Pozorování:**
Testovací set obsahuje výhradně "dobré" konfigurace (QS ≈ 0.33–0.37, daleko pod prahem 0.75). Model proto nikdy nespustí předčasné ukončení — trivialní přesnost 100 %. Konfuzní matice má TP=0, FP=0, FN=0, TN=60.

Pro smysluplné vyhodnocení early stopping je potřeba testovací set obsahující i "špatné" konfigurace s QS > 0.75. Ty vznikají z penalizovaných nebo vysloveně špatných EA běhů. Doporučení: po dalším EA runu zahrnout i penalizované jedince do testovacích dat (odděleně od trénovacích).

---

## Srovnání dvou modelů

Po přetrénování na více datech:

```bash
# Záloha
cp models/lstm_latest.keras models/lstm_v1.keras
cp models/lstm_scaler_latest.pkl models/lstm_scaler_v1.pkl

# Přetrénování (přepíše lstm_latest)
python3 src/train.py

# Srovnání
python3 visualize_model.py --label v2 --compare models/lstm_v1.keras --label_b v1
```

Výstup `visualizations/v2_vs_v1/comparison.png` ukáže MAE side-by-side + tabulku delta (terminálový výstup):

```
Target                               v2          v1       Δ (v1-v2)
----------------------------------------------------------------------
MQE improvement ratio            0.0312      0.0410     ▼ 0.0098
Topographic error                0.0280      0.0380     ▼ 0.0100
Dead neuron ratio                0.0028      0.0035     ▼ 0.0007
Quality score (composite)        0.0210      0.0273     ▼ 0.0063
```

▼ = zlepšení, ▲ = zhoršení oproti předchozí verzi.

---

## Hlavní rozdíly oproti MLP vizualizaci

| Aspekt | MLP | LSTM |
|---|---|---|
| 4. graf | Feature importance (permutační) | Prefix accuracy by K% |
| Zvláštní plot | — | Early stopping konfuzní matice |
| Testovací set | ~500 vzorků z CSV | 60 oken (10 individ × 6 K-frakcí) |
| Inference | Batch (jeden forward pass) | Sample-by-sample (různé délky sekvencí) |

Feature importance nemá pro LSTM smysl stejnou metodou — permutace jednoho sloupce na všech timestepech ruší časové vzory, nikoliv jen informaci konkrétního příznaku. Místo toho prefix-accuracy plot ukazuje, kolik přidané hodnoty přináší delší kontext.
