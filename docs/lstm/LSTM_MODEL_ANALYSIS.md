# LSTM Model Analysis — Vizualizace a hodnocení

**Verze**: 1.0  
**Datum**: 2026-05-16

---

## Trénovací korpus (oba modely)

| Parametr | Hodnota |
|---|---|
| Datasety | 4 (BreastCancer, IndianLiver, LungCancer, Pima) |
| Seedů na dataset | 5 |
| Generací na seed | 5 |
| Populace | 30 individuí |
| Celkem evaluací | 4 × 5 × 5 × 30 = **3 000** |
| Feasible individua | ~2 200 (odhadem ~73 % bez penalizace) |

Jde o první fázi sběru dat — EA bez zapnutých NN modelů, plné 300-checkpoint trajektorie.

---

## Phase 2 — LSTM Early Stopping Predictor

**Účel**: Z prvních K % trénovací sekvence predikovat finální kvalitu SOM a rozhodnout, zda běh zastavit.  
**Výstup**: `raw_mqe_improvement_ratio`, `raw_topographic_error`, `dead_neuron_ratio`  
**Model**: `app/lstm/models/lstm_latest.keras`

---

### scatter.png + residuals.png — Predikce jednotlivých metrik

| Metrika | r | MAE | Hodnocení |
|---|---|---|---|
| MQE improvement ratio | 0.970 | nízká | ✅ Výborný — LSTM spolehlivě předpovídá trend kvantizační chyby z prefixu |
| Dead neuron ratio | 0.786 | nízká | ✅ Dobrý — predikce neaktivních neuronů dostatečně přesná pro early stopping |
| Topographic error | 0.499 | vyšší | ⚠️ Slabé místo — bimodální residua, nízká korelace |

**MQE improvement ratio** — residua mají mírný systematický posun (`mean = −0.011`): model hodnotu mírně podhodnocuje. Pro early stopping je to bezpečný konzervativní přístup — model raději nechá běh doběhnout, než aby ho předčasně zastavil.

**Topographic error** — bimodalita residuí (shluky kolem −0.05 a +0.03) není náhodný šum. Příčina je strukturální: malé a velké mapy mají přirozeně jiné režimy TE, a model z prefixu neví, do jakého režimu konfigurace patří, pokud `map_size` není explicitně v sekvenci. TE navíc klesá skokovitě (diskrétní přepínání topologie), zatímco MQE klesá spojitě — LSTM sekvence funguje pro TE hůř ze strukturálních důvodů.

---

### prefix_accuracy.png — Vliv délky prefixu

MAE quality score klesá lineárně s délkou prefixu:

| Prefix | Quality MAE |
|---|---|
| 20 % | 0.0384 |
| 30 % | 0.0361 |
| 50 % | 0.0318 |
| 70 % | 0.0294 |

Trend je správný — čím déle SOM běží, tím přesněji LSTM predikuje finální stav. Per-target analýza potvrzuje: MQE a Dead ratio mají stabilně nízkou MAE již od 20 %, TE klesá nejpomaleji. Metodicky solidní výsledek pro daný rozsah trénovacích dat.

---

### early_stopping.png — Anomálie v testovací sadě

**Pozorování**: Model dosáhl `Acc=1.0, Prec=1.0, Rec=1.0` na confusion matrix. To je podezřelé.

**Příčina**: Testovací sada neobsahuje žádného jedince s `quality_score < 0.75` — všechna testovací individua padla do třídy "True STOP". Model predikoval STOP pro vše, a protože to byla pravda, dosáhl 100 % přesnosti bez jakékoli reálné schopnosti rozlišovat.

**Proč k tomu došlo**: Threshold 0.75 je špatně kalibrovaný vůči skutečné distribuci quality score. Pokud má velká část EA individuí QS > 0.75 (což je běžné — EA generuje hodně průměrných konfigurací), pak threshold 0.75 není informativní dělicí čára. Problém není v nedostatku dat z "neúspěšných běhů" — ta existují. Problém je nerovnováha tříd při splitu (uid-level split mohl náhodně soustředit špatné konfigurace do test setu).

**Řešení**:
1. Kalibrovat threshold jako percentil na trénovací sadě (např. medián nebo 60. percentil quality score), ne jako pevnou hodnotu 0.75.
2. Ověřit distribuci tříd v train/val/test splitech — přidat stratifikaci při splitu (aktuálně split je čistě na uid úrovni bez kontroly distribuce tříd).

---

### Souhrn Phase 2

| Kategorie | Stav |
|---|---|
| MQE predikce | ✅ Použitelná pro early stopping |
| Dead ratio predikce | ✅ Použitelná |
| Topographic error predikce | ⚠️ Slabá, strukturální problém |
| Threshold kalibrace | ❌ 0.75 není kalibrováno na skutečnou distribuci |
| Confusion matrix validace | ❌ Test set neobsahuje obě třídy — klasifikační výsledky nelze interpretovat |

**Závěr**: Model je pro early stopping použitelný s výhradami. MQE a dead ratio jsou predikované dobře. Confusion matrix vyžaduje rekalibraci thresholdu a opravu splitu. TE zůstane nejslabší metrikou — zvážit snížení její váhy v quality score pro rané prefixy (20–30 %), kde je TE predikce nejméně spolehlivá.

---

## Phase 3 — LSTM Dynamic Schedule Controller

**Účel**: Per-checkpoint predikce `(lr_factor, radius_factor)` — multiplikativní korekce learning rate a radiusu v průběhu SOM tréninku.  
**Vstup sekvence**: `(progress, mqe_rel, topo_error, dead_ratio, lr_rel, radius_rel)`  
**Výstup**: `(lr_factor, radius_factor)` ∈ [0.5, 1.5] per checkpoint  
**Model**: `app/lstm/models/lstm_controller_latest.keras`

Trénovací data: perturbace Pareto individuí z Phase 1 EA dat (`generate_phase3_data.py`), 5 individuí × 9 variant (1 baseline + 8 perturbací) na seed, ~45 trajektorií na dataset.

---

### scatter_p3.png — Kritická anomálie: kolaps na 1.0

**Pozorování**: Predikované hodnoty se shlukují na diskrétních hodnotách 0.5 a 1.0 místo plynulé regrese přes celý rozsah [0.75, 1.25]. Model se chová jako diskrétní přepínač, ne jako spojitý kontroler.

**Příčiny (dvě, kumulativní)**:

**1) Fyzikálně neplatná trénovací data.** Perturbační funkce v `generate_phase3_data.py` vracely faktory bez znalosti aktuálních hodnot LR a radiusu — aplikovaly ±25 % na cumulative faktor bez horní meze. Výsledek: radius přes 80 při mapě 20×20, LR rostoucí namísto klesání. Model trénoval na nesmyslných trajektoriích.

**2) Distribuce targetů — 60 % hodnot = 1.0.** `PERTURB_PROB = 0.4` znamená, že 60 % timestepů nemá perturbaci (`lr_factor = 1.0`, `radius_factor = 1.0`). Model se naučí predikovat 1.0 pro vše a dostane nízkou loss. Advantage-weighted loss nezachraňuje, protože váhy jsou per-trajektorie, ne per-timestep.

Zdánlivě vysoká korelace `r ≈ 0.90` je artefakt z těch 60 % shod na hodnotě 1.0.

**Oprava (implementována)**:
- `generate_phase3_data.py`: nová funkce `make_constrained_perturb_fn` čte aktuální hodnoty z checkpointu a clipuje výsledek na fyzikální meze: `lr ∈ [1e-4, start_lr]`, `radius ∈ [1.0, max(map_m, map_n)]`.
- Worker předává `max_radius = max(map_m, map_n)` a `max_lr = start_learning_rate` z `row_dict`.
- Při příštím `prepare_phase3_dataset.py`: filtrovat timestepy kde `abs(factor − 1.0) < ε` z trénovacích dat, spočítat r a MAE jen na přeturbovaných timestepech.

---

### residuals_p3.png — Padding zeros jako falešné targety

**Pozorování**: Residua mají extrémní špičku na 0.0 (správné predikce) ale také izolovaný shluk přesně na +0.5.

**Příčina**: `pad_ragged()` v `prepare_phase3_dataset.py` doplňuje kratší sekvence nulami — padded timestepy mají `y = [0.0, 0.0]`. Model pro ně predikuje ~0.5 (spodní limit sigmoid výstupu × 1.0 + 0.5 = 0.5). Residuum je tedy přesně `0.5 − 0.0 = +0.5`. Nejde o chybu SOM, kde by se "vypínalo učení" — je to čistě artefakt paddingu.

**Oprava**:
- Při evaluaci a při vykreslování scatter/residuals maskovat padded pozice: `mask = y_test[j].sum(axis=1) != 0.0` (nebo srovnávat délky sekvencí a ořezat).
- Přidat sequence masking do Keras modelu při tréninku (`Masking(mask_value=0.0)` vrstva), aby padded pozice nedostávaly gradient.

---

### advantage_p3.png — Q1 prázdný, uniformní MAE

**Pozorování**: Q1 (0–25 % advantage) má `n=0`. MAE je napříč Q2–Q4 téměř identická (~0.10–0.12).

**Příčina Q1=0**: Advantage je `max(0, delta_mqe) / max_delta` — záporné delta jsou ořezány na 0. Pokud velká část trajektorií má `advantage = 0.0` přesně (perturbace zhoršila výsledek), pak kvartilové hranice Q1 vychází `(adv >= 0.0) & (adv < 0.0)` = prázdná množina. Jde o bug ve výpočtu kvartilů při mnoha tie hodnotách na minimu.

**Příčina uniformní MAE**: Potvrzuje kolaps modelu na predikci 1.0. Model ignoruje advantage signál, protože jeho loss je dominovaná 60 % timestepy s target=1.0.

**Oprava Q1**:
```python
# Místo quartile hranic přes percentil → binování přes argsort
sorted_idx = np.argsort(adv_test)
bins = np.array_split(sorted_idx, 4)   # Q1, Q2, Q3, Q4 — vždy stejně velké
```

**Oprava uniformní MAE**: vyřeší se filtrací timestepů (viz scatter_p3 oprava).

---

### Souhrn Phase 3

| Kategorie | Stav | Příčina |
|---|---|---|
| Plynulá regrese lr_factor / radius_factor | ❌ Kolaps na 1.0 | 60 % targetů = 1.0, nepřeturbované timestepy dominují |
| Korelace r≈0.90 | ❌ Artefakt | Pochází z shody na 1.0, ne z reálné regrese |
| Residuální spike +0.5 | ❌ Padding artefakt | Padded pozice mají target=0.0, model predikuje 0.5 |
| Q1 prázdný | ❌ Bug | Quartile split nefunguje při mnoha tie=0 |
| Uniformní MAE přes advantage | ❌ Kolaps | Model nerozlišuje advantage úrovně |

**Závěr**: Model v tomto stavu není použitelný pro jemné dynamické řízení. Architektura ani aktivace nejsou problém. Oprava je primárně v **přípravě dat**:

1. Filtrovat pouze přeturbované timestepy pro trénink (kde faktor ≠ 1.0)
2. Maskovat padding při tréninku i evaluaci
3. Opravit quartile split pro vizualizaci
4. Zvýšit `PERTURB_PROB` při příští generaci dat

---

## Srovnání obou modelů

| Aspekt | Phase 2 (early stopping) | Phase 3 (controller) |
|---|---|---|
| Základní funkcionalita | ✅ Funguje pro MQE a dead ratio | ❌ Kolapsuje na predikci 1.0 |
| Hlavní problém | Threshold kalibrace + test set imbalance | 60 % targetů = 1.0, padding artefakty |
| Topo error | Strukturálně slabší metrika | — |
| Připravený k nasazení? | ⚠️ Podmíněně (MQE+dead fungují) | ❌ Vyžaduje opravu dat a přetrénování |
| Prioritní oprava | Rekalibrovat threshold, opravit split | Filtrovat timestepy v prepare_phase3_dataset.py |

---

## Plán oprav

### Phase 2 — krátkodobé

```bash
# 1. Spočítat distribuci quality score na trénovací sadě
# 2. Nastavit threshold jako percentil (např. 60. percentil QS)
# 3. Přidat stratifikaci tříd do split_records()
# 4. Přetrénovat nebo jen rekalibrovat threshold bez přetrénování
```

### Phase 3 — vyžaduje nová data nebo opravu přípravy

```bash
# Možnost A — filtrace existujících dat (rychlejší)
# V prepare_phase3_dataset.py přidat filtr timestepů:
#   mask = (abs(lr_f - 1.0) > 0.01) | (abs(rad_f - 1.0) > 0.01)

# Možnost B — nová data s vyšší PERTURB_PROB (doporučeno pro Phase 3 Část 2)
# V generate_phase3_data.py zvýšit PERTURB_PROB: 0.4 → 0.75

# Po opravě dat:
python app/lstm/prepare_phase3_dataset.py
cd app/lstm && python src/train_phase3.py
python visualize_model.py --phase 3
```
