# LSTM — Strategie dynamického řízení SOM tréninku

**Verze**: 2.1  
**Aktualizováno**: 2026-05-14  
**Stav**: Část 1 hotova ✅ — pipeline end-to-end funkční, model natrénován (test MAE=0.041)

---

## Kde jsme teď

LSTM je v Phase 2 — **early stopping prediktor**. Z prefixu trénovací sekvence (prvních K % checkpointů) predikuje finální kvalitu SOM a rozhoduje, zda trénink zastavit. Neovlivňuje průběh tréninku samotného.

Statické decay křivky (learning_rate, radius, batch_size) jsou v EA nastaveny fixně pro každého jedince a mění se deterministicky dle konfigurace.

---

## Proč statické křivky nahradit dynamickým řízením

Výsledky permutační důležitosti MLP (trénovaného na ~5 800 EA individuích) ukazují:

```
ds_active_dimensions  ████████████████████████  0.0093   (dataset vlastnost)
ds_numeric            ████████████              0.0049   (dataset vlastnost)
ds_n_samples          ████████████              0.0048   (dataset vlastnost)
ds_categorical        ██████████                0.0038   (dataset vlastnost)
start_radius_init_ratio  ██              0.0008
epoch_multiplier         █               0.0004
map_m / map_n            █               0.0003
start_learning_rate      █               0.0002
growth_g                 ≈ 0
num_batches              ≈ 0
batch_growth_type        ≈ 0             ← FORMA křivky nezáleží
lr_decay_type            ≈ 0             ← FORMA křivky nezáleží
radius_decay_type        ≈ 0             ← FORMA křivky nezáleží
end_learning_rate        ≈ 0
end_batch_percent        ≈ 0
```

**Závěr**: Parametry popisující *tvar* decay křivky (exponenciální / lineární / step-down, growth_g, end_lr) mají na finální kvalitu SOM zanedbatelný vliv. Záleží na *počátečních hodnotách* (start_radius, start_lr) a na tom *kolik dat SOM zpracuje* (epoch_multiplier). Konkrétní průběh poklesu — dynamicky adaptovaný — může přinést zlepšení bez ztráty flexibility.

---

## Co dynamické řízení mění

Namísto pevné křivky pro každý parametr dostane SOM v každém MQE checkpointu (cca každých 300 iterací) korekční faktory od LSTM kontroleru:

```
Checkpoint t  →  LSTM(hidden_{t-1})  →  hidden_t + (lr_factor, radius_factor, batch_factor)
                                              ↓
                              SOM: lr        *= lr_factor
                                   radius    *= radius_factor
                                   batch_pct *= batch_factor   (volitelně)
```

LSTM vidí aktuální stav konvergence (MQE trend, topographic error, dead neurons, kde na progress ose jsme) a přizpůsobí, jak agresivně snižovat parametry v dalším úseku. Statický schedule je pak jen fallback výchozí hodnota, ne hlavní mechanismus.

**Řízené parametry:**
- `learning_rate` — jak moc se mapa přizpůsobuje nové informaci
- `radius` — jak daleko sousedství zasahuje; klíčový pro topologickou organizaci

`batch_size` / `batch_pct` se **neřídí** — tvar dávkového průchodu není pro kvalitu mapy zásadní (viz permutační důležitost: `batch_growth_type ≈ 0`). Počet zpracovávaných vzorků v průběhu je okrajová úprava, ne primární řízení; lze přidat jako třetí výstup v pozdější fázi pokud se ukáže potřeba.

---

## Jak se parametry skutečně mění — multiplikativní kumulativní přístup

Controller nevydává "přidej X k radiusu". Vydává **multiplikátor** v rozsahu [0.5, 1.5]. Efektivní hodnota parametru v každé iteraci je:

```python
current_lr     = static_schedule(t) * _cum_lr_factor
current_radius = static_schedule(t) * _cum_radius_factor
```

Kumulativní faktor se aktualizuje při každém MQE checkpointu:

```python
lr_f, rad_f = dynamic_schedule_fn(cp)          # controller výstup
lr_f   = clip(lr_f,   0.5, 2.0)                # per-krok clamp
rad_f  = clip(rad_f,  0.5, 2.0)
_cum_lr_factor     = clip(_cum_lr_factor * lr_f,   0.05, 5.0)
_cum_radius_factor = clip(_cum_radius_factor * rad_f, 0.05, 5.0)
```

**Efekt na lineární decay:** pokud radius klesá lineárně z R_start na R_end a `_cum_rad_factor = 1.1`, celá křivka se posune **proporcionálně nahoru** o 10 %. Tvar (lineární sklon) zůstává zachován.

**Proč multiplikativní, ne aditivní:**
- Additivní offset `R_actual = R_static + delta` by měl stejně velký absolutní efekt na začátku tréninku (velký R) i na konci (malý R).
- Multiplikativní přístup dává efekt úměrný aktuální hodnotě — tam kde R je malé, je i zásah malý. To odpovídá fyzice SOM: u konce tréninku jsou změny radiusu citlivější.

**Kumulace drobných kroků:** na první pohled nevýznamný krok `rad_f = 0.998` se po 300 checkpointech zkumuluje na `0.998³⁰⁰ ≈ 0.55` — na konci tréninku je effective radius jen 55 % statického plánu. Akumulace je reálný mechanismus, ne okrajový efekt.

---

## Logování zásahů controlleru

V `som.py` jsou **dva typy log řádků**:

### 1. Milestone log — každých ~10 % tréninku

Loguje se vždy, i když controller vydá neutrální hodnoty (lr_f ≈ 1.0). Slouží ke sledování kumulativního driftu:

```
LSTM ctrl @ 20%: step lr_f=0.9981 rad_f=0.9988 | cum_lr=0.9412 cum_rad=0.9701
LSTM ctrl @ 30%: step lr_f=0.9979 rad_f=0.9990 | cum_lr=0.8893 cum_rad=0.9512
```

### 2. Intervention log — pouze při zásahu > 1 %

Spustí se pouze když `|lr_f − 1| > 0.01` nebo `|rad_f − 1| > 0.01`. Ukazuje **absolutní hodnoty** parametrů v momentu zásahu:

```
LSTM ctrl INTERVENTION @ 43.2%: lr_f=1.0523 (Δ+0.0523) rad_f=0.9312 (Δ-0.0688) | effective lr=0.00312 R=3.1840 | cum_lr=1.0180 cum_rad=0.8943
```

Pole `effective lr` a `R` jsou hodnoty, které SOM **reálně dostane** — tedy `static_schedule(t) × cum_factor` po aplikaci zásahu.

| Pole | Význam |
|---|---|
| `Δ+` | controller zvyšuje (zpomaluje pokles parametru) |
| `Δ-` | controller snižuje (urychluje pokles parametru) |
| `effective lr` / `R` | výsledná hodnota po kumulativním faktoru |
| `cum_lr` / `cum_rad` | celkový kumulovaný efekt od začátku běhu |

**Aktuální model (Část 1, 24 trajektorií)** produkuje odchylky ~0.3 % → intervention log se téměř nespustí. To je správné — model je konzervativní. Po Části 2 (víc dat) by intervention log měl ukazovat smysluplné zásahy v blízkosti elbow bodů MQE křivky.

---

## Datová mezera Phase 3

Všechna existující data (5 853 individuí) tréninky s **fixními** decay scheduly. LSTM kontroler nemůže být natrénován z dat, kde nikdy nedošlo ke změně parametrů uprostřed tréninku — neexistují příklady "v bodě X jsem zvýšil radius, výsledek byl..."

### Řešení: stochastická perturbace při nových EA bězích

Při dalších EA kampaních přidat do SOM tréninku náhodné perturbace parametrů v MQE checkpointech. Tím vzniknou různorodé trajektorie, na kterých lze trénovat LSTM kontroler.

**Mechanismus**:
```python
# V som.py, při každém MQE checkpointu (pro označená individua):
if perturbation_enabled and rng.random() < perturbation_prob:
    lr_factor     = rng.uniform(1 - lr_perturb,     1 + lr_perturb)
    radius_factor = rng.uniform(1 - radius_perturb, 1 + radius_perturb)
    current_lr    *= lr_factor
    current_radius *= radius_factor
    # (batch_factor volitelně)
```

**Co se uloží do checkpointu**: `lr_factor` a `radius_factor` aplikované v daném bodě — jako trénovací label pro LSTM kontroler.

### [ROZHODNUTÍ 1] Jaké procento individuí perturbovar?

| Možnost | Výhoda | Nevýhoda |
|---|---|---|
| **30 %** perturb / 70 % čisté | Zachová čisté referenční data pro Phase 2 prediktor | Méně perturbovaných trajektorií |
| **50 %** perturb / 50 % čisté | Vyvážený dataset | Méně čistých dat pro prediktor |
| **70 %** perturb / 30 % čisté | Maximální variabilita pro kontroler | Méně čistých dat |

**Doporučení**: 50 % — Phase 2 prediktor má 5 853 čistých trajektorií, čisté data pro Phase 2 trénink jsou již dostatečná. Nová kampaň primárně slouží Phase 3.

### [ROZHODNUTÍ 2] Jak velká perturbace?

| Parametr | Navrhovaný rozsah | Odůvodnění |
|---|---|---|
| `lr_factor` | ±25 % (0.75–1.25) | LR je citlivý — velké skoky destabilizují |
| `radius_factor` | ±25 % (0.75–1.25) | Stejná logika jako LR |

`batch_factor` se neperturbuje — batch není řízen (viz výše). Pokud se v budoucnu přidá jako třetí výstup, rozsah ±15 % by byl dostatečný.

**Poznámka**: Perturbace se aplikují multiplikativně při každém checkpointu, takže kumulativní drift je větší než jednotlivý krok. Rozsah ±25 % na krok je konzervativní.

---

## Zmenšení search space po implementaci Phase 3

Jakmile LSTM kontroler řídí průběh křivek, lze ze search space EA odstranit parametry pro jejich tvar:

| Odstraňuje se | Proč |
|---|---|
| `lr_decay_type` (4 typy) | LSTM řídí průběh LR dynamicky — tvar křivky nehraje roli |
| `radius_decay_type` (4 typy) | LSTM řídí průběh R dynamicky |
| `growth_g` | součást decay křivky, odpadá s decay typem |
| `end_learning_rate` | konečná hodnota — LSTM adaptuje průběh, pevný cíl je zbytečný |

`batch_growth_type`, `end_batch_percent`, `num_batches` zůstávají (batch se neřídí LSTMem — pokud ho chceme v EA, přebírá statický průběh jako dosud).

**Zůstává v EA search space** (~8 parametrů místo ~15):
- `map_m`, `map_n` (velikost mapy)
- `epoch_multiplier` (celková délka tréninku)
- `start_learning_rate` (výchozí LR — LSTM adaptuje od tohoto bodu)
- `start_radius_init_ratio` (výchozí radius)
- `start_batch_percent`, `end_batch_percent`, `batch_growth_type` (batch stále statický)

Tato redukce zkrátí evaluaci a umožní EA prohledat efektivnější prostor.

---

## Konkrétní plán sběru dat Phase 3 — bez nového EA

Místo spuštění nové EA kampaně (stovky hodin) stačí vzít hotové Pareto konfigurace a znovu spustit jen **SOM training** s řízenou perturbací. SOM trénink jednoho individua trvá desítky sekund.

### Rozsah

| | Počet |
|---|---|
| Datasety | 4 (BreastCancer, IndianLiver, LungCancer, Pima) |
| Pareto individuí na dataset | 3 |
| Variant na individuum | 5 (viz níže) |
| **Celkem SOM běhů** | **~60** |
| Odhadovaný čas | 30–90 minut |

### Varianty na každou konfiguraci

| Varianta | Co je dynamické | Baseline |
|---|---|---|
| 0 — baseline | nic (deterministický schedule) | ano — referenční výsledek |
| 1 — LR | `lr_factor = U(0.75, 1.25)` při každém checkpointu | ne |
| 2 — radius | `radius_factor = U(0.75, 1.25)` | ne |
| 3 — lr + radius | obojí najednou | ne |

Oddělené varianty (1–2) izolují efekt každého parametru. Varianta 3 testuje interakci. Porovnání všech s baselinovou kvalitou dá přímý signál, zda LR nebo radius má větší dopad na finální kvalitu SOM.

Batch varianta se nesbírá — batch není řízen.

### Co se ukládá

Do každého checkpointu se přidají aplikované faktory:

```json
{
  "iteration": 3200,
  "progress": 0.23,
  "mqe": 0.412,
  "topographic_error": 0.071,
  "dead_neuron_ratio": 0.18,
  "learning_rate": 0.312,
  "radius": 5.1,
  "lr_factor": 1.18,
  "radius_factor": 0.89,
  "batch_factor": 1.0
}
```

Trénovací příklad pro Phase 3 LSTM = (stav konvergence v bodě t, aplikovaný faktor, Δ finální kvality vs baseline).

### Fundamentální datová mezera

Všechny existující checkpointy (5 853 individuí) mají `lr_factor=1.0, radius_factor=1.0`
vždy — SOM nikdy neměl parametry upravované uprostřed tréninku. Kontroler se nemá
z čeho naučit "při tomto stavu konvergence jsem zpomalil radius, výsledek byl lepší".
**Bez nových dat s perturbacemi Phase 3 nelze natrénovat.**

---

## Implementační plán Phase 3

### Část 1 — Smoke test ✅ HOTOVO

Cíl byl ověřit celý pipeline end-to-end. Data jsou správná, model se natrénuje,
predikuje nenáhodné hodnoty. **Splněno 2026-05-13.**

**Krok 1 — `som.py`: `dynamic_schedule_fn` callback ✅**

Implementováno: `train()` přijímá `dynamic_schedule_fn=None`. Při každém MQE checkpointu:
- Volá callback s aktuálním checkpoint dictem
- Faktory se ořezávají na `[0.5, 2.0]` per krok a `[0.05, 5.0]` kumulativně
- Checkpoint ukládá `lr_factor` a `radius_factor` (default 1.0 pokud bez perturbace)

**Krok 2 — `app/lstm/generate_phase3_data.py` ✅**

5 Pareto individuí × (1 baseline + 8 variant) = **45 trajektorií**, **14 013 checkpointů**

| Varianta (střídají se 0–7 mod 3) | Popis |
|---|---|
| lr+radius | `lr_factor, radius_factor = U(0.75, 1.25)`, prob=0.4 |
| lr_only | pouze `lr_factor` |
| radius_only | pouze `radius_factor` |

**Výsledky dat:**
- 26/40 perturbovaných běhů bylo **lepších** než baseline (65 %)
- Δ MQE: −0.12 až +0.23 (baseline MQE ≈ 1.63–1.65)
- Perturbační parametry: `LR_PERTURB=0.25`, `RADIUS_PERTURB=0.25`, `PERTURB_PROB=0.4`

**Krok 3 — `app/lstm/src/model_controller.py` ✅**

```
Vstup (per checkpoint):  (progress, mqe_rel, te, dead, lr_rel, radius_rel)  # 6 features
LSTM(64) → Dropout(0.2) → LSTM(32) → Dense(32, relu)
+ context: Dense(16, relu), tiled across time
concat → Dense(16, relu) → Dense(2, sigmoid) → výstup × 1.0 + 0.5 → [0.5, 1.5]
```

Dvě varianty: `create_controller()` (stateful, batch_shape=(1,1,6), inference) a
`create_controller_trainable()` (non-stateful, trénink batch processing).

**Krok 4 — `app/lstm/prepare_phase3_dataset.py` + `app/lstm/src/train_phase3.py` ✅**

Dataset (40 perturbovaných trajektorií, split 24/8/8 na uid úrovni):
- `X: (N, T, 6)`, `y: (N, T, 2)`, `ctx: (N, 4)`, `adv: (N,)` — normalized advantage
- advantage = `max(0, delta_mqe) / max_delta` → [0, 1]

Trénink: advantage-weighted MSE — weights tiled na `(N, T)` kvůli Keras seq2seq loss.

**Výsledky tréninku:**
- 48 epoch (early stopping), val_mae = 0.044, **test_mae = 0.041**
- Model uložen: `app/lstm/models/lstm_controller_latest.keras`
- Scaler: `app/lstm/models/lstm_controller_scaler_latest.pkl`

**Ověření části 1 ✅:**
- Ztráta klesala (0.048 → 0.041 MAE)
- Predikce nejsou konstantně `(1.0, 1.0)` — model rozlišuje trajektorie
- Pipeline end-to-end funkční bez chyb

---

### Část 2 — Pořádný trénink (~200–400 SOM běhů, ~1–3 hodiny)

Rozšíření dat na všechny datasety:

| | |
|---|---|
| Datasety | 4 (BreastCancer, IndianLiver, LungCancer, Pima) |
| Pareto individuí na dataset | 5–10 |
| Variant na individuum | 10 (více perturbačních semen) |
| **Celkem SOM běhů** | **~200–400** |

Vylepšení modelu:
- Přidat **elbow features** (2. derivace MQE) jako extra dimenze sekvence — model pak explicitně vidí, v jaké fázi konvergence se SOM nachází (viz sekce Elbow body)
- Trénink s více daty umožní smysluplnější advantage-weighted loss
- Hyperparameter tuning (LSTM units, dropout, learning rate)

Validace:
- Srovnávací EA běh: statický schedule vs. LSTM kontroler → měřitelné Δ na Pareto frontě
- Per-dataset holdout: model natrénovaný na 3 datasetech testovaný na 4.

---

### Pořadí kroků

```
Krok 1  ✅ dynamic_schedule_fn v som.py
Krok 2  ✅ generate_phase3_data.py — Část 1 (45 trajektorií, LungCancer)
Krok 3  ✅ model_controller.py + prepare_phase3_dataset.py + train_phase3.py
Krok 4  ✅ Ověření Části 1: ztráta klesá, test MAE=0.041
Krok 5  ❌ Rozšíření generate_phase3_data.py na všechny datasety (Část 2)
Krok 6  ❌ Trénink na celém datasetu + elbow features
Krok 7  ❌ Srovnávací validační EA běh: statický schedule vs. LSTM kontroler
```

EA integrace je hotová — `use_lstm_controller: true` v config-ea.json aktivuje kontroler.
Krok 7 ověří, zda model v praxi zlepšuje Pareto frontu.

---

## Závislosti a pořadí kroků (přehled)

```
Phase 2 ✅ hotovo               Phase 3 (příští)
──────────────────────────────  ──────────────────────────────────────────
prepare_dataset.py              dynamic_schedule_fn v som.py
model.py (hybrid LSTM+context)  generate_phase3_data.py (60 SOM běhů)
train.py                        Trénink Phase 3 LSTM kontroleru
lstm_latest.keras natrénován    Srovnávací validační běh
FR-LSTM-2.6: ověření v EA ❌
```

---

## Pozorování z reálných SOM běhů (LungCancer, 18×18, Pareto config)

Z porovnávacích běhů v `data/results/analysis/som-comparison/`:

- **U-matrix bez výrazných hranic** — očekávané pro datasety s převahou kategorických
  features (LungCancer: 15 kat. / 2 num.). Gradients jsou plynulé. Component diagramy
  jsou relevantnější vizualizací než U-matrix pro tento typ dat.

- **dead_neuron_ratio = 0, pokrytí celé mapy** — mapa 18×18 je přiměřená pro 3000 vzorků
  (coverage_ratio ≈ 9.3). Vesantovo pravidlo dobře kalibrovalo search space.

---

## Elbow body v MQE křivce jako fázové přechody

### Pozorování

MQE křivka reálného SOM tréninku (300 MQE checkpointů) obsahuje typicky **3 elbow body**
odpovídající přirozeným fázovým přechodům:

```
MQE
│
│\         ← fáze 1: globální organizace (velký LR, velký radius)
│ \
│  ·─·     ← elbow 1: topologie stabilizována, začíná lokální doladění
│     \
│      ·─  ← elbow 2: lokální struktura ustálena, začíná konvergenční plato
│        ──── ← elbow 3: marginální zlepšení, trénink se vyčerpává
└──────────────── progress
    20%  50%  80%
```

### Implikace pro Phase 2 (early stopping)

Aktuální přístup: pevné K% okno (20–70 % délky sekvence).
Lepší přístup: predikovat po dosažení elbow 2 — v tomto bodě je lokální struktura
ustálena a zbývající trénink přináší jen marginální zlepšení. Závisí na datasetu
a konfiguraci, ale empiricky odpovídá ~50–65 % progress.

Přínos: menší variance v délce prefixu, modelu se vždy předá sémanticky ekvivalentní
bod ("po druhé fázi"), ne jen procento délky.

### Implikace pro Phase 3 (dynamický controller)

Elbow body jsou přirozené momenty pro intervenci:
- **Elbow 1** (~20–30 %): globální fáze končí → zpomalit pokles radius (nechat
  lokální organizaci proběhnout déle)
- **Elbow 2** (~50–65 %): lokální fáze končí → agresivněji snížit LR, přepnout
  batch do fine-tuning velikosti
- **Elbow 3** (~80–90 %): konvergence → případné early stop, dál trénovat nemá smysl

### Technická implementace: druhá derivace MQE

Elbow = lokální maximum druhé derivace MQE sekvence (největší změna sklonu).

```python
def compute_mqe_derivatives(mqe_values: list) -> tuple[list, list]:
    """Vrátí první a druhou derivaci MQE sekvence (numericky)."""
    mqe = np.array(mqe_values)
    d1 = np.gradient(mqe)           # rychlost poklesu
    d2 = np.gradient(d1)            # zrychlení poklesu (elbow = max |d2|)
    return d1.tolist(), d2.tolist()
```

Detekce elbow bodů:
```python
peaks, _ = scipy.signal.find_peaks(np.abs(d2), prominence=threshold)
# peaks[0], peaks[1], peaks[2] → tři hlavní přechody
```

### Použití jako feature pro LSTM

**Možnost A — přidané scalar features do context vstupu:**
```
context vstup (aktuálně 4 dimenze) rozšířit o:
  d2_mqe_current    — aktuální druhá derivace (jsme na elbow?)
  elbow1_passed     — 0/1 flag, zda byl detekován první přechod
  elbow2_passed     — 0/1 flag
  progress_since_elbow2  — jak daleko jsme od druhého elbow
```

**Možnost B — přidat d1 a d2 jako 2 nové dimenze sekvence:**
```
sekvence vstup (aktuálně 6 dimenzí) rozšířit na 8:
  + d1_mqe (normalizovaná první derivace)
  + d2_mqe (normalizovaná druhá derivace)
```

Možnost B je přirozenější — LSTM pak sám naučí korelaci mezi tvarem křivky
a vhodnou intervencí, aniž bychom explicitně definovali prahy pro elbow detekci.

### Priorita implementace

Elbow features nejsou podmínkou pro základní Phase 3 — lze začít bez nich
a přidat jako vylepšení po validaci základní architektury kontroleru.
Přidáno jako otázka č. 6 v souhrnu.

---

## Souhrn otevřených otázek

| # | Otázka | Rozhodnutí | Stav |
|---|---|---|---|
| 1 | Rozsah perturbace LR/radius | ±25 % (`LR_PERTURB=0.25`, `RADIUS_PERTURB=0.25`) | ✅ implementováno |
| 2 | Perturbace při každém checkpointu, nebo s pravděpodobností p? | `PERTURB_PROB=0.4` — čistší trajektorie | ✅ implementováno |
| 3 | Zahrnout batch_factor do řízení? | Ne — controller řídí pouze lr+radius; batch zůstává statický; možno přidat jako třetí výstup v budoucnu pokud se ukáže potřeba | ✅ rozhodnuto |
| 4 | Kolik Pareto individuí na dataset (Část 2) | 5–10, cíl ~200–400 SOM běhů | ❓ čeká na Část 2 |
| 5 | Přidat elbow features (d1, d2 MQE)? | Část 2 — nezatěžovat smoke test | ❓ čeká na Část 2 |
| 6 | Srovnávací EA validace (`use_lstm_controller: true`) | Kdy spustit? | ❓ čeká na víc dat |
