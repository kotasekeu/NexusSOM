# LSTM — Strategie dynamického řízení SOM tréninku

**Verze**: 1.1  
**Aktualizováno**: 2026-05-06  
**Stav**: návrh — otevřené otázky k rozhodnutí označeny `[ROZHODNUTÍ]`

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

**Primárně řízené parametry** (dle dynamiky SOM):
- `learning_rate` — ovlivňuje jak moc se mapa přizpůsobuje nové informaci
- `radius` — ovlivňuje jak daleko sousedství zasahuje, klíčové pro topologickou organizaci
- `batch_size` / `batch_pct` — ovlivňuje jak hluboko SOM vidí do dat v jednom průchodu

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
| `batch_factor` | ±15 % (0.85–1.15) | Batch méně citlivý, ale zachovat opatrnost |

**Poznámka**: Perturbace se aplikují multiplikativně při každém checkpointu, takže kumulativní drift je větší než jednotlivý krok. Rozsah ±25 % na krok je konzervativní.

---

## Zmenšení search space po implementaci Phase 3

Jakmile LSTM kontroler řídí průběh křivek, lze ze search space EA odstranit parametry pro jejich tvar:

| Odstraňuje se | Proč | Zůstává / co LSTM nahrazuje |
|---|---|---|
| `lr_decay_type` (4 typy) | LSTM řídí průběh dynamicky | výchozí start_lr |
| `radius_decay_type` (4 typy) | LSTM řídí průběh dynamicky | výchozí start_radius |
| `batch_growth_type` (2 typy) | LSTM řídí batch_pct | výchozí start/end_batch_pct |
| `growth_g` | součást decay křivky | — |
| `end_learning_rate` | konečná hodnota fixní (LSTM ji adaptuje) | — |
| `end_batch_percent` | konečná hodnota fixní | — |
| `num_batches` | méně důležité | — |

**Zůstává v EA search space** (~7 parametrů místo ~15):
- `map_m`, `map_n` (velikost mapy)
- `epoch_multiplier` (celková délka tréninku)
- `start_learning_rate` (výchozí LR)
- `start_radius_init_ratio` (výchozí radius)
- `start_batch_percent` (výchozí batch)

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

### 5 variant na každou konfiguraci

| Varianta | Co je dynamické | Baseline |
|---|---|---|
| 0 — baseline | nic (deterministický schedule) | ano — referenční výsledek |
| 1 — LR | `lr_factor = U(0.75, 1.25)` při každém checkpointu | ne |
| 2 — radius | `radius_factor = U(0.75, 1.25)` | ne |
| 3 — batch | `batch_factor = U(0.85, 1.15)` | ne |
| 4 — vše | lr + radius + batch najednou | ne |

Oddělené varianty (1–3) izolují efekt každého parametru. Varianta 4 testuje interakci. Porovnání všech 5 s baselinovou kvalitou dá přímý signál, který parametr dynamické řízení nejvíce pomáhá.

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

### Potřebné implementační kroky

**1. `som.py` — přidat `dynamic_schedule_fn` callback**

```python
def train(self, data, ignore_mask=None, working_dir='.',
          lstm_early_stop_fn=None,      # Phase 2 — zůstává
          dynamic_schedule_fn=None):    # Phase 3 — nový

# V tréninkovém loopu při každém MQE checkpointu:
if dynamic_schedule_fn is not None:
    lr_f, rad_f, batch_f = dynamic_schedule_fn(checkpoint)
    current_lr     *= lr_f
    current_radius *= rad_f
    current_batch  *= batch_f
    checkpoint['lr_factor']     = lr_f
    checkpoint['radius_factor'] = rad_f
    checkpoint['batch_factor']  = batch_f
```

**2. `app/lstm/generate_phase3_data.py`** — nový skript

```
Pro každý dataset:
  Načti pareto_front.csv → vezmi top 3 individua
  Pro každé individuum:
    Pro každou variantu (0–4):
      Načti konfiguraci individua z results.csv
      Spusť som.train() s příslušnou dynamic_schedule_fn
      Ulož training_checkpoints.json s lr_factor/radius_factor/batch_factor
      Ulož výsledek (finální mqe, te, dead) pro porovnání s baselineem
```

### Pořadí kroků po vygenerování dat

```
Krok 1 (nyní)      Ověřit Phase 2 v EA běhu (FR-LSTM-2.6)
Krok 2             Implementovat dynamic_schedule_fn v som.py
Krok 3             Spustit generate_phase3_data.py (~60 SOM běhů, 1–2 h)
Krok 4             Natrénovat Phase 3 LSTM kontroler na vygenerovaných datech
Krok 5 (validace)  Srovnávací EA běh:
                     • čistý SOM (statický schedule)
                     • SOM + LSTM early stopping (Phase 2)
                     • SOM + LSTM dynamický kontroler (Phase 3)
```

Kroky 1–3 jsou vzájemně nezávislé a mohou probíhat paralelně.

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

## Souhrn otevřených otázek

| # | Otázka | Navrhovaná odpověď | Stav |
|---|---|---|---|
| 1 | Rozsah perturbace LR/radius | ±25 % (0.75–1.25) | ❓ k rozhodnutí |
| 2 | Rozsah perturbace batch | ±15 % (0.85–1.15) | ❓ k rozhodnutí |
| 3 | Kolik Pareto individuí na dataset | 3 | ❓ k rozhodnutí |
| 4 | Perturbace při každém checkpointu, nebo jen náhodně s pravděpodobností p? | každý checkpoint | ❓ k rozhodnutí |
| 5 | Spustit generate_phase3_data.py před nebo po ověření Phase 2? | po ověření Phase 2 | doporučeno |
