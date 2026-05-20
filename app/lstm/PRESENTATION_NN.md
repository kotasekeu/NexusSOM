# NexusSOM — Neuronové sítě: MLP a LSTM
## Formát: [Slide] / [Komentář řečníka]

---

## Slide 1 — Název

**Slide:**
> Neuronové sítě v NexusSOM
> MLP predikce kvality · LSTM early stopping · LSTM dynamický kontroler

**Komentář:**
NexusSOM obsahuje tři neuronové modely, každý na jiné vrstvě evolučního procesu. MLP funguje jako statický filtr ještě před spuštěním SOM. LSTM Phase 2 sleduje běžící trénink a rozhoduje o předčasném zastavení. LSTM Phase 3 aktivně ovlivňuje průběh tréninku v každém checkpointu. Výsledkem je EA, která dokáže prohledat prostor hyperparametrů efektivněji a přitom využívat historii předchozích běhů.

---

## Slide 2 — Proč neuronové sítě?

**Slide:**
- Spuštění SOM tréninku = 300–500 epoch, stovky individuí na generaci
- EA generuje tisíce konfigurací — ne všechny stojí za spuštění
- **MLP**: predikuje finální kvalitu bez spuštění SOM → pre-screen filtr
- **LSTM Phase 2**: zastaví běh, pokud prefix trajektorie ukazuje slabé výsledky
- **LSTM Phase 3**: upravuje learning rate a radius v každém checkpointu

**Komentář:**
Základní problém je výpočetní efektivita. Každá EA generace čítá desítky individuí a každý z nich vyžaduje plný SOM trénink. Neuronové sítě představují tři různé intervence: před spuštěním, v průběhu a jako adaptivní řízení. Každý model byl trénován na datech z předchozích EA běhů — systém se s každou iterací zlepšuje.

---

## Slide 3 — Přehled vrstev

**Slide:**

```
EA generuje jedince (hyperparametry)
        ↓
[MLP — The Prophet]          ← statický pre-screen
  25 vstupů → 3 výstupy       konfigurace bez spuštění SOM
  MAE = 0.030                 filtruje "jistě špatné" konfigurace
        ↓ (konfigurace prošla filtrem)
[SOM trénink — checkpoint loop]
        ↓
[LSTM Phase 2 — Early Stopping]  ← za běhu
  sekvence[0..K] → 3 výstupy     predikuje finální stav z prefixu
  MAE ≈ 0.023                     zastaví trénink pokud quality < threshold
        ↓ (trénink pokračuje)
[LSTM Phase 3 — Controller]      ← per-checkpoint
  checkpoint[t] → lr_factor, radius_factor
  MAE = 0.041                     dynamicky upravuje decay curves
        ↓
Finální SOM + metriky
```

**Komentář:**
Tři vrstvy jsou vzájemně nezávislé — každou lze zapnout/vypnout v konfiguraci. V produkci lze kombinovat MLP filtr s Phase 2, ale Phase 3 je zatím ve vývoji a vyžaduje opravu dat před nasazením.

---

## Slide 4 — MLP: The Prophet

**Slide:**
```
Vstup (25): one-hot(lr_decay, radius_decay, batch_growth) +
            log(start_lr), log(end_lr), log(start_radius) +
            epoch_multiplier, map_m, map_n +
            ds_n_samples, ds_n_dimensions, ds_n_numeric, ...

Dense(256) → BN → Dropout(0.3)
Dense(128) → BN → Dropout(0.3)
Dense(64)  → BN → Dropout(0.2)
Dense(32)  →      Dropout(0.2)
Dense(3, linear)

Výstup: mqe_improvement_ratio, topographic_error, dead_neuron_ratio
```

| | Hodnota |
|---|---|
| Parametry | ~100 000 |
| Trénovací data | 2 277 individuí (4 datasety, 5 seedů) |
| Test MAE | **0.030** |
| Stav | ✅ Produkčně nasazený |

**Komentář:**
MLP je feedforward síť s plným přístupem ke všem hyperparametrům a vlastnostem datasetu. Klíčový přínos: predikce proběhne v milisekundách, zatímco plný SOM trénink trvá sekundy až minuty. Dropout a BatchNorm zajišťují generalizaci na nové datasety.

---

## Slide 5 — MLP: Co predikuje a proč to funguje

**Slide:**

Permutační důležitost vstupních featur (z natrénovaného MLP):

```
ds_active_dimensions  ████████████████████████  0.0093   ← důležité
ds_numeric            ████████████              0.0049
ds_n_samples          ████████████              0.0048
ds_categorical        ██████████                0.0038
start_radius_init_ratio  ██              0.0008
epoch_multiplier         █               0.0004
map_m / map_n            █               0.0003
batch_growth_type        ≈ 0             ← tvar křivky nezáleží
lr_decay_type            ≈ 0
radius_decay_type        ≈ 0
```

**Závěr**: Na finální kvalitu SOM mají největší vliv vlastnosti datasetu, ne tvar decay křivek.

**Komentář:**
Toto zjištění je klíčové pro celý projekt. Pokud tvar decay křivky nezáleží, pak Phase 3 LSTM (dynamické řízení tvaru) musí fungovat jinak než predikce finálního čísla — jde o jemné online úpravy. Permutační důležitost navíc motivuje kontextový vstup (vlastnosti datasetu) ve všech třech modelech.

---

## Slide 6 — LSTM Phase 2: Early Stopping Predictor

**Slide:**

```
Vstup sekvence (batch, T, 6):
  progress, mqe_rel, topographic_error,
  dead_neuron_ratio, lr_rel, radius_rel

Vstup kontext (batch, 4):
  ds_n_samples, ds_n_active_dims, ds_n_numeric, ds_n_categorical

LSTM(64, return_sequences=True)
LSTM(32)                         ← konec sekvence
                    Dense(16, relu) ← kontext branch
          Concat
          Dense(32, relu)
          Dense(3, linear)

Výstup: mqe_improvement_ratio, topographic_error, dead_neuron_ratio
```

| | Hodnota |
|---|---|
| Trénovací okna | 18 282 (5 853 individuí × K ∈ 20–70 %) |
| Test MAE | **≈ 0.023** |
| Threshold | quality_score > 0.75 → STOP |

**Komentář:**
Hybridní architektura kombinuje LSTM pro zpracování sekvence checkpointů s paralelní Dense větví pro statický kontext datasetu. Masking ignoruje nulové padding hodnoty. Model dostane prvních K % trénovacích checkpointů a predikuje, jak SOM dopadne na konci.

---

## Slide 7 — LSTM Phase 2: Výsledky

**Slide:**

| Metrika | Korelace r | Stav |
|---|---|---|
| MQE improvement ratio | **0.970** | ✅ Výborný |
| Dead neuron ratio | **0.786** | ✅ Dobrý |
| Topographic error | **0.499** | ⚠️ Slabý |

Přesnost podle délky prefixu (MAE quality score):

| Prefix | MAE |
|---|---|
| 20 % | 0.038 |
| 30 % | 0.036 |
| 50 % | 0.032 |
| 70 % | 0.029 |

**Komentář:**
MQE predikce je spolehlivá od 20 % prefixu. Topographic error má bimodální residua — malé a velké mapy přirozeně přecházejí do jiných topologických režimů, a LSTM z prvních 20–70 % checkpointů nedokáže map_size z dynamiky sekvence odvodit. Toto je strukturální problém, ne problém s modelem.

---

## Slide 8 — LSTM Phase 3: Dynamic Schedule Controller

**Slide:**

```
V každém MQE checkpointu (každých ~300 SOM iterací):

  checkpoint[t] = (progress, mqe_rel, te, dead, lr_rel, radius_rel)
  kontext       = (n_samples, active_dims, n_numeric, n_categorical)

  LSTM(64, stateful, return_sequences=False)
  Dense(32, relu)
               Dense(16, relu)  ← kontext branch
  Concat
  Dense(16, relu)
  Dense(2, sigmoid) → ScaleSigmoid [0,1] → [0.5, 1.5]

  Výstup: lr_factor, radius_factor  ∈ [0.5, 1.5]
```

Aplikace:
```python
current_lr     = static_schedule(t) * cumulative_lr_factor
cumulative_lr_factor = clip(cumulative_lr_factor * lr_factor, 0.05, 5.0)
```

**Komentář:**
Stateful LSTM si drží hidden state mezi checkpointy — vidí celou historii tréninku, ne jen aktuální stav. Kumulativní faktor zajišťuje, že controller může trvale zpomalit nebo zrychlit decay, nejen dočasně. Rozsah [0.5, 1.5] per-krok a [0.05, 5.0] kumulativně zabraňuje divergenci.

---

## Slide 9 — LSTM Phase 3: Trénování (Advantage-Weighted Behavioral Cloning)

**Slide:**

```
Data:
  Baseline run:    SOM s původním statickým schedule → MQE_baseline
  Perturbed run:   SOM s náhodnou perturbací faktoru v náhodných checkpointech → MQE_pert

Advantage:
  Δmqe = MQE_baseline − MQE_pert    (kladné = perturbace pomohla)
  advantage = min_max_normalize((Δmqe/σ) + (Δte/σ) + (Δdead/σ))  ∈ [0, 1]

Loss (advantage-weighted MSE):
  loss[t] = advantage × perturb_mask[t] × MSE(pred[t], target[t])
  kde perturb_mask[t] = 1 pokud faktor ≠ 1.0, jinak 0
```

| | Hodnota |
|---|---|
| Trénovací trajektorie | 45 (LungCancer dataset) |
| Test MAE | **0.041** |
| Stav | ✅ Pipeline end-to-end funkční |

**Komentář:**
Behavioral cloning s advantage weighting: model se učí napodobovat akce (lr_factor, radius_factor), ale s váhami podle toho, jak moc perturbace zlepšila výsledek. Trajektorie, kde perturbace pomohla výrazně, mají váhu blízkou 1. Trajektorie, kde perturbace nezlepšila nic, mají váhu blízkou 0.

---

## Slide 10 — Problémy a Known Issues

**Slide:**

**Phase 2 — Early Stopping:**

| Problém | Příčina | Řešení |
|---|---|---|
| Topographic error r=0.499 | Strukturální: TE skáče diskrétně, LSTM sekvence to nedokáže předpovědět z prefixu | Přidat map_m/map_n do sekvence |
| Confusion matrix Acc=1.0 (podezřelé) | Testovací sada neobsahuje jedince s QS < 0.75 | Kalibrovat threshold jako percentil, stratifikovat split |

**Phase 3 — Controller:**

| Problém | Příčina | Řešení |
|---|---|---|
| Model kolapsuje na predikci 1.0 | 60 % timestepů má target=1.0 (bez perturbace) — perturb_mask zabrání backprop | Filtrovat nepřeturbované timestepy z tréninku |
| Padding artefakt (spike u 0.5) | Padded pozice (target=0.0) dostávají predikci 0.5 (střed sigmoidu) | Masking vrstvy při tréninku |
| Q1 advantage vždy prázdný | Kvartilový split selhává při mnoha tie hodnotách na 0.0 | Binování přes argsort místo percentilů |
| Malý trénovací dataset | Jen 45 trajektorií, 1 dataset | Rozšíření na 4 datasety, 200+ SOM běhů |

**Komentář:**
Phase 2 je podmíněně nasaditelná — MQE a dead ratio fungují spolehlivě, TE predikce je slabá ale neohrožuje funkci early stoppingu. Phase 3 má architektonicky správný design, ale trpí datovým problémem: trénovací dataset obsahuje příliš málo perturbovaných timestepů. Oprava nevyžaduje změnu modelu, jen způsob přípravy dat.

---

## Slide 11 — Roadmapa a Co Zbývá

**Slide:**

| Úkol | Priorita | Stav |
|---|---|---|
| Phase 2: Rekalibrovat threshold (percentil místo 0.75) | Střední | ❌ Chybí |
| Phase 2: Stratifikovaný split (třída-balancovaný) | Střední | ❌ Chybí |
| Phase 2: Ověřit early stopping v reálném EA běhu | Střední | ❌ Chybí |
| Phase 3: Filtrovat nepřeturbované timestepy v prepare_phase3_dataset.py | Vysoká | ❌ Chybí |
| Phase 3: Přidat Masking při tréninku kontroleru | Střední | ❌ Chybí |
| Phase 3: Rozšíření dat — 4 datasety, 200+ SOM běhů | Vysoká | ❌ Chybí |
| Phase 3: Srovnávací EA validace: statický schedule vs. LSTM kontroler | Vysoká | ❌ Chybí |
| MLP: Přetrénování na větším korpusu (průběžně) | Nízká | ✅ Funguje |

**Komentář:**
Nejkritičtější krok pro Phase 3 je filtrace timestepů a sběr dat z více datasetů. Bez toho model zůstane na kolapsovaném řešení. Phase 2 je v použitelném stavu pro MQE-based early stopping, threshold kalibrace je malá oprava.

---

## Slide 12 — Srovnání tří modelů

**Slide:**

| | **MLP** | **LSTM Phase 2** | **LSTM Phase 3** |
|---|---|---|---|
| Typ úlohy | Regrese (statická) | Regrese (sekvenční) | Imitační učení (sekvenční) |
| Vstup | Hyperparametry + dataset | Prefix sekvence checkpointů | Per-checkpoint stav |
| Výstup | mqe, te, dead | mqe, te, dead | lr_factor, radius_factor |
| Tréninkový signál | Supervised (EA výsledky) | Supervised (EA výsledky) | Advantage-weighted (perturbace) |
| Dataset | 2 277 individuí | 18 282 oken (5 853 ind.) | 45 trajektorií |
| Přesnost | MAE = 0.030 ✅ | MAE ≈ 0.023 ✅ | MAE = 0.041 ⚠️ |
| Nasazení | ✅ Produkce | ⚠️ Podmíněně | ❌ Vyžaduje opravu dat |

**Komentář:**
Tři modely řeší tři různé problémy. MLP je nejjednodušší a nejspolehlivější — feedforward síť na statických vstupech. LSTM Phase 2 přidává časovou dimenzi a datový kontext. LSTM Phase 3 je nejsložitější: stateful inference, advantage-weighted loss, behavioral cloning — a zároveň nejnáchylnější na kvalitu trénovacích dat.
