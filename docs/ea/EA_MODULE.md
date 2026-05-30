# EA modul — technická dokumentace

Popis implementace evoluční optimalizace hyperparametrů SOM.  
Kód: [`app/ea/ea.py`](../../app/ea/ea.py) · Konfigurace: [`data/datasets/<ds>/config-ea.json`](../../data/datasets/Iris/config-ea.json)

---

## 1. Formulace problému

**Co optimalizujeme:** hyperparametry trénování Kohonenovy SOM — velikost mapy, rychlost učení, poloměr sousedství, počet epoch, typ poklesu parametrů.

**Proč evoluční optimalizace:** prostor je smíšený (spojité floaty, celá čísla, kategorické proměnné), fitness je černá skřínka (vyžaduje spuštění SOM), neexistuje gradient.

**Proč multi-objektivní:** tři metriky kvality jsou v přirozeném trade-offu — lze je zlepšovat pouze na úkor jiných. Výsledkem je Pareto fronta kompromisních konfigurací, ne jedno "nejlepší" řešení.

---

## 2. Algoritmus — NSGA-II

Použit algoritmus **NSGA-II** (Deb et al., 2002, *IEEE Transactions on Evolutionary Computation*).

### Proč NSGA-II, ne SPEA2 nebo jiný

- SPEA2 vyžaduje předem nastavenou velikost externího archivu; pro neznámou velikost Pareto fronty je proměnlivý archiv NSGA-II výhodnější.
- NSGA-II s crowding distance capem (`max_archive_size`) zabraňuje explozi archivu ve 4D+ prostorech.
- Standardní volba pro black-box multi-objektivní optimalizaci s populací 30–200.

### Průběh jedné generace

#### Krok 0 — Počáteční populace P₀

Před první generací se náhodně vygeneruje `population_size` konfigurací (každý parametr vzorkován rovnoměrně ze svého rozsahu). Každá konfigurace se **evaluuje** — spustí se SOM trénink a změří se tři cíle. Tato populace P₀ tvoří startovní bod.

---

#### Krok a — Tournament selection (turnajový výběr)

**Co to je:** mechanismus výběru rodičů pro křížení. Nechceme vybírat rodiče zcela náhodně (mohli bychom vybrat špatná řešení) ani deterministicky nejlepší (ztratíme diverzitu).

**Jak funguje:** z aktuální populace se náhodně vytáhne `k` jedinců (u nás `tournament_k = 5`). Z těchto 5 se vybere vítěz podle pravidel:
1. Nižší rank (rank 0 = Pareto fronta) vítězí vždy.
2. Při stejném ranku vítězí ten s **větší crowding distance** (= je v méně zalidněné části objective prostoru → zachovává diverzitu).

Tento proces se zopakuje dvakrát → dva rodiče. Pak se opakuje pro dalších `population_size` párů.

**Proč k=5:** kompromis mezi selekčním tlakem (velké k preferuje elitu) a diverzitou (malé k dává šanci i horším řešením). k=2 je příliš slabý tlak, k=population_size je deterministický.

Kód: [`tournament_selection()` ea.py:309](../../app/ea/ea.py#L309)

---

#### Krok b — SBX crossover + polynomiální mutace → offspring

**Offspring** = potomci. Jsou to nové konfigurace vytvořené kombinací dvou rodičovských konfigurací. Potomci zatím **nebyli evaluováni** — neznáme jejich MQE, TE ani ρ.

##### SBX crossover (Simulated Binary Crossover)

**Co to je:** způsob křížení pro spojité parametry. Binární křížení v reálném prostoru.

**Jak funguje:** pro každý float parametr se vypočítá rozptylový koeficient β z náhodného čísla a distribučního indexu η:

```
u ~ Uniform(0, 1)
β = (2u)^(1/(η+1))         pokud u ≤ 0.5
β = (1/(2(1−u)))^(1/(η+1))  pokud u > 0.5

potomek1 = 0.5 × ((1+β)×rodič1 + (1−β)×rodič2)
potomek2 = 0.5 × ((1−β)×rodič1 + (1+β)×rodič2)
```

**Intuice:** η=20 znamená, že β je typicky blízko 1 → potomci leží blízko rodičů (exploatace). Malé η → větší rozptyl (explorace). SBX zachovává vlastnost binárního křížení: průměr potomků = průměr rodičů.

**Pro kategorické parametry** (lr_decay_type apod.): uniform swap — každý gen se s pravděpodobností 50 % přepíše hodnotou druhého rodiče.

**Pro celočíselné parametry:** SBX + zaokrouhlení na nejbližší celé číslo.

Kód: [`crossover()` ea.py:648](../../app/ea/ea.py#L648)

##### Polynomiální mutace

**Co to je:** náhodná malá změna jednoho genu po křížení. Zabraňuje uvíznutí v lokálním optimu.

**Jak funguje:** s pravděpodobností `mutation_prob = 0.1` se aplikuje na každý gen perturbace z polynomiálního rozdělení s indexem η=20. Výsledek je vždy v [min, max] daného parametru. Distribuce je symetrická kolem aktuální hodnoty — typicky malé změny, velké skoky vzácné.

**Proč 0.1:** standardní hodnota. Příliš vysoká mutace ničí dobré řešení; příliš nízká zpomalí exploraci.

---

#### Krok c — Deduplikace offspring

Každá konfigurace má deterministický **UID** = MD5 hash jejích parametrů. Pokud SBX+mutace vytvoří dvě identické konfigurace (může nastat u celočíselných parametrů v úzkém rozsahu), druhý duplikát se zahodí a generuje se nový potomek.

Cíl: zaručit, že populace obsahuje přesně `population_size` **unikátních** konfigurací. Bez deduplikace by EA zbytečně evaluovala stejnou konfiguraci dvakrát.

---

#### Krok d — Evaluace offspring O

Každý potomek se **skutečně spustí** — trénink SOM na datasetu, změření MQE ratio, TE, Spearman ρ, constraint violation. Toto je výpočetně nejnáročnější část (paralelně přes Pool).

---

#### Krok e — Combined P ∪ O

`combined = aktuální populace P + noví potomci O`

Velikost: `2 × population_size` jedinců. Smícháním rodičů a potomků zaručujeme **elitismus** — dobří rodiče mohou přežít do další generace, pokud potomci nejsou lepší.

Toto je klíčový rozdíl NSGA-II oproti jednodušším EA: místo nahrazení rodičů potomky se sloučí obě generace a pak se kompetitivně vybírá nejlepší polovina.

---

#### Krok f — Non-dominated sort → fronty F₀, F₁, …

**Co to je:** rozdělení všech `2N` řešení do vrstev (front) podle Pareto dominance.

**Fronta F₀** (rank 0) = **Pareto fronta** — řešení, která nejsou dominována žádným jiným řešením v combined. Řešení A dominuje B, pokud A není horší v žádném cíli a je lepší alespoň v jednom.

**Fronta F₁** = řešení dominovaná pouze F₀. **F₂** = dominovaná F₀ ∪ F₁. Atd.

```
combined (2N řešení)
  └── F₀: [A, B, C, D]        ← Pareto optimální — nikdo je nepřekoná
  └── F₁: [E, F, G]           ← dominuje je jen F₀
  └── F₂: [H, I]              ← dominuje je F₀ ∪ F₁
  └── ...
```

**S constrained dominance:** infeasible řešení (CV > 0) jsou vždy v horších frontách než feasible, i kdyby měla lepší raw objectives.

Složitost: O(M·N²) kde M = počet cílů, N = velikost populace. Pro N=30, M=3: 30²×3 = 2700 operací — zanedbatelné.

Kód: [`non_dominated_sort()` ea.py:211](../../app/ea/ea.py#L211)

---

#### Krok g — Sestavení nové populace P_{g+1}

Z front se plní nová populace přesně na `population_size`:

```
P_{g+1} = []
for fronta F_k v pořadí F₀, F₁, F₂, ...:
    if |P_{g+1}| + |F_k| ≤ population_size:
        přidej celou F_k          ← všechna Pareto-optimální řešení přežijí
    else:
        seřaď F_k dle crowding distance (sestupně)
        přidej prvních (population_size − |P_{g+1}|) řešení  ← vyber nejrůznorodější
        break
```

**Crowding distance** jako tie-breaker zajišťuje, že z přeplněné fronty přežijí řešení v méně hustých oblastech objective prostoru → udržuje diverzitu Pareto fronty.

---

#### Krok h — Archiv (Pareto fronta výsledků)

**Archiv ≠ populace.** Populace P_{g+1} slouží pro genetické operátory příští generace. Archiv je permanentní paměť nejlepších nalezených řešení přes všechny generace.

`ARCHIVE = F₀ z combined` → všechna non-dominovaná feasible řešení z aktuální generace. Pokud `|F₀| > max_archive_size` (default 25), ořezává se dle crowding distance (vypadnou nejhustěji seskupená řešení).

Archiv se loguje do `pareto_front.csv` na konci každé generace.

---

#### Přehled toku dat

```
P₀ (náhodná, evaluovaná)
    │
    ▼ pro každou generaci:
tournament_selection(P) ──► parent pairs
    │
    ▼
SBX crossover + polynomial mutation ──► O (potomci, neevaluovaní)
    │
    ▼
deduplikace ──► O (unikátní)
    │
    ▼
evaluace SOM pro každý ∈ O
    │
    ▼
combined = P ∪ O  (2N řešení, všechna evaluovaná)
    │
    ▼
non-dominated sort + constrained dominance ──► F₀, F₁, F₂, ...
    │
    ├──► P_{g+1} = nejlepší N řešení (elitismus + diverzita)
    │
    └──► ARCHIV = F₀, ořezán na max_archive_size
```

Implementace: [`run_evolution()` ea.py:788](../../app/ea/ea.py#L788)

---

## 3. Účelové funkce (Pareto cíle)

Všechny tři cíle: **nižší = lepší**.

| # | Název | Vzorec | Co měří |
|---|-------|--------|---------|
| 1 | `raw_mqe_ratio` | `MQE_final / MQE_init` | Relativní zlepšení kvantizační chyby vůči náhodné inicializaci; dataset-independent |
| 2 | `raw_te` | `#{vzorků kde BMU a 2.BMU nejsou sousedé} / N` | Topografická chyba — lokální zachování sousedství |
| 3 | `1 − ρ` | `1 − Spearman(dist_data, dist_grid)` | Globální topologická korelace — zachování manifoldu |

#### MQE ratio (cíl 1)
Absolutní MQE závisí na velikosti mapy a datasetu, proto se používá poměr vůči výchozí hodnotě (checkpoint[0]). Hodnota 0.85 = SOM zlepšila QE o 15 % oproti náhodné inicializaci.  
Kód: [`_mqe_obj()` ea.py:891](../../app/ea/ea.py#L891)

#### Topografická chyba — TE (cíl 2)
Procento vzorků, u nichž první a druhý BMU nejsou přímí sousedé na mřížce. Pro hex mapy se používá cube-coordinate vzdálenost (cube distance = 1 ↔ přímý soused); pro čtvercové mapy Moore sousedství.  
Kód: [`calculate_topographic_error()` som.py](../../app/som/som.py)

#### Spearmanovo ρ (cíl 3)
Spearmanovo pořadové korelační koeficient mezi pairwise vzdálenostmi vah v datovém prostoru a pairwise fyzickými vzdálenostmi neuronů na mřížce. ρ → 1.0 = mapa globálně rozvinuta, ρ → 0 = mapa zmačkaná.

**Proč ne jen TE:** TE detekuje pouze lokální sousedské chyby. Globálně zmačkaná mapa (několik vzdálených clusterů) může mít nízké TE, pokud jsou jednotlivé clustery interně dobře organizovány. Spearman zachytí tuto globální deformaci.  
Kód: [`calculate_topological_correlation()` som.py](../../app/som/som.py)

---

## 4. Omezení (Constraints) — Deb 2002

Místo penalizace objectives se používá **constrained dominance** (Deb et al., 2002):

1. Feasible řešení vždy dominuje infeasible (bez ohledu na raw hodnoty).
2. Mezi dvěma infeasible: menší `constraint_violation` (CV) dominuje.
3. Mezi dvěma feasible: standardní Pareto dominance na 3 raw cílech.

```python
# _dominates() ea.py:189
if cv_p == 0 and cv_q > 0: return True   # p feasible, q ne → p dominuje
if cv_p > 0 and cv_q == 0: return False
if cv_p > 0 and cv_q > 0:  return cv_p < cv_q
# oba feasible → standardní Pareto
```

### CV skalár — co vstupuje do omezení

**Organizační omezení** (`org_cv`):  
`max(0, max(u_matrix_max, dist_map_max) − org_threshold)`  
Threshold je kalibrován analytickou sondou před G0 (viz sekce 8).

**Dead neuron omezení** (`dead_cv`) — kaskádní pásma:

| dead_excess | faktor | důvod |
|-------------|--------|-------|
| `< 0.2` | × 1.5 | mírné varování |
| `< 0.4` | × 2.5 | střední penalizace |
| `≥ 0.4` | × 5.0 | mapa zcela nevhodná |

`dead_excess = dead_ratio − dynamic_threshold`

**Dynamický práh dead neurons:**  
`threshold = clamp(1 − coverage_ratio/10, 0.30, 0.85)`  
kde `coverage_ratio = n_samples / (m × n)`. Velká mapa s malým datasetem má přirozeně více mrtvých neuronů → uvolněný práh.

Kód: [`compute_constraint_violation()` ea.py:161](../../app/ea/ea.py#L161)

---

## 5. Prohledávací prostor

### Parametry

| Parametr | Typ | Rozsah | Poznámka |
|----------|-----|--------|---------|
| `map_size` | `discrete_int_pair` | dynamický | Vesantovo pravidlo (viz sekce 8) |
| `start_learning_rate` | `float` | [0.5, 1.0] | log-scale |
| `end_learning_rate` | `float` | [0.001, 0.2] | log-scale |
| `lr_decay_type` | `categorical` | linear-drop / exp-drop / step-down | |
| `start_radius_init_ratio` | fixed | 1.0 | celá mapa — zajišťuje globální fázi |
| `end_radius` | fixed | 1.0 | min. poloměr 1 neuron |
| `radius_decay_type` | `categorical` | linear-drop / exp-drop / step-down | |
| `epoch_multiplier` | `float` | dynamický | počet epoch = multiplier × n_samples |
| `start_batch_percent` | `float` | [0.0, 3.0] | % datasetu v prvním batchi |
| `end_batch_percent` | `float` | [1.0, 8.0] | % datasetu v posledním batchi |
| `batch_growth_type` | `categorical` | linear-growth / exp-growth | |
| `growth_g` | `int` | [10, 40] | GSOM growth threshold |
| `num_batches` | `int` | [1, 10] | počet mini-batchů |

**Fixované parametry** (nejsou v search space):  
`map_type=hex`, `normalize_weights_flag=false`, `start_radius_init_ratio=1.0`

### Logaritmické vzorkování

LR a radius jsou v log-scale prostoru. SBX crossover a polynomiální mutace pracují na `log(x)`, výsledek se zpětně převede `exp(c)`. Důvod: rozdíl 0.01→0.02 je +100 %, ale 0.51→0.52 je +2 %; lineární prostor by podvzorkoval malé hodnoty.

---

## 6. Genetické operátory

### SBX crossover (Simulated Binary Crossover)

**Parametr:** `sbx_eta = 20.0` (distribuční index)

SBX generuje potomky symetricky kolem rodičů s pravděpodobností koncentrovanou blízko rodičovských hodnot (velké η = exploatace; malé η = explorace). Pracuje per-gen nezávisle pro float parametry.

```
η = 20 → offspring typicky v ±10 % okolí průměru rodičů
```

Pro **kategorické** parametry: uniform swap (50 % pravděpodobnost přepnutí hodnoty).  
Pro **int** parametry: SBX + zaokrouhlení.  
Kód: [`crossover()` ea.py:648](../../app/ea/ea.py#L648), [`crossover_mixed()` operators.py](../../app/ea/operators.py)

### Polynomiální mutace

**Parametry:** `mutation_prob = 0.1` (per-gen), `mutation_eta = 20.0`

Aplikuje se po crossoveru s pravděpodobností `mutation_prob` na každý gen. Stejná distribuce jako SBX — malé perturbace kolem aktuální hodnoty.

### Repair

Po každém crossoveru i mutaci se aplikuje `validate_and_repair()` — deterministická oprava:
- `start_lr > end_lr`: prohození
- `start_batch < end_batch`: prohození
- typy: int round, float clamp na [min, max]

Kód: [`validate_and_repair()` ea.py:100](../../app/ea/ea.py#L100)

### Tournament selection

`tournament_k = 5` (konfigurovatelné). Výběr nejlepšího z k náhodných účastníků dle:
1. Nižší rank (fronta F₀ > F₁ > …)
2. Při stejném ranku: větší crowding distance (zachování diverzity)

Kód: [`tournament_selection()` ea.py:309](../../app/ea/ea.py#L309)

---

## 7. Archiv a crowding distance

**Crowding distance** (Deb 2002): pro každé řešení ve frontě F_k se počítá průměrná vzdálenost k sousedům v objective prostoru. Krajní body dostávají ∞. Řešení s vysokou crowding distance jsou "vzácná" — zachovávají diverzitu fronty.

**Normalizace objectives před HV/crowding:** globální running min/max aktualizovaný každou generaci přes všechna feasible řešení v daném sedu. `norm = (x − running_min) / span`. Nezkresluje HV dimenzemi s různým absolutním rozsahem.

**max_archive_size** (default 25): po naplnění fronty F₀ se ořezávají řešení s nejnižší crowding distance.

Kód: [`crowding_distance_assignment()` ea.py:266](../../app/ea/ea.py#L266)

---

## 8. Předzpracování před G0

### Analytická sonda — kalibrace `org_threshold`

#### Jaký problém řeší

SOM trénink produkuje metriky `u_matrix_max` a `distance_map_max` — maximální hodnoty U-matice a distance mapy. Tyto hodnoty závisí na datasetu, velikosti mapy a hyperparametrech. Normální, správně natrénovaná SOM typicky dává hodnoty **1.02–1.16**.

Pokud bychom použili statický práh `org_threshold = 1.0` (= "cokoliv nad 1.0 je špatně"), penalizujeme prakticky každý běh — i ten, který je objektivně dobrý. EA pak nedokáže rozlišit skutečně špatnou organizaci od normálního výsledku. V jednom testovacím běhu dosáhla míra penalizace **100 %** všech evaluací kvůli tomuto nastavení.

#### Jak sonda funguje

Před startem G0 se spustí `n_probes` (default 15) rychlých SOM tréninků (`epoch_multiplier` snížen na ~30 % normálního) z náhodně vzorkovaných konfigurací v search space — stejném, který bude EA prohledávat.

Ze změřených hodnot `max(u_matrix_max, dist_map_max)` přes všechny sondy se vezme **70. percentil** jako `org_threshold`. To znamená: "horních 30 % nejhorší organizace = penalizace, zbylých 70 % = normální výsledek."

```
sondy: [1.04, 1.08, 1.11, 1.09, 1.15, 1.06, ...]
                                  ↑
                           70. percentil = org_threshold
```

**Výsledek:** práh se automaticky přizpůsobí konkrétnímu datasetu a rozsahu map v search space. EA penalizuje pouze skutečně nadprůměrně špatnou organizaci, ne normální rozptyl.

Kód: [`run_calibration_probe()` ea.py:709](../../app/ea/ea.py#L709) · Výstup: `calibration_probe.csv`

---

### Dynamický search space — heuristika velikosti mapy

#### Co řeší

Kdybychom nechali EA volit velikost mapy z fixního rozsahu (např. 5×5 až 25×25), zbytečně prohledáváme konfigurace které jsou předem nevhodné. Mapa 5×5 = 25 neuronů pro 10 000 vzorků je strukturálně nevhodná (průměr 400 vzorků na neuron). Mapa 25×25 = 625 neuronů pro 150 vzorků Iris je předimenzovaná (průměr 0.24 vzorku na neuron).

#### Heuristika `5 × √n`

Empirické pravidlo publikované Vesantem & Alhoniemim (2000) v kontextu SOM clusteringu: optimální počet neuronů je přibližně `5 × √n_samples`.

**Původ:** pravidlo není matematicky odvozeno, jde o empirické pozorování z rozsáhlé sady experimentů. V literatuře se občas nazývá "SOM sizing heuristic" nebo "5√n rule". Samotný název "Vesantovo pravidlo" je neformální — při hledání použij "SOM map size heuristic Vesanto" nebo přímo citaci níže.

**Citace:** Vesanto, J. & Alhoniemi, E. (2000). *Clustering of the Self-Organizing Map*. IEEE Transactions on Neural Networks, 11(3), 586–600.

```python
U = 5 * sqrt(n_samples)      # optimální počet neuronů celkem
optimal_side = sqrt(U)        # strana čtvercové mapy
map_size_range = [max(8, round(optimal_side * 0.7)),
                  round(optimal_side * 1.3)]
```

Příklad: BreastCancer (569 vzorků) → U ≈ 119, optimal_side ≈ 10.9 → rozsah **[8, 14]**.  
Příklad: dataset 5 000 vzorků → U ≈ 354, optimal_side ≈ 18.8 → rozsah **[13, 24]**.

EA stále prohledává různé konkrétní velikosti a tvary v tomto koridoru — heuristika pouze eliminuje extremy.

`epoch_multiplier` se kalibruje na cílový počet iterací [3 000, 20 000] / n_samples, aby byl čas trénování srovnatelný pro různě velké datasety.

Kód: [`apply_dynamic_search_space()` ea.py:566](../../app/ea/ea.py#L566)

---

## 9. Multi-seed strategie

EA běží opakovaně s různými random seedy (default: `[42]`, pro sběr NN dat doporučeno `[42, 1337, 7, 101, 2026]`).

**Proč:** po konvergenci (~gen 5–8) generuje SBX potomky v úzké oblasti kolem konvergovaného bodu. Jeden dlouhý běh dá N evaluací s nízkou diverzitou. Pět kratších běhů pokryje prostor lépe.

Každý seed dostane vlastní adresář `results/seed_{seed}/`. Kalibrační sonda a preprocessing jsou sdílené (proběhnou jednou).

Kód: [`main()` ea.py:1561](../../app/ea/ea.py#L1561), sekce `seeds`

---

## 10. Pareto metriky

Logováno per-generace do `pareto_metrics.csv`.

| Metrika | Popis |
|---------|-------|
| **Hypervolume (HV)** | Objem prostoru dominovaného frontou vůči referenčnímu bodu [1.1, 1.1, 1.1] v normalizovaném prostoru. Vyšší = lepší. Počítáno přes pymoo. |
| **Spacing** | Průměrná NN vzdálenost mezi body fronty. 0 = dokonale uniformní distribuce. |
| **Spread** | max−min per dimenzi v normalizovaném prostoru. Hodnota blízko 1.0 = fronta pokrývá celý pozorovaný rozsah. |

Do HV a Spacing vstupují pouze **feasible** řešení.

Kód: [`_compute_pareto_metrics()` ea.py:374](../../app/ea/ea.py#L374)

---

## 11. Integrace neuronových sítí (volitelná)

Aktivace: `"use_nn": true` v config-ea.json. Každá vrstva je nezávislá, lze kombinovat.

### MLP pre-screen (Phase 2)
Natrénovaný MLP predikuje `raw_mqe_ratio` z konfigurace PŘED spuštěním SOM. Pokud predikce < `mlp_bad_quality_threshold`, konfigurace se přeskočí.

**Upozornění:** predikovaná hodnota je `raw_mqe_ratio` kde **vyšší = lepší** (ratio 0.9 = dobré). Podmínka přeskočení: `pred < threshold` (ne `>`).

### LSTM early stopping (Phase 2)
LSTM sleduje průběh trénování SOM (sekvenci checkpointů) a rozhoduje, zda pokračovat. Aktivuje se po minimálně 20 % délky tréninku. `quality_score = (1−mqe_ratio) + te + dead×0.5`; stop pokud score > `lstm_quality_threshold`.

### LSTM controller (Phase 3 — experimentální)
Dynamicky upravuje `lr_factor` a `radius_factor` za běhu SOM. Aktuálně omezená prediktivní síla (credit assignment problem — viz ISSUES.md #79).

### CNN vizuální kvalita (Phase 2)
CNN hodnotí U-matrix jako obrázek → `cnn_quality_score ∈ [0, 1]`. Přidává 4. cíl do NSGA-II pokud `use_cnn: true`.

---

## 12. Výstupní soubory

| Soubor | Obsah |
|--------|-------|
| `pareto_front.csv` | archiv per generace — raw objectives, CV, parametry, ds_* metadata |
| `pareto_metrics.csv` | HV, Spacing, Spread per generace |
| `results.csv` | všechna evaluovaná řešení |
| `calibration_probe.csv` | výsledky kalibrační sondy |
| `status.csv` | stav evaluace každého jedince |
| `individuals/<uid>/` | mapy, run_metrics.json, checkpointy pro každé evaluované řešení |

---

## 13. Klíčové reference

- **NSGA-II:** Deb, K. et al. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*. IEEE TEC 6(2).
- **SBX / polynomiální mutace:** Deb, K. & Agrawal, R. B. (1995). *Simulated Binary Crossover for Continuous Search Space*. Complex Systems 9(2).
- **Vesantovo pravidlo:** Vesanto, J. & Alhoniemi, E. (2000). *Clustering of the Self-Organizing Map*. IEEE TNN 11(3). Doporučený počet neuronů ≈ 5√n.
- **Constrained dominance:** Deb, K. et al. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*. Sekce III-B.
- **Topografická chyba:** Kiviluoto, K. (1996). *Topology Preservation in Self-Organizing Maps*. ICNN.
- **Spearman topological correlation:** vlastní implementace; metodologicky blízká Trustworthiness (Venna & Kaski, 2001).
