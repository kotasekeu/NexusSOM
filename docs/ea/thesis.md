# Kapitola 5: Evoluční optimalizace hyperparametrů

> Stav: osnova — obsah bude dopracován

---

## 5.1 NSGA-II algoritmus

### Motivace volby
- Víceúčelový smíšený prohledávací prostor → jednoobjektový GA nevyhovuje
- Oproti DE a PSO: přirozená podpora kategorických parametrů, vestavěná Pareto archivace
- Lit: Deb et al. (2002) *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*, IEEE TEVC

### Non-dominated sorting
- Definice dominance: `A dominuje B` ⟺ `∀i: A_i ≤ B_i` ∧ `∃i: A_i < B_i`
- Jedinci rozděleni do front F_0, F_1, ..., F_k (F_0 = Pareto fronta)
- Složitost: O(M·N²) kde M = počet cílů, N = velikost populace
- Vzorec: viz originál Deb 2002, Alg. 1

### Crowding distance
- Míra "hustoty okolí" jedince na frontě — zabránění shlukování
- Krajní jedinci v každé dimenzi: ∞
- Ostatní: součet normalizovaných vzdáleností k sousedům přes všechny cíle
- Vzorec: `CD_i = Σ_m (f_m(i+1) - f_m(i-1)) / (f_m^max - f_m^min)`

### Turnajová selekce
- Velikost turnaje k=3 (kompromis: selekční tlak vs diverzita)
- Priorita 1: rank (nižší = lepší)
- Priorita 2 při shodném ranku: crowding distance (vyšší = lepší)

### SBX křížení (Simulated Binary Crossover)
- Simulace binárního křížení v reálném prostoru
- Distribuční index η_c = 20 → potomci blízko rodičů (exploatace)
- Rozptylový faktor β závisí na náhodném u ∈ [0,1]:
  - `u ≤ 0.5: β = (2u)^(1/(η+1))`
  - `u > 0.5: β = (1/(2(1-u)))^(1/(η+1))`
- Potomci: `c_1 = 0.5·((1+β)·p_1 + (1-β)·p_2)`, `c_2 = 0.5·((1-β)·p_1 + (1+β)·p_2)`
- Lit: Deb & Agrawal (1995) *Simulated Binary Crossover for Continuous Search Space*

### Polynomiální mutace
- Perturbace zachovávající meze intervalu
- Parametr η_m = 20, pravděpodobnost mutace p = 0.1 na gen
- Vzorec: `δ_q = (2u)^(1/(η+1)) - 1` pro u < 0.5, symetricky pro u ≥ 0.5
- Mutovaná hodnota: `x' = x + δ_q · (x_max - x_min)`
- Lit: Deb & Goyal (1996) *A Combined Genetic Adaptive Search for Engineering Design*

### Elitismus a archiv
- Po evaluaci: kombinace populace + archiv fronty F_0 z předchozí generace
- Seřazení dle (rank ASC, crowding distance DESC) → výběr nejlepších N
- Archiv = aktuální F_0, přenáší se do další generace

---

## 5.2 Prohledávací prostor

### Typy parametrů
- **Kategorické:** typ topologie (Hex/Square), funkce útlumu LR (linear/exp/log/step), funkce útlumu σ, typ růstu dávky
  - Křížení: uniformní výběr z rodiče
  - Mutace: náhodná náhrada z přípustných hodnot
- **Diskrétní (int pair):** rozměry mapy m×n — mapování ze spojitého intervalu + zaokrouhlení
- **Celočíselné:** epoch_multiplier, growth_g, num_batches — SBX + zaokrouhlení
- **Spojité (lineární měřítko):** velikost dávky (start/end %)
- **Spojité (logaritmické měřítko):** rychlost učení α, počáteční poloměr σ_init

### Logaritmické měřítko
- Motivace: α má exponenciální vliv → lineární vzorkování plýtvá kapacitou poblíž středu
- Příklad: Δ(0.01 → 0.02) = +100 %, Δ(0.51 → 0.52) = +2 %
- Implementace: SBX a polynomiální mutace operují na log(α), výsledek = exp(·)
- Vzorkování: `log_val ~ U(log(min), log(max))`, `α = exp(log_val)`
- Parametry s log-scale: `start_learning_rate` [0.01, 1.0], `end_learning_rate` [0.001, 0.5], `start_radius_init_ratio` [0.05, 1.0]

### Oprava omezení (Repair Mechanism)
- Po křížení a mutaci deterministická kontrola doménových vazeb:
  - α_start ≥ α_end (LR musí klesat)
  - σ_start ≥ σ_end (poloměr musí klesat)
  - batch_start ≤ batch_end (dávka musí růst)
  - `growth_g = 0` pokud všechny křivky lineární (prevence fenotypových duplikátů → cache)
- Alternativa penalizace záměrně odmítnuta: zbytečně komplikuje fitness krajinu

---

## 5.3 Víceúčelová fitness funkce

### Čtyři cíle (všechny minimalizovat)
1. **MQE** — střední kvantizační chyba; primární metrika přesnosti mapování
2. **Topografická chyba (TE)** — podíl vzorků, jejichž 2 nejbližší neurony nejsou sousedé
3. **Poměr mrtvých neuronů** — podíl neuronů bez přiřazeného vzorku
4. **Doba trénování** — výpočetní náklady v sekundách

### Proč ne skalární agregace
- Váhování cílů je subjektivní a skrývá trade-off informaci
- Pareto fronta poskytuje celé spektrum kompromisů — uživatel volí bod dle priority

### Volitelný 5. cíl
- CNN skóre vizuální kvality (viz kapitola 6)
- Minimalizuje se `1 - cnn_score` → vyšší vizuální kvalita = lepší
- Poznámka: pouze pro druhý a další běh EA po natrénování CNN

---

## 5.4 EA vs hybridní SOM

- Hybridní SOM adaptuje průběh trénování *uvnitř* jednoho běhu (dynamická dávka, decay křivky)
- EA optimalizuje *nastavení* trénování *přes běhy* — hledá nejlepší startovní konfiguraci
- Tyto dvě úrovně optimalizace se nevylučují — EA nastaví rámec, hybridní SOM ho naplní
- Klíčový argument pro EA: **generace trénovacích dat** pro MLP, LSTM a CNN
  - Bez EA bychom neměli dostatek strukturovaných dat pro trénování predikčních modelů
  - EA je tedy nejen optimalizátor ale i datový generátor pro zbytek systému

---

## 5.5 Výstup

### Pareto fronta
- Sada konfigurací z fronty F_0 po posledním generaci
- Každá konfigurace: žádná jiná ji nedominuje ve všech čtyřech cílech
- Logována průběžně každou generaci: `pareto_front_log.txt`
- Export: kompletní konfigurace + metriky pro reprodukovatelné trénování

### Trénovací data pro downstream moduly
- **MLP / Oracle:** záznamy `(konfigurace → výsledné metriky)` pro predikci kvality
- **LSTM / Brain:** `checkpoint_count = 10` snímků průběhu trénování per run → časové řady
- **CNN / Eye:** vizualizace map (U-matice, vzdálenostní mapa, mrtvé neurony) → RGB obrázky

### Deduplikace a cache
- `UID = MD5(sorted(config))` — deterministický hash konfigurace
- Výsledky cachované cross-generačně → opakující se konfigurace se nehodnotí znovu
- Statistika: total_requested / cache_hits / new_evaluations

---

## 5.6 Problémy a výzvy

### Výpočetní náklady evaluace
- Jedno trénování SOM: sekundy až minuty v závislosti na epoch_multiplier a map_size
- Řešení: `multiprocessing.Pool(min(10, cpu_count))` — paralelní evaluace celé populace
- Problém: spawnování procesů inicializuje TF modely opakovaně → per-process NN cache

### Checkpointy a early stopping
- Checkpointy se ukládají v intervalech `evaluations / checkpoint_count`
- Early stopping může ukončit trénování před dosažením všech 10 checkpointů → kratší sekvence
- Řešení pro LSTM: padding poslední hodnotou na cílovou délku
- Důsledek: LSTM trénován převážně na simulovaných nebo paddovaných sekvencích

### Nominální vs reálný prohledávací prostor
- `map_size` — malé změny v genotypu nemusí změnit fenotyp (zaokrouhlení na int)
- `growth_g` — ignorován pokud jsou křivky lineární → bez opravy by generoval funkčně stejné ale odlišné UID
- Obě vyžadovaly speciální handling v `validate_and_repair()`

### Záporné predikce MLP
- MLP predikuje záporné hodnoty pro TE a dead_ratio (fyzicky nemožné)
- Nutné ořezání výstupu — clip nebo ReLU na výstupní vrstvě

---

## 5.7 Hodnocení

### Co funguje
- NSGA-II nalezl Pareto-optimální konfigurace autonomně, bez ruční expertízy
- Log-scale sampling výrazně zlepšuje pokrytí prostoru pro citlivé parametry (LR, σ)
- Pareto fronta poskytuje interpretovatelné kompromisy, ne černou skříňku
- Datový výstup EA: základní vstup pro trénování MLP, LSTM, CNN

### Otevřené otázky
- Jak velká populace / kolik generací je dostatečné? (benchmarking chybí)
- Přenositelnost konfigurací z jednoho datasetu na jiný — jak obecná je Pareto fronta?
- Reálný přínos CNN jako 5. cíle — zlepšuje konvergenci nebo jen přidává šum?
- Srovnání EA vs náhodné prohledávání (random search) — klasický baseline

### Poznámky k experimentům
- Validace na BreastCancer dataset (569 vzorků, 30 příznaků)
- Virtuální datasety: 9 synteticky generovaných ze stejného schématu → 1500 EA evaluací celkem
- Výsledky: viz příloha D
