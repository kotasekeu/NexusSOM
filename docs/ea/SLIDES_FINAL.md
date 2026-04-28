# NexusSom — EA Podklady pro prezentaci
## Formát: [Slide] / [Komentář řečníka]

---

## Slide 1 — Název

**Slide:**
> Víceúčelová evoluční optimalizace hyperparametrů SOM
> NSGA-II · SBX · Pareto

**Komentář:**
Projekt řeší konkrétní problém: SOM má ~15 hyperparametrů a jejich nastavení výrazně ovlivňuje kvalitu výsledné mapy. Cílem EA není najít jedno "nejlepší" řešení, ale celou sadu kompromisních konfigurací — Pareto frontu. Tento výstup pak slouží jako základ pro trénování predikčních modelů.

---

## Slide 2 — Proč evoluční algoritmus?

**Slide:**
- ~15 hyperparametrů: trénovací dynamika, topologie, dávkování
- Prostor smíšený: spojité + diskrétní + kategorické
- Grid search: kombinatorická exploze, nefunkční pro spojité intervaly
- Ruční ladění: neobjektivní, nepřenositelné
- **Cíl: autonomní prohledávání → Pareto fronta kompromisů**

**Komentář:**
Prostor nelze systematicky prohledat. Samotná rychlost učení je spojitá — mezi 0.01 a 0.9 je nekonečně mnoho hodnot. Ke kombinaci s typem útlumové křivky, rozměry mapy a dalšími parametry je grid search nerealistický. EA prohledává prostor řízeně: přežívají konfigurace, které fungují dobře na více frontách naráz.

---

## Slide 3 — Čtyři navzájem si konkurující cíle

**Slide:**
| Cíl | Směr |
|-----|------|
| MQE — přesnost mapování dat | ↓ min |
| Topografická chyba | ↓ min |
| Poměr mrtvých neuronů | ↓ min |
| Doba trénování | ↓ min |

→ Žádné váhování. Výstup: **Pareto fronta**, ne jedno číslo.

**Komentář:**
Tato čtyři kritéria jsou přirozeně v konfliktu. Přesnější mapa vyžaduje více epoch → delší čas. Větší mapa má méně mrtvých neuronů ale horší topografii pro malé datasety. Jednoobjektový přístup by vyžadoval váhování — to je subjektivní a skrývá trade-off. NSGA-II vrací celou hranici možného.

---

## Slide 4 — Tři ortogonální vrstvy návrhu

**Slide:**

```
Vrstva 1: Typ problému          Vrstva 2: Algoritmus         Vrstva 3: Operátory
─────────────────────────       ──────────────────────       ──────────────────────
single-objective  → 1 skóre    GA, DE, PSO                  SBX křížení
multi-objective   → Pareto  ←  NSGA-II, SPEA2           ←   Polynomiální mutace
many-objective    → 8+ cílů    NSGA-III, MOEA/D             Uniformní křížení
                                                             ...
```

**Volba operátorů je nezávislá na volbě algoritmu.**

Náš případ:
- Vrstva 1: **multi-objective** (4 cíle)
- Vrstva 2: **NSGA-II** (viz dále)
- Vrstva 3: **SBX + polynomiální + uniformní** (viz dále)

**Komentář:**
Toto oddělení je klíčové pro správné porozumění. NSGA-II neurčuje, jaké křížení se použije — do NSGA-II lze zapojit DE operátory, uniformní křížení nebo cokoliv jiného. Stejně tak SBX lze použít uvnitř SPEA2. Algoritmus (vrstva 2) řeší selekci a archivaci; operátory (vrstva 3) řeší jak generovat nové jedince. NSGA-III a MOEA/D nejsou "lepší verze NSGA-II pro stejný problém" — jsou to odpovědi na jiný typ problému (many-objective, vrstva 1).

---

## Slide 5 — Proč NSGA-II? *(vrstva 2: algoritmus)*

**Slide:**
- **Typ problému: multi-objective** (4 cíle) → potřebujeme Pareto archivaci

Proč ne single-objective alternativy?
- DE, PSO: jeden cíl, nebo váhovaná suma → ztrácí trade-off informaci

Proč NSGA-II, ne SPEA2?
- Nedominované třídění → rank bez parametrů
- Crowding distance → diverzita bez fixní velikosti archivu
- Pareto fronta proměnné velikosti — nevíme předem kolik řešení bude

*(Detailní srovnání se SPEA2 — viz poslední slide)*

**Komentář:**
DE a PSO jsou schopny multi-objective optimalizace ve svých rozšířených variantách (MODE, MOPSO), ale jejich základní design je single-objective a rozšíření jsou méně přímočará. Klíčový argument pro NSGA-II oproti SPEA2 je praktický: SPEA2 vyžaduje předem nastavit velikost externího archivu — pro náš případ, kde nevíme jak velká bude Pareto fronta, je to nežádoucí parametr.

## Slide 6 — NSGA-II vs SPEA2 *(vrstva 2: algoritmus, oba pro multi-objective)*

**Slide:**

| | **NSGA-II** | **SPEA2** |
|--|-------------|-----------|
| Fitness | Rank (číslo fronty) | Strength: počet dominovaných řešení |
| Diverzita | Crowding distance | k-NN hustota v cílovém prostoru |
| Archiv | Pareto fronta F₀ — proměnná velikost | Fixní archiv — parametr |
| Truncation | Dle (rank, crowding dist.) | Boundary solutions zachovány explicitně |
| Složitost | O(MN²) | O(N² log N) — k-NN |

**Oba algoritmy:** multi-objective (2–5 cílů) · libovolné operátory (SBX, DE, …)

**Naše volba — NSGA-II:**
- Velikost Pareto fronty není předem známa → fixní archiv SPEA2 = nežádoucí parametr
- Crowding distance pro 4 cíle srovnatelně účinná jako k-NN
- Wider adoption → snazší srovnání s literaturou

*(NSGA-III / MOEA/D = jiný typ problému: many-objective, 8+ cílů — jiná vrstva 1)*

**Komentář:**
SPEA2 (Zitzler et al., 2001) používá sofistikovanější hustotní estimaci přes k nejbližších sousedů — to dává lepší pokrytí při nepravidelném tvaru fronty. Klíčový praktický rozdíl: SPEA2 vyžaduje předem nastavit velikost externího archivu. Pokud nastavíme příliš malý, odřežeme validní řešení; příliš velký = degradace selekčního tlaku. Pro náš případ nevíme dopředu, kolik Pareto-optimálních konfigurací SOM existuje. Důležité: NSGA-III a MOEA/D nejsou "lepší verze NSGA-II" — jsou to odpovědi na many-objective problém (vrstva 1), kde Pareto dominance slábne a crowding distance ztrácí diskriminační sílu. Naše 4 cíle jsou stále v komfortní zóně NSGA-II.


---

## Slide 5 — Prohledávací prostor

**Slide:**
- **Kategorické:** typ topologie (Hex/Square), funkce útlumu (linear / exp / log / step)
- **Diskrétní (int pair):** rozměry mapy m×n — mapování ze spojitého intervalu
- **Spojité (lineární měřítko):** velikost dávky, epoch multiplier
- **Spojité (logaritmické měřítko):** rychlost učení α, počáteční poloměr σ

**Proč log-scale pro α?**
Δ(0.01 → 0.02) = +100 %    ×    Δ(0.51 → 0.52) = +2 %

**Komentář:**
Lineární rovnoměrné vzorkování by zahltilo prostor hodnotami kolem středu intervalu. Rychlost učení a poloměr sousedství mají exponenciální vliv — rozdíl 0.01 vs 0.02 je klíčový, 0.51 vs 0.52 prakticky irelevantní. Operátory SBX a polynomiální mutace proto pracují v log-prostoru: cross-over a mutace jsou tam proporcionálně symetrické.

---

## Slide 6 — SBX a polynomiální mutace *(vrstva 3: operátory)*

**Slide:**
**SBX (Simulated Binary Crossover), η = 20:**
- Potomci v těsné blízkosti rodičů → exploatace
- Symetrické rozprostření kolem rodičů
- Log-space varianta: SBX pracuje na log(α), výsledek = exp(c)

**Polynomiální mutace, η = 20, p = 0.1 / gen:**
- Perturbace zachovávající meze intervalu
- Vyšší η = menší skok, přesnější doladění

*Oba operátory jsou algoritmicky nezávislé — fungují stejně v NSGA-II i SPEA2.*

**Komentář:**
SBX a polynomiální mutace pochází z prací Deba — jsou historicky asociovány s NSGA-II, ale jsou to obecné operátory pro spojité prostory, nikoli součást NSGA-II samotného. Lze je použít v libovolném EA frameworku. Distribuční index η=20 volíme tak, aby potomci byli blízko rodičů — preferujeme exploataci v pokročilé fázi evoluce. Polynomiální mutace s p=0.1 znamená, že každý parametr se mutuje s 10% pravděpodobností per gen — dostatečná diverzifikace bez destrukce dobrých řešení.

---

## Slide 7 — Oprava omezení (Repair Mechanism)

**Slide:**
Po křížení a mutaci — deterministická oprava:

- α\_start ≥ α\_end (LR musí klesat)
- σ\_start ≥ σ\_end (poloměr musí klesat)
- batch\_start ≤ batch\_end (dávka musí růst)
- growth\_g = 0 pro čistě lineární konfigurace (prevence fenotypových duplikátů)

**Komentář:**
SBX a mutace pracují každý parametr nezávisle — mohou narušit vztahy mezi parametry. Oprava je jednoduchá: prohození hodnot nebo nastavení konstanty. Poslední pravidlo je subtilní: parametr growth_g ovlivňuje tvar exponenciálních křivek, ale pro lineární funkce nemá žádný vliv. Pokud jsou všechny křivky lineární, fixujeme growth_g=0, aby identické konfigurace měly stejné UID → cache deduplikace funguje správně.

---

## Slide 8 — NSGA-II: tok algoritmu

**Slide:**
```
Inicializace: N náhodných konfigurací (log-uniform pro α, σ)
        ↓
┌── Generace 1..G ──────────────────────────────────┐
│  Paralelní evaluace populace (multiprocessing Pool) │
│  ↓                                                  │
│  Kombinace populace + archiv (elitismus)            │
│  ↓                                                  │
│  Non-dominated sorting → Pareto fronty (ranky)     │
│  ↓                                                  │
│  Crowding distance → diverzita uvnitř fronty       │
│  ↓                                                  │
│  Turnajová selekce (k=3): rank, pak crowding dist. │
│  ↓                                                  │
│  SBX + polynomiální mutace → nová generace        │
│  ↓                                                  │
│  Oprava omezení → UID → cache                      │
└───────────────────────────────────────────────────┘
        ↓
Archiv fronty 0 = finální Pareto fronta
```

**Komentář:**
Klíčové jsou dva mechanismy zachování kvality: (1) elitismus — nejlepší fronta přežívá celá do další generace, (2) crowding distance — zabraňuje shlukování řešení v jednom místě fronty. Turnaj k=3 balancuje selekční tlak: příliš malý = pomalá konvergence, příliš velký = ztráta diverzity.

---

## Slide 9 — Efektivita: cache a paralelismus

**Slide:**
- **Paralelní Pool:** celá populace trénuje simultánně (max 10 procesů)
- **UID = MD5(konfigurace)** → deterministický hash
- Duplikáty napříč generacemi → přímé načtení z cache
- V pokročilých generacích: velká část populace z cache

*Reálná výpočetní náročnost ≪ N × doba_trénování*

**Komentář:**
Trénování jedné SOM konfigurace trvá sekundy až minuty. Bez paralelismu by jedna generace trvala desítky minut. Cache je kritická zejména ve fázích konvergence — populace obsahuje opakující se konfigurace z archivů a elitismu, které se nemusí přepočítávat.

---

## Slide 10 — Výstup EA

**Slide:**
**Primární výstup:**
- Pareto fronta: sada optimálních kompromisů pro nasazení SOM

**Vedlejší produkt — strukturovaná data:**
- Průběhy trénování (checkpointy) → trénovací sekvence pro LSTM
- Vizualizace map (U-matice, vzdálenostní mapy) → vstup pro CNN
- Konfigurace + výsledné metriky → vstup pro MLP

*Sítě neovlivňují aktuální běh EA. Po přetrénování mohou akcelerovat další iteraci.*

**Komentář:**
EA zde funguje dvakrát. Primárně jako optimalizátor — hledáme nejlepší způsob trénovat SOM. Vedlejším produktem je rozsáhlý dataset: tisíce konfigurací s jejich výsledky, průběhy trénování a vizualizacemi. Tyto data slouží k trénování predikčních modelů, které v budoucí iteraci EA mohou filtrovat špatné konfigurace ještě před spuštěním SOM nebo zastavit trénování, pokud průběh nevypadá perspektivně.

---

## Slide 11 — Výsledky

**Slide:**
- Po 30 generacích, populace 50: typicky 5–20 Pareto-optimálních řešení
- Fronta pokrývá celé spektrum: rychlé mapy ↔ přesné mapy
- Konfiguraci z fronty nelze zlepšit v jednom cíli bez degradace jiného

*(ukázka grafu: MQE vs. čas, barevně podle topografické chyby)*

**Komentář:**
Výstupem není jedno číslo, ale hranice možného pro daný dataset. Analytik si pak vybírá bod na frontě podle priority — chce rychlou analýzu nebo maximální přesnost? Klíčové je, že EA tuto hranici nalezl autonomně bez znalosti doménové expertízy.

---

## Slide 12 — Shrnutí

**Slide:**
| Co | Proč |
|----|------|
| NSGA-II | Víceúčelový smíšený prostor — přirozená volba |
| SBX + polynomiální mutace | Exploatace blízkosti rodičů, log-space pro α, σ |
| Log-scale sampling | Proporcionální diverzita v citlivých intervalech |
| Oprava omezení | Fenotypová platnost bez fitness penalizace |
| UID cache | Eliminace duplikátů → reálná efektivita |
| Pareto archiv | Elitismus bez ztráty diverzity |

**→ EA jako generátor dat i jako optimalizátor**

**Komentář:**
Kombinace NSGA-II s SBX a logaritmickým měřítkem dává robustní základ pro prohledávání prostoru hyperparametrů SOM. Výstupem je nejen sada nejlepších konfigurací, ale i rozsáhlý trénovací dataset pro predikční modely, které mohou v další iteraci výrazně snížit výpočetní náklady.

---

## Slide — Smíšené genetické operátory: typové větvení

**Slide:**

| Typ parametru | Křížení | Mutace (p = 0.1 / gen) |
|---------------|---------|------------------------|
| `float` (lin.) | SBX, ořez na [min, max] | Polynomiální, ořez na [min, max] |
| `float` (log) | SBX v log-prostoru → exp(·) | Polynomiální v log-prostoru → exp(·) |
| `int` | SBX + round() | Polynomiální + round() |
| `categorical` | Uniform swap (p = 0.5 / gen) | Náhodná náhrada z přípustných hodnot |
| `int pair` (m×n) | SBX na skalár s → [s, s] | Polynomiální na s → [s, s] |

Každý gen zpracován **nezávisle** podle svého typu.

**Komentář:**
Standardní NSGA-II předpokládá čistě spojitý prostor — aplikuje SBX na celý chromozom jako vektor reálných čísel. Zde máme čtyři různé typy genů a pro každý platí jiná operace. Uniformní swap pro kategorické je správná volba: SBX by nedával smysl pro hodnoty jako "linear-drop" vs "exp-drop". Mutace pro `int` a `float` sdílí polynomiální základ, ale celočíselné parametry se zaokrouhlují. Pravděpodobnost p=0.1 je per-gen, nikoli per-jedinec — v průměru se zmutuje 1–2 parametry z 14.

---

## Slide 14 — SBX a polynomiální mutace

**Slide:**

**SBX** — u ~ U[0,1], p₁ ≤ p₂:
```
β = (2u)^(1/(η+1))          u ≤ 0.5
β = (1/(2(1−u)))^(1/(η+1))  u > 0.5

c₁ = ½·((1+β)·p₁ + (1−β)·p₂)
c₂ = ½·((1−β)·p₁ + (1+β)·p₂)
```
η = 20 → β ≈ 1 pro většinu u → **potomci blízko rodičů**

**Polynomiální mutace** — x' = x + δ_q · (x_max − x_min):
```
δ_q ∈ (−1, 1),  |δ_q| ≪ 1 pro η = 20
```
→ malá perturbace symetricky kolem x, zachovává meze

**Log-space (α, σ):** SBX/mutace pracuje na log(x), výsledek = exp(·)
→ β=1 v log-prostoru = násobení konstantou, ne additivní posun

**Komentář:**
β je rozptylový faktor — u blízké 0.5 dává β≈1, potomci splývají s rodiči. U krajních hodnot u→0 nebo u→1 je β>1, potomci mohou přeskočit za rodiče (explorace). Distribuční index η=20 tlačí β k 1: přednost exploataci nad explorací. U polynomiální mutace je δ_q normalizované na délku intervalu — malá absolutní hodnota, ale proporcionální k rozsahu parametru. Log-space je klíčová myšlenka: v lineárním prostoru by mutace z α=0.01 na intervalu [0.001, 1.0] generovala skoro vždy hodnoty poblíž středu 0.5 — δ_q·0.999 dominuje. V log-prostoru je interval [log(0.001), log(1.0)] ≈ [-7, 0] a mutace je symetrická v řádech velikosti.

---

