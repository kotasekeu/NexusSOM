# Ablation Study — kompletní plán experimentů

**Verze**: 1.0
**Datum**: 2026-06-10
**Účel**: Podklad pro jednání s vedoucím katedry. Plán „nice to have all" —
realizovat se může jen část. Slouží k rozhodnutí, co patří do diplomové práce,
co do navazujícího článku a co do případné rigorózní práce.

**Hlavní otázka celé studie:**
> Má hybridní přístup + automatizace neuronovými sítěmi měřitelně lepší
> výsledky než klasický Kohonen s manuální konfigurací — nebo je klasický
> Kohonen „v pohodě" a skutečná hodnota je v analytické vrstvě kolem něj?

Obě odpovědi jsou publikovatelné. Negativní výsledek (automatizace nepřináší
zlepšení úměrné komplexitě) je stejně cenný jako pozitivní — viz uzavření CNN
tracku (`docs/cnn/CNN_REQUIREMENTS.md`).

---

## Výzkumné otázky

| RQ | Otázka | Část |
|----|--------|------|
| **RQ1** | Přináší hybridní trénink měřitelné zlepšení proti deterministickému a stochastickému režimu? | DP |
| **RQ2** | Jaký je přínos každé komponenty (preprocess, EA, MLP, LSTM×2, analýza) vůči její komplexitě? | DP |
| **RQ3** | Je automatická konfigurace (EA + MLP) srovnatelná s manuální expertní konfigurací? | DP |
| **RQ4** | Detekuje systém spolehlivě zanesené chyby a udrží topologii dat? | DP |
| **RQ5** | Jak si systém stojí proti jiným SOM variantám a shlukovacím metodám? | RIG/článek |
| **RQ6** | Kde jsou limity velikosti, dimenzionality a struktury dat? Generalizuje přístup? | RIG/článek |

---

## Společná metodika (platí pro obě části)

### Měřené metriky

| Kategorie | Metriky |
|---|---|
| **Kvalita mapy** | MQE / raw_mqe_ratio, topografická chyba, dead neuron ratio, spatial_quality_score |
| **Kvalita shlukování** | silhouette (cluster = neuron/BMU), Trustworthiness & Continuity, ARI vůči ground truth labelům (kde existují) |
| **Detekce anomálií** | precision, recall, F1 — per typ chyby |
| **Náklady** | čas tréninku, čas optimalizace (EA overhead), špičková paměť |
| **Komplexita komponenty** | LOC, závislosti, nutnost trénovacích dat, údržba — kvalitativní škála |
| **Stabilita** | std všech metrik přes seedy, ARI shody clusterizace mezi běhy |

### Protokol

- **Multi-seed**: každá konfigurace N nezávislých běhů (DP: N=10, RIG: N=30),
  report mean ± std; nefixovaný vs. fixovaný seed jako explicitní dimenze
  (viz otevřený coverage problém)
- **Statistika** (RIG): Friedmanův test + Nemenyi post-hoc napříč metodami,
  Wilcoxon pro párová srovnání, velikost efektu (Cliffova delta)
- **Reprodukovatelnost**: každý běh = uložený config JSON + seed + verze kódu;
  výsledky v timestamped results dirs, agregace do DB přes API

### Infrastrukturní předpoklady (z `docs/som/article_implementation.md`)

1. Multi-seed tool s porovnáním stability (bod 1) — **blokuje téměř vše**
2. Zanášení chyb do generovaných datasetů + P/R vyhodnocení (bod 2)
3. Kvantitativní topologické metriky vůči ground truth (bod 4)
4. Škálovací benchmark (bod 5)
5. Vyřešený coverage tracking (bod 3) — jinak nelze tvrdit přínos hybridu

---

## ČÁST A — Diplomová práce

Rozsah odpovídá cílům 4 a 5 (`THESIS_GOALS.md`): validace na reálné datové
sadě a sadě s uměle vloženými chybami; objektivní zhodnocení přínosu každé
komponenty vůči komplexitě.

### A1 — Komponentní ablace (RQ2, RQ3) — jádro DP

Princip: jeden referenční pipeline, vypínání po jedné komponentě
(leave-one-out), vše multi-seed na 3 referenčních datasetech.

| # | Experiment | Varianty | Známý/očekávaný výsledek |
|---|---|---|---|
| A1.1 | **Preprocess vypnut** | bez normalizace / bez masky šumových dimenzí / bez obojího | ⚠ **Známo: totální rozpad organizace.** Zdokumentovat řízeným experimentem — kvantifikovat rozpad (MQE, TE, vizuál). Nejsilnější argument pro přínos preprocess modulu |
| A1.2 | **Režim tréninku** | deterministický / stochastický / hybridní | RQ1 — hlavní trojice srovnání, stejné datasety, stejné configy |
| A1.3 | **Konfigurace** | EA-optimalizovaná / manuální (Vesanto heuristika) / default config | RQ3 — kolik kvality přidává EA proti rozumnému manuálu |
| A1.4 | **MLP Oracle** | EA s MLP doporučením startu / EA s random init | úspora generací do konvergence, kvalita finální fronty |
| A1.5 | **LSTM early stopping** | zapnuto / vypnuto | ušetřený čas vs. ztracená kvalita (falešně zastavené běhy) |
| A1.6 | **LSTM controller** | zapnuto / vypnuto | zlepší dynamické řízení útlumu kvalitu/čas proti statické křivce? |
| A1.7 | **Celek vs. části** | plný automat / nejlepší jednotlivá komponenta / čistý Kohonen | syntéza — odpověď na hlavní otázku v malém |

Odhad: 7 experimentů × ~3 varianty × 3 datasety × 10 seedů ≈ **600–700 běhů
SOM** (bez EA běhů v A1.3/A1.4). Při minutách na běh řádově dny strojového času.

### A2 — Detekce zanesených chyb (RQ4, cíl 4 DP)

- Generované datasety (swiss roll, space filling, obecný generátor) +
  injektované chyby se známými ID
- Typy chyb pro DP: izolované bodové outliery, malá anomální subgroup,
  chybné hodnoty dle datové masky
- Analýza s chybou / bez chyby; P/R/F1 per typ; vliv chyb na organizaci
  (MQE/TE delta)
- Jeden reálný dataset s ručně vloženými chybami jako sanity check

### A3 — Zachování topologie (RQ4)

- Swiss roll + space filling: kvantitativní metrika vůči ground truth
  (T&C, korelace geodetických vzdáleností), ne jen vizuální topo grafy
- Srovnání det/stoch/hybrid na téže metrice — doplňuje A1.2

### A4 — Datasety pro DP

| Dataset | Typ | Role |
|---|---|---|
| Iris, Wine Quality, BreastCancer | reálné, malé | referenční trojice pro A1 |
| LungCancer (3000×17) | reálný, střední | větší reálná validace |
| swiss roll, space filling | generované, ground truth | A2, A3 |
| 1 větší dataset (≥50k×30+) | reálný/generovaný | důkaz škálování pro DP (plný scaling až RIG) |

### A5 — Minimální obhajitelná varianta DP

Pokud bude čas kritický, neredukovat počet komponent, ale: datasety 3→2,
seedy 10→5, vypustit A1.4 (MLP) a A1.6 (controller) — obě lze popsat
kvalitativně z existujících EA dat. **A1.1, A1.2, A1.3, A2 jsou nepodkročitelné**
— bez nich nelze odpovědět na cíle 4 a 5.

---

## ČÁST B — Rigorózní práce / článek

Cíl: prokázat schopnost samostatné odborné práce — systematická studie
s externími baselinami, statistickou výbavou a hledáním limitů. Staví na
infrastruktuře části A, žádná práce z A se nezahazuje.

### B1 — Plná komponentní matice (RQ2 rozšířená)

- Místo leave-one-out **faktoriální design**: preprocess × režim × konfigurace
  × MLP × LSTM-ES × LSTM-ctrl = až 2⁴×3×3 = 144 kombinací konfigurace
- Frakční design (DOE) pro udržení rozsahu; interakce komponent
  (např. „MLP pomáhá jen s EA", „controller pomáhá jen stochastickému režimu")
- N=30 seedů, plná statistika

### B2 — Externí baseline metody (RQ5)

| Skupina | Metody | Co se srovnává |
|---|---|---|
| Klasické SOM | minisom (online), batch SOM, somoclu | MQE, TE, T&C, čas — přímý souboj s hybridem |
| SOM varianty | Growing SOM / GNG, hexagonal vs. rect | přínos pevné vs. rostoucí topologie |
| Shlukování | k-means, HDBSCAN, GMM, hierarchické | ARI/silhouette na ground truth datasetech; detekce anomálií HDBSCAN vs. naše analýza |
| Redukce dimenze | UMAP/t-SNE/PCA + shlukování | kvalita projekce (T&C) vs. SOM; SOM unikum = obojí najednou |
| Moderní | autoencoder + k-means, SOM-VAE (dle kapacity) | „moderní metody" z hlavní otázky |

Pozn.: analytická vrstva (anomálie, LLM reporty) je na metodě nezávislá —
experiment může ukázat, že právě ona je hlavní přínos (druhá větev hlavní otázky).

### B3 — Škálovací studie (RQ6)

- n_samples: 10² → 10³ → 10⁴ → 10⁵ → 10⁶ (generované + reálné kde existují)
- n_dims: 4 → 17 → 50 → 100 → 500 → 1000; poměr informativních vs. šumových
  dimenzí (test masky)
- map_size škálování (Vesanto vs. EA volba) — kdy mapa přestává stačit
- Výstup: škálovací křivky čas/paměť/kvalita, **zlomové body** = praktické
  limity aplikace

### B4 — Struktury dat (RQ6)

- Gaussian mixtures (kontrolovaný počet/překryv clusterů)
- Manifoldy: swiss roll, S-curve, torus, propletené spirály
- Kategorická / smíšená data, různé kardinality
- Nevyvážené clustery (1:100), hierarchické struktury
- Datasety bez struktury (uniformní šum) — **negativní kontrola**: systém musí
  říct „nic tu není", ne vymýšlet clustery

### B5 — Taxonomie zanesených chyb (RQ4 rozšířená)

- Bodové outliery (vzdálenost ×k od centroidu — citlivostní křivka detekce)
- Subgroup anomálie (velikost 0.1–5 % datasetu)
- Missing values (MCAR/MNAR, 1–30 %)
- Sensor drift (postupný posun části vzorků)
- Duplicity a téměř-duplicity
- Label noise v kategorické dimenzi
- Chyby porušující datovou masku (formátové)
- Pro každý typ: detekční křivka P/R vs. intenzita chyby; srovnání s HDBSCAN
  outlier skóre a izolačním lesem (Isolation Forest)

### B6 — Citlivostní analýzy hyperparametrů

- Decay typy (linear/exp/log/step-down) × granularita × batch growth
- Mapa citlivosti: které hyperparametry reálně rozhodují (rozptyl výsledku
  per parametr — Sobol/Morris screening dle kapacity)
- Zpětně ospravedlňuje EA search space; podklad pro MLP feature importance

### B7 — Výpočetní studie EA (RQ3 rozšířená)

- EA vs. brute-force grid vs. random search vs. Bayesovská optimalizace
  (stejný budget evaluací)
- Konvergenční křivky hypervolume; zranitelnost vůči lokálním minimům
  (multi-seed EA — odpověď na výtku R1)
- Cena: kolik EA evaluací „zaplatí" jedna manuální konfigurace experta

### Odhad rozsahu části B

Řádově **10⁴–10⁵ běhů SOM** + baseline běhy. Realizovatelné jen s dávkovým
spouštěním (worker z `app/main.py`), DB agregací přes API a frakčním designem.
Plný rozsah je na 6–12 měsíců strojového i lidského času — **proto se vybírá**.

---

## Rozhodovací tabulka pro jednání s vedoucím

| Blok | RQ | Odhad běhů | Odhad práce | DP | Článek | RIG |
|---|---|---|---|---|---|---|
| A1 komponentní ablace | RQ1–3 | ~700 | 2–3 týdny | ✅ jádro | výsledky přebírá | rozšiřuje B1 |
| A2 detekce chyb | RQ4 | ~200 | 1–2 týdny | ✅ cíl 4 | ano | rozšiřuje B5 |
| A3 topologie | RQ4 | ~100 | 1 týden | ✅ | ano | — |
| A4 větší dataset | RQ6 | ~50 | 3 dny | ✅ ochutnávka | — | rozšiřuje B3 |
| B1 plná matice | RQ2 | ~4000 | 3–4 týdny | ❌ | částečně | ✅ |
| B2 externí baseliny | RQ5 | ~1000 | 3–4 týdny | textově | ✅ jádro článku | ✅ |
| B3 škálování | RQ6 | ~500 | 2 týdny | ❌ | ano | ✅ |
| B4 struktury dat | RQ6 | ~2000 | 2–3 týdny | ❌ | částečně | ✅ |
| B5 taxonomie chyb | RQ4 | ~3000 | 3 týdny | ❌ | ano | ✅ |
| B6 citlivost | — | ~5000 | 2 týdny | ❌ | — | ✅ |
| B7 EA studie | RQ3 | ~2000 EA evals | 2 týdny | ❌ | ano | ✅ |

**Navrhované dělení k diskusi:**
- **DP** = A1 + A2 + A3 (+ A4 jako ochutnávka škálování). Odpovídá přesně cílům 1–5.
- **Článek** (revize/nový) = B2 + výběr z B3/B5 — externí srovnání je to,
  co recenzenti explicitně chtěli (R2.5).
- **RIG** = B1 + B3–B7 kompletně, se statistickou výbavou B-protokolu.

---

## Známé předběžné výsledky (vstupují do studie jako hypotézy)

1. **Preprocess je kritický**: bez normalizace nebo bez aplikace masky dochází
   k totálnímu rozpadu organizace datasetu (pozorováno; A1.1 to kvantifikuje).
2. **CNN track je slepá ulice** — uzavřeno s plnou dokumentací; nahrazeno
   matematickou prostorovou analýzou (`spatial_quality_score`). Vzor pro
   formát závěru „komponenta nepřináší hodnotu úměrnou komplexitě".
3. **Coverage hybridu je neověřená** — tracking nepotvrzuje očekávané lepší
   pokrytí (otevřený problém; musí být vyřešen před A1.2, jinak hybrid
   obhajujeme bez svého hlavního argumentu).
4. MLP predikuje kvalitu z konfigurace s MAE≈0.02–0.05 (z EA dat) — A1.4
   testuje, jestli se to propíše do reálné úspory.
