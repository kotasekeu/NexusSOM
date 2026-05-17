# EA — Problémy a rozhodnutí

## Architektura a návrh

1. **Volba NSGA-II místo single-objective EA** — původně čtyři cíle (MQE ratio, TE, dead ratio, čas); přešlo se na tři cíle po zjištění, že čas jako Pareto cíl přitahuje malé rychlé mapy bez ohledu na kvalitu (viz bod 32); Pareto fronta zachovává spektrum kompromisů kvality.

2. **Volba SBX + polynomiální mutace místo DE nebo uniformního křížení** — SBX zajišťuje exploataci blízkosti rodičů a proporcionální skok v celém intervalu; polynomiální mutace zachovává meze bez ořezu.

3. **Oddělení tří vrstev: typ problému / algoritmus / operátory** — NSGA-II neurčuje typ křížení; SBX lze použít i v SPEA2; záměna vrstev vedla k nesprávnému pochopení architektury.

4. **Volba NSGA-II místo SPEA2** — SPEA2 vyžaduje předem nastavenou velikost externího archivu; pro neznámou velikost Pareto fronty je proměnná fronta NSGA-II výhodnější.

---

## Problémy v kódu EA

5. **Archiv buildovaný ze stale indexů po sort()** — `ARCHIVE = [combined_population[i] for i in fronts[0]]` bylo voláno po `combined_population.sort()`, takže indexy z `fronts[0]` ukazovaly na špatné jedince; dobrá řešení (nízké MQE) se ztrácela z archivu a nahrazovala penalizovanými; opraveno filtrováním `rank == 0` ze seřazeného seznamu.

6. **Hardcoded `tournament_k = 3` nevhodný pro malé populace** — pro populaci 5 jedinců by k=3 eliminovalo 60 % genetické diverzity; nahrazeno dynamickým `max(2, population_size // 10)` s možností přepisu v configu.

7. **`cnn_quality_score` jako NaN v results.csv** — `log_result_to_csv` byl volán před CNN hodnocením; přesunuto za CNN blok.

8. **Stale indexy v `fronts[0]` způsobovaly ztrátu elitismu** — viz bod 5; bug byl přítomen ve všech bězích od začátku a znemožňoval skutečné uchovávání nejlepších konfigurací.

---

## Fitness a hodnocení

9. **Absolutní MQE jako fitness ukazatel je nepřesný** — MQE závisí na velikosti mapy i datasetu; SOM 18×18 na Iris s MQE=0.1→0.095 vypadá skvěle, ale delta je minimální; nahrazeno `mqe_improvement_ratio = final_mqe / initial_mqe` (z checkpoint[0]).

10. **Topographic error a dead ratio nepotřebují normalizaci** — obě metriky jsou přirozeně v [0, 1] jako podíly, nezávislé na datasetu ani mapě; pokus o normalizaci ratiem initial hodnoty měnil charakter dat bez přínosu.

11. **Penalizace špatné organizace mapy deformuje MQE** — u_matrix_max > 1.0 nebo distance_map_max > 1.0 násobí `best_mqe` faktorem 10+; penalizovaná hodnota (QE=14.8) je pak srovnávána s legitimní (QE=0.55) na stejné škále — bylo nutné přejít na improvement ratio.

---

## Prohledávací prostor

12. **Lineární vzorkování learning rate zkresluje prostor** — rozdíl 0.01→0.02 je +100 %, ale 0.51→0.52 je +2 %; logaritmické vzorkování zajišťuje proporcionální pokrytí; přidáno `log_scale: true` pro LR a radius v configu.

13. **SBX a polynomiální mutace v lineárním prostoru pro LR** — křížení a mutace v lineárním prostoru generují potomky u středu intervalu; přidána log-space varianta: SBX pracuje na `log(x)`, výsledek je `exp(c)`.

14. **Smíšený prohledávací prostor vyžaduje typové větvení operátorů** — `float`, `float(log)`, `int`, `categorical`, `int pair` mají každý jinou operaci; uniformní swap pro kategorické, SBX+round pro int.

15. **`processing_type` jako zbytečný parametr** — SOM vždy funguje v hybrid módu; přítomnost v search space vedla k redundantní diversifikaci; odstraněno ze všech konfigů a SOM kódu.

16. **Oprava omezení (`repair`) je nutná po každém křížení i mutaci** — SBX pracuje na každém genu nezávisle a může narušit vztahy (LR_start < LR_end apod.); deterministická oprava prohozením nebo konstantou aplikována konsistentně.

---

## Pareto fronta a diverzita

17. **Pareto fronta exploduje se čtyřmi cíli** — matematicky ~39 % náhodných řešení je non-dominovaných ve 4D prostoru; s populací 50 to znamená ~20 řešení v archivu bez smysluplné diskriminace; řešeno `max_archive_size` capem s výběrem dle crowding distance.

18. **Malá populace generuje velkou frontu bez smysluplného pokrytí** — 5 jedinců, 3 generace = 15 hodnocení; i penalizovaná řešení (QE=11–15) byla non-dominovaná protože chyběly alternativy; pro reálné výsledky nutná populace ≥ 30.

---

## Efektivita a paralelismus

19. **Paralelizace na Windows vyžaduje `freeze_support` a spawn mode** — na macOS/Linux funguje fork; Windows vždy spawnuje nový proces; zdánlivě sekvenční chování bylo způsobeno spawn overhead, ne chybou paralelismu.

20. **EVALUATED_CACHE není sdílen přes worker procesy** — při `Pool.starmap` má každý worker prázdný cache; jedinci jsou re-evaluováni v každé generaci pokud jsou v populaci; cache funguje správně pouze v sequenčním módu.

21. **Datová sada pro trénování NN vyžaduje normalizaci MQE přes datasety** — MQE 18×18 mapy na BreastCancer není srovnatelné s 5×5 mapou na Iris; `mqe_improvement_ratio` normalizuje oboje; topo a dead ratio nepotřebují normalizaci.

22. **Dataset jako feature pro NN modely** — trénování MLP/LSTM na jednom datasetu nedává generalizovatelný model; statistiky datasetu (`n_samples`, `n_features`, `n_categorical` atd.) přidány jako vstupní features.

---

## Architektura cílů a penalizace

32. **`duration` odstraněno jako Pareto cíl — nahrazeno loggingem** — tréninkový čas jako plnohodnotný cíl NSGA-II trvale zvýhodňoval malé rychlé mapy (5×5, 3–5 s) bez ohledu na kvalitu; tato řešení obsazovala krajní pozice TIME osy → vysoká crowding distance → přežívala cap a vytlačovala kvalitní řešení; čas je nyní logován jako metadata, ale nevstupuje do dominance; cíle přešly na 3: `[raw_mqe_improvement_ratio, raw_topographic_error, dead_neuron_ratio]`.

33. **Penalizace přepisovala raw hodnoty — ztráta informace** — multiplykace `best_mqe *= penalty_factor` a `topographic_error *= penalty_factor` zničila původní naměřené hodnoty; nebylo možné rozlišit "skutečně špatný výsledek" od "dobrého výsledku s penalizací za organizaci"; přidány oddělené sloupce `raw_best_mqe`, `raw_topographic_error`, `raw_mqe_improvement_ratio`, `penalty_factor`, `is_penalized`, `penalty_reason`.

34. **NSGA-II cíle přešly na raw hodnoty** — po oddělení raw a penalizovaných metrik používá NSGA-II pro dominance a ranking raw hodnoty (`raw_mqe_improvement_ratio`, `raw_topographic_error`); penalizované hodnoty jsou logovány v CSV pro referenci, ale nevstupují do Pareto výpočtů; penalizace stále tlačí EA správným směrem přes selection pressure, ale nesnižuje raw kvalitu v objectives.

35. **Penalizace zůstává jako selection pressure, ne jako archivní filtr** — penalizovaná řešení zůstávají v `combined_population` jako potenciální rodiče (genetická diverzita); v archivu se objeví pouze pokud mají dobré raw hodnoty na některém z cílů; Pareto log zobrazuje pro penalizovaná řešení primárně raw hodnoty + druhý řádek s penalizovanými hodnotami a důvodem penalizace.

---

## Dynamická penalizace a thresholdy

36. **Dynamická penalizace je architektonicky nekompatibilní s NSGA-II** — změna prahů penalizace mezi generacemi (slabší penalizace v G1, silnější v G5) mění fitness landscape; archivní řešení hodnocená v G1 jsou pak dominována nebo dominují G5 řešení na základě jiných pravidel, ne lepší kvality; NSGA-II předpokládá stabilní a konzistentní fitness funkci přes celý běh; dynamická penalizace by vyžadovala re-evaluaci celého archivu v každé generaci, což je výpočetně neúnosné; penalizace musí být statická (konstantní prahy od G0) nebo odstraněna.

37. **Dead penalty a `dead_neuron_ratio` jako Pareto cíl plní odlišné role** — `dead_neuron_ratio` jako Pareto cíl minimalizuje podíl neaktivních neuronů v rámci trade-offů (EA může zvolit menší mapu s nižším dead ratio); dead penalty jako penalizace MQE signalizuje fundamentálně špatný poměr mapa/dataset a tlačí EA pryč od takových konfigurací dříve, než se dostanou do archivu; dvě role nejsou redundantní — Pareto cíl řídí jemný trade-off, penalizace zabraňuje propagaci strukturálně vadných konfigurací; dead penalty zůstává, ale musí mít správně nastavený threshold (viz bod 38).

38. **`DEAD_NEURON_THRESHOLD=10%` nerespektuje vztah velikosti mapy a datasetu** — statický práh 10 % ignoruje, že dead neuron ratio je primárně funkcí `coverage_ratio = n_samples / (m * n)`; pro mapu 10×10 a 569 vzorků je `coverage_ratio ≈ 5.7` — stále vznikají desítky neaktivních neuronů i při dobré topologii; threshold musí vycházet z `coverage_ratio`: při pokrytí ≥ 10 vzorků/neuron je threshold 30 %, při pokrytí 5–10 je threshold 50 %, při pokrytí 2–5 je threshold 70 %, při pokrytí < 2 je threshold 85 %; vzorec: `threshold = clamp(1 - coverage_ratio / 10, 0.3, 0.85)`; tím se práh automaticky zpřísňuje pro velká data (kde by mnoho dead neurons bylo opravdu problém) a uvolňuje pro malé datasety na velké mapě.

41. **Krokové (graduated) penalizace dead ratio místo binárního prahu** — jeden pevný práh přechází z nulové penalizace na plnou naráz, bez gradientu; lepší přístup je krokové pásmo: dead_ratio v `[threshold, threshold+0.2)` → `penalty_factor 1.5` (mírné varování), v `[threshold+0.2, threshold+0.4)` → `penalty_factor 2.5` (střední penalizace), v `[threshold+0.4, 1.0]` → `penalty_factor 5.0` (těžká penalizace — mapa zcela nevhodná); graduated penalizace zachovává informaci (o kolik je mapa předimenzovaná) a dává EA spojitý gradient pro výběr lepší velikosti mapy.

42. **Omezení search space pro `map_size` dle analýzy datasetu** — velikost mapy by neměla být volena čistě EA; analýza datasetu před EA rundem odhalí `n_samples` a `n_features`; empirické pravidlo pro SOM: optimální počet neuronů ≈ `5 * sqrt(n_samples)`; pro BreastCancer (569 vzorků) → ~119 neuronů → mapa 11×11; search space pro `map_size` lze zúžit na `[max(8, optimal*0.5), optimal*1.5]`; EA stále prohledává různé tvary mapy (obdélník, čtverec) a různé velikosti v rozumném rozsahu, ale eliminuje extremy (5×5 nebo 25×25) které jsou dopředu nevhodné.

39. **`ORGANIZATION_THRESHOLD=1.0` pro `dist_map_max` příliš přísný** — typické hodnoty `dist_map_max` při správném tréninku jsou v rozmezí 1.02–1.16; threshold 1.0 penalizuje téměř každý běh za hraniční překročení (o 2–16 %); penalizace by měla postihovat pouze skutečně špatnou organizaci (rozpadlé mapy, U-Matrix s hodnotami 5–10), ne normální rozptyl dobře naučených map; pevný práh bez kalibrace na dataset vede ke 100% míře penalizace.

40. **Kalibrační přístup pro organization threshold** — správný postup je spustit 10–20 rychlých náhodných konfigurací SOM před zahájením EA (1–2 % celkového výpočetního budgetu); z naměřených hodnot `dist_map_max` vzít 70. percentil jako práh penalizace; tím se práh automaticky přizpůsobí datasetu, velikosti mapy a tréninkovým podmínkám; threshold ≤ 70. percentilu = normální výsledek, threshold > 70. percentilu = skutečně špatná organizace hodná penalizace; tato kalibrace se provede jednou před G0 a prahy zůstanou konstantní po celý běh EA (viz bod 36).

46. **Analytická sonda před EA kalibruje organization threshold** — před spuštěním G0 se spustí N rychlých SOM tréninků (`epoch_multiplier=0.3`, paralelně) z náhodných konfigurací v prohledávacím prostoru; ze změřených hodnot `max(u_matrix_max, dist_map_max)` se vypočítá 70. percentil jako `org_threshold`; výsledky jsou logovány do `calibration_probe.csv`; kalibrovaný práh se předá do `fixed_params['org_threshold']` a odtud do `evaluate_individual` a `compute_constraint_violation`; výchozí chování: 15 sond, konfigurovatelné přes sekci `CALIBRATION: {n_probes: 15, probe_epoch_multiplier: 0.3}` v EA configu; při `n_probes=0` se sonda přeskočí a použije fallback 1.0.

47. **Dynamický search space pro `map_size` dle Vesantova pravidla** — velikost mapy se neprohledává ve fixním rozsahu; před startem EA se vypočítá optimální počet neuronů `U = 5 * sqrt(n_samples)` a optimální strana `optimal_side = sqrt(U)`; search space se upraví na `[max(8, round(optimal_side*0.7)), round(optimal_side*1.3)]`; pro BreastCancer (569 vzorků): U≈119, optimal_side≈10.9, rozsah [8, 14]; funkce `apply_dynamic_search_space(search_space, n_samples)` vrátí upravenou kopii bez mutace originálu; EA stále prohledává různé kvadratické velikosti v rozumném koridoru, eliminuje extremy (5×5 nebo 25×25) které jsou předem nevhodné.

48. **Graduated dead threshold implementován přes `compute_constraint_violation`** — pevný binární threshold `DEAD_NEURON_THRESHOLD=0.10` nahrazen dynamickým výpočtem v `compute_constraint_violation`; threshold = `clamp(1 - coverage_ratio/10, 0.3, 0.85)` kde `coverage_ratio = n_samples / (m*n)`; violation se počítá ve třech pásmech nad prahem: `dead_excess < 0.2` → `cv = excess * 1.5`, `dead_excess < 0.4` → `cv = excess * 2.5`, `dead_excess >= 0.4` → `cv = excess * 5.0`; tím se eliminovalo 100% penalizovaných map — EA nyní správně rozlišuje strukturálně nevhodnou kombinaci mapa/dataset od normálního výsledku.

43. **Constrained Dominance (Deb 2002) nahrazuje multiplicativní penalizaci** — multiplicativní penalizace `best_mqe *= factor` deformovala raw hodnoty a snižovala přesnost NSGA-II; přechod na constrained dominance (Deb et al. 2002): feasible řešení vždy dominuje infeasible bez ohledu na raw kvalitu; mezi dvěma infeasible rozhoduje skalár `constraint_violation` (menší CV dominuje); mezi dvěma feasible platí standardní Pareto dominance na 3 raw cílech; implementováno v `non_dominated_sort(objectives, violations)` přes helper `_dominates(objectives, violations, p, q)`.

44. **`constraint_violation` jako skalár agregující všechna omezení** — organizační omezení: `max(0, max(u_matrix_max, distance_map_max) - ORG_THRESHOLD)`; dead neuron omezení: `graduated_dead_cv` podle hloubky překročení dynamického prahu (faktor 1.5 / 2.5 / 5.0 podle pásma); součet tvoří jeden skalár `cv`; `cv = 0` → feasible, `cv > 0` → infeasible; magnitude CV umožňuje NSGA-II seřadit infeasible řešení od "lehce porušující" po "zcela nevhodné"; uloženo do `results.csv` a `pareto_front.csv` jako samostatný sloupec.

45. **`penalty_factor` zachován jako reference-only metadata** — po přechodu na constrained dominance se `penalty_factor` nepřikládá k žádné objective hodnotě; je logován jako `1.0 + constraint_violation` pro zpětnou kompatibilitu s `verify_ea_run.py` a pro diagnostiku míry porušení; `best_mqe` a `topographic_error` jsou nyní vždy raw naměřené hodnoty, nikdy penalizované; `penalty_reason` popisuje konkrétní příčinu (`org(u=...,d=...)`, `dead=X%(thresh=Y%)`).

---

## Diagnostika a správnost výsledků

26. **`normalize_weights_flag=True` vede ke 100% penalizaci** — ve všech 712 běhu s `normalize_weights_flag=True` byl výsledek penalizovaný (ratio ≥ 2); normalizace vah konzistentně produkuje špatnou organizaci mapy (u_matrix_max > 1 nebo distance_map_max > 1); hodnota `true` odebrána z search space.

27. **`map_size min=5` způsobuje dominanci penalizovaných běhů** — 79 % map 5×5 je penalizovaných; mapy jsou příliš malé pro organizaci (U-Matrix/Distance thresholds > 1); 48 % všech evaluací tvořily mapy 5×5; doporučeno zvýšit `map_size min` na 10.

28. **`max_archive_size=10` příliš malý — dobré výsledky vytlačeny crowding distance capem** — penalizovaná rychlá řešení (5×5, čas 3–5 s) obsazovala krajní pozice na TIME ose s vysokou crowding distance a vytlačovala validní `ratio < 1` řešení z archivu; z 39 unikátních "good" UID (ratio < 1) zůstaly v závěrečném archivu pouze 4; větší archiv problém neřeší — přibyde jen více balastu; kořen odstraněn odebráním `duration` jako cíle (bod 32) a přechodem na raw objectives (bod 34).

29. **`np.random.seed()` aplikován po inicializaci vah — nondeterministická evaluace** — v `som.py` byla linie `self.weights = np.random.rand(...)` volána PŘED `np.random.seed(self.random_seed)`; váhy byly vždy náhodné bez ohledu na seed; stejný config evaluovaný dvakrát dával různé výsledky; v EA běhu 30×50 vzniklo 45 duplicitních UID v `results.csv` s různými hodnotami; opraveno prohozením pořadí.

30. **`dead_neuron_ratio` chyběl v `pareto_front_log.txt`** — čtvrtý cíl NSGA-II nebyl logován; ověření dominance bylo neúplné (3/4 cíle); přidáno pole `Dead=` do `log_pareto_front()`.

31. **Diagnostický nástroj `verify_ea_run.py`** — vytvořen nástroj `app/tools/verify_ea_run.py` pro ověření výsledků EA běhu; sekce: přehled, penalizace podle velikosti mapy, vývoj archivu po generacích, elitismus/UID tracking, ověření dominance závěrečného archivu (4 cíle), analýza crowding ejection, korelace parametrů s penalizací, doporučení.

49. **`epoch_multiplier min=0` zničil 2hodinový běh EA** — search space povoloval hodnotu 0; repair clampal na `max(1.0, ...)`, tedy každý SOM trénoval jen ~569 iterací (1× n_samples); SBX z rodičů se stejnou hodnotou 1 generuje potomky blízko 1; bez `duration` jako Pareto cíle nebyl žádný tlak EA opustit tuto hodnotu; všechny generace konvergovaly na `epoch_multiplier=1` a SOMs byly systematicky podtrénované; opraveno: `min` v search space configu nastaveno na 5 (pro BreastCancer → ~2845 iterací minimálně); repair opraven z `max(0.1, ...)` na `max(1, int(...))` aby zachoval celočíselný typ.

50. **`epoch_multiplier` range nerespektuje velikost datasetu** — fixní rozsah (min=5, max=30) je vhodný pro 569 vzorků (~2845–17070 iterací), ale pro 15 000 vzorků dává multiplier=30 → 450 000 iterací (hodiny trénování); EA nemá způsob jak toto zjistit samo; rozšířena funkce `apply_dynamic_search_space` aby přepočítávala rozsah `epoch_multiplier` z cílového počtu celkových iterací `[3 000, 20 000] / n_samples`; pro 569 vzorků → `[5, 35]`, pro 15 000 vzorků → `[1, 3]`; hodnoty `min`/`max` v configu jsou nyní zbytečné — dynamic funkce je vždy přepíše; proto jsou odstraněny z `config-ea.json` a nahrazeny komentářem o dynamické kalibraci.

51. **Duplikátní potomci zahazováni po generování — skutečná populace menší než `population_size`** — intra-generační deduplikace probíhala až po vygenerování celé populace; duplicitní jedinci (stejné UID po SBX + repair) se zahazovali bez náhrady; efektivní populace mohla být výrazně menší než nastavená hodnota; opraveno přesunutím deduplikace dovnitř generovací smyčky — potomci se generují dokud není nashromážděno přesně `population_size` unikátních jedinců nebo není vyčerpán limit `population_size × 3` crossoverů; při sytosti search space se pokračuje s menší populací a vypíše informační hláška.

52. **EA konvergovala k malému LR a malému poloměru — přeskočení globální organizační fáze** — analýza archivu z běhu 80×15 odhalila, že medián `start_learning_rate` klesl z 0.13 na 0.03 a medián `start_radius_init_ratio` z 0.12 na 0.07 již po gen 5; EA se naučila, že malý počáteční radius a LR dává lepší `mqe_improvement_ratio` (rychlá lokální konvergence), ale SOM nikdy neprojde globální organizační fází potřebnou pro smysluplnou topologii; log scale vzorkování ani Gaussiánová sousedská funkce (která pracuje spojitě, ne binárně) toto nezachrání — problém leží v selekčním tlaku; opraveno zvýšením dolních mezí: `start_learning_rate min 0.01→0.5` (standardní SOM rozsah), `start_radius_init_ratio min 0.05→0.5` (zajistí pokrytí alespoň poloviny mapy, tj. `0.5 × max(m,n)` neuronů).

53. **Post-konvergenční generace produkují biased trénovací data pro NN** — po konvergenci EA (~gen 5 v běhu 80×15) generuje SBX potomky v úzké oblasti parametrového prostoru kolem konvergovaného bodu; generace 6–15 přidávají ~800 evaluací, které jsou si navzájem podobné; MLP trénovaná na těchto datech by se přeučila na konvergovaný region a špatně generalizovala na zbytek prostoru; prodloužení běhu na 15+ generací tedy zvyšuje počet evaluací, ale snižuje jejich diverzitu; řešení: místo jednoho dlouhého běhu spustit více nezávislých kratších běhů s různými random seedy — každý start exploruje prostor z jiné náhodné populace.

54. **Multi-seed strategie pro sběr NN trénovacích dat** — zvolena strategie `5 × 50 × 6` (seedy × populace × generace) místo jednoho běhu 80×15; fixní seedy `[42, 1337, 7, 101, 2026]` definovány v `EA_SETTINGS.seeds`; každý seed dostane vlastní složku `results/seed_{seed}/`; preprocessing dat a kalibrační sonda proběhnou jednou (sdílené pro všechny seedy, výsledky v `results/`); `main()` iteruje přes seeds, nastavuje `FIXED_PARAMS.random_seed` a resetuje `ARCHIVE` mezi běhy; celkem ~1500 evaluací na dataset s výrazně lepším pokrytím parametrového prostoru než 80×15.

---

## Integrace NN modelů (MLP + LSTM) — Phase 2

55. **LSTM early stopping bylo gated za `save_checkpoints` — LSTM nikdy nespustilo** — checkpoint blok v `som.py` byl podmíněn `if self.save_checkpoints and iteration % checkpoint_interval == 0:`; LSTM callback `lstm_early_stop_fn` byl volaný uvnitř tohoto bloku; pokud `save_checkpoints=False` nebo nebyl splněn interval, LSTM se nikdy nevyvolalo; opraveno přidáním samostatného seznamu `lstm_checkpoints = []` nezávislého na `checkpoints[]`; podmínka pro budování checkpointu rozšířena na `if self.save_checkpoints and ... or lstm_early_stop_fn is not None:` přičemž LSTM vždy dostane přehledný stream checkpointů bez ohledu na nastavení ukládání souborů.

56. **LSTM se spustilo po 2 checkpointech — příliš brzy (< 20 % trénování)** — model byl natrénován na K-prefix oknech s K ∈ {20,30,...,70}% délky sekvence; minimální K = 20 %, ale LSTM callback se aktivoval po `len(checkpoints) >= 2` tj. po ~2 % trénování; predikce ze 2 % dat byly nesmyslné (model viděl takové situace jen náhodně); opraveno zavedením `lstm_min_checkpoints = max(2, mqe_evaluations_per_run // 5)` = 60 pro 300 evaluací (= 20 % trénování); LSTM callback se volá pouze po nashromáždění dostatečného prefix okna.

57. **`quality_score` formule invertovaná — LSTM zastavovalo dobré běhy** — `should_stop_early` v `nn_integration.py` počítalo `quality_score = final_mqe_ratio + te + dead*0.5` kde `final_mqe_ratio` je `raw_mqe_improvement_ratio`; tato hodnota je **vyšší = lepší** (ratio ≈ 0.7 → 30% MQE improvement = špatné; ratio ≈ 0.95 → 5% improvement = výborné); přičtením raw ratia místo jeho invertování skript zastavoval konfigurace s vysokým (dobrým) mqe_improvement_ratio; opraveno na `quality_score = (1.0 - final_mqe_ratio) + te + dead*0.5` — nižší = lepší, LSTM zastavuje pouze skutečně slabé konfigurace; práh `lstm_quality_threshold = 0.75` kalibrován na p75 distribuce quality_score z testovacích dat.

58. **MLP filtr invertovaný — přeskakoval dobré konfigurace** — podmínka v `ea.py` pro pre-screen MLP byla `if pred_mqe > threshold: skip`; `pred_mqe` je predikovaný `raw_mqe_improvement_ratio` kde **vyšší = lepší** (ratio 0.9 = dobré); podmínka `> 0.5` přeskakovala konfigurace s vysokým predikovaným zlepšením a propouštela slabé; opraveno na `if pred_mqe < threshold: skip` — přeskočit pouze když model predikuje nízké zlepšení (špatná konfigurace); chyba existovala od první implementace filtru a způsobovala opačný efekt optimalizace.

59. **MLP metadata cesta selhávala pro `mlp_latest.keras` stable path** — `_load_mlp` v `nn_integration.py` odvozoval cestu k metadatům z cesty modelu pomocí `model_path.replace('_best.keras', '_metadata.json')`; pro stable path `mlp_latest.keras` (bez suffixu `_best`) nahrazení nenašlo nic a cesta zůstala `mlp_latest.keras`; pokus o `json.load()` na binárním `.keras` souboru způsobil `'utf-8' codec can't decode` nebo `JSONDecodeError`; MLP se deaktivovalo i při existujícím modelu a metadatech; opraveno seznam kandidátních cest: `['mlp_latest_metadata.json', model_path.replace('_best.keras','_metadata.json'), 'metadata.json']` — první existující soubor se použije.

---

## Phase 3 — LSTM dynamický kontrolér

60. **Lambda vrstvy v `model_controller.py` nebyly serializovatelné** — první architektura používala `layers.Lambda(lambda ctx, seq: tf.repeat(...))` pro tiling kontextu a `layers.Lambda(lambda x: x * 1.0 + 0.5)` pro škálování výstupu; Keras odmítl deserializovat takový model z `.keras` souboru s chybou `Lambda whose function is a Python lambda cannot be saved` nebo `NotImplementedError: output shape`; pokus o `keras.config.enable_unsafe_deserialization()` obešel první chybu, ale narazil na druhou (Keras nedokáže odvodit výstupní tvar Lambda vrstvy při načítání); opraveno náhradou obou Lambda vrstev za správné `layers.Layer` podtřídy `_TileContext` a `_ScaleSigmoid` s implementací `compute_output_shape()` a `get_config()`; model byl po opravě přetrénován (108 epoch, MAE=0.036).

61. **`sample_weight` shape mismatch při seq2seq tréninku Phase 3** — při `model.fit()` s `return_sequences=True` produkuje Keras interní loss tvar `(batch, time)` před redukcí; předání `sample_weight=(N,)` způsobilo `ValueError: Dimensions must be equal, but are 307 and 8`; opraveno tílováním advantage array: `adv_tiled = np.tile(adv[:, np.newaxis], (1, T)).astype(np.float32)` — tvar se změní na `(N, T)` a broadcasting projde; toto platí jak pro `model.fit()` tak `model.evaluate()`.

62. **`_load_lstm_controller` chyběly `custom_objects` — model by selhal při načítání** — `nn_integration.py` volal `_keras.models.load_model(model_path)` bez předání `custom_objects`; po přetrénování modelu s `_TileContext` a `_ScaleSigmoid` vrstvami by EA při `use_lstm_controller: true` selhala s `Unknown layer: _TileContext`; opraveno přidáním importu `from model_controller import _TileContext, _ScaleSigmoid` a předáním `custom_objects={'_TileContext': _TileContext, '_ScaleSigmoid': _ScaleSigmoid}` do `load_model()`.

68. **Phase 2 confusion matrix 100 % přesnost — artefakt nevyvážené testovací sady** — model dosáhl `Acc=1.0, Prec=1.0, Rec=1.0`; zdánlivě perfektní výsledek způsoben tím, že testovací sada neobsahuje žádného jedince s `quality_score < 0.75` — všechna testovací individua patří do třídy "True STOP"; model predikoval STOP pro vše a byl správně; threshold 0.75 je špatně kalibrovaný vůči skutečné distribuci quality score — pokud má velká část EA individuí QS > 0.75, není threshold informativní dělicí čárou; problém není nedostatek dat z "neúspěšných běhů" (ta existují), ale nerovnováha tříd v uid-level splitu a špatná volba thresholdu; oprava: kalibrovat threshold jako percentil distribuce quality score na trénovací sadě (např. 60. percentil), přidat stratifikaci distribuce tříd při uid-level splitu.

69. **Topographic error — strukturálně slabší predikce LSTM Phase 2** — scatter analýza ukázala `r=0.499` pro TE vs `r=0.970` pro MQE improvement ratio; residua vykazují bimodalitu (shluky kolem −0.05 a +0.03); příčina je strukturální: malé a velké mapy mají přirozeně různé TE režimy (map_size není vstupem LSTM sekvence, jen kontextem), TE navíc klesá skokovitě (diskrétní přepínání topologie) zatímco MQE klesá spojitě — LSTM na TE funguje hůř ze strukturálních důvodů; rozhodnutí: pro early stopping se TE váha může snížit v quality score pro rané prefixy (20–30 %), kde je predikce TE nejméně spolehlivá; aktuálně ponecháno s plnou váhou jako otevřená otázka pro Phase 2 rozšíření.

70. **Phase 3 perturbační funkce ignorovaly fyzikální meze — radius > mapa, LR rostl** — původní `make_random_perturb_fn`, `make_lr_only_fn`, `make_radius_only_fn` vracely faktory ±25 % bez znalosti aktuálních hodnot LR a radiusu; kumulativní drift v `som.py` (clamp [0.05, 5.0]) vedl k radiusu 80+ u map 20×20, LR rostlo místo klesání; trénovací data Phase 3 obsahovala fyzikálně nesmyslné trajektorie; opraveno: stará trojice funkcí nahrazena jednou `make_constrained_perturb_fn(rng, max_radius, max_lr, apply_lr, apply_radius)`, která čte `checkpoint['learning_rate']` a `checkpoint['radius']` (aktuální skutečné hodnoty), vypočítá navrhovanou novou absolutní hodnotu, clipuje ji na fyzikální meze `lr ∈ [1e-4, start_lr]`, `radius ∈ [1.0, max(map_m, map_n)]` a vrátí faktor `new / current`; worker předává `max_radius` a `max_lr` z `row_dict` každého individua.

71. **Phase 3 trénovací data — 60 % timestepů má target = 1.0, model kolabuje** — `PERTURB_PROB=0.4` znamená, že 60 % timestepů v každé trajektorii nemá perturbaci (`lr_factor=1.0`, `radius_factor=1.0` přesně); model se naučí predikovat 1.0 pro vše a dostane nízkou loss; zdánlivě dobrá korelace `r≈0.90` je artefakt těchto shod, ne reálná regresní schopnost; advantage-weighted loss (per-trajektorie) situaci nezachraňuje — nepřeturbované a přeturbované timestepy mají uvnitř jedné trajektorie stejnou váhu; oprava: v `prepare_phase3_dataset.py` filtrovat pouze timestepy kde `abs(lr_factor − 1.0) > ε` nebo `abs(radius_factor − 1.0) > ε`; alternativně zvýšit `PERTURB_PROB` z 0.4 na 0.7–0.8 při regeneraci dat; r a MAE měřit výhradně na přeturbovaných timestepech.

75. **Phase 3 advantage — statické váhy α, β způsobují dominanci jedné dimenze přes datasety** — `advantage = 1.0·δMQE + 0.5·δTE + 0.3·δDead` s pevnými koeficienty: pokud δMQE nabývá hodnot 0.1–0.3 a δTE 0.001–0.01 na menším datasetu, δTE prakticky nepřispívá; na jiném datasetu se poměry otočí; LSTM dostává nekonzistentní gradient přes datasety; opraveno Z-score normalizací v `compute_advantages()` v `prepare_phase3_dataset.py`: každá složka se vydělí svým globálním σ přes všechny záznamy ze všech seedů a datasetů před součtem; výsledek `(δmqe/σ_mqe) + (δte/σ_te) + (δdead/σ_dead)` má konzistentní měřítko; σ hodnoty se vypisují při přípravě datasetu pro diagnostiku.

72. **Phase 3 evaluace — padding zeros v `y_test` vytvářely falešný spike residuí +0.5** — `pad_ragged()` v `prepare_phase3_dataset.py` doplňuje kratší sekvence nulami; padded timestepy mají `y = [0.0, 0.0]`; model pro ně predikuje ~0.5 (spodní limit sigmoid výstupu); residuum `0.5 − 0.0 = +0.5` přesně tvoří izolovaný shluk v `residuals_p3.png`; nejde o stav kde SOM "vypíná učení", ale o čistý artefakt paddingu; oprava: při evaluaci a vizualizaci maskovat padded pozice `mask = y_test[j].sum(axis=1) != 0.0`; při tréninku přidat `Masking(mask_value=0.0)` vrstvu aby padded pozice nedostávaly gradient.

73. **Phase 3 `advantage_p3.png` — Q1 prázdný při many-tie hodnotách 0.0** — advantage je `max(0, delta_mqe) / max_delta`; záporné delta jsou ořezány na 0; pokud velká část trajektorií má `advantage=0.0` přesně, quartilové hranice vychází `(adv >= 0.0) & (adv < 0.0)` = prázdná množina; `n=0` pro Q1 je bug ve výpočtu, ne absence negativních příkladů; oprava: binovat advantage přes `np.array_split(np.argsort(adv), 4)` — vždy stejně velké kvartily bez závislosti na distribuci hodnot.

74. **`_TileContext` custom Keras vrstva chyběla v `custom_objects` při načítání v `visualize_model.py`** — `load_model_and_scaler()` volal `tf.keras.models.load_model(model_path)` bez `custom_objects`; Phase 3 controller model obsahuje vrstvy `_TileContext` a `_ScaleSigmoid` definované v `src/model_controller.py`; načítání selhalo s `TypeError: Could not locate class '_TileContext'`; opraveno: pokud `'controller' in model_path`, před načtením se importují obě vrstvy z `model_controller` a předají jako `custom_objects={'_TileContext': ..., '_ScaleSigmoid': ...}`.

76. **`load_splits` vracel masky interně ale nepředával je — `NameError: msk_train is not defined`** — `load_splits()` v `train_phase3.py` načítal `msk_train/val/test` uvnitř `_load()` a korektně je rozkládal, ale return tuple na řádku 72–75 obsahoval jen `(X, ctx, y, adv, scaler, meta)` bez masek; volající `train()` pak rozbaloval stejně starý tuple bez masek, přičemž `msk_train` nikdy nevznikl; první pokus o `_combined_weight(adv_train, msk_train)` skončil `NameError`; opraveno rozšířením obou tuple na `(X, ctx, y, adv, msk, ..., scaler, meta)`.

77. **Phase 3 prediction display ukazoval padding pozici — target vždy (0.000, 0.000)** — evaluační smyčka na konci `train_phase3.py` používala `last = -1` jako "last valid checkpoint"; sekvence jsou padded na délku nejdelší trajektorie (588 timestepů); pro kratší trajectorie je pozice -1 čistý padding s `y = [0.0, 0.0]`; výpis tak vždy ukazoval target (0.000, 0.000) bez ohledu na skutečné cílové hodnoty; model predikoval ≈1.001, padding-target je 0.000 — zdálo se, že model predikuje špatně, ale chyba byla v displayi; opraveno nalezením posledního aktivního (perturbed) timestepsoru přes `np.argwhere(msk_test[i] > 0.0)` a použitím jeho indexu; výpis nyní zahrnuje i `t=N` — číslo timestepsu pro diagnostiku délky sekvencí.

78. **Phase 3 advantage clipping na 0 → 61 % trajektorií má nulovou váhu → model kolabuje na 1.001** — `compute_advantages()` v `prepare_phase3_dataset.py` počítala Z-score normalizovaný composite advantage a pak aplikovala `np.clip(raw, 0.0, None)` a dělila maximem; záporné advantages (= trajektorie horší než baseline) dostaly váhu 0.0; při trénování s 560 trajektoriemi bylo nonzero jen 216 (38.6 %); z těch 26 % timestepů bylo aktivních (perturb_mask=1); efektivně model dostával gradient z ~10 % všech timestep-batch kombinací; zbylých 90 % nepřispívalo k losu → model se naučil predikovat střed výstupního rozsahu (1.001) jako optimální konstantu; `train_mae=0.46` přes celý trénink; opraveno nahrazením clipping → min-max normalizace: `adv = (raw - raw.min()) / (raw.max() - raw.min())`; nejhorší trajektorie dostane váhu 0.0, nejlepší 1.0, všechny ostatní mají nenulovou váhu; mean advantage přejde z 0.061 na ≈0.5; všechny trajektorie přispívají k učení se správnou relativní vahou.

63. **Phase 3 model kolaboval na konstantní výstup ≈ 0.997** — analýza modelu natrénovaného na 24 trajektoriích (45 s baseline) odhalila: predikovaný `lr_f` std=0.005 vs target std=0.079; Pearsonovo r ≈ −0.10; model MAE=0.038 horší než naivní baseline 1.0 (MAE=0.036); kořenová příčina: 70–75 % timestepů má target=1.0 (PERTURB_PROB=0.4 → 60 % checkpointů bez perturbace, zbytek baseline trajektorie); advantage-weighted loss nedokázal překonat tuto nevyváženost; slabý pozitivní signál existuje (Pearson advantage vs mean_pred_rad = +0.576); výsledek: controller funguje end-to-end technicky, ale reálný efekt je ≈ 8–11 % kumulativní odchylka v jednom směru pro všechny trajektorie bez adaptivity; oprava vyžaduje víc dat a/nebo přepracování tréninkového schématu (viz FR-LSTM-3.6, LSTM_DYNAMIC_CONTROL.md).

64. **Chybělo logování Phase 3 faktoru v `som.py` — nebylo vidět co controller dělá** — `som.py` aplikoval `lr_f` a `rad_f` na cumulative faktory, ale nic nelogoval; nebylo možné ověřit, zda controller funguje bez debuggeru; přidán log každých `max(1, mqe_evaluations_per_run // 10)` checkpointů: `LSTM ctrl @ XX%: step lr_f=... rad_f=... | cum_lr=... cum_rad=...`; přidán čítač `_ctrl_cp_count` a konstanta `_ctrl_log_every` k inicializaci před tréninkovou smyčkou.

---

## Pareto metriky — HV, Spacing, Spread

65. **`np.clip` bez normalizace měřítka zkresloval HV** — původní implementace používala `np.clip(objectives, 0.0, 1.1)` jako "normalizaci" před výpočtem hypervolumu; pokud `raw_mqe_ratio` nabývá hodnot 0.1–0.5 a `topo_error` 0.01–0.05, má mqe_ratio ~10× větší vliv na objem HV — dimenze s malým absolutním rozsahem jsou v HV neviditelné; per-generace min-max normalizace by měřítka srovnala, ale zlomila by cross-generace srovnatelnost (reference bod [1.1,1.1,1.1] by měnil smysl každou generaci); opraveno globálním running min/max (`_OBJ_RUNNING_MIN`, `_OBJ_RUNNING_MAX`) aktualizovaným každou generaci přes všechny feasible solutions v daném sedu; normalizace `(x - running_min) / span` zajišťuje konzistentní HV přes celý běh; running stats se resetují na začátku každého sedu.

66. **HV a Spacing nebyly implementovány — chyběla per-generace metrika kvality fronty** — pro Phase 5 srovnání (statický vs dynamický LSTM schedule) je nutné sledovat, jak se kvalita Pareto fronty vyvíjí přes generace; implementovány funkce `_compute_pareto_metrics()` (pymoo HV s reference point [1.1,1.1,1.1] v normalizovaném prostoru + nearest-neighbor Spacing) a `_log_pareto_metrics()` (zapíše do `pareto_metrics.csv`); `log_pareto_front()` volá obě automaticky na konci každé generace; do HV a Spacing vstupují pouze feasible (nepenalizovaná) řešení; pymoo přidán do `python/requirements.txt`.

67. **Spacing bez Maximum Spread nezachycuje pokrytí fronty** — Spacing měří rovnoměrnost rozložení bodů (0 = dokonale uniformní), ale nic neříká o rozsahu; fronta s dokonalým Spacing může pokrývat jen 1 % objective prostoru (body jsou rovnoměrně u sebe); pro Phase 5 analýzu je důležité vědět, jak široké spektrum trade-offů fronta nabízí; přidán Maximum Spread per dimenzi (`max − min` v normalizovaném prostoru) jako `spread_mqe`, `spread_te`, `spread_dead` do `pareto_metrics.csv`; hodnota blízko 1.0 znamená, že fronta pokrývá celý pozorovaný rozsah dané dimenze.

---

## Checkpointy a LSTM data

23. **Řídké checkpointy při dlouhém tréninku SOM** — 15 000 iterací a 25 checkpointů = 1 checkpoint na 600 iterací; příliš málo dat pro LSTM trénink; přidán flag `checkpoint_every_mqe` pro uložení při každém výpočtu MQE (~500 checkpointů na běh).

24. **Výpočet topographic error způsoboval zásadní zpomalení logování** — Python smyčka přes všechny vzorky; nahrazeno NumPy vektorizací s broadcasting; zrychlení ~100×.

25. **Variabilní délka LSTM sekvencí kvůli early stopping** — běhy s early stopping mají méně checkpointů; `collect_training_data.py` paduje/ořezává na fixní délku; default `--checkpoints 10` byl nevhodný pro reálná data (~500 checkpointů); opraveno na `--checkpoints 500`.
