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

---

## Checkpointy a LSTM data

23. **Řídké checkpointy při dlouhém tréninku SOM** — 15 000 iterací a 25 checkpointů = 1 checkpoint na 600 iterací; příliš málo dat pro LSTM trénink; přidán flag `checkpoint_every_mqe` pro uložení při každém výpočtu MQE (~500 checkpointů na běh).

24. **Výpočet topographic error způsoboval zásadní zpomalení logování** — Python smyčka přes všechny vzorky; nahrazeno NumPy vektorizací s broadcasting; zrychlení ~100×.

25. **Variabilní délka LSTM sekvencí kvůli early stopping** — běhy s early stopping mají méně checkpointů; `collect_training_data.py` paduje/ořezává na fixní délku; default `--checkpoints 10` byl nevhodný pro reálná data (~500 checkpointů); opraveno na `--checkpoints 500`.
