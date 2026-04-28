# EA — Problémy a rozhodnutí

## Architektura a návrh

1. **Volba NSGA-II místo single-objective EA** — čtyři navzájem si konkurující cíle (MQE, TE, dead ratio, čas) nelze sloučit bez subjektivního váhování; Pareto fronta zachovává celé spektrum kompromisů.

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

## Checkpointy a LSTM data

23. **Řídké checkpointy při dlouhém tréninku SOM** — 15 000 iterací a 25 checkpointů = 1 checkpoint na 600 iterací; příliš málo dat pro LSTM trénink; přidán flag `checkpoint_every_mqe` pro uložení při každém výpočtu MQE (~500 checkpointů na běh).

24. **Výpočet topographic error způsoboval zásadní zpomalení logování** — Python smyčka přes všechny vzorky; nahrazeno NumPy vektorizací s broadcasting; zrychlení ~100×.

25. **Variabilní délka LSTM sekvencí kvůli early stopping** — běhy s early stopping mají méně checkpointů; `collect_training_data.py` paduje/ořezává na fixní délku; default `--checkpoints 10` byl nevhodný pro reálná data (~500 checkpointů); opraveno na `--checkpoints 500`.
