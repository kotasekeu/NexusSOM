## 50 klíčových požadavků na EA pro SOM

Legenda: ✅ implementováno · ⚠️ částečně · ❌ chybí

---

### Architektura a Genetický kód (Reprezentace)

1. ✅ **Definice genomu:** Pole reálných/celočíselných hodnot reprezentující parametry SOM (radius, learning rate, epochy, $g$-modifikátor, batch size).
   > `genetic_operators.py` + JSON config `SEARCH_SPACE` — genome je slovník s typovanými geny (`float`, `int`, `categorical`, `discrete_int_pair`).

2. ✅ **Typy genů:** Kombinace spojitých (float pro $LR$, $g$) a diskrétních (int pro epochy, velikost mřížky) proměnných.
   > `random_config_continuous()` v `genetic_operators.py` obsluhuje všechny 4 typy. `map_size` je `discrete_int_pair`, LR je `float` s log-scale podporou.

3. ✅ **Validace mezí (Bounds check):** Každý gen musí striktně dodržovat min/max hranice (např. $LR \in \langle0.01; 1.0\rangle$).
   > `validate_and_repair()` v `ea.py` — opravuje pořadí LR, batch, radius. `np.clip()` v `sbx_crossover()` a `polynomial_mutation()` v `genetic_operators.py`.

4. ✅ **Inicializace populace:** Generování diversifikované startovní sady jedinců (např. pomocí pseudonáhodného rozdělení nebo LHS).
   > `run_evolution()` řádek 850: `[validate_and_repair(random_config_continuous(search_space)) for _ in range(population_size)]`

5. ✅ **Konfigurace velikosti populace:** Dynamicky nastavitelný počet jedinců v generaci podle dimenzionality problému.
   > `EA_SETTINGS.population_size` v configu. `apply_dynamic_search_space()` dynamicky zužuje `map_size` a `epoch_multiplier` bounds podle `n_samples` (Vesantova heuristika).

---

### Fitness Funkce a Selekce

6. ✅ **MQE integrace:** Výpočet průměrné kvantizační chyby z finální mapy jako primární fitness složka.
   > `raw_mqe_improvement_ratio` je 1. cíl NSGA-II. Computed v `evaluate_individual()` přes `som.train()` a checkpoint baseline.

7. ✅ **Výpočet Topographic Error (TE):** Měření podílu dat, kde první a druhý nejbližší neuron nesousedí (klíčová obrana proti kolapsu topologie).
   > `som.calculate_topographic_error(data, mask=ignore_mask)` v `evaluate_individual()`, předáno jako 2. cíl NSGA-II.

8. ❌ **Penalizace výpočetního času:** Zahrnutí časové náročnosti tréninku do fitness pro eliminaci neefektivně vysokých epoch.
   > `training_duration` je logováno do `results.csv`, ale NENÍ součástí Pareto cílů. Tři cíle jsou: MQE ratio, TE, Spearman ρ.

9. ❌ **Normalizace vah fitness:** Možnost vážit MQE vs. TE vs. čas pro různé optimalizační cíle.
   > Implementována normalizace cílů na [0,1] přes `_normalize_objectives()` (running min/max). Chybí uživatelsky konfigurovatelné váhy jednotlivých cílů — všechny jsou rovnoměrné.

10. ✅ **Turnajová selekce (Tournament):** Výběr rodičů porovnáním náhodného podvzorku populace (stabilní tlak na selekci).
    > `tournament_selection()` v `ea.py` — srovnává rank + crowding distance. Velikost turnaje `tournament_k` konfigurovatelná přes `GENETIC_OPERATORS.tournament_k`.

11. ❌ **Ruletová selekce (Roulette wheel):** Pravděpodobnost výběru přímo úměrná fitness (pro vyšší diverzitu).
    > Není implementováno. Používá se výhradně turnajová selekce.

12. ✅ **Elitářství (Elitism):** Automatické zachování $N$ nejlepších jedinců do další generace bez mutace.
    > `ARCHIVE` uchovává celou Pareto frontu (rank=0). `combined_population = evaluated_population + ARCHIVE` kombinuje aktuální populaci s archivem každou generaci.

13. ⚠️ **Detekce invalidních jedinců:** Při selhání tréninku SOM (NaN hodnoty) přidělit nejhorší možnou fitness.
    > Jedinci vyfiltrovatelní MLP pre-screenem dostávají `constraint_violation=999.0`. Jedinci které selžou výjimkou jsou z populace vynechání (ne penalizovaní nejhorší fitness). Chybí přidělení worst-case fitness při technickém selhání evaluace.

---

### Genetické operátory (Křížení a Mutace)

14. ❌ **Jednobodové křížení (Single-point crossover):** Výměna bloků genů mezi dvěma rodiči v náhodném bodě.
    > Není implementováno. Používá se SBX (continuous) + uniform (categorical).

15. ✅ **Aritmetické křížení:** Výpočet váženého průměru genů rodičů pro spojité parametry ($LR, g$).
    > SBX (`sbx_crossover()` v `genetic_operators.py`): `child1 = 0.5 * ((1+β)·p1 + (1-β)·p2)` — aritmetické blending s distribučním indexem eta. Podporuje log-scale pro LR.

16. ✅ **Uniformní křížení:** Každý gen potomka je vybrán z jednoho z rodičů s pravděpodobností 50 %.
    > `crossover_mixed()` pro `categorical` parametry: `if random.random() < 0.5: child1[key] = parent1[key]`. Pro kontinuální geny používá SBX.

17. ❌ **Gaussovská mutace:** Přičtení malé hodnoty z normálního rozdělení ke spojitým genům pro jemné doladění.
    > Není implementováno. Používá se polynomiální mutace (`polynomial_mutation()` v `genetic_operators.py`), nikoli Gaussova.

18. ⚠️ **Uniformní mutace:** Náhodná změna hodnoty genu v rámci definovaných mezí (pro únik z lokálního minima).
    > Pro `categorical` geny: `random.choice(spec['values'])` — uniformní náhrada. Pro spojité geny: polynomiální mutace (ne uniformní). Chybí čistá uniformní mutace pro float parametry.

19. ❌ **Adaptivní mutace:** Zvyšování míry mutace, pokud populace začíná stagnovat.
    > Není implementováno. `mutation_prob` je pevné číslo z configu, nemění se za běhu.

20. ✅ **Řízení pravděpodobnosti operátorů:** Nezávislé nastavení šance na křížení ($P_c \approx 0.8$) a mutaci ($P_m \approx 0.1$).
    > `GENETIC_OPERATORS.mutation_prob`, `sbx_eta`, `mutation_eta`, `tournament_k` v JSON configu. SBX má interní 50% skip. `mutation_prob` předáno do `mutate_mixed()`.

---

### Řízení konvergence a Ukončení

21. ✅ **Maximální počet generací:** Hard stop po dosažení limitu epoch algoritmu.
    > `for gen in range(generations)` v `run_evolution()`. Hodnota z `EA_SETTINGS.generations`.

22. ❌ **Stagnace fitness:** Ukončení, pokud se nejlepší fitness nezlepší o $\epsilon$ po $X$ generací.
    > Není implementováno na úrovni EA. `max_epochs_without_improvement` a `early_stopping_window` v FIXED_PARAMS platí pro vnitřní SOM trénink, ne pro EA.

23. ❌ **Diverzita populace:** Sledování rozptylu fitness; stop při totálním splynutí populace (předčasná konvergence).
    > Crowding distance udržuje diverzitu uvnitř NSGA-II, ale není implementována stop podmínka při kolapsu diverzity. `pareto_metrics.csv` loguje spacing, ale nevyvolává stop.

24. ❌ **Ukládání checkpointů:** Serializace stavu EA po každé generaci pro možnost obnovení při pádu.
    > Stav EA (populace, ARCHIVE, random state) není serializován. `pareto_front.csv` a `results.csv` jsou inkrementálně zapisovány, ale nestačí k plné obnově. SOM-level checkpointy (`save_checkpoints: True`) jsou nezávislé.

25. ⚠️ **Archivace historicky nejlepšího:** Nezávislé uložení absolutně nejlepšího nalezeného jedince napříč všemi generacemi.
    > `ARCHIVE` uchovává celou finální Pareto frontu. `pareto_front.csv` loguje stav fronty po každé generaci. `log_final_best()` je definováno v `ea.py`, ale není voláno v `main()` ani `run_evolution()` — soubor `final_best.txt` se nezapisuje.

---

### Datová efektivita a Škálovatelnost

26. ❌ **Sub-sampling pro fitness:** Trénování SOM během fitness hodnocení na reprezentativním vzorku dat (např. 10–20 %) pro rapidní zrychlení EA.
    > `DATA_PARAMS.sample_size` funguje pouze pro syntetická data. Pro reálná data se používá celý dataset (`loaded_data = np.load(training_data_path)`). Sub-sampling pro zrychlení EA evaluací chybí.

27. ✅ **Paralelizace hodnocení:** Spouštění fitness funkcí pro různé jedince v populaci asynchronně na více jádrech CPU.
    > `Pool(processes=min(10, cpu_count(), len(population)))` + `pool.apply_async(evaluate_individual, ...)` v `run_evolution()`.

28. ✅ **Determinismus (Seeding):** Možnost nastavit pevný random seed pro replikovatelnost celého běhu EA.
    > `FIXED_PARAMS.random_seed` předáno do SOM. Multi-seed runs přes `EA_SETTINGS.seeds` — každý seed dá nezávislý běh v `seed_{N}/`.

29. ✅ **Nezávislost na typu dat:** EA nesmí reflektovat význam dat, pracuje pouze se strukturou matice.
    > EA pracuje výhradně s numpy array. Preprocessing (normalizace, ignore_mask) je oddělen v `som.preprocess`.

30. ⚠️ **Nízká paměťová stopa:** Průběžné promazávání instancí SOM z paměti po vyhodnocení jejich fitness.
    > SOM instance žijí v subprocesech — jsou uvolněny při ukončení procesu. `EVALUATED_CACHE` v hlavním procesu roste neomezeně (bez size limitu). `ARCHIVE` je periodicky deduplikován.

---

### Stabilita a Robustnost (Obrana proti chybám)

31. ✅ **Ochrana proti dělení nulou:** Ošetření mezních stavů v metrikách (např. prázdné neurony).
    > `max(1, map_m * map_n)` v `_dead_neuron_threshold()`, `if max_val - min_val < 1e-9: return value` v `polynomial_mutation()`, `if obj_range < 1e-8: continue` v `crowding_distance_assignment()`.

32. ✅ **Omezení explodujících parametrů:** Ochrana proti kombinacím parametrů, které vedou k $NaN$ nebo nekonečným vahám.
    > `validate_and_repair()` — opravuje pořadí LR/radius/batch, vynucuje `epoch_multiplier >= 0.1`, `num_batches >= 1`, `growth_g >= 1.0`. Kalibrační probe nastavuje `org_threshold`.

33. ✅ **Validace velikosti mřížky vůči datům:** EA nesmí navrhnout mřížku $100\times100$ pro dataset o 50 prvcích.
    > `apply_dynamic_search_space()` — Vesantova heuristika `U = 5·sqrt(n_samples)`, bounds `[0.7·sqrt(U), 1.3·sqrt(U)]`. `_dead_neuron_threshold()` kalibruje toleranci mrtvých neuronů k poměru map/data.

34. ✅ **Ošetření diskrétních skoků:** Zaokrouhlování a validace genů reprezentujících indexy nebo celá čísla.
    > `int(round(c1))` v `crossover_mixed()` a `mutate_mixed()` pro `int` a `discrete_int_pair`. `max(1, int(...))` pro `num_batches` v `validate_and_repair()`.

35. ✅ **Imunita vůči outlierům:** Fitness nesmí zkolabovat kvůli jednomu extrémnímu prvku v datech.
    > `ignore_mask` propagováno do SOM tréninku. Graduated dead neuron penalty v `compute_constraint_violation()` zabraňuje dominanci extrémních konfigurací.

---

### Integrace a Výstupy pro Neuronové Sítě

36. ⚠️ **Export optimální konfigurace:** Výstup ve formátu JSON/YAML připravený pro přímé sestavení produkčního SOM.
    > Všechny parametry jsou v `pareto_front.csv` a `results.csv`. Chybí explicitní export nejlepší konfigurace jako JSON soubor připravený pro `python run_som.py --config best.json`.

37. ✅ **Logování průběhu optimalizace:** Generování CSV s vývojem min/max/avg fitness v každé generaci.
    > `results.csv` (všichni jedinci), `pareto_front.csv` (Pareto fronta per generace), `pareto_metrics.csv` (HV, spacing, spread per generace), `status.csv`, `log.txt`.

38. ✅ **Generování matice příznaků (Feature matrix):** Příprava mapy pro transformaci dat do formátu vhodného jako vstup pro následné NN (MLP/CNN).
    > `weights.npy` per jedinec jako feature matrix. `maps_dataset/` s PNG mapami pro CNN. `combine_maps_to_rgb()` vytváří RGB obrazy pro CNN trénink. `nn_integration.py` obsluhuje MLP/CNN/LSTM pipeline.

39. ✅ **Konzistentní distribuce vah:** Výsledné parametry musí zaručit, že neurony pokrývají celý datový prostor rovnoměrně.
    > Dead neuron constraint (`_dead_neuron_threshold()`, `compute_constraint_violation()`) vyřazuje konfigurace s nedostatečným pokrytím jako infeasible v NSGA-II.

40. ⚠️ **Stabilita transformace:** Při opakovaném spuštění finálního SOM na stejných datech musí vzniknout topologicky identická reprezentace.
    > `random_seed` předáván do SOM. Multiprocessing (fork) však může způsobit nedeterministické chování při paralelní evaluaci. Přesná reprodukovatelnost není garantována.

---

### Uživatelská automatizace (Podpora pro no-code web)

41. ✅ **Heuristické nastavení mezí:** Automatické zúžení/rozšíření vyhledávacího prostoru na základě řádků a sloupců nahraného CSV.
    > `apply_dynamic_search_space()` — automaticky nastaví `map_size` bounds z `n_samples` a `epoch_multiplier` bounds z log-lineární interpolace přes empirické kotvy.

42. ❌ **Predikce času běhu:** Odhad celkové doby trvání EA na základě velikosti populace a datasetu.
    > Není implementováno.

43. ✅ **Bezobslužný režim:** Schopnost běžet na defaultní konfiguraci bez nutnosti uživatelského zásahu.
    > `python ea.py -i data.csv` — běží s `ea_config.py` defaults bez dalších parametrů.

44. ⚠️ **Asynchronní hlášení stavu:** Posílání procentuálního progresu do webového rozhraní.
    > `log_progress()` zapisuje do `progress.log`. `status.csv` aktualizován per jedinec. Chybí aktivní push do web rozhraní — pouze file-based polling.

45. ❌ **Detekce nevhodných dat:** Včasné zastavení EA, pokud vstupní data vykazují nulový rozptyl (konstantní sloupce).
    > `validate_input_data()` v `som.preprocess` provádí základní validaci. Explicitní EA-level detekce nulového rozptylu a časné zastavení před první generací chybí.

---

### Připravenost pro pokročilé techniky (Ablation & Extensions)

46. ⚠️ **Izolace operátorů:** Možnost vypnout křížení nebo mutaci pro účely ablation study.
    > `mutation_prob=0` efektivně vypne mutaci. Křížení nelze vypnout konfigurací — SBX má hardcoded 50% skip, ale žádný explicitní `crossover_prob=0` přepínač neexistuje.

47. ❌ **Podpora dynamické změny fitness:** Možnost za běhu přepínat mezi optimalizací čistě na MQE, TE nebo jejich kombinaci.
    > Tři cíle jsou hardcoded v `run_evolution()`. CNN lze přidat jako 4. cíl (`use_cnn_objective`), ale nelze za běhu přepínat mezi subsetem cílů.

48. ❌ **Sledování úspěšnosti mutací:** Statistiky, kolik mutací vedlo k lepší fitness (ladění genetického tlaku).
    > Není implementováno. `EVALUATION_STATS` sleduje cache hits/misses, ne efektivitu mutací.

49. ✅ **Kompatibilita s hybridním SOM:** Genom musí nativně podporovat specifické parametry hybridního režimu (např. $g$-modifier).
    > `growth_g` je součástí genomu (`SEARCH_SPACE`). `validate_and_repair()` explicitně ošetřuje jeho sémantiku (0 pro čistě lineární, ≥1.0 pro nelineární křivky).

50. ❌ **Možnost fixace genů:** Schopnost uživatelsky zamknout např. rozměr mřížky a nechat EA optimalizovat pouze učící křivky.
    > Není implementováno jako feature. Workaround: vyňmout parametr z `SEARCH_SPACE` a přesunout do `FIXED_PARAMS`, ale to vyžaduje ruční úpravu configu, ne config flag `locked: true`.
