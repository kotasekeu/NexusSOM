---

## Plán testování, Unit Testy a Ablation Study pro EA

### Unit Testy (Izolované testy komponent v Pythonu)

| ID | Testovaná komponenta | Ověřovaný scénář (Problém $\rightarrow$ Řešení v jedné větě) | Výsledná aserce (`assert`) |
| --- | --- | --- | --- |
| **U1** | Inicializace populace | Genom obsahuje hodnoty mimo definované min/max meze. <br>

<br>$\rightarrow$ Zkontroluj, že každý vygenerovaný jedinec striktně splňuje hardcoded boundary constraints. | `assert min_b <= gene <= max_b` |
| **U2** | Křížení (Crossover) | Potomek po aritmetickém křížení identických rodičů mutuje nebo vybočuje. <br>

<br>$\rightarrow$ Ověř, že křížení dvou stejných rodičů produkuje identického potomka. | `assert child == parent1` |
| **U3** | Gaussovská mutace | Mutace mění hodnoty skokově mimo rozumné okolí. <br>

<br>$\rightarrow$ Otestuj, že delta změny genu odpovídá nastavené směrodatné odchylce $\sigma$. | `assert abs(orig - mut) < 3 * sigma` |
| **U4** | Elitářství (Elitism) | Nejlepší jedinec z generace $G_n$ se ztratí v generaci $G_{n+1}$. <br>

<br>$\rightarrow$ Ověř, že jedinec s nejvyšší fitness je stoprocentně zkopírován do nové populace. | `assert best_Gn in population_Gn1` |
| **U5** | Turnajová selekce | Selekce vybírá nejhoršího jedince z turnaje. <br>

<br>$\rightarrow$ Zafixuj náhodný výběr a ověř, že vybraný vítěz má prokazatelně nejlepší fitness v dané sub-skupině. | `assert winner.fitness == max(sub_group)` |
| **U6** | Výpočet fitness (NaN) | Trénink SOM vygeneruje podtečení/přetečení ($NaN$). <br>

<br>$\rightarrow$ Ošetři výstup fitness funkce tak, aby při výskytu $NaN$ vrátila nejhorší možnou hodnotu (`inf`), místo pádu programu. | `assert fitness == float('inf')` |
| **U7** | Ukončení při stagnaci | Algoritmus běží dál i přes nulový posun ve fitness. <br>

<br>$\rightarrow$ Simuluj konstantní fitness po dobu $X$ generací a ověř, že se EA korektně předčasně ukončí. | `assert ea.has_stopped == True` |
| **U8** | Seedování (Reprodukovatelnost) | Dva po sobě jdoucí běhy EA na stejném seedu dají jiné výsledky. <br>

<br>$\rightarrow$ Nastav pevný `random.seed()` a ověř shodu výsledných fitness i genomů u obou nezávislých běhů. | `assert run1.best_genes == run2.best_genes` |

### Integrační a Systémové Testy (End-to-End)

* **Test zřetězení (Pipeline Integration):** Ověření, že JSON konfigurace vyexportovaná z EA modulu je bez jakékoliv úpravy validně parsována modulem SOM a ten podle ní úspěšně odstartuje trénink.
* **Test datové dimenzionality:** Spuštění EA nad extrémně širokým datasetem (např. 100+ sloupců) a extrémně dlouhým datasetem (10k+ řádků) pro ověření, že maticové operace ve fitness funkci nezpůsobí `MemoryError`.
* **Asynchronní stress test:** Spuštění EA s maximální paralelizací na všech dostupných CPU jádrech nad syntetickými daty pro detekci race-conditions (souběhů) při zápisu do sdílené historie generací.

### Metodika pro Ablation Study (Ověření přínosu komponent)

Pro provedení ablation study na standardních datasetech (např. *Swiss Roll*, *Iris*, *Wine*) odstavuješ jednotlivé komponenty algoritmu a měříš dopad na rychlost a kvalitu (MQE/TE):

1. **Ablace Topographic Error (TE):** Nastav váhu TE ve fitness funkci na 0 (optimalizuje se pouze MQE).
* *Očekávaný výsledek:* Algoritmus zkolabuje do lokálního minima (jako u Student Habits) – potvrdí se nutnost TE pro udržení topologie.


2. **Ablace Křížení (Crossover):** Nastav pravděpodobnost křížení $P_c = 0$ (běží pouze mutace – Random Walk).
* *Očekávaný výsledek:* Výrazné zpomalení konvergence, algoritmus potřebuje k nalezení optima násobně více generací.


3. **Ablace Mutace:** Nastav pravděpodobnost mutace $P_m = 0$ (běží pouze křížení).
* *Očekávaný výsledek:* Rychlá degradace diverzity populace a uvíznutí v prvním nalezeném lokálním optimu bez šance na únik.


4. **Ablace Sub-samplingu:** Vypni podvzorkování dat ve fitness (každá generace trénuje SOM na 100 % dat).
* *Očekávaný výsledek:* Drastický nárůst výpočetního času (např. $10\times$), přičemž výsledná kvalita nalezených hyperparametrů se liší zanedbatelně.

