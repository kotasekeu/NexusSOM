Slide 2 — Motivace: Proč optimalizovat SOM?
Obsah:
•	Komplexita prostoru: ~15 hyperparametrů (spojité, diskrétní i kategorické).
•	Selhání tradičních metod:
o	Grid Search: Nepoužitelný kvůli kombinatorické explozi a spojitým intervalům.
o	Ruční ladění: Časově náročné a neobjektivní.
•	Cíl optimalizace:
o	Autonomní nalezení Pareto-optimálních konfigurací.
o	Maximalizace kvality mapy při minimalizaci výpočetních nákladů.
•	Role v projektu: Generování kvalitních datových podkladů pro následný trénink neuronových sítí.
Komentář:
SOM je extrémně citlivý na nastavení. Víme, že zkoušet kombinace ručně je neefektivní. Proto volíme globální optimalizaci. Tento krok je kritický – bez precizně optimalizovaného SOM by celá následná struktura (MLP, CNN, LSTM) pracovala s nekvalitními daty. EA zde slouží jako "vrták", který prozkoumá terén, než tam postavíme neuronové modely.
 
Slide 3 — Proč vícekriteriální (multi-objective) optimalizace?
Obsah (na slide):
•	Absence jediného optima: Kvalitu SOM nelze popsat jedním číslem bez ztráty informace.
•	Konfigurace pro NexusSom (4 cíle):
1.	MQE (Mean Quantization Error): Přesnost reprezentace dat (minimalizovat).
2.	Topografická chyba (TE): Zachování topologie (minimalizovat).
3.	Doba trénování: Časová efektivita (minimalizovat).
4.	Poměr mrtvých neuronů: Využití kapacity mřížky (minimalizovat).
•	Výstup: Pareto fronta – sada kompromisních řešení namísto jednoho výsledku.
Komentář (pro řečníka):
V reálné praxi vždy řešíme trade-off. Chceme mapu rychle, nebo ji chceme mít velmi přesnou? NSGA-II nám neřekne, co je "nejlepší", ale nabídne nám hranici možného. Z této fronty pak automatizovaně vybíráme kandidáty pro trénink dalších modulů. Pátým, experimentálním kritériem, je vizuální skóre z CNN, které se snaží kvantifikovat to, co analytik vidí okem.

 
Slide 4 — Komparativní analýza: Proč NSGA-II?
Obsah (na slide):
•	Srovnání s jednokriteriálními metodami:
o	Tabu Search / Horolezecký algoritmus: Riziko uvíznutí v lokálním optimu a neschopnost řešit konfliktní cíle (např. přesnost vs. čas).
•	Srovnání s ostatními populací řízenými EA:
o	Diferenciální evoluce (DE): Vhodná pro spojité problémy, obtížná adaptace na diskrétní parametry (např. topologie).
o	PSO (Particle Swarm Optimization): Tendence k předčasné konvergenci a horší diverzita na Pareto frontě.
•	Výhody NSGA-II:
o	Nedominované třídění: Efektivní hledání optimálních kompromisů.
o	Crowding Distance: Zajištění rovnoměrného pokrytí Pareto fronty.
o	Robustnost: Stabilní výsledky i v diskontinuálním fitness prostoru.
Komentář (pro řečníka):
Volba NSGA-II je podložena potřebou zachovat diverzitu řešení napříč celou Pareto frontou. Na rozdíl od algoritmů jako DE nebo PSO, které jsou primárně určeny pro spojitý prostor, NSGA-II efektivně pracuje se smíšeným prohledávacím prostorem hyperparametrů SOM. Mechanismus crowding distance nám navíc zaručuje, že nezískáme shluk podobných řešení, ale reprezentativní vzorek celého spektra možností trénování.
 
Slide 5 — Definice prohledávacího prostoru

Obsah (na slide):
•	Struktura chromosomu: Smíšený typ parametrů (Mixed-Integer Programming).
•	Kategorické parametry: Typ topologie (Hex/Square), funkce útlumu (Linear, Exp, Log, Step).
•	Diskrétní parametry: Rozměry mapy (map_size $m \times n$).
o	Implementace: Mapování ze spojitého intervalu $[5, 20]$ s diskretizací.
•	Spojité parametry:
o	Počáteční a koncová rychlost učení ($\alpha$).
o	Počáteční poloměr okolí ($\sigma$) jako poměr k rozměru mapy.
•	Ošetření přesnosti: Kontrola rozsahu u parametrů s vysokou citlivostí (learning rate) pro zamezení stagnace v mikro-rozsahu.
Komentář (pro řečníka):
Prohledávací prostor je definován tak, aby pokryl všechny aspekty trénovacího procesu SOM. Specifickou výzvou je diskretizace rozměrů mapy, kde malé změny v genotypech EA nemusí nutně vést ke změně fenotypu (reálného rozměru mřížky). Algoritmus je proto konfigurován s distribučním indexem $\eta=20$ pro křížení a mutaci, což umožňuje jak jemné doladění spojitých hodnot, tak dostatečnou sílu pro překonání plochých oblastí fitness krajiny u diskrétních parametrů.

 
Slide 6 — Genetické operátory pro smíšený prostor
Obsah (na slide):
•	Křížení (SBX – Simulated Binary Crossover):
o	Simulace binárního křížení v reálném prostoru.
o	Parametr distribučního indexu $\eta_c = 20$.
•	Polynomiální mutace:
o	Lokální perturbace parametrů se zachováním mezí intervalů.
o	Parametr distribučního indexu $\eta_m = 20$.
•	Validace a oprava jedinců:
o	Automatická kontrola logických vazeb parametrů.
o	Příklad: počáteční poloměr $\ge$ koncový poloměr, klesající rychlost učení.
Komentář (pro řečníka): Pro efektivní prohledávání kombinovaného prostoru využíváme operátory SBX (Simulated Binary Crossover – křížení simulující binární operace na reálných číslech) a polynomiální mutaci. Hodnota distribučního indexu ($\eta$) nastavená na 20 zajišťuje, že potomci jsou generováni v blízkosti rodičů, což umožňuje precizní lokální ladění (exploataci). Klíčovou součástí je modul validace (repair mechanism), který po mutaci kontroluje, zda parametry dávají smysl – například že poloměr okolí během trénování skutečně klesá a neroste, což by vedlo k degradaci mapy.

 
Slide 7 — Mechanismus výběru a zachování diverzity
Obsah (na slide):
•	Binární turnajový výběr:
o	Priorita 1: Hodnost (pořadí Pareto fronty).
o	Priorita 2: Vzdálenost vytěsnění (Crowding distance).
•	Nedominované třídění:
o	Rozdělení populace do hierarchických vrstev podle dominance.
•	Elitismus:
o	Přímý přenos nejlepších řešení (Pareto fronty) do následující generace.
•	Archivace:
o	Průběžné logování optimálních řešení napříč generacemi.
Komentář (pro řečníka): Algoritmus využívá standardní nedominované třídění (Non-dominated sorting), které rozřazuje jedince do front podle toho, jak se vzájemně dominují v cílových funkcích. Výběr rodičů probíhá formou turnaje, kde vítězí jedinec s lepší hodností (rankem). Pokud mají dva jedinci stejnou hodnost, rozhoduje vzdálenost vytěsnění (Crowding distance), která upřednostňuje jedince v méně zaplněných částech fronty, čímž udržujeme diverzitu řešení podél celého spektra kompromisů.

 
Slide 8 — Paralelní evaluace a efektivita výpočtu
Obsah (na slide):
•	Vícejádrové zpracování:
o	Souběžné trénování celé populace pomocí procesního fondu (Multiprocessing Pool).
•	Mechanismus jednoznačných identifikátorů (UID):
o	Generování hash kódu pro každou unikátní konfiguraci.
•	Deduplikace a mezipaměť:
o	Detekce identických jedinců v různých generacích.
o	Opětovné využití výsledků bez nutnosti nového trénování.
•	Monitoring prostředků:
o	Sledování využití operační paměti a procesorového času.
Komentář (pro řečníka): Vzhledem k tomu, že trénování jedné mapy může trvat sekundy až minuty, je paralelní vyhodnocování nezbytností. Projekt využívá UID (Unique Identifier – unikátní identifikátor založený na MD5 hashi konfigurace), který umožňuje identifikovat identické jedince. Pokud se stejná konfigurace objeví v pozdější generaci, systém načte výsledky z mezipaměti (cache), čímž se radikálně snižuje celková výpočetní náročnost v pokročilých fázích evoluce.
Zde jsou závěrečné slidy zaměřené na výstupy optimalizačního procesu, generování dat pro návazné modely a celkové shrnutí přínosu.

 
Slide 9 — Analýza a vizualizace výsledků
Obsah (na slide):
•	Identifikace Pareto-optimálních řešení:
o	Výběr konfigurací z první fronty (Rank 0).
•	Vizualizace kompromisních variant:
o	Grafické porovnání chybovosti (MQE) vůči času a topologii.
•	Kvantifikace vizuální kvality:
o	Korelace mezi statistickými metrikami a vizuální strukturou mapy.
•	Export parametrů:
o	Kompletní konfigurační sady pro reprodukovatelné trénování.
Komentář (pro řečníka): Výstupem evolučního procesu je soubor optimálních řešení, která tvoří tzv. Pareto frontu – tedy řešení, kde nelze vylepšit jedno kritérium, aniž by se zhoršilo jiné. V rámci projektu analyzujeme především vztah mezi přesností mapování (MQE – Mean Quantization Error) a výpočetní náročností. Součástí výstupu je i vizuální hodnocení, které pomáhá identifikovat konfigurace generující nejlépe interpretovatelné mapy pro koncového uživatele.

 
Slide 10 — Evoluční algoritmus jako generátor dat
Obsah (na slide):
•	Strukturovaný sběr experimentálních dat:
o	Automatické ukládání každého vyhodnoceného jedince.
•	Příprava podkladů pro neuronové sítě:
o	MLP: Vztah mezi hyperparametry a výslednou kvalitou.
o	LSTM: Časové řady metrik v průběhu trénování.
o	CNN: Vícekanálové obrazy (U-matice, vzdálenostní mapy).
•	Rozsah datasetu:
o	Tisíce unikátních konfigurací a jejich tréninkových trajektorií.
Komentář (pro řečníka): Evoluční algoritmus zde neplní pouze roli optimalizátoru, ale slouží jako masivní generátor dat. Každý proběhlý trénink SOM (Self-Organizing Map) ukládá své parametry a metriky pro MLP (Multi-Layer Perceptron), časové průběhy pro LSTM (Long Short-Term Memory) a vizuální reprezentace pro CNN (Convolutional Neural Network). Tato data jsou klíčová pro druhou fázi práce, kdy neuronové sítě převezmou roli "urychlovačů" optimalizačního procesu.

 
Slide 11 — Zhodnocení a přínos automatizace
Obsah (na slide):
•	Eliminace expertního ladění:
o	Autonomní prohledávání bez nutnosti hluboké znalosti vnitřních parametrů.
•	Objektivita procesu:
o	Nahrazení subjektivního výběru parametrů statisticky podloženou optimalizací.
•	Efektivita vyhledávání:
o	Schopnost nalézt netradiční konfigurace mimo standardní doporučení.
•	Škálovatelnost:
o	Snadná adaptace na různé typy datových sad.
Komentář (pro řečníka): Hlavním přínosem nasazení evolučních algoritmů je transformace procesu, který dříve vyžadoval hodiny expertní práce, v plně autonomní systém. Algoritmus dokáže efektivně prohledat prostor, který je pro člověka neintuitivní, a nalézt kombinace parametrů (např. specifické tvary útlumových křivek rychlosti učení), které vedou k lepším výsledkům než standardní, manuálně volené postupy. Evoluce tak tvoří stabilní základ pro celou modulární platformu.

 
Slide 12 — Shrnutí a závěr
Obsah (na slide):
•	NSGA-II jako robustní jádro:
o	Efektivní zvládnutí smíšeného prohledávacího prostoru.
•	Vícekriteriální pohled:
o	Nalezení rovnováhy mezi přesností, časem a topologií.
•	Základ pro AI moduly:
o	Vytvoření robustní znalostní báze pro trénink dalších komponent.
•	Směr dalšího vývoje:
o	Integrace predikčních modelů pro zrychlení evolučního cyklu.
Komentář (pro řečníka): Práce prokázala, že použití víceúčelové evoluční optimalizace je pro konfiguraci samoorganizujících se map vysoce efektivní cestou. Algoritmus NSGA-II (Non-dominated Sorting Genetic Algorithm II) poskytuje potřebnou stabilitu a diversitu řešení. Aktuálně dokončená fáze optimalizace bez účasti neuronových sítí tvoří nezbytný podklad, na kterém bude postaveno budoucí zrychlení celého analytického pipeline pomocí predikčních modelů.

