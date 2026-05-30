# verify_ea_run.py

Diagnostický nástroj pro analýzu výsledků EA běhu. Výstup do terminálu.

Kód: [`app/tools/verify_ea_run.py`](../../app/tools/verify_ea_run.py)

---

## Účel

Ověřuje správnost EA běhu — odhaluje problémy jako ztrátu elitismu, dominanci penalizovaných řešení v archivu, konvergenci do špatné oblasti search space nebo interní dominanci ve finálním archivu.

---

## Použití

```bash
python app/tools/verify_ea_run.py data/datasets/BreastCancer/results/<run>/

# Jen vybrané sekce (čísla 1–8)
python app/tools/verify_ea_run.py <results_dir> --sections 1 5 8
```

### Parametry

| Parametr | Popis |
|----------|-------|
| `results_dir` | Složka EA běhu (obsahuje `results.csv`, `pareto_front.csv`) |
| `--sections` | Vypsat jen vybrané sekce (default: všechny) |

---

## Sekce reportu

| # | Název | Co kontroluje |
|---|-------|---------------|
| 1 | **Overview** | Celkový počet evaluací, míra penalizace, distribuce `raw_mqe_ratio` |
| 2 | **Map size vs penalty** | Korelace velikosti mapy s penalizací — odhalí příliš malé/velké mapy |
| 3 | **Archive evolution** | Vývoj archivu per generace — počet feasible/infeasible, nejlepší ratio |
| 4 | **Elitism check** | Sleduje zda nejlepší UID přežívají do dalších generací |
| 5 | **Final archive** | Detailní výpis finálního archivu + kontrola interní dominance (žádné řešení by nemělo dominovat jiné ve stejné frontě) |
| 6 | **Crowding ejection** | Dobrá řešení ztracená cap-em archivu vs. skutečnou dominancí |
| 7 | **Parameter correlation** | Které parametry korelují s penalizací — vodítko pro úpravu search space |
| 8 | **Recommendations** | Automatické doporučení na základě nalezených problémů |

---

## Typické nálezy a jejich interpretace

**Míra penalizace 100 %** → `org_threshold` příliš přísný, kalibrační sonda nepomohla nebo nebyla spuštěna.

**Elitism selhání** → archiv ztrácí dobrá řešení mezi generacemi. Příčina: bug v indexování po sort() (viz ISSUES.md #5).

**Interní dominance v archivu** → NSGA-II řadí špatně. Příčina: nekonzistentní CV výpočet nebo bug v `_dominates()`.

**Všechna řešení v archivu mají stejný `map_size`** → EA konvergovala předčasně, search space příliš úzký nebo `epoch_multiplier` min příliš nízký.

---

## Vstupy

- `results.csv` — všechna evaluovaná řešení
- `pareto_front.csv` — archiv per generace (preferováno)
- `pareto_front_log.txt` — starší formát (fallback)
