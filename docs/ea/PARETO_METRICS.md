# Pareto metriky — assessment

**Verze**: 1.0  
**Datum**: 2026-05-14

---

## Objectives (pro minimalizaci)

| # | Objective | Výpočet | Rozsah |
|---|---|---|---|
| 0 | `raw_mqe_ratio` | `final_mqe / initial_mqe` | 0–1 (nižší = lepší SOM) |
| 1 | `topo_error` | topographic error | 0–1 |
| 2 | `dead_ratio` | podíl mrtvých neuronů | 0–1 |

---

## 12 položek — stav implementace

| # | Položka | Stav | Detail |
|---|---|---|---|
| 1 | **Hypervolume 3D** | ✅ Hotovo | `_compute_pareto_metrics()` v `ea.py`, pymoo `HV` |
| 2 | **Fixed reference point** | ✅ Hotovo | `_HV_REFERENCE = [1.1, 1.1, 1.1]` — konzistentní přes generace i seedy |
| 3 | **Normalizace před HV** | ✅ Hotovo | Globální running min/max přes všechny feasible solutions v daném sedu — `_normalize_objectives()` + `np.clip(0.0, 1.1)`; srovnatelné HV přes generace |
| 4 | **Crowding distance** | ✅ Hotovo | `crowding_distance_assignment()` — standardní NSGA-II, line 256 |
| 5 | **Elitismus / archiv** | ✅ Hotovo | `ARCHIVE` = pouze rank==0 jedinci, přenesen do každé generace |
| 6 | **Penalizovaní do nejhoršího frontu** | ✅ Hotovo | `non_dominated_sort()` s `violations` param — infeasible dominováni všemi feasible |
| 7 | **State-action cross-reference** | ❌ Chybí | LSTM controller loguje per-step faktory, `pareto_front.csv` per-gen řešení — žádné propojení (HV delta vs. průměrné LR/R zásahy) |
| 8 | **Penalizace jako kontinuum** | ✅ Hotovo | `compute_constraint_violation()` — graduated bands 1.5/2.5/5.0, `cv > 0` = infeasible s magnitudou |
| 9 | **HV + Spacing kombinace** | ✅ Hotovo | Obojí v `_compute_pareto_metrics()`, výstup do `pareto_metrics.csv` |
| 10 | **n-D Spacing (3D)** | ✅ Hotovo | Nearest-neighbor distance v 3D objective space; `sqrt(mean((d_bar - d_i)^2))` |
| 11 | **Normalizace před Spacing** | ✅ Hotovo | Sdílí normalizaci s HV — `_normalize_objectives()` před výpočtem obou metrik |
| 12 | **Spacing jako LSTM signál** | ❌ Odloženo | HV/Spacing jsou per-generace metriky; LSTM controller pracuje per-checkpoint — přímé zapojení nedává smysl. Využití: post-hoc korelace v `pareto_metrics.csv` |

**Stav**: 10/12 hotovo. Položky 7 a 12 jsou záměrně odloženy (architektonické důvody).

---

## Výstupní soubory

| Soubor | Obsah | Granularita |
|---|---|---|
| `pareto_front.csv` | Každé řešení na Pareto frontě | per-generace × per-řešení |
| `pareto_metrics.csv` | HV + Spacing souhrn | per-generace (1 řádek) |

### Sloupce `pareto_metrics.csv`

```
generation, front_size, hv, spacing, spread_mqe, spread_te, spread_dead
```

- `hv` — hypervolume dominovaný frontou vůči referenčnímu bodu [1.1,1.1,1.1] v normalizovaném prostoru; maximum = 1.1³ = 1.331
- `spacing` — rovnoměrnost rozložení bodů na frontě; 0 = dokonale uniformní
- `spread_mqe/te/dead` — Maximum Spread per dimenzi (max−min v normalizovaném prostoru); blízko 1.0 = front pokrývá celý rozsah dané dimenze

---

## Implementace

```python
# ea.py — relevantní funkce
_HV_REFERENCE = np.array([1.1, 1.1, 1.1])

_update_obj_running_stats(raw_objectives)  # aktualizuje globální min/max per seed
_normalize_objectives(raw_objectives)      # → normalizováno + clip [0, 1.1]
_compute_pareto_metrics(norm_objectives)   # → {front_size, hv, spacing, spread_*}
_log_pareto_metrics(generation, metrics)   # → zapíše do pareto_metrics.csv
```

Pořadí volání v `log_pareto_front()`: `_update_obj_running_stats` → `_normalize_objectives` → `_compute_pareto_metrics` → `_log_pareto_metrics`. Running stats se resetují pro každý seed.

`log_pareto_front()` volá obě funkce automaticky na konci každé generace. Do HV/Spacing vstupují pouze **feasible** řešení (bez penalizace).

---

## Otevřené položky

| # | Položka | Poznámka |
|---|---|---|
| 7 | State-action cross-reference | Pokud controller skutečně zasahuje (Fáze 5), dává smysl párovat HV delta s průměrným `|lr_f - 1|` per-generace — post-hoc analýza, ne live signál |
| 12 | Spacing jako LSTM signál | Spacing charakterizuje celou frontu, ne jeden run — nevhodné pro per-checkpoint controller |
