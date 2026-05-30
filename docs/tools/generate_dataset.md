# generate_dataset.py

Generátor syntetických datasetů podle schématu reálného datasetu.

Kód: [`app/tools/generate_dataset.py`](../../app/tools/generate_dataset.py)

---

## Účel

Vytváří syntetická data, která zachovávají statistické vlastnosti (typy sloupců, distribuce, kardinalitu kategorií) reálného datasetu. Vstupem je `preprocessing_info.json` vygenerovaný SOM pipeline z reálných dat.

Použití: testování pipeline na datech podobné struktury bez nutnosti mít originální data, generování větších verzí malých datasetů.

---

## Použití

```bash
python app/tools/generate_dataset.py \
  --schema data/datasets/Iris/results/<run>/preprocessing_info.json \
  --output data/datasets/IrisSynthetic/iris_synthetic.csv \
  --rows 1000

# S fixním seedem
python app/tools/generate_dataset.py \
  --schema path/to/preprocessing_info.json \
  --output output.csv \
  --rows 500 --seed 42
```

### Parametry

| Parametr | Popis |
|----------|-------|
| `--schema` | Cesta k `preprocessing_info.json` |
| `--output` | Výstupní CSV soubor |
| `--rows` | Počet řádků |
| `--seed` | Random seed (volitelné) |

---

## Jak funguje

Ze schématu čte pro každý sloupec: typ (`float`, `int`, kategorický), počet unikátních hodnot, `nunique_ratio`. Podle těchto vlastností generuje:

- **Numerické s vysokou unikátností** → normální rozdělení
- **Celočíselné ID sloupce** (`nunique_ratio = 1.0`) → sekvenční nebo náhodná unikátní čísla
- **Kategorické** → vzorkování z původních kategorií s zachováním frekvencí

---

## Výstup

CSV soubor se stejnou strukturou sloupců jako originální dataset.
