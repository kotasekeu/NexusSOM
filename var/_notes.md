tensorboard --logdir=logs/som_quality_standard_20260112_210424
spuštění vizualizací nad CNN NS dostupne pak na 
http://localhost:6006/





# ============================================
# PŘÍPRAVA PROSTŘEDÍ
# ============================================

# Vytvoření virtuálního prostředí pro TensorFlow (pro NN)
python3 -m venv venv_tf
source venv_tf/bin/activate
pip install -r requirements.txt
pip install -r requirements_nn.txt

# Deaktivace venv (pro běh bez NN)
deactivate


# ============================================
# SPUŠTĚNÍ SOM
# ============================================

# SOM BEZ NN (běžné prostředí, bez TensorFlow)
cd app/som
python3 som.py --input ../data/iris.csv --map-size 10 10

# SOM S NN (není přímá integrace - SOM funguje samostatně)
# SOM nemá NN závislosti, běží vždy stejně
python3 som.py --input ../data/iris.csv --map-size 15 15


# ============================================
# SPUŠTĚNÍ EA (Genetický algoritmus)
# ============================================

# EA BEZ NN (všechny NN vypnuté v config)
cd app/ea
python3 ea.py

# EA S NN (zapnout v ea_config.py: use_mlp=True, use_cnn=True)
# Nejdřív editovat ea/ea_config.py:
# "NEURAL_NETWORKS": {
#     "use_mlp": True,
#     "use_lstm": False,
#     "use_cnn": True,
#     ...
# }
source ../venv_tf/bin/activate
python3 ea.py


# ============================================
# CNN - PŘÍPRAVA DAT
# ============================================

cd app/cnn

# 1. Validace dat (kontrola integrity)
python3 src/validate_data.py

# 2. Příprava datasetu (výpočet quality scores z results.csv)
python3 src/prepare_data.py


# ============================================
# CNN - LABELOVÁNÍ DAT
# ============================================

# Interaktivní labelovací nástroj
python3 label_maps.py --results_dir ../test/results/20260110_220147

# Klávesy během labelování:
# G - Good (dobrá mapa)
# B - Bad (špatná mapa)
# S - Save (uložit obrázek)
# Q - Quit (uložit a ukončit)


# ============================================
# TRÉNOVÁNÍ NEURONOVÝCH SÍTÍ
# ============================================

# Aktivovat TensorFlow prostředí
source venv_tf/bin/activate

# ------------------------------
# CNN "The Eye" - Vizuální hodnocení kvality
# ------------------------------
cd app/cnn

# Trénování standardního modelu (50 epoch)
python3 src/train.py --model standard --epochs 50 --batch-size 32

# Trénování lightweight modelu (rychlejší)
python3 src/train.py --model lite --epochs 50 --batch-size 32

# S vlastními parametry
python3 src/train.py --model standard --epochs 100 --batch-size 16 --learning-rate 0.001


# ------------------------------
# MLP "The Prophet" - Predikce kvality z hyperparametrů
# ------------------------------
cd app/mlp

# 1. Příprava datasetu z EA results.csv
python3 prepare_dataset.py --input ../test/results/20260110_220147/results.csv

# 2. Trénování standardního modelu
python3 src/train.py --dataset data/dataset.csv --epochs 50

# 3. Trénování lightweight modelu
python3 src/train.py --dataset data/dataset.csv --epochs 50 --model lite


# ------------------------------
# LSTM "The Oracle" - Predikce finální kvality z průběhu trénování
# ------------------------------
cd app/lstm

# 1. Sběr trénovacích dat (POC - simulovaná data)
python3 collect_training_data.py

# 2. Trénování standardního modelu
python3 src/train.py --dataset data/dataset.csv --epochs 50

# 3. Trénování lightweight modelu
python3 src/train.py --dataset data/dataset.csv --epochs 50 --model lite


# ============================================
# EVALUACE MODELŮ
# ============================================

# CNN evaluace
cd app/cnn
python3 src/evaluate.py \
    --model models/som_quality_standard_20260112_210424_best.keras \
    --test-csv models/som_quality_standard_20260112_210424_test_set.csv

# MLP evaluace
cd app/mlp
python3 evaluate_model.py \
    --model models/mlp_prophet_lite_20260113_073440_best.keras

# LSTM evaluace
cd app/lstm
python3 evaluate_model.py \
    --model models/lstm_oracle_lite_20260113_073755_best.keras


# ============================================
# PREDIKCE S NATRÉNOVANÝMI MODELY
# ============================================

# CNN - predikce jednoho obrázku
cd app/cnn
python3 src/predict.py \
    --model models/som_quality_standard_20260112_210424_best.keras \
    --image ../test/results/20260110_220147/maps_dataset/rgb/example.png

# CNN - batch predikce (celá složka)
python3 src/predict.py \
    --model models/som_quality_standard_20260112_210424_best.keras \
    --image-dir ../test/results/20260110_220147/maps_dataset/rgb/ \
    --output predictions.csv


# ============================================
# MONITORING TRÉNOVÁNÍ
# ============================================

# TensorBoard pro CNN
cd app/cnn
tensorboard --logdir=logs/som_quality_standard_20260112_210424 --port=6006
# Otevřít v prohlížeči: http://localhost:6006

# TensorBoard pro MLP
cd app/mlp
tensorboard --logdir=logs/ --port=6007

# TensorBoard pro LSTM
cd app/lstm
tensorboard --logdir=logs/ --port=6008


# ============================================
# TESTOVÁNÍ NN INTEGRACE
# ============================================

# Test integračního modulu (bez TensorFlow)
deactivate
cd app/ea
python3 nn_integration.py
# Očekávaný výstup: ⚠ TensorFlow not available

# Test integračního modulu (s TensorFlow)
source ../venv_tf/bin/activate
python3 nn_integration.py
# Očekávaný výstup: ✓ All models loaded successfully


# ============================================
# QUICK START - KOMPLETNÍ WORKFLOW
# ============================================

# 1. Spustit EA optimalizaci (vygeneruje results.csv a RGB mapy)
cd app/ea
python3 ea.py

# 2. Připravit CNN data
cd ../cnn
python3 src/prepare_data.py

# 3. (Volitelně) Labelovat mapy ručně
python3 label_maps.py --results_dir ../test/results/YYYYMMDD_HHMMSS

# 4. Natrénovat CNN
source ../venv_tf/bin/activate
python3 src/train.py --model standard --epochs 50

# 5. Připravit MLP data
cd ../mlp
python3 prepare_dataset.py --input ../test/results/YYYYMMDD_HHMMSS/results.csv

# 6. Natrénovat MLP
python3 src/train.py --dataset data/dataset.csv --epochs 50 --model lite

# 7. Natrénovat LSTM (s POC daty)
cd ../lstm
python3 collect_training_data.py
python3 src/train.py --dataset data/dataset.csv --epochs 50 --model lite

# 8. Zapnout NN v EA konfiguraci a spustit optimalizaci
cd ../ea
# Editovat ea_config.py: use_mlp=True, use_cnn=True
python3 ea.py
# Výsledek: 10-20× rychlejší než bez NN