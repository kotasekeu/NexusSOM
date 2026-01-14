# Project Flow #
## From Install to final step ##

1. Install virtual env for SOM, EA and all NN with all dependencies
2. /opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv

3. source .venv/bin/activate
4. install all dependencies 
   <br /> python -m pip install -r ./python/requirements.txt
   <br /> and for NN
   <br /> python -m pip install -r ./python/requirements_nn.txt
   <br /> and for Jupyter testing
   <br /> python -m pip install -r ./python/requirements_jupyter.txt
   <br /> run smoke test 
   <br /> python ./tests/smoke_tests/smoke_test.py 
   <br /> and for Mac Silicon CPU
   <br /> python ./tests/smoke_tests/mac_gpu.py

### SOM ###
Run SOM on BC dataset

python ./app/run_som.py -i ./data/datasets/BreastCancer/breast-cancer.csv -c ./data/datasets/BreastCancer/config-som.json 


### EA ###
Run EA on BC dataset

python ./app/run_ea.py -i ./data/datasets/BreastCancer/breast-cancer.csv -c ./data/datasets/BreastCancer/config-ea.json 

Folder results is created in same folder as is input file (.csv) 

### CNN ###
labelování map
python ./app/cnn/src/label_maps.py --results_dir ./data/datasets/BreastCancer/results/EA --auto_bad_threshold 0.1
Labeling Session Summary
Total labeled: 982/982
  - Good: 337
  - Bad: 645
    • Manual: 34
    • Auto (dead_neuron_ratio > 10%): 611

vytvoření dataset souboru, spouštět z app/cnn protože cesty
python ./src/prepare_dataset.py --results_dir ../../data/datasets/BreastCancer/results/EA
Dataset saved to: ./data/datasets/BreastCancer/results/EA/dataset.csv


spouštět z cnn kvuli složkam log a model

Trenovani modelu
python ./src/train.py --dataset ../../data/datasets/BreastCancer/results/EA/dataset.csv

Total params: 1,276,449 (4.87 MB)
Trainable params: 1,273,761 (4.86 MB)
Non-trainable params: 2,688 (10.50 KB)
================================================================================
Total parameters: 1,276,449
Trainable parameters: 1,273,761







Jupyter Test 
Inside venv run
jupyter notebook --no-browser

copy local url 
http://localhost:8888/tree?token=9beee9a70bf42402a07d6895c5b8cec6f5ef8beb1ed5e1b2