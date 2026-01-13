================================================================================
LSTM TRAINING PROGRESS PREDICTOR - TRAINING
================================================================================
Model Type: lite
Epochs: 50
Batch Size: 32
Learning Rate: 0.001
================================================================================
Loading dataset from: data/test_dataset.csv
Total sequences: 1000
✓ Loaded metadata: 10 checkpoints per sequence

Parsing training sequences...
Sequence shape: (1000, 10, 3)
Target shape: (1000, 3)

Training sequences: 722
Validation sequences: 128
Test sequences: 150

Creating model...

================================================================================
MODEL ARCHITECTURE
================================================================================

Model: "lstm_oracle_lite"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm1 (LSTM)                         │ (None, 10, 64)              │          17,408 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout1 (Dropout)                   │ (None, 10, 64)              │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm2 (LSTM)                         │ (None, 32)                  │          12,416 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense1 (Dense)                       │ (None, 16)                  │             528 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output (Dense)                       │ (None, 3)                   │              51 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 30,403 (118.76 KB)
 Trainable params: 30,403 (118.76 KB)
 Non-trainable params: 0 (0.00 B)

================================================================================
MODEL DETAILS
================================================================================
Total parameters: 30,403
Input shape: (None, 10, 3)
Output shape: (None, 3)
================================================================================


Starting training...
Epoch 1/50
15/23 ━━━━━━━━━━━━━━━━━━━━ 0s 4ms/step - loss: 0.1931 - mae: 0.3341 - mse: 0.1931
Epoch 1: val_loss improved from None to 0.01394, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 1: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 1s 14ms/step - loss: 0.1069 - mae: 0.2303 - mse: 0.1069 - val_loss: 0.0139 - val_mae: 0.0816 - val_mse: 0.0139 - learning_rate: 0.0010
Epoch 2/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0157 - mae: 0.0947 - mse: 0.0157 
Epoch 2: val_loss improved from 0.01394 to 0.00813, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 2: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0119 - mae: 0.0794 - mse: 0.0119 - val_loss: 0.0081 - val_mae: 0.0664 - val_mse: 0.0081 - learning_rate: 0.0010
Epoch 3/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0082 - mae: 0.0671 - mse: 0.0082 
Epoch 3: val_loss improved from 0.00813 to 0.00595, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 3: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 9ms/step - loss: 0.0070 - mae: 0.0602 - mse: 0.0070 - val_loss: 0.0060 - val_mae: 0.0537 - val_mse: 0.0060 - learning_rate: 0.0010
Epoch 4/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0062 - mae: 0.0563 - mse: 0.0062 
Epoch 4: val_loss improved from 0.00595 to 0.00518, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 4: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0056 - mae: 0.0537 - mse: 0.0056 - val_loss: 0.0052 - val_mae: 0.0486 - val_mse: 0.0052 - learning_rate: 0.0010
Epoch 5/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0053 - mae: 0.0523 - mse: 0.0053 
Epoch 5: val_loss improved from 0.00518 to 0.00449, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 5: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0049 - mae: 0.0494 - mse: 0.0049 - val_loss: 0.0045 - val_mae: 0.0439 - val_mse: 0.0045 - learning_rate: 0.0010
Epoch 6/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0047 - mae: 0.0479 - mse: 0.0047 
Epoch 6: val_loss improved from 0.00449 to 0.00385, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 6: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0043 - mae: 0.0458 - mse: 0.0043 - val_loss: 0.0039 - val_mae: 0.0405 - val_mse: 0.0039 - learning_rate: 0.0010
Epoch 7/50
16/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0043 - mae: 0.0458 - mse: 0.0043 
Epoch 7: val_loss improved from 0.00385 to 0.00337, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 7: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0039 - mae: 0.0428 - mse: 0.0039 - val_loss: 0.0034 - val_mae: 0.0379 - val_mse: 0.0034 - learning_rate: 0.0010
Epoch 8/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0036 - mae: 0.0420 - mse: 0.0036 
Epoch 8: val_loss improved from 0.00337 to 0.00285, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 8: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0032 - mae: 0.0393 - mse: 0.0032 - val_loss: 0.0029 - val_mae: 0.0351 - val_mse: 0.0029 - learning_rate: 0.0010
Epoch 9/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0033 - mae: 0.0395 - mse: 0.0033 
Epoch 9: val_loss improved from 0.00285 to 0.00246, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 9: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.0029 - mae: 0.0363 - mse: 0.0029 - val_loss: 0.0025 - val_mae: 0.0307 - val_mse: 0.0025 - learning_rate: 0.0010
Epoch 10/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0029 - mae: 0.0365 - mse: 0.0029
Epoch 10: val_loss improved from 0.00246 to 0.00242, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 10: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.0027 - mae: 0.0346 - mse: 0.0027 - val_loss: 0.0024 - val_mae: 0.0334 - val_mse: 0.0024 - learning_rate: 0.0010
Epoch 11/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0026 - mae: 0.0356 - mse: 0.0026 
Epoch 11: val_loss improved from 0.00242 to 0.00232, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 11: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0025 - mae: 0.0337 - mse: 0.0025 - val_loss: 0.0023 - val_mae: 0.0339 - val_mse: 0.0023 - learning_rate: 0.0010
Epoch 12/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0025 - mae: 0.0342 - mse: 0.0025 
Epoch 12: val_loss improved from 0.00232 to 0.00208, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 12: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0023 - mae: 0.0327 - mse: 0.0023 - val_loss: 0.0021 - val_mae: 0.0327 - val_mse: 0.0021 - learning_rate: 0.0010
Epoch 13/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0022 - mae: 0.0326 - mse: 0.0022 
Epoch 13: val_loss improved from 0.00208 to 0.00196, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 13: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0021 - mae: 0.0315 - mse: 0.0021 - val_loss: 0.0020 - val_mae: 0.0322 - val_mse: 0.0020 - learning_rate: 0.0010
Epoch 14/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0021 - mae: 0.0325 - mse: 0.0021 
Epoch 14: val_loss improved from 0.00196 to 0.00172, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 14: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0019 - mae: 0.0307 - mse: 0.0019 - val_loss: 0.0017 - val_mae: 0.0300 - val_mse: 0.0017 - learning_rate: 0.0010
Epoch 15/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0018 - mae: 0.0317 - mse: 0.0018 
Epoch 15: val_loss improved from 0.00172 to 0.00145, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 15: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0017 - mae: 0.0294 - mse: 0.0017 - val_loss: 0.0015 - val_mae: 0.0250 - val_mse: 0.0015 - learning_rate: 0.0010
Epoch 16/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0016 - mae: 0.0278 - mse: 0.0016 
Epoch 16: val_loss improved from 0.00145 to 0.00128, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 16: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0016 - mae: 0.0276 - mse: 0.0016 - val_loss: 0.0013 - val_mae: 0.0228 - val_mse: 0.0013 - learning_rate: 0.0010
Epoch 17/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0013 - mae: 0.0264 - mse: 0.0013 
Epoch 17: val_loss improved from 0.00128 to 0.00128, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 17: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0014 - mae: 0.0261 - mse: 0.0014 - val_loss: 0.0013 - val_mae: 0.0231 - val_mse: 0.0013 - learning_rate: 0.0010
Epoch 18/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0015 - mae: 0.0278 - mse: 0.0015 
Epoch 18: val_loss improved from 0.00128 to 0.00127, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 18: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0014 - mae: 0.0266 - mse: 0.0014 - val_loss: 0.0013 - val_mae: 0.0239 - val_mse: 0.0013 - learning_rate: 0.0010
Epoch 19/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0016 - mae: 0.0280 - mse: 0.0016 
Epoch 19: val_loss did not improve from 0.00127
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.0016 - mae: 0.0270 - mse: 0.0016 - val_loss: 0.0013 - val_mae: 0.0249 - val_mse: 0.0013 - learning_rate: 0.0010
Epoch 20/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0014 - mae: 0.0269 - mse: 0.0014
Epoch 20: val_loss did not improve from 0.00127
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.0013 - mae: 0.0257 - mse: 0.0013 - val_loss: 0.0013 - val_mae: 0.0247 - val_mse: 0.0013 - learning_rate: 0.0010
Epoch 21/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0014 - mae: 0.0269 - mse: 0.0014
Epoch 21: val_loss improved from 0.00127 to 0.00112, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 21: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0013 - mae: 0.0255 - mse: 0.0013 - val_loss: 0.0011 - val_mae: 0.0215 - val_mse: 0.0011 - learning_rate: 0.0010
Epoch 22/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0014 - mae: 0.0265 - mse: 0.0014 
Epoch 22: val_loss did not improve from 0.00112
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.0014 - mae: 0.0256 - mse: 0.0014 - val_loss: 0.0011 - val_mae: 0.0226 - val_mse: 0.0011 - learning_rate: 0.0010
Epoch 23/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0014 - mae: 0.0264 - mse: 0.0014
Epoch 23: val_loss improved from 0.00112 to 0.00107, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 23: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0013 - mae: 0.0255 - mse: 0.0013 - val_loss: 0.0011 - val_mae: 0.0218 - val_mse: 0.0011 - learning_rate: 0.0010
Epoch 24/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0013 - mae: 0.0260 - mse: 0.0013
Epoch 24: val_loss improved from 0.00107 to 0.00105, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 24: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.0012 - mae: 0.0244 - mse: 0.0012 - val_loss: 0.0011 - val_mae: 0.0222 - val_mse: 0.0011 - learning_rate: 0.0010
Epoch 25/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0013 - mae: 0.0261 - mse: 0.0013 
Epoch 25: val_loss improved from 0.00105 to 0.00097, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 25: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.0012 - mae: 0.0246 - mse: 0.0012 - val_loss: 9.7304e-04 - val_mae: 0.0208 - val_mse: 9.7304e-04 - learning_rate: 0.0010
Epoch 26/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0013 - mae: 0.0260 - mse: 0.0013
Epoch 26: val_loss improved from 0.00097 to 0.00097, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 26: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0012 - mae: 0.0242 - mse: 0.0012 - val_loss: 9.7094e-04 - val_mae: 0.0215 - val_mse: 9.7094e-04 - learning_rate: 0.0010
Epoch 27/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0012 - mae: 0.0246 - mse: 0.0012
Epoch 27: val_loss improved from 0.00097 to 0.00097, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 27: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0012 - mae: 0.0236 - mse: 0.0012 - val_loss: 9.7053e-04 - val_mae: 0.0219 - val_mse: 9.7053e-04 - learning_rate: 0.0010
Epoch 28/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0011 - mae: 0.0247 - mse: 0.0011
Epoch 28: val_loss improved from 0.00097 to 0.00096, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 28: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0011 - mae: 0.0232 - mse: 0.0011 - val_loss: 9.6488e-04 - val_mae: 0.0220 - val_mse: 9.6488e-04 - learning_rate: 0.0010
Epoch 29/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0011 - mae: 0.0243 - mse: 0.0011 
Epoch 29: val_loss improved from 0.00096 to 0.00081, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 29: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.0010 - mae: 0.0227 - mse: 0.0010 - val_loss: 8.0925e-04 - val_mae: 0.0190 - val_mse: 8.0925e-04 - learning_rate: 0.0010
Epoch 30/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0011 - mae: 0.0236 - mse: 0.0011        
Epoch 30: val_loss did not improve from 0.00081
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 0.0011 - mae: 0.0229 - mse: 0.0011 - val_loss: 8.6805e-04 - val_mae: 0.0207 - val_mse: 8.6805e-04 - learning_rate: 0.0010
Epoch 31/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0011 - mae: 0.0238 - mse: 0.0011
Epoch 31: val_loss improved from 0.00081 to 0.00076, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 31: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 9.6873e-04 - mae: 0.0220 - mse: 9.6873e-04 - val_loss: 7.6208e-04 - val_mae: 0.0185 - val_mse: 7.6208e-04 - learning_rate: 0.0010
Epoch 32/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 9.3748e-04 - mae: 0.0222 - mse: 9.3748e-04 
Epoch 32: val_loss improved from 0.00076 to 0.00069, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 32: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 8.9852e-04 - mae: 0.0214 - mse: 8.9852e-04 - val_loss: 6.8663e-04 - val_mae: 0.0177 - val_mse: 6.8663e-04 - learning_rate: 0.0010
Epoch 33/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 9.1967e-04 - mae: 0.0217 - mse: 9.1967e-04
Epoch 33: val_loss did not improve from 0.00069
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 9.1253e-04 - mae: 0.0213 - mse: 9.1253e-04 - val_loss: 7.2196e-04 - val_mae: 0.0191 - val_mse: 7.2196e-04 - learning_rate: 0.0010
Epoch 34/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 9.3398e-04 - mae: 0.0225 - mse: 9.3398e-04
Epoch 34: val_loss did not improve from 0.00069
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 9.3821e-04 - mae: 0.0219 - mse: 9.3821e-04 - val_loss: 7.6271e-04 - val_mae: 0.0199 - val_mse: 7.6271e-04 - learning_rate: 0.0010
Epoch 35/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0010 - mae: 0.0231 - mse: 0.0010
Epoch 35: val_loss improved from 0.00069 to 0.00066, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 35: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 9.0591e-04 - mae: 0.0217 - mse: 9.0591e-04 - val_loss: 6.6041e-04 - val_mae: 0.0181 - val_mse: 6.6041e-04 - learning_rate: 0.0010
Epoch 36/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 9.0818e-04 - mae: 0.0218 - mse: 9.0818e-04 
Epoch 36: val_loss did not improve from 0.00066
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 8.5667e-04 - mae: 0.0208 - mse: 8.5667e-04 - val_loss: 6.9503e-04 - val_mae: 0.0200 - val_mse: 6.9503e-04 - learning_rate: 0.0010
Epoch 37/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 9.5930e-04 - mae: 0.0227 - mse: 9.5930e-04
Epoch 37: val_loss improved from 0.00066 to 0.00062, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 37: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 9.4862e-04 - mae: 0.0216 - mse: 9.4862e-04 - val_loss: 6.1855e-04 - val_mae: 0.0173 - val_mse: 6.1855e-04 - learning_rate: 0.0010
Epoch 38/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 8.5393e-04 - mae: 0.0214 - mse: 8.5393e-04 
Epoch 38: val_loss did not improve from 0.00062
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 8.1799e-04 - mae: 0.0201 - mse: 8.1799e-04 - val_loss: 6.2515e-04 - val_mae: 0.0187 - val_mse: 6.2515e-04 - learning_rate: 0.0010
Epoch 39/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 8.4327e-04 - mae: 0.0217 - mse: 8.4327e-04
Epoch 39: val_loss improved from 0.00062 to 0.00055, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 39: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 7.7399e-04 - mae: 0.0201 - mse: 7.7399e-04 - val_loss: 5.5177e-04 - val_mae: 0.0169 - val_mse: 5.5177e-04 - learning_rate: 0.0010
Epoch 40/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 8.0734e-04 - mae: 0.0208 - mse: 8.0734e-04 
Epoch 40: val_loss improved from 0.00055 to 0.00054, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 40: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 7.9292e-04 - mae: 0.0200 - mse: 7.9292e-04 - val_loss: 5.4135e-04 - val_mae: 0.0171 - val_mse: 5.4135e-04 - learning_rate: 0.0010
Epoch 41/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 7.3746e-04 - mae: 0.0197 - mse: 7.3746e-04 
Epoch 41: val_loss improved from 0.00054 to 0.00048, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 41: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 6.9913e-04 - mae: 0.0188 - mse: 6.9913e-04 - val_loss: 4.8436e-04 - val_mae: 0.0159 - val_mse: 4.8436e-04 - learning_rate: 0.0010
Epoch 42/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 7.0550e-04 - mae: 0.0192 - mse: 7.0550e-04 
Epoch 42: val_loss did not improve from 0.00048
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 6.7783e-04 - mae: 0.0186 - mse: 6.7783e-04 - val_loss: 5.4493e-04 - val_mae: 0.0174 - val_mse: 5.4493e-04 - learning_rate: 0.0010
Epoch 43/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 8.5274e-04 - mae: 0.0213 - mse: 8.5274e-04
Epoch 43: val_loss did not improve from 0.00048
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 7.9353e-04 - mae: 0.0201 - mse: 7.9353e-04 - val_loss: 5.3151e-04 - val_mae: 0.0170 - val_mse: 5.3151e-04 - learning_rate: 0.0010
Epoch 44/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 7.3996e-04 - mae: 0.0199 - mse: 7.3996e-04 
Epoch 44: val_loss improved from 0.00048 to 0.00039, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 44: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 6.6874e-04 - mae: 0.0184 - mse: 6.6874e-04 - val_loss: 3.9019e-04 - val_mae: 0.0143 - val_mse: 3.9019e-04 - learning_rate: 0.0010
Epoch 45/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 7.1256e-04 - mae: 0.0191 - mse: 7.1256e-04
Epoch 45: val_loss did not improve from 0.00039
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 7.0039e-04 - mae: 0.0188 - mse: 7.0039e-04 - val_loss: 4.9081e-04 - val_mae: 0.0166 - val_mse: 4.9081e-04 - learning_rate: 0.0010
Epoch 46/50
18/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 6.3268e-04 - mae: 0.0186 - mse: 6.3268e-04
Epoch 46: val_loss did not improve from 0.00039
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 6.8200e-04 - mae: 0.0188 - mse: 6.8200e-04 - val_loss: 4.5685e-04 - val_mae: 0.0159 - val_mse: 4.5685e-04 - learning_rate: 0.0010
Epoch 47/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 7.0211e-04 - mae: 0.0194 - mse: 7.0211e-04
Epoch 47: val_loss did not improve from 0.00039
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 6.8862e-04 - mae: 0.0188 - mse: 6.8862e-04 - val_loss: 4.6740e-04 - val_mae: 0.0162 - val_mse: 4.6740e-04 - learning_rate: 0.0010
Epoch 48/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 5.4692e-04 - mae: 0.0173 - mse: 5.4692e-04
Epoch 48: val_loss improved from 0.00039 to 0.00036, saving model to models/lstm_oracle_lite_20260113_073755_best.keras

Epoch 48: finished saving model to models/lstm_oracle_lite_20260113_073755_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 5.3711e-04 - mae: 0.0166 - mse: 5.3711e-04 - val_loss: 3.5841e-04 - val_mae: 0.0139 - val_mse: 3.5841e-04 - learning_rate: 0.0010
Epoch 49/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 5.8909e-04 - mae: 0.0178 - mse: 5.8909e-04 
Epoch 49: val_loss did not improve from 0.00036
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 5.9474e-04 - mae: 0.0173 - mse: 5.9474e-04 - val_loss: 5.2501e-04 - val_mae: 0.0177 - val_mse: 5.2501e-04 - learning_rate: 0.0010
Epoch 50/50
17/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 7.6656e-04 - mae: 0.0197 - mse: 7.6656e-04
Epoch 50: val_loss did not improve from 0.00036
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - loss: 6.7609e-04 - mae: 0.0188 - mse: 6.7609e-04 - val_loss: 3.9402e-04 - val_mae: 0.0152 - val_mse: 3.9402e-04 - learning_rate: 0.0010
Restoring model weights from the end of the best epoch: 48.

================================================================================
EVALUATING ON TEST SET
================================================================================
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 7.8933e-04 - mae: 0.0156 - mse: 7.8933e-04  

Test Results:
  loss: 0.000789
  compile_metrics: 0.015587

================================================================================
SAMPLE PREDICTIONS
================================================================================

Actual                                   | Predicted                               
--------------------------------------------------------------------------------
[0.719225, 0.295833, 0.875000]           | [0.736609, 0.282099, 0.838575]          
[0.276893, 0.018933, 0.520000]           | [0.282574, 0.014294, 0.521524]          
[0.215540, 0.014736, 0.719388]           | [0.210441, -0.004339, 0.681642]         
[0.200420, 0.028000, 0.500000]           | [0.210651, 0.020027, 0.509659]          
[0.176732, 0.102000, 0.630000]           | [0.165518, 0.088884, 0.636647]          
[0.686915, 0.121733, 0.760000]           | [0.699656, 0.129523, 0.758398]          
[0.687897, 0.064800, 0.720000]           | [0.735194, 0.066190, 0.721115]          
[0.694914, 0.123037, 0.777778]           | [0.697116, 0.092810, 0.760091]          
[0.240001, 0.057111, 0.527778]           | [0.219209, 0.044002, 0.517912]          
[0.116854, 0.055833, 0.775000]           | [0.103378, 0.057687, 0.765489]          

Final model saved to: models/lstm_oracle_lite_20260113_073755_final.keras
Metadata saved to: models/lstm_oracle_lite_20260113_073755_metadata.json

================================================================================
TRAINING COMPLETED SUCCESSFULLY!
================================================================================

Best model: models/lstm_oracle_lite_20260113_073755_best.keras
Training logs: logs/lstm_oracle_lite_20260113_073755

To view training progress:
  tensorboard --logdir=logs/lstm_oracle_lite_20260113_073755