python3 ./cnn/src/train.py --dataset ./test/results/20260112_140511/dataset.csv
/Users/tomas/OSU/Python/NexusSom/app/venv_tf/lib/python3.11/site-packages/keras/src/export/tf2onnx_lib.py:8: FutureWarning: In the future `np.object` will be defined as the corresponding NumPy scalar.
  if not hasattr(np, "object"):
================================================================================
SOM QUALITY PREDICTION - TRAINING
================================================================================
Model Type: standard
Epochs: 50
Batch Size: 32
Image Size: (224, 224)
Learning Rate: 0.001
================================================================================
Loading dataset from: ./test/results/20260112_140511/dataset.csv
Total samples: 1000
Training samples: 722
Validation samples: 128
Test samples: 150

Creating data generators...

Creating model...
================================================================================
MODEL ARCHITECTURE SUMMARY
================================================================================
Model: "SOM_Quality_CNN"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ image_input (InputLayer)             │ (None, 224, 224, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1_1 (Conv2D)                     │ (None, 224, 224, 32)        │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn1_1 (BatchNormalization)           │ (None, 224, 224, 32)        │             128 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ relu1_1 (Activation)                 │ (None, 224, 224, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv1_2 (Conv2D)                     │ (None, 224, 224, 32)        │           9,248 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn1_2 (BatchNormalization)           │ (None, 224, 224, 32)        │             128 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ relu1_2 (Activation)                 │ (None, 224, 224, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ pool1 (MaxPooling2D)                 │ (None, 112, 112, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout1 (Dropout)                   │ (None, 112, 112, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2_1 (Conv2D)                     │ (None, 112, 112, 64)        │          18,496 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn2_1 (BatchNormalization)           │ (None, 112, 112, 64)        │             256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ relu2_1 (Activation)                 │ (None, 112, 112, 64)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2_2 (Conv2D)                     │ (None, 112, 112, 64)        │          36,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn2_2 (BatchNormalization)           │ (None, 112, 112, 64)        │             256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ relu2_2 (Activation)                 │ (None, 112, 112, 64)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ pool2 (MaxPooling2D)                 │ (None, 56, 56, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout2 (Dropout)                   │ (None, 56, 56, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv3_1 (Conv2D)                     │ (None, 56, 56, 128)         │          73,856 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn3_1 (BatchNormalization)           │ (None, 56, 56, 128)         │             512 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ relu3_1 (Activation)                 │ (None, 56, 56, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv3_2 (Conv2D)                     │ (None, 56, 56, 128)         │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn3_2 (BatchNormalization)           │ (None, 56, 56, 128)         │             512 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ relu3_2 (Activation)                 │ (None, 56, 56, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ pool3 (MaxPooling2D)                 │ (None, 28, 28, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout3 (Dropout)                   │ (None, 28, 28, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv4_1 (Conv2D)                     │ (None, 28, 28, 256)         │         295,168 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn4_1 (BatchNormalization)           │ (None, 28, 28, 256)         │           1,024 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ relu4_1 (Activation)                 │ (None, 28, 28, 256)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv4_2 (Conv2D)                     │ (None, 28, 28, 256)         │         590,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn4_2 (BatchNormalization)           │ (None, 28, 28, 256)         │           1,024 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ relu4_2 (Activation)                 │ (None, 28, 28, 256)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ pool4 (MaxPooling2D)                 │ (None, 14, 14, 256)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout4 (Dropout)                   │ (None, 14, 14, 256)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ global_avg_pool                      │ (None, 256)                 │               0 │
│ (GlobalAveragePooling2D)             │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense1 (Dense)                       │ (None, 256)                 │          65,792 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn_dense1 (BatchNormalization)       │ (None, 256)                 │           1,024 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ relu_dense1 (Activation)             │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_dense1 (Dropout)             │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense2 (Dense)                       │ (None, 128)                 │          32,896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn_dense2 (BatchNormalization)       │ (None, 128)                 │             512 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ relu_dense2 (Activation)             │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_dense2 (Dropout)             │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ quality_output (Dense)               │ (None, 1)                   │             129 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 1,276,449 (4.87 MB)
 Trainable params: 1,273,761 (4.86 MB)
 Non-trainable params: 2,688 (10.50 KB)
================================================================================
Total parameters: 1,276,449
Trainable parameters: 1,273,761
================================================================================

Starting training...
Training batches per epoch: 23
Validation batches per epoch: 4
Epoch 1/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.6172 - mae: 0.3702 - mse: 0.1945 - rmse: 0.4392  
Epoch 1: val_loss improved from None to 0.55673, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 1: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 66s 3s/step - loss: 0.5771 - mae: 0.3252 - mse: 0.1596 - rmse: 0.3994 - val_loss: 0.5567 - val_mae: 0.3732 - val_mse: 0.1517 - val_rmse: 0.3895 - learning_rate: 0.0010
Epoch 2/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.4823 - mae: 0.1965 - mse: 0.0833 - rmse: 0.2883 
Epoch 2: val_loss improved from 0.55673 to 0.48968, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 2: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 57s 3s/step - loss: 0.4741 - mae: 0.1845 - mse: 0.0810 - rmse: 0.2847 - val_loss: 0.4897 - val_mae: 0.1886 - val_mse: 0.1105 - val_rmse: 0.3324 - learning_rate: 0.0010
Epoch 3/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.4338 - mae: 0.1531 - mse: 0.0608 - rmse: 0.2460 
Epoch 3: val_loss improved from 0.48968 to 0.46515, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 3: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 51s 2s/step - loss: 0.4294 - mae: 0.1498 - mse: 0.0624 - rmse: 0.2498 - val_loss: 0.4651 - val_mae: 0.1844 - val_mse: 0.1119 - val_rmse: 0.3345 - learning_rate: 0.0010
Epoch 4/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.3925 - mae: 0.1143 - mse: 0.0454 - rmse: 0.2126 
Epoch 4: val_loss improved from 0.46515 to 0.44677, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 4: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 55s 2s/step - loss: 0.3801 - mae: 0.1062 - mse: 0.0390 - rmse: 0.1974 - val_loss: 0.4468 - val_mae: 0.1347 - val_mse: 0.1194 - val_rmse: 0.3456 - learning_rate: 0.0010
Epoch 5/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.3727 - mae: 0.1135 - mse: 0.0514 - rmse: 0.2249 
Epoch 5: val_loss improved from 0.44677 to 0.42465, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 5: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.3662 - mae: 0.1131 - mse: 0.0507 - rmse: 0.2251 - val_loss: 0.4246 - val_mae: 0.1280 - val_mse: 0.1223 - val_rmse: 0.3497 - learning_rate: 0.0010
Epoch 6/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.3526 - mae: 0.1108 - mse: 0.0560 - rmse: 0.2358 
Epoch 6: val_loss improved from 0.42465 to 0.40173, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 6: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 48s 2s/step - loss: 0.3398 - mae: 0.1046 - mse: 0.0486 - rmse: 0.2205 - val_loss: 0.4017 - val_mae: 0.1279 - val_mse: 0.1228 - val_rmse: 0.3505 - learning_rate: 0.0010
Epoch 7/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.3127 - mae: 0.0930 - mse: 0.0391 - rmse: 0.1971 
Epoch 7: val_loss improved from 0.40173 to 0.38144, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 7: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.3043 - mae: 0.0888 - mse: 0.0356 - rmse: 0.1886 - val_loss: 0.3814 - val_mae: 0.1258 - val_mse: 0.1241 - val_rmse: 0.3523 - learning_rate: 0.0010
Epoch 8/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.3046 - mae: 0.0990 - mse: 0.0522 - rmse: 0.2277 
Epoch 8: val_loss improved from 0.38144 to 0.36138, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 8: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.2908 - mae: 0.0902 - mse: 0.0431 - rmse: 0.2075 - val_loss: 0.3614 - val_mae: 0.1251 - val_mse: 0.1243 - val_rmse: 0.3525 - learning_rate: 0.0010
Epoch 9/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.2667 - mae: 0.0849 - mse: 0.0342 - rmse: 0.1847 
Epoch 9: val_loss improved from 0.36138 to 0.34280, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 9: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 49s 2s/step - loss: 0.2608 - mae: 0.0823 - mse: 0.0327 - rmse: 0.1808 - val_loss: 0.3428 - val_mae: 0.1249 - val_mse: 0.1247 - val_rmse: 0.3531 - learning_rate: 0.0010
Epoch 10/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.2559 - mae: 0.0812 - mse: 0.0422 - rmse: 0.2034 
Epoch 10: val_loss improved from 0.34280 to 0.32421, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 10: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 49s 2s/step - loss: 0.2527 - mae: 0.0816 - mse: 0.0430 - rmse: 0.2074 - val_loss: 0.3242 - val_mae: 0.1251 - val_mse: 0.1237 - val_rmse: 0.3518 - learning_rate: 0.0010
Epoch 11/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.2397 - mae: 0.0882 - mse: 0.0431 - rmse: 0.2063 
Epoch 11: val_loss improved from 0.32421 to 0.30781, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 11: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.2358 - mae: 0.0844 - mse: 0.0428 - rmse: 0.2068 - val_loss: 0.3078 - val_mae: 0.1255 - val_mse: 0.1228 - val_rmse: 0.3505 - learning_rate: 0.0010
Epoch 12/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.2138 - mae: 0.0722 - mse: 0.0323 - rmse: 0.1782 
Epoch 12: val_loss improved from 0.30781 to 0.29345, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 12: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.2134 - mae: 0.0752 - mse: 0.0353 - rmse: 0.1879 - val_loss: 0.2934 - val_mae: 0.1243 - val_mse: 0.1230 - val_rmse: 0.3508 - learning_rate: 0.0010
Epoch 13/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.1910 - mae: 0.0650 - mse: 0.0239 - rmse: 0.1540 
Epoch 13: val_loss improved from 0.29345 to 0.26980, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 13: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.1906 - mae: 0.0674 - mse: 0.0267 - rmse: 0.1634 - val_loss: 0.2698 - val_mae: 0.1187 - val_mse: 0.1130 - val_rmse: 0.3362 - learning_rate: 0.0010
Epoch 14/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.1946 - mae: 0.0805 - mse: 0.0409 - rmse: 0.2017 
Epoch 14: val_loss improved from 0.26980 to 0.26669, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 14: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 51s 2s/step - loss: 0.1985 - mae: 0.0861 - mse: 0.0475 - rmse: 0.2180 - val_loss: 0.2667 - val_mae: 0.1249 - val_mse: 0.1221 - val_rmse: 0.3494 - learning_rate: 0.0010
Epoch 15/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.1824 - mae: 0.0795 - mse: 0.0404 - rmse: 0.1992 
Epoch 15: val_loss improved from 0.26669 to 0.23715, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 15: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 48s 2s/step - loss: 0.1744 - mae: 0.0747 - mse: 0.0350 - rmse: 0.1870 - val_loss: 0.2371 - val_mae: 0.1108 - val_mse: 0.1036 - val_rmse: 0.3218 - learning_rate: 0.0010
Epoch 16/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.1540 - mae: 0.0604 - mse: 0.0230 - rmse: 0.1497 
Epoch 16: val_loss did not improve from 0.23715
23/23 ━━━━━━━━━━━━━━━━━━━━ 47s 2s/step - loss: 0.1554 - mae: 0.0650 - mse: 0.0267 - rmse: 0.1633 - val_loss: 0.2384 - val_mae: 0.1223 - val_mse: 0.1152 - val_rmse: 0.3394 - learning_rate: 0.0010
Epoch 17/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.1456 - mae: 0.0622 - mse: 0.0247 - rmse: 0.1554 
Epoch 17: val_loss improved from 0.23715 to 0.20979, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 17: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 48s 2s/step - loss: 0.1477 - mae: 0.0681 - mse: 0.0290 - rmse: 0.1704 - val_loss: 0.2098 - val_mae: 0.1090 - val_mse: 0.0961 - val_rmse: 0.3099 - learning_rate: 0.0010
Epoch 18/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.1301 - mae: 0.0541 - mse: 0.0185 - rmse: 0.1357 
Epoch 18: val_loss improved from 0.20979 to 0.18650, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 18: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 51s 2s/step - loss: 0.1318 - mae: 0.0587 - mse: 0.0222 - rmse: 0.1491 - val_loss: 0.1865 - val_mae: 0.1009 - val_mse: 0.0815 - val_rmse: 0.2855 - learning_rate: 0.0010
Epoch 19/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.1220 - mae: 0.0584 - mse: 0.0189 - rmse: 0.1372 
Epoch 19: val_loss improved from 0.18650 to 0.18256, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 19: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.1206 - mae: 0.0569 - mse: 0.0193 - rmse: 0.1389 - val_loss: 0.1826 - val_mae: 0.1143 - val_mse: 0.0853 - val_rmse: 0.2920 - learning_rate: 0.0010
Epoch 20/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.1185 - mae: 0.0535 - mse: 0.0230 - rmse: 0.1508 
Epoch 20: val_loss improved from 0.18256 to 0.15499, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 20: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 49s 2s/step - loss: 0.1194 - mae: 0.0558 - mse: 0.0255 - rmse: 0.1597 - val_loss: 0.1550 - val_mae: 0.1064 - val_mse: 0.0649 - val_rmse: 0.2548 - learning_rate: 0.0010
Epoch 21/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.1129 - mae: 0.0605 - mse: 0.0244 - rmse: 0.1547 
Epoch 21: val_loss did not improve from 0.15499
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.1148 - mae: 0.0657 - mse: 0.0278 - rmse: 0.1666 - val_loss: 0.2178 - val_mae: 0.1895 - val_mse: 0.1340 - val_rmse: 0.3661 - learning_rate: 0.0010
Epoch 22/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.1244 - mae: 0.0814 - mse: 0.0419 - rmse: 0.2043 
Epoch 22: val_loss improved from 0.15499 to 0.13817, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 22: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 51s 2s/step - loss: 0.1178 - mae: 0.0755 - mse: 0.0365 - rmse: 0.1910 - val_loss: 0.1382 - val_mae: 0.0976 - val_mse: 0.0596 - val_rmse: 0.2442 - learning_rate: 0.0010
Epoch 23/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.1095 - mae: 0.0640 - mse: 0.0321 - rmse: 0.1789 
Epoch 23: val_loss improved from 0.13817 to 0.12654, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 23: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 49s 2s/step - loss: 0.1041 - mae: 0.0600 - mse: 0.0279 - rmse: 0.1669 - val_loss: 0.1265 - val_mae: 0.0808 - val_mse: 0.0529 - val_rmse: 0.2300 - learning_rate: 0.0010
Epoch 24/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.1208 - mae: 0.0800 - mse: 0.0484 - rmse: 0.2191 
Epoch 24: val_loss did not improve from 0.12654
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.1138 - mae: 0.0750 - mse: 0.0425 - rmse: 0.2062 - val_loss: 0.1531 - val_mae: 0.0972 - val_mse: 0.0844 - val_rmse: 0.2905 - learning_rate: 0.0010
Epoch 25/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0936 - mae: 0.0589 - mse: 0.0260 - rmse: 0.1610 
Epoch 25: val_loss did not improve from 0.12654
23/23 ━━━━━━━━━━━━━━━━━━━━ 49s 2s/step - loss: 0.0920 - mae: 0.0572 - mse: 0.0255 - rmse: 0.1597 - val_loss: 0.1512 - val_mae: 0.0990 - val_mse: 0.0872 - val_rmse: 0.2953 - learning_rate: 0.0010
Epoch 26/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0894 - mae: 0.0584 - mse: 0.0265 - rmse: 0.1617 
Epoch 26: val_loss improved from 0.12654 to 0.12426, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 26: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.0854 - mae: 0.0565 - mse: 0.0235 - rmse: 0.1534 - val_loss: 0.1243 - val_mae: 0.0840 - val_mse: 0.0649 - val_rmse: 0.2547 - learning_rate: 0.0010
Epoch 27/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0881 - mae: 0.0661 - mse: 0.0296 - rmse: 0.1712 
Epoch 27: val_loss improved from 0.12426 to 0.09424, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 27: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.0828 - mae: 0.0591 - mse: 0.0252 - rmse: 0.1586 - val_loss: 0.0942 - val_mae: 0.0646 - val_mse: 0.0387 - val_rmse: 0.1966 - learning_rate: 0.0010
Epoch 28/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0892 - mae: 0.0693 - mse: 0.0343 - rmse: 0.1842 
Epoch 28: val_loss did not improve from 0.09424
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.0822 - mae: 0.0606 - mse: 0.0280 - rmse: 0.1674 - val_loss: 0.0987 - val_mae: 0.0852 - val_mse: 0.0462 - val_rmse: 0.2148 - learning_rate: 0.0010
Epoch 29/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0805 - mae: 0.0632 - mse: 0.0286 - rmse: 0.1679 
Epoch 29: val_loss did not improve from 0.09424
23/23 ━━━━━━━━━━━━━━━━━━━━ 48s 2s/step - loss: 0.0762 - mae: 0.0588 - mse: 0.0250 - rmse: 0.1580 - val_loss: 0.1537 - val_mae: 0.1196 - val_mse: 0.1040 - val_rmse: 0.3224 - learning_rate: 0.0010
Epoch 30/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0723 - mae: 0.0585 - mse: 0.0232 - rmse: 0.1516 
Epoch 30: val_loss did not improve from 0.09424
23/23 ━━━━━━━━━━━━━━━━━━━━ 48s 2s/step - loss: 0.0782 - mae: 0.0657 - mse: 0.0298 - rmse: 0.1727 - val_loss: 0.1005 - val_mae: 0.1047 - val_mse: 0.0535 - val_rmse: 0.2314 - learning_rate: 0.0010
Epoch 31/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0714 - mae: 0.0550 - mse: 0.0250 - rmse: 0.1572 
Epoch 31: val_loss did not improve from 0.09424
23/23 ━━━━━━━━━━━━━━━━━━━━ 49s 2s/step - loss: 0.0724 - mae: 0.0579 - mse: 0.0265 - rmse: 0.1629 - val_loss: 0.1593 - val_mae: 0.1239 - val_mse: 0.1149 - val_rmse: 0.3390 - learning_rate: 0.0010
Epoch 32/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0612 - mae: 0.0489 - mse: 0.0175 - rmse: 0.1307 
Epoch 32: val_loss did not improve from 0.09424

Epoch 32: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.0647 - mae: 0.0511 - mse: 0.0216 - rmse: 0.1470 - val_loss: 0.1372 - val_mae: 0.1051 - val_mse: 0.0956 - val_rmse: 0.3092 - learning_rate: 0.0010
Epoch 33/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0683 - mae: 0.0554 - mse: 0.0270 - rmse: 0.1631 
Epoch 33: val_loss did not improve from 0.09424
23/23 ━━━━━━━━━━━━━━━━━━━━ 51s 2s/step - loss: 0.0642 - mae: 0.0491 - mse: 0.0233 - rmse: 0.1525 - val_loss: 0.1312 - val_mae: 0.1057 - val_mse: 0.0910 - val_rmse: 0.3017 - learning_rate: 5.0000e-04
Epoch 34/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0554 - mae: 0.0415 - mse: 0.0156 - rmse: 0.1239 
Epoch 34: val_loss improved from 0.09424 to 0.06435, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 34: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 51s 2s/step - loss: 0.0594 - mae: 0.0456 - mse: 0.0199 - rmse: 0.1409 - val_loss: 0.0643 - val_mae: 0.0586 - val_mse: 0.0256 - val_rmse: 0.1600 - learning_rate: 5.0000e-04
Epoch 35/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0589 - mae: 0.0408 - mse: 0.0204 - rmse: 0.1420 
Epoch 35: val_loss did not improve from 0.06435
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.0614 - mae: 0.0460 - mse: 0.0232 - rmse: 0.1523 - val_loss: 0.0861 - val_mae: 0.0733 - val_mse: 0.0487 - val_rmse: 0.2206 - learning_rate: 5.0000e-04
Epoch 36/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0578 - mae: 0.0444 - mse: 0.0206 - rmse: 0.1432 
Epoch 36: val_loss did not improve from 0.06435
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.0548 - mae: 0.0432 - mse: 0.0180 - rmse: 0.1341 - val_loss: 0.1292 - val_mae: 0.1048 - val_mse: 0.0931 - val_rmse: 0.3051 - learning_rate: 5.0000e-04
Epoch 37/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0497 - mae: 0.0360 - mse: 0.0139 - rmse: 0.1166 
Epoch 37: val_loss did not improve from 0.06435
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.0532 - mae: 0.0416 - mse: 0.0178 - rmse: 0.1333 - val_loss: 0.1368 - val_mae: 0.1145 - val_mse: 0.1020 - val_rmse: 0.3193 - learning_rate: 5.0000e-04
Epoch 38/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0579 - mae: 0.0466 - mse: 0.0234 - rmse: 0.1522 
Epoch 38: val_loss did not improve from 0.06435
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.0568 - mae: 0.0478 - mse: 0.0225 - rmse: 0.1501 - val_loss: 0.0943 - val_mae: 0.0851 - val_mse: 0.0606 - val_rmse: 0.2462 - learning_rate: 5.0000e-04
Epoch 39/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0572 - mae: 0.0483 - mse: 0.0237 - rmse: 0.1530 
Epoch 39: val_loss did not improve from 0.06435

Epoch 39: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.0544 - mae: 0.0489 - mse: 0.0212 - rmse: 0.1455 - val_loss: 0.0712 - val_mae: 0.0741 - val_mse: 0.0385 - val_rmse: 0.1961 - learning_rate: 5.0000e-04
Epoch 40/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0519 - mae: 0.0423 - mse: 0.0193 - rmse: 0.1355     
Epoch 40: val_loss improved from 0.06435 to 0.05398, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 40: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.0517 - mae: 0.0422 - mse: 0.0192 - rmse: 0.1385 - val_loss: 0.0540 - val_mae: 0.0696 - val_mse: 0.0218 - val_rmse: 0.1476 - learning_rate: 2.5000e-04
Epoch 41/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0506 - mae: 0.0424 - mse: 0.0185 - rmse: 0.1358 
Epoch 41: val_loss improved from 0.05398 to 0.04795, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 41: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 50s 2s/step - loss: 0.0513 - mae: 0.0445 - mse: 0.0194 - rmse: 0.1393 - val_loss: 0.0479 - val_mae: 0.0523 - val_mse: 0.0164 - val_rmse: 0.1279 - learning_rate: 2.5000e-04
Epoch 42/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0468 - mae: 0.0377 - mse: 0.0153 - rmse: 0.1210     
Epoch 42: val_loss did not improve from 0.04795
23/23 ━━━━━━━━━━━━━━━━━━━━ 49s 2s/step - loss: 0.0481 - mae: 0.0393 - mse: 0.0168 - rmse: 0.1298 - val_loss: 0.0995 - val_mae: 0.0846 - val_mse: 0.0685 - val_rmse: 0.2617 - learning_rate: 2.5000e-04
Epoch 43/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0405 - mae: 0.0306 - mse: 0.0096 - rmse: 0.0964 
Epoch 43: val_loss did not improve from 0.04795
23/23 ━━━━━━━━━━━━━━━━━━━━ 49s 2s/step - loss: 0.0457 - mae: 0.0371 - mse: 0.0149 - rmse: 0.1222 - val_loss: 0.0518 - val_mae: 0.0465 - val_mse: 0.0214 - val_rmse: 0.1462 - learning_rate: 2.5000e-04
Epoch 44/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0440 - mae: 0.0364 - mse: 0.0137 - rmse: 0.1168 
Epoch 44: val_loss did not improve from 0.04795
23/23 ━━━━━━━━━━━━━━━━━━━━ 51s 2s/step - loss: 0.0437 - mae: 0.0371 - mse: 0.0136 - rmse: 0.1166 - val_loss: 0.0774 - val_mae: 0.0656 - val_mse: 0.0475 - val_rmse: 0.2181 - learning_rate: 2.5000e-04
Epoch 45/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0422 - mae: 0.0334 - mse: 0.0125 - rmse: 0.1108 
Epoch 45: val_loss improved from 0.04795 to 0.04476, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 45: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 49s 2s/step - loss: 0.0446 - mae: 0.0358 - mse: 0.0150 - rmse: 0.1226 - val_loss: 0.0448 - val_mae: 0.0424 - val_mse: 0.0155 - val_rmse: 0.1247 - learning_rate: 2.5000e-04
Epoch 46/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0522 - mae: 0.0454 - mse: 0.0231 - rmse: 0.1483 
Epoch 46: val_loss improved from 0.04476 to 0.04314, saving model to models/som_quality_standard_20260112_210424_best.keras

Epoch 46: finished saving model to models/som_quality_standard_20260112_210424_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 49s 2s/step - loss: 0.0434 - mae: 0.0364 - mse: 0.0144 - rmse: 0.1200 - val_loss: 0.0431 - val_mae: 0.0391 - val_mse: 0.0145 - val_rmse: 0.1202 - learning_rate: 2.5000e-04
Epoch 47/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0402 - mae: 0.0340 - mse: 0.0117 - rmse: 0.1070 
Epoch 47: val_loss did not improve from 0.04314
23/23 ━━━━━━━━━━━━━━━━━━━━ 48s 2s/step - loss: 0.0403 - mae: 0.0337 - mse: 0.0119 - rmse: 0.1091 - val_loss: 0.0559 - val_mae: 0.0502 - val_mse: 0.0278 - val_rmse: 0.1668 - learning_rate: 2.5000e-04
Epoch 48/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0397 - mae: 0.0305 - mse: 0.0117 - rmse: 0.1075 
Epoch 48: val_loss did not improve from 0.04314
23/23 ━━━━━━━━━━━━━━━━━━━━ 48s 2s/step - loss: 0.0424 - mae: 0.0357 - mse: 0.0146 - rmse: 0.1208 - val_loss: 0.0896 - val_mae: 0.0749 - val_mse: 0.0620 - val_rmse: 0.2489 - learning_rate: 2.5000e-04
Epoch 49/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0405 - mae: 0.0338 - mse: 0.0131 - rmse: 0.1128 
Epoch 49: val_loss did not improve from 0.04314
23/23 ━━━━━━━━━━━━━━━━━━━━ 49s 2s/step - loss: 0.0405 - mae: 0.0344 - mse: 0.0131 - rmse: 0.1145 - val_loss: 0.0687 - val_mae: 0.0593 - val_mse: 0.0416 - val_rmse: 0.2039 - learning_rate: 2.5000e-04
Epoch 50/50
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2s/step - loss: 0.0408 - mae: 0.0356 - mse: 0.0138 - rmse: 0.1169 
Epoch 50: val_loss did not improve from 0.04314
23/23 ━━━━━━━━━━━━━━━━━━━━ 51s 2s/step - loss: 0.0390 - mae: 0.0335 - mse: 0.0121 - rmse: 0.1101 - val_loss: 0.0539 - val_mae: 0.0487 - val_mse: 0.0272 - val_rmse: 0.1650 - learning_rate: 2.5000e-04
Restoring model weights from the end of the best epoch: 46.

================================================================================
EVALUATING ON TEST SET
================================================================================
5/5 ━━━━━━━━━━━━━━━━━━━━ 5s 1s/step - loss: 0.0353 - mae: 0.0301 - mse: 0.0066 - rmse: 0.0812   

Test Results:
  loss: 0.035270
  compile_metrics: 0.030074

Final model saved to: models/som_quality_standard_20260112_210424_final.keras
Test set saved to: models/som_quality_standard_20260112_210424_test_set.csv

================================================================================
TRAINING COMPLETED SUCCESSFULLY!
================================================================================

Best model: models/som_quality_standard_20260112_210424_best.keras
Training logs: logs/som_quality_standard_20260112_210424

To view training progress:
  tensorboard --logdir=logs/som_quality_standard_20260112_210424
