================================================================================
MLP HYPERPARAMETER QUALITY PREDICTOR - TRAINING
================================================================================
Model Type: lite
Epochs: 50
Batch Size: 32
Learning Rate: 0.001
================================================================================
Loading dataset from: data/test_dataset.csv
Total samples: 1000
✓ Loaded metadata: 21 features, 3 targets
Training samples: 722
Validation samples: 128
Test samples: 150

Creating model...

================================================================================
MODEL ARCHITECTURE
================================================================================

Model: "mlp_prophet_lite"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense1 (Dense)                       │ (None, 128)                 │           2,816 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout1 (Dropout)                   │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense2 (Dense)                       │ (None, 64)                  │           8,256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout2 (Dropout)                   │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense3 (Dense)                       │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output (Dense)                       │ (None, 3)                   │              99 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 13,251 (51.76 KB)
 Trainable params: 13,251 (51.76 KB)
 Non-trainable params: 0 (0.00 B)

================================================================================
MODEL DETAILS
================================================================================
Total parameters: 13,251
Input shape: (None, 21)
Output shape: (None, 3)
================================================================================


Starting training...
Epoch 1/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 11s 505ms/step - loss: 0.4260 - mae: 0.5383 - mse: 0.4260
Epoch 1: val_loss improved from None to 0.03114, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 1: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.1433 - mae: 0.2825 - mse: 0.1433 - val_loss: 0.0311 - val_mae: 0.1311 - val_mse: 0.0311 - learning_rate: 0.0010
Epoch 2/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0790 - mae: 0.2179 - mse: 0.0790
Epoch 2: val_loss improved from 0.03114 to 0.02106, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 2: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0467 - mae: 0.1665 - mse: 0.0467 - val_loss: 0.0211 - val_mae: 0.1084 - val_mse: 0.0211 - learning_rate: 0.0010
Epoch 3/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0400 - mae: 0.1534 - mse: 0.0400
Epoch 3: val_loss improved from 0.02106 to 0.01349, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 3: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0307 - mae: 0.1353 - mse: 0.0307 - val_loss: 0.0135 - val_mae: 0.0847 - val_mse: 0.0135 - learning_rate: 0.0010
Epoch 4/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0373 - mae: 0.1400 - mse: 0.0373
Epoch 4: val_loss improved from 0.01349 to 0.01248, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 4: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0261 - mae: 0.1227 - mse: 0.0261 - val_loss: 0.0125 - val_mae: 0.0803 - val_mse: 0.0125 - learning_rate: 0.0010
Epoch 5/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0340 - mae: 0.1343 - mse: 0.0340
Epoch 5: val_loss improved from 0.01248 to 0.01089, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 5: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0219 - mae: 0.1143 - mse: 0.0219 - val_loss: 0.0109 - val_mae: 0.0756 - val_mse: 0.0109 - learning_rate: 0.0010
Epoch 6/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0333 - mae: 0.1337 - mse: 0.0333
Epoch 6: val_loss improved from 0.01089 to 0.01024, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 6: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0204 - mae: 0.1067 - mse: 0.0204 - val_loss: 0.0102 - val_mae: 0.0747 - val_mse: 0.0102 - learning_rate: 0.0010
Epoch 7/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0222 - mae: 0.1116 - mse: 0.0222
Epoch 7: val_loss improved from 0.01024 to 0.00953, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 7: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0184 - mae: 0.1029 - mse: 0.0184 - val_loss: 0.0095 - val_mae: 0.0709 - val_mse: 0.0095 - learning_rate: 0.0010
Epoch 8/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0205 - mae: 0.1090 - mse: 0.0205
Epoch 8: val_loss improved from 0.00953 to 0.00895, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 8: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0174 - mae: 0.1002 - mse: 0.0174 - val_loss: 0.0090 - val_mae: 0.0692 - val_mse: 0.0090 - learning_rate: 0.0010
Epoch 9/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0206 - mae: 0.1018 - mse: 0.0206
Epoch 9: val_loss improved from 0.00895 to 0.00852, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 9: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0141 - mae: 0.0896 - mse: 0.0141 - val_loss: 0.0085 - val_mae: 0.0665 - val_mse: 0.0085 - learning_rate: 0.0010
Epoch 10/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0192 - mae: 0.1046 - mse: 0.0192
Epoch 10: val_loss improved from 0.00852 to 0.00796, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 10: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0138 - mae: 0.0887 - mse: 0.0138 - val_loss: 0.0080 - val_mae: 0.0631 - val_mse: 0.0080 - learning_rate: 0.0010
Epoch 11/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0202 - mae: 0.1036 - mse: 0.0202
Epoch 11: val_loss did not improve from 0.00796
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0128 - mae: 0.0837 - mse: 0.0128 - val_loss: 0.0081 - val_mae: 0.0645 - val_mse: 0.0081 - learning_rate: 0.0010
Epoch 12/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0138 - mae: 0.0896 - mse: 0.0138
Epoch 12: val_loss did not improve from 0.00796
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0114 - mae: 0.0802 - mse: 0.0114 - val_loss: 0.0082 - val_mae: 0.0653 - val_mse: 0.0082 - learning_rate: 0.0010
Epoch 13/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0206 - mae: 0.1005 - mse: 0.0206
Epoch 13: val_loss improved from 0.00796 to 0.00782, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 13: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0114 - mae: 0.0780 - mse: 0.0114 - val_loss: 0.0078 - val_mae: 0.0619 - val_mse: 0.0078 - learning_rate: 0.0010
Epoch 14/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0160 - mae: 0.0939 - mse: 0.0160
Epoch 14: val_loss improved from 0.00782 to 0.00715, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 14: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0115 - mae: 0.0800 - mse: 0.0115 - val_loss: 0.0072 - val_mae: 0.0587 - val_mse: 0.0072 - learning_rate: 0.0010
Epoch 15/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0168 - mae: 0.0899 - mse: 0.0168
Epoch 15: val_loss did not improve from 0.00715
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0110 - mae: 0.0776 - mse: 0.0110 - val_loss: 0.0076 - val_mae: 0.0619 - val_mse: 0.0076 - learning_rate: 0.0010
Epoch 16/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0182 - mae: 0.1018 - mse: 0.0182
Epoch 16: val_loss did not improve from 0.00715
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0104 - mae: 0.0756 - mse: 0.0104 - val_loss: 0.0074 - val_mae: 0.0606 - val_mse: 0.0074 - learning_rate: 0.0010
Epoch 17/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0123 - mae: 0.0795 - mse: 0.0123
Epoch 17: val_loss did not improve from 0.00715
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0101 - mae: 0.0737 - mse: 0.0101 - val_loss: 0.0080 - val_mae: 0.0624 - val_mse: 0.0080 - learning_rate: 0.0010
Epoch 18/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0152 - mae: 0.0884 - mse: 0.0152
Epoch 18: val_loss improved from 0.00715 to 0.00704, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 18: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0101 - mae: 0.0737 - mse: 0.0101 - val_loss: 0.0070 - val_mae: 0.0579 - val_mse: 0.0070 - learning_rate: 0.0010
Epoch 19/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0116 - mae: 0.0778 - mse: 0.0116
Epoch 19: val_loss improved from 0.00704 to 0.00627, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 19: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0093 - mae: 0.0702 - mse: 0.0093 - val_loss: 0.0063 - val_mae: 0.0537 - val_mse: 0.0063 - learning_rate: 0.0010
Epoch 20/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0105 - mae: 0.0744 - mse: 0.0105
Epoch 20: val_loss did not improve from 0.00627
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0087 - mae: 0.0687 - mse: 0.0087 - val_loss: 0.0066 - val_mae: 0.0567 - val_mse: 0.0066 - learning_rate: 0.0010
Epoch 21/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0094 - mae: 0.0716 - mse: 0.0094
Epoch 21: val_loss did not improve from 0.00627
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0079 - mae: 0.0657 - mse: 0.0079 - val_loss: 0.0065 - val_mae: 0.0566 - val_mse: 0.0065 - learning_rate: 0.0010
Epoch 22/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0099 - mae: 0.0734 - mse: 0.0099
Epoch 22: val_loss did not improve from 0.00627
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0079 - mae: 0.0663 - mse: 0.0079 - val_loss: 0.0065 - val_mae: 0.0553 - val_mse: 0.0065 - learning_rate: 0.0010
Epoch 23/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0106 - mae: 0.0747 - mse: 0.0106
Epoch 23: val_loss improved from 0.00627 to 0.00616, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 23: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0078 - mae: 0.0644 - mse: 0.0078 - val_loss: 0.0062 - val_mae: 0.0529 - val_mse: 0.0062 - learning_rate: 0.0010
Epoch 24/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0095 - mae: 0.0718 - mse: 0.0095
Epoch 24: val_loss did not improve from 0.00616
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0079 - mae: 0.0649 - mse: 0.0079 - val_loss: 0.0066 - val_mae: 0.0560 - val_mse: 0.0066 - learning_rate: 0.0010
Epoch 25/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0087 - mae: 0.0641 - mse: 0.0087
Epoch 25: val_loss did not improve from 0.00616
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0070 - mae: 0.0611 - mse: 0.0070 - val_loss: 0.0067 - val_mae: 0.0572 - val_mse: 0.0067 - learning_rate: 0.0010
Epoch 26/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0131 - mae: 0.0785 - mse: 0.0131
Epoch 26: val_loss improved from 0.00616 to 0.00546, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 26: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0072 - mae: 0.0617 - mse: 0.0072 - val_loss: 0.0055 - val_mae: 0.0495 - val_mse: 0.0055 - learning_rate: 0.0010
Epoch 27/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0109 - mae: 0.0774 - mse: 0.0109
Epoch 27: val_loss did not improve from 0.00546
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0069 - mae: 0.0603 - mse: 0.0069 - val_loss: 0.0060 - val_mae: 0.0537 - val_mse: 0.0060 - learning_rate: 0.0010
Epoch 28/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0121 - mae: 0.0756 - mse: 0.0121
Epoch 28: val_loss did not improve from 0.00546
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0071 - mae: 0.0612 - mse: 0.0071 - val_loss: 0.0061 - val_mae: 0.0536 - val_mse: 0.0061 - learning_rate: 0.0010
Epoch 29/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0085 - mae: 0.0665 - mse: 0.0085
Epoch 29: val_loss did not improve from 0.00546
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0071 - mae: 0.0596 - mse: 0.0071 - val_loss: 0.0056 - val_mae: 0.0507 - val_mse: 0.0056 - learning_rate: 0.0010
Epoch 30/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0096 - mae: 0.0712 - mse: 0.0096
Epoch 30: val_loss did not improve from 0.00546
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0066 - mae: 0.0589 - mse: 0.0066 - val_loss: 0.0060 - val_mae: 0.0531 - val_mse: 0.0060 - learning_rate: 0.0010
Epoch 31/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0087 - mae: 0.0672 - mse: 0.0087
Epoch 31: val_loss did not improve from 0.00546
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0063 - mae: 0.0568 - mse: 0.0063 - val_loss: 0.0058 - val_mae: 0.0525 - val_mse: 0.0058 - learning_rate: 0.0010
Epoch 32/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0123 - mae: 0.0755 - mse: 0.0123
Epoch 32: val_loss improved from 0.00546 to 0.00526, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 32: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0065 - mae: 0.0575 - mse: 0.0065 - val_loss: 0.0053 - val_mae: 0.0485 - val_mse: 0.0053 - learning_rate: 0.0010
Epoch 33/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0103 - mae: 0.0716 - mse: 0.0103
Epoch 33: val_loss did not improve from 0.00526
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0061 - mae: 0.0562 - mse: 0.0061 - val_loss: 0.0053 - val_mae: 0.0496 - val_mse: 0.0053 - learning_rate: 0.0010
Epoch 34/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0053 - mae: 0.0534 - mse: 0.0053
Epoch 34: val_loss improved from 0.00526 to 0.00498, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 34: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0059 - mae: 0.0551 - mse: 0.0059 - val_loss: 0.0050 - val_mae: 0.0474 - val_mse: 0.0050 - learning_rate: 0.0010
Epoch 35/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0098 - mae: 0.0692 - mse: 0.0098
Epoch 35: val_loss did not improve from 0.00498
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0059 - mae: 0.0539 - mse: 0.0059 - val_loss: 0.0056 - val_mae: 0.0517 - val_mse: 0.0056 - learning_rate: 0.0010
Epoch 36/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0086 - mae: 0.0651 - mse: 0.0086
Epoch 36: val_loss did not improve from 0.00498
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0056 - mae: 0.0524 - mse: 0.0056 - val_loss: 0.0051 - val_mae: 0.0474 - val_mse: 0.0051 - learning_rate: 0.0010
Epoch 37/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0091 - mae: 0.0607 - mse: 0.0091
Epoch 37: val_loss did not improve from 0.00498
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0054 - mae: 0.0523 - mse: 0.0054 - val_loss: 0.0052 - val_mae: 0.0486 - val_mse: 0.0052 - learning_rate: 0.0010
Epoch 38/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0098 - mae: 0.0665 - mse: 0.0098
Epoch 38: val_loss did not improve from 0.00498
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0056 - mae: 0.0521 - mse: 0.0056 - val_loss: 0.0051 - val_mae: 0.0480 - val_mse: 0.0051 - learning_rate: 0.0010
Epoch 39/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0057 - mae: 0.0552 - mse: 0.0057
Epoch 39: val_loss improved from 0.00498 to 0.00458, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 39: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0054 - mae: 0.0510 - mse: 0.0054 - val_loss: 0.0046 - val_mae: 0.0437 - val_mse: 0.0046 - learning_rate: 0.0010
Epoch 40/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0074 - mae: 0.0628 - mse: 0.0074
Epoch 40: val_loss did not improve from 0.00458
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0050 - mae: 0.0498 - mse: 0.0050 - val_loss: 0.0049 - val_mae: 0.0468 - val_mse: 0.0049 - learning_rate: 0.0010
Epoch 41/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0090 - mae: 0.0658 - mse: 0.0090
Epoch 41: val_loss did not improve from 0.00458
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0055 - mae: 0.0511 - mse: 0.0055 - val_loss: 0.0049 - val_mae: 0.0463 - val_mse: 0.0049 - learning_rate: 0.0010
Epoch 42/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0056 - mae: 0.0525 - mse: 0.0056
Epoch 42: val_loss did not improve from 0.00458
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0050 - mae: 0.0496 - mse: 0.0050 - val_loss: 0.0054 - val_mae: 0.0497 - val_mse: 0.0054 - learning_rate: 0.0010
Epoch 43/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0067 - mae: 0.0605 - mse: 0.0067
Epoch 43: val_loss did not improve from 0.00458
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0051 - mae: 0.0504 - mse: 0.0051 - val_loss: 0.0050 - val_mae: 0.0457 - val_mse: 0.0050 - learning_rate: 0.0010
Epoch 44/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0080 - mae: 0.0559 - mse: 0.0080
Epoch 44: val_loss did not improve from 0.00458
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0047 - mae: 0.0474 - mse: 0.0047 - val_loss: 0.0049 - val_mae: 0.0457 - val_mse: 0.0049 - learning_rate: 0.0010
Epoch 45/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0080 - mae: 0.0591 - mse: 0.0080
Epoch 45: val_loss did not improve from 0.00458
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0047 - mae: 0.0474 - mse: 0.0047 - val_loss: 0.0050 - val_mae: 0.0465 - val_mse: 0.0050 - learning_rate: 0.0010
Epoch 46/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0064 - mae: 0.0575 - mse: 0.0064
Epoch 46: val_loss did not improve from 0.00458
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0045 - mae: 0.0472 - mse: 0.0045 - val_loss: 0.0048 - val_mae: 0.0448 - val_mse: 0.0048 - learning_rate: 0.0010
Epoch 47/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0066 - mae: 0.0556 - mse: 0.0066
Epoch 47: val_loss improved from 0.00458 to 0.00453, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 47: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.0045 - mae: 0.0463 - mse: 0.0045 - val_loss: 0.0045 - val_mae: 0.0417 - val_mse: 0.0045 - learning_rate: 0.0010
Epoch 48/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0070 - mae: 0.0565 - mse: 0.0070
Epoch 48: val_loss improved from 0.00453 to 0.00444, saving model to models/mlp_prophet_lite_20260113_073440_best.keras

Epoch 48: finished saving model to models/mlp_prophet_lite_20260113_073440_best.keras
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0046 - mae: 0.0459 - mse: 0.0046 - val_loss: 0.0044 - val_mae: 0.0417 - val_mse: 0.0044 - learning_rate: 0.0010
Epoch 49/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0061 - mae: 0.0580 - mse: 0.0061
Epoch 49: val_loss did not improve from 0.00444
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0046 - mae: 0.0463 - mse: 0.0046 - val_loss: 0.0045 - val_mae: 0.0430 - val_mse: 0.0045 - learning_rate: 0.0010
Epoch 50/50
 1/23 ━━━━━━━━━━━━━━━━━━━━ 0s 6ms/step - loss: 0.0074 - mae: 0.0558 - mse: 0.0074
Epoch 50: val_loss did not improve from 0.00444
23/23 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0048 - mae: 0.0474 - mse: 0.0048 - val_loss: 0.0045 - val_mae: 0.0421 - val_mse: 0.0045 - learning_rate: 0.0010
Restoring model weights from the end of the best epoch: 48.

================================================================================
EVALUATING ON TEST SET
================================================================================
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.0073 - mae: 0.0477 - mse: 0.0073 

Test Results:
  loss: 0.007341
  compile_metrics: 0.047676

================================================================================
SAMPLE PREDICTIONS
================================================================================

Actual                                   | Predicted                               
--------------------------------------------------------------------------------
[0.719225, 0.295833, 0.875000]           | [0.627941, 0.224761, 0.675774]          
[0.276893, 0.018933, 0.520000]           | [0.242646, 0.066622, 0.537066]          
[0.215540, 0.014736, 0.719388]           | [0.137379, 0.022544, 0.654441]          
[0.200420, 0.028000, 0.500000]           | [0.207702, 0.070607, 0.518522]          
[0.176732, 0.102000, 0.630000]           | [0.164206, 0.037925, 0.610411]          
[0.686915, 0.121733, 0.760000]           | [0.683083, 0.096184, 0.711728]          
[0.687897, 0.064800, 0.720000]           | [0.652064, 0.091278, 0.700852]          
[0.694914, 0.123037, 0.777778]           | [0.662333, 0.100347, 0.717561]          
[0.240001, 0.057111, 0.527778]           | [0.239707, 0.044623, 0.542529]          
[0.116854, 0.055833, 0.775000]           | [0.096448, 0.049657, 0.724486]          

Final model saved to: models/mlp_prophet_lite_20260113_073440_final.keras
Scaler saved to: models/mlp_prophet_lite_20260113_073440_scaler.pkl
Metadata saved to: models/mlp_prophet_lite_20260113_073440_metadata.json

================================================================================
TRAINING COMPLETED SUCCESSFULLY!
================================================================================

Best model: models/mlp_prophet_lite_20260113_073440_best.keras
Training logs: logs/mlp_prophet_lite_20260113_073440

To view training progress:
  tensorboard --logdir=logs/mlp_prophet_lite_20260113_073440