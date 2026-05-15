======================================================================
MLP TRAINING
======================================================================
Loading dataset: data/all_combined_mlp.csv
Total samples: 87
Features: 25  |  Targets: ['raw_mqe_improvement_ratio', 'raw_topographic_error', 'dead_neuron_ratio']
Train: 62  |  Val: 11  |  Test: 14
2026-05-14 16:52:26.544509: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

================================================================================
MODEL ARCHITECTURE
================================================================================

Model: "mlp_prophet"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ dense1 (Dense)                       │ (None, 256)                 │           6,656 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn1 (BatchNormalization)             │ (None, 256)                 │           1,024 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout1 (Dropout)                   │ (None, 256)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense2 (Dense)                       │ (None, 128)                 │          32,896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn2 (BatchNormalization)             │ (None, 128)                 │             512 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout2 (Dropout)                   │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense3 (Dense)                       │ (None, 64)                  │           8,256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bn3 (BatchNormalization)             │ (None, 64)                  │             256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout3 (Dropout)                   │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense4 (Dense)                       │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout4 (Dropout)                   │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output (Dense)                       │ (None, 3)                   │              99 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 51,779 (202.26 KB)
 Trainable params: 50,883 (198.76 KB)
 Non-trainable params: 896 (3.50 KB)

================================================================================
MODEL DETAILS
================================================================================
Total parameters: 51,779
Input shape: (None, 25)
Output shape: (None, 3)
================================================================================


Training (300 epochs max, early stopping patience=30)...
Epoch 1/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 1s 151ms/step - loss: 1.6929 - mae: 1.0601 - val_loss: 0.2540 - val_mae: 0.4318 - learning_rate: 0.0010
Epoch 2/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 1.7973 - mae: 1.0711 - val_loss: 0.2262 - val_mae: 0.4083 - learning_rate: 0.0010
Epoch 3/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 1.4550 - mae: 0.9931 - val_loss: 0.2067 - val_mae: 0.3889 - learning_rate: 0.0010
Epoch 4/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 1.2351 - mae: 0.8598 - val_loss: 0.1897 - val_mae: 0.3721 - learning_rate: 0.0010
Epoch 5/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 1.0883 - mae: 0.8567 - val_loss: 0.1766 - val_mae: 0.3575 - learning_rate: 0.0010
Epoch 6/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step - loss: 1.3631 - mae: 0.9020 - val_loss: 0.1659 - val_mae: 0.3458 - learning_rate: 0.0010
Epoch 7/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 1.2457 - mae: 0.8852 - val_loss: 0.1562 - val_mae: 0.3355 - learning_rate: 0.0010
Epoch 8/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 1.0772 - mae: 0.8251 - val_loss: 0.1467 - val_mae: 0.3246 - learning_rate: 0.0010
Epoch 9/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 1.2040 - mae: 0.8740 - val_loss: 0.1388 - val_mae: 0.3156 - learning_rate: 0.0010
Epoch 10/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 0.9818 - mae: 0.8011 - val_loss: 0.1305 - val_mae: 0.3056 - learning_rate: 0.0010
Epoch 11/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.7953 - mae: 0.7101 - val_loss: 0.1233 - val_mae: 0.2968 - learning_rate: 0.0010
Epoch 12/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 1.0427 - mae: 0.8017 - val_loss: 0.1180 - val_mae: 0.2889 - learning_rate: 0.0010
Epoch 13/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.8095 - mae: 0.7031 - val_loss: 0.1142 - val_mae: 0.2835 - learning_rate: 0.0010
Epoch 14/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.7253 - mae: 0.6306 - val_loss: 0.1105 - val_mae: 0.2787 - learning_rate: 0.0010
Epoch 15/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.8537 - mae: 0.7066 - val_loss: 0.1079 - val_mae: 0.2744 - learning_rate: 0.0010
Epoch 16/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.6475 - mae: 0.6038 - val_loss: 0.1054 - val_mae: 0.2702 - learning_rate: 0.0010
Epoch 17/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step - loss: 0.9323 - mae: 0.7379 - val_loss: 0.1027 - val_mae: 0.2661 - learning_rate: 0.0010
Epoch 18/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.8080 - mae: 0.6729 - val_loss: 0.1016 - val_mae: 0.2642 - learning_rate: 0.0010
Epoch 19/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.6698 - mae: 0.6598 - val_loss: 0.1015 - val_mae: 0.2639 - learning_rate: 0.0010
Epoch 20/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.7215 - mae: 0.6587 - val_loss: 0.1010 - val_mae: 0.2639 - learning_rate: 0.0010
Epoch 21/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.5810 - mae: 0.5965 - val_loss: 0.1001 - val_mae: 0.2630 - learning_rate: 0.0010
Epoch 22/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.6485 - mae: 0.5909 - val_loss: 0.0993 - val_mae: 0.2619 - learning_rate: 0.0010
Epoch 23/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.6316 - mae: 0.6350 - val_loss: 0.0981 - val_mae: 0.2607 - learning_rate: 0.0010
Epoch 24/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.6194 - mae: 0.6302 - val_loss: 0.0973 - val_mae: 0.2600 - learning_rate: 0.0010
Epoch 25/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.4878 - mae: 0.5625 - val_loss: 0.0958 - val_mae: 0.2588 - learning_rate: 0.0010
Epoch 26/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.5544 - mae: 0.5928 - val_loss: 0.0945 - val_mae: 0.2574 - learning_rate: 0.0010
Epoch 27/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.5837 - mae: 0.5872 - val_loss: 0.0936 - val_mae: 0.2568 - learning_rate: 0.0010
Epoch 28/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.5146 - mae: 0.5652 - val_loss: 0.0929 - val_mae: 0.2561 - learning_rate: 0.0010
Epoch 29/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.5937 - mae: 0.5868 - val_loss: 0.0924 - val_mae: 0.2564 - learning_rate: 0.0010
Epoch 30/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.5368 - mae: 0.5534 - val_loss: 0.0912 - val_mae: 0.2550 - learning_rate: 0.0010
Epoch 31/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.4981 - mae: 0.5547 - val_loss: 0.0899 - val_mae: 0.2529 - learning_rate: 0.0010
Epoch 32/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.4801 - mae: 0.5283 - val_loss: 0.0895 - val_mae: 0.2516 - learning_rate: 0.0010
Epoch 33/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.4075 - mae: 0.5051 - val_loss: 0.0888 - val_mae: 0.2498 - learning_rate: 0.0010
Epoch 34/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 0.4666 - mae: 0.5210 - val_loss: 0.0878 - val_mae: 0.2473 - learning_rate: 0.0010
Epoch 35/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.3890 - mae: 0.4863 - val_loss: 0.0859 - val_mae: 0.2435 - learning_rate: 0.0010
Epoch 36/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.4263 - mae: 0.5313 - val_loss: 0.0848 - val_mae: 0.2406 - learning_rate: 0.0010
Epoch 37/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.4525 - mae: 0.5044 - val_loss: 0.0837 - val_mae: 0.2384 - learning_rate: 0.0010
Epoch 38/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.4119 - mae: 0.4898 - val_loss: 0.0830 - val_mae: 0.2367 - learning_rate: 0.0010
Epoch 39/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.3990 - mae: 0.4819 - val_loss: 0.0829 - val_mae: 0.2361 - learning_rate: 0.0010
Epoch 40/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.4154 - mae: 0.4823 - val_loss: 0.0833 - val_mae: 0.2361 - learning_rate: 0.0010
Epoch 41/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.4795 - mae: 0.5434 - val_loss: 0.0833 - val_mae: 0.2358 - learning_rate: 0.0010
Epoch 42/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.4517 - mae: 0.5371 - val_loss: 0.0837 - val_mae: 0.2359 - learning_rate: 0.0010
Epoch 43/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.4052 - mae: 0.5183 - val_loss: 0.0829 - val_mae: 0.2346 - learning_rate: 0.0010
Epoch 44/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.3959 - mae: 0.4889 - val_loss: 0.0826 - val_mae: 0.2337 - learning_rate: 0.0010
Epoch 45/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.2786 - mae: 0.4155 - val_loss: 0.0823 - val_mae: 0.2338 - learning_rate: 0.0010
Epoch 46/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.3656 - mae: 0.4602 - val_loss: 0.0815 - val_mae: 0.2327 - learning_rate: 0.0010
Epoch 47/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.2965 - mae: 0.4348 - val_loss: 0.0803 - val_mae: 0.2304 - learning_rate: 0.0010
Epoch 48/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.3082 - mae: 0.4408 - val_loss: 0.0795 - val_mae: 0.2286 - learning_rate: 0.0010
Epoch 49/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.2861 - mae: 0.4282 - val_loss: 0.0783 - val_mae: 0.2265 - learning_rate: 0.0010
Epoch 50/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.4147 - mae: 0.4942 - val_loss: 0.0777 - val_mae: 0.2255 - learning_rate: 0.0010
Epoch 51/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.3005 - mae: 0.4351 - val_loss: 0.0765 - val_mae: 0.2239 - learning_rate: 0.0010
Epoch 52/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.3510 - mae: 0.4638 - val_loss: 0.0757 - val_mae: 0.2230 - learning_rate: 0.0010
Epoch 53/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.3987 - mae: 0.4950 - val_loss: 0.0749 - val_mae: 0.2222 - learning_rate: 0.0010
Epoch 54/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.3060 - mae: 0.4426 - val_loss: 0.0746 - val_mae: 0.2220 - learning_rate: 0.0010
Epoch 55/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.3036 - mae: 0.4451 - val_loss: 0.0734 - val_mae: 0.2201 - learning_rate: 0.0010
Epoch 56/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.2489 - mae: 0.3901 - val_loss: 0.0724 - val_mae: 0.2181 - learning_rate: 0.0010
Epoch 57/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.3317 - mae: 0.4690 - val_loss: 0.0718 - val_mae: 0.2168 - learning_rate: 0.0010
Epoch 58/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.3695 - mae: 0.4731 - val_loss: 0.0710 - val_mae: 0.2149 - learning_rate: 0.0010
Epoch 59/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.2862 - mae: 0.4194 - val_loss: 0.0700 - val_mae: 0.2129 - learning_rate: 0.0010
Epoch 60/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.2888 - mae: 0.4160 - val_loss: 0.0700 - val_mae: 0.2120 - learning_rate: 0.0010
Epoch 61/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.2689 - mae: 0.3973 - val_loss: 0.0691 - val_mae: 0.2096 - learning_rate: 0.0010
Epoch 62/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.2235 - mae: 0.3676 - val_loss: 0.0677 - val_mae: 0.2064 - learning_rate: 0.0010
Epoch 63/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 51ms/step - loss: 0.2913 - mae: 0.4207 - val_loss: 0.0668 - val_mae: 0.2039 - learning_rate: 0.0010
Epoch 64/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.2597 - mae: 0.4135 - val_loss: 0.0661 - val_mae: 0.2019 - learning_rate: 0.0010
Epoch 65/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.2613 - mae: 0.3918 - val_loss: 0.0655 - val_mae: 0.1999 - learning_rate: 0.0010
Epoch 66/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.2642 - mae: 0.3924 - val_loss: 0.0650 - val_mae: 0.1987 - learning_rate: 0.0010
Epoch 67/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.2688 - mae: 0.4086 - val_loss: 0.0642 - val_mae: 0.1969 - learning_rate: 0.0010
Epoch 68/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.2532 - mae: 0.3901 - val_loss: 0.0627 - val_mae: 0.1942 - learning_rate: 0.0010
Epoch 69/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.2610 - mae: 0.3980 - val_loss: 0.0605 - val_mae: 0.1907 - learning_rate: 0.0010
Epoch 70/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.2280 - mae: 0.3849 - val_loss: 0.0583 - val_mae: 0.1860 - learning_rate: 0.0010
Epoch 71/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.2032 - mae: 0.3567 - val_loss: 0.0569 - val_mae: 0.1831 - learning_rate: 0.0010
Epoch 72/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.2707 - mae: 0.4115 - val_loss: 0.0561 - val_mae: 0.1811 - learning_rate: 0.0010
Epoch 73/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1940 - mae: 0.3387 - val_loss: 0.0552 - val_mae: 0.1789 - learning_rate: 0.0010
Epoch 74/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.2298 - mae: 0.3838 - val_loss: 0.0541 - val_mae: 0.1767 - learning_rate: 0.0010
Epoch 75/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.2194 - mae: 0.3612 - val_loss: 0.0535 - val_mae: 0.1762 - learning_rate: 0.0010
Epoch 76/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.2229 - mae: 0.3733 - val_loss: 0.0532 - val_mae: 0.1756 - learning_rate: 0.0010
Epoch 77/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.2455 - mae: 0.3857 - val_loss: 0.0519 - val_mae: 0.1724 - learning_rate: 0.0010
Epoch 78/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.2208 - mae: 0.3553 - val_loss: 0.0512 - val_mae: 0.1703 - learning_rate: 0.0010
Epoch 79/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.2270 - mae: 0.3775 - val_loss: 0.0502 - val_mae: 0.1681 - learning_rate: 0.0010
Epoch 80/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1893 - mae: 0.3347 - val_loss: 0.0493 - val_mae: 0.1666 - learning_rate: 0.0010
Epoch 81/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1967 - mae: 0.3419 - val_loss: 0.0486 - val_mae: 0.1661 - learning_rate: 0.0010
Epoch 82/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.2102 - mae: 0.3634 - val_loss: 0.0486 - val_mae: 0.1670 - learning_rate: 0.0010
Epoch 83/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.2383 - mae: 0.3718 - val_loss: 0.0484 - val_mae: 0.1677 - learning_rate: 0.0010
Epoch 84/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.2088 - mae: 0.3718 - val_loss: 0.0482 - val_mae: 0.1678 - learning_rate: 0.0010
Epoch 85/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.1913 - mae: 0.3427 - val_loss: 0.0480 - val_mae: 0.1679 - learning_rate: 0.0010
Epoch 86/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.2129 - mae: 0.3691 - val_loss: 0.0475 - val_mae: 0.1673 - learning_rate: 0.0010
Epoch 87/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1789 - mae: 0.3355 - val_loss: 0.0475 - val_mae: 0.1678 - learning_rate: 0.0010
Epoch 88/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.1732 - mae: 0.3424 - val_loss: 0.0474 - val_mae: 0.1674 - learning_rate: 0.0010
Epoch 89/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.1357 - mae: 0.2923 - val_loss: 0.0475 - val_mae: 0.1675 - learning_rate: 0.0010
Epoch 90/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.2067 - mae: 0.3444 - val_loss: 0.0476 - val_mae: 0.1676 - learning_rate: 0.0010
Epoch 91/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1744 - mae: 0.3290 - val_loss: 0.0477 - val_mae: 0.1676 - learning_rate: 0.0010
Epoch 92/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.1846 - mae: 0.3243 - val_loss: 0.0478 - val_mae: 0.1678 - learning_rate: 0.0010
Epoch 93/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1695 - mae: 0.3218 - val_loss: 0.0481 - val_mae: 0.1681 - learning_rate: 0.0010
Epoch 94/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.1805 - mae: 0.3187 - val_loss: 0.0483 - val_mae: 0.1683 - learning_rate: 0.0010
Epoch 95/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1994 - mae: 0.3510 - val_loss: 0.0480 - val_mae: 0.1682 - learning_rate: 0.0010
Epoch 96/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1738 - mae: 0.3235 - val_loss: 0.0479 - val_mae: 0.1681 - learning_rate: 0.0010
Epoch 97/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1628 - mae: 0.3071 - val_loss: 0.0477 - val_mae: 0.1684 - learning_rate: 0.0010
Epoch 98/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1588 - mae: 0.3162 - val_loss: 0.0477 - val_mae: 0.1689 - learning_rate: 0.0010
Epoch 99/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1876 - mae: 0.3358 - val_loss: 0.0479 - val_mae: 0.1700 - learning_rate: 0.0010
Epoch 100/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1842 - mae: 0.3293 - val_loss: 0.0478 - val_mae: 0.1705 - learning_rate: 0.0010
Epoch 101/300
1/2 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.1565 - mae: 0.3189
Epoch 101: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1795 - mae: 0.3396 - val_loss: 0.0476 - val_mae: 0.1706 - learning_rate: 0.0010
Epoch 102/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1539 - mae: 0.3047 - val_loss: 0.0475 - val_mae: 0.1705 - learning_rate: 5.0000e-04
Epoch 103/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1653 - mae: 0.3129 - val_loss: 0.0475 - val_mae: 0.1705 - learning_rate: 5.0000e-04
Epoch 104/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1545 - mae: 0.3083 - val_loss: 0.0475 - val_mae: 0.1707 - learning_rate: 5.0000e-04
Epoch 105/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1523 - mae: 0.3174 - val_loss: 0.0473 - val_mae: 0.1705 - learning_rate: 5.0000e-04
Epoch 106/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1688 - mae: 0.3233 - val_loss: 0.0471 - val_mae: 0.1701 - learning_rate: 5.0000e-04
Epoch 107/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1515 - mae: 0.3028 - val_loss: 0.0468 - val_mae: 0.1696 - learning_rate: 5.0000e-04
Epoch 108/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1535 - mae: 0.3071 - val_loss: 0.0465 - val_mae: 0.1691 - learning_rate: 5.0000e-04
Epoch 109/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 0.1805 - mae: 0.3321 - val_loss: 0.0462 - val_mae: 0.1685 - learning_rate: 5.0000e-04
Epoch 110/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1602 - mae: 0.3153 - val_loss: 0.0458 - val_mae: 0.1676 - learning_rate: 5.0000e-04
Epoch 111/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 0.1299 - mae: 0.2855 - val_loss: 0.0455 - val_mae: 0.1668 - learning_rate: 5.0000e-04
Epoch 112/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1794 - mae: 0.3384 - val_loss: 0.0451 - val_mae: 0.1661 - learning_rate: 5.0000e-04
Epoch 113/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.1628 - mae: 0.3176 - val_loss: 0.0448 - val_mae: 0.1651 - learning_rate: 5.0000e-04
Epoch 114/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1601 - mae: 0.3137 - val_loss: 0.0446 - val_mae: 0.1643 - learning_rate: 5.0000e-04
Epoch 115/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1564 - mae: 0.3116 - val_loss: 0.0443 - val_mae: 0.1632 - learning_rate: 5.0000e-04
Epoch 116/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.1389 - mae: 0.2922 - val_loss: 0.0443 - val_mae: 0.1628 - learning_rate: 5.0000e-04
Epoch 117/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1499 - mae: 0.2905 - val_loss: 0.0440 - val_mae: 0.1626 - learning_rate: 5.0000e-04
Epoch 118/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1458 - mae: 0.3142 - val_loss: 0.0440 - val_mae: 0.1627 - learning_rate: 5.0000e-04
Epoch 119/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 51ms/step - loss: 0.1492 - mae: 0.3016 - val_loss: 0.0439 - val_mae: 0.1624 - learning_rate: 5.0000e-04
Epoch 120/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.1546 - mae: 0.3077 - val_loss: 0.0439 - val_mae: 0.1624 - learning_rate: 5.0000e-04
Epoch 121/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.1609 - mae: 0.3143 - val_loss: 0.0441 - val_mae: 0.1627 - learning_rate: 5.0000e-04
Epoch 122/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1321 - mae: 0.2759 - val_loss: 0.0442 - val_mae: 0.1628 - learning_rate: 5.0000e-04
Epoch 123/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step - loss: 0.1352 - mae: 0.2848 - val_loss: 0.0441 - val_mae: 0.1626 - learning_rate: 5.0000e-04
Epoch 124/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.1529 - mae: 0.3069 - val_loss: 0.0440 - val_mae: 0.1624 - learning_rate: 5.0000e-04
Epoch 125/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0962 - mae: 0.2455 - val_loss: 0.0439 - val_mae: 0.1621 - learning_rate: 5.0000e-04
Epoch 126/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1401 - mae: 0.2933 - val_loss: 0.0441 - val_mae: 0.1625 - learning_rate: 5.0000e-04
Epoch 127/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.1113 - mae: 0.2644 - val_loss: 0.0439 - val_mae: 0.1625 - learning_rate: 5.0000e-04
Epoch 128/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 0.1267 - mae: 0.2773 - val_loss: 0.0439 - val_mae: 0.1624 - learning_rate: 5.0000e-04
Epoch 129/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.1256 - mae: 0.2710 - val_loss: 0.0439 - val_mae: 0.1627 - learning_rate: 5.0000e-04
Epoch 130/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1297 - mae: 0.2866 - val_loss: 0.0439 - val_mae: 0.1625 - learning_rate: 5.0000e-04
Epoch 131/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1337 - mae: 0.2761 - val_loss: 0.0438 - val_mae: 0.1622 - learning_rate: 5.0000e-04
Epoch 132/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1251 - mae: 0.2701 - val_loss: 0.0436 - val_mae: 0.1616 - learning_rate: 5.0000e-04
Epoch 133/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1287 - mae: 0.2905 - val_loss: 0.0435 - val_mae: 0.1609 - learning_rate: 5.0000e-04
Epoch 134/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1430 - mae: 0.2979 - val_loss: 0.0432 - val_mae: 0.1600 - learning_rate: 5.0000e-04
Epoch 135/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1564 - mae: 0.3079 - val_loss: 0.0430 - val_mae: 0.1592 - learning_rate: 5.0000e-04
Epoch 136/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1105 - mae: 0.2600 - val_loss: 0.0427 - val_mae: 0.1584 - learning_rate: 5.0000e-04
Epoch 137/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1324 - mae: 0.2886 - val_loss: 0.0427 - val_mae: 0.1581 - learning_rate: 5.0000e-04
Epoch 138/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1476 - mae: 0.3095 - val_loss: 0.0425 - val_mae: 0.1574 - learning_rate: 5.0000e-04
Epoch 139/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.1340 - mae: 0.2821 - val_loss: 0.0426 - val_mae: 0.1572 - learning_rate: 5.0000e-04
Epoch 140/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.1683 - mae: 0.3174 - val_loss: 0.0424 - val_mae: 0.1569 - learning_rate: 5.0000e-04
Epoch 141/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1160 - mae: 0.2654 - val_loss: 0.0420 - val_mae: 0.1560 - learning_rate: 5.0000e-04
Epoch 142/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1307 - mae: 0.2874 - val_loss: 0.0418 - val_mae: 0.1551 - learning_rate: 5.0000e-04
Epoch 143/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1405 - mae: 0.2882 - val_loss: 0.0415 - val_mae: 0.1542 - learning_rate: 5.0000e-04
Epoch 144/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1212 - mae: 0.2651 - val_loss: 0.0412 - val_mae: 0.1536 - learning_rate: 5.0000e-04
Epoch 145/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1419 - mae: 0.2869 - val_loss: 0.0407 - val_mae: 0.1525 - learning_rate: 5.0000e-04
Epoch 146/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1139 - mae: 0.2796 - val_loss: 0.0403 - val_mae: 0.1514 - learning_rate: 5.0000e-04
Epoch 147/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.1214 - mae: 0.2780 - val_loss: 0.0400 - val_mae: 0.1509 - learning_rate: 5.0000e-04
Epoch 148/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1482 - mae: 0.3059 - val_loss: 0.0399 - val_mae: 0.1505 - learning_rate: 5.0000e-04
Epoch 149/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.1305 - mae: 0.2751 - val_loss: 0.0400 - val_mae: 0.1504 - learning_rate: 5.0000e-04
Epoch 150/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1216 - mae: 0.2675 - val_loss: 0.0399 - val_mae: 0.1498 - learning_rate: 5.0000e-04
Epoch 151/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1379 - mae: 0.2972 - val_loss: 0.0396 - val_mae: 0.1491 - learning_rate: 5.0000e-04
Epoch 152/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1375 - mae: 0.2831 - val_loss: 0.0392 - val_mae: 0.1480 - learning_rate: 5.0000e-04
Epoch 153/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1011 - mae: 0.2469 - val_loss: 0.0387 - val_mae: 0.1470 - learning_rate: 5.0000e-04
Epoch 154/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1087 - mae: 0.2580 - val_loss: 0.0387 - val_mae: 0.1466 - learning_rate: 5.0000e-04
Epoch 155/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1118 - mae: 0.2653 - val_loss: 0.0384 - val_mae: 0.1459 - learning_rate: 5.0000e-04
Epoch 156/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 0.1135 - mae: 0.2585 - val_loss: 0.0381 - val_mae: 0.1453 - learning_rate: 5.0000e-04
Epoch 157/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1358 - mae: 0.2927 - val_loss: 0.0380 - val_mae: 0.1448 - learning_rate: 5.0000e-04
Epoch 158/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1266 - mae: 0.2841 - val_loss: 0.0377 - val_mae: 0.1442 - learning_rate: 5.0000e-04
Epoch 159/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1216 - mae: 0.2559 - val_loss: 0.0377 - val_mae: 0.1442 - learning_rate: 5.0000e-04
Epoch 160/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1077 - mae: 0.2499 - val_loss: 0.0375 - val_mae: 0.1436 - learning_rate: 5.0000e-04
Epoch 161/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.0920 - mae: 0.2317 - val_loss: 0.0373 - val_mae: 0.1431 - learning_rate: 5.0000e-04
Epoch 162/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1249 - mae: 0.2792 - val_loss: 0.0370 - val_mae: 0.1424 - learning_rate: 5.0000e-04
Epoch 163/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1459 - mae: 0.2951 - val_loss: 0.0368 - val_mae: 0.1418 - learning_rate: 5.0000e-04
Epoch 164/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1114 - mae: 0.2737 - val_loss: 0.0369 - val_mae: 0.1419 - learning_rate: 5.0000e-04
Epoch 165/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1216 - mae: 0.2754 - val_loss: 0.0369 - val_mae: 0.1417 - learning_rate: 5.0000e-04
Epoch 166/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1204 - mae: 0.2666 - val_loss: 0.0370 - val_mae: 0.1415 - learning_rate: 5.0000e-04
Epoch 167/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0993 - mae: 0.2452 - val_loss: 0.0371 - val_mae: 0.1422 - learning_rate: 5.0000e-04
Epoch 168/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0927 - mae: 0.2423 - val_loss: 0.0372 - val_mae: 0.1426 - learning_rate: 5.0000e-04
Epoch 169/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.1159 - mae: 0.2637 - val_loss: 0.0372 - val_mae: 0.1428 - learning_rate: 5.0000e-04
Epoch 170/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0959 - mae: 0.2374 - val_loss: 0.0370 - val_mae: 0.1427 - learning_rate: 5.0000e-04
Epoch 171/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.1141 - mae: 0.2683 - val_loss: 0.0369 - val_mae: 0.1429 - learning_rate: 5.0000e-04
Epoch 172/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.1085 - mae: 0.2595 - val_loss: 0.0369 - val_mae: 0.1430 - learning_rate: 5.0000e-04
Epoch 173/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1179 - mae: 0.2688 - val_loss: 0.0371 - val_mae: 0.1435 - learning_rate: 5.0000e-04
Epoch 174/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1164 - mae: 0.2562 - val_loss: 0.0372 - val_mae: 0.1437 - learning_rate: 5.0000e-04
Epoch 175/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0962 - mae: 0.2414 - val_loss: 0.0373 - val_mae: 0.1444 - learning_rate: 5.0000e-04
Epoch 176/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1028 - mae: 0.2417 - val_loss: 0.0372 - val_mae: 0.1445 - learning_rate: 5.0000e-04
Epoch 177/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0991 - mae: 0.2485 - val_loss: 0.0370 - val_mae: 0.1438 - learning_rate: 5.0000e-04
Epoch 178/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 59ms/step - loss: 0.1161 - mae: 0.2619 - val_loss: 0.0366 - val_mae: 0.1428 - learning_rate: 5.0000e-04
Epoch 179/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 0.1101 - mae: 0.2669 - val_loss: 0.0366 - val_mae: 0.1427 - learning_rate: 5.0000e-04
Epoch 180/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.0945 - mae: 0.2467 - val_loss: 0.0363 - val_mae: 0.1417 - learning_rate: 5.0000e-04
Epoch 181/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1492 - mae: 0.2847 - val_loss: 0.0359 - val_mae: 0.1407 - learning_rate: 5.0000e-04
Epoch 182/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 0.0992 - mae: 0.2548 - val_loss: 0.0357 - val_mae: 0.1402 - learning_rate: 5.0000e-04
Epoch 183/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 0.0892 - mae: 0.2343 - val_loss: 0.0355 - val_mae: 0.1398 - learning_rate: 5.0000e-04
Epoch 184/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1266 - mae: 0.2715 - val_loss: 0.0352 - val_mae: 0.1393 - learning_rate: 5.0000e-04
Epoch 185/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.0965 - mae: 0.2405 - val_loss: 0.0350 - val_mae: 0.1388 - learning_rate: 5.0000e-04
Epoch 186/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.1270 - mae: 0.2902 - val_loss: 0.0348 - val_mae: 0.1387 - learning_rate: 5.0000e-04
Epoch 187/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1178 - mae: 0.2702 - val_loss: 0.0346 - val_mae: 0.1384 - learning_rate: 5.0000e-04
Epoch 188/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.0947 - mae: 0.2368 - val_loss: 0.0345 - val_mae: 0.1382 - learning_rate: 5.0000e-04
Epoch 189/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0977 - mae: 0.2485 - val_loss: 0.0345 - val_mae: 0.1383 - learning_rate: 5.0000e-04
Epoch 190/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step - loss: 0.0987 - mae: 0.2548 - val_loss: 0.0344 - val_mae: 0.1381 - learning_rate: 5.0000e-04
Epoch 191/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.1077 - mae: 0.2537 - val_loss: 0.0344 - val_mae: 0.1384 - learning_rate: 5.0000e-04
Epoch 192/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1067 - mae: 0.2605 - val_loss: 0.0342 - val_mae: 0.1379 - learning_rate: 5.0000e-04
Epoch 193/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0949 - mae: 0.2436 - val_loss: 0.0342 - val_mae: 0.1376 - learning_rate: 5.0000e-04
Epoch 194/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.0930 - mae: 0.2386 - val_loss: 0.0341 - val_mae: 0.1373 - learning_rate: 5.0000e-04
Epoch 195/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 57ms/step - loss: 0.1063 - mae: 0.2578 - val_loss: 0.0338 - val_mae: 0.1363 - learning_rate: 5.0000e-04
Epoch 196/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 56ms/step - loss: 0.0744 - mae: 0.2135 - val_loss: 0.0338 - val_mae: 0.1357 - learning_rate: 5.0000e-04
Epoch 197/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.0950 - mae: 0.2419 - val_loss: 0.0338 - val_mae: 0.1353 - learning_rate: 5.0000e-04
Epoch 198/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.1025 - mae: 0.2419 - val_loss: 0.0338 - val_mae: 0.1350 - learning_rate: 5.0000e-04
Epoch 199/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.0974 - mae: 0.2430 - val_loss: 0.0337 - val_mae: 0.1348 - learning_rate: 5.0000e-04
Epoch 200/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.0894 - mae: 0.2307 - val_loss: 0.0337 - val_mae: 0.1345 - learning_rate: 5.0000e-04
Epoch 201/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0990 - mae: 0.2417 - val_loss: 0.0337 - val_mae: 0.1346 - learning_rate: 5.0000e-04
Epoch 202/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0927 - mae: 0.2431 - val_loss: 0.0337 - val_mae: 0.1347 - learning_rate: 5.0000e-04
Epoch 203/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.1056 - mae: 0.2512 - val_loss: 0.0337 - val_mae: 0.1347 - learning_rate: 5.0000e-04
Epoch 204/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0793 - mae: 0.2171 - val_loss: 0.0337 - val_mae: 0.1347 - learning_rate: 5.0000e-04
Epoch 205/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1086 - mae: 0.2437 - val_loss: 0.0337 - val_mae: 0.1346 - learning_rate: 5.0000e-04
Epoch 206/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.0815 - mae: 0.2285 - val_loss: 0.0336 - val_mae: 0.1345 - learning_rate: 5.0000e-04
Epoch 207/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1218 - mae: 0.2695 - val_loss: 0.0336 - val_mae: 0.1346 - learning_rate: 5.0000e-04
Epoch 208/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.0843 - mae: 0.2266 - val_loss: 0.0334 - val_mae: 0.1345 - learning_rate: 5.0000e-04
Epoch 209/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.0807 - mae: 0.2261 - val_loss: 0.0334 - val_mae: 0.1347 - learning_rate: 5.0000e-04
Epoch 210/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.0881 - mae: 0.2458 - val_loss: 0.0332 - val_mae: 0.1343 - learning_rate: 5.0000e-04
Epoch 211/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.0823 - mae: 0.2279 - val_loss: 0.0330 - val_mae: 0.1339 - learning_rate: 5.0000e-04
Epoch 212/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.0898 - mae: 0.2308 - val_loss: 0.0329 - val_mae: 0.1338 - learning_rate: 5.0000e-04
Epoch 213/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 55ms/step - loss: 0.0924 - mae: 0.2280 - val_loss: 0.0329 - val_mae: 0.1338 - learning_rate: 5.0000e-04
Epoch 214/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.0970 - mae: 0.2372 - val_loss: 0.0328 - val_mae: 0.1337 - learning_rate: 5.0000e-04
Epoch 215/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.0777 - mae: 0.2265 - val_loss: 0.0328 - val_mae: 0.1337 - learning_rate: 5.0000e-04
Epoch 216/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.0854 - mae: 0.2326 - val_loss: 0.0326 - val_mae: 0.1336 - learning_rate: 5.0000e-04
Epoch 217/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.1113 - mae: 0.2554 - val_loss: 0.0326 - val_mae: 0.1337 - learning_rate: 5.0000e-04
Epoch 218/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.0976 - mae: 0.2340 - val_loss: 0.0326 - val_mae: 0.1338 - learning_rate: 5.0000e-04
Epoch 219/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.0856 - mae: 0.2316 - val_loss: 0.0325 - val_mae: 0.1336 - learning_rate: 5.0000e-04
Epoch 220/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.0896 - mae: 0.2355 - val_loss: 0.0323 - val_mae: 0.1332 - learning_rate: 5.0000e-04
Epoch 221/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step - loss: 0.0841 - mae: 0.2328 - val_loss: 0.0322 - val_mae: 0.1328 - learning_rate: 5.0000e-04
Epoch 222/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.0735 - mae: 0.2081 - val_loss: 0.0322 - val_mae: 0.1327 - learning_rate: 5.0000e-04
Epoch 223/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 52ms/step - loss: 0.0829 - mae: 0.2256 - val_loss: 0.0321 - val_mae: 0.1325 - learning_rate: 5.0000e-04
Epoch 224/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0667 - mae: 0.1943 - val_loss: 0.0321 - val_mae: 0.1324 - learning_rate: 5.0000e-04
Epoch 225/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.0967 - mae: 0.2516 - val_loss: 0.0321 - val_mae: 0.1323 - learning_rate: 5.0000e-04
Epoch 226/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0714 - mae: 0.2145 - val_loss: 0.0321 - val_mae: 0.1323 - learning_rate: 5.0000e-04
Epoch 227/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0748 - mae: 0.2227 - val_loss: 0.0322 - val_mae: 0.1329 - learning_rate: 5.0000e-04
Epoch 228/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0880 - mae: 0.2272 - val_loss: 0.0322 - val_mae: 0.1331 - learning_rate: 5.0000e-04
Epoch 229/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.1191 - mae: 0.2706 - val_loss: 0.0321 - val_mae: 0.1327 - learning_rate: 5.0000e-04
Epoch 230/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0770 - mae: 0.2133 - val_loss: 0.0322 - val_mae: 0.1329 - learning_rate: 5.0000e-04
Epoch 231/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0788 - mae: 0.2168 - val_loss: 0.0321 - val_mae: 0.1326 - learning_rate: 5.0000e-04
Epoch 232/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 53ms/step - loss: 0.0782 - mae: 0.2196 - val_loss: 0.0320 - val_mae: 0.1321 - learning_rate: 5.0000e-04
Epoch 233/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.0715 - mae: 0.2059 - val_loss: 0.0321 - val_mae: 0.1321 - learning_rate: 5.0000e-04
Epoch 234/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0934 - mae: 0.2419 - val_loss: 0.0322 - val_mae: 0.1324 - learning_rate: 5.0000e-04
Epoch 235/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0853 - mae: 0.2311 - val_loss: 0.0324 - val_mae: 0.1327 - learning_rate: 5.0000e-04
Epoch 236/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0901 - mae: 0.2296 - val_loss: 0.0326 - val_mae: 0.1332 - learning_rate: 5.0000e-04
Epoch 237/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0731 - mae: 0.2095 - val_loss: 0.0325 - val_mae: 0.1332 - learning_rate: 5.0000e-04
Epoch 238/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0812 - mae: 0.2218 - val_loss: 0.0325 - val_mae: 0.1333 - learning_rate: 5.0000e-04
Epoch 239/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0672 - mae: 0.1913 - val_loss: 0.0325 - val_mae: 0.1332 - learning_rate: 5.0000e-04
Epoch 240/300
1/2 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.0957 - mae: 0.2427
Epoch 240: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0952 - mae: 0.2423 - val_loss: 0.0325 - val_mae: 0.1333 - learning_rate: 5.0000e-04
Epoch 241/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0953 - mae: 0.2363 - val_loss: 0.0325 - val_mae: 0.1334 - learning_rate: 2.5000e-04
Epoch 242/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0735 - mae: 0.2095 - val_loss: 0.0325 - val_mae: 0.1336 - learning_rate: 2.5000e-04
Epoch 243/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0743 - mae: 0.2211 - val_loss: 0.0326 - val_mae: 0.1336 - learning_rate: 2.5000e-04
Epoch 244/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0688 - mae: 0.2072 - val_loss: 0.0325 - val_mae: 0.1335 - learning_rate: 2.5000e-04
Epoch 245/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.0704 - mae: 0.2055 - val_loss: 0.0325 - val_mae: 0.1337 - learning_rate: 2.5000e-04
Epoch 246/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0788 - mae: 0.2191 - val_loss: 0.0325 - val_mae: 0.1338 - learning_rate: 2.5000e-04
Epoch 247/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.0703 - mae: 0.2141 - val_loss: 0.0325 - val_mae: 0.1341 - learning_rate: 2.5000e-04
Epoch 248/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0879 - mae: 0.2294 - val_loss: 0.0325 - val_mae: 0.1340 - learning_rate: 2.5000e-04
Epoch 249/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0719 - mae: 0.2063 - val_loss: 0.0325 - val_mae: 0.1341 - learning_rate: 2.5000e-04
Epoch 250/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0757 - mae: 0.2182 - val_loss: 0.0326 - val_mae: 0.1343 - learning_rate: 2.5000e-04
Epoch 251/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 34ms/step - loss: 0.0598 - mae: 0.2035 - val_loss: 0.0326 - val_mae: 0.1345 - learning_rate: 2.5000e-04
Epoch 252/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0696 - mae: 0.2019 - val_loss: 0.0326 - val_mae: 0.1345 - learning_rate: 2.5000e-04
Epoch 253/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0695 - mae: 0.2101 - val_loss: 0.0326 - val_mae: 0.1347 - learning_rate: 2.5000e-04
Epoch 254/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0740 - mae: 0.2094 - val_loss: 0.0326 - val_mae: 0.1347 - learning_rate: 2.5000e-04
Epoch 255/300
1/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - loss: 0.0594 - mae: 0.1957
Epoch 255: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0748 - mae: 0.2133 - val_loss: 0.0326 - val_mae: 0.1347 - learning_rate: 2.5000e-04
Epoch 256/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0794 - mae: 0.2131 - val_loss: 0.0326 - val_mae: 0.1347 - learning_rate: 1.2500e-04
Epoch 257/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0653 - mae: 0.2066 - val_loss: 0.0325 - val_mae: 0.1343 - learning_rate: 1.2500e-04
Epoch 258/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0849 - mae: 0.2172 - val_loss: 0.0324 - val_mae: 0.1341 - learning_rate: 1.2500e-04
Epoch 259/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0655 - mae: 0.2008 - val_loss: 0.0323 - val_mae: 0.1338 - learning_rate: 1.2500e-04
Epoch 260/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0655 - mae: 0.1934 - val_loss: 0.0322 - val_mae: 0.1337 - learning_rate: 1.2500e-04
Epoch 261/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 33ms/step - loss: 0.0751 - mae: 0.2207 - val_loss: 0.0322 - val_mae: 0.1338 - learning_rate: 1.2500e-04
Epoch 262/300
2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 32ms/step - loss: 0.0806 - mae: 0.2245 - val_loss: 0.0321 - val_mae: 0.1338 - learning_rate: 1.2500e-04
Epoch 262: early stopping
Restoring model weights from the end of the best epoch: 232.

======================================================================
TEST SET EVALUATION
======================================================================
  loss: 0.030692
  compile_metrics: 0.134794

Sample predictions (first 8 test rows):
             mqe_improvement_ratio       topographic_error       dead_neuron_ratio
  actual:                    0.5290                  0.0949                  0.5972
  predicted:                 0.5772                  0.0752                  0.5015

  actual:                    0.5545                  0.0316                  0.6480
  predicted:                 0.3755                 -0.1229                  0.4390

  actual:                    0.4298                  0.0791                  0.6509
  predicted:                 0.2071                 -0.0057                  0.3589

  actual:                    0.3870                  0.0264                  0.6888
  predicted:                 0.2863                  0.0587                  0.7123

  actual:                    0.5781                  0.0123                  0.6509
  predicted:                 0.1995                  0.0816                  0.1517

  actual:                    0.5184                  0.0176                  0.6633
  predicted:                 0.3814                  0.1492                  0.6152

  actual:                    0.4528                  0.0158                  0.6786
  predicted:                 0.4068                  0.0767                  0.4511

  actual:                    0.4944                  0.1019                  0.5972
  predicted:                 0.4247                  0.1150                  0.4582

======================================================================
Best model (timestamped): models/mlp_standard_20260514_165226_best.keras
Stable path (EA config):  models/mlp_latest.keras
Scaler (EA config):       models/mlp_scaler_latest.pkl
======================================================================

Update config-ea.json NEURAL_NETWORKS section:
  "mlp_model_path":  "models/mlp_latest.keras"
  "mlp_scaler_path": "models/mlp_scaler_latest.pkl"