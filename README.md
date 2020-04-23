# MyEA (I no longer maintain this repository! (for the time being))
AI library in C++ (with OpenMP) and CUDA.

This library contains the following implementations : 
  1. Activation:
  - Cosine
  - Cosine Symmetric
  - Elliot
  - Elliot Symmetric
  - ELU (Exponential Linear Unit)
  - Gaussian
  - Gaussian Symmetric
  - Gaussian Stepwise
  - ISRU (Inverse Square Root Units)
  - ISRLU (Inverse Square Root Linear Units)
  - Linear
  - Linear piece
  - Linear piece Symmetric
  - LReLU (Leaky Rectified Linear Unit)
  - ReLU (Rectified Linear Unit)
  - SELU (Scaled Exponential Linear Unit)
  - Sine
  - Sine Symmetric
  - Sigmoid
  - Sigmoid Stepwise
  - Softmax
  - Tanh
  - Tanh Stepwise
  - Threshold
  - Threshold Symmetric

  2. Layers:
  - Average Pooling
  - Fully Connected (aka MLP, Dense, Linear)
  - Fully Connected Independently Recurrent (aka IndRNN, w/ Bidirectional)
  - Fully Connected Recurrent (aka RNN, w/ Bidirectional)
  - LSTM  w/ peephole (w/ Bidirectional)
  - Max Pooling
  - Residual Block (aka ResNet)

  3. Loss functions:
  - Bit
  - Cross-entropy
  - L1
  - L2
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
  - ME (Mean Error)
  - MSE (Mean Square Error)
  - RMSE (Root Mean Square Error)
  - SMAPE (Symmetric Mean Absolute Percentage Error)

  4. Normalization:
  - Batch Normalization
  - Batch Renormalization

  5. Optimizers:
  - AdaBound
  - Adam
  - Adamax
  - AMSBound
  - AMSGrad
  - GD (aka SGD, with momentum and nesterov)
  - iRPROP-
  - iRPROP+
  - NosAdam

  6. Regularization:
  - Alpha
  - Clip Gradient
  - Dropout bernoulli (aka Dropout)
  - Dropout bernoulli inverted
  - Dropout Gaussian
  - L1
  - L2 (aka Weight Decay With SGD)
  - ShakeDrop
  - Uout
  - Zoneout

  7. Weights initializer:
  - Glorot (Gaussian or Uniform)
  - Identity
  - LSUV
  - Orthogonal
  - Uniform
