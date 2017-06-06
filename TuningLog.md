# Tuning log

# DNN model 
mini-batch size: 100
Learning rate: 0.001
optimizer: ProximalAdagradOptimizer
regularization_strength = 0.001

# input data 
no one hot encoder, direct 0-7 number: 88.7
One hot encoder, with D, N, S, R composed by A, G, C, T: 95.6%
One hot encoder, with D, N, S, R built by individual number: 95.3%