import numpy as np
import matplotlib.pyplot as plt

input_size = 500
hidden_layer_sizes = [500]*10

act_type = 'relu'  # one of ['relu', 'tanh']
act_funcs = {'relu': lambda x: np.maximum(x, 0), 'tanh': lambda x: np.tanh(x)}

if __name__ == '__main__':
  depth = len(hidden_layer_sizes)
  x = np.random.normal(size=input_size)
  activations = [x]

  for i in range(depth):
    fan_in = x.shape[0]
    fan_out = hidden_layer_sizes[i]
    W = np.random.normal(0, 1, (fan_in, fan_out)) / np.sqrt(fan_in/2)
    x = np.dot(W, x)
    x = act_funcs[act_type](x)
    activations.append(x)

  fig = plt.figure(figsize=(16, 4))
  for i in range(len(activations)):
    plt.subplot(1, len(activations), i+1)
    plt.hist(activations[i], 30, range=(-1, 1))
  plt.show()


