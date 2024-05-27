import random
import functools
from micrograd.value import Value

class Neuron():
  def __init__(self, inputs) -> None:
    self.w = [Value(random.uniform(-1, 1)) for _ in range(inputs)]
    self.b = Value(random.uniform(-1, 1))
  
  def __repr__(self):
    return f"Neuron(weights= {self.w}, bias={self.b})"

  def __call__(self, x):
    return sum([wi * xi for wi, xi in zip(self.w, x)], self.b).tanh()
    
  def parameters(self):
    return self.w + [self.b]

class Layer():
  def __init__(self, inputs, outputs) -> None:
    self.neurons = [Neuron(inputs) for _ in range(outputs)]
  
  def __call__(self, x):
    y = [neuron(x) for neuron in self.neurons]
    return y[0] if len(y) == 1 else y

  def parameters(self):
    return [parameter for neuron in self.neurons for parameter in neuron.parameters()]

class MLP:
  def __init__(self, inputs, outputs):
    sz = [inputs] + outputs
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(outputs))]

  def __call__(self, x):
    out = functools.reduce(lambda inputs, layer: layer(inputs), self.layers, x)
    return out

  def parameters(self):
    return [parameter for layer in self.layers for parameter in layer.parameters()]

  def train(self, xs, ys):
    for step in range(1000):
      predictions = [self(x) for x in xs]
      loss = sum([(target - prediction)**2 for prediction, target in zip(predictions, ys)])
      loss.backward()
      print(step, loss.data)
      for parameter in self.parameters():
        parameter.data -= 0.01 * parameter.grad
        parameter.grad = 0