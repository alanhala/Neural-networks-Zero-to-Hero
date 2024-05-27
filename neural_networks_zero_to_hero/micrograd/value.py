import math

class Value():
  def __init__(self, data, children=(), operation = None, label="") -> None:
    self.data = data
    self.label = label
    self.children = children
    self.operation = operation
    self.grad = 0.0
    self.grad_fn = lambda : None

  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    value = Value(self.data + other.data, (self, other), operation="+")
    def grad_fn():
      self.grad += 1.0 * value.grad
      other.grad += 1.0 * value.grad
    value.grad_fn = grad_fn
    return value
  
  def __sub__(self, other):
    return Value(self.data - other.data, (self, other), operation="-")

  def __sub__(self, other):
    return self + (-other)
  
  def __neg__(self):
    return self * -1
  
  def __radd__(self, other):
    return self + other
  
  def __rsub__(self, other):
      return self - other

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    value = Value(self.data * other.data, (self, other), operation="*")
    def grad_fn():
      self.grad += other.data * value.grad
      other.grad += self.data * value.grad
    value.grad_fn = grad_fn
    return value
  
  def __rmul__(self, other):
      return self * other
    
  def __truediv__(self, other):
    return self * other**-1
  
  def __pow__(self, other):
    assert isinstance(other, (int, float)) # only supporting int/float powers for now
    value = Value(self.data ** other, (self,), f"**{other}")
    def grad_fn():
      self.grad += (other * self.data ** (other - 1)) * value.grad
    value.grad_fn = grad_fn
    return value

  def exp(self): 
    value = Value(math.exp(self.data), (self, ), 'exp')
    def grad_fn():
      self.grad += value.data * value.grad
    value.grad_fn = grad_fn
    return value
  
  def tanh(self):
    value = Value((math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1), (self,), 'tanh')
    def grad_fn():
      self.grad += (1 - value.data**2) * value.grad
    value.grad_fn = grad_fn
    return value
  
  def backward(self):
    self.grad = 1.0
    self._backward()

  def _backward(self):
    self.grad_fn()
    for value in self.children:
      value._backward()

    
