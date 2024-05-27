from graphviz import Digraph
from micrograd.value import Value

class Graph():
  def __init__(self, value: Value, graph = None) -> None:
    self.value = value
    self.graph = graph or Digraph(graph_attr={'rankdir': 'LR'})
  
  def print(self):
    nodes, edges = set(), set()
    def trace(value):
      nodes.add(value)
      for child in value.children:
        edges.add((child, value))
        trace(child)
    trace(self.value)
    for node in nodes:
      uid = str(id(node))
      self.graph.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (node.label, node.data, node.grad), shape='record')
      if node.operation:
        self.graph.node(name = uid + node.operation, label = node.operation)
        self.graph.edge(uid + node.operation, uid)
    for node1, node2 in edges:
      self.graph.edge(str(id(node1)), str(id(node2)) + node2.operation)
    return self.graph
