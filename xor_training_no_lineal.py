"""
Ejercicio: Red neuronal para predecir XOR con activaciones no lineales
Usando michigrad, con ReLU en la capa oculta.
"""

import os
import numpy as np
from michigrad.engine import Value
from michigrad.nn import MLP, relu
from michigrad.visualize import show_graph

# Configurar Graphviz
import graphviz
graphviz.backend.EXECUTABLE_EXTENSION = '.exe'
os.environ['PATH'] += os.pathsep + r'C:\Program Files\Graphviz\bin'

xs = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
ys = [0.0, 1.0, 1.0, 0.0]

print("Dataset XOR")
for x, y in zip(xs, ys):
    print(f"  {x} -> {y}")

# Crear modelo con ReLU en capa oculta
model = MLP(2, [2, 1], relu())

print("\nModelo:", model)
print("Parámetros:", len(model.parameters()))

learning_rate = 0.1
epochs = 125
print(f"Learning rate: {learning_rate}, Épocas: {epochs}\n")

# Primera iteración
print("Primera iteración:")
inputs_first = [[Value(x[0], name=f'x0_{i}'), 
                  Value(x[1], name=f'x1_{i}')] for i, x in enumerate(xs)]
predictions_first = [model(xi) for xi in inputs_first]

print(f"Predicciones: {[round(p.data, 4) for p in predictions_first]}")
print(f"Esperadas:    {ys}")

loss_first = sum(((yi - pred)**2 for yi, pred in zip(ys, predictions_first))) / len(ys)
loss_first.name = "Loss"
print(f"Loss: {loss_first.data:.6f}\n")

print("Generando gráfos...")
dot_graph_1 = show_graph(loss_first, format='svg', rankdir='TB')
dot_graph_1.render('grafo_nl_01_after_forward', cleanup=True)

model.zero_grad()
loss_first.backward()

dot_graph_2 = show_graph(loss_first, format='svg', rankdir='TB')
dot_graph_2.render('grafo_nl_02_after_backward', cleanup=True)

for p in model.parameters():
    p.data += -learning_rate * p.grad

# Entrenamiento
print("\nEntrenamiento:")
losses = []

for epoch in range(1, epochs + 1):
    inputs = [[Value(x[0]), Value(x[1])] for x in xs]
    predictions = [model(xi) for xi in inputs]
    
    loss = sum(((yi - pred)**2 for yi, pred in zip(ys, predictions))) / len(ys)
    losses.append(loss.data)
    
    model.zero_grad()
    loss.backward()
    
    for p in model.parameters():
        p.data += -learning_rate * p.grad
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"  Época {epoch:2d} - Loss: {loss.data:.6f}")

# Resultados
print("\nResultados finales:")
inputs_final = [[Value(x[0]), Value(x[1])] for x in xs]
predictions_final = [model(xi) for xi in inputs_final]

for x, pred, expected in zip(xs, predictions_final, ys):
    error = abs(pred.data - expected)
    print(f"  {x} -> Pred: {pred.data:.4f} | Esperado: {expected:.4f} | Error: {error:.4f}")

print(f"\nPérdida inicial: {losses[0]:.6f}")
print(f"Pérdida final:   {losses[-1]:.6f}")
print(f"Reducción:       {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")

# Gráfico
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), losses, linewidth=2, color='blue')
    plt.xlabel('Época')
    plt.ylabel('Loss (MSE)')
    plt.title('Pérdida durante el entrenamiento (con ReLU)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss_curve_no_lineal.png', dpi=150, bbox_inches='tight')
    print("\nGráfico guardado como 'training_loss_curve_no_lineal.png'")
    
except ImportError:
    print("\nmatplotlib no instalado")

print("\nArchivos generados:")
print("  - grafo_nl_01_after_forward.svg")
print("  - grafo_nl_02_after_backward.svg")
print("  - training_loss_curve_no_lineal.png")
