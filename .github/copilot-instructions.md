# Michigrad: AI Coding Agent Instructions

## Project Overview
Michigrad is an educational autograd (automatic differentiation) engine for scalar values. It's a simplified clone of Karpathy's Micrograd, designed for learning deep learning fundamentals.

**Core Purpose**: Build computation graphs for forward/backward pass (backpropagation) to compute gradients for neural network training.

## Architecture

### Core Components

1. **`engine.py` - `Value` class**: The foundation
   - Wraps scalar values with gradient tracking (`data`, `grad`)
   - Implements arithmetic operators (`__add__`, `__mul__`, etc.) that build computation graph via closures
   - Key pattern: Each operation stores a `_backward()` lambda that applies chain rule
   - Operations: `+`, `-`, `*`, `/`, `**`, `relu()`, `exp()`
   - Entry point for backprop: Call `.backward()` on root node (typically loss)

2. **`nn.py` - Neural Network Modules**
   - `Module`: Base class with `parameters()` and `zero_grad()` methods
   - `Neuron`: Single perceptron with weights `w` (list of Values) and bias `b`; applies ReLU by default unless `nonlin=False`
   - `Layer`: Collection of neurons; forward pass returns list or single Value
   - `MLP`: Multi-layer perceptron; chains layers, only last layer is linear (`nonlin=False`)

3. **`visualize.py` - Graph Visualization**
   - `show_graph()`: Renders computation graph as SVG/PNG using Graphviz
   - `show_graph_interactive()`: Interactive NetworkX visualization
   - Both trace nodes/edges from a root Value backward through `_prev` pointers

## Key Patterns & Conventions

### Computation Graph Construction
- **Lazy evaluation**: Operations don't compute immediately; they build the graph structure
- **Chain rule implementation**: Each Value stores `_backward()` function that accumulates gradients to children using `+=`
- **Topological sorting**: `backward()` builds DAG order to ensure gradients computed in reverse execution order

```python
# Pattern: All backward functions accumulate via +=
def _backward():
    self.grad += derivative_wrt_self * out.grad  # Chain rule
```

### Training Loop Pattern
1. `zero_grad()` on all modules
2. Forward pass: call MLP with input Values
3. Compute loss (typically MSE): `(y_true - y_pred) ** 2`
4. `loss.backward()` to propagate gradients
5. Update weights: `w.data -= learning_rate * w.grad`

### Value Naming
Use `.name` attribute for readable graph visualization. Convention: mathematical notation (e.g., `"W₀"`, `"ŷ"` for predictions, `"L"` for loss).

## Testing
Tests in `test/test_engine.py` validate gradients against PyTorch using tolerance `1e-6`. Tests verify:
- Forward pass correctness
- Backward pass gradient computation
- Numerical stability across complex operation sequences

**Run tests**: `pytest test/test_engine.py`

## Common Workflows

### Adding New Operations
1. Implement `__method__` in `Value` (or `__rmethod__` for reverse)
2. Create output `Value` with operands as `_children` and operation string as `_op`
3. Define `_backward()` closure applying chain rule to all children
4. Add corresponding test in `test_engine.py` validating against PyTorch

Example (already implemented - exponentiation):
```python
def exp(self):
    out = Value(math.exp(self.data), (self,), f'e^{self.data}')
    def _backward():
        self.grad += out.data * out.grad  # d/dx(e^x) = e^x
    out._backward = _backward
    return out
```

### Building Neural Networks
```python
from michigrad.nn import MLP

# Define architecture: 2 inputs → 16 hidden → 16 hidden → 1 output
model = MLP(nin=2, nouts=[16, 16, 1])

# Training step
x = [Value(xi) for xi in input_data]  # Wrap inputs
y_pred = model(x)
loss = (y_true - y_pred) ** 2
loss.backward()
for p in model.parameters():
    p.data -= learning_rate * p.grad
model.zero_grad()
```

## Dependencies
- `numpy`: Random data generation
- `torch`: Gradient validation in tests (not used in core library)
- `graphviz`: Computation graph rendering
- `networkx`, `pyvis`: Interactive visualization

## File Structure Notes
- No external configuration files (no config.yaml, setup.py beyond standard)
- Module-level imports in `__init__.py` are empty (explicit imports required)
- Tests use pytest convention (`test_*.py` in `test/` directory)
- Jupyter notebook (`pruebas.ipynb`) serves as educational walkthrough

## Critical Implementation Detail
**Gradient Accumulation**: Always use `+=` in backward functions, never `=`. This handles cases where a Value appears multiple times in the graph (e.g., `x * x + x`).

## When Modifying Core
- Preserve numeric stability (test against PyTorch with `tol=1e-6`)
- Ensure `_backward()` functions follow chain rule strictly
- Update `__init__.py` if adding public APIs
- Add tests before merging changes to backward pass
