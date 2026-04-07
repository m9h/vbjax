# Contributing to vbjax

Thank you for your interest in contributing to vbjax!  This guide covers
everything you need to get started, from setting up a development
environment to submitting a pull request.

## 1. Development Setup

```bash
git clone https://github.com/ins-amu/vbjax
cd vbjax

# Create a virtual environment (uv recommended)
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify the installation
python -c "import vbjax; print(vbjax.__version__)"
pytest -x
```

If you have a CUDA GPU, install the appropriate JAX wheel:

```bash
uv pip install --upgrade "jax[cuda12]"
```

## 2. Project Structure

```
vbjax/
  neural_mass.py   # Neural mass models (JR, MPR, CMC, BOLD, ...)
  loops.py         # make_sde, make_ode, make_dde integrators
  coupling.py      # Coupling functions for network models
  connectome.py    # Structural connectivity utilities
  layers.py        # Dense layer helpers
  monitor.py       # Online monitors (BOLD, FC, time-average)
  sparse.py        # Sparse matrix-vector products
  shtlc.py         # Spherical harmonic diffusion
  diagnostics.py   # Statistical diagnostics
examples/          # Runnable scripts demonstrating models
docs/              # Sphinx documentation (RST format)
```

## 3. Adding a New Neural Mass Model

All models in `vbjax/neural_mass.py` follow a consistent pattern:

1. **Define a parameter namedtuple** (`ModelTheta`) with all model
   constants.
2. **Define a state namedtuple** (`ModelState`) naming each state
   variable.
3. **Create defaults** (`model_default_theta`, `model_default_state`)
   with physiologically motivated values.
4. **Implement the dynamics** as `model_dfun(ys, c, p)` where:
   - `ys` is the state vector (shape `(n_states,)` or
     `(n_states, n_nodes)`)
   - `c` is the coupling input
   - `p` is the parameter namedtuple
   - Returns `dys`, the state derivatives (same shape as `ys`)
5. **(Optional) Network wrapper** `model_net_dfun(ys, p)` with
   signature compatible with `vbjax.make_sde`.

Example skeleton:

```python
import collections
import jax.numpy as np

MyModelTheta = collections.namedtuple('MyModelTheta', 'alpha beta I')
my_model_default_theta = MyModelTheta(alpha=1.0, beta=0.5, I=0.0)

def my_model_dfun(ys, c, p: MyModelTheta) -> np.ndarray:
    """My model dynamics.

    Parameters
    ----------
    ys : array, shape (2,) or (2, n_nodes)
        State vector [x, v].
    c : array
        Coupling input.
    p : MyModelTheta
        Model parameters.

    Returns
    -------
    dys : array, same shape as ys
        State derivatives.
    """
    x, v = ys
    dx = v
    dv = -p.alpha * x - p.beta * v + p.I + c
    return np.array([dx, dv])
```

Export the new model from `vbjax/__init__.py` and add tests.

## 4. JAX Compatibility Rules

vbjax models must be compatible with JAX transforms (`jit`, `grad`,
`vmap`, `jacobian`).  Follow these rules:

- **Use `jax.numpy`**, not NumPy, for all array operations inside
  `dfun`.
- **Avoid Python control flow** that depends on array values.  Use
  `jax.lax.cond`, `jax.lax.switch`, or `jnp.where` instead of
  `if`/`else`.
- **No in-place mutation.**  Use `x.at[i].set(v)` rather than
  `x[i] = v`.
- **Keep signatures pure** -- functions should have no side effects.
- **Test differentiability** with `jax.grad` or `jax.jacobian` in your
  test suite.

## 5. Testing

Tests live in `vbjax/tests/` and use plain pytest (no unittest classes).

```bash
# Run all tests
pytest

# Run a specific test file
pytest vbjax/tests/test_cmc.py -v

# Run tests in parallel
pytest -n auto
```

Every new model should have tests covering at minimum:

- Shape correctness (single-node and multi-node)
- Default parameter/state field counts
- Resting-state stability (bounded dynamics with I=0)
- Response to input (non-trivial activity with drive)
- Noise-driven oscillations
- Differentiability with respect to state and coupling input
- Network coupling via `make_sde`

## 6. Documentation

Documentation uses **Sphinx** with **NumPy-style docstrings** and is
hosted on Read the Docs.

- **Docstrings**: Use NumPy-style for all public functions, classes,
  and module-level constants.  Include `Parameters`, `Returns`, and
  `References` sections.
- **Tutorials**: Place `.rst` files in `docs/tutorials/`.  Include
  runnable code blocks and cross-reference the API with `:func:` and
  `:class:` roles.
- **Building locally**:

```bash
cd docs
uv pip install -r requirements.txt
make html
open _build/html/index.html
```

## 7. Pull Request Process

1. **Fork and branch** from `main`.  Use descriptive branch names like
   `feature/my-new-model` or `fix/bold-stability`.
2. **Write tests first** if possible.  All CI checks must pass.
3. **Keep commits focused**.  One logical change per commit.
4. **Update documentation** for any public API changes -- add docstrings,
   update tutorials, and extend the changelog.
5. **Open a PR** against `main` with a clear description of what the
   change does and why.
6. A maintainer will review.  Address feedback via additional commits
   (no force-push during review).
