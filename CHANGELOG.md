# Changelog

All notable changes to vbjax are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] -- feature/cmc-linearization

### Added

- **Canonical Microcircuit (CMC) model** -- 4-population neural mass model
  (spiny stellate, superficial pyramidal, inhibitory interneurons, deep
  pyramidal) with 8 state variables and 16 parameters.
  - `cmc_dfun` -- single/multi-node dynamics compatible with `make_sde`.
  - `cmc_net_dfun` -- network form with structural connectivity coupling.
  - `cmc_hier_dfun` -- hierarchical variant with separate forward (-> ss)
    and backward (-> sp + dp) coupling pathways for predictive coding.
  - `cmc_hier_2node_dfun` -- two-node hierarchy for predictive coding
    experiments (lower V1-like area + higher V4-like area).
  - `cmc_hier_Nnode_dfun` -- N-node generalization with separate forward
    and backward structural connectivity matrices.
  - `cmc_to_layer_activity` -- bridge from CMC state to 3-layer cortical
    activity (deep/middle/superficial) for coupling with vpjax hemodynamic
    models.
  - `cmc_observe_sp` / `cmc_observe_dp` -- observation functions for
    EEG-like (superficial pyramidal) and LFP-like (deep pyramidal) signals.
  - `CMCTheta`, `CMCState` namedtuples with documented fields.
  - `cmc_default_theta` tuned via differential evolution for alpha-band
    oscillations; `cmc_default_state` at resting zero.
- Comprehensive CMC test suite (`vbjax/tests/test_cmc.py`): shape checks,
  sigmoid consistency with JR, resting stability, noise-driven oscillations,
  network coupling, hierarchical predictive coding, laminar separation,
  layer bridge, differentiability.
- `examples/cmc_comparison.py` -- multi-dimensional comparison of CMC vs
  JR vs MPR: spectral fingerprints, bifurcation analysis, laminar
  decomposition, step response, compute cost, forward/backward asymmetry,
  2-node hierarchy, cross-frequency coupling, layer-resolved BOLD bridge.
- `examples/cmc_tune_defaults.py` -- parameter tuning via CMA-ES /
  differential evolution targeting alpha-band oscillations.
- CMC tutorials covering single-node dynamics, hierarchical predictive
  coding, and model comparison.
- `CONTRIBUTING.md` with development setup, model-addition guide, JAX
  compatibility rules, testing and documentation conventions.
- NumPy-style docstrings and type annotations for all CMC public API.
- `autodoc_mock_imports` and `sphinx_autodoc_typehints` in Sphinx config.

## [0.0.18] -- v0.0.18 (main)

### Added

- SDDE pytree gfun support (PR #97).
- Integrators guide (`docs/integrators.rst`): ODE, SDE, DDE, SDDE with
  API reference, accuracy comparison, and advanced usage examples.

### Existing

- Jansen-Rit (JR) neural mass model.
- Montbrio-Pazo-Roxin (MPR) mean-field model.
- Balloon-Windkessel BOLD model.
- Bilinear DCM.
- Dopamine-modulated adaptive QIF model.
- BVEP epileptor model.
- SDE/ODE/DDE/SDDE integrators (`make_sde`, `make_ode`, `make_dde`,
  `make_sdde`).
- Structural connectivity and region mapping utilities.
- Sparse matrix-vector products.
- Spherical harmonic diffusion.
- Online monitors (BOLD, FC, time-average, covariance).
