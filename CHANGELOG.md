# Changelog

All notable changes to vbjax are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] -- feature/cmc-linearization

### Added

- **Analytical transfer functions** (`vbjax/transfer.py`) -- linearized
  spectral analysis for all models without time-domain simulation.
  - Generic `linearized_transfer_function()`: Jacobian-based T(ω) = C(iωI-J)⁻¹B
  - `jr_analytical_psd`, `cmc_analytical_psd`, `liley_analytical_psd`,
    `cbei_analytical_psd` -- model-specific wrappers
  - `rrw_transfer_function` / `rrw_analytical_psd` -- purpose-built
    Robinson corticothalamic transfer function with explicit delay terms
    (based on braintrak/NFTsim implementation)
- **Bayesian model selection** (`vbjax/bms.py`) -- SPM-style BMS:
  - Laplace free energy via `jax.hessian`
  - BIC approximation for tractable model comparison
  - Fixed-effects and random-effects BMS (Stephan et al. 2009)
  - Exceedance and protected exceedance probabilities (Rigoux et al. 2014)
- **Spectral fitting pipeline** (`vbjax/spectral.py`) -- differentiable
  Welch PSD, model inversion, MAP estimation via Adam
- **Hartoyo EEG data loader** (`vbjax/hartoyo.py`) -- 82-subject resting
  and alpha-blocking spectra from Hartoyo et al. (2019, 2020)
- **RRW SDDE** -- corticothalamic model with explicit delay via `make_sdde`:
  - `rrw_sdde_dfun`, `make_rrw_sdde`, `rrw_delay_steps`
  - 10-state model with proper second-order NFTsim dendritic filters
  - NFTsim canonical parameters (Robinson 2005, Bastiaens et al. 2025)
- **Liley SDDE** -- network form with delayed inter-regional coupling:
  - `liley_sdde_dfun`, `liley_sdde_net_dfun`
- **Liley adhoc** -- membrane potential clamping for numerical stability
- **AgentSciML adapter** -- evolutionary optimization of fitting strategies
- **DGX Spark scripts** -- Slurm job submission for GPU cluster runs
- **Liley mean-field cortical model** (Liley, Cadusch & Dafilis 2002) --
  14 state variables with conductance-based (shunting) synapses, alpha-function
  PSP kernels, and damped-wave long-range axonal propagation.
  - `liley_dfun`, `liley_net_dfun`, `liley_observe_eeg`.
  - `LileyTheta`, `LileyState` namedtuples with all 26 parameters from the
    original publication.
  - `liley_default_theta` / `liley_default_state`.
- **Bojak-Liley pharmacological extension** (Bojak & Liley 2005) --
  drug-dependent modulation of inhibitory PSP kernel parameters for
  modeling GABAergic anesthetics (propofol, isoflurane, etc.).
  - `liley_pharma_dfun` with `LileyPharmaTheta`.
- **Coombes-Byrne next-generation neural mass** (Byrne et al. 2017,
  Coombes & Byrne 2019) -- exact mean-field of QIF network with
  alpha-function conductance-based synapses.
  - Single population (4D): `cb_dfun`, `cb_net_dfun`, `cb_r_positive`.
  - E-I two-population (8D): `cbei_dfun`, `cbei_net_dfun`.
- **Robinson-Rennie-Wright corticothalamic model** (Robinson et al. 2002) --
  4-population (cortical E/I + thalamic relay/reticular) model with
  corticothalamic loop delay generating alpha rhythm.
  - `rrw_dfun`, `rrw_net_dfun`, `rrw_observe_phi`.
- All new models exported from `vbjax.__init__`.
- Comprehensive test suites for Liley, CB, CBEI, and RRW models.

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
