# Model Comparison Findings

## Overview

Bayesian model comparison of 5 neural mass model families fitted to
82 subjects of resting-state EEG from Hartoyo et al. (2019, 2020),
using both time-domain SDE simulation and analytical transfer functions.

## Models

| Model | States | Origin | Key mechanism |
|-------|--------|--------|---------------|
| **Jansen-Rit (JR)** | 6 | Jansen & Rit 1995 | 3-population, 2nd-order PSP kernels |
| **CMC** | 8 | Bastos et al. 2012 | 4-population laminar microcircuit |
| **Liley** | 14 | Liley et al. 2002 | Conductance-based synapses, axonal propagation |
| **RRW** | 10 | Robinson et al. 2001/2002 | Corticothalamic loop delay (~85 ms) |
| **CBEI** | 8 | Coombes & Byrne 2019 | Exact QIF mean-field, conductance synapses |

## Key Findings

### 1. Fitting methodology dominates model selection

The choice of **how** to fit (analytical transfer function vs time-domain
simulation + Welch PSD) has a larger effect on model comparison results
than the choice of neural mass model.

**Analytical transfer functions** (linearized spectral analysis):
- RRW: loss = 10-20 per subject (purpose-built transfer function with delay)
- JR: loss = 85-150 (generic Jacobian linearization)
- CMC: loss = 9000+ (generic linearization misses CMC's spectral structure)

**Time-domain SDE simulation** (simulate → Welch PSD → fit):
- CMC: loss = 1900-2200 per subject
- Liley: loss = 1900-2800
- RRW: loss = 6000-9000 (ODE form without delay cannot produce alpha)
- CBEI: loss = 15000-20000

### 2. RRW analytical transfer function is the best spectral model

When using the purpose-built Robinson transfer function (with explicit
`exp(iωt₀)` delay terms), the RRW model fits real EEG spectra
dramatically better than any other model:

| Model | Eyes-Closed mean F | Eyes-Open mean F |
|-------|-------------------|-----------------|
| **RRW (analytical)** | **-27** | **-19** |
| JR (analytical) | -102 | -396 |
| CMC (analytical) | -9194 | -12520 |

However, this comparison is partially unfair: the RRW uses a hand-derived
transfer function that captures the delay analytically, while JR and CMC
use a generic Jacobian-based linearization that cannot represent delays.

### 3. Time-domain SDE results: condition-dependent model selection

When using the SDE simulation approach (which has its own limitations):

**Eyes-closed resting EEG:**
- CMC wins (FFX p=1.0, 82/82 subjects)
- CMC's laminar architecture is more parsimonious for resting spectra

**Eyes-open alpha-blocking:**
- Liley wins (FFX p=1.0, 82/82 subjects)
- Liley's conductance-based synapses and time constants capture the
  state-dependent shift during alpha suppression

This condition-dependent switch is physiologically meaningful but the
margins are large and every subject agrees, which suggests the
differences are driven by model structure rather than individual variation.

### 4. RRW without delays is not a valid competitor

The RRW model's alpha-generating mechanism IS the corticothalamic loop
delay. Without it (ODE form), the model cannot produce an alpha peak and
fits are 10x worse than CMC/Liley. Our SDDE implementation with the delay
produces active dynamics (eigenvalue analysis confirms 9.8 Hz mode at
damping -0.00004) but the alpha resonance doesn't emerge as a clear peak
in noisy time-domain simulation — it appears as elevated alpha-band power
(2.25x broadband) rather than a sharp spectral peak.

The sharp alpha peaks in Robinson's publications come from the **analytical
transfer function**, not from time-domain simulation. This is standard
practice in the neural field theory literature.

### 5. AgentSciML evolutionary optimization improved fitting

The AgentSciML multi-agent loop evolved the fitting strategy from
baseline (score 0.094) to best (score 0.155), a 65% improvement:
- Discovered that **connectivity parameters** (gains, time constants)
  matter more than input drive for spectral fitting
- RRW `I` parameter was stuck at 0; replacing with `nu_ee, nu_ei, nu_se, nu_re`
  fixed the problem
- Optimal settings: lr=0.05, noise_sigma=0.0003, 200 steps, 5 free params

### 6. Practical recommendations

For spectral model comparison of neural mass models:
1. **Use analytical transfer functions** where available — they are faster,
   more accurate, and avoid simulation noise confounds
2. **Hand-derive model-specific transfer functions** rather than relying on
   generic Jacobian linearization — the generic approach misses model-specific
   features (delays, specific PSP kernel structure)
3. **Report the fitting methodology** alongside model comparison results —
   the method has as much influence as the model choice
4. **Use condition-dependent comparison** (resting vs perturbation) to
   reveal model-specific advantages that aren't visible in resting data alone

## Data

- Hartoyo et al. (2019): 82 subjects, eyes-closed resting EEG, 2-20 Hz
- Hartoyo et al. (2020): same 82 subjects, eyes-closed + eyes-open (alpha-blocking)
- Frequency resolution: 0.25 Hz, 72 bins

## Infrastructure

- Models implemented in JAX via vbjax (fork of ins-amu/vbjax)
- BMS: SPM-style Laplace/BIC free energy + FFX/RFX (Stephan et al. 2009)
- GPU: RTX 2080 (local), DGX Spark GB10 (cluster)
- AgentSciML: multi-agent evolutionary optimization of fitting strategies

## References

- Bastos AM et al. (2012) Neuron 76(4):695-711
- Bojak I & Liley DTJ (2005) Phys Rev E 71:041902
- Coombes S & Byrne A (2019) J Neurophysiol 122:1275-1287
- Hartoyo A et al. (2019) PLoS Comp Biol 15(5):e1006694
- Hartoyo A et al. (2020) PLoS Comp Biol 16(4):e1007662
- Jansen BH & Rit VG (1995) Biol Cybern 73:357-366
- Liley DTJ et al. (2002) Network 13:67-113
- Robinson PA et al. (2001) Phys Rev E 63:021903
- Robinson PA et al. (2002) Phys Rev E 65:041924
- Stephan KE et al. (2009) NeuroImage 46(4):1004-1017
- Bastiaens SP et al. (2025) PLoS Comp Biol — Alpha models comparison
