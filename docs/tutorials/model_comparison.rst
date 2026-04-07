Comparing Neural Mass Models: CMC vs JR vs MPR
================================================

This tutorial compares the Canonical Microcircuit (CMC) against the
Jansen-Rit (JR) and Montbrio-Pazo-Roxin (MPR) models across several
analysis dimensions.  The goal is to help you choose the right model for
your research question and to understand the trade-offs in complexity,
interpretability, and computational cost.

See ``examples/cmc_comparison.py`` for runnable code producing all
comparisons below.

.. contents:: In this tutorial
   :local:
   :depth: 2

Model Summary
-------------

.. list-table::
   :header-rows: 1
   :widths: 12 8 8 18 12 42

   * - Model
     - States
     - Params
     - Populations
     - PSP order
     - Unique capability
   * - **CMC**
     - 8
     - 16
     - ss, sp, ii, dp
     - 2nd-order
     - Laminar decomposition, forward/backward asymmetry, predictive coding
   * - **JR**
     - 6
     - 14
     - pyr, ei, ii
     - 2nd-order
     - Established workhorse, well-characterized bifurcation
   * - **MPR**
     - 2
     - 7
     - r, V (mean-field)
     - 1st-order
     - Exact mean-field of QIF, analytical tractability
   * - **Dopa**
     - 6
     - 27
     - r, V, u, Sa, Sg, Dp
     - Mixed
     - Dopamine modulation, 3-channel coupling

Spectral Fingerprints
---------------------

Both CMC and JR use 2nd-order PSP kernels and the same sigmoid firing
rate function, so spectral differences isolate the effect of **circuit
architecture** alone.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import numpy as np
   from scipy.signal import welch
   import vbjax as vb

   dt = 0.5
   n_steps = 16000  # 8 seconds
   key = jax.random.PRNGKey(42)

   # CMC
   p_cmc = vb.cmc_default_theta
   _, loop_cmc = vb.make_sde(
       dt=dt, dfun=lambda y, p: vb.cmc_dfun(y, 0.0, p), gfun=1e-3)
   zs_cmc = jax.random.normal(key, (n_steps, 8))
   ys_cmc = loop_cmc(jnp.zeros(8), zs_cmc, p_cmc)
   sp_cmc = np.array(ys_cmc[4000:, 1])  # superficial pyramidal

   # JR
   p_jr = vb.jr_default_theta._replace(I=220.0)
   _, loop_jr = vb.make_sde(
       dt=dt, dfun=lambda y, p: vb.jr_dfun(y, 0.0, p), gfun=1e-3)
   zs_jr = jax.random.normal(key, (n_steps, 6))
   ys_jr = loop_jr(jnp.zeros(6), zs_jr, p_jr)
   pyr_jr = np.array(ys_jr[4000:, 1] - ys_jr[4000:, 2])  # y1 - y2

   # Compare PSD
   fs = 1000.0 / dt
   f_cmc, pxx_cmc = welch(sp_cmc, fs=fs, nperseg=1024)
   f_jr, pxx_jr = welch(pyr_jr, fs=fs, nperseg=1024)

   print(f"CMC peak: {f_cmc[np.argmax(pxx_cmc[1:])+1]:.1f} Hz")
   print(f"JR  peak: {f_jr[np.argmax(pxx_jr[1:])+1]:.1f} Hz")

**Expected outcome**: JR produces a single alpha peak (~10 Hz).  The CMC
may produce a richer spectrum with contributions from both the fast
sp-ii loop and the slower dp-ii loop.

**CMC strength**: Multi-band spectra from a single cortical column.

**CMC weakness**: May require more careful parameter tuning for a clean
alpha peak.

Bifurcation Analysis
--------------------

Sweeping the external input current ``I`` reveals the oscillation onset
(Hopf bifurcation) and the relationship between drive strength and
oscillation amplitude.

.. code-block:: python

   I_range = np.linspace(0, 500, 30)
   cmc_amps, jr_amps = [], []

   for I_val in I_range:
       # CMC
       p = vb.cmc_default_theta._replace(I=float(I_val))
       _, loop = vb.make_sde(
           dt=0.5,
           dfun=lambda y, p: vb.cmc_dfun(y, 0.0, p),
           gfun=1e-4)
       ys = loop(jnp.zeros(8),
                 jax.random.normal(key, (6000, 8)), p)
       cmc_amps.append(float(jnp.std(ys[3000:, 1])))

       # JR
       p = vb.jr_default_theta._replace(I=float(I_val))
       _, loop = vb.make_sde(
           dt=0.5,
           dfun=lambda y, p: vb.jr_dfun(y, 0.0, p),
           gfun=1e-4)
       ys = loop(jnp.zeros(6),
                 jax.random.normal(key, (6000, 6)), p)
       jr_amps.append(float(jnp.std(ys[3000:, 1] - ys[3000:, 2])))

   cmc_amps = np.array(cmc_amps)
   jr_amps = np.array(jr_amps)

   # Find onset
   cmc_onset = I_range[np.argmax(cmc_amps > 0.01)]
   jr_onset = I_range[np.argmax(jr_amps > 0.01)]
   print(f"CMC onset: I ~ {cmc_onset:.0f}")
   print(f"JR  onset: I ~ {jr_onset:.0f}")

**Expected outcome**: Both models show a supercritical Hopf bifurcation.
The CMC may show a secondary bifurcation at higher ``I`` due to its
richer recurrent structure.

**CMC strength**: Richer bifurcation structure (potential period doubling
or torus bifurcation at high drive).

**CMC weakness**: No analytical bifurcation results available (unlike
MPR, which has closed-form Hopf conditions).

Laminar Decomposition
---------------------

This is the CMC's unique capability.  No other vbjax model separates
superficial and deep pyramidal output.

.. code-block:: python

   p = vb.cmc_default_theta._replace(I=250.0)
   _, loop = vb.make_sde(
       dt=0.5,
       dfun=lambda y, p: vb.cmc_dfun(y, 0.0, p),
       gfun=1e-3)
   ys = loop(jnp.zeros(8),
             jax.random.normal(key, (8000, 8)), p)

   sp = np.array(ys[4000:, 1])  # EEG-like (prediction errors)
   dp = np.array(ys[4000:, 3])  # LFP-like (predictions)

   corr = np.corrcoef(sp, dp)[0, 1]
   print(f"sp-dp correlation: {corr:.3f}")
   print(f"sp amplitude: {np.std(sp):.4f}")
   print(f"dp amplitude: {np.std(dp):.4f}")

The sp and dp signals are correlated but not identical, reflecting
their distinct circuit positions.  Under feedforward input (bottom-up),
sp leads dp.  Under feedback input (top-down), dp leads sp.  This is
the predictive coding signature.

**CMC strength**: Testable predictions for simultaneous EEG-LFP
recordings.

**CMC weakness**: Adds 2 extra unobserved states if you only measure EEG.

Step Response
-------------

The CMC's 4-population circuit creates a temporal cascade absent in
JR's 3-population model:

.. code-block:: python

   # Baseline (no drive)
   p_off = vb.cmc_default_theta._replace(I=0.0)
   _, loop_off = vb.make_sde(
       dt=0.5,
       dfun=lambda y, _: vb.cmc_dfun(y, 0.0, p_off),
       gfun=0.0)
   ys_pre = loop_off(jnp.zeros(8), jnp.zeros((500, 8)), None)

   # Step on
   p_on = vb.cmc_default_theta._replace(I=300.0)
   _, loop_on = vb.make_sde(
       dt=0.5,
       dfun=lambda y, _: vb.cmc_dfun(y, 0.0, p_on),
       gfun=0.0)
   ys_post = loop_on(ys_pre[-1], jnp.zeros((2000, 8)), None)

   # Check peak latencies
   for i, name in enumerate(['ss', 'sp', 'ii', 'dp']):
       peak_t = int(jnp.argmax(jnp.abs(ys_post[:, i]))) * 0.5
       print(f"  {name}: peak latency = {peak_t:.1f} ms")

**Expected outcome**: Sequential activation: ss responds first (direct
thalamic target), then sp, then ii, then dp.  JR activates all three
populations near-simultaneously because its circuit lacks the ss -> sp
-> dp relay chain.

**CMC strength**: Captures temporal sequencing across cortical layers.

**CMC weakness**: Longer transient to reach steady state.

Computational Cost
------------------

More state variables per node means more computation per integration
step.  The cost comparison matters for whole-brain simulations with
hundreds of nodes:

.. code-block:: python

   import time

   models = {
       'CMC (8D)': (lambda y, _: vb.cmc_dfun(y, 0.0, vb.cmc_default_theta),
                    jnp.zeros(8)),
       'JR  (6D)': (lambda y, _: vb.jr_dfun(y, 0.0, vb.jr_default_theta),
                    jnp.zeros(6)),
       'MPR (2D)': (lambda y, _: vb.mpr_dfun(y, jnp.array([0., 0.]),
                    vb.mpr_default_theta), jnp.zeros(2)),
   }

   for name, (dfun, y0) in models.items():
       _, loop = vb.make_sde(dt=0.5, dfun=dfun, gfun=1e-3)
       zs = jax.random.normal(key, (1000,) + y0.shape)
       _ = loop(y0, zs, None)  # warmup JIT
       times = []
       for _ in range(5):
           t0 = time.perf_counter()
           _ = loop(y0, zs, None).block_until_ready()
           times.append(time.perf_counter() - t0)
       print(f"  {name}: {np.mean(times)*1000:.2f} ms / 1000 steps")

The CMC is typically ~20% slower than JR per node and ~3--4x slower
than MPR.  For whole-brain simulations with 68+ nodes, this overhead
is usually negligible compared to the matrix-vector product for network
coupling.

When to Use Which Model
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Research question
     - Recommended model
   * - Resting-state FC fitting
     - **JR** or **MPR** (fewer parameters, well-characterized)
   * - Laminar-resolved fMRI / simultaneous EEG-LFP
     - **CMC** (only model with layer-specific output)
   * - Hierarchical predictive coding
     - **CMC** (forward/backward pathway asymmetry)
   * - Analytical bifurcation analysis
     - **MPR** (exact mean-field, closed-form Hopf)
   * - Dopaminergic modulation
     - **Dopa** (explicit dopamine dynamics)
   * - Quick prototyping / teaching
     - **JR** (most documentation, simplest circuit)
   * - Whole-brain with >100 nodes, GPU
     - **MPR** (2 states/node, minimal memory)

References
----------

* Bastos AM et al. (2012) Canonical microcircuits for predictive coding.
  *Neuron* 76(4):695-711.
* Jansen BH, Rit VG (1995) Electroencephalogram and visual evoked
  potential generation in a mathematical model of coupled cortical
  columns. *Biol Cybern* 73:357-366.
* Montbrio E, Pazo D, Roxin A (2015) Macroscopic description for
  networks of spiking neurons. *Phys Rev X* 5:021028.
* Deco G et al. (2013) Resting-state functional connectivity emerges
  from structurally and dynamically shaped slow linear fluctuations.
  *J Neurosci* 33(27):11239-11252.
