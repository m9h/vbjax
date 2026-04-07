The Canonical Microcircuit: 4-Population Neural Mass Model
===========================================================

This tutorial introduces the Canonical Microcircuit (CMC) implementation in
vbjax.  The CMC is a neural mass model of a cortical column with four
populations arranged by cortical layer, based on Bastos et al. (2012) and
Douglas (2025).

.. contents:: In this tutorial
   :local:
   :depth: 2

Architecture
------------

The CMC contains four neural populations:

.. list-table::
   :header-rows: 1
   :widths: 15 40 25 20

   * - Population
     - Role
     - Cortical layer
     - PSP kernel
   * - **ss** (spiny stellate)
     - Thalamic input / feedforward relay
     - Granular (IV)
     - Excitatory (He, a)
   * - **sp** (superficial pyramidal)
     - Prediction error output, EEG/MEG generator
     - Supragranular (II/III)
     - Excitatory (He, a)
   * - **ii** (inhibitory interneurons)
     - Local inhibition across layers
     - All layers
     - Inhibitory (Hi, b)
   * - **dp** (deep pyramidal)
     - Prediction output, feedback source
     - Infragranular (V/VI)
     - Excitatory (He, a)

Intrinsic connectivity:

* **Excitatory**: ss -> sp, sp -> ii, sp -> dp, dp -> ii, dp -> sp
* **Inhibitory**: ii -> ss, ii -> sp, ii -> dp

This circuit creates two recurrent loops: a fast supragranular loop
(sp <-> ii) and a slower infragranular loop (dp <-> ii), which can
generate multi-band oscillatory spectra from a single cortical column.

State Variables
---------------

Each population has a 2nd-order post-synaptic potential (PSP) kernel,
converted to a 1st-order system of 2 variables (membrane potential ``x``
and its derivative ``v``), giving 8 state variables total:

.. code-block:: text

   [x_ss, x_sp, x_ii, x_dp, v_ss, v_sp, v_ii, v_dp]

The dynamics for each excitatory population follow:

.. math::

   \dot{v}_k = H_e \cdot a \cdot I_k - 2a \cdot v_k - a^2 \cdot x_k

and for the inhibitory population:

.. math::

   \dot{v}_{ii} = H_i \cdot b \cdot I_{ii} - 2b \cdot v_{ii} - b^2 \cdot x_{ii}

where :math:`I_k` is the total synaptic input to population :math:`k`.

Parameters
----------

The parameter set (:class:`~vbjax.neural_mass.CMCTheta`) has 16 fields:

* 7 PSP/sigmoid parameters (``He``, ``Hi``, ``a``, ``b``, ``r``, ``v0``,
  ``nu_max``) -- directly comparable to Jansen-Rit
* 8 intrinsic connectivity gains (``g_ss_sp``, ``g_sp_ii``, ``g_sp_dp``,
  ``g_dp_ii``, ``g_dp_sp``, ``g_ii_ss``, ``g_ii_sp``, ``g_ii_dp``)
* 1 external drive (``I``)

The default parameters were tuned via differential evolution to produce a
spectral peak in the alpha band (8--13 Hz) with stable bounded dynamics.
See ``examples/cmc_tune_defaults.py`` for the optimization procedure.

Single-Node Simulation
----------------------

A minimal example that integrates the CMC under stochastic drive:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import vbjax as vb

   # Parameters and initial state
   theta = vb.cmc_default_theta
   y0 = jnp.zeros(8)

   # Build an SDE integrator (dt=0.5 ms, additive noise sigma=1e-3)
   _, loop = vb.make_sde(
       dt=0.5,
       dfun=lambda y, p: vb.cmc_dfun(y, 0.0, p),
       gfun=1e-3,
   )

   # Generate noise and integrate
   key = jax.random.PRNGKey(42)
   n_steps = 10000  # 5 seconds at dt=0.5 ms
   zs = jax.random.normal(key, (n_steps, 8))
   ys = loop(y0, zs, theta)

   # ys has shape (10000, 8)
   # ys[:, 1] is the superficial pyramidal membrane potential (EEG-like)

Observing the Output
^^^^^^^^^^^^^^^^^^^^

The CMC provides two observation functions that extract physiologically
meaningful signals:

.. code-block:: python

   # EEG/MEG-like signal (superficial pyramidal prediction errors)
   eeg = vb.cmc_observe_sp(ys.T)  # shape (n_steps,)

   # LFP-like signal (deep pyramidal predictions)
   lfp = vb.cmc_observe_dp(ys.T)  # shape (n_steps,)

Noise-Driven Oscillations
--------------------------

The CMC generates oscillatory dynamics when driven by noise.  The circuit
architecture -- specifically the excitatory-inhibitory loops through the
four populations -- shapes the spectral content of the output.

.. code-block:: python

   import numpy as np
   from scipy.signal import welch

   # Use the superficial pyramidal signal (skip 2s transient)
   sp = np.array(ys[4000:, 1])
   fs = 1000.0 / 0.5  # sampling rate in Hz

   f, pxx = welch(sp, fs=fs, nperseg=1024)
   peak_freq = f[np.argmax(pxx[1:]) + 1]
   print(f"Peak frequency: {peak_freq:.1f} Hz")

With the default parameters, the CMC typically produces a peak in the
alpha range.  The separate E/I loops through sp-ii (fast) and dp-ii
(slower) can produce richer multi-band spectra compared to the 3-population
Jansen-Rit model.

Modifying Parameters
--------------------

All parameters are accessible via namedtuple ``_replace``:

.. code-block:: python

   # Increase external drive
   theta_driven = vb.cmc_default_theta._replace(I=400.0)

   # Weaken inhibition to ss
   theta_disinhibit = vb.cmc_default_theta._replace(g_ii_ss=50.0)

   # Zero input (resting state)
   theta_rest = vb.cmc_default_theta._replace(I=0.0)

Differentiability
-----------------

All CMC functions are compatible with JAX transforms.  You can compute
gradients through the dynamics for parameter fitting, sensitivity
analysis, or optimal control:

.. code-block:: python

   # Jacobian of the dynamics w.r.t. state
   J = jax.jacobian(lambda y: vb.cmc_dfun(y, 0.0, theta))(jnp.zeros(8))
   # J has shape (8, 8) -- the linearized system matrix

   # Gradient of a loss w.r.t. coupling input
   def loss(c):
       dys = vb.cmc_dfun(jnp.zeros(8), c, theta)
       return jnp.sum(dys ** 2)

   grad_c = jax.grad(loss)(0.0)

Network Simulation
------------------

For multi-node simulations, use :func:`~vbjax.neural_mass.cmc_net_dfun`
with a structural connectivity matrix:

.. code-block:: python

   n_nodes = 68
   SC = ...  # (68, 68) structural connectivity matrix
   G = 1.0   # global coupling strength

   p = (SC, G, vb.cmc_default_theta._replace(I=200.0))
   _, loop = vb.make_sde(dt=0.5, dfun=vb.cmc_net_dfun, gfun=1e-3)

   y0 = jnp.zeros((8, n_nodes))
   zs = jax.random.normal(key, (n_steps, 8, n_nodes))
   ys = loop(y0, zs, p)
   # ys has shape (n_steps, 8, n_nodes)

References
----------

* Bastos AM et al. (2012) Canonical microcircuits for predictive coding.
  *Neuron* 76(4):695-711.
* Douglas PK (2025) Computing with canonical microcircuits.
  arXiv:2508.06501.
* Moran RJ et al. (2013) Neural masses and fields in dynamic causal
  modeling. *Frontiers in Computational Neuroscience* 7:57.
