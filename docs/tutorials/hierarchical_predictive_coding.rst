Hierarchical Predictive Coding with CMC Networks
=================================================

This tutorial shows how to build hierarchical predictive coding
networks using the CMC model.  The CMC's 4-population architecture
naturally separates feedforward and feedback pathways, implementing the
key prediction in Bastos et al. (2012): forward connections target
granular layer IV (spiny stellate cells) while backward connections
target agranular layers (superficial and deep pyramidal cells).

.. contents:: In this tutorial
   :local:
   :depth: 2

Background: Forward and Backward Pathways
------------------------------------------

In the cortical hierarchy, information flows asymmetrically:

* **Forward (bottom-up)**: Prediction errors ascend from lower to higher
  areas, entering the granular layer (ss).  The source is the
  superficial pyramidal population (sp) of the lower area.
* **Backward (top-down)**: Predictions descend from higher to lower
  areas, entering the agranular layers (sp + dp).  The source is the
  deep pyramidal population (dp) of the higher area.

This asymmetry is absent in standard models (JR, MPR, RWW) where
coupling enters through a single undifferentiated channel.

The Hierarchical CMC Functions
------------------------------

vbjax provides three levels of hierarchical CMC:

1. :func:`~vbjax.neural_mass.cmc_hier_dfun` -- base function with
   explicit forward and backward coupling inputs
2. :func:`~vbjax.neural_mass.cmc_hier_2node_dfun` -- two-node
   hierarchy (e.g. V1 <-> V4)
3. :func:`~vbjax.neural_mass.cmc_hier_Nnode_dfun` -- N-node hierarchy
   with separate forward and backward connectivity matrices

Base Hierarchical Dynamics
^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~vbjax.neural_mass.cmc_hier_dfun` extends the standard CMC by
routing coupling to the correct populations:

.. code-block:: python

   import jax.numpy as jnp
   import vbjax as vb

   ys = jnp.zeros(8)
   p = vb.cmc_default_theta

   # Forward coupling enters ss (granular layer IV)
   dys_fwd = vb.cmc_hier_dfun(ys, c_fwd=200.0, c_bwd=0.0, p=p)

   # Backward coupling enters sp + dp (agranular layers)
   dys_bwd = vb.cmc_hier_dfun(ys, c_fwd=0.0, c_bwd=200.0, p=p)

   # With c_bwd=0, reduces exactly to cmc_dfun
   dys_standard = vb.cmc_dfun(ys, 200.0, p)
   # dys_fwd == dys_standard  (numerically identical)

Two-Node Hierarchy
------------------

The simplest predictive coding circuit has two nodes: a lower sensory
area and a higher association area.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import vbjax as vb

   # Node 0 = lower area (e.g. V1)
   # Node 1 = higher area (e.g. V4)
   G_fwd = 80.0   # forward coupling gain
   G_bwd = 80.0   # backward coupling gain
   node_p = vb.cmc_default_theta._replace(I=220.0)
   p = (G_fwd, G_bwd, node_p)

   # Build SDE integrator
   dt = 0.5
   _, loop = vb.make_sde(dt=dt, dfun=vb.cmc_hier_2node_dfun, gfun=1e-3)

   # Integrate
   y0 = jnp.zeros((8, 2))
   key = jax.random.PRNGKey(3)
   n_steps = 10000
   zs = jax.random.normal(key, (n_steps, 8, 2))
   ys = loop(y0, zs, p)
   # ys shape: (10000, 8, 2)

   # Extract signals from each node
   sp_lower  = ys[:, 1, 0]  # prediction errors from V1
   sp_higher = ys[:, 1, 1]  # prediction errors from V4
   dp_higher = ys[:, 3, 1]  # predictions from V4

Forward/Backward Coupling Asymmetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The key predictive coding prediction is that forward and backward
coupling produce qualitatively different response patterns.  You can
test this by comparing three conditions:

.. code-block:: python

   import numpy as np

   conditions = {
       'Forward only':     (80.0,  0.0),
       'Backward only':    ( 0.0, 80.0),
       'Bidirectional':    (80.0, 80.0),
   }

   for label, (gf, gb) in conditions.items():
       p = (gf, gb, node_p)
       _, loop = vb.make_sde(dt=0.5, dfun=vb.cmc_hier_2node_dfun, gfun=1e-3)
       ys = loop(jnp.zeros((8, 2)), zs, p)

       # Cross-node sp correlation (skip transient)
       sp0 = np.array(ys[5000:, 1, 0])
       sp1 = np.array(ys[5000:, 1, 1])
       corr = np.corrcoef(sp0, sp1)[0, 1]
       print(f"  {label:20s}  sp0-sp1 correlation = {corr:+.3f}")

Forward-only coupling synchronizes prediction errors across areas.
Backward-only coupling introduces predictions that modulate the lower
area.  Bidirectional coupling creates the full predictive coding loop,
with characteristic lead-lag relationships between sp and dp signals.

N-Node Generalization
---------------------

For larger hierarchies, use :func:`~vbjax.neural_mass.cmc_hier_Nnode_dfun`
with separate forward and backward structural connectivity matrices:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import vbjax as vb

   n_nodes = 6

   # Define hierarchy via separate connectivity matrices
   # SC_fwd[i, j] > 0 means j sends forward projections to i
   # SC_bwd[i, j] > 0 means j sends backward projections to i
   key = jax.random.PRNGKey(0)
   k1, k2 = jax.random.split(key)
   SC_fwd = jnp.abs(jax.random.normal(k1, (n_nodes, n_nodes))) * 0.1
   SC_bwd = jnp.abs(jax.random.normal(k2, (n_nodes, n_nodes))) * 0.1

   G_fwd = 1.0
   G_bwd = 1.0
   node_p = vb.cmc_default_theta._replace(I=200.0)
   p = (SC_fwd, SC_bwd, G_fwd, G_bwd, node_p)

   # Integrate
   _, loop = vb.make_sde(dt=0.5, dfun=vb.cmc_hier_Nnode_dfun, gfun=1e-3)
   y0 = jnp.zeros((8, n_nodes))
   zs = jax.random.normal(jax.random.PRNGKey(1), (8000, 8, n_nodes))
   ys = loop(y0, zs, p)
   # ys shape: (8000, 8, 6)

Constructing Hierarchy Matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In practice, the forward and backward matrices encode the cortical
hierarchy.  A simple approach for a linear chain:

.. code-block:: python

   n = 5  # 5-area hierarchy: V1 -> V2 -> V4 -> TEO -> TE

   SC_fwd = jnp.zeros((n, n))
   SC_bwd = jnp.zeros((n, n))

   for i in range(n - 1):
       SC_fwd = SC_fwd.at[i + 1, i].set(1.0)  # forward: lower -> higher
       SC_bwd = SC_bwd.at[i, i + 1].set(1.0)  # backward: higher -> lower

For real anatomical hierarchies, derive forward/backward labels from
tract-tracing data (e.g. Markov et al. 2014) or from the laminar
patterns in diffusion MRI connectivity.

Layer-Resolved BOLD via vpjax Bridge
-------------------------------------

A unique advantage of the CMC is that each population maps to a specific
cortical layer, enabling layer-resolved hemodynamic modeling.  The
:func:`~vbjax.neural_mass.cmc_to_layer_activity` function provides this
bridge:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import vbjax as vb

   # Run a CMC simulation
   theta = vb.cmc_default_theta._replace(I=250.0)
   _, loop = vb.make_sde(
       dt=0.5,
       dfun=lambda y, p: vb.cmc_dfun(y, 0.0, p),
       gfun=1e-3,
   )
   y0 = jnp.zeros(8)
   key = jax.random.PRNGKey(17)
   zs = jax.random.normal(key, (8000, 8))
   ys = loop(y0, zs, theta)

   # Map to 3-layer activity (matches vpjax.LayerNVCParams ordering)
   #   Layer 0: deep (dp -> layers V/VI)
   #   Layer 1: middle (ss -> layer IV)
   #   Layer 2: superficial (sp -> layers II/III)
   layer_activity = jax.vmap(vb.cmc_to_layer_activity)(ys)
   # layer_activity shape: (8000, 3)

These per-layer neural activity time series can be fed directly into
per-layer Balloon-Windkessel models (e.g. the Riera model in vpjax)
to generate layer-resolved BOLD predictions, replacing the heuristic
``feedforward_frac`` / ``feedback_frac`` parameters.

The bridge is fully differentiable, so it supports end-to-end gradient-
based fitting of CMC + hemodynamic model parameters from laminar fMRI
data.

References
----------

* Bastos AM et al. (2012) Canonical microcircuits for predictive coding.
  *Neuron* 76(4):695-711.
* Douglas PK (2025) Computing with canonical microcircuits.
  arXiv:2508.06501.
* Markov NT et al. (2014) A weighted and directed interareal connectivity
  matrix for macaque cerebral cortex. *Cerebral Cortex* 24(1):17-36.
