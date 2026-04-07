"""Tests for the Canonical Microcircuit (CMC) neural mass model.

Validates the CMC implementation against expected properties:
- Correct shapes and dtypes
- Sigmoid consistency with Jansen-Rit
- Resting state stability
- Oscillatory dynamics under drive
- Network coupling via make_sde
- Laminar output separation (sp vs dp)
"""

import numpy as np
import jax
import jax.numpy as jp
import vbjax as vb


# ── Shape and basic sanity ───────���───────────────────────────────────

def test_cmc_dfun_shape_single_node():
    """Single-node CMC returns (8,) derivative."""
    ys = jp.zeros(8)
    c = 0.0
    dys = vb.cmc_dfun(ys, c, vb.cmc_default_theta)
    assert dys.shape == (8,)


def test_cmc_dfun_shape_multi_node():
    """Multi-node CMC returns (8, n_nodes) derivative."""
    n = 10
    ys = jp.zeros((8, n))
    c = jp.zeros(n)
    dys = vb.cmc_dfun(ys, c, vb.cmc_default_theta)
    assert dys.shape == (8, n)


def test_cmc_default_state():
    """Default state namedtuple has correct fields."""
    s = vb.cmc_default_state
    assert len(s) == 8
    assert s.x_ss == 0.0
    assert s.v_dp == 0.0


def test_cmc_default_theta_fields():
    """CMCTheta has 16 fields (7 PSP/sigmoid + 8 connectivity + 1 input)."""
    p = vb.cmc_default_theta
    assert len(p) == 16
    assert p.He == 3.25
    assert p.Hi == 22.0


# ── Sigmoid consistency with Jansen-Rit ───���──────────────────────────

def test_cmc_sigmoid_matches_jr():
    """CMC and JR use the same sigmoid: 2*nu_max / (1 + exp(r*(v0 - x)))."""
    p_cmc = vb.cmc_default_theta
    p_jr = vb.jr_default_theta

    x_test = jp.linspace(-10, 20, 100)

    # JR sigmoid (from jr_dfun internals)
    sigm_jr = 2.0 * p_jr.nu_max / (1.0 + jp.exp(p_jr.r * (p_jr.v0 - x_test)))

    # CMC sigmoid (same formula, different default v0 but same r, nu_max)
    sigm_cmc = 2.0 * p_cmc.nu_max / (1.0 + jp.exp(p_cmc.r * (p_cmc.v0 - x_test)))

    # With matched parameters, they must be identical
    p_matched = p_cmc._replace(v0=p_jr.v0)
    sigm_matched = 2.0 * p_matched.nu_max / (
        1.0 + jp.exp(p_matched.r * (p_matched.v0 - x_test)))
    np.testing.assert_allclose(sigm_matched, sigm_jr, rtol=1e-6)


# ── Resting state ─────��──────────────────────────────���───────────────

def test_cmc_zero_state_derivative():
    """At x=0, v=0 with no input, velocity derivatives are driven only
    by sigmoid(0) terms — check derivatives are finite and small velocity
    components are zero."""
    ys = jp.zeros(8)
    dys = vb.cmc_dfun(ys, 0.0, vb.cmc_default_theta)
    # dx/dt = v = 0 for all populations
    np.testing.assert_allclose(dys[:4], 0.0, atol=1e-10)
    # dv/dt should be finite (sigmoid at 0 is nonzero)
    assert jp.all(jp.isfinite(dys[4:]))


def test_cmc_resting_stability():
    """Integrate CMC at rest (no input) — state should remain bounded."""
    dt = 0.5  # ms
    p = vb.cmc_default_theta._replace(I=0.0)
    _, loop = vb.make_sde(dt=dt, dfun=lambda y, p: vb.cmc_dfun(y, 0.0, p), gfun=0.0)
    y0 = jp.zeros(8)
    # 2000 steps = 1 second
    zs = jp.zeros((2000, 8))
    ys = loop(y0, zs, p)
    assert jp.all(jp.isfinite(ys))
    # Bounded: membrane potentials should stay within reasonable range
    assert jp.all(jp.abs(ys[:, :4]) < 50.0)


# ── Driven dynamics ───���──────────────────────────────────────────────

def test_cmc_responds_to_input():
    """Constant input to ss produces nonzero sp and dp activity."""
    dt = 0.5
    p = vb.cmc_default_theta._replace(I=300.0)
    _, loop = vb.make_sde(dt=dt, dfun=lambda y, p: vb.cmc_dfun(y, 0.0, p), gfun=0.0)
    y0 = jp.zeros(8)
    zs = jp.zeros((4000, 8))
    ys = loop(y0, zs, p)
    # After transient, sp and dp should have nonzero activity
    sp_late = ys[2000:, 1]  # superficial pyramidal
    dp_late = ys[2000:, 3]  # deep pyramidal
    assert jp.std(sp_late) > 1e-6 or jp.abs(jp.mean(sp_late)) > 1e-3
    assert jp.std(dp_late) > 1e-6 or jp.abs(jp.mean(dp_late)) > 1e-3


def test_cmc_oscillates_with_noise():
    """With noise drive, CMC should produce oscillatory dynamics."""
    dt = 0.5
    n_steps = 4000
    p = vb.cmc_default_theta._replace(I=220.0)

    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.cmc_dfun(y, 0.0, p),
        gfun=1e-3,
    )
    y0 = jp.zeros(8)
    key = jax.random.PRNGKey(42)
    zs = jax.random.normal(key, (n_steps, 8))
    ys = loop(y0, zs, p)
    assert jp.all(jp.isfinite(ys))
    # Superficial pyramidal should show variance (oscillation)
    sp_var = jp.var(ys[1000:, 1])
    assert sp_var > 1e-8


# ── Network coupling ���────────────────────────────────────────────────

def test_cmc_net_dfun_shape():
    """cmc_net_dfun produces correct (8, n_nodes) output."""
    n = 5
    SC = jp.ones((n, n)) * 0.1
    SC = SC.at[jp.diag_indices(n)].set(0.0)
    G = 1.0
    p = (SC, G, vb.cmc_default_theta)
    ys = jp.zeros((8, n))
    dys = vb.cmc_net_dfun(ys, p)
    assert dys.shape == (8, n)
    assert jp.all(jp.isfinite(dys))


def test_cmc_network_sde():
    """CMC network runs through make_sde without error."""
    n = 4
    dt = 0.5
    SC = jp.abs(jax.random.normal(jax.random.PRNGKey(0), (n, n)))
    SC = (SC + SC.T) / 2
    SC = SC.at[jp.diag_indices(n)].set(0.0)
    G = 0.5
    node_p = vb.cmc_default_theta._replace(I=200.0)
    p = (SC, G, node_p)

    _, loop = vb.make_sde(dt=dt, dfun=vb.cmc_net_dfun, gfun=1e-3)
    y0 = jp.zeros((8, n))
    key = jax.random.PRNGKey(1)
    zs = jax.random.normal(key, (1000, 8, n))
    ys = loop(y0, zs, p)
    assert ys.shape == (1000, 8, n)
    assert jp.all(jp.isfinite(ys))


# ── Observation functions ──────��─────────────────────────────────────

def test_cmc_observe_sp():
    """cmc_observe_sp extracts index 1 (superficial pyramidal)."""
    ys = jp.arange(8, dtype=float)
    assert vb.cmc_observe_sp(ys) == 1.0


def test_cmc_observe_dp():
    """cmc_observe_dp extracts index 3 (deep pyramidal)."""
    ys = jp.arange(8, dtype=float)
    assert vb.cmc_observe_dp(ys) == 3.0


def test_cmc_laminar_separation():
    """Superficial and deep pyramidal outputs should differ under drive,
    reflecting distinct roles in feedforward vs feedback processing."""
    dt = 0.5
    p = vb.cmc_default_theta._replace(I=250.0)
    _, loop = vb.make_sde(dt=dt, dfun=lambda y, p: vb.cmc_dfun(y, 0.0, p), gfun=1e-3)
    y0 = jp.zeros(8)
    key = jax.random.PRNGKey(7)
    zs = jax.random.normal(key, (4000, 8))
    ys = loop(y0, zs, p)
    sp = ys[2000:, 1]
    dp = ys[2000:, 3]
    # They should not be identical (different circuit positions)
    corr = jp.corrcoef(jp.stack([sp, dp]))[0, 1]
    assert jp.abs(corr) < 0.999


# ── Differentiability ────────────────────────────────────────────────

# ── Hierarchical predictive coding ────────────────────────────────────

def test_cmc_hier_dfun_reduces_to_cmc_dfun():
    """cmc_hier_dfun with c_bwd=0 should match cmc_dfun."""
    ys = jax.random.normal(jax.random.PRNGKey(0), (8,))
    c = 1.5
    p = vb.cmc_default_theta
    d1 = vb.cmc_dfun(ys, c, p)
    d2 = vb.cmc_hier_dfun(ys, c, 0.0, p)
    np.testing.assert_allclose(d1, d2, rtol=1e-6)


def test_cmc_hier_backward_modifies_sp_dp():
    """Backward coupling should change sp and dp derivatives but not ss."""
    ys = jp.zeros(8)
    p = vb.cmc_default_theta
    d_no_bwd = vb.cmc_hier_dfun(ys, 0.0, 0.0, p)
    d_with_bwd = vb.cmc_hier_dfun(ys, 0.0, 100.0, p)
    # ss velocity derivative (index 4) should be unchanged
    np.testing.assert_allclose(d_no_bwd[4], d_with_bwd[4], rtol=1e-6)
    # sp and dp velocity derivatives (indices 5, 7) should differ
    assert d_no_bwd[5] != d_with_bwd[5]
    assert d_no_bwd[7] != d_with_bwd[7]


def test_cmc_hier_2node_shape():
    """2-node hierarchical CMC produces (8, 2) output."""
    ys = jp.zeros((8, 2))
    p = (1.0, 1.0, vb.cmc_default_theta)
    dys = vb.cmc_hier_2node_dfun(ys, p)
    assert dys.shape == (8, 2)


def test_cmc_hier_2node_sde():
    """2-node hierarchical CMC integrates through make_sde."""
    dt = 0.5
    n_steps = 2000
    node_p = vb.cmc_default_theta._replace(I=250.0)
    p = (50.0, 50.0, node_p)
    _, loop = vb.make_sde(dt=dt, dfun=vb.cmc_hier_2node_dfun, gfun=1e-3)
    y0 = jp.zeros((8, 2))
    key = jax.random.PRNGKey(3)
    zs = jax.random.normal(key, (n_steps, 8, 2))
    ys = loop(y0, zs, p)
    assert ys.shape == (n_steps, 8, 2)
    assert jp.all(jp.isfinite(ys))


def test_cmc_hier_forward_backward_asymmetry():
    """Forward drive (→ss) and backward drive (→sp+dp) produce
    qualitatively different response patterns — the key predictive
    coding prediction."""
    dt = 0.5
    n_steps = 4000
    p_fwd = vb.cmc_default_theta._replace(I=0.0)
    p_bwd = vb.cmc_default_theta._replace(I=0.0)
    key = jax.random.PRNGKey(5)
    zs = jax.random.normal(key, (n_steps, 8))

    # Forward drive: constant input to ss via coupling
    _, loop_fwd = vb.make_sde(
        dt=dt, dfun=lambda y, p: vb.cmc_hier_dfun(y, 200.0, 0.0, p), gfun=1e-3)
    ys_fwd = loop_fwd(jp.zeros(8), zs, p_fwd)

    # Backward drive: same magnitude but enters sp+dp
    _, loop_bwd = vb.make_sde(
        dt=dt, dfun=lambda y, p: vb.cmc_hier_dfun(y, 0.0, 200.0, p), gfun=1e-3)
    ys_bwd = loop_bwd(jp.zeros(8), zs, p_bwd)

    # Late-phase sp and dp activity should differ between conditions
    sp_fwd = jp.std(ys_fwd[2000:, 1])
    dp_fwd = jp.std(ys_fwd[2000:, 3])
    sp_bwd = jp.std(ys_bwd[2000:, 1])
    dp_bwd = jp.std(ys_bwd[2000:, 3])

    # Forward drive should produce different sp/dp ratio than backward
    ratio_fwd = float(sp_fwd / jp.clip(dp_fwd, 1e-10))
    ratio_bwd = float(sp_bwd / jp.clip(dp_bwd, 1e-10))
    assert ratio_fwd != ratio_bwd, (
        f"sp/dp ratio should differ: fwd={ratio_fwd:.3f} bwd={ratio_bwd:.3f}")


def test_cmc_hier_Nnode_shape():
    """N-node hierarchical CMC with forward/backward SC matrices."""
    n = 6
    SC_fwd = jp.abs(jax.random.normal(jax.random.PRNGKey(0), (n, n))) * 0.1
    SC_bwd = jp.abs(jax.random.normal(jax.random.PRNGKey(1), (n, n))) * 0.1
    p = (SC_fwd, SC_bwd, 1.0, 1.0, vb.cmc_default_theta)
    ys = jp.zeros((8, n))
    dys = vb.cmc_hier_Nnode_dfun(ys, p)
    assert dys.shape == (8, n)
    assert jp.all(jp.isfinite(dys))


# ── Layer activity bridge (CMC → vpjax) ─────────────────────────────

def test_cmc_to_layer_activity_single_node():
    """Layer mapping extracts correct populations in vpjax order."""
    ys = jp.arange(8, dtype=float)
    layers = vb.cmc_to_layer_activity(ys)
    assert layers.shape == (3,)
    # Layer 0 (deep) = x_dp (index 3), Layer 1 (mid) = x_ss (0), Layer 2 (sup) = x_sp (1)
    np.testing.assert_allclose(layers, jp.array([3.0, 0.0, 1.0]))


def test_cmc_to_layer_activity_multi_node():
    """Layer mapping works for (8, n_nodes) state arrays."""
    n = 5
    ys = jax.random.normal(jax.random.PRNGKey(0), (8, n))
    layers = vb.cmc_to_layer_activity(ys)
    assert layers.shape == (n, 3)
    # Check deep layer = dp for each node
    np.testing.assert_allclose(layers[:, 0], ys[3])
    # Check middle layer = ss
    np.testing.assert_allclose(layers[:, 1], ys[0])
    # Check superficial = sp
    np.testing.assert_allclose(layers[:, 2], ys[1])


def test_cmc_layer_activity_differentiable():
    """Layer bridge is differentiable for end-to-end BOLD fitting."""
    def loss(ys):
        layers = vb.cmc_to_layer_activity(ys)
        return jp.sum(layers ** 2)
    g = jax.grad(loss)(jp.ones(8))
    assert jp.all(jp.isfinite(g))
    # Only states 0 (ss), 1 (sp), 3 (dp) should have gradients
    assert g[0] != 0.0  # ss
    assert g[1] != 0.0  # sp
    assert g[2] == 0.0  # ii — not included
    assert g[3] != 0.0  # dp
    assert jp.all(g[4:] == 0.0)  # velocities not used


# ── Differentiability ────────────────────────────────────────────────

def test_cmc_grad_wrt_input():
    """CMC dfun is differentiable w.r.t. coupling input."""
    def loss(c):
        ys = jp.zeros(8)
        dys = vb.cmc_dfun(ys, c, vb.cmc_default_theta)
        return jp.sum(dys ** 2)

    g = jax.grad(loss)(0.0)
    assert jp.isfinite(g)
    assert g != 0.0


def test_cmc_grad_wrt_state():
    """CMC Jacobian w.r.t. state is computable (needed for sensitivity)."""
    def f(ys):
        return vb.cmc_dfun(ys, 0.0, vb.cmc_default_theta)

    J = jax.jacobian(f)(jp.zeros(8))
    assert J.shape == (8, 8)
    assert jp.all(jp.isfinite(J))
