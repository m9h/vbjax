"""Tests for next-generation neural mass models in vbjax.

Covers three models:
- Coombes-Byrne (CB): exact mean-field of QIF, 4D, single population
- Coombes-Byrne E-I (CBEI): 8D excitatory-inhibitory extension
- Robinson-Rennie-Wright (RRW): 8D corticothalamic with 4 populations

Validates shape correctness, resting stability, driven dynamics,
network coupling, observation functions, differentiability, and
model-specific properties (r_positive, E-I interaction, thalamic loop).
"""

import numpy as np
import jax
import jax.numpy as jp
import vbjax as vb


# ====================================================================
# Coombes-Byrne single population (4D)
# ====================================================================

# -- Shape and basic sanity -----------------------------------------------

def test_cb_dfun_shape_single_node():
    """Single-node CB returns (4,) derivative."""
    ys = jp.zeros(4)
    c = 0.0
    dys = vb.cb_dfun(ys, c, vb.cb_default_theta)
    assert dys.shape == (4,)


def test_cb_dfun_shape_multi_node():
    """Multi-node CB returns (4, n_nodes) derivative."""
    n = 10
    ys = jp.zeros((4, n))
    c = jp.zeros(n)
    dys = vb.cb_dfun(ys, c, vb.cb_default_theta)
    assert dys.shape == (4, n)


def test_cb_default_theta_fields():
    """CBTheta has 7 fields: tau_m, Delta, eta, tau_s, V_syn, kappa_s, I."""
    p = vb.cb_default_theta
    assert len(p) == 7
    assert p.tau_m == 10.0
    assert p.Delta == 1.0
    assert p.eta == -5.0


def test_cb_default_state_fields():
    """CBState has 4 fields: r, V, g, z."""
    s = vb.cb_default_state
    assert len(s) == 4
    assert hasattr(s, 'r')
    assert hasattr(s, 'V')
    assert hasattr(s, 'g')
    assert hasattr(s, 'z')


# -- Resting stability ----------------------------------------------------

def test_cb_resting_stability():
    """Integrate CB at rest (no input) -- state should remain bounded."""
    dt = 0.1  # ms
    p = vb.cb_default_theta._replace(I=0.0)
    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.cb_dfun(y, 0.0, p),
        gfun=0.0,
    )
    y0 = jp.array(list(vb.cb_default_state))
    # 20000 steps * 0.1 ms = 2 seconds
    zs = jp.zeros((20000, 4))
    ys = loop(y0, zs, p)
    assert jp.all(jp.isfinite(ys))
    assert jp.all(jp.abs(ys) < 100.0)


# -- Driven dynamics ------------------------------------------------------

def test_cb_responds_to_input():
    """Constant external drive I produces nonzero rate and voltage."""
    dt = 0.1
    p = vb.cb_default_theta._replace(I=10.0)
    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.cb_dfun(y, 0.0, p),
        gfun=0.0,
    )
    y0 = jp.array(list(vb.cb_default_state))
    zs = jp.zeros((20000, 4))
    ys = loop(y0, zs, p)
    # After transient, rate and voltage should differ from initial
    r_late = ys[15000:, 0]
    V_late = ys[15000:, 1]
    assert jp.std(r_late) > 1e-8 or jp.abs(jp.mean(r_late) - 0.01) > 1e-4
    assert jp.std(V_late) > 1e-8 or jp.abs(jp.mean(V_late) + 2.0) > 1e-4


def test_cb_responds_to_coupling():
    """Nonzero coupling input c produces different dynamics than c=0."""
    ys = jp.array(list(vb.cb_default_state))
    p = vb.cb_default_theta
    dys_no_c = vb.cb_dfun(ys, 0.0, p)
    dys_with_c = vb.cb_dfun(ys, 5.0, p)
    # The V equation (index 1) receives c additively, so dV should differ
    assert dys_no_c[1] != dys_with_c[1]
    # r equation (index 0) does not directly receive c, so dr unchanged
    np.testing.assert_allclose(dys_no_c[0], dys_with_c[0], rtol=1e-6)


# -- Network coupling via make_sde ----------------------------------------

def test_cb_net_dfun_shape():
    """cb_net_dfun produces correct (4, n_nodes) output."""
    n = 5
    SC = jp.ones((n, n)) * 0.1
    SC = SC.at[jp.diag_indices(n)].set(0.0)
    G = 1.0
    p = (SC, G, vb.cb_default_theta)
    ys = jp.zeros((4, n))
    dys = vb.cb_net_dfun(ys, p)
    assert dys.shape == (4, n)
    assert jp.all(jp.isfinite(dys))


def test_cb_network_sde():
    """CB network runs through make_sde without error."""
    n = 4
    dt = 0.1
    SC = jp.abs(jax.random.normal(jax.random.PRNGKey(0), (n, n)))
    SC = (SC + SC.T) / 2
    SC = SC.at[jp.diag_indices(n)].set(0.0)
    G = 0.5
    node_p = vb.cb_default_theta._replace(I=5.0)
    p = (SC, G, node_p)

    _, loop = vb.make_sde(dt=dt, dfun=vb.cb_net_dfun, gfun=1e-3)
    y0 = jp.tile(jp.array(list(vb.cb_default_state)), (n, 1)).T
    key = jax.random.PRNGKey(1)
    zs = jax.random.normal(key, (2000, 4, n))
    ys = loop(y0, zs, p)
    assert ys.shape == (2000, 4, n)
    assert jp.all(jp.isfinite(ys))


# -- r_positive adhoc ------------------------------------------------------

def test_cb_r_positive_enforces_nonnegative_rate():
    """cb_r_positive clamps negative firing rate r to zero."""
    y = jp.array([-0.5, -2.0, 0.1, 0.0])
    y_fixed = vb.cb_r_positive(y, None)
    assert y_fixed[0] >= 0.0
    # Other state variables should be unchanged
    np.testing.assert_allclose(y_fixed[1:], y[1:])


def test_cb_r_positive_preserves_positive_rate():
    """cb_r_positive does not alter an already positive rate."""
    y = jp.array([0.05, -2.0, 0.1, 0.0])
    y_fixed = vb.cb_r_positive(y, None)
    np.testing.assert_allclose(y_fixed, y)


def test_cb_r_positive_in_sde():
    """CB integration with r_positive adhoc keeps rate non-negative."""
    dt = 0.1
    p = vb.cb_default_theta._replace(I=0.0)
    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.cb_dfun(y, 0.0, p),
        gfun=1e-2,
        adhoc=vb.cb_r_positive,
    )
    y0 = jp.array(list(vb.cb_default_state))
    key = jax.random.PRNGKey(42)
    zs = jax.random.normal(key, (5000, 4))
    ys = loop(y0, zs, p)
    # All recorded firing rates should be non-negative
    assert jp.all(ys[:, 0] >= 0.0)


# -- Differentiability -----------------------------------------------------

def test_cb_grad_wrt_input():
    """CB dfun is differentiable w.r.t. coupling input c."""
    def loss(c):
        ys = jp.array(list(vb.cb_default_state))
        dys = vb.cb_dfun(ys, c, vb.cb_default_theta)
        return jp.sum(dys ** 2)

    g = jax.grad(loss)(0.0)
    assert jp.isfinite(g)
    assert g != 0.0


def test_cb_grad_wrt_state():
    """CB Jacobian w.r.t. state is computable."""
    def f(ys):
        return vb.cb_dfun(ys, 0.0, vb.cb_default_theta)

    J = jax.jacobian(f)(jp.array(list(vb.cb_default_state)))
    assert J.shape == (4, 4)
    assert jp.all(jp.isfinite(J))


# ====================================================================
# Coombes-Byrne E-I (8D)
# ====================================================================

# -- Shape and basic sanity -----------------------------------------------

def test_cbei_dfun_shape_single_node():
    """Single-node CBEI returns (8,) derivative."""
    ys = jp.zeros(8)
    c = 0.0
    dys = vb.cbei_dfun(ys, c, vb.cbei_default_theta)
    assert dys.shape == (8,)


def test_cbei_dfun_shape_multi_node():
    """Multi-node CBEI returns (8, n_nodes) derivative."""
    n = 10
    ys = jp.zeros((8, n))
    c = jp.zeros(n)
    dys = vb.cbei_dfun(ys, c, vb.cbei_default_theta)
    assert dys.shape == (8, n)


def test_cbei_default_theta_fields():
    """CBEITheta has 15 fields (2 tau_m, 2 Delta, 2 eta, 2 tau_s,
    2 V_syn, 4 kappa, 1 I)."""
    p = vb.cbei_default_theta
    assert len(p) == 15
    assert p.tau_m_e == 10.0
    assert p.V_syn_i == -80.0
    assert p.kappa_ee == 10.0


def test_cbei_default_state_fields():
    """CBEIState has 8 fields: r_e, V_e, g_e, z_e, r_i, V_i, g_i, z_i."""
    s = vb.cbei_default_state
    assert len(s) == 8
    assert hasattr(s, 'r_e')
    assert hasattr(s, 'r_i')
    assert hasattr(s, 'V_i')
    assert hasattr(s, 'z_i')


# -- Resting stability ----------------------------------------------------

def test_cbei_resting_stability():
    """Integrate CBEI at rest (no input) -- state should remain bounded."""
    dt = 0.1
    p = vb.cbei_default_theta._replace(I=0.0)
    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.cbei_dfun(y, 0.0, p),
        gfun=0.0,
    )
    y0 = jp.array(list(vb.cbei_default_state))
    zs = jp.zeros((20000, 8))
    ys = loop(y0, zs, p)
    assert jp.all(jp.isfinite(ys))
    assert jp.all(jp.abs(ys) < 200.0)


# -- Driven dynamics ------------------------------------------------------

def test_cbei_responds_to_input():
    """Constant external drive produces nonzero excitatory rate."""
    dt = 0.1
    p = vb.cbei_default_theta._replace(I=10.0)
    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.cbei_dfun(y, 0.0, p),
        gfun=0.0,
    )
    y0 = jp.array(list(vb.cbei_default_state))
    zs = jp.zeros((20000, 8))
    ys = loop(y0, zs, p)
    r_e_late = ys[15000:, 0]  # excitatory rate
    assert jp.std(r_e_late) > 1e-8 or jp.abs(jp.mean(r_e_late) - 0.01) > 1e-4


# -- E-I interaction (not independent) ------------------------------------

def test_cbei_populations_interact():
    """E and I populations should not be independent: changing kappa_ie
    (I->E coupling) should alter excitatory dynamics."""
    ys = jp.array(list(vb.cbei_default_state))

    p_coupled = vb.cbei_default_theta
    p_uncoupled = vb.cbei_default_theta._replace(kappa_ie=0.0, kappa_ei=0.0)

    dys_coupled = vb.cbei_dfun(ys, 0.0, p_coupled)
    dys_uncoupled = vb.cbei_dfun(ys, 0.0, p_uncoupled)

    # E population derivatives (indices 0-3) should differ when I->E
    # coupling is removed, because kappa_ie * g_i * (V_syn_i - V_e)
    # enters the V_e equation.
    # At default state g_i=0 so the immediate derivative is the same;
    # integrate briefly to see the effect.
    dt = 0.1
    _, loop_c = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.cbei_dfun(y, 0.0, p),
        gfun=0.0,
    )
    _, loop_u = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.cbei_dfun(y, 0.0, p),
        gfun=0.0,
    )

    y0 = jp.array(list(vb.cbei_default_state))
    zs = jp.zeros((10000, 8))

    ys_c = loop_c(y0, zs, p_coupled)
    ys_u = loop_u(y0, zs, p_uncoupled)

    # After integration, the E population rate should differ between
    # the coupled and uncoupled cases
    diff_r_e = jp.abs(ys_c[-1, 0] - ys_u[-1, 0])
    diff_V_e = jp.abs(ys_c[-1, 1] - ys_u[-1, 1])
    assert diff_r_e > 1e-6 or diff_V_e > 1e-6, (
        "E-I coupling should produce different E dynamics")


def test_cbei_inhibition_reduces_excitatory_rate():
    """Stronger inhibitory coupling (kappa_ie) should reduce or alter
    the excitatory rate compared to no inhibition."""
    dt = 0.1
    p_no_inh = vb.cbei_default_theta._replace(I=10.0, kappa_ie=0.0)
    p_strong_inh = vb.cbei_default_theta._replace(I=10.0, kappa_ie=20.0)

    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.cbei_dfun(y, 0.0, p),
        gfun=0.0,
    )
    y0 = jp.array(list(vb.cbei_default_state))
    zs = jp.zeros((20000, 8))

    ys_no_inh = loop(y0, zs, p_no_inh)
    ys_strong_inh = loop(y0, zs, p_strong_inh)

    # The final states should differ
    diff = jp.max(jp.abs(ys_no_inh[-1] - ys_strong_inh[-1]))
    assert diff > 1e-4, "Inhibition should change dynamics"


# -- Network coupling via make_sde ----------------------------------------

def test_cbei_net_dfun_shape():
    """cbei_net_dfun produces correct (8, n_nodes) output."""
    n = 5
    SC = jp.ones((n, n)) * 0.1
    SC = SC.at[jp.diag_indices(n)].set(0.0)
    G = 1.0
    p = (SC, G, vb.cbei_default_theta)
    ys = jp.zeros((8, n))
    dys = vb.cbei_net_dfun(ys, p)
    assert dys.shape == (8, n)
    assert jp.all(jp.isfinite(dys))


def test_cbei_network_sde():
    """CBEI network runs through make_sde without error."""
    n = 4
    dt = 0.1
    SC = jp.abs(jax.random.normal(jax.random.PRNGKey(0), (n, n)))
    SC = (SC + SC.T) / 2
    SC = SC.at[jp.diag_indices(n)].set(0.0)
    G = 0.5
    node_p = vb.cbei_default_theta._replace(I=5.0)
    p = (SC, G, node_p)

    _, loop = vb.make_sde(dt=dt, dfun=vb.cbei_net_dfun, gfun=1e-3)
    y0 = jp.tile(jp.array(list(vb.cbei_default_state)), (n, 1)).T
    key = jax.random.PRNGKey(1)
    zs = jax.random.normal(key, (2000, 8, n))
    ys = loop(y0, zs, p)
    assert ys.shape == (2000, 8, n)
    assert jp.all(jp.isfinite(ys))


# -- Observation functions -------------------------------------------------

def test_cbei_observe_r():
    """cbei_observe_r extracts index 0 (excitatory rate)."""
    ys = jp.arange(8, dtype=float)
    assert vb.cbei_observe_r(ys) == 0.0


def test_cbei_observe_V():
    """cbei_observe_V extracts index 1 (excitatory membrane potential)."""
    ys = jp.arange(8, dtype=float)
    assert vb.cbei_observe_V(ys) == 1.0


def test_cbei_observe_r_multi_node():
    """cbei_observe_r works for (8, n_nodes) state arrays."""
    n = 5
    ys = jax.random.normal(jax.random.PRNGKey(0), (8, n))
    obs = vb.cbei_observe_r(ys)
    np.testing.assert_allclose(obs, ys[0])


def test_cbei_observe_V_multi_node():
    """cbei_observe_V works for (8, n_nodes) state arrays."""
    n = 5
    ys = jax.random.normal(jax.random.PRNGKey(0), (8, n))
    obs = vb.cbei_observe_V(ys)
    np.testing.assert_allclose(obs, ys[1])


# -- Differentiability -----------------------------------------------------

def test_cbei_grad_wrt_input():
    """CBEI dfun is differentiable w.r.t. coupling input c."""
    def loss(c):
        ys = jp.array(list(vb.cbei_default_state))
        dys = vb.cbei_dfun(ys, c, vb.cbei_default_theta)
        return jp.sum(dys ** 2)

    g = jax.grad(loss)(0.0)
    assert jp.isfinite(g)
    assert g != 0.0


def test_cbei_grad_wrt_state():
    """CBEI Jacobian w.r.t. state is computable."""
    def f(ys):
        return vb.cbei_dfun(ys, 0.0, vb.cbei_default_theta)

    J = jax.jacobian(f)(jp.array(list(vb.cbei_default_state)))
    assert J.shape == (8, 8)
    assert jp.all(jp.isfinite(J))


# ====================================================================
# Robinson-Rennie-Wright corticothalamic (8D)
# ====================================================================

# -- Shape and basic sanity -----------------------------------------------

def test_rrw_dfun_shape_single_node():
    """Single-node RRW returns (8,) derivative."""
    ys = jp.zeros(8)
    c = 0.0
    dys = vb.rrw_dfun(ys, c, vb.rrw_default_theta)
    assert dys.shape == (8,)


def test_rrw_dfun_shape_multi_node():
    """Multi-node RRW returns (8, n_nodes) derivative."""
    n = 10
    ys = jp.zeros((8, n))
    c = jp.zeros(n)
    dys = vb.rrw_dfun(ys, c, vb.rrw_default_theta)
    assert dys.shape == (8, n)


def test_rrw_default_theta_fields():
    """RRWTheta has 16 fields spanning cortical/thalamic parameters."""
    p = vb.rrw_default_theta
    assert len(p) == 16
    assert p.Q_max == 250.0
    assert p.theta == 15.0
    assert p.t0 == 85.0
    assert p.alpha == 50.0
    assert p.beta == 200.0


def test_rrw_default_state_fields():
    """RRWState has 8 fields: phi_e, dphi_e, V_e, V_i, V_s, V_r, dV_s, dV_r."""
    s = vb.rrw_default_state
    assert len(s) == 8
    assert hasattr(s, 'phi_e')
    assert hasattr(s, 'V_s')
    assert hasattr(s, 'V_r')
    assert hasattr(s, 'dV_r')


# -- Resting stability ----------------------------------------------------

def test_rrw_resting_stability():
    """Integrate RRW at rest (no input) -- state should remain bounded."""
    dt = 0.1
    p = vb.rrw_default_theta._replace(I=0.0)
    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.rrw_dfun(y, 0.0, p),
        gfun=0.0,
    )
    y0 = jp.array(list(vb.rrw_default_state))
    # 20000 steps * 0.1 ms = 2 seconds
    zs = jp.zeros((20000, 8))
    ys = loop(y0, zs, p)
    assert jp.all(jp.isfinite(ys))
    assert jp.all(jp.abs(ys) < 500.0)


# -- Driven dynamics ------------------------------------------------------

def test_rrw_responds_to_input():
    """Constant external drive I produces nonzero cortical field."""
    dt = 0.1
    p = vb.rrw_default_theta._replace(I=5.0)
    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.rrw_dfun(y, 0.0, p),
        gfun=0.0,
    )
    y0 = jp.array(list(vb.rrw_default_state))
    zs = jp.zeros((20000, 8))
    ys = loop(y0, zs, p)
    phi_e_late = ys[15000:, 0]  # cortical field
    # phi_e should have changed from its initial value
    assert jp.std(phi_e_late) > 1e-8 or jp.abs(jp.mean(phi_e_late) - 5.0) > 1e-4


def test_rrw_responds_to_coupling():
    """Nonzero coupling c enters relay neuron and alters dynamics."""
    ys = jp.array(list(vb.rrw_default_state))
    p = vb.rrw_default_theta
    dys_no_c = vb.rrw_dfun(ys, 0.0, p)
    dys_with_c = vb.rrw_dfun(ys, 5.0, p)
    # Coupling enters relay neuron (V_s pathway via dV_s, index 6)
    assert dys_no_c[6] != dys_with_c[6], (
        "Coupling should affect relay neuron derivative")


# -- Corticothalamic populations respond to cortical drive -----------------

def test_rrw_thalamic_response_to_cortical_drive():
    """Thalamic relay (V_s) and reticular (V_r) populations should
    respond to cortical drive through nu_se and nu_re couplings."""
    dt = 0.1
    p_driven = vb.rrw_default_theta._replace(I=5.0)
    p_rest = vb.rrw_default_theta._replace(I=0.0)

    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.rrw_dfun(y, 0.0, p),
        gfun=0.0,
    )
    y0 = jp.array(list(vb.rrw_default_state))
    zs = jp.zeros((20000, 8))

    ys_driven = loop(y0, zs, p_driven)
    ys_rest = loop(y0, zs, p_rest)

    # V_s (index 4) and V_r (index 5) should differ between conditions
    diff_Vs = jp.max(jp.abs(ys_driven[10000:, 4] - ys_rest[10000:, 4]))
    diff_Vr = jp.max(jp.abs(ys_driven[10000:, 5] - ys_rest[10000:, 5]))
    assert diff_Vs > 1e-6, "Relay nucleus should respond to cortical drive"
    assert diff_Vr > 1e-6, "Reticular nucleus should respond to cortical drive"


def test_rrw_thalamic_severed():
    """With nu_se=0 and nu_re=0, thalamic populations should decouple
    from cortex."""
    dt = 0.1
    p_coupled = vb.rrw_default_theta._replace(I=5.0)
    p_severed = vb.rrw_default_theta._replace(I=5.0, nu_se=0.0, nu_re=0.0)

    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.rrw_dfun(y, 0.0, p),
        gfun=0.0,
    )
    y0 = jp.array(list(vb.rrw_default_state))
    zs = jp.zeros((20000, 8))

    ys_coupled = loop(y0, zs, p_coupled)
    ys_severed = loop(y0, zs, p_severed)

    # V_s dynamics should differ (coupled case receives cortical output)
    diff_Vs = jp.max(jp.abs(ys_coupled[10000:, 4] - ys_severed[10000:, 4]))
    assert diff_Vs > 1e-6, (
        "Severing corticothalamic connections should change relay dynamics")


# -- Network coupling via make_sde ----------------------------------------

def test_rrw_net_dfun_shape():
    """rrw_net_dfun produces correct (8, n_nodes) output."""
    n = 5
    SC = jp.ones((n, n)) * 0.1
    SC = SC.at[jp.diag_indices(n)].set(0.0)
    G = 1.0
    p = (SC, G, vb.rrw_default_theta)
    ys = jp.zeros((8, n))
    dys = vb.rrw_net_dfun(ys, p)
    assert dys.shape == (8, n)
    assert jp.all(jp.isfinite(dys))


def test_rrw_network_sde():
    """RRW network runs through make_sde without error."""
    n = 4
    dt = 0.1
    SC = jp.abs(jax.random.normal(jax.random.PRNGKey(0), (n, n)))
    SC = (SC + SC.T) / 2
    SC = SC.at[jp.diag_indices(n)].set(0.0)
    G = 0.5
    node_p = vb.rrw_default_theta._replace(I=2.0)
    p = (SC, G, node_p)

    _, loop = vb.make_sde(dt=dt, dfun=vb.rrw_net_dfun, gfun=1e-3)
    y0 = jp.tile(jp.array(list(vb.rrw_default_state)), (n, 1)).T
    key = jax.random.PRNGKey(1)
    zs = jax.random.normal(key, (2000, 8, n))
    ys = loop(y0, zs, p)
    assert ys.shape == (2000, 8, n)
    assert jp.all(jp.isfinite(ys))


# -- Observation function --------------------------------------------------

def test_rrw_observe_phi():
    """rrw_observe_phi extracts index 0 (cortical excitatory field)."""
    ys = jp.arange(8, dtype=float)
    assert vb.rrw_observe_phi(ys) == 0.0


def test_rrw_observe_phi_multi_node():
    """rrw_observe_phi works for (8, n_nodes) state arrays."""
    n = 5
    ys = jax.random.normal(jax.random.PRNGKey(0), (8, n))
    obs = vb.rrw_observe_phi(ys)
    np.testing.assert_allclose(obs, ys[0])


# -- Differentiability -----------------------------------------------------

def test_rrw_grad_wrt_input():
    """RRW dfun is differentiable w.r.t. coupling input c."""
    def loss(c):
        ys = jp.array(list(vb.rrw_default_state))
        dys = vb.rrw_dfun(ys, c, vb.rrw_default_theta)
        return jp.sum(dys ** 2)

    g = jax.grad(loss)(0.0)
    assert jp.isfinite(g)
    assert g != 0.0


def test_rrw_grad_wrt_state():
    """RRW Jacobian w.r.t. state is computable."""
    def f(ys):
        return vb.rrw_dfun(ys, 0.0, vb.rrw_default_theta)

    J = jax.jacobian(f)(jp.array(list(vb.rrw_default_state)))
    assert J.shape == (8, 8)
    assert jp.all(jp.isfinite(J))


def test_rrw_grad_wrt_theta_param():
    """RRW dfun is differentiable w.r.t. a parameter (e.g. nu_ee)."""
    def loss(nu_ee):
        ys = jp.array(list(vb.rrw_default_state))
        p = vb.rrw_default_theta._replace(nu_ee=nu_ee)
        dys = vb.rrw_dfun(ys, 0.0, p)
        return jp.sum(dys ** 2)

    g = jax.grad(loss)(1.0)
    assert jp.isfinite(g)
    assert g != 0.0
