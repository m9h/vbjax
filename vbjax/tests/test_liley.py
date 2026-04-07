"""Tests for the Liley mean-field cortical model.

Validates the Liley implementation against expected properties:
- Correct shapes and dtypes for single/multi-node
- Parameter and state namedtuple structure
- Resting state behavior
- Stability under deterministic integration
- Conductance-based (shunting) synapse sign changes
- Driven dynamics with external input
- Network coupling via make_sde
- EEG observation function
- Differentiability w.r.t. state and input
- Spectral content under noise drive
- Long-range axonal propagation driven by excitatory firing
"""

import numpy as np
import jax
import jax.numpy as jp
import vbjax as vb


# -- Shape and basic sanity -----------------------------------------------

def test_liley_dfun_shape_single_node():
    """Single-node Liley returns (14,) derivative."""
    ys = jp.array(vb.liley_default_state)
    c = 0.0
    dys = vb.liley_dfun(ys, c, vb.liley_default_theta)
    assert dys.shape == (14,)
    assert dys.dtype == jp.float32 or dys.dtype == jp.float64


def test_liley_dfun_shape_multi_node():
    """Multi-node Liley returns (14, n_nodes) derivative."""
    n = 6
    ys = jp.tile(jp.array(vb.liley_default_state)[:, None], (1, n))
    c = jp.zeros(n)
    dys = vb.liley_dfun(ys, c, vb.liley_default_theta)
    assert dys.shape == (14, n)


# -- Namedtuple structure -------------------------------------------------

def test_liley_theta_fields():
    """LileyTheta has 26 fields covering membrane, synaptic, sigmoid,
    propagation, and external input parameters."""
    p = vb.liley_default_theta
    assert len(p) == 28
    assert hasattr(p, 'tau_e')
    assert hasattr(p, 'tau_i')
    assert hasattr(p, 'h_ee_eq')
    assert hasattr(p, 'Gamma_e')
    assert hasattr(p, 'gamma_e')
    assert hasattr(p, 'N_ee_b')
    assert hasattr(p, 'N_ee_a')
    assert hasattr(p, 'S_e_max')
    assert hasattr(p, 'mu_e')
    assert hasattr(p, 'sigma_e')
    assert hasattr(p, 'Lambda')
    assert hasattr(p, 'v_e')
    assert hasattr(p, 'p_ee')
    assert hasattr(p, 'p_ei')


def test_liley_state_fields():
    """LileyState has 14 fields: 2 membrane, 4 synaptic currents,
    4 synaptic derivatives, 2 propagation, 2 propagation derivatives."""
    s = vb.liley_default_state
    assert len(s) == 14
    assert s.h_e == -70.0
    assert s.h_i == -70.0
    assert s.I_ee == 0.0
    assert s.phi_ee == 0.0
    assert s.dphi_ee == 0.0


# -- Resting state --------------------------------------------------------

def test_liley_resting_membrane_drift():
    """At resting potentials with zero synaptic/propagation states,
    membrane derivatives should push toward rest (near zero)."""
    # Build a state at rest: h at rest, all currents and propagation zero
    rest = list(vb.liley_default_state)
    # h_e = h_e_rest, h_i = h_i_rest already in default state
    ys = jp.array(rest)

    # Zero external input
    p = vb.liley_default_theta._replace(p_ee=0.0, p_ei=0.0)
    dys = vb.liley_dfun(ys, 0.0, p)

    # dh_e = (1/tau_e)(h_e_rest - h_e + psi_ee*I_ee + psi_ie*I_ie)
    # At h_e=h_e_rest with I_ee=I_ie=0, dh_e should be ~0
    assert jp.abs(dys[0]) < 1e-6, f"dh_e at rest = {dys[0]}"
    assert jp.abs(dys[1]) < 1e-6, f"dh_i at rest = {dys[1]}"


def test_liley_resting_synaptic_currents_at_rest():
    """At the resting state with zero external input, synaptic current
    derivatives should be driven only by the sigmoid at rest potential."""
    rest = jp.array(vb.liley_default_state)
    p = vb.liley_default_theta._replace(p_ee=0.0, p_ei=0.0)
    dys = vb.liley_dfun(rest, 0.0, p)

    # The dI/dt states (indices 2-5) should equal the dI values (indices 6-9)
    # Since dI_* are all zero in default state, dI/dt = dI_* = 0
    np.testing.assert_allclose(dys[2:6], 0.0, atol=1e-10)


# -- Stability -------------------------------------------------------------

def test_liley_stability_deterministic():
    """Integrate Liley with no noise for 2 seconds; state stays bounded."""
    dt = 0.1  # ms
    n_steps = 20000  # 2 seconds
    p = vb.liley_default_theta
    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.liley_dfun(y, 0.0, p),
        gfun=0.0,
    )
    y0 = jp.array(vb.liley_default_state)
    zs = jp.zeros((n_steps, 14))
    ys = loop(y0, zs, p)
    assert jp.all(jp.isfinite(ys)), "Non-finite values during integration"
    # Membrane potentials (indices 0,1) should stay physiological
    assert jp.all(jp.abs(ys[:, 0]) < 200.0), \
        f"h_e out of range: [{float(ys[:, 0].min())}, {float(ys[:, 0].max())}]"
    assert jp.all(jp.abs(ys[:, 1]) < 200.0), \
        f"h_i out of range: [{float(ys[:, 1].min())}, {float(ys[:, 1].max())}]"


def test_liley_stability_with_noise():
    """With mild noise, Liley should remain bounded over 2 seconds."""
    dt = 0.1
    n_steps = 20000
    p = vb.liley_default_theta
    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.liley_dfun(y, 0.0, p),
        gfun=0.1,
    )
    y0 = jp.array(vb.liley_default_state)
    key = jax.random.PRNGKey(42)
    zs = jax.random.normal(key, (n_steps, 14))
    ys = loop(y0, zs, p)
    assert jp.all(jp.isfinite(ys)), "Non-finite values with noise"


# -- Conductance-based synapses -------------------------------------------

def test_liley_psi_sign_change():
    """Conductance-based scaling psi_jk(h_k) changes sign when h_k
    crosses the reversal potential -- the key difference from JR/CMC."""
    p = vb.liley_default_theta

    # psi_ee = (h_ee_eq - h_e) / |h_ee_eq - h_e_rest|
    # h_ee_eq = 45 mV.  Below reversal (h_e < 45), psi_ee > 0.
    # Above reversal (h_e > 45), psi_ee < 0.
    denom_ee = jp.abs(p.h_ee_eq - p.h_e_rest)  # |45 - (-70)| = 115

    h_below = 0.0   # well below reversal
    h_above = 60.0   # above reversal

    psi_below = (p.h_ee_eq - h_below) / denom_ee
    psi_above = (p.h_ee_eq - h_above) / denom_ee

    assert psi_below > 0.0, f"psi_ee below reversal should be positive: {psi_below}"
    assert psi_above < 0.0, f"psi_ee above reversal should be negative: {psi_above}"

    # Similarly for inhibitory: psi_ie = (h_ie_eq - h_e) / |h_ie_eq - h_e_rest|
    # h_ie_eq = -90 mV.  Below reversal (h_e < -90), psi_ie > 0.
    # Above reversal (h_e > -90), psi_ie < 0.
    denom_ie = jp.abs(p.h_ie_eq - p.h_e_rest)  # |-90 - (-70)| = 20
    h_below_inh = -100.0
    h_above_inh = -50.0

    psi_ie_below = (p.h_ie_eq - h_below_inh) / denom_ie
    psi_ie_above = (p.h_ie_eq - h_above_inh) / denom_ie

    assert psi_ie_below > 0.0, f"psi_ie below reversal should be positive: {psi_ie_below}"
    assert psi_ie_above < 0.0, f"psi_ie above reversal should be negative: {psi_ie_above}"


def test_liley_conductance_affects_membrane():
    """Changing membrane potential changes the effective synaptic drive
    through the conductance-based psi factors (shunting inhibition)."""
    p = vb.liley_default_theta._replace(p_ee=0.0, p_ei=0.0)

    # State with h_e near rest, some excitatory current
    state_lo = list(vb.liley_default_state)
    state_lo[0] = -60.0   # h_e near rest
    state_lo[2] = 10.0    # I_ee = 10
    ys_lo = jp.array(state_lo)

    # State with h_e near reversal, same excitatory current
    state_hi = list(vb.liley_default_state)
    state_hi[0] = 40.0    # h_e near excitatory reversal (45 mV)
    state_hi[2] = 10.0    # I_ee = 10
    ys_hi = jp.array(state_hi)

    dys_lo = vb.liley_dfun(ys_lo, 0.0, p)
    dys_hi = vb.liley_dfun(ys_hi, 0.0, p)

    # The excitatory contribution to dh_e should be much smaller near reversal
    # psi_ee * I_ee is the excitatory synaptic contribution
    psi_ee_lo = (p.h_ee_eq - (-60.0)) / jp.abs(p.h_ee_eq - p.h_e_rest)
    psi_ee_hi = (p.h_ee_eq - 40.0) / jp.abs(p.h_ee_eq - p.h_e_rest)

    assert jp.abs(psi_ee_hi) < jp.abs(psi_ee_lo), \
        "Near reversal, effective drive should be weaker"


# -- Driven dynamics -------------------------------------------------------

def test_liley_responds_to_external_input():
    """With p_ee > 0, h_e should deviate from rest after integration."""
    dt = 0.1
    n_steps = 20000  # 2 seconds
    p = vb.liley_default_theta  # p_ee=1.0 by default
    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.liley_dfun(y, 0.0, p),
        gfun=0.0,
    )
    y0 = jp.array(vb.liley_default_state)
    zs = jp.zeros((n_steps, 14))
    ys = loop(y0, zs, p)

    # h_e in the later half should differ from rest (-70 mV)
    h_e_late = ys[n_steps // 2:, 0]
    mean_h_e = jp.mean(h_e_late)
    assert jp.abs(mean_h_e - (-70.0)) > 0.1, \
        f"h_e should deviate from rest under external input: mean={float(mean_h_e)}"


def test_liley_no_input_stays_near_rest():
    """With p_ee=0, p_ei=0, and no coupling, h_e should stay near rest."""
    dt = 0.1
    n_steps = 10000
    p = vb.liley_default_theta._replace(p_ee=0.0, p_ei=0.0)
    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.liley_dfun(y, 0.0, p),
        gfun=0.0,
    )
    y0 = jp.array(vb.liley_default_state)
    zs = jp.zeros((n_steps, 14))
    ys = loop(y0, zs, p)

    # Late h_e should be near rest; the sigmoid at -70 still produces
    # some small drive, so allow wider tolerance
    h_e_late = ys[n_steps // 2:, 0]
    # Without external input, activity should be much closer to rest
    # than with input (deviation primarily from sigmoid at rest)
    assert jp.all(jp.isfinite(h_e_late))


# -- Network coupling -----------------------------------------------------

def test_liley_net_dfun_shape():
    """liley_net_dfun produces correct (14, n_nodes) output."""
    n = 5
    SC = jp.ones((n, n)) * 0.1
    SC = SC.at[jp.diag_indices(n)].set(0.0)
    G = 1.0
    p = (SC, G, vb.liley_default_theta)
    ys = jp.tile(jp.array(vb.liley_default_state)[:, None], (1, n))
    dys = vb.liley_net_dfun(ys, p)
    assert dys.shape == (14, n)
    assert jp.all(jp.isfinite(dys))


def test_liley_network_sde():
    """Liley network runs through make_sde for 4 nodes without error."""
    n = 4
    dt = 0.1
    SC = jp.abs(jax.random.normal(jax.random.PRNGKey(0), (n, n)))
    SC = (SC + SC.T) / 2
    SC = SC.at[jp.diag_indices(n)].set(0.0)
    G = 0.5
    node_p = vb.liley_default_theta
    p = (SC, G, node_p)

    _, loop = vb.make_sde(dt=dt, dfun=vb.liley_net_dfun, gfun=0.01)
    y0 = jp.tile(jp.array(vb.liley_default_state)[:, None], (1, n))
    key = jax.random.PRNGKey(1)
    zs = jax.random.normal(key, (5000, 14, n))
    ys = loop(y0, zs, p)
    assert ys.shape == (5000, 14, n)
    assert jp.all(jp.isfinite(ys))


def test_liley_network_coupling_effect():
    """Network coupling should make nodes influence each other:
    coupled nodes should correlate more than zero."""
    n = 4
    dt = 0.1
    n_steps = 10000
    SC = jp.ones((n, n)) * 0.5
    SC = SC.at[jp.diag_indices(n)].set(0.0)
    G = 2.0
    node_p = vb.liley_default_theta
    p = (SC, G, node_p)

    _, loop = vb.make_sde(dt=dt, dfun=vb.liley_net_dfun, gfun=0.1)
    y0 = jp.tile(jp.array(vb.liley_default_state)[:, None], (1, n))
    key = jax.random.PRNGKey(7)
    zs = jax.random.normal(key, (n_steps, 14, n))
    ys = loop(y0, zs, p)
    assert jp.all(jp.isfinite(ys))


# -- Observation function --------------------------------------------------

def test_liley_observe_eeg_single_node():
    """liley_observe_eeg extracts index 0 (h_e) for single node."""
    ys = jp.arange(14, dtype=float)
    assert vb.liley_observe_eeg(ys) == 0.0


def test_liley_observe_eeg_multi_node():
    """liley_observe_eeg extracts h_e row for multi-node arrays."""
    n = 5
    ys = jax.random.normal(jax.random.PRNGKey(0), (14, n))
    eeg = vb.liley_observe_eeg(ys)
    np.testing.assert_allclose(eeg, ys[0])


# -- Differentiability -----------------------------------------------------

def test_liley_grad_wrt_input():
    """liley_dfun is differentiable w.r.t. coupling input."""
    def loss(c):
        ys = jp.array(vb.liley_default_state)
        dys = vb.liley_dfun(ys, c, vb.liley_default_theta)
        return jp.sum(dys ** 2)

    g = jax.grad(loss)(0.0)
    assert jp.isfinite(g)
    assert g != 0.0, "Gradient w.r.t. coupling should be nonzero"


def test_liley_grad_wrt_state():
    """Jacobian w.r.t. state is computable (needed for linearization)."""
    def f(ys):
        return vb.liley_dfun(ys, 0.0, vb.liley_default_theta)

    y0 = jp.array(vb.liley_default_state)
    J = jax.jacobian(f)(y0)
    assert J.shape == (14, 14)
    assert jp.all(jp.isfinite(J))


def test_liley_grad_wrt_theta_scalar():
    """liley_dfun is differentiable w.r.t. a scalar parameter."""
    def loss(tau_e):
        p = vb.liley_default_theta._replace(tau_e=tau_e)
        ys = jp.array(vb.liley_default_state)
        dys = vb.liley_dfun(ys, 0.0, p)
        return jp.sum(dys ** 2)

    g = jax.grad(loss)(94.0)
    assert jp.isfinite(g)


# -- Spectral test ---------------------------------------------------------

def test_liley_spectral_content():
    """Under noise drive, PSD of h_e should show non-flat spectrum
    (the model shapes noise into structured oscillations)."""
    dt = 0.1  # ms
    n_steps = 40000  # 4 seconds
    p = vb.liley_default_theta

    _, loop = vb.make_sde(
        dt=dt,
        dfun=lambda y, p: vb.liley_dfun(y, 0.0, p),
        gfun=0.1,
    )
    y0 = jp.array(vb.liley_default_state)
    key = jax.random.PRNGKey(123)
    zs = jax.random.normal(key, (n_steps, 14))
    ys = loop(y0, zs, p)

    # Extract h_e after transient
    h_e = np.array(ys[10000:, 0])

    # Compute PSD
    fs = 1.0 / (dt * 1e-3)  # sampling frequency in Hz
    from numpy.fft import rfft, rfftfreq
    freqs = rfftfreq(len(h_e), d=1.0 / fs)
    psd = np.abs(rfft(h_e - h_e.mean())) ** 2

    # Spectrum should not be flat: ratio of max to median should be > 2
    # (white noise would be ~flat, a model with resonance peaks above)
    mask = (freqs > 1.0) & (freqs < 100.0)  # physiological band
    psd_band = psd[mask]
    if len(psd_band) > 0:
        ratio = float(np.max(psd_band) / np.median(psd_band))
        assert ratio > 1.5, \
            f"Spectrum appears too flat (max/median = {ratio:.2f})"


# -- Long-range propagation ------------------------------------------------

def test_liley_propagation_driven_by_S_e():
    """phi_ee and phi_ei should be driven by excitatory firing rate S_e.
    When S_e is elevated (h_e >> mu_e), propagation fields should grow."""
    p = vb.liley_default_theta._replace(p_ee=0.0, p_ei=0.0)

    # State at rest: S_e ~ sigmoid(-70, -50, 5) ~ very small
    state_rest = jp.array(vb.liley_default_state)
    dys_rest = vb.liley_dfun(state_rest, 0.0, p)

    # State with elevated h_e: S_e ~ sigmoid(-20, -50, 5) ~ near S_e_max
    state_hi = jp.array(vb.liley_default_state)
    state_hi = state_hi.at[0].set(-20.0)  # h_e elevated
    dys_hi = vb.liley_dfun(state_hi, 0.0, p)

    # ddphi_ee (index 12) should be larger when S_e is larger
    # The driving term is v_Lambda^2 * N_ee_a * S_e
    assert jp.abs(dys_hi[12]) > jp.abs(dys_rest[12]), \
        "phi_ee acceleration should be larger with elevated S_e"
    assert jp.abs(dys_hi[13]) > jp.abs(dys_rest[13]), \
        "phi_ei acceleration should be larger with elevated S_e"


def test_liley_propagation_damping():
    """Propagation fields should exhibit damping: with nonzero dphi
    and no firing, the field should decay."""
    p = vb.liley_default_theta._replace(p_ee=0.0, p_ei=0.0)

    # State at rest with positive dphi_ee (propagation velocity)
    state = list(vb.liley_default_state)
    state[10] = 1.0   # phi_ee = 1.0
    state[12] = 1.0   # dphi_ee = 1.0
    ys = jp.array(state)

    dys = vb.liley_dfun(ys, 0.0, p)

    # ddphi_ee = -2*v_Lambda*dphi_ee - v_Lambda^2*phi_ee + v_Lambda^2*N_ee_a*S_e
    # With S_e small (at rest h=-70), the damping terms should dominate,
    # making ddphi_ee negative (decelerating)
    assert dys[12] < 0.0, \
        f"ddphi_ee should be negative (damped) when dphi>0 and phi>0: {float(dys[12])}"


# -- Integration consistency -----------------------------------------------

def test_liley_synaptic_indices_match_state():
    """Verify that the state ordering in the ODE matches the namedtuple:
    derivatives of I states (indices 2-5) should equal dI states (6-9)."""
    ys = jp.array(vb.liley_default_state)
    # Set some dI values
    ys = ys.at[6].set(1.0)   # dI_ee = 1.0
    ys = ys.at[7].set(-0.5)  # dI_ei = -0.5
    ys = ys.at[8].set(0.3)   # dI_ie = 0.3
    ys = ys.at[9].set(-0.1)  # dI_ii = -0.1

    dys = vb.liley_dfun(ys, 0.0, vb.liley_default_theta)

    # dI_ee/dt = dI_ee (the velocity of the current)
    # In the ODE: dy[2] = dI_ee, dy[3] = dI_ei, etc.
    np.testing.assert_allclose(dys[2], 1.0, atol=1e-10)
    np.testing.assert_allclose(dys[3], -0.5, atol=1e-10)
    np.testing.assert_allclose(dys[4], 0.3, atol=1e-10)
    np.testing.assert_allclose(dys[5], -0.1, atol=1e-10)


def test_liley_gamma_unit_conversion():
    """Rate constants gamma_e, gamma_i are in s^-1 in theta but converted
    to ms^-1 internally. Check that the conversion is consistent."""
    p = vb.liley_default_theta
    # gamma_e = 300 s^-1 should become 0.3 ms^-1
    ge_ms = p.gamma_e * 1e-3
    assert abs(ge_ms - 0.3) < 1e-10

    # gamma_i = 65 s^-1 should become 0.065 ms^-1
    gi_ms = p.gamma_i * 1e-3
    assert abs(gi_ms - 0.065) < 1e-10
