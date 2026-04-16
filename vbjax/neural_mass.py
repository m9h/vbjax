"""Neural mass models for virtual brain simulations.

Implements several neural mass models commonly used in computational
neuroscience, including the Jansen-Rit (JR), Montbrio-Pazo-Roxin (MPR),
Balloon-Windkessel BOLD, DCM, dopamine-modulated QIF, and the Canonical
Microcircuit (CMC).  All models expose a ``dfun(state, coupling, params)``
interface compatible with :func:`vbjax.make_sde` and :func:`vbjax.make_ode`.
"""

import collections
from typing import Tuple, Union

import jax.numpy as np


JRTheta = collections.namedtuple(
    typename='JRTheta',
    field_names='A B a b v0 nu_max r J a_1 a_2 a_3 a_4 mu I'.split(' '))

jr_default_theta = JRTheta(
    A=3.25, B=22.0, a=0.1, b=0.05, v0=5.52, nu_max=0.0025, 
    r=0.56, J=135.0, a_1=1.0, a_2=0.8, a_3=0.25, a_4=0.25, mu=0.22, I=0.0)

JRState = collections.namedtuple(
    typename='JRState',
    field_names='y0 y1 y2 y3 y4 y5'.split(' '))
    
def jr_dfun(ys, c, p):
    y0, y1, y2, y3, y4, y5 = ys

    sigm_y1_y2 = 2.0 * p.nu_max / (1.0 + np.exp(p.r * (p.v0 - (y1 - y2))))
    sigm_y0_1  = 2.0 * p.nu_max / (1.0 + np.exp(p.r * (p.v0 - (p.a_1 * p.J * y0))))
    sigm_y0_3  = 2.0 * p.nu_max / (1.0 + np.exp(p.r * (p.v0 - (p.a_3 * p.J * y0))))

    return np.array([y3,
        y4,
        y5,
        p.A * p.a * sigm_y1_y2 - 2.0 * p.a * y3 - p.a ** 2 * y0,
        p.A * p.a * (p.mu + p.a_2 * p.J * sigm_y0_1 + c)
            - 2.0 * p.a * y4 - p.a ** 2 * y1,
        p.B * p.b * (p.a_4 * p.J * sigm_y0_3) - 2.0 * p.b * y5 - p.b ** 2 * y2,
                     ])


BVEPTheta = collections.namedtuple(
    typename='BVEPTheta',
    field_names='tau0 I1 x0'
)

bvep_default_theta = BVEPTheta(
    tau0=10.0, I1=3.1, x0=-3.5
)

def bvep_dfun(ys, c, p: BVEPTheta):
    x, z = ys
    x2 = x*x
    dx = 1 - x*x2 - 2*x2 - z + p.I1
    dz = (1/p.tau0)*(4*(x - p.x0) - z - c)
    return np.array([dx, dz])


# Montbrio-Pazo-Roxin
MPRTheta = collections.namedtuple(
    typename='MPRTheta',
    field_names='tau I Delta J eta cr cv'.split(' '))

mpr_default_theta = MPRTheta(
    tau=1.0,
    I=0.0,
    Delta=1.0,
    J=15.0,
    eta=-5.0,
    cr=1.0,
    cv=0.0
)

MPRState = collections.namedtuple(
    typename='MPRState',
    field_names='r V'.split(' '))

mpr_default_state = MPRState(r=0.0, V=-2.0)

def mpr_dfun(ys, c, p):
    r, V = ys

    # bound rate to be positive
    r = r * (r > 0)

    I_c = p.cr * c[0] + p.cv * c[1]

    return np.array([
        (1 / p.tau) * (p.Delta / (np.pi * p.tau) + 2 * r * V),
        (1 / p.tau) * (V ** 2 + p.eta + p.J * p.tau *
         r + p.I + I_c - (np.pi ** 2) * (r ** 2) * (p.tau ** 2))
    ])

def mpr_r_positive(rv, _):
    r, v = rv
    return np.array([ r*(r>0), v ])


BOLDTheta = collections.namedtuple(
    typename='BOLDTheta',
    field_names='tau_s,tau_f,tau_o,alpha,te,v0,e0,epsilon,nu_0,'
                'r_0,recip_tau_s,recip_tau_f,recip_tau_o,recip_alpha,'
                'recip_e0,k1,k2,k3'
)

def compute_bold_theta(
        tau_s=0.65,
        tau_f=0.41,
        tau_o=0.98,
        alpha=0.32,
        te=0.04,
        v0=4.0,
        e0=0.4,
        epsilon=0.5,
        nu_0=40.3,
        r_0=25.0,
    ):
    recip_tau_s = 1.0 / tau_s
    recip_tau_f = 1.0 / tau_f
    recip_tau_o = 1.0 / tau_o
    recip_alpha = 1.0 / alpha
    recip_e0 = 1.0 / e0
    k1 = 4.3 * nu_0 * e0 * te
    k2 = epsilon * r_0 * e0 * te
    k3 = 1.0 - epsilon
    return BOLDTheta(**locals())

bold_default_theta = compute_bold_theta()

def bold_dfun(sfvq, x, p: BOLDTheta):
    s, f, v, q = sfvq
    ds = x - p.recip_tau_s * s - p.recip_tau_f * (f - 1)
    df = s
    dv = p.recip_tau_o * (f - v ** p.recip_alpha)
    dq = p.recip_tau_o * (f * (1 - (1 - p.e0) ** (1 / f)) * p.recip_e0
                          - v ** p.recip_alpha * (q / v))
    return np.array([ds, df, dv, dq])


DCMTheta = collections.namedtuple(typename='DCMTheta',
                                  field_names='A,B,C')

def dcm_dfun(x, u, p: DCMTheta):
    """Implements the classical bilinear DCM
    \\dot{x} = (A + sum_j u_j B_j ) x + C u
    """
    return (p.A + np.sum(u * p.B,axis=-1)) @ x + p.C @ u


# TODO other models
# TODO codim3 https://gist.github.com/maedoc/01cea5cad9c833c56349392ee7d9b627


DopaTheta = collections.namedtuple(
    typename='dopaTheta',
    field_names='a, b, c, ga, gg, Eta, Delta, Iext, Ea, Eg, Sja, Sjg, tauSa, tauSg, alpha, beta, ud, k, Vmax, Km, Bd, Ad, tau_Dp, wi, we, wd, sigma')


DopaState = collections.namedtuple(
    typename='DopaState',
    field_names='r V u Sa Sg Dp')

dopa_default_theta = DopaTheta(
    a=0.04, b=5., c=140., ga=12., gg=12.,
    Delta=1., Eta=18., Iext=0., Ea=0., Eg=-80., tauSa=5., tauSg=5., Sja=0.8, Sjg=1.2,
    ud=12., alpha=0.013, beta=.4, k=10e4, Vmax=1300., Km=150., Bd=0.2, Ad=1., tau_Dp=500.,
    wi=1.e-4, we=1.e-4, wd=1.e-4, sigma=1e-3,
    )

dopa_default_initial_state = DopaState(
    r=0.03, V=-67.0, u=0.0, Sa=0.0, Sg=0.0, Dp=0.5)

def dopa_dfun(y, cy, p: DopaTheta):
    "Adaptive QIF model with dopamine modulation."

    r, V, u, Sa, Sg, Dp = y
    c_inh, c_exc, c_dopa = cy
    a, b, c, ga, gg, Eta, Delta, Iext, Ea, Eg, Sja, Sjg, tauSa, tauSg, alpha, beta, ud, k, Vmax, Km, Bd, Ad, tau_Dp, *_ = p

    dr = 2. * a * r * V + b * r - ga * Sa * r - gg * Sg * r + (a * Delta) / np.pi
    dV = a * V**2 + b * V + c + Eta - (np.pi**2 * r**2) / a + (Ad * Dp + Bd) * ga * Sa * (Ea - V) + gg * Sg * (Eg - V) + Iext - u
    du = alpha * (beta * V - u) + ud * r
    dSa = -Sa / tauSa + Sja * c_exc
    dSg = -Sg / tauSg + Sjg * c_inh
    dDp = (k * c_dopa - Vmax * Dp / (Km + Dp)) / tau_Dp
    
    return np.array([dr, dV, du, dSa, dSg, dDp])

def dopa_net_dfun(y, p):
    "Canonical form for network of dopa nodes."
    Ci, Ce, Cd, node_params = p
    r = y[0]
    c_inh = node_params.wi * Ci @ r
    c_exc = node_params.we * Ce @ r
    c_dopa = node_params.wd * Cd @ r
    return dopa_dfun(y, (c_inh, c_exc, c_dopa), node_params)

def dopa_r_positive(y, _):
    # same as mpr but keep separate name for now
    y = y.at[0].set( np.where(y[0]<0, 0, y[0]) )
    return y

def dopa_gfun_mulr(y, p):
    "Provide a multiplicative r, additive V gfun."
    r, *_ = y
    g = np.r_[r, 1, 0, 0, 0, 0] * p.sigma
    if r.ndim == 2:
        g = g.reshape((-1, 1))
    return g

def dopa_gfun_add(y, p):
    "Provides an additive noise gfun."
    return p.sigma


# Canonical Microcircuit (Bastos et al. 2012, Douglas 2025)
#
# 4 populations arranged by cortical layer:
#   ss  - spiny stellate cells        (granular, layer IV)
#   sp  - superficial pyramidal cells (supragranular, layers II/III)
#   ii  - inhibitory interneurons     (all layers)
#   dp  - deep pyramidal cells        (infragranular, layers V/VI)
#
# 8 state variables (2nd-order ODE per population → 1st-order system):
#   [x_ss, x_sp, x_ii, x_dp, v_ss, v_sp, v_ii, v_dp]
#   where x = mean membrane potential, v = dx/dt
#
# Intrinsic connectivity (Bastos et al. 2012, Fig. 3):
#   ss → sp  (excitatory feedforward)
#   sp → ii  (excitatory, drives inhibition)
#   sp → dp  (excitatory, descending)
#   dp → ii  (excitatory, drives inhibition)
#   dp → sp  (excitatory, ascending feedback)
#   ii → ss  (inhibitory)
#   ii → sp  (inhibitory)
#   ii → dp  (inhibitory)
#
# Inter-regional coupling enters via:
#   Forward:  ss (granular layer, thalamic/feedforward target)
#   Backward: sp + dp (supragranular + infragranular targets)
#
# Each excitatory population (ss, sp, dp) uses the excitatory PSP kernel
# (He, a), and the inhibitory population (ii) uses the inhibitory kernel
# (Hi, b).  Same sigmoid as Jansen-Rit for direct comparability.
#
# References:
#   Bastos AM et al. (2012) Canonical microcircuits for predictive coding.
#       Neuron 76(4):695-711.
#   Douglas PK (2025) Computing with canonical microcircuits.
#       arXiv:2508.06501.
#   Moran RJ et al. (2013) Neural masses and fields in dynamic causal
#       modeling. Frontiers in Computational Neuroscience 7:57.

CMCTheta = collections.namedtuple(
    typename='CMCTheta',
    field_names='He Hi a b r v0 nu_max '
                'g_ss_sp g_sp_ii g_sp_dp g_dp_ii g_dp_sp '
                'g_ii_ss g_ii_sp g_ii_dp '
                'I'.split(' '))
"""Parameters for the Canonical Microcircuit (CMC) neural mass model.

Fields
------
He : float
    Excitatory post-synaptic potential amplitude (mV).
Hi : float
    Inhibitory post-synaptic potential amplitude (mV).
a : float
    Excitatory rate constant (ms^-1).
b : float
    Inhibitory rate constant (ms^-1).
r : float
    Sigmoid steepness (mV^-1).
v0 : float
    Sigmoid midpoint (mV).
nu_max : float
    Maximum firing rate (kHz).
g_ss_sp : float
    Excitatory gain, spiny stellate to superficial pyramidal.
g_sp_ii : float
    Excitatory gain, superficial pyramidal to inhibitory interneurons.
g_sp_dp : float
    Excitatory gain, superficial pyramidal to deep pyramidal.
g_dp_ii : float
    Excitatory gain, deep pyramidal to inhibitory interneurons.
g_dp_sp : float
    Excitatory gain, deep pyramidal to superficial pyramidal.
g_ii_ss : float
    Inhibitory gain, interneurons to spiny stellate.
g_ii_sp : float
    Inhibitory gain, interneurons to superficial pyramidal.
g_ii_dp : float
    Inhibitory gain, interneurons to deep pyramidal.
I : float
    External (thalamic) drive current.

References
----------
Bastos AM et al. (2012) Canonical microcircuits for predictive coding.
    Neuron 76(4):695-711.
"""

cmc_default_theta = CMCTheta(
    He=3.25,        # excitatory PSP amplitude (mV), same as JR A
    Hi=22.0,        # inhibitory PSP amplitude (mV), same as JR B
    a=0.1,          # excitatory rate constant (ms^-1), same as JR a
    b=0.05,         # inhibitory rate constant (ms^-1), same as JR b
    r=0.56,         # sigmoid steepness (mV^-1)
    v0=6.0,         # sigmoid midpoint (mV)
    nu_max=0.0025,  # max firing rate (kHz)
    # excitatory intrinsic connections (tuned via DE for alpha oscillations)
    g_ss_sp=86.3,   # ss -> sp, feedforward
    g_sp_ii=23.8,   # sp -> ii
    g_sp_dp=188.0,  # sp -> dp, descending
    g_dp_ii=68.1,   # dp -> ii
    g_dp_sp=125.1,  # dp -> sp, ascending feedback
    # inhibitory intrinsic connections
    g_ii_ss=120.3,  # ii -> ss
    g_ii_sp=101.4,  # ii -> sp
    g_ii_dp=158.5,  # ii -> dp
    # external drive
    I=362.5,
)
"""Default CMC parameters tuned via differential evolution for alpha
oscillations.  Connectivity gains were optimized to produce a spectral
peak in the 8--13 Hz (alpha) band with stable bounded dynamics.  See
``examples/cmc_tune_defaults.py`` for the optimization procedure.
"""

CMCState = collections.namedtuple(
    typename='CMCState',
    field_names='x_ss x_sp x_ii x_dp v_ss v_sp v_ii v_dp'.split(' '))
"""State vector for the CMC neural mass model.

The 8 state variables arise from a 2nd-order ODE per population,
converted to a 1st-order system: ``[x, v]`` where ``x`` is the mean
membrane potential and ``v = dx/dt``.

Fields
------
x_ss : float
    Mean membrane potential of spiny stellate cells (layer IV).
x_sp : float
    Mean membrane potential of superficial pyramidal cells (layers II/III).
x_ii : float
    Mean membrane potential of inhibitory interneurons.
x_dp : float
    Mean membrane potential of deep pyramidal cells (layers V/VI).
v_ss, v_sp, v_ii, v_dp : float
    Corresponding time derivatives (velocities).
"""

cmc_default_state = CMCState(
    x_ss=0.0, x_sp=0.0, x_ii=0.0, x_dp=0.0,
    v_ss=0.0, v_sp=0.0, v_ii=0.0, v_dp=0.0)
"""Default initial state for the CMC model (all zeros / resting)."""


def cmc_dfun(ys, c, p: CMCTheta) -> np.ndarray:
    """Canonical microcircuit dynamics (Bastos et al. 2012).

    Parameters
    ----------
    ys : array, shape (8,) or (8, n_nodes)
        State vector [x_ss, x_sp, x_ii, x_dp, v_ss, v_sp, v_ii, v_dp].
    c : array
        Coupling input (enters spiny stellate / granular layer).
    p : CMCTheta
        Model parameters.

    Returns
    -------
    dys : array, same shape as ys
        State derivatives.
    """
    x_ss, x_sp, x_ii, x_dp, v_ss, v_sp, v_ii, v_dp = ys

    # Sigmoid (identical to Jansen-Rit for direct comparison)
    sigm = lambda x: 2.0 * p.nu_max / (1.0 + np.exp(p.r * (p.v0 - x)))

    s_ss = sigm(x_ss)
    s_sp = sigm(x_sp)
    s_ii = sigm(x_ii)
    s_dp = sigm(x_dp)

    # Synaptic input currents per population
    I_ss = -p.g_ii_ss * s_ii + c + p.I
    I_sp = p.g_ss_sp * s_ss - p.g_ii_sp * s_ii + p.g_dp_sp * s_dp
    I_ii = p.g_sp_ii * s_sp + p.g_dp_ii * s_dp
    I_dp = p.g_sp_dp * s_sp - p.g_ii_dp * s_ii

    a2 = p.a ** 2
    b2 = p.b ** 2

    # 2nd-order PSP kernels as 1st-order system:
    #   dv/dt = H*κ*I - 2κ*v - κ²*x    (damped harmonic oscillator)
    return np.array([
        v_ss,
        v_sp,
        v_ii,
        v_dp,
        p.He * p.a * I_ss - 2.0 * p.a * v_ss - a2 * x_ss,
        p.He * p.a * I_sp - 2.0 * p.a * v_sp - a2 * x_sp,
        p.Hi * p.b * I_ii - 2.0 * p.b * v_ii - b2 * x_ii,
        p.He * p.a * I_dp - 2.0 * p.a * v_dp - a2 * x_dp,
    ])


def cmc_net_dfun(ys, p: Tuple) -> np.ndarray:
    """Network form: computes linear coupling from superficial pyramidal
    activity and calls cmc_dfun.  Compatible with vbjax.make_sde.

    Parameters
    ----------
    ys : array, shape (8, n_nodes)
        State matrix.
    p : tuple (SC, G, node_theta)
        SC : (n, n) structural connectivity matrix
        G  : float, global coupling strength
        node_theta : CMCTheta, per-node parameters

    Returns
    -------
    dys : array, shape (8, n_nodes)
    """
    SC, G, node_p = p
    x_sp = ys[1]  # superficial pyramidal = long-range output
    c = G * (SC @ x_sp)
    return cmc_dfun(ys, c, node_p)


def cmc_hier_dfun(ys, c_fwd, c_bwd, p: CMCTheta) -> np.ndarray:
    """CMC with separate forward and backward inter-regional coupling.

    Implements the hierarchical predictive coding architecture of
    Bastos et al. (2012):
    - Forward connections target ss (granular layer IV)
    - Backward connections target sp + dp (agranular layers)

    Parameters
    ----------
    ys : array, shape (8,) or (8, n_nodes)
        State vector.
    c_fwd : array
        Forward coupling input (enters ss / granular layer).
    c_bwd : array
        Backward coupling input (enters sp + dp / agranular layers).
    p : CMCTheta
        Model parameters.

    Returns
    -------
    dys : array, same shape as ys
    """
    x_ss, x_sp, x_ii, x_dp, v_ss, v_sp, v_ii, v_dp = ys

    sigm = lambda x: 2.0 * p.nu_max / (1.0 + np.exp(p.r * (p.v0 - x)))

    s_ss = sigm(x_ss)
    s_sp = sigm(x_sp)
    s_ii = sigm(x_ii)
    s_dp = sigm(x_dp)

    I_ss = -p.g_ii_ss * s_ii + c_fwd + p.I
    I_sp = p.g_ss_sp * s_ss - p.g_ii_sp * s_ii + p.g_dp_sp * s_dp + c_bwd
    I_ii = p.g_sp_ii * s_sp + p.g_dp_ii * s_dp
    I_dp = p.g_sp_dp * s_sp - p.g_ii_dp * s_ii + c_bwd

    a2 = p.a ** 2
    b2 = p.b ** 2

    return np.array([
        v_ss,
        v_sp,
        v_ii,
        v_dp,
        p.He * p.a * I_ss - 2.0 * p.a * v_ss - a2 * x_ss,
        p.He * p.a * I_sp - 2.0 * p.a * v_sp - a2 * x_sp,
        p.Hi * p.b * I_ii - 2.0 * p.b * v_ii - b2 * x_ii,
        p.He * p.a * I_dp - 2.0 * p.a * v_dp - a2 * x_dp,
    ])


def cmc_hier_2node_dfun(ys, p: Tuple) -> np.ndarray:
    """Two-node hierarchical CMC for predictive coding experiments.

    Node 0 = lower area (e.g. V1), Node 1 = higher area (e.g. V4).
    Forward: sp of lower → ss of higher (prediction errors ascend).
    Backward: dp of higher → sp+dp of lower (predictions descend).

    Compatible with vbjax.make_sde: signature is dfun(state, params).

    Parameters
    ----------
    ys : array, shape (8, 2)
        Column 0 = lower node, column 1 = higher node.
    p : tuple (G_fwd, G_bwd, node_theta)
        G_fwd : float, forward coupling gain
        G_bwd : float, backward coupling gain
        node_theta : CMCTheta, intrinsic parameters (shared)

    Returns
    -------
    dys : array, shape (8, 2)
    """
    G_fwd, G_bwd, node_p = p

    sp_lower = ys[1, 0]
    dp_higher = ys[3, 1]

    c_fwd = np.array([0.0, G_fwd * sp_lower])
    c_bwd = np.array([G_bwd * dp_higher, 0.0])

    return cmc_hier_dfun(ys, c_fwd, c_bwd, node_p)


def cmc_hier_Nnode_dfun(ys, p: Tuple) -> np.ndarray:
    """N-node hierarchical CMC with forward/backward connectivity.

    Generalizes the 2-node case to arbitrary hierarchies defined
    by separate forward and backward structural connectivity matrices.

    Compatible with vbjax.make_sde.

    Parameters
    ----------
    ys : array, shape (8, n_nodes)
        State matrix.
    p : tuple (SC_fwd, SC_bwd, G_fwd, G_bwd, node_theta)
        SC_fwd : (n, n) forward connectivity (source sp → target ss)
        SC_bwd : (n, n) backward connectivity (source dp → target sp+dp)
        G_fwd  : float, forward coupling gain
        G_bwd  : float, backward coupling gain
        node_theta : CMCTheta

    Returns
    -------
    dys : array, shape (8, n_nodes)
    """
    SC_fwd, SC_bwd, G_fwd, G_bwd, node_p = p

    x_sp = ys[1]  # forward source: superficial pyramidal
    x_dp = ys[3]  # backward source: deep pyramidal

    c_fwd = G_fwd * (SC_fwd @ x_sp)
    c_bwd = G_bwd * (SC_bwd @ x_dp)

    return cmc_hier_dfun(ys, c_fwd, c_bwd, node_p)


def cmc_to_layer_activity(ys) -> np.ndarray:
    """Map CMC state to 3-layer cortical activity for vpjax coupling.

    Provides principled per-layer neural activity, replacing the
    heuristic feedforward/feedback fractions in vpjax.layer_stimulus().
    Feed directly into per-layer Balloon-Windkessel models.

    Mapping (matches vpjax.LayerNVCParams layer ordering):
        Layer 0 (deep, V-VI)         ← x_dp (deep pyramidal)
        Layer 1 (middle, IV)         ← x_ss (spiny stellate)
        Layer 2 (superficial, I-III) ← x_sp (superficial pyramidal)

    Parameters
    ----------
    ys : array, shape (8,) or (8, n_nodes)
        CMC state vector.

    Returns
    -------
    layer_activity : array, shape (3,) or (n_nodes, 3)
        Per-layer neural activity for vpjax hemodynamic models.
    """
    x_ss = ys[0]
    x_sp = ys[1]
    x_dp = ys[3]
    if ys.ndim == 1:
        return np.array([x_dp, x_ss, x_sp])
    else:
        return np.stack([x_dp, x_ss, x_sp], axis=-1)


def cmc_observe_sp(ys) -> np.ndarray:
    """Return superficial pyramidal membrane potential (EEG/MEG-like).

    In predictive coding, superficial pyramidal cells encode prediction
    errors and are the primary generators of EEG/MEG signals measured at
    the scalp.
    """
    return ys[1]


def cmc_observe_dp(ys) -> np.ndarray:
    """Return deep pyramidal membrane potential (LFP/feedback-like).

    Deep pyramidal cells encode predictions and project to subcortical
    structures and lower cortical areas.
    """
    return ys[3]


# ====================================================================
# Coombes-Byrne single population (Byrne et al. 2017, Coombes & Byrne 2019)
#
# Exact mean-field reduction of a QIF neuron network with
# alpha-function conductance-based synapses and Lorentzian
# heterogeneity.
#
# 4 state variables:
#   r  - mean firing rate
#   V  - mean membrane potential
#   g  - synaptic conductance (alpha-function kernel)
#   z  - auxiliary variable for 2nd-order conductance dynamics
#
# Equations (Coombes & Byrne 2019, Eq. 7-10):
#   tau_m * dr/dt = Delta/pi + 2*r*V
#   tau_m * dV/dt = V^2 + eta + kappa_s*g*(V_syn - V) - (pi*tau_m*r)^2
#   tau_s * dg/dt = z
#   tau_s * dz/dt = r - 2*z - g
#
# References:
#   Byrne A, Avitabile D, Coombes S (2017) Next generation neural mass
#       models. Lecture Notes in Nonlinear Dynamics, Springer.
#   Coombes S & Byrne A (2019) Next-generation neural mass and field
#       modeling. J Neurophysiol 122:1275-1287.
#   Cakir Y et al. (2023) Comparison between an exact and a heuristic
#       neural mass model. Biol Cybern 117:79-98.
# ====================================================================

CBTheta = collections.namedtuple(
    typename='CBTheta',
    field_names='tau_m Delta eta tau_s V_syn kappa_s I'.split(' '))

cb_default_theta = CBTheta(
    tau_m=10.0,     # membrane time constant (ms)
    Delta=1.0,      # heterogeneity width (Lorentzian half-width, mV)
    eta=-5.0,       # mean external current (mV^2/ms), center of Lorentzian
    tau_s=5.0,      # synaptic time constant (ms), alpha-function
    V_syn=0.0,      # synaptic reversal potential (mV), 0 for excitatory
    kappa_s=1.0,    # synaptic coupling strength (dimensionless)
    I=0.0,          # external drive (mV^2/ms)
)

CBState = collections.namedtuple(
    typename='CBState',
    field_names='r V g z'.split(' '))

cb_default_state = CBState(r=0.01, V=-2.0, g=0.0, z=0.0)


def cb_dfun(ys, c, p):
    """Coombes-Byrne next-generation neural mass (4D).

    Parameters
    ----------
    ys : array, shape (4,) or (4, n_nodes)
        State vector [r, V, g, z].
    c : array
        External coupling input (enters V equation additively).
    p : CBTheta
        Model parameters.

    Returns
    -------
    dys : array, same shape as ys
    """
    r, V, g, z = ys
    r = r * (r > 0)

    pi_tau_r = np.pi * p.tau_m * r

    dr = (1.0 / p.tau_m) * (p.Delta / np.pi + 2.0 * r * V)
    dV = (1.0 / p.tau_m) * (V ** 2 + p.eta + p.I + c
                             + p.kappa_s * g * (p.V_syn - V)
                             - pi_tau_r ** 2)
    dg = z / p.tau_s
    dz = (1.0 / p.tau_s) * (r - 2.0 * z - g)

    return np.array([dr, dV, dg, dz])


def cb_net_dfun(ys, p):
    """Network form for Coombes-Byrne model. Compatible with make_sde.

    Parameters
    ----------
    ys : array, shape (4, n_nodes)
    p : tuple (SC, G, node_theta)
    """
    SC, G, node_p = p
    r = ys[0]
    c = G * (SC @ r)
    return cb_dfun(ys, c, node_p)


def cb_r_positive(y, _):
    """Enforce non-negative firing rate."""
    return y.at[0].set(np.where(y[0] < 0, 0, y[0]))


# ====================================================================
# Liley mean-field model (Liley, Cadusch & Dafilis 2002)
#
# Spatially homogeneous (neural mass) form of the Liley continuum
# model of electrocortical activity.
#
# 2 populations: Excitatory (e) and Inhibitory (i)
# Conductance-based synapses with reversal potentials.
#
# 14 first-order ODEs (from 2 membrane + 4 second-order synaptic
# + 2 second-order long-range axonal equations):
#
# Key feature: the conductance-based (shunting) synapse
#   psi_jk(h_k) = (h_j^eq - h_k) / |h_j^eq - h_k^rest|
#
# References:
#   Liley DTJ, Cadusch PJ, Dafilis MP (2002) A spatially continuous
#       mean field theory of electrocortical activity. Network 13:67-113.
#   Bojak I & Liley DTJ (2005) Modeling the effects of anesthesia on
#       the electroencephalogram. Phys Rev E 71:041902.
# ====================================================================

LileyTheta = collections.namedtuple(
    typename='LileyTheta',
    field_names=(
        'tau_e tau_i '
        'h_e_rest h_i_rest '
        'h_ee_eq h_ei_eq h_ie_eq h_ii_eq '
        'Gamma_e Gamma_i '
        'gamma_e gamma_i '
        'N_ee_b N_ei_b N_ie_b N_ii_b '
        'N_ee_a N_ei_a '
        'S_e_max S_i_max '
        'mu_e mu_i '
        'sigma_e sigma_i '
        'Lambda v_e '
        'p_ee p_ei '
    ).split())

liley_default_theta = LileyTheta(
    tau_e=94.0,      # excitatory membrane time constant (ms)
    tau_i=42.0,      # inhibitory membrane time constant (ms)
    h_e_rest=-70.0,  # excitatory resting potential (mV)
    h_i_rest=-70.0,  # inhibitory resting potential (mV)
    h_ee_eq=45.0,    # E→E reversal potential (mV)
    h_ei_eq=45.0,    # E→I reversal potential (mV)
    h_ie_eq=-90.0,   # I→E reversal potential (mV)
    h_ii_eq=-90.0,   # I→I reversal potential (mV)
    Gamma_e=0.3,     # peak EPSP amplitude (mV)
    Gamma_i=0.065,   # peak IPSP amplitude (mV)
    gamma_e=300.0,   # EPSP rate constant (s^-1 → 0.3 ms^-1)
    gamma_i=65.0,    # IPSP rate constant (s^-1 → 0.065 ms^-1)
    N_ee_b=3034.0,   # local E→E connections
    N_ei_b=3034.0,   # local E→I connections
    N_ie_b=536.0,    # local I→E connections
    N_ii_b=536.0,    # local I→I connections
    N_ee_a=4000.0,   # distant E→E connections (long-range)
    N_ei_a=2000.0,   # distant E→I connections (long-range)
    S_e_max=0.5,     # max excitatory firing rate (kHz)
    S_i_max=0.5,     # max inhibitory firing rate (kHz)
    mu_e=-50.0,      # excitatory sigmoid midpoint (mV)
    mu_i=-50.0,      # inhibitory sigmoid midpoint (mV)
    sigma_e=5.0,     # excitatory sigmoid width (mV)
    sigma_i=5.0,     # inhibitory sigmoid width (mV)
    Lambda=0.4,      # spatial decay rate (cm^-1)
    v_e=140.0,       # axonal conduction speed (cm/s)
    p_ee=1.0,        # external E→E input (kHz)
    p_ei=1.0,        # external E→I input (kHz)
)

LileyState = collections.namedtuple(
    typename='LileyState',
    field_names=(
        'h_e h_i '
        'I_ee I_ei I_ie I_ii '
        'dI_ee dI_ei dI_ie dI_ii '
        'phi_ee phi_ei dphi_ee dphi_ei'
    ).split())

liley_default_state = LileyState(
    h_e=-70.0, h_i=-70.0,
    I_ee=0.0, I_ei=0.0, I_ie=0.0, I_ii=0.0,
    dI_ee=0.0, dI_ei=0.0, dI_ie=0.0, dI_ii=0.0,
    phi_ee=0.0, phi_ei=0.0, dphi_ee=0.0, dphi_ei=0.0,
)


def liley_dfun(ys, c, p):
    """Liley mean-field cortical model (14D).

    Parameters
    ----------
    ys : array, shape (14,) or (14, n_nodes)
        State vector.
    c : array
        External coupling input (adds to phi_ee).
    p : LileyTheta
        Model parameters.

    Returns
    -------
    dys : array, same shape as ys
    """
    (h_e, h_i,
     I_ee, I_ei, I_ie, I_ii,
     dI_ee, dI_ei, dI_ie, dI_ii,
     phi_ee, phi_ei, dphi_ee, dphi_ei) = ys

    # Convert rate constants to ms^-1 for consistent time units
    ge = p.gamma_e * 1e-3  # s^-1 → ms^-1
    gi = p.gamma_i * 1e-3

    # Sigmoids (firing rate functions)
    S_e = p.S_e_max / (1.0 + np.exp(-2.0 * (h_e - p.mu_e) / p.sigma_e))
    S_i = p.S_i_max / (1.0 + np.exp(-2.0 * (h_i - p.mu_i) / p.sigma_i))

    # Conductance-based (shunting) scaling factors: psi_jk(h_k)
    psi_ee = (p.h_ee_eq - h_e) / np.abs(p.h_ee_eq - p.h_e_rest)
    psi_ei = (p.h_ei_eq - h_i) / np.abs(p.h_ei_eq - p.h_i_rest)
    psi_ie = (p.h_ie_eq - h_e) / np.abs(p.h_ie_eq - p.h_e_rest)
    psi_ii = (p.h_ii_eq - h_i) / np.abs(p.h_ii_eq - p.h_i_rest)

    # Membrane potential dynamics (conductance-based)
    dh_e = (1.0 / p.tau_e) * (p.h_e_rest - h_e + psi_ee * I_ee + psi_ie * I_ie)
    dh_i = (1.0 / p.tau_i) * (p.h_i_rest - h_i + psi_ei * I_ei + psi_ii * I_ii)

    # Axonal propagation rate (spatial decay * velocity)
    v_Lambda = p.v_e * p.Lambda * 1e-3  # cm/s * cm^-1 → ms^-1

    # Synaptic input dynamics (second-order alpha-function kernels)
    e_const = np.e  # Euler's number for alpha-function peak normalization

    ddI_ee = -2.0 * ge * dI_ee - ge**2 * I_ee + p.Gamma_e * ge * e_const * (
        p.N_ee_b * S_e + phi_ee + c + p.p_ee)
    ddI_ei = -2.0 * ge * dI_ei - ge**2 * I_ei + p.Gamma_e * ge * e_const * (
        p.N_ei_b * S_e + phi_ei + p.p_ei)
    ddI_ie = -2.0 * gi * dI_ie - gi**2 * I_ie + p.Gamma_i * gi * e_const * (
        p.N_ie_b * S_i)
    ddI_ii = -2.0 * gi * dI_ii - gi**2 * I_ii + p.Gamma_i * gi * e_const * (
        p.N_ii_b * S_i)

    # Long-range axonal propagation (damped, spatially homogeneous)
    ddphi_ee = (-2.0 * v_Lambda * dphi_ee - v_Lambda**2 * phi_ee
                + v_Lambda**2 * p.N_ee_a * S_e)
    ddphi_ei = (-2.0 * v_Lambda * dphi_ei - v_Lambda**2 * phi_ei
                + v_Lambda**2 * p.N_ei_a * S_e)

    return np.array([
        dh_e, dh_i,
        dI_ee, dI_ei, dI_ie, dI_ii,
        ddI_ee, ddI_ei, ddI_ie, ddI_ii,
        dphi_ee, dphi_ei, ddphi_ee, ddphi_ei,
    ])


def liley_net_dfun(ys, p):
    """Network form for Liley model. Compatible with make_sde.

    Parameters
    ----------
    ys : array, shape (14, n_nodes)
    p : tuple (SC, G, node_theta)
    """
    SC, G, node_p = p
    h_e = ys[0]
    S_e = node_p.S_e_max / (1.0 + np.exp(
        -2.0 * (h_e - node_p.mu_e) / node_p.sigma_e))
    c = G * (SC @ S_e)
    return liley_dfun(ys, c, node_p)


def liley_observe_eeg(ys):
    """Return h_e (excitatory membrane potential, EEG-like observable)."""
    return ys[0]


# ====================================================================
# Bojak-Liley pharmacological extension (Bojak & Liley 2005)
#
# GABAergic anaesthetics (propofol, isoflurane, etc.) modulate the
# inhibitory post-synaptic potential by:
#   1. Scaling peak IPSP amplitude: Gamma_i → Gamma_i * (1 + c_drug * rho_amp)
#   2. Slowing IPSP decay rate:     gamma_i → gamma_i / (1 + c_drug * rho_rate)
#
# Reference:
#   Bojak I & Liley DTJ (2005) Modeling the effects of anesthesia on
#       the electroencephalogram. Phys Rev E 71:041902.
# ====================================================================

LileyPharmaTheta = collections.namedtuple(
    typename='LileyPharmaTheta',
    field_names=(
        'tau_e tau_i '
        'h_e_rest h_i_rest '
        'h_ee_eq h_ei_eq h_ie_eq h_ii_eq '
        'Gamma_e Gamma_i '
        'gamma_e gamma_i '
        'N_ee_b N_ei_b N_ie_b N_ii_b '
        'N_ee_a N_ei_a '
        'S_e_max S_i_max '
        'mu_e mu_i '
        'sigma_e sigma_i '
        'Lambda v_e '
        'p_ee p_ei '
        'c_drug rho_amp rho_rate'
    ).split())

liley_pharma_default_theta = LileyPharmaTheta(
    tau_e=94.0,      # excitatory membrane time constant (ms)
    tau_i=42.0,      # inhibitory membrane time constant (ms)
    h_e_rest=-70.0,  # excitatory resting potential (mV)
    h_i_rest=-70.0,  # inhibitory resting potential (mV)
    h_ee_eq=45.0,    # E→E reversal potential (mV)
    h_ei_eq=45.0,    # E→I reversal potential (mV)
    h_ie_eq=-90.0,   # I→E reversal potential (mV)
    h_ii_eq=-90.0,   # I→I reversal potential (mV)
    Gamma_e=0.3,     # peak EPSP amplitude (mV)
    Gamma_i=0.065,   # peak IPSP amplitude (mV)
    gamma_e=300.0,   # EPSP rate constant (s^-1 → 0.3 ms^-1)
    gamma_i=65.0,    # IPSP rate constant (s^-1 → 0.065 ms^-1)
    N_ee_b=3034.0,   # local E→E connections
    N_ei_b=3034.0,   # local E→I connections
    N_ie_b=536.0,    # local I→E connections
    N_ii_b=536.0,    # local I→I connections
    N_ee_a=4000.0,   # distant E→E connections (long-range)
    N_ei_a=2000.0,   # distant E→I connections (long-range)
    S_e_max=0.5,     # max excitatory firing rate (kHz)
    S_i_max=0.5,     # max inhibitory firing rate (kHz)
    mu_e=-50.0,      # excitatory sigmoid midpoint (mV)
    mu_i=-50.0,      # inhibitory sigmoid midpoint (mV)
    sigma_e=5.0,     # excitatory sigmoid width (mV)
    sigma_i=5.0,     # inhibitory sigmoid width (mV)
    Lambda=0.4,      # spatial decay rate (cm^-1)
    v_e=140.0,       # axonal conduction speed (cm/s)
    p_ee=1.0,        # external E→E input (kHz)
    p_ei=1.0,        # external E→I input (kHz)
    c_drug=0.0,      # dimensionless drug concentration (0=none)
    rho_amp=0.0,     # amplitude scaling factor
    rho_rate=0.0,    # rate slowing factor
)

liley_pharma_propofol_theta = LileyPharmaTheta(
    tau_e=94.0, tau_i=42.0,
    h_e_rest=-70.0, h_i_rest=-70.0,
    h_ee_eq=45.0, h_ei_eq=45.0, h_ie_eq=-90.0, h_ii_eq=-90.0,
    Gamma_e=0.3, Gamma_i=0.065,
    gamma_e=300.0, gamma_i=65.0,
    N_ee_b=3034.0, N_ei_b=3034.0, N_ie_b=536.0, N_ii_b=536.0,
    N_ee_a=4000.0, N_ei_a=2000.0,
    S_e_max=0.5, S_i_max=0.5,
    mu_e=-50.0, mu_i=-50.0,
    sigma_e=5.0, sigma_i=5.0,
    Lambda=0.4, v_e=140.0,
    p_ee=1.0, p_ei=1.0,
    c_drug=1.0,      # clinical concentration
    rho_amp=1.5,     # propofol amplitude scaling
    rho_rate=1.2,    # propofol rate slowing
)

liley_pharma_isoflurane_theta = LileyPharmaTheta(
    tau_e=94.0, tau_i=42.0,
    h_e_rest=-70.0, h_i_rest=-70.0,
    h_ee_eq=45.0, h_ei_eq=45.0, h_ie_eq=-90.0, h_ii_eq=-90.0,
    Gamma_e=0.3, Gamma_i=0.065,
    gamma_e=300.0, gamma_i=65.0,
    N_ee_b=3034.0, N_ei_b=3034.0, N_ie_b=536.0, N_ii_b=536.0,
    N_ee_a=4000.0, N_ei_a=2000.0,
    S_e_max=0.5, S_i_max=0.5,
    mu_e=-50.0, mu_i=-50.0,
    sigma_e=5.0, sigma_i=5.0,
    Lambda=0.4, v_e=140.0,
    p_ee=1.0, p_ei=1.0,
    c_drug=1.0,      # clinical concentration
    rho_amp=1.0,     # isoflurane amplitude scaling
    rho_rate=0.8,    # isoflurane rate slowing
)


def liley_pharma_dfun(ys, c, p):
    """Liley mean-field model with Bojak-Liley pharmacological extension (14D).

    GABAergic anaesthetics modulate the inhibitory PSP by scaling the
    peak amplitude and slowing the decay rate.  When c_drug=0 this
    reduces exactly to the standard Liley model.

    Parameters
    ----------
    ys : array, shape (14,) or (14, n_nodes)
        State vector (same layout as liley_dfun).
    c : array
        External coupling input (adds to phi_ee).
    p : LileyPharmaTheta
        Model parameters including c_drug, rho_amp, rho_rate.

    Returns
    -------
    dys : array, same shape as ys
    """
    (h_e, h_i,
     I_ee, I_ei, I_ie, I_ii,
     dI_ee, dI_ei, dI_ie, dI_ii,
     phi_ee, phi_ei, dphi_ee, dphi_ei) = ys

    # Convert rate constants to ms^-1 for consistent time units
    ge = p.gamma_e * 1e-3  # s^-1 → ms^-1

    # Drug-modulated inhibitory parameters (Bojak & Liley 2005)
    Gamma_i_eff = p.Gamma_i * (1.0 + p.c_drug * p.rho_amp)
    gamma_i_eff = p.gamma_i / (1.0 + p.c_drug * p.rho_rate)
    gi = gamma_i_eff * 1e-3  # s^-1 → ms^-1

    # Sigmoids (firing rate functions)
    S_e = p.S_e_max / (1.0 + np.exp(-2.0 * (h_e - p.mu_e) / p.sigma_e))
    S_i = p.S_i_max / (1.0 + np.exp(-2.0 * (h_i - p.mu_i) / p.sigma_i))

    # Conductance-based (shunting) scaling factors: psi_jk(h_k)
    psi_ee = (p.h_ee_eq - h_e) / np.abs(p.h_ee_eq - p.h_e_rest)
    psi_ei = (p.h_ei_eq - h_i) / np.abs(p.h_ei_eq - p.h_i_rest)
    psi_ie = (p.h_ie_eq - h_e) / np.abs(p.h_ie_eq - p.h_e_rest)
    psi_ii = (p.h_ii_eq - h_i) / np.abs(p.h_ii_eq - p.h_i_rest)

    # Membrane potential dynamics (conductance-based)
    dh_e = (1.0 / p.tau_e) * (p.h_e_rest - h_e + psi_ee * I_ee + psi_ie * I_ie)
    dh_i = (1.0 / p.tau_i) * (p.h_i_rest - h_i + psi_ei * I_ei + psi_ii * I_ii)

    # Axonal propagation rate (spatial decay * velocity)
    v_Lambda = p.v_e * p.Lambda * 1e-3  # cm/s * cm^-1 → ms^-1

    # Synaptic input dynamics (second-order alpha-function kernels)
    e_const = np.e  # Euler's number for alpha-function peak normalization

    ddI_ee = -2.0 * ge * dI_ee - ge**2 * I_ee + p.Gamma_e * ge * e_const * (
        p.N_ee_b * S_e + phi_ee + c + p.p_ee)
    ddI_ei = -2.0 * ge * dI_ei - ge**2 * I_ei + p.Gamma_e * ge * e_const * (
        p.N_ei_b * S_e + phi_ei + p.p_ei)

    # Inhibitory kernels use drug-modulated Gamma_i_eff and gi
    ddI_ie = -2.0 * gi * dI_ie - gi**2 * I_ie + Gamma_i_eff * gi * e_const * (
        p.N_ie_b * S_i)
    ddI_ii = -2.0 * gi * dI_ii - gi**2 * I_ii + Gamma_i_eff * gi * e_const * (
        p.N_ii_b * S_i)

    # Long-range axonal propagation (damped, spatially homogeneous)
    ddphi_ee = (-2.0 * v_Lambda * dphi_ee - v_Lambda**2 * phi_ee
                + v_Lambda**2 * p.N_ee_a * S_e)
    ddphi_ei = (-2.0 * v_Lambda * dphi_ei - v_Lambda**2 * phi_ei
                + v_Lambda**2 * p.N_ei_a * S_e)

    return np.array([
        dh_e, dh_i,
        dI_ee, dI_ei, dI_ie, dI_ii,
        ddI_ee, ddI_ei, ddI_ie, ddI_ii,
        dphi_ee, dphi_ei, ddphi_ee, ddphi_ei,
    ])


def liley_pharma_net_dfun(ys, p):
    """Network form for Liley pharmacological model. Compatible with make_sde.

    Parameters
    ----------
    ys : array, shape (14, n_nodes)
    p : tuple (SC, G, node_theta)
    """
    SC, G, node_p = p
    h_e = ys[0]
    S_e = node_p.S_e_max / (1.0 + np.exp(
        -2.0 * (h_e - node_p.mu_e) / node_p.sigma_e))
    c = G * (SC @ S_e)
    return liley_pharma_dfun(ys, c, node_p)


def liley_adhoc(ys, *_):
    """Clamp Liley membrane potentials to physiological range.

    Use as ``adhoc`` argument in ``make_sde`` to prevent divergence
    at high pharmacological doses or strong coupling.
    """
    h_e = np.clip(ys[0], -90.0, 50.0)
    h_i = np.clip(ys[1], -90.0, 50.0)
    return ys.at[0].set(h_e).at[1].set(h_i)


# ====================================================================
# Coombes-Byrne E-I two-population model (8D)
#
# Exact mean-field of two interacting QIF populations (E and I) with
# alpha-function conductance-based synapses.
#
# 8 state variables:
#   r_e, V_e, g_e, z_e  - excitatory population
#   r_i, V_i, g_i, z_i  - inhibitory population
#
# References:
#   Coombes S & Byrne A (2019) Next-generation neural mass and field
#       modeling. J Neurophysiol 122:1275-1287.
# ====================================================================

CBEITheta = collections.namedtuple(
    typename='CBEITheta',
    field_names=(
        'tau_m_e tau_m_i Delta_e Delta_i eta_e eta_i '
        'tau_s_e tau_s_i '
        'V_syn_e V_syn_i '
        'kappa_ee kappa_ei kappa_ie kappa_ii '
        'I'
    ).split())

cbei_default_theta = CBEITheta(
    tau_m_e=10.0,    # E membrane time constant (ms)
    tau_m_i=10.0,    # I membrane time constant (ms)
    Delta_e=1.0,     # E heterogeneity width (mV)
    Delta_i=1.0,     # I heterogeneity width (mV)
    eta_e=-5.0,      # E mean drive (mV^2/ms)
    eta_i=-5.0,      # I mean drive (mV^2/ms)
    tau_s_e=5.0,     # E synaptic time constant (ms)
    tau_s_i=5.0,     # I synaptic time constant (ms)
    V_syn_e=0.0,     # excitatory reversal potential (mV)
    V_syn_i=-80.0,   # inhibitory reversal potential (mV)
    kappa_ee=10.0,   # E→E coupling strength
    kappa_ei=10.0,   # E→I coupling strength
    kappa_ie=10.0,   # I→E coupling strength
    kappa_ii=5.0,    # I→I coupling strength
    I=0.0,           # external drive to E (mV^2/ms)
)

CBEIState = collections.namedtuple(
    typename='CBEIState',
    field_names='r_e V_e g_e z_e r_i V_i g_i z_i'.split(' '))

cbei_default_state = CBEIState(
    r_e=0.01, V_e=-2.0, g_e=0.0, z_e=0.0,
    r_i=0.01, V_i=-2.0, g_i=0.0, z_i=0.0,
)


def cbei_dfun(ys, c, p):
    """Coombes-Byrne E-I next-generation neural mass (8D).

    Parameters
    ----------
    ys : array, shape (8,) or (8, n_nodes)
        State vector [r_e, V_e, g_e, z_e, r_i, V_i, g_i, z_i].
    c : array
        External coupling input (enters V_e additively).
    p : CBEITheta
        Model parameters.

    Returns
    -------
    dys : array, same shape as ys
    """
    r_e, V_e, g_e, z_e, r_i, V_i, g_i, z_i = ys
    r_e = r_e * (r_e > 0)
    r_i = r_i * (r_i > 0)

    pi_tau_e_r = np.pi * p.tau_m_e * r_e
    pi_tau_i_r = np.pi * p.tau_m_i * r_i

    # Excitatory population
    dr_e = (1.0 / p.tau_m_e) * (p.Delta_e / np.pi + 2.0 * r_e * V_e)
    dV_e = (1.0 / p.tau_m_e) * (
        V_e ** 2 + p.eta_e + p.I + c
        + p.kappa_ee * g_e * (p.V_syn_e - V_e)
        + p.kappa_ie * g_i * (p.V_syn_i - V_e)
        - pi_tau_e_r ** 2)
    dg_e = z_e / p.tau_s_e
    dz_e = (1.0 / p.tau_s_e) * (r_e - 2.0 * z_e - g_e)

    # Inhibitory population
    dr_i = (1.0 / p.tau_m_i) * (p.Delta_i / np.pi + 2.0 * r_i * V_i)
    dV_i = (1.0 / p.tau_m_i) * (
        V_i ** 2 + p.eta_i
        + p.kappa_ei * g_e * (p.V_syn_e - V_i)
        + p.kappa_ii * g_i * (p.V_syn_i - V_i)
        - pi_tau_i_r ** 2)
    dg_i = z_i / p.tau_s_i
    dz_i = (1.0 / p.tau_s_i) * (r_i - 2.0 * z_i - g_i)

    return np.array([dr_e, dV_e, dg_e, dz_e, dr_i, dV_i, dg_i, dz_i])


def cbei_net_dfun(ys, p):
    """Network form for Coombes-Byrne E-I model. Compatible with make_sde.

    Parameters
    ----------
    ys : array, shape (8, n_nodes)
    p : tuple (SC, G, node_theta)
    """
    SC, G, node_p = p
    r_e = ys[0]
    c = G * (SC @ r_e)
    return cbei_dfun(ys, c, node_p)


def cbei_observe_r(ys):
    """Return excitatory firing rate."""
    return ys[0]


def cbei_observe_V(ys):
    """Return excitatory mean membrane potential."""
    return ys[1]


# ====================================================================
# Robinson-Rennie-Wright corticothalamic model (Robinson et al. 2002)
#
# 4 populations: cortical excitatory (e), cortical inhibitory (i),
# thalamic relay (s), thalamic reticular nucleus (r).
#
# Alpha from corticothalamic loop delay (~85 ms round-trip → ~12 Hz).
#
# 12 first-order ODEs (from 2nd-order dendritic filters + wave eq):
#   phi_e, dphi_e      - cortical excitatory axonal field
#   V_e, dV_e          - cortical excitatory soma (2nd-order filter)
#   V_i, dV_i          - cortical inhibitory soma (2nd-order filter)
#   V_s, dV_s          - thalamic relay soma (2nd-order filter)
#   V_r, dV_r          - thalamic reticular soma (2nd-order filter)
#   (12 total)
#
# Each soma potential obeys the dendritic filter (NFTsim form):
#   dV/dt = W
#   dW/dt = alpha*beta*(drive - V) - (alpha+beta)*W
#
# Steady-state operating point (Bastiaens et al. 2025):
#   phi_e=3.175, V_e=0.634, V_i≈V_e, V_s=-3.234, V_r=5.676
#
# References:
#   Robinson PA, Rennie CJ, Wright JJ (2002) Prediction of EEG
#       spectra from neurophysiology. Phys Rev E 65:041924.
#   Bastiaens et al. (2025) Alpha models. PLoS Comp Biol.
# ====================================================================

RRWTheta = collections.namedtuple(
    typename='RRWTheta',
    field_names=(
        'Q_max theta sigma_prime '
        'gamma_e '
        'alpha beta '
        'nu_ee nu_ei nu_es '
        'nu_se nu_sr nu_sn '
        'nu_re nu_rs '
        't0 '
        'I'
    ).split())

rrw_default_theta = RRWTheta(
    # NFTsim canonical set (Robinson 2005, PMC1854922)
    # Produces alpha-band oscillations via corticothalamic delay
    Q_max=340.0,       # maximum firing rate (s^-1)
    theta=12.92,       # sigmoid threshold (mV)
    sigma_prime=3.8,   # sigmoid width (mV)
    gamma_e=116.0,     # cortical damping rate (s^-1)
    alpha=83.33,       # synaptic rise rate (s^-1)  [1/alpha = 12 ms]
    beta=769.23,       # synaptic decay rate (s^-1)  [1/beta = 1.3 ms]
    nu_ee=1.525,       # E→E cortical (mV*s)
    nu_ei=-3.023,      # I→E cortical (mV*s, negative)
    nu_es=0.567,       # relay→E cortical (mV*s)
    nu_se=3.447,       # E→relay (mV*s)
    nu_sr=-1.465,      # reticular→relay (mV*s, negative)
    nu_sn=3.593,       # external→relay (mV*s)
    nu_re=0.170,       # E→reticular (mV*s)
    nu_rs=0.051,       # relay→reticular (mV*s)
    t0=85.0,           # round-trip corticothalamic delay (ms)
    I=1.0,             # external stimulus rate (kHz), tonic thalamic drive
)

RRWState = collections.namedtuple(
    typename='RRWState',
    field_names='phi_e dphi_e V_e dV_e V_i dV_i V_s dV_s V_r dV_r'.split(' '))

rrw_default_state = RRWState(
    # Steady-state operating point from Bastiaens et al. (2025)
    phi_e=3.175, dphi_e=0.0,
    V_e=0.634, dV_e=0.0,
    V_i=0.634, dV_i=0.0,
    V_s=-3.234, dV_s=0.0,
    V_r=5.676, dV_r=0.0,
)


def rrw_dfun(ys, c, p):
    """Robinson-Rennie-Wright corticothalamic model (12D).

    Proper second-order dendritic filter for ALL populations, matching
    NFTsim's DendriteDE::rhs().  ODE form (no delay).

    Parameters
    ----------
    ys : array, shape (12,) or (12, n_nodes)
        [phi_e, dphi_e, V_e, dV_e, V_i, dV_i, V_s, dV_s, V_r, dV_r].
    c : array
        External coupling input (enters relay neuron).
    p : RRWTheta

    Returns
    -------
    dys : array, same shape as ys
    """
    phi_e, dphi_e, V_e, dV_e, V_i, dV_i, V_s, dV_s, V_r, dV_r = ys

    ge = p.gamma_e      # s^-1
    ab = p.alpha * p.beta  # s^-2
    apb = p.alpha + p.beta  # s^-1

    def S(V):
        return p.Q_max / (1.0 + np.exp(-(V - p.theta) / p.sigma_prime))

    Q_e = S(V_e)
    Q_i = S(V_i)
    Q_s = S(V_s)
    Q_r = S(V_r)

    # Cortical wave equation (spatially homogeneous)
    ddphi_e = ge**2 * (Q_e - phi_e) - 2.0 * ge * dphi_e

    # ODE: no delay
    phi_e_delayed = phi_e

    # NFTsim dendritic filter: dV/dt = W, dW/dt = ab*(drive - V) - apb*W
    drive_e = p.nu_ee * phi_e + p.nu_ei * Q_i + p.nu_es * Q_s
    ddV_e = ab * (drive_e - V_e) - apb * dV_e

    drive_i = p.nu_ee * phi_e + p.nu_ei * Q_i + p.nu_es * Q_s
    ddV_i = ab * (drive_i - V_i) - apb * dV_i

    drive_s = p.nu_se * phi_e_delayed + p.nu_sr * Q_r + p.nu_sn * (p.I + c)
    ddV_s = ab * (drive_s - V_s) - apb * dV_s

    drive_r = p.nu_re * phi_e_delayed + p.nu_rs * Q_s
    ddV_r = ab * (drive_r - V_r) - apb * dV_r

    # Convert s^-1 → ms^-1
    ms = 1e-3
    return np.array([
        dphi_e * ms, ddphi_e * ms,
        dV_e * ms, ddV_e * ms,
        dV_i * ms, ddV_i * ms,
        dV_s * ms, ddV_s * ms,
        dV_r * ms, ddV_r * ms,
    ])


def rrw_net_dfun(ys, p):
    """Network form for RRW model. Compatible with make_sde.

    Parameters
    ----------
    ys : array, shape (12, n_nodes)
    p : tuple (SC, G, node_theta)
    """
    SC, G, node_p = p
    phi_e = ys[0]
    c = G * (SC @ phi_e)
    return rrw_dfun(ys, c, node_p)


def rrw_observe_phi(ys):
    """Return cortical excitatory field phi_e (EEG-like observable)."""
    return ys[0]


def rrw_transfer_function(freqs_hz, p, phi_n_sq=1.0):
    """Analytical power spectrum of the RRW corticothalamic model.

    Computes the EEG power spectrum from the linearized transfer function
    without any time-domain simulation.  This is how Robinson's group
    actually uses the model (Robinson et al. 2001, 2002).

    Based on the braintrak ``nus_mass`` implementation and Bastiaens
    et al. (2025).

    Parameters
    ----------
    freqs_hz : array, shape (n_freq,)
        Frequencies in Hz.
    p : RRWTheta
        Model parameters.
    phi_n_sq : float
        Power of the white noise input (default 1.0).

    Returns
    -------
    psd : array, shape (n_freq,)
        Power spectral density at each frequency.
    """
    w = 2.0 * np.pi * freqs_hz  # rad/s

    # Sigmoid slope at the fixed point (same for all populations
    # in the standard symmetric parameterization)
    rho = p.Q_max / (4.0 * p.sigma_prime)  # s^-1 / mV

    # Linearized gains: G_ab = rho * nu_ab
    G_ee = rho * p.nu_ee
    G_ei = rho * p.nu_ei
    G_es = rho * p.nu_es
    G_se = rho * p.nu_se
    G_sr = rho * p.nu_sr
    G_sn = rho * p.nu_sn
    G_re = rho * p.nu_re
    G_rs = rho * p.nu_rs

    # Composite loop gains
    G_ese = G_es * G_se
    G_esre = G_es * G_sr * G_re
    G_srs = G_sr * G_rs

    # Dendritic filter: L(w) = 1 / ((1 - iw/alpha)(1 - iw/beta))
    iw = 1j * w
    L = 1.0 / ((1.0 - iw / p.alpha) * (1.0 - iw / p.beta))

    # Delay factor
    t0_s = p.t0 * 1e-3  # ms -> s
    delay = np.exp(iw * t0_s)

    # Denominator: spatially uniform (k=0) transfer function
    # A(w) = (1 - G_ei*L)(1 - iw/gamma_e)^2
    #       - G_ee*L*(1 - G_srs*L^2)
    #       - (G_ese*L^2 + G_esre*L^3) * exp(iw*t0)
    gamma_factor = (1.0 - iw / p.gamma_e) ** 2
    A = ((1.0 - G_ei * L) * gamma_factor
         - G_ee * L * (1.0 - G_srs * L**2)
         - (G_ese * L**2 + G_esre * L**3) * delay)

    # Numerator: noise enters via thalamic relay
    # phi_e(w) = G_esn * L^2 * phi_n * exp(iw*t0/2) / A(w)
    # (the half-delay accounts for noise entering at the thalamus)
    N = G_sn * L**2 * delay**(0.5)  # exp(iw*t0/2) = delay^0.5

    # Power spectrum
    T = N / A
    psd = np.abs(T) ** 2 * phi_n_sq

    return psd


def rrw_analytical_psd(freqs_hz, p, phi_n_sq=1.0):
    """Convenience alias for ``rrw_transfer_function``."""
    return rrw_transfer_function(freqs_hz, p, phi_n_sq)


# ====================================================================
# RRW with corticothalamic delay (SDDE form)
#
# The critical feature of the RRW model is the corticothalamic loop
# delay t0 (~85 ms round-trip).  phi_e reaches the thalamus at
# t - t0/2.  Without this delay the model cannot generate alpha.
#
# SDDE dfun signature: dfun(buf, x, t, p)
#   buf : history buffer, shape (buf_len, n_states)
#   x   : current state, shape (n_states,)
#   t   : current time index into buf
#   p   : parameters
# ====================================================================

def rrw_delay_steps(dt, t0=85.0):
    """Compute delay in time steps for the half round-trip.

    Parameters
    ----------
    dt : float
        Integration time step (ms).
    t0 : float
        Full round-trip corticothalamic delay (ms). Default 85 ms.

    Returns
    -------
    delay_steps : int
        Number of time steps for t0/2.
    """
    return int(np.floor(t0 / 2.0 / dt))


def rrw_sdde_dfun(buf, x, t, p):
    """RRW corticothalamic model with explicit delay (SDDE form).

    The cortical field phi_e reaches thalamic populations with a delay
    of t0/2 (half the round-trip).  This is the mechanism that produces
    alpha-band oscillations at ~1/t0 ≈ 12 Hz.

    Compatible with ``vbjax.make_sdde``.

    Parameters
    ----------
    buf : array, shape (buf_len, 8) or (buf_len, 8, n_nodes)
        History buffer of states.
    x : array, shape (8,) or (8, n_nodes)
        Current state [phi_e, dphi_e, V_e, V_i, V_s, V_r, dV_s, dV_r].
    t : int
        Current time index into buf.
    p : tuple (delay_steps, c_ext, node_theta)
        delay_steps : int, number of steps for t0/2
        c_ext : external coupling input
        node_theta : RRWTheta

    Returns
    -------
    dx : array, same shape as x
    """
    delay_steps, c, theta = p

    phi_e, dphi_e, V_e, dV_e, V_i, dV_i, V_s, dV_s, V_r, dV_r = x

    # Delayed cortical field: phi_e at t - delay_steps
    phi_e_delayed = buf[t - delay_steps, 0]

    ge = theta.gamma_e
    ab = theta.alpha * theta.beta
    apb = theta.alpha + theta.beta

    def S(V):
        return theta.Q_max / (1.0 + np.exp(-(V - theta.theta) / theta.sigma_prime))

    Q_e = S(V_e)
    Q_i = S(V_i)
    Q_s = S(V_s)
    Q_r = S(V_r)

    ddphi_e = ge**2 * (Q_e - phi_e) - 2.0 * ge * dphi_e

    # Cortical somas (local — no delay)
    drive_e = theta.nu_ee * phi_e + theta.nu_ei * Q_i + theta.nu_es * Q_s
    ddV_e = ab * (drive_e - V_e) - apb * dV_e

    drive_i = theta.nu_ee * phi_e + theta.nu_ei * Q_i + theta.nu_es * Q_s
    ddV_i = ab * (drive_i - V_i) - apb * dV_i

    # Thalamic — DELAYED cortical field
    drive_s = theta.nu_se * phi_e_delayed + theta.nu_sr * Q_r + theta.nu_sn * (theta.I + c)
    ddV_s = ab * (drive_s - V_s) - apb * dV_s

    drive_r = theta.nu_re * phi_e_delayed + theta.nu_rs * Q_s
    ddV_r = ab * (drive_r - V_r) - apb * dV_r

    ms = 1e-3
    return np.array([
        dphi_e * ms, ddphi_e * ms,
        dV_e * ms, ddV_e * ms,
        dV_i * ms, ddV_i * ms,
        dV_s * ms, ddV_s * ms,
        dV_r * ms, ddV_r * ms,
    ])


def make_rrw_sdde(dt=0.5, t0=85.0, gfun=0.1):
    """Convenience factory for RRW with corticothalamic delay.

    Parameters
    ----------
    dt : float
        Time step (ms).
    t0 : float
        Round-trip corticothalamic delay (ms).
    gfun : float or callable
        Diffusion coefficient.

    Returns
    -------
    step : callable
    loop : callable
        ``loop(buf, p)`` where buf shape is ``(nh + n_steps, 8)``
        and ``p = (delay_steps, c_ext, rrw_theta)``.
    nh : int
        History length (max delay in steps).
    delay_steps : int
        Steps for t0/2.

    Example
    -------
    >>> step, loop, nh, ds = make_rrw_sdde(dt=0.5, t0=85.0)
    >>> buf = jnp.zeros((nh + 10000, 8))  # history + simulation
    >>> buf = buf.at[nh:].set(vb.randn(10000, 8) * 0.1)  # noise
    >>> p = (ds, 0.0, vb.rrw_default_theta)
    >>> buf, xs = loop(buf, p)
    """
    delay_steps = rrw_delay_steps(dt, t0)
    nh = delay_steps + 1  # need at least delay_steps of history

    from .loops import make_sdde as _make_sdde
    step, loop = _make_sdde(dt, nh, rrw_sdde_dfun, gfun)

    return step, loop, nh, delay_steps


# ====================================================================
# Liley with inter-regional propagation delays (SDDE form)
#
# In the network form, excitatory firing rate from region j arrives at
# region i after a conduction delay d_ij = length_ij / v_e.
# The phi_ee and phi_ei long-range inputs become delayed sums.
#
# SDDE dfun signature: dfun(buf, x, t, p)
# ====================================================================

def liley_sdde_dfun(buf, x, t, p):
    """Liley model with delayed inter-regional coupling (SDDE form).

    For single-node (no network), this reduces to the standard Liley model
    with c=0.  For network use, the coupling is computed from the delayed
    buffer and passed via the parameter tuple.

    Compatible with ``vbjax.make_sdde``.

    Parameters
    ----------
    buf : array, shape (buf_len, 14) or (buf_len, 14, n_nodes)
        History buffer.
    x : array, shape (14,) or (14, n_nodes)
        Current state.
    t : int
        Current time index into buf.
    p : tuple (delay_coupling, node_theta)
        delay_coupling : array or scalar
            Pre-computed delayed coupling input (added to phi_ee).
            For single-node use, pass 0.0.
        node_theta : LileyTheta

    Returns
    -------
    dx : array, same shape as x
    """
    c_delayed, theta = p

    (h_e, h_i,
     I_ee, I_ei, I_ie, I_ii,
     dI_ee, dI_ei, dI_ie, dI_ii,
     phi_ee, phi_ei, dphi_ee, dphi_ei) = x

    ge = theta.gamma_e * 1e-3
    gi = theta.gamma_i * 1e-3

    S_e = theta.S_e_max / (1.0 + np.exp(-2.0 * (h_e - theta.mu_e) / theta.sigma_e))
    S_i = theta.S_i_max / (1.0 + np.exp(-2.0 * (h_i - theta.mu_i) / theta.sigma_i))

    psi_ee = (theta.h_ee_eq - h_e) / np.abs(theta.h_ee_eq - theta.h_e_rest)
    psi_ei = (theta.h_ei_eq - h_i) / np.abs(theta.h_ei_eq - theta.h_i_rest)
    psi_ie = (theta.h_ie_eq - h_e) / np.abs(theta.h_ie_eq - theta.h_e_rest)
    psi_ii = (theta.h_ii_eq - h_i) / np.abs(theta.h_ii_eq - theta.h_i_rest)

    dh_e = (1.0 / theta.tau_e) * (theta.h_e_rest - h_e + psi_ee * I_ee + psi_ie * I_ie)
    dh_i = (1.0 / theta.tau_i) * (theta.h_i_rest - h_i + psi_ei * I_ei + psi_ii * I_ii)

    v_Lambda = theta.v_e * theta.Lambda * 1e-3

    e_const = np.e

    ddI_ee = -2.0 * ge * dI_ee - ge**2 * I_ee + theta.Gamma_e * ge * e_const * (
        theta.N_ee_b * S_e + phi_ee + c_delayed + theta.p_ee)
    ddI_ei = -2.0 * ge * dI_ei - ge**2 * I_ei + theta.Gamma_e * ge * e_const * (
        theta.N_ei_b * S_e + phi_ei + theta.p_ei)
    ddI_ie = -2.0 * gi * dI_ie - gi**2 * I_ie + theta.Gamma_i * gi * e_const * (
        theta.N_ie_b * S_i)
    ddI_ii = -2.0 * gi * dI_ii - gi**2 * I_ii + theta.Gamma_i * gi * e_const * (
        theta.N_ii_b * S_i)

    ddphi_ee = (-2.0 * v_Lambda * dphi_ee - v_Lambda**2 * phi_ee
                + v_Lambda**2 * theta.N_ee_a * S_e)
    ddphi_ei = (-2.0 * v_Lambda * dphi_ei - v_Lambda**2 * phi_ei
                + v_Lambda**2 * theta.N_ei_a * S_e)

    return np.array([
        dh_e, dh_i,
        dI_ee, dI_ei, dI_ie, dI_ii,
        ddI_ee, ddI_ei, ddI_ie, ddI_ii,
        dphi_ee, dphi_ei, ddphi_ee, ddphi_ei,
    ])


def liley_sdde_net_dfun(buf, x, t, p):
    """Network Liley with delayed connectome coupling (SDDE form).

    Each region j's excitatory firing rate S_e arrives at region i
    after a conduction delay d_ij.  The delay helper pre-computes
    the delay indices from tract lengths and conduction velocity.

    Compatible with ``vbjax.make_sdde``.

    Parameters
    ----------
    buf : array, shape (buf_len, 14, n_nodes)
        History buffer.
    x : array, shape (14, n_nodes)
        Current state.
    t : int
        Current time index.
    p : tuple (delay_helper, G, node_theta)
        delay_helper : DelayHelper from vbjax.coupling
        G : float, global coupling gain
        node_theta : LileyTheta

    Returns
    -------
    dx : array, shape (14, n_nodes)
    """
    from .coupling import delay_apply

    dh, G, node_theta = p

    # Compute delayed coupling: weighted sum of delayed h_e across regions
    # h_e is state index 0
    h_e_delayed = delay_apply(dh, t, buf[:, 0:1, :])  # (1, n_nodes)

    # Firing rate of delayed h_e
    S_e_delayed = node_theta.S_e_max / (1.0 + np.exp(
        -2.0 * (h_e_delayed[0] - node_theta.mu_e) / node_theta.sigma_e))

    c_delayed = G * S_e_delayed

    return liley_sdde_dfun(buf, x, t, (c_delayed, node_theta))
