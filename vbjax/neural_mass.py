import collections
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

cmc_default_theta = CMCTheta(
    He=3.25,        # excitatory PSP amplitude (mV), same as JR A
    Hi=22.0,        # inhibitory PSP amplitude (mV), same as JR B
    a=0.1,          # excitatory rate constant (ms⁻¹), same as JR a
    b=0.05,         # inhibitory rate constant (ms⁻¹), same as JR b
    r=0.56,         # sigmoid steepness (mV⁻¹)
    v0=6.0,         # sigmoid midpoint (mV)
    nu_max=0.0025,  # max firing rate (kHz)
    # excitatory intrinsic connections (tuned via DE for alpha oscillations)
    g_ss_sp=86.3,   # ss → sp, feedforward
    g_sp_ii=23.8,   # sp → ii
    g_sp_dp=188.0,  # sp → dp, descending
    g_dp_ii=68.1,   # dp → ii
    g_dp_sp=125.1,  # dp → sp, ascending feedback
    # inhibitory intrinsic connections
    g_ii_ss=120.3,  # ii → ss
    g_ii_sp=101.4,  # ii → sp
    g_ii_dp=158.5,  # ii → dp
    # external drive
    I=362.5,
)

CMCState = collections.namedtuple(
    typename='CMCState',
    field_names='x_ss x_sp x_ii x_dp v_ss v_sp v_ii v_dp'.split(' '))

cmc_default_state = CMCState(
    x_ss=0.0, x_sp=0.0, x_ii=0.0, x_dp=0.0,
    v_ss=0.0, v_sp=0.0, v_ii=0.0, v_dp=0.0)


def cmc_dfun(ys, c, p):
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


def cmc_net_dfun(ys, p):
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


def cmc_hier_dfun(ys, c_fwd, c_bwd, p):
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


def cmc_hier_2node_dfun(ys, p):
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


def cmc_hier_Nnode_dfun(ys, p):
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


def cmc_to_layer_activity(ys):
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


def cmc_observe_sp(ys):
    """Return superficial pyramidal membrane potential (EEG/MEG-like).

    In predictive coding, superficial pyramidal cells encode prediction
    errors and are the primary generators of EEG/MEG signals measured at
    the scalp.
    """
    return ys[1]


def cmc_observe_dp(ys):
    """Return deep pyramidal membrane potential (LFP/feedback-like).

    Deep pyramidal cells encode predictions and project to subcortical
    structures and lower cortical areas.
    """
    return ys[3]
