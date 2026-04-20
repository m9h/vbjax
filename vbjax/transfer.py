"""Analytical transfer functions for neural mass models.

Computes EEG power spectra from linearized models without time-domain
simulation.  This is how SPM's DCM and the Robinson group's NFTsim
compute spectra — and provides apples-to-apples model comparison.

For any neural mass model dx/dt = f(x, p):
  1. Find fixed point x* where f(x*, p) ≈ 0
  2. Jacobian J = df/dx at x*
  3. Transfer function T(ω) = C @ (iωI - J)^{-1} @ B
  4. PSD P(ω) = |T(ω)|² × noise_power

References:
    Robinson et al. (2001, 2002) — corticothalamic transfer function
    David & Friston (2003) — JR transfer function for DCM
    Moran et al. (2013) — CMC transfer function for DCM
    Liley et al. (2002) — Liley spectral analysis
"""

import jax
import jax.numpy as jnp
import numpy as np


# =====================================================================
# Generic linearized transfer function
# =====================================================================

def find_fixed_point(dfun, x0, p, n_steps=50000, dt=0.01):
    """Find approximate fixed point by forward integration.

    Parameters
    ----------
    dfun : callable
        ``dfun(x, coupling, p)`` returning dx/dt.
    x0 : array
        Initial state.
    p : pytree
        Parameters.
    n_steps : int
        Integration steps.
    dt : float
        Step size.

    Returns
    -------
    x_star : array
        Approximate fixed point.
    """
    x = x0
    for _ in range(n_steps):
        dx = dfun(x, 0.0, p)
        x = x + dt * dx
    return x


def linearized_transfer_function(dfun, x_star, p, freqs_hz,
                                  B=None, C=None, noise_power=1.0):
    """Compute analytical PSD from a linearized neural mass model.

    Parameters
    ----------
    dfun : callable
        ``dfun(x, coupling, p)`` returning dx/dt.  Must work with the
        model's time units (typically ms^-1 for vbjax models).
    x_star : array, shape (n,)
        Fixed point (equilibrium state).
    p : pytree
        Model parameters.
    freqs_hz : array, shape (n_freq,)
        Frequencies in Hz.
    B : array, shape (n,) or (n, n_inputs), optional
        Input (noise) matrix.  Default: identity (noise on all states).
    C : array, shape (n,) or (n_outputs, n), optional
        Observation matrix.  Default: first state variable [1, 0, ...].
    noise_power : float
        Input noise power spectral density.

    Returns
    -------
    psd : array, shape (n_freq,)
        Power spectral density at each frequency.
    """
    n = x_star.shape[0]

    # Jacobian at the fixed point
    J = jax.jacobian(lambda x: dfun(x, 0.0, p))(x_star)
    J = jnp.array(J)

    # Default input: noise on all states equally
    if B is None:
        B = jnp.eye(n)
    B = jnp.atleast_2d(B)
    if B.shape[0] == 1 and B.shape[1] == n:
        B = B.T  # row vector -> column

    # Default observation: first state variable
    if C is None:
        C = jnp.zeros(n).at[0].set(1.0)
    C = jnp.atleast_2d(C)
    if C.shape[0] == n and C.shape[1] == 1:
        C = C.T  # column vector -> row

    # Convert Hz to rad per model time unit
    # vbjax models output derivatives in ms^-1, so ω = 2π f × 1e-3
    w = 2.0 * np.pi * freqs_hz * 1e-3  # rad/ms

    I = jnp.eye(n)
    psd = jnp.zeros(len(w))

    # T(ω) = C @ (iωI - J)^{-1} @ B
    # P(ω) = trace(T @ T^H) × noise_power  (for multi-input)
    def compute_psd_at_freq(wi):
        resolvent = jnp.linalg.inv(1j * wi * I - J)  # (n, n)
        T = C @ resolvent @ B  # (n_out, n_in)
        return jnp.sum(jnp.abs(T)**2) * noise_power

    psd = jnp.array([compute_psd_at_freq(wi) for wi in w])
    return psd


# =====================================================================
# Model-specific wrappers
# =====================================================================

def jr_analytical_psd(freqs_hz, p, noise_power=1.0):
    """Analytical PSD for the Jansen-Rit model.

    Observable: y1 - y2 (pyramidal membrane potential, EEG-like).
    Noise input: all states (broadband cortical noise).
    """
    import vbjax as vb
    x_star = jnp.zeros(6)
    C = jnp.zeros(6).at[1].set(1.0).at[2].set(-1.0)
    return linearized_transfer_function(
        vb.jr_dfun, x_star, p, freqs_hz, C=C, noise_power=noise_power)


def cmc_analytical_psd(freqs_hz, p, noise_power=1.0):
    """Analytical PSD for the Canonical Microcircuit model.

    Observable: x_sp (superficial pyramidal = EEG generator).
    Noise input: all states (broadband cortical noise).
    """
    import vbjax as vb
    x_star = jnp.zeros(8)
    C = jnp.zeros(8).at[1].set(1.0)
    return linearized_transfer_function(
        vb.cmc_dfun, x_star, p, freqs_hz, C=C, noise_power=noise_power)


def liley_analytical_psd(freqs_hz, p, noise_power=1.0):
    """Analytical PSD for the Liley mean-field cortical model.

    Observable: h_e (excitatory membrane potential = EEG).
    Noise input: all states (broadband cortical noise).
    """
    import vbjax as vb
    x_star = jnp.array([v for v in vb.liley_default_state])
    C = jnp.zeros(14).at[0].set(1.0)
    return linearized_transfer_function(
        vb.liley_dfun, x_star, p, freqs_hz, C=C, noise_power=noise_power)


def cbei_analytical_psd(freqs_hz, p, noise_power=1.0):
    """Analytical PSD for the Coombes-Byrne E-I model.

    Observable: V_e (excitatory mean membrane potential).
    Noise input: all states (broadband cortical noise).
    """
    import vbjax as vb
    x_star = jnp.array([v for v in vb.cbei_default_state])
    C = jnp.zeros(8).at[1].set(1.0)
    return linearized_transfer_function(
        vb.cbei_dfun, x_star, p, freqs_hz, C=C, noise_power=noise_power)
