"""Differentiable spectral fitting pipeline for neural mass models.

Provides JAX-differentiable Welch PSD estimation, forward spectral prediction,
log-spectral distance loss, and MAP-based model inversion. All functions use
only ``jax.numpy`` operations so they are compatible with ``jax.grad``.

Typical usage::

    import vbjax as vb
    from vbjax.spectral import make_model_inverter

    def dfun(ys, p):
        return vb.jr_dfun(ys, 0, p)

    inverter = make_model_inverter(
        model_dfun=dfun, dt=2.0, n_steps=30000, n_warmup=5000,
        n_states=6, target_psd=target_psd, target_freqs=target_freqs,
        prior_mean=prior_mean, prior_std=prior_std,
    )
    result = inverter['fit'](theta_init)
"""

import jax
import jax.numpy as jnp
import vbjax as vb


# ---------------------------------------------------------------------------
# 1. Welch PSD estimator (fully differentiable)
# ---------------------------------------------------------------------------

def welch_psd_jax(x, fs, nperseg=256, noverlap=None):
    """Compute Welch power spectral density estimate using only JAX ops.

    Parameters
    ----------
    x : jnp.ndarray, shape (n_samples,)
        Input time series (1-D).
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Length of each segment (FFT window size).
    noverlap : int or None
        Number of overlapping samples. Defaults to ``nperseg // 2``.

    Returns
    -------
    freqs : jnp.ndarray, shape (nperseg // 2 + 1,)
        Frequency axis (one-sided, non-negative).
    psd : jnp.ndarray, shape (nperseg // 2 + 1,)
        Power spectral density estimate.
    """
    if noverlap is None:
        noverlap = nperseg // 2

    step = nperseg - noverlap
    n_samples = x.shape[0]
    n_segments = (n_samples - nperseg) // step + 1

    # Build segment indices via lax-friendly gather
    starts = jnp.arange(n_segments) * step          # (n_segments,)
    offsets = jnp.arange(nperseg)                    # (nperseg,)
    indices = starts[:, None] + offsets[None, :]     # (n_segments, nperseg)
    segments = x[indices]                            # (n_segments, nperseg)

    # Hanning window  (periodic=False convention, matching scipy.signal.hann)
    window = jnp.hanning(nperseg)
    win_norm = jnp.sum(window ** 2)

    segments = segments * window[None, :]

    # One-sided FFT
    fft_vals = jnp.fft.rfft(segments, axis=-1)       # (n_segments, nperseg//2+1)
    power = jnp.real(fft_vals * jnp.conj(fft_vals))  # magnitude squared

    # Average across segments
    psd = jnp.mean(power, axis=0)

    # Normalise:  PSD = (1 / (fs * win_norm)) * |X|^2
    # Factor of 2 for one-sided spectrum (except DC and Nyquist)
    psd = psd / (fs * win_norm)
    n_freqs = psd.shape[0]
    scale = jnp.ones(n_freqs)
    scale = scale.at[1:-1].set(2.0)
    psd = psd * scale

    freqs = jnp.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, psd


# ---------------------------------------------------------------------------
# 2. Forward spectral predictor factory
# ---------------------------------------------------------------------------

def make_spectral_predictor(model_dfun, dt, n_steps, n_warmup, fs,
                            nperseg=256, noise_sigma=0.5, observe_fn=None,
                            key=None):
    """Build a differentiable function ``predict_psd(theta) -> (freqs, psd)``.

    Parameters
    ----------
    model_dfun : callable
        Drift function with signature ``dfun(ys, p)`` compatible with
        :func:`vbjax.make_sde`.
    dt : float
        Integration time step (in the same units as *fs*, typically ms if
        ``fs`` is in kHz, or seconds if ``fs`` is in Hz). The caller is
        responsible for consistency.
    n_steps : int
        Total number of integration steps (including warm-up).
    n_warmup : int
        Number of initial steps to discard as transient.
    fs : float
        Sampling frequency of the simulated signal (Hz).
    nperseg : int
        Welch window length.
    noise_sigma : float
        Diffusion coefficient for additive noise.
    observe_fn : callable or None
        Function ``observe_fn(states)`` applied to the full state trajectory
        ``(n_time, n_states, ...)`` to extract a 1-D observable. Defaults to
        selecting the first state variable: ``states[:, 0]``.
    key : jax.random.PRNGKey or None
        RNG key for generating frozen noise. Defaults to ``PRNGKey(42)``.

    Returns
    -------
    predict_psd : callable
        ``predict_psd(theta) -> (freqs, psd)`` where *theta* is passed
        directly to the SDE loop as the parameter argument.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    if observe_fn is None:
        observe_fn = lambda states: states[:, 0]

    _, sde_loop = vb.make_sde(dt, model_dfun, noise_sigma)

    def predict_psd(theta):
        # Infer initial state shape from a probe call
        # We need the caller to set up initial conditions; use zeros as safe
        # default for shape discovery.  The frozen noise drives the system
        # away from the fixed point anyway.
        n_states = _infer_n_states(model_dfun, theta)
        x0 = jnp.zeros(n_states)
        zs = jax.random.normal(key, (n_steps, n_states))
        xs = sde_loop(x0, zs, theta)        # (n_steps, n_states)
        xs = xs[n_warmup:]                    # discard transient
        obs = observe_fn(xs)                  # 1-D signal
        # Flatten to 1-D in case observe_fn returns (T, 1)
        obs = obs.reshape(-1)
        freqs, psd = welch_psd_jax(obs, fs, nperseg=nperseg)
        # Band-limit to 1 -- 100 Hz
        mask = (freqs >= 1.0) & (freqs <= 100.0)
        return freqs[mask], psd[mask]

    return predict_psd


def _infer_n_states(dfun, theta):
    """Try calling dfun with progressively larger state vectors to discover
    the expected state size.  This is a build-time helper, not traced."""
    # Common neural mass state dimensions
    for n in [2, 4, 6, 8, 10, 12, 16]:
        try:
            out = dfun(jnp.zeros(n), theta)
            if out.shape == (n,):
                return n
        except Exception:
            continue
    raise RuntimeError("Could not infer n_states from model_dfun; "
                       "pass n_states explicitly or check dfun signature.")


# ---------------------------------------------------------------------------
# 3. Spectral loss (log-spectral distance)
# ---------------------------------------------------------------------------

def spectral_loss(predicted_psd, target_psd):
    """Log-spectral distance between two PSDs.

    Parameters
    ----------
    predicted_psd : jnp.ndarray
        Predicted power spectral density.
    target_psd : jnp.ndarray
        Target (observed) power spectral density.

    Returns
    -------
    loss : scalar
        Sum of squared log-spectral differences.
    """
    eps = 1e-10
    return jnp.sum(
        (jnp.log(predicted_psd + eps) - jnp.log(target_psd + eps)) ** 2
    )


# ---------------------------------------------------------------------------
# 4. Model inverter factory (MAP estimation + Bayesian model comparison)
# ---------------------------------------------------------------------------

def make_model_inverter(model_dfun, dt, n_steps, n_warmup, n_states,
                        target_psd, target_freqs, prior_mean, prior_std,
                        noise_sigma=0.5, observe_fn=None, key=None):
    """Build functions for MAP estimation and Bayesian model comparison.

    Parameters
    ----------
    model_dfun : callable
        Drift function ``dfun(ys, p)`` compatible with :func:`vbjax.make_sde`.
    dt : float
        Integration time step.
    n_steps : int
        Total simulation steps.
    n_warmup : int
        Transient steps to discard.
    n_states : int
        Dimensionality of the state vector.
    target_psd : jnp.ndarray
        Observed power spectral density to fit (already band-limited to the
        same frequency grid that the predictor will produce).
    target_freqs : jnp.ndarray
        Corresponding frequency axis (used only for ``fs`` inference and
        consistency checks).
    prior_mean : jnp.ndarray
        Mean of the Gaussian prior on *theta*.
    prior_std : jnp.ndarray
        Standard deviation of the Gaussian prior on *theta*.
    noise_sigma : float
        SDE diffusion coefficient.
    observe_fn : callable or None
        Observation function (see :func:`make_spectral_predictor`).
    key : jax.random.PRNGKey or None
        RNG key for frozen noise.

    Returns
    -------
    dict with keys:

    - ``'neg_log_joint'`` : ``theta -> scalar`` -- negative log posterior
      (up to a constant).
    - ``'predict_psd'`` : ``theta -> (freqs, psd)`` -- forward prediction.
    - ``'fit'`` : ``(theta_init, n_iter, lr) -> dict`` -- MAP optimisation.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    if observe_fn is None:
        observe_fn = lambda states: states[:, 0]

    prior_mean = jnp.asarray(prior_mean, dtype=jnp.float32)
    prior_std = jnp.asarray(prior_std, dtype=jnp.float32)
    target_psd = jnp.asarray(target_psd, dtype=jnp.float32)

    # Infer sampling frequency from the target frequency grid
    df = target_freqs[1] - target_freqs[0]
    # For rfft: fs = 2 * f_nyquist, but we only have the band-limited grid.
    # Instead, we reconstruct fs from df and nperseg at the call-site later.
    # A simpler approach: fs = 1/dt  (dt in seconds).
    # The caller must ensure dt is in seconds for fs to be in Hz.
    fs = 1.0 / dt

    _, sde_loop = vb.make_sde(dt, model_dfun, noise_sigma)

    # Pre-generate frozen noise for differentiability
    zs = jax.random.normal(key, (n_steps, n_states))
    x0 = jnp.zeros(n_states)

    # Figure out a good nperseg from the target frequency grid
    # df = fs / nperseg  =>  nperseg = fs / df
    nperseg = int(round(float(fs / df)))
    # Ensure it is even (for nice rfft behaviour)
    if nperseg % 2 != 0:
        nperseg += 1

    def predict_psd(theta):
        xs = sde_loop(x0, zs, theta)   # (n_steps, n_states)
        xs = xs[n_warmup:]
        obs = observe_fn(xs).reshape(-1)
        freqs, psd = welch_psd_jax(obs, fs, nperseg=nperseg)
        mask = (freqs >= 1.0) & (freqs <= 100.0)
        return freqs[mask], psd[mask]

    def neg_log_likelihood(theta):
        _, pred_psd = predict_psd(theta)
        # Truncate / pad to match target length (they should already match
        # if the caller set up target_freqs correctly).
        n = jnp.minimum(pred_psd.shape[0], target_psd.shape[0])
        return spectral_loss(pred_psd[:n], target_psd[:n])

    def neg_log_prior(theta):
        return 0.5 * jnp.sum(((theta - prior_mean) / prior_std) ** 2)

    def neg_log_joint(theta):
        return neg_log_likelihood(theta) + neg_log_prior(theta)

    def fit(theta_init, n_iter=200, lr=0.01):
        """Run MAP estimation via Adam gradient descent.

        Parameters
        ----------
        theta_init : jnp.ndarray
            Initial parameter vector.
        n_iter : int
            Number of optimisation steps.
        lr : float
            Learning rate.

        Returns
        -------
        dict with keys ``'theta'``, ``'loss_trace'``, ``'freqs'``, ``'psd'``.
        """
        theta_init = jnp.asarray(theta_init, dtype=jnp.float32)

        val_and_grad = jax.jit(jax.value_and_grad(neg_log_joint))

        # Simple Adam optimiser (no dependency on optax)
        m = jnp.zeros_like(theta_init)
        v = jnp.zeros_like(theta_init)
        beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
        theta = theta_init

        loss_trace = []
        for i in range(n_iter):
            loss_val, grad = val_and_grad(theta)
            loss_trace.append(float(loss_val))
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad ** 2
            m_hat = m / (1.0 - beta1 ** (i + 1))
            v_hat = v / (1.0 - beta2 ** (i + 1))
            theta = theta - lr * m_hat / (jnp.sqrt(v_hat) + eps_adam)

        freqs, psd = predict_psd(theta)
        return {
            'theta': theta,
            'loss_trace': jnp.array(loss_trace),
            'freqs': freqs,
            'psd': psd,
        }

    return {
        'neg_log_joint': neg_log_joint,
        'predict_psd': predict_psd,
        'fit': fit,
    }


# ---------------------------------------------------------------------------
# 5. Synthetic EEG generator
# ---------------------------------------------------------------------------

def generate_synthetic_eeg(model_dfun, theta, dt, n_steps, n_states,
                           noise_sigma=0.5, observe_fn=None, key=None):
    """Generate synthetic EEG from a neural mass model.

    Parameters
    ----------
    model_dfun : callable
        Drift function ``dfun(ys, p)`` compatible with :func:`vbjax.make_sde`.
    theta : pytree
        Parameters passed to the SDE loop.
    dt : float
        Integration time step (seconds).
    n_steps : int
        Number of simulation steps.
    n_states : int
        Dimensionality of the state vector.
    noise_sigma : float
        Diffusion coefficient.
    observe_fn : callable or None
        Maps ``(n_steps, n_states)`` trajectory to 1-D signal.
        Defaults to first state variable.
    key : jax.random.PRNGKey or None
        RNG key.

    Returns
    -------
    time_series : jnp.ndarray, shape (n_steps,)
        Observed (1-D) signal.
    freqs : jnp.ndarray
        Frequency axis of the PSD.
    psd : jnp.ndarray
        Welch PSD of the observed signal.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    if observe_fn is None:
        observe_fn = lambda states: states[:, 0]

    fs = 1.0 / dt

    _, sde_loop = vb.make_sde(dt, model_dfun, noise_sigma)
    x0 = jnp.zeros(n_states)
    zs = jax.random.normal(key, (n_steps, n_states))

    xs = sde_loop(x0, zs, theta)     # (n_steps, n_states)
    obs = observe_fn(xs).reshape(-1)

    freqs, psd = welch_psd_jax(obs, fs)
    return obs, freqs, psd
