"""
SPM-style Bayesian Model Selection for comparing neural mass models.

Implements the complete DCM-style Bayesian model comparison pipeline:
  - Laplace approximation to log model evidence
  - Variational Laplace (MAP estimation + free energy)
  - Fixed-effects BMS (Bayesian model selection)
  - Random-effects BMS (Stephan et al. 2009)
  - Protected exceedance probability (Rigoux et al. 2014)
  - Spectral and Gaussian log-likelihood / log-prior helpers

References
----------
Stephan, K.E. et al. (2009). Bayesian model selection for group studies.
    NeuroImage, 46(4), 1004-1017.
Rigoux, L. et al. (2014). Bayesian model selection for group studies -
    Revisited. NeuroImage, 84, 971-985.
Friston, K. et al. (2007). Variational free energy and the Laplace
    approximation. NeuroImage, 34(1), 220-234.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import digamma


# ---------------------------------------------------------------------------
# Helpers: log-likelihood and log-prior
# ---------------------------------------------------------------------------

def spectral_log_likelihood(observed_psd, predicted_psd, precision=1.0):
    """Gaussian log-likelihood in the spectral (log-PSD) domain.

    Operating on log-PSD is standard for EEG spectral fitting because
    it stabilises variance across frequency bins.

    Parameters
    ----------
    observed_psd : array, shape (n_freq,) or (n_channels, n_freq)
        Observed power spectral density.
    predicted_psd : array, shape matching *observed_psd*
        Model-predicted power spectral density.
    precision : float
        Observation precision (inverse variance).

    Returns
    -------
    ll : scalar
        Log-likelihood value.
    """
    residual = jnp.log(observed_psd) - jnp.log(predicted_psd)
    return -0.5 * precision * jnp.sum(residual ** 2)


def gaussian_log_prior(theta, prior_mean, prior_precision):
    """Gaussian log-prior with diagonal precision.

    Parameters
    ----------
    theta : array, shape (k,)
        Parameter vector.
    prior_mean : array, shape (k,)
        Prior mean.
    prior_precision : array, shape (k,)
        Diagonal of the prior precision matrix (inverse variance per
        parameter).

    Returns
    -------
    lp : scalar
        Log-prior value.
    """
    diff = theta - prior_mean
    return -0.5 * jnp.dot(diff, prior_precision * diff)


# ---------------------------------------------------------------------------
# Laplace free energy
# ---------------------------------------------------------------------------

def laplace_free_energy(neg_log_joint_fn, theta_map, n_data):
    """Laplace approximation to the log model evidence.

    Computes the negative variational free energy:

        F = -NLJ(theta_MAP) + (k/2)*log(2*pi) - 0.5*log|H|

    where NLJ is the negative log joint, H is the Hessian of NLJ
    evaluated at the MAP estimate, and k = len(theta_MAP).

    Parameters
    ----------
    neg_log_joint_fn : callable
        Maps theta -> scalar negative log joint (= -log p(y|theta) - log p(theta)).
    theta_map : array, shape (k,)
        MAP parameter estimate.
    n_data : int
        Number of data points (unused in the formula but kept for
        interface consistency with information-criterion methods).

    Returns
    -------
    F : scalar
        Laplace approximation to the log model evidence.
    posterior_cov : array, shape (k, k)
        Posterior covariance (inverse Hessian at MAP).
    """
    k = theta_map.shape[0]
    nlj_map = neg_log_joint_fn(theta_map)

    # Hessian of the negative log joint at the MAP
    H = jax.hessian(neg_log_joint_fn)(theta_map)

    # log|H| via slogdet for numerical stability
    sign, logdet = jnp.linalg.slogdet(H)

    # Posterior covariance = H^{-1}
    posterior_cov = jnp.linalg.inv(H)

    # Laplace free energy (log model evidence approximation)
    F = -nlj_map + 0.5 * k * jnp.log(2.0 * jnp.pi) - 0.5 * logdet

    return F, posterior_cov


# ---------------------------------------------------------------------------
# Variational Laplace (MAP + Laplace free energy)
# ---------------------------------------------------------------------------

def variational_laplace(neg_log_joint_fn, theta_init, lr=0.01,
                        max_iter=500, tol=1e-6):
    """MAP estimation via Adam, then Laplace free-energy evaluation.

    Uses a manual Adam implementation (no external optimiser dependency).

    Parameters
    ----------
    neg_log_joint_fn : callable
        Maps theta -> scalar negative log joint.
    theta_init : array, shape (k,)
        Initial parameter vector.
    lr : float
        Learning rate for Adam.
    max_iter : int
        Maximum number of optimisation iterations.
    tol : float
        Convergence tolerance on the absolute change in the objective.

    Returns
    -------
    result : dict
        theta_map       : MAP parameter estimate
        free_energy     : Laplace free energy F
        posterior_cov   : posterior covariance (inverse Hessian)
        neg_log_joint   : NLJ value at MAP
        converged       : bool, whether tolerance was reached
    """
    grad_fn = jax.grad(neg_log_joint_fn)

    # Adam state
    theta = theta_init.copy()
    m = jnp.zeros_like(theta)   # first moment
    v = jnp.zeros_like(theta)   # second moment
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    prev_loss = neg_log_joint_fn(theta)
    converged = False

    for t in range(1, max_iter + 1):
        g = grad_fn(theta)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g ** 2
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        theta = theta - lr * m_hat / (jnp.sqrt(v_hat) + eps)

        loss = neg_log_joint_fn(theta)
        if jnp.abs(loss - prev_loss) < tol:
            converged = True
            break
        prev_loss = loss

    n_data = 0  # placeholder; n_data not used in the formula
    F, posterior_cov = laplace_free_energy(neg_log_joint_fn, theta, n_data)

    return {
        'theta_map': theta,
        'free_energy': F,
        'posterior_cov': posterior_cov,
        'neg_log_joint': neg_log_joint_fn(theta),
        'converged': converged,
    }


# ---------------------------------------------------------------------------
# Fixed-effects BMS
# ---------------------------------------------------------------------------

def bms_ffx(free_energies):
    """Fixed-effects Bayesian model selection.

    Sums the log model evidence across subjects and converts to
    posterior model probabilities via softmax.

    Parameters
    ----------
    free_energies : array, shape (n_subjects, n_models)
        Log model evidence (free energy) per subject and model.

    Returns
    -------
    posterior_prob : array, shape (n_models,)
        Posterior model probabilities.
    winning_model : int
        Index of the winning model.
    """
    summed = jnp.sum(free_energies, axis=0)
    # softmax for numerical stability
    log_probs = summed - jnp.max(summed)
    probs = jnp.exp(log_probs)
    posterior_prob = probs / jnp.sum(probs)
    winning_model = jnp.argmax(posterior_prob)
    return posterior_prob, winning_model


# ---------------------------------------------------------------------------
# Monte-Carlo exceedance probability
# ---------------------------------------------------------------------------

def exceedance_probability(alpha, n_samples=100_000, key=None):
    """Monte Carlo exceedance probability from a posterior Dirichlet.

    For each sample from Dir(alpha), records which model has the
    highest frequency, then averages across samples.

    Parameters
    ----------
    alpha : array, shape (K,)
        Posterior Dirichlet concentration parameters.
    n_samples : int
        Number of Monte Carlo draws.
    key : jax.random.PRNGKey or None
        PRNG key.  If *None*, a deterministic seed is used.

    Returns
    -------
    xp : array, shape (K,)
        Exceedance probabilities.
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    samples = jax.random.dirichlet(key, alpha, shape=(n_samples,))
    # For each sample, which model has the highest frequency?
    winners = (samples == samples.max(axis=1, keepdims=True)).astype(jnp.float32)
    xp = winners.mean(axis=0)
    return xp


# ---------------------------------------------------------------------------
# Protected exceedance probability (Rigoux et al. 2014)
# ---------------------------------------------------------------------------

def protected_exceedance_probability(xp, free_energies, alpha):
    """Protected exceedance probability with Bayesian omnibus risk.

    Corrects exceedance probabilities for the possibility that the
    observed log-evidence differences are due to chance (all models
    equally likely).

    Parameters
    ----------
    xp : array, shape (K,)
        Raw exceedance probabilities.
    free_energies : array, shape (n_subjects, n_models)
        Log model evidence per subject and model.
    alpha : array, shape (K,)
        Posterior Dirichlet parameters from RFX BMS.

    Returns
    -------
    pxp : array, shape (K,)
        Protected exceedance probabilities.
    bor : scalar
        Bayesian omnibus risk.
    """
    K = free_energies.shape[1]

    # F_1: evidence that models differ (best model per subject)
    F1 = jnp.sum(jnp.max(free_energies, axis=1))

    # F_0: evidence that all models are equally likely
    F0 = jnp.sum(jnp.full(free_energies.shape[0], jnp.log(1.0 / K)))

    # Bayesian omnibus risk
    # bor = 1 / (1 + exp(F1 - F0))  =  sigmoid(F0 - F1)
    bor = 1.0 / (1.0 + jnp.exp(F1 - F0))

    # Protected exceedance probability
    pxp = (1.0 - bor) * xp + bor / K
    return pxp, bor


# ---------------------------------------------------------------------------
# Random-effects BMS (Stephan et al. 2009)
# ---------------------------------------------------------------------------

def bms_rfx(free_energies, max_iter=1000, tol=1e-6):
    """Random-effects Bayesian model selection.

    Implements the variational Bayes algorithm of Stephan et al. (2009)
    to estimate a Dirichlet posterior over model frequencies given
    per-subject log model evidences.

    Parameters
    ----------
    free_energies : array, shape (n_subjects, n_models)
        Log model evidence (free energy) per subject and model.
    max_iter : int
        Maximum VB iterations.
    tol : float
        Convergence tolerance on the Dirichlet parameter update.

    Returns
    -------
    result : dict
        alpha                       : posterior Dirichlet parameters
        exp_r                       : expected model frequencies
        exceedance_prob             : Monte Carlo exceedance probabilities
        protected_exceedance_prob   : BOR-corrected exceedance probabilities
    """
    free_energies = jnp.asarray(free_energies)
    N, K = free_energies.shape
    lme = free_energies  # alias for clarity

    alpha = jnp.ones(K)  # uniform Dirichlet prior

    for _ in range(max_iter):
        # Expected log model frequencies under current Dirichlet
        log_u = lme + (digamma(alpha) - digamma(jnp.sum(alpha)))[None, :]

        # Numerically stable softmax per subject
        log_u = log_u - jnp.max(log_u, axis=1, keepdims=True)
        u = jnp.exp(log_u)
        g = u / jnp.sum(u, axis=1, keepdims=True)

        # Update Dirichlet (prior count = 1)
        alpha_new = jnp.ones(K) + jnp.sum(g, axis=0)

        if jnp.linalg.norm(alpha_new - alpha) < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    # Expected model frequencies
    exp_r = alpha / jnp.sum(alpha)

    # Exceedance and protected exceedance probabilities
    xp = exceedance_probability(alpha)
    pxp, bor = protected_exceedance_probability(xp, free_energies, alpha)

    return {
        'alpha': alpha,
        'exp_r': exp_r,
        'exceedance_prob': xp,
        'protected_exceedance_prob': pxp,
    }
