"""Autoresearch infrastructure — READ ONLY.

Provides neural mass model simulation, spectral fitting, and BMS utilities
for the AgentSciML evolutionary loop. The experiment.py modifies fitting
*strategy*; this file provides the stable API.
"""

from __future__ import annotations

import csv
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import vbjax as vb
from vbjax.spectral import welch_psd_jax
from vbjax.bms import bms_ffx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_FILE = Path(__file__).parent / "results.tsv"
TIMEOUT_SECONDS = 300  # 5 min max

# Simulation defaults (RTX 2080 feasible)
DEFAULT_DT_S = 0.5e-3       # 0.5 ms -> 2 kHz
DEFAULT_FS = 2000.0
DEFAULT_N_STEPS = 10_000     # 5 seconds
DEFAULT_N_WARMUP = 2_000
DEFAULT_NPERSEG = 512

# Precompute frequency grid
_freqs = np.fft.rfftfreq(DEFAULT_NPERSEG, d=DEFAULT_DT_S)
_band = np.where((_freqs >= 1.0) & (_freqs <= 100.0))[0]
BAND_START = int(_band[0])
BAND_END = int(_band[-1]) + 1
N_FREQ_BINS = BAND_END - BAND_START
TARGET_FREQS = _freqs[BAND_START:BAND_END]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    commit: str = ""
    strategy: str = ""
    model_name: str = ""
    n_free_params: int = 0
    n_subjects: int = 0
    n_opt_steps: int = 0
    lr: float = 0.0
    noise_sigma: float = 0.0
    dt_s: float = DEFAULT_DT_S
    n_steps: int = DEFAULT_N_STEPS
    spectral_loss: float = float("inf")
    free_energy: float = float("-inf")
    theta_map: list[float] = field(default_factory=list)
    wall_time: float = 0.0
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)


def get_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def print_result(result: ExperimentResult) -> None:
    """Print RESULT| line for AgentSciML parsing."""
    score = -result.spectral_loss  # higher is better
    print(
        f"RESULT|model={result.model_name}"
        f"|strategy={result.strategy}"
        f"|spectral_fit={score:.6f}"
        f"|loss={result.spectral_loss:.6f}"
        f"|free_energy={result.free_energy:.2f}"
        f"|n_params={result.n_free_params}"
        f"|n_steps={result.n_opt_steps}"
        f"|lr={result.lr}"
        f"|wall_time={result.wall_time:.1f}"
        f"|theta={result.theta_map}"
        f"|status={result.status}"
    )


def log_result(result: ExperimentResult) -> None:
    """Append result to results.tsv."""
    fieldnames = [
        "commit", "strategy", "model_name", "n_free_params", "n_subjects",
        "n_opt_steps", "lr", "noise_sigma", "spectral_loss", "free_energy",
        "theta_map", "wall_time", "status",
    ]
    write_header = not RESULTS_FILE.exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow({k: getattr(result, k) for k in fieldnames})


# ---------------------------------------------------------------------------
# Ground truth generator
# ---------------------------------------------------------------------------

def generate_synthetic_eeg(n_subjects: int = 2, dt_s: float = DEFAULT_DT_S,
                           n_steps: int = DEFAULT_N_STEPS,
                           n_warmup: int = DEFAULT_N_WARMUP,
                           nperseg: int = DEFAULT_NPERSEG,
                           noise_sigma: float = 0.5e-3,
                           seed: int = 0) -> jnp.ndarray:
    """Generate synthetic EEG PSDs from the Liley model (ground truth).

    Returns array of shape (n_subjects, N_FREQ_BINS).
    """
    def liley_gt(ys, p):
        return vb.liley_dfun(ys, 0.0, p)

    fs = 1.0 / dt_s
    _, loop = vb.make_sde(dt_s, liley_gt, noise_sigma)
    y0 = jnp.array([v for v in vb.liley_default_state])

    psds = []
    for i in range(n_subjects):
        key = jax.random.PRNGKey(seed + i)
        zs = jax.random.normal(key, (n_steps, 14))
        xs = loop(y0, zs, vb.liley_default_theta)
        obs = xs[n_warmup:, 0]
        _, psd_full = welch_psd_jax(obs, fs, nperseg=nperseg)
        psd_band = jax.lax.dynamic_slice(psd_full, (BAND_START,), (N_FREQ_BINS,))
        psds.append(psd_band)

    return jnp.stack(psds)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_model_dfun(model_name: str, free_param_names: list[str]):
    """Create a dfun(ys, theta) where theta is a flat array of log-deviations.

    Parameters
    ----------
    model_name : str
        One of 'liley', 'cmc', 'rrw', 'cbei'.
    free_param_names : list[str]
        Names of parameters to expose (must be valid fields of the model's
        default theta).

    Returns
    -------
    dfun : callable
        Compatible with vb.make_sde.
    n_states : int
    defaults : dict
        Default parameter values for the free params.
    """
    model_map = {
        'liley': (vb.liley_dfun, vb.liley_default_theta, 14),
        'cmc':   (vb.cmc_dfun,   vb.cmc_default_theta,   8),
        'rrw':   (vb.rrw_dfun,   vb.rrw_default_theta,   8),
        'cbei':  (vb.cbei_dfun,  vb.cbei_default_theta,   8),
    }

    base_dfun, default_theta, n_states = model_map[model_name]
    defaults = {name: getattr(default_theta, name) for name in free_param_names}

    def dfun(ys, theta):
        replacements = {}
        for i, name in enumerate(free_param_names):
            replacements[name] = jnp.exp(theta[i]) * defaults[name]
        p = default_theta._replace(**replacements)
        return base_dfun(ys, 0.0, p)

    return dfun, n_states, defaults


# ---------------------------------------------------------------------------
# Spectral fitting
# ---------------------------------------------------------------------------

def fit_model(model_name: str, free_param_names: list[str],
              target_psd: jnp.ndarray, *,
              n_opt_steps: int = 100, lr: float = 0.01,
              noise_sigma: float = 1e-3, dt_s: float = DEFAULT_DT_S,
              n_steps: int = DEFAULT_N_STEPS, n_warmup: int = DEFAULT_N_WARMUP,
              nperseg: int = DEFAULT_NPERSEG,
              theta_init: jnp.ndarray | None = None,
              noise_seed: int = 42,
              grad_clip: float = 10.0,
              prior_std: float = 1.0,
              lr_schedule: str = "constant",
              ) -> ExperimentResult:
    """Fit a model to a target PSD and return the result.

    This is the core function the experiment.py should call.
    """
    k = len(free_param_names)
    dfun, n_states, defaults = make_model_dfun(model_name, free_param_names)
    fs = 1.0 / dt_s

    _, sde_loop = vb.make_sde(dt_s, dfun, noise_sigma)
    noise_key = jax.random.PRNGKey(noise_seed)
    zs = jax.random.normal(noise_key, (n_steps, n_states))
    x0 = jnp.zeros(n_states)

    prior_mean = jnp.zeros(k)
    prior_std_arr = jnp.ones(k) * prior_std

    def forward_psd(theta):
        xs = sde_loop(x0, zs, theta)
        obs = xs[n_warmup:, 0]
        _, psd_full = welch_psd_jax(obs, fs, nperseg=nperseg)
        return jax.lax.dynamic_slice(psd_full, (BAND_START,), (N_FREQ_BINS,))

    def neg_log_joint(theta):
        pred = forward_psd(theta)
        eps = 1e-10
        nll = jnp.sum((jnp.log(pred + eps) - jnp.log(target_psd + eps))**2)
        nlp = 0.5 * jnp.sum(((theta - prior_mean) / prior_std_arr)**2)
        return nll + nlp

    # MAP estimation
    vg = jax.jit(jax.value_and_grad(neg_log_joint))
    theta = theta_init if theta_init is not None else jnp.zeros(k)
    m, v = jnp.zeros(k), jnp.zeros(k)

    t0 = time.perf_counter()
    for step in range(n_opt_steps):
        loss, grad = vg(theta)
        grad = jnp.clip(grad, -grad_clip, grad_clip)

        # Learning rate schedule
        if lr_schedule == "cosine":
            cur_lr = lr * 0.5 * (1 + np.cos(np.pi * step / n_opt_steps))
        elif lr_schedule == "warmup":
            warmup_steps = min(20, n_opt_steps // 5)
            cur_lr = lr * min(1.0, (step + 1) / warmup_steps)
        else:
            cur_lr = lr

        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        mh = m / (1 - 0.9**(step + 1))
        vh = v / (1 - 0.999**(step + 1))
        theta = theta - cur_lr * mh / (jnp.sqrt(vh) + 1e-8)

    wall_time = time.perf_counter() - t0
    final_loss = float(neg_log_joint(theta))

    # BIC free energy
    F_bic = -final_loss - 0.5 * k * np.log(N_FREQ_BINS)

    return ExperimentResult(
        commit=get_commit_hash(),
        model_name=model_name,
        n_free_params=k,
        n_opt_steps=n_opt_steps,
        lr=lr,
        noise_sigma=noise_sigma,
        dt_s=dt_s,
        n_steps=n_steps,
        spectral_loss=final_loss,
        free_energy=F_bic,
        theta_map=[float(t) for t in theta],
        wall_time=wall_time,
    )


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def run_comparison(models_config: list[dict], target_psds: jnp.ndarray,
                   **fit_kwargs) -> list[ExperimentResult]:
    """Fit multiple models to multiple subjects and return all results.

    Parameters
    ----------
    models_config : list of dict
        Each dict has 'name' and 'free_params' keys.
    target_psds : array (n_subjects, N_FREQ_BINS)
        Target PSDs to fit.
    **fit_kwargs
        Passed to fit_model.

    Returns
    -------
    results : list of ExperimentResult
    """
    strategy = fit_kwargs.pop('strategy', 'default')
    all_results = []
    for mcfg in models_config:
        for i in range(target_psds.shape[0]):
            result = fit_model(
                mcfg['name'], mcfg['free_params'],
                target_psds[i],
                noise_seed=42 + i,
                **fit_kwargs,
            )
            result.n_subjects = target_psds.shape[0]
            result.strategy = strategy
            all_results.append(result)
    return all_results
