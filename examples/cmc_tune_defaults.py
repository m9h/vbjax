"""Tune CMC default parameters for physiological alpha oscillations.

Uses CMA-ES (or optionally LLaMEA) to find connectivity weights that
produce oscillatory dynamics matching Jansen-Rit's spectral properties:
- Peak frequency in alpha band (8-13 Hz)
- Sufficient oscillation amplitude
- Stable bounded dynamics

Usage:
    python examples/cmc_tune_defaults.py                    # CMA-ES (no API key needed)
    python examples/cmc_tune_defaults.py --optimizer llamea  # LLaMEA (needs ANTHROPIC_API_KEY)
    python examples/cmc_tune_defaults.py --budget 200        # more evaluations
"""

import sys
import time
import numpy as np
import jax
import jax.numpy as jp
from scipy.signal import welch
from scipy.optimize import differential_evolution

import vbjax as vb


# ── Search space ─────────────────────────────────────────────────────

PARAM_BOUNDS = {
    "g_ss_sp": (20.0, 500.0),
    "g_sp_ii": (5.0, 250.0),
    "g_sp_dp": (20.0, 500.0),
    "g_dp_ii": (5.0, 250.0),
    "g_dp_sp": (20.0, 500.0),
    "g_ii_ss": (5.0, 250.0),
    "g_ii_sp": (5.0, 250.0),
    "g_ii_dp": (5.0, 250.0),
    "I":       (50.0, 500.0),
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())
BOUNDS_ARRAY = [PARAM_BOUNDS[k] for k in PARAM_NAMES]


# ── Simulation ───────────────────────────────────────────────────────

DT = 0.5
N_STEPS = 10000
WARMUP = 3000
FS = 1000.0 / DT
SIGMA = 1e-3

# Fixed noise for reproducible evaluation
KEY = jax.random.PRNGKey(42)
ZS = jax.random.normal(KEY, (N_STEPS, 8))

# Pre-compile loop
_, LOOP = vb.make_sde(
    dt=DT,
    dfun=lambda y, p: vb.cmc_dfun(y, 0.0, p),
    gfun=SIGMA,
)


def simulate(params_vec):
    """Run CMC with given parameter vector, return sp time series."""
    params_dict = dict(zip(PARAM_NAMES, params_vec))
    theta = vb.cmc_default_theta._replace(**params_dict)
    y0 = jp.zeros(8)
    ys = LOOP(y0, ZS, theta)
    return np.array(ys)


def fitness(params_vec):
    """Negative fitness (for minimization). Higher is worse."""
    try:
        ys = simulate(params_vec)
    except Exception:
        return 10.0  # penalty

    if not np.all(np.isfinite(ys)):
        return 10.0

    sp = ys[WARMUP:, 1]
    amplitude = np.std(sp)

    if amplitude < 1e-8:
        return 5.0  # no oscillation

    # Check all states bounded
    if np.max(np.abs(ys[WARMUP:, :4])) > 1e5:
        return 8.0  # blowup

    # PSD
    f, pxx = welch(sp, fs=FS, nperseg=1024)

    # Alpha band (8-13 Hz)
    alpha_mask = (f >= 8.0) & (f <= 13.0)
    total_power = np.sum(pxx[1:])
    alpha_power = np.sum(pxx[alpha_mask])
    alpha_frac = alpha_power / max(total_power, 1e-20)

    # Peak frequency
    peak_idx = np.argmax(pxx[1:]) + 1
    peak_freq = f[peak_idx]

    # Amplitude score (target: comparable to JR, std ~ 1-5 mV)
    amp_score = np.tanh(amplitude / 0.5)  # half-max at 0.5 mV

    # Frequency penalty: prefer peak in alpha
    if 8.0 <= peak_freq <= 13.0:
        freq_bonus = 1.0
    elif 5.0 <= peak_freq <= 20.0:
        freq_bonus = 0.5
    else:
        freq_bonus = 0.1

    # Laminar separation bonus (sp and dp should differ)
    dp = ys[WARMUP:, 3]
    if np.std(dp) > 1e-8:
        corr = abs(np.corrcoef(sp, dp)[0, 1])
        sep_bonus = 1.0 - corr  # reward decorrelation
    else:
        sep_bonus = 0.0

    # Combined (negate for minimization)
    score = alpha_frac * amp_score * freq_bonus * (0.7 + 0.3 * sep_bonus)
    return -score


# ── Optimization ─────────────────────────────────────────────────────

def run_scipy_de(budget=200, seed=42):
    """Differential Evolution (scipy) — no external deps needed."""
    print(f"Running scipy Differential Evolution (budget≈{budget})...")
    t0 = time.perf_counter()

    result = differential_evolution(
        fitness,
        bounds=BOUNDS_ARRAY,
        maxiter=budget // 15,  # DE does ~15 evals per iteration
        popsize=15,
        seed=seed,
        tol=1e-6,
        disp=True,
    )

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.1f}s, {result.nfev} evaluations")
    return result.x, -result.fun


def run_cmaes(budget=200, seed=42):
    """CMA-ES via cmaes package if available."""
    try:
        from cmaes import CMA
    except ImportError:
        print("cmaes not installed, falling back to scipy DE")
        return run_scipy_de(budget, seed)

    print(f"Running CMA-ES (budget={budget})...")
    t0 = time.perf_counter()

    # Start from current defaults
    x0 = np.array([getattr(vb.cmc_default_theta, k) for k in PARAM_NAMES])
    bounds_lo = np.array([b[0] for b in BOUNDS_ARRAY])
    bounds_hi = np.array([b[1] for b in BOUNDS_ARRAY])

    # Clip x0 to bounds
    x0 = np.clip(x0, bounds_lo, bounds_hi)
    sigma0 = np.mean(bounds_hi - bounds_lo) * 0.3

    optimizer = CMA(mean=x0, sigma=sigma0, bounds=np.array(BOUNDS_ARRAY),
                    seed=seed)

    best_x, best_score = None, float('inf')
    for gen in range(budget):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            score = fitness(x)
            solutions.append((x, score))
            if score < best_score:
                best_score = score
                best_x = x.copy()

        optimizer.tell(solutions)

        if gen % 10 == 0:
            print(f"  gen {gen:3d}  best={-best_score:.4f}  "
                  f"pop_best={-min(s for _, s in solutions):.4f}")

        if optimizer.should_stop():
            print(f"  CMA-ES converged at gen {gen}")
            break

    elapsed = time.perf_counter() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    return best_x, -best_score


# ── Analysis ─────────────────────────────────────────────────────────

def analyze_result(params_vec, score):
    """Print detailed analysis of optimized parameters."""
    params_dict = dict(zip(PARAM_NAMES, params_vec))

    print("\n═══ Optimized CMC Parameters ═══")
    print(f"Fitness score: {score:.4f}")
    print()
    print("cmc_tuned_theta = CMCTheta(")
    print(f"    He=3.25, Hi=22.0, a=0.1, b=0.05, r=0.56, v0=6.0, nu_max=0.0025,")
    for k, v in params_dict.items():
        comma = "," if k != "I" else ","
        print(f"    {k}={v:.1f}{comma}")
    print(")")

    # Simulate and analyze
    ys = simulate(params_vec)
    sp = ys[WARMUP:, 1]
    dp = ys[WARMUP:, 3]
    ss = ys[WARMUP:, 0]

    print(f"\nDynamics:")
    print(f"  sp amplitude (std): {np.std(sp):.4f}")
    print(f"  dp amplitude (std): {np.std(dp):.4f}")
    print(f"  ss amplitude (std): {np.std(ss):.4f}")
    print(f"  sp-dp correlation:  {np.corrcoef(sp, dp)[0,1]:.3f}")

    # Spectral analysis
    f, pxx = welch(sp, fs=FS, nperseg=1024)
    peak_idx = np.argmax(pxx[1:]) + 1
    alpha_mask = (f >= 8) & (f <= 13)
    alpha_frac = np.sum(pxx[alpha_mask]) / np.sum(pxx[1:])
    print(f"\nSpectral:")
    print(f"  Peak frequency:     {f[peak_idx]:.1f} Hz")
    print(f"  Alpha power frac:   {alpha_frac:.3f}")
    print(f"  Total power:        {np.sum(pxx[1:]):.2e}")

    # Compare with JR
    print(f"\nJR reference (defaults, I=220):")
    p_jr = vb.jr_default_theta._replace(I=220.0)
    _, jr_loop = vb.make_sde(dt=DT, dfun=lambda y, p: vb.jr_dfun(y, 0.0, p), gfun=SIGMA)
    jr_ys = jr_loop(jp.zeros(6), ZS[:, :6], p_jr)
    jr_sp = np.array(jr_ys[WARMUP:, 1] - jr_ys[WARMUP:, 2])
    f_jr, pxx_jr = welch(jr_sp, fs=FS, nperseg=1024)
    jr_peak = f_jr[np.argmax(pxx_jr[1:])+1]
    print(f"  JR amplitude (std): {np.std(jr_sp):.4f}")
    print(f"  JR peak frequency:  {jr_peak:.1f} Hz")

    # Layer activity for vpjax bridge
    layer_act = np.array(jax.vmap(vb.cmc_to_layer_activity)(jp.array(ys[WARMUP:])))
    print(f"\nLayer activity (vpjax bridge):")
    for i, name in enumerate(['deep (dp)', 'middle (ss)', 'superficial (sp)']):
        print(f"  {name}: mean={np.mean(layer_act[:, i]):.3f}, std={np.std(layer_act[:, i]):.4f}")

    return params_dict


if __name__ == "__main__":
    optimizer = "de"  # default
    budget = 150

    for arg in sys.argv[1:]:
        if arg == "--optimizer":
            idx = sys.argv.index(arg)
            optimizer = sys.argv[idx + 1]
        elif arg.startswith("--budget"):
            idx = sys.argv.index(arg)
            budget = int(sys.argv[idx + 1])
        elif arg == "llamea":
            optimizer = "llamea"

    if optimizer == "cmaes":
        best_x, best_score = run_cmaes(budget)
    elif optimizer == "llamea":
        print("LLaMEA requires neurojax CMCSpectralAdapter — use:")
        print("  from neurojax.bench.adapters.cmc_spectral_adapter import CMCSpectralAdapter")
        print("  from neurojax.bench.optimizers.llamea_wrapper import LLaMEAWrapper")
        sys.exit(0)
    else:
        best_x, best_score = run_scipy_de(budget)

    params = analyze_result(best_x, best_score)
