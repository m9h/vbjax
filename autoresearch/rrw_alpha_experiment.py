"""RRW alpha-band parameter search via Jacobian eigenvalue analysis.

Strategy: find parameter regimes where the linearized RRW has eigenvalues
with imaginary part near 2*pi*10 (alpha ~10 Hz) and real part just below
zero (near Hopf bifurcation = maximum resonance amplification).

This is the MUTABLE experiment for AgentSciML.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
import vbjax as vb
from scipy.signal import welch


def rrw_jacobian_eigenvalues(theta):
    """Compute eigenvalues of the RRW ODE Jacobian at the fixed point.

    Returns eigenvalues sorted by imaginary part (frequency).
    The alpha resonance appears when an eigenvalue pair has:
      - imag ≈ 2*pi*10 (≈ 63 rad/s for 10 Hz)
      - real just below 0 (damped but near instability)
    """
    y0 = jnp.array([v for v in vb.rrw_default_state])

    def f(y):
        return vb.rrw_dfun(y, 0.0, theta)

    J = jax.jacobian(f)(y0)
    eigs = np.linalg.eigvals(np.array(J))
    return eigs


def eigenvalue_alpha_score(eigs, target_hz=10.0):
    """Score how close the most alpha-like eigenvalue is to the target.

    Returns (score, best_freq_hz, best_damping).
    Higher score = better alpha candidate.
    """
    target_rad = 2 * np.pi * target_hz * 1e-3  # rad/ms (since dfun outputs ms^-1)

    best_score = 0.0
    best_freq = 0.0
    best_damp = 0.0

    for e in eigs:
        freq_rad = abs(e.imag)
        freq_hz = freq_rad / (2 * np.pi * 1e-3)
        damping = e.real

        if freq_hz < 1.0:  # skip DC modes
            continue

        # Score: high when freq is near target and damping is near zero (from below)
        freq_match = np.exp(-0.5 * ((freq_hz - target_hz) / 3.0)**2)
        # Damping: want real part close to 0 but negative (stable)
        # More negative = more damped = weaker resonance
        if damping > 0:
            damp_score = 0.5  # unstable — still informative
        else:
            damp_score = np.exp(damping * 50)  # e.g., -0.01 → 0.61, -0.1 → 0.007

        score = freq_match * damp_score

        if score > best_score:
            best_score = score
            best_freq = freq_hz
            best_damp = damping

    return best_score, best_freq, best_damp


def run_sdde_and_score(theta, dt=0.5, n_sim=15000, noise_sig=0.3, seed=42):
    """Run SDDE simulation and score the PSD for alpha content."""
    key = jax.random.PRNGKey(seed)
    y0 = jnp.array([v for v in vb.rrw_default_state])
    ns = len(y0)

    step, loop, nh, ds = vb.make_rrw_sdde(dt=dt, t0=theta.t0, gfun=noise_sig)
    buf = jnp.zeros((nh + n_sim, ns))
    for i in range(nh + 1):
        buf = buf.at[i].set(y0)
    buf = buf.at[nh+1:].set(jax.random.normal(key, (n_sim-1, ns)) * noise_sig)

    _, xs = loop(buf, (ds, 0.0, theta))
    phi = np.array(xs[5000:, 0])

    if not np.all(np.isfinite(phi)) or phi.std() < 0.01:
        return 0.0, 0.0

    fs = 1000.0 / dt
    f, psd = welch(phi, fs=fs, nperseg=2048)
    m = (f > 1) & (f < 50)
    peak = f[m][np.argmax(psd[m])]
    alpha = psd[(f > 8) & (f < 13)].mean()
    total = psd[m].mean()

    return float(alpha / (total + 1e-20)), float(peak)


def run_experiment():
    """Two-phase search: eigenvalue analysis then SDDE validation."""
    base = vb.rrw_default_theta

    print("=" * 60)
    print("Phase 1: Jacobian eigenvalue analysis (fast)")
    print("=" * 60)

    # Sweep parameters that affect the loop gain and resonance
    results = []

    for nu_ee in np.linspace(0.5, 4.0, 8):
        for nu_ei in np.linspace(-6.0, -1.0, 6):
            for nu_se in np.linspace(0.5, 5.0, 6):
                for nu_es in np.linspace(0.2, 2.0, 5):
                    theta = base._replace(
                        nu_ee=float(nu_ee), nu_ei=float(nu_ei),
                        nu_se=float(nu_se), nu_es=float(nu_es))

                    try:
                        eigs = rrw_jacobian_eigenvalues(theta)
                        score, freq, damp = eigenvalue_alpha_score(eigs)
                    except Exception:
                        continue

                    if score > 0.01:
                        results.append((score, freq, damp, nu_ee, nu_ei, nu_se, nu_es))

    results.sort(key=lambda x: -x[0])
    print(f"\nFound {len(results)} candidates with score > 0.01")
    print(f"{'Score':>8s} {'Freq':>6s} {'Damp':>8s} {'nu_ee':>8s} {'nu_ei':>8s} {'nu_se':>8s} {'nu_es':>8s}")

    for score, freq, damp, nee, nei, nse, nes in results[:20]:
        tag = " <--" if 8 < freq < 13 else ""
        print(f"{score:8.4f} {freq:6.1f} {damp:8.5f} {nee:8.3f} {nei:8.3f} {nse:8.3f} {nes:8.3f}{tag}",
              flush=True)

    if not results:
        print("No alpha candidates found. Try wider parameter ranges.")
        print("RESULT|model=RRW_alpha|alpha_score=0.0|peak=0.0|status=no_candidates")
        return

    print(f"\n{'=' * 60}")
    print(f"Phase 2: SDDE validation of top {min(10, len(results))} candidates")
    print(f"{'=' * 60}")

    best_sdde_score = 0
    best_config = None

    for score, freq, damp, nee, nei, nse, nes in results[:10]:
        theta = base._replace(
            nu_ee=float(nee), nu_ei=float(nei),
            nu_se=float(nse), nu_es=float(nes))

        alpha_ratio, peak = run_sdde_and_score(theta)
        tag = " <-- ALPHA!" if 8 < peak < 13 and alpha_ratio > 0.3 else ""
        print(f"  eig_freq={freq:.1f} -> SDDE peak={peak:.1f} Hz, "
              f"alpha_ratio={alpha_ratio:.3f}{tag}", flush=True)

        if alpha_ratio > best_sdde_score:
            best_sdde_score = alpha_ratio
            best_config = dict(nu_ee=nee, nu_ei=nei, nu_se=nse, nu_es=nes,
                              eig_freq=freq, sdde_peak=peak)

    if best_config:
        print(f"\nBest SDDE config: {best_config}")
        print(f"RESULT|model=RRW_alpha|alpha_score={best_sdde_score:.6f}"
              f"|peak={best_config['sdde_peak']:.1f}"
              f"|eig_freq={best_config['eig_freq']:.1f}"
              f"|nu_ee={best_config['nu_ee']:.3f}"
              f"|nu_ei={best_config['nu_ei']:.3f}"
              f"|nu_se={best_config['nu_se']:.3f}"
              f"|nu_es={best_config['nu_es']:.3f}")
    else:
        print("No valid SDDE results")
        print("RESULT|model=RRW_alpha|alpha_score=0.0|peak=0.0|status=no_valid_sdde")


if __name__ == "__main__":
    run_experiment()
