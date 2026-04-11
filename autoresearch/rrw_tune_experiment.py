"""RRW SDDE parameter tuning: find alpha-band regime.

The RRW corticothalamic model with delay should produce ~10-12 Hz alpha
from the t0~85ms loop delay. Current NFTsim defaults give 21.5 Hz peak.
Sweep parameters to find the alpha regime.

This is a standalone experiment — run directly, no AgentSciML needed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
import vbjax as vb
from scipy.signal import welch


def score_alpha(phi_e, fs, target_freq=10.0):
    """Score how close the peak PSD frequency is to the target alpha."""
    freqs, psd = welch(phi_e, fs=fs, nperseg=1024)
    mask = (freqs > 1) & (freqs < 50)
    if psd[mask].max() == 0:
        return 0.0, 0.0
    peak = freqs[mask][np.argmax(psd[mask])]
    alpha_power = psd[(freqs > 8) & (freqs < 13)].mean()
    total_power = psd[mask].mean()
    # Score: high when peak is near target and alpha dominates
    freq_score = np.exp(-0.5 * ((peak - target_freq) / 2.0)**2)
    alpha_ratio = alpha_power / (total_power + 1e-20)
    return float(freq_score * alpha_ratio), float(peak)


def run_rrw_sdde(theta, dt=0.5, n_sim=10000, noise_sig=0.5, seed=42):
    """Run RRW SDDE and return phi_e time series."""
    key = jax.random.PRNGKey(seed)
    step, loop, nh, ds = vb.make_rrw_sdde(dt=dt, t0=theta.t0, gfun=noise_sig)
    y0 = jnp.array([v for v in vb.rrw_default_state])

    buf = jnp.zeros((nh + n_sim, 8))
    for i in range(nh + 1):
        buf = buf.at[i].set(y0)
    buf = buf.at[nh+1:].set(jax.random.normal(key, (n_sim-1, 8)) * noise_sig)

    p = (ds, 0.0, theta)
    _, xs = loop(buf, p)
    return np.array(xs[3000:, 0])  # discard transient


def main():
    dt = 0.5
    fs = 1000.0 / dt
    base = vb.rrw_default_theta

    print("=== RRW SDDE Alpha-Band Parameter Search ===\n")

    # Sweep key parameters
    best_score = 0
    best_config = None

    # gamma_e controls cortical damping — lower = slower oscillation
    # nu_se/nu_es control thalamocortical gain — key for loop resonance
    # alpha/beta control synaptic filter — affect resonant frequency

    results = []

    for gamma_e in [50, 75, 100, 116, 150]:
        for alpha_val in [50, 60, 83.33]:
            for nu_se in [1.0, 2.0, 3.447, 5.0]:
                for nu_es in [0.3, 0.567, 1.0, 2.0]:
                    theta = base._replace(
                        gamma_e=gamma_e,
                        alpha=alpha_val,
                        nu_se=nu_se,
                        nu_es=nu_es,
                    )

                    try:
                        phi_e = run_rrw_sdde(theta)
                        if not np.all(np.isfinite(phi_e)):
                            continue
                        score, peak = score_alpha(phi_e, fs)
                    except Exception:
                        continue

                    results.append((score, peak, gamma_e, alpha_val, nu_se, nu_es))

                    if score > best_score:
                        best_score = score
                        best_config = (gamma_e, alpha_val, nu_se, nu_es)
                        print(f"  NEW BEST: score={score:.4f}, peak={peak:.1f} Hz, "
                              f"gamma_e={gamma_e}, alpha={alpha_val}, "
                              f"nu_se={nu_se}, nu_es={nu_es}")

    # Sort and show top 10
    results.sort(key=lambda x: -x[0])
    print(f"\n=== Top 10 configurations ===")
    print(f"{'Score':>8s} {'Peak':>6s} {'gamma_e':>8s} {'alpha':>8s} {'nu_se':>8s} {'nu_es':>8s}")
    for score, peak, ge, al, nse, nes in results[:10]:
        marker = " <--" if 8 < peak < 14 else ""
        print(f"{score:8.4f} {peak:6.1f} {ge:8.1f} {al:8.2f} {nse:8.3f} {nes:8.3f}{marker}")

    if best_config:
        print(f"\nBest: gamma_e={best_config[0]}, alpha={best_config[1]}, "
              f"nu_se={best_config[2]}, nu_es={best_config[3]}")

        # Print RESULT line for AgentSciML compatibility
        print(f"\nRESULT|model=RRW_SDDE|score={best_score:.6f}|peak={results[0][1]:.1f}")


if __name__ == "__main__":
    main()
