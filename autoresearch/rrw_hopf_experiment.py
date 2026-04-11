"""RRW Hopf bifurcation boundary search for alpha oscillations.

Goal: find parameter configurations where the RRW SDDE model produces
a clear alpha peak (8-13 Hz) in the time-domain PSD.

Strategy: eigenvalue analysis to find parameters where the alpha mode
has damping very close to zero, then validate with SDDE simulation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
import vbjax as vb
from scipy.signal import welch

sys.stdout.reconfigure(line_buffering=True)


def rrw_eigenvalues(theta):
    y0 = jnp.array([v for v in vb.rrw_default_state])
    J = jax.jacobian(lambda y: vb.rrw_dfun(y, 0.0, theta))(y0)
    return np.linalg.eigvals(np.array(J))


def find_alpha_mode(eigs, target_hz=10.0):
    best_dist = 1e10
    best_eig = None
    for e in eigs:
        freq_hz = abs(e.imag) / (2 * np.pi * 1e-3)
        if freq_hz < 1.0:
            continue
        dist = abs(freq_hz - target_hz)
        if dist < best_dist:
            best_dist = dist
            best_eig = e
    if best_eig is None:
        return 0.0, 0.0
    return abs(best_eig.imag) / (2 * np.pi * 1e-3), best_eig.real


def run_sdde(theta, noise_sig=0.3, n_sim=20000, seed=42):
    dt = 0.5
    fs = 1000.0 / dt
    key = jax.random.PRNGKey(seed)
    y0 = jnp.array([v for v in vb.rrw_default_state])
    ns = len(y0)
    step, loop, nh, ds = vb.make_rrw_sdde(dt=dt, t0=theta.t0, gfun=noise_sig)
    buf = jnp.zeros((nh + n_sim, ns))
    for i in range(nh + 1):
        buf = buf.at[i].set(y0)
    buf = buf.at[nh+1:].set(jax.random.normal(key, (n_sim-1, ns)) * noise_sig)
    _, xs = loop(buf, (ds, 0.0, theta))
    phi = np.array(xs[8000:, 0])
    if not np.all(np.isfinite(phi)) or phi.std() < 0.001:
        return 0.0, 0.0, 0.0
    f, p = welch(phi, fs=fs, nperseg=4096)
    m = (f > 1) & (f < 50)
    peak = f[m][np.argmax(p[m])]
    alpha = p[(f > 8) & (f < 13)].mean()
    total = p[m].mean()
    return float(peak), float(alpha / (total + 1e-20)), float(phi.std())


def main():
    base = vb.rrw_default_theta
    print("=== RRW Hopf Boundary Search ===\n")

    # Phase 1: Fine eigenvalue search targeting damping -> 0
    print("Phase 1: Eigenvalue search (damping in [-0.005, 0.001])")
    candidates = []

    for nu_ee in np.linspace(0.5, 5.0, 10):
        for nu_ei in np.linspace(-5.0, -0.5, 10):
            for nu_se in np.linspace(0.2, 4.0, 8):
                for nu_es in np.linspace(0.2, 3.0, 7):
                    theta = base._replace(
                        nu_ee=float(nu_ee), nu_ei=float(nu_ei),
                        nu_se=float(nu_se), nu_es=float(nu_es))
                    try:
                        eigs = rrw_eigenvalues(theta)
                        freq, damp = find_alpha_mode(eigs)
                    except:
                        continue
                    if 7 < freq < 14 and -0.005 < damp < 0.001:
                        candidates.append((abs(damp), freq, damp,
                                          nu_ee, nu_ei, nu_se, nu_es))

    candidates.sort()
    print(f"  Found {len(candidates)} near-critical alpha modes")

    if candidates:
        print(f"  Top 20:")
        print(f"  {'|damp|':>8s} {'freq':>6s} {'damp':>10s} {'nu_ee':>8s} {'nu_ei':>8s} {'nu_se':>8s} {'nu_es':>8s}")
        for ad, freq, damp, nee, nei, nse, nes in candidates[:20]:
            print(f"  {ad:8.6f} {freq:6.1f} {damp:10.7f} {nee:8.3f} {nei:8.3f} {nse:8.3f} {nes:8.3f}")

    # Phase 2: SDDE validation
    print(f"\nPhase 2: SDDE validation of top {min(20, len(candidates))}")
    best = None

    for ad, freq, damp, nee, nei, nse, nes in candidates[:20]:
        theta = base._replace(nu_ee=float(nee), nu_ei=float(nei),
                              nu_se=float(nse), nu_es=float(nes))
        peak, afrac, std = run_sdde(theta)
        tag = " <-- ALPHA!" if 8 <= peak <= 13 else ""
        print(f"  damp={damp:.6f} eig={freq:.1f}Hz -> peak={peak:.1f}Hz "
              f"alpha={afrac:.3f} std={std:.3f}{tag}")

        if best is None or afrac > best[0]:
            best = (afrac, peak, freq, damp, nee, nei, nse, nes, std)

    if best:
        print(f"\nBest: alpha_frac={best[0]:.3f}, peak={best[1]:.1f}Hz, "
              f"eig_freq={best[2]:.1f}Hz, damp={best[3]:.6f}")
        print(f"  nu_ee={best[4]:.3f}, nu_ei={best[5]:.3f}, "
              f"nu_se={best[6]:.3f}, nu_es={best[7]:.3f}")
        print(f"RESULT|model=RRW_SDDE|alpha_frac={best[0]:.6f}|peak={best[1]:.1f}"
              f"|eig_freq={best[2]:.1f}|damp={best[3]:.6f}")


if __name__ == "__main__":
    main()
