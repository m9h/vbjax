#!/usr/bin/env python3
"""Bayesian Model Comparison on DGX Spark (128GB unified memory).

Compares 4 neural mass model families against synthetic EEG:
  Liley (14D), CMC (8D), RRW (8D), CBEI (8D)

Uses full Laplace free energy (exact Hessian via jax.hessian),
vmap over subjects, longer simulations, and more free parameters
than the RTX 2080 version.

Usage:
    python scripts/dgx_bms.py                    # full comparison
    python scripts/dgx_bms.py --quick             # quick test (2 subjects, 50 opt steps)
    python scripts/dgx_bms.py --pharma            # include pharmacological perturbation
    python scripts/dgx_bms.py --output results/   # save results to directory

References:
    Friston et al. (2007) Variational free energy and the Laplace approximation.
    Stephan et al. (2009) Bayesian model selection for group studies.
    Rigoux et al. (2014) Bayesian model selection for group studies - Revisited.
"""

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Must set before importing vbjax (which sets XLA_FLAGS)
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_deterministic_ops=false')

import vbjax as vb
from vbjax.spectral import welch_psd_jax, spectral_loss
from vbjax.bms import bms_ffx, bms_rfx, laplace_free_energy


# =====================================================================
# Configuration
# =====================================================================

def get_config(quick=False):
    if quick:
        return dict(
            dt_s=0.5e-3, n_steps=10_000, n_warmup=2_000, nperseg=512,
            n_subjects=2, n_opt_steps=50, lr=0.02,
            n_free_params=2,
        )
    return dict(
        dt_s=0.1e-3,       # 0.1 ms -> 10 kHz sampling
        n_steps=100_000,    # 10 seconds of simulated EEG
        n_warmup=20_000,    # 2 seconds warmup
        nperseg=2048,       # ~5 Hz spectral resolution
        n_subjects=12,      # 12 synthetic subjects
        n_opt_steps=200,    # Adam iterations
        lr=0.01,
        n_free_params=4,    # 4 free params per model
    )


# =====================================================================
# Model definitions
# =====================================================================

def make_liley_dfun(n_params):
    """Liley model with n_params free parameters (log-deviations)."""
    def dfun(ys, theta):
        replacements = {}
        if n_params >= 1:
            replacements['p_ee'] = jnp.exp(theta[0]) * 1.0
        if n_params >= 2:
            replacements['sigma_e'] = jnp.exp(theta[1]) * 5.0
        if n_params >= 3:
            replacements['Gamma_e'] = jnp.exp(theta[2]) * 0.3
        if n_params >= 4:
            replacements['gamma_i'] = jnp.exp(theta[3]) * 65.0
        p = vb.liley_default_theta._replace(**replacements)
        return vb.liley_dfun(ys, 0.0, p)
    return dfun


def make_cmc_dfun(n_params):
    """CMC model with n_params free parameters."""
    def dfun(ys, theta):
        replacements = {}
        if n_params >= 1:
            replacements['I'] = jnp.exp(theta[0]) * 362.5
        if n_params >= 2:
            replacements['g_ss_sp'] = jnp.exp(theta[1]) * 86.3
        if n_params >= 3:
            replacements['He'] = jnp.exp(theta[2]) * 3.25
        if n_params >= 4:
            replacements['Hi'] = jnp.exp(theta[3]) * 22.0
        p = vb.cmc_default_theta._replace(**replacements)
        return vb.cmc_dfun(ys, 0.0, p)
    return dfun


def make_rrw_dfun(n_params):
    """Robinson-Rennie-Wright with n_params free parameters."""
    def dfun(ys, theta):
        replacements = {}
        if n_params >= 1:
            replacements['I'] = jnp.exp(theta[0]) * 1.0
        if n_params >= 2:
            replacements['nu_ee'] = jnp.exp(theta[1]) * 1.0
        if n_params >= 3:
            replacements['nu_es'] = jnp.exp(theta[2]) * 1.0
        if n_params >= 4:
            replacements['gamma_e'] = jnp.exp(theta[3]) * 100.0
        p = vb.rrw_default_theta._replace(**replacements)
        return vb.rrw_dfun(ys, 0.0, p)
    return dfun


def make_cbei_dfun(n_params):
    """Coombes-Byrne E-I with n_params free parameters."""
    def dfun(ys, theta):
        replacements = {}
        if n_params >= 1:
            replacements['I'] = jnp.exp(theta[0]) * 3.0
        if n_params >= 2:
            replacements['kappa_ee'] = jnp.exp(theta[1]) * 10.0
        if n_params >= 3:
            replacements['kappa_ie'] = jnp.exp(theta[2]) * 10.0
        if n_params >= 4:
            replacements['tau_s_e'] = jnp.exp(theta[3]) * 5.0
        p = vb.cbei_default_theta._replace(**replacements)
        return vb.cbei_dfun(ys, 0.0, p)
    return dfun


def make_liley_pharma_dfun(n_params):
    """Liley with pharmacological extension — for perturbation paradigm."""
    def dfun(ys, theta):
        replacements = {}
        if n_params >= 1:
            replacements['p_ee'] = jnp.exp(theta[0]) * 1.0
        if n_params >= 2:
            replacements['sigma_e'] = jnp.exp(theta[1]) * 5.0
        if n_params >= 3:
            replacements['c_drug'] = jnp.sigmoid(theta[2]) * 2.0
        if n_params >= 4:
            replacements['rho_amp'] = jnp.exp(theta[3]) * 1.5
        p = vb.liley_pharma_propofol_theta._replace(**replacements)
        return vb.liley_pharma_dfun(ys, 0.0, p)
    return dfun


# =====================================================================
# Core fitting engine
# =====================================================================

def make_forward_psd(dfun, n_states, noise_sigma, dt_s, n_steps,
                     n_warmup, nperseg, band_start, n_freq_bins, noise_key):
    """Build a differentiable forward PSD prediction function."""
    _, sde_loop = vb.make_sde(dt_s, dfun, noise_sigma)
    fs = 1.0 / dt_s
    zs = jax.random.normal(noise_key, (n_steps, n_states))
    x0 = jnp.zeros(n_states)

    def forward_psd(theta):
        xs = sde_loop(x0, zs, theta)
        obs = xs[n_warmup:, 0]
        _, psd_full = welch_psd_jax(obs, fs, nperseg=nperseg)
        return jax.lax.dynamic_slice(psd_full, (band_start,), (n_freq_bins,))

    return forward_psd


def fit_single(forward_psd, target_psd, prior_mean, prior_std,
               n_opt_steps, lr):
    """MAP estimation via Adam + Laplace free energy."""
    k = prior_mean.shape[0]

    def neg_log_joint(theta):
        pred = forward_psd(theta)
        eps = 1e-10
        nll = jnp.sum((jnp.log(pred + eps) - jnp.log(target_psd + eps))**2)
        nlp = 0.5 * jnp.sum(((theta - prior_mean) / prior_std)**2)
        return nll + nlp

    # MAP: Adam
    vg = jax.value_and_grad(neg_log_joint)
    theta = jnp.zeros(k)
    m, v = jnp.zeros(k), jnp.zeros(k)

    def adam_step(carry, _):
        theta, m, v, step = carry
        loss, grad = vg(theta)
        grad = jnp.clip(grad, -10.0, 10.0)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        mh = m / (1 - 0.9**(step + 1))
        vh = v / (1 - 0.999**(step + 1))
        theta = theta - lr * mh / (jnp.sqrt(vh) + 1e-8)
        return (theta, m, v, step + 1), loss

    init = (theta, m, v, jnp.array(1.0))
    (theta, _, _, _), losses = jax.lax.scan(adam_step, init, None, length=n_opt_steps)

    # Laplace free energy with exact Hessian
    nlj_map = neg_log_joint(theta)
    H = jax.hessian(neg_log_joint)(theta)
    sign, logdet = jnp.linalg.slogdet(H)
    F = -nlj_map + 0.5 * k * jnp.log(2 * jnp.pi) - 0.5 * logdet
    F = jnp.where(sign > 0, F, jnp.array(-1e10))

    # Posterior covariance
    post_cov = jnp.linalg.inv(H + 1e-6 * jnp.eye(k))

    return {
        'theta_map': theta,
        'free_energy': F,
        'neg_log_joint': nlj_map,
        'posterior_cov': post_cov,
        'loss_final': losses[-1],
        'loss_trace': losses,
    }


# =====================================================================
# Main comparison
# =====================================================================

def run_comparison(cfg, include_pharma=False):
    """Run full Bayesian model comparison."""
    dt_s = cfg['dt_s']
    fs = 1.0 / dt_s
    n_steps = cfg['n_steps']
    n_warmup = cfg['n_warmup']
    nperseg = cfg['nperseg']
    n_subjects = cfg['n_subjects']
    n_opt_steps = cfg['n_opt_steps']
    lr = cfg['lr']
    k = cfg['n_free_params']

    # Precompute frequency grid
    _freqs = np.fft.rfftfreq(nperseg, d=dt_s)
    _band = np.where((_freqs >= 1.0) & (_freqs <= 100.0))[0]
    band_start = int(_band[0])
    band_end = int(_band[-1]) + 1
    n_freq_bins = band_end - band_start
    target_freqs = _freqs[band_start:band_end]

    print(f"  PSD: {n_freq_bins} bins, {target_freqs[0]:.1f}-{target_freqs[-1]:.1f} Hz")
    print(f"  Simulation: {n_steps} steps @ dt={dt_s*1e3:.1f} ms = {n_steps*dt_s:.1f}s")
    print(f"  Free params per model: {k}")

    # ── Generate synthetic EEG from Liley ────────────────────────────
    print("\n--- Generating ground truth (Liley model) ---")

    def liley_gt(ys, p):
        return vb.liley_dfun(ys, 0.0, p)

    _, gt_loop = vb.make_sde(dt_s, liley_gt, 0.5e-3)
    y0_liley = jnp.array([v for v in vb.liley_default_state])

    t0 = time.perf_counter()
    subject_psds = []
    for i in range(n_subjects):
        sk = jax.random.PRNGKey(i)
        zs = jax.random.normal(sk, (n_steps, 14))
        xs = gt_loop(y0_liley, zs, vb.liley_default_theta)
        obs = xs[n_warmup:, 0]
        _, psd_full = welch_psd_jax(obs, fs, nperseg=nperseg)
        psd_band = jax.lax.dynamic_slice(psd_full, (band_start,), (n_freq_bins,))
        subject_psds.append(psd_band)

    subject_psds = jnp.stack(subject_psds)
    t_gen = time.perf_counter() - t0
    peak_idx = int(jnp.argmax(subject_psds.mean(axis=0)))
    print(f"  Generated {n_subjects} subjects in {t_gen:.1f}s")
    print(f"  Mean peak: {target_freqs[peak_idx]:.1f} Hz")

    # ── Pharmacological perturbation data (optional) ─────────────────
    pharma_psds = None
    if include_pharma:
        print("\n--- Generating pharmacological perturbation (propofol) ---")
        def liley_pharma_gt(ys, p):
            return vb.liley_pharma_dfun(ys, 0.0, p)

        _, pharma_loop = vb.make_sde(dt_s, liley_pharma_gt, 0.5e-3,
                                      adhoc=vb.liley_adhoc)
        p_propofol = vb.liley_pharma_propofol_theta._replace(c_drug=0.8)

        pharma_psds_list = []
        for i in range(n_subjects):
            sk = jax.random.PRNGKey(100 + i)
            zs = jax.random.normal(sk, (n_steps, 14))
            xs = pharma_loop(y0_liley, zs, p_propofol)
            obs = xs[n_warmup:, 0]
            _, psd_full = welch_psd_jax(obs, fs, nperseg=nperseg)
            psd_band = jax.lax.dynamic_slice(psd_full, (band_start,), (n_freq_bins,))
            pharma_psds_list.append(psd_band)

        pharma_psds = jnp.stack(pharma_psds_list)
        print(f"  Generated {n_subjects} pharma subjects")

    # ── Define candidate models ──────────────────────────────────────
    model_specs = [
        ('Liley',  make_liley_dfun(k),  14, 0.5e-3),
        ('CMC',    make_cmc_dfun(k),     8, 1e-3),
        ('RRW',    make_rrw_dfun(k),     8, 0.1e-3),
        ('CBEI',   make_cbei_dfun(k),    8, 0.01e-3),
    ]
    if include_pharma:
        model_specs.append(
            ('LileyPh', make_liley_pharma_dfun(k), 14, 0.5e-3))

    names = [s[0] for s in model_specs]
    n_models = len(model_specs)

    prior_mean = jnp.zeros(k)
    prior_std = jnp.ones(k) * 1.0  # broad prior

    # ── Fit each model to each subject ───────────────────────────────
    conditions = [('baseline', subject_psds)]
    if pharma_psds is not None:
        conditions.append(('propofol', pharma_psds))

    all_results = {}

    for cond_name, cond_psds in conditions:
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond_name}")
        print(f"{'='*60}")

        free_energies = np.zeros((n_subjects, n_models))

        for j, (mname, dfun, n_st, noise_sig) in enumerate(model_specs):
            t0 = time.perf_counter()
            print(f"\n  Fitting {mname} ({n_st}D, {k} free params)...")

            for i in range(n_subjects):
                noise_key = jax.random.PRNGKey(1000 + i)
                target_psd_i = cond_psds[i]

                fwd = make_forward_psd(
                    dfun, n_st, noise_sig, dt_s, n_steps, n_warmup,
                    nperseg, band_start, n_freq_bins, noise_key)

                result = fit_single(fwd, target_psd_i, prior_mean, prior_std,
                                    n_opt_steps, lr)

                free_energies[i, j] = float(result['free_energy'])

                if i == 0:
                    print(f"    s0: F={float(result['free_energy']):.2f}, "
                          f"loss={float(result['neg_log_joint']):.2f}, "
                          f"theta={np.array(result['theta_map']).round(3).tolist()}")

            dt_m = time.perf_counter() - t0
            mean_F = free_energies[:, j].mean()
            print(f"    {n_subjects} subjects in {dt_m:.1f}s, mean F={mean_F:.2f}")

        all_results[cond_name] = {
            'free_energies': free_energies.tolist(),
            'model_names': names,
        }

        # ── BMS ──────────────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print(f"BMS RESULTS: {cond_name}")
        print(f"{'─'*60}")

        # Table
        hdr = "          " + "".join(f"  {n:>10s}" for n in names)
        print(hdr)
        for i in range(n_subjects):
            row = f"  subj {i:2d}:" + "".join(
                f"  {free_energies[i,j]:>10.2f}" for j in range(n_models))
            best = names[np.argmax(free_energies[i])]
            print(f"{row}  <- {best}")

        # FFX
        post_prob, winner = bms_ffx(jnp.array(free_energies))
        print(f"\n  Fixed-effects BMS:")
        for j, nm in enumerate(names):
            print(f"    {nm:8s}: sum F = {free_energies[:,j].sum():>12.2f}, "
                  f"p(m|y) = {float(post_prob[j]):.6f}")
        print(f"    Winner: {names[int(winner)]}")

        all_results[cond_name]['ffx_winner'] = names[int(winner)]
        all_results[cond_name]['ffx_probs'] = [float(p) for p in post_prob]

        # RFX
        rfx = bms_rfx(jnp.array(free_energies))
        print(f"\n  Random-effects BMS (Stephan et al. 2009):")
        print(f"    {'Model':8s}  {'Exp freq':>10s}  {'XP':>10s}  {'PXP':>10s}")
        for j, nm in enumerate(names):
            print(f"    {nm:8s}  {float(rfx['exp_r'][j]):>10.4f}  "
                  f"{float(rfx['exceedance_prob'][j]):>10.4f}  "
                  f"{float(rfx['protected_exceedance_prob'][j]):>10.4f}")

        all_results[cond_name]['rfx_exp_r'] = [float(r) for r in rfx['exp_r']]
        all_results[cond_name]['rfx_xp'] = [float(x) for x in rfx['exceedance_prob']]
        all_results[cond_name]['rfx_pxp'] = [float(x) for x in rfx['protected_exceedance_prob']]

        # Bayes factors
        sums = free_energies.sum(axis=0)
        order = np.argsort(-sums)
        print(f"\n  Model ranking:")
        for rank, idx in enumerate(order):
            bf = sums[idx] - sums[order[-1]]
            print(f"    {rank+1}. {names[idx]:8s}: "
                  f"sum F = {sums[idx]:.1f}, log BF vs worst = {bf:.1f}")

    return all_results, target_freqs


# =====================================================================
# Entry point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Bayesian Model Comparison on DGX Spark')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--pharma', action='store_true',
                        help='Include pharmacological perturbation condition')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results JSON')
    args = parser.parse_args()

    print("=" * 60)
    print("vbjax Bayesian Model Comparison")
    print("=" * 60)
    print(f"Device: {jax.devices()[0]}")
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print(f"Pharma: {args.pharma}")

    cfg = get_config(quick=args.quick)
    print(f"Config: {cfg}")

    t_start = time.perf_counter()
    results, target_freqs = run_comparison(cfg, include_pharma=args.pharma)
    t_total = time.perf_counter() - t_start

    print(f"\n{'='*60}")
    print(f"Total wall time: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"{'='*60}")

    # Save results
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'bms_results.json'
        results['config'] = cfg
        results['target_freqs'] = target_freqs.tolist()
        results['wall_time_s'] = t_total
        results['device'] = str(jax.devices()[0])
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
