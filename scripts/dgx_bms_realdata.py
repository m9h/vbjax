#!/usr/bin/env python3
"""Bayesian Model Comparison on real Hartoyo et al. EEG data.

Fits 4 neural mass models (Liley, CMC, RRW, CBEI) to 82 subjects of
resting-state eyes-closed EEG from Hartoyo et al. (2019), using the
exact Laplace free energy for Bayesian model selection.

Optionally includes alpha-blocking (eyes-open) condition from
Hartoyo et al. (2020) for perturbation-based model comparison.

Usage:
    python scripts/dgx_bms_realdata.py                  # resting only
    python scripts/dgx_bms_realdata.py --alpha-blocking  # + eyes-open
    python scripts/dgx_bms_realdata.py --n-subjects 20   # subset
    python scripts/dgx_bms_realdata.py --quick            # 10 subjects, 50 steps
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import vbjax as vb
from vbjax.hartoyo import load_resting_spectra, load_alpha_blocking_spectra
from vbjax.spectral import welch_psd_jax
from vbjax.bms import bms_ffx, bms_rfx


# =====================================================================
# Model definitions — spectral domain fitting
# =====================================================================
# The Hartoyo data is already PSD (not time series), so we fit in the
# spectral domain: simulate -> compute PSD -> compare to observed PSD.
# The freq grid is 2-19.75 Hz at 0.25 Hz resolution.

def make_liley_spectral(free_params, dt_s, n_steps, n_warmup, nperseg,
                        target_freqs, noise_sigma=0.5e-3):
    """Build Liley forward PSD predictor matched to Hartoyo freq grid."""
    defaults = {name: getattr(vb.liley_default_theta, name) for name in free_params}

    def dfun(ys, theta):
        replacements = {}
        for i, name in enumerate(free_params):
            replacements[name] = jnp.exp(theta[i]) * defaults[name]
        p = vb.liley_default_theta._replace(**replacements)
        return vb.liley_dfun(ys, 0.0, p)

    _, sde_loop = vb.make_sde(dt_s, dfun, noise_sigma)
    fs = 1.0 / dt_s

    # Precompute freq grid mapping
    sim_freqs = np.fft.rfftfreq(nperseg, d=dt_s)
    # Find nearest sim freq for each target freq
    target_indices = np.array([np.argmin(np.abs(sim_freqs - tf)) for tf in target_freqs])

    return sde_loop, target_indices, 14


def make_cmc_spectral(free_params, dt_s, n_steps, n_warmup, nperseg,
                      target_freqs, noise_sigma=1e-3):
    defaults = {name: getattr(vb.cmc_default_theta, name) for name in free_params}

    def dfun(ys, theta):
        replacements = {}
        for i, name in enumerate(free_params):
            replacements[name] = jnp.exp(theta[i]) * defaults[name]
        p = vb.cmc_default_theta._replace(**replacements)
        return vb.cmc_dfun(ys, 0.0, p)

    _, sde_loop = vb.make_sde(dt_s, dfun, noise_sigma)
    fs = 1.0 / dt_s
    sim_freqs = np.fft.rfftfreq(nperseg, d=dt_s)
    target_indices = np.array([np.argmin(np.abs(sim_freqs - tf)) for tf in target_freqs])

    return sde_loop, target_indices, 8


def make_rrw_spectral(free_params, dt_s, n_steps, n_warmup, nperseg,
                      target_freqs, noise_sigma=0.1e-3):
    defaults = {name: getattr(vb.rrw_default_theta, name) for name in free_params}

    def dfun(ys, theta):
        replacements = {}
        for i, name in enumerate(free_params):
            replacements[name] = jnp.exp(theta[i]) * defaults[name]
        p = vb.rrw_default_theta._replace(**replacements)
        return vb.rrw_dfun(ys, 0.0, p)

    _, sde_loop = vb.make_sde(dt_s, dfun, noise_sigma)
    sim_freqs = np.fft.rfftfreq(nperseg, d=dt_s)
    target_indices = np.array([np.argmin(np.abs(sim_freqs - tf)) for tf in target_freqs])

    return sde_loop, target_indices, 8


def make_cbei_spectral(free_params, dt_s, n_steps, n_warmup, nperseg,
                       target_freqs, noise_sigma=0.01e-3):
    defaults = {name: getattr(vb.cbei_default_theta, name) for name in free_params}

    def dfun(ys, theta):
        replacements = {}
        for i, name in enumerate(free_params):
            replacements[name] = jnp.exp(theta[i]) * defaults[name]
        p = vb.cbei_default_theta._replace(**replacements)
        return vb.cbei_dfun(ys, 0.0, p)

    _, sde_loop = vb.make_sde(dt_s, dfun, noise_sigma)
    sim_freqs = np.fft.rfftfreq(nperseg, d=dt_s)
    target_indices = np.array([np.argmin(np.abs(sim_freqs - tf)) for tf in target_freqs])

    return sde_loop, target_indices, 8


# =====================================================================
# Fitting engine
# =====================================================================

def fit_to_spectrum(sde_loop, n_states, target_indices, target_psd,
                    dt_s, n_steps, n_warmup, nperseg, k,
                    noise_key, n_opt_steps=200, lr=0.05, use_bic=False):
    """Fit a model to a single observed PSD. Returns (F, loss, theta)."""
    fs = 1.0 / dt_s
    zs = jax.random.normal(noise_key, (n_steps, n_states))
    x0 = jnp.zeros(n_states)
    target_indices_jnp = jnp.array(target_indices)

    prior_mean = jnp.zeros(k)
    prior_std = jnp.ones(k) * 1.5

    def forward_psd(theta):
        xs = sde_loop(x0, zs, theta)
        obs = xs[n_warmup:, 0]
        _, psd_full = welch_psd_jax(obs, fs, nperseg=nperseg)
        return psd_full[target_indices_jnp]

    def neg_log_joint(theta):
        pred = forward_psd(theta)
        eps = 1e-10
        nll = jnp.sum((jnp.log(pred + eps) - jnp.log(target_psd + eps))**2)
        nlp = 0.5 * jnp.sum(((theta - prior_mean) / prior_std)**2)
        return nll + nlp

    # Adam MAP
    vg = jax.jit(jax.value_and_grad(neg_log_joint))
    theta = jnp.zeros(k)
    m, v = jnp.zeros(k), jnp.zeros(k)

    for step in range(n_opt_steps):
        loss, grad = vg(theta)
        grad = jnp.clip(grad, -5.0, 5.0)
        cur_lr = lr * 0.5 * (1 + np.cos(np.pi * step / n_opt_steps))
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        mh = m / (1 - 0.9**(step + 1))
        vh = v / (1 - 0.999**(step + 1))
        theta = theta - cur_lr * mh / (jnp.sqrt(vh) + 1e-8)

    final_nlj = float(neg_log_joint(theta))

    if use_bic:
        # BIC approximation: F ≈ -NLJ - (k/2)*log(n_data)
        n_data = len(target_indices)
        F = -final_nlj - 0.5 * k * np.log(n_data)
    else:
        # Exact Laplace free energy
        try:
            H = jax.hessian(neg_log_joint)(theta)
            sign, logdet = jnp.linalg.slogdet(H)
            F = -final_nlj + 0.5 * k * float(jnp.log(2 * jnp.pi)) - 0.5 * float(logdet)
            if int(sign) <= 0:
                F = -1e10
        except Exception:
            F = -1e10

    return F, final_nlj, theta


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha-blocking', action='store_true')
    parser.add_argument('--n-subjects', type=int, default=82)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--bic', action='store_true',
                        help='Use BIC approximation instead of exact Hessian')
    parser.add_argument('--output', type=str, default='results/realdata')
    args = parser.parse_args()

    if args.quick:
        args.n_subjects = min(args.n_subjects, 10)
        n_opt_steps = 50
        n_steps = 20_000
    else:
        n_opt_steps = 200
        n_steps = 50_000

    dt_s = 0.5e-3  # 2 kHz
    n_warmup = n_steps // 5
    nperseg = 1024

    print("=" * 60)
    print("vbjax BMS on Hartoyo et al. EEG data")
    print("=" * 60)
    print(f"Device: {jax.devices()[0]}")
    print(f"Subjects: {args.n_subjects}")
    print(f"Opt steps: {n_opt_steps}, sim steps: {n_steps}")

    # Load real data
    spectra_ec, freqs = load_resting_spectra()
    n_sub = min(args.n_subjects, spectra_ec.shape[0])
    spectra_ec = spectra_ec[:n_sub]
    n_freq = len(freqs)
    print(f"Loaded {n_sub} subjects, {n_freq} freq bins ({float(freqs[0]):.1f}-{float(freqs[-1]):.1f} Hz)")

    # Model specs: name, factory, free_params
    model_specs = [
        ('Liley', make_liley_spectral,
         ['p_ee', 'sigma_e', 'Gamma_e', 'gamma_i']),
        ('CMC', make_cmc_spectral,
         ['I', 'g_ss_sp', 'He', 'Hi']),
        ('RRW', make_rrw_spectral,
         ['I', 'nu_ee', 'nu_es', 'gamma_e']),
        ('CBEI', make_cbei_spectral,
         ['I', 'kappa_ee', 'kappa_ie', 'tau_s_e']),
    ]

    conditions = [('eyes_closed', spectra_ec)]

    if args.alpha_blocking:
        spectra_ab, _ = load_alpha_blocking_spectra()
        spectra_eo = spectra_ab[:n_sub, 1, :]  # eyes-open
        conditions.append(('eyes_open', spectra_eo))

    all_results = {}

    for cond_name, cond_spectra in conditions:
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond_name}")
        print(f"{'='*60}")

        free_energies = np.zeros((n_sub, len(model_specs)))
        names = [s[0] for s in model_specs]

        for j, (mname, factory, free_params) in enumerate(model_specs):
            k = len(free_params)
            sde_loop, target_indices, n_states = factory(
                free_params, dt_s, n_steps, n_warmup, nperseg,
                np.array(freqs))

            t0 = time.perf_counter()
            print(f"\n  Fitting {mname} ({n_states}D, {k} params: {free_params})...")

            for i in range(n_sub):
                noise_key = jax.random.PRNGKey(1000 + i)
                target_psd_i = cond_spectra[i]

                F, loss, theta = fit_to_spectrum(
                    sde_loop, n_states, target_indices, target_psd_i,
                    dt_s, n_steps, n_warmup, nperseg, k, noise_key,
                    n_opt_steps=n_opt_steps, lr=0.05,
                    use_bic=args.bic)

                free_energies[i, j] = F

                if i < 3:
                    print(f"    s{i}: F={F:.2f}, loss={loss:.2f}, "
                          f"theta={[round(float(t), 3) for t in theta]}")

            dt_m = time.perf_counter() - t0
            mean_F = free_energies[:, j].mean()
            valid = (free_energies[:, j] > -1e9).sum()
            print(f"    {n_sub} subjects in {dt_m:.1f}s ({dt_m/60:.1f} min), "
                  f"mean F={mean_F:.2f}, valid Hessians: {valid}/{n_sub}")

        # BMS
        print(f"\n{'─'*60}")
        print(f"BMS: {cond_name}")
        print(f"{'─'*60}")

        post_prob, winner = bms_ffx(jnp.array(free_energies))
        print(f"\n  FFX: winner = {names[int(winner)]}")
        for jj, nm in enumerate(names):
            print(f"    {nm:8s}: sum F={free_energies[:,jj].sum():>12.1f}, "
                  f"p={float(post_prob[jj]):.6f}")

        rfx = bms_rfx(jnp.array(free_energies))
        print(f"\n  RFX (Stephan et al. 2009):")
        print(f"    {'Model':8s}  {'Exp r':>8s}  {'XP':>8s}  {'PXP':>8s}")
        for jj, nm in enumerate(names):
            print(f"    {nm:8s}  {float(rfx['exp_r'][jj]):>8.4f}  "
                  f"{float(rfx['exceedance_prob'][jj]):>8.4f}  "
                  f"{float(rfx['protected_exceedance_prob'][jj]):>8.4f}")

        all_results[cond_name] = {
            'free_energies': free_energies.tolist(),
            'model_names': names,
            'ffx_winner': names[int(winner)],
            'ffx_probs': [float(p) for p in post_prob],
            'rfx_exp_r': [float(r) for r in rfx['exp_r']],
            'rfx_xp': [float(x) for x in rfx['exceedance_prob']],
            'rfx_pxp': [float(x) for x in rfx['protected_exceedance_prob']],
        }

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_out = {
        **all_results,
        'n_subjects': n_sub,
        'n_opt_steps': n_opt_steps,
        'n_steps': n_steps,
        'freqs': freqs.tolist(),
        'device': str(jax.devices()[0]),
    }
    out_path = out_dir / 'bms_realdata.json'
    with open(out_path, 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
