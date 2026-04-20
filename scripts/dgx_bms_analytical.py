#!/usr/bin/env python3
"""Apples-to-apples BMS: ALL models use analytical transfer functions.

Fits linearized transfer function PSDs to 82 subjects of Hartoyo et al.
real EEG data. No time-domain simulation — pure complex arithmetic.

Models: JR, CMC, Liley, RRW, CBEI
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import os
os.environ.setdefault('TMPDIR', '/data/mhough/tmp')
# Don't set XLA_FLAGS here — let vbjax.__init__ handle it

import vbjax as vb
from vbjax.transfer import (jr_analytical_psd, cmc_analytical_psd,
                             liley_analytical_psd, cbei_analytical_psd)
from vbjax.hartoyo import load_resting_spectra, load_alpha_blocking_spectra
from vbjax.bms import bms_ffx, bms_rfx


# =====================================================================
# Fitting: analytical PSD to observed PSD
# =====================================================================

def make_analytical_fitter(model_name, free_param_names, freqs_hz):
    """Build a fitter for an analytical transfer function model."""

    model_map = {
        'jr':    (jr_analytical_psd,    vb.jr_default_theta),
        'cmc':   (cmc_analytical_psd,   vb.cmc_default_theta),
        'liley': (liley_analytical_psd, vb.liley_default_theta),
        'cbei':  (cbei_analytical_psd,  vb.cbei_default_theta),
        'rrw':   (vb.rrw_analytical_psd, vb.rrw_default_theta),
    }

    psd_fn, default_theta = model_map[model_name]
    defaults = {name: getattr(default_theta, name) for name in free_param_names}
    k = len(free_param_names)
    freqs_np = np.array(freqs_hz)

    def predict_psd(theta):
        replacements = {}
        for i, name in enumerate(free_param_names):
            replacements[name] = jnp.exp(theta[i]) * defaults[name]
        p = default_theta._replace(**replacements)
        return psd_fn(freqs_np, p)

    return predict_psd, k, defaults


def fit_analytical(predict_psd, target_psd, k,
                   n_opt_steps=200, lr=0.05, use_bic=True):
    """Fit analytical PSD to target. Returns (F, loss, theta)."""
    prior_mean = jnp.zeros(k)
    prior_std = jnp.ones(k) * 1.5

    def neg_log_joint(theta):
        pred = predict_psd(theta)
        eps = 1e-10
        nll = jnp.sum((jnp.log(pred + eps) - jnp.log(target_psd + eps))**2)
        nlp = 0.5 * jnp.sum(((theta - prior_mean) / prior_std)**2)
        return nll + nlp

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
        n_data = len(target_psd)
        F = -final_nlj - 0.5 * k * np.log(n_data)
    else:
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
    parser.add_argument('--bic', action='store_true', default=True)
    parser.add_argument('--laplace', action='store_true')
    parser.add_argument('--output', type=str, default='results/analytical')
    args = parser.parse_args()

    if args.laplace:
        args.bic = False

    n_opt = 100 if args.quick else 200
    if args.quick:
        args.n_subjects = min(args.n_subjects, 10)

    print("=" * 60)
    print("Analytical Transfer Function BMS — Hartoyo et al. EEG")
    print("=" * 60)
    print(f"Device: {jax.devices()[0]}")
    print(f"Free energy: {'BIC' if args.bic else 'Laplace'}")
    print(f"Opt steps: {n_opt}")

    # Load data
    spectra_ec, freqs = load_resting_spectra()
    n_sub = min(args.n_subjects, spectra_ec.shape[0])
    spectra_ec = spectra_ec[:n_sub]
    print(f"Loaded {n_sub} subjects, {len(freqs)} freq bins "
          f"({float(freqs[0]):.1f}-{float(freqs[-1]):.1f} Hz)")

    # Model specs: (display_name, model_key, free_params)
    model_specs = [
        ('JR',    'jr',    ['A', 'B', 'a', 'b', 'mu']),
        ('CMC',   'cmc',   ['I', 'g_ss_sp', 'g_sp_ii', 'g_ii_ss', 'g_ii_sp']),
        ('Liley', 'liley', ['p_ee', 'sigma_e', 'tau_e', 'p_ei', 'tau_i']),
        ('RRW',   'rrw',   ['nu_ee', 'nu_ei', 'nu_se', 'nu_re']),
        ('CBEI',  'cbei',  ['kappa_ee', 'kappa_ei', 'kappa_ie', 'kappa_ii']),
    ]
    names = [s[0] for s in model_specs]

    conditions = [('eyes_closed', spectra_ec)]
    if args.alpha_blocking:
        spectra_ab, _ = load_alpha_blocking_spectra()
        spectra_eo = spectra_ab[:n_sub, 1, :]
        conditions.append(('eyes_open', spectra_eo))

    all_results = {}

    for cond_name, cond_spectra in conditions:
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond_name}")
        print(f"{'='*60}")

        free_energies = np.zeros((n_sub, len(model_specs)))

        for j, (mname, mkey, free_params) in enumerate(model_specs):
            predict_psd, k, defaults = make_analytical_fitter(
                mkey, free_params, np.array(freqs))

            t0 = time.perf_counter()
            print(f"\n  Fitting {mname} (analytical, {k} params: {free_params})...",
                  flush=True)

            for i in range(n_sub):
                target_psd_i = cond_spectra[i]
                F, loss, theta = fit_analytical(
                    predict_psd, target_psd_i, k,
                    n_opt_steps=n_opt, lr=0.05, use_bic=args.bic)
                free_energies[i, j] = F

                if i < 3:
                    print(f"    s{i}: F={F:.2f}, loss={loss:.2f}, "
                          f"theta={[round(float(t), 3) for t in theta]}",
                          flush=True)

            dt_m = time.perf_counter() - t0
            mean_F = free_energies[:, j].mean()
            valid = (free_energies[:, j] > -1e9).sum()
            print(f"    {n_sub} subjects in {dt_m:.1f}s ({dt_m/60:.1f} min), "
                  f"mean F={mean_F:.2f}, valid: {valid}/{n_sub}", flush=True)

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

        # Per-subject winners
        winners = np.argmax(free_energies, axis=1)
        print(f"\n  Per-subject winners:")
        for jj, nm in enumerate(names):
            print(f"    {nm:8s}: {(winners==jj).sum()}/{n_sub}")

        all_results[cond_name] = {
            'free_energies': free_energies.tolist(),
            'model_names': names,
            'ffx_winner': names[int(winner)],
            'ffx_probs': [float(p) for p in post_prob],
            'rfx_exp_r': [float(r) for r in rfx['exp_r']],
            'rfx_xp': [float(x) for x in rfx['exceedance_prob']],
            'rfx_pxp': [float(x) for x in rfx['protected_exceedance_prob']],
            'per_subject_winners': {nm: int((winners==jj).sum())
                                    for jj, nm in enumerate(names)},
        }

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_out = {
        **all_results,
        'method': 'analytical_transfer_function',
        'free_energy_type': 'BIC' if args.bic else 'Laplace',
        'n_subjects': n_sub,
        'n_opt_steps': n_opt,
        'freqs': [float(f) for f in freqs],
        'device': str(jax.devices()[0]),
    }
    out_path = out_dir / 'bms_analytical.json'
    with open(out_path, 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
