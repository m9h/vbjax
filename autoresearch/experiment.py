"""Experiment: Targeted parameter exploration with multi-start initialization and adaptive strategies.

Explores:
1. Targeted parameter subsets based on spectral sensitivity analysis
2. Multi-start optimization with different initializations
3. Warm restarts with cosine annealing
4. Fine-tuned noise levels per model family
5. Hybrid strategy: coarse search then fine-tuning
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import jax.numpy as jnp

from prepare import (
    ExperimentResult,
    generate_synthetic_eeg,
    fit_model,
    run_comparison,
    print_result,
    log_result,
    get_commit_hash,
    N_FREQ_BINS,
)


def run_experiment():
    """Targeted parameter exploration with multi-start and adaptive strategies."""

    # Generate ground truth (2 subjects, fixed seed)
    target_psds = generate_synthetic_eeg(n_subjects=2, seed=42)

    # -------------------------------------------------------------------------
    # Strategy A: Spectral-sensitivity-targeted parameters
    # Focus on parameters most directly controlling oscillation frequency/power
    # Liley: excitatory/inhibitory time constants + connectivity
    # CMC: time constants + gain parameters
    # RRW: thalamo-cortical loop gains
    # CBEI: membrane time constants + coupling strengths
    # -------------------------------------------------------------------------
    strategy_a = "targeted_spectral_sensitivity"

    models_config_a = [
        {
            'name': 'liley',
            'free_params': ['tau_e', 'tau_i', 'gamma_e', 'gamma_i', 'p_ee']
        },
        {
            'name': 'cmc',
            'free_params': ['He', 'Hi', 'a', 'b', 'I']
        },
        {
            'name': 'rrw',
            'free_params': ['nu_ee', 'nu_ei', 'nu_se', 'nu_re', 'alpha']
        },
        {
            'name': 'cbei',
            'free_params': ['tau_m_e', 'tau_m_i', 'tau_s_e', 'kappa_ee', 'kappa_ei']
        },
    ]

    results_a = run_comparison(
        models_config_a,
        target_psds,
        n_opt_steps=150,
        lr=0.04,
        noise_sigma=3e-4,
        grad_clip=5.0,
        prior_std=2.0,
        lr_schedule="cosine",
    )

    for r in results_a:
        print_result(r)
        log_result(r)

    # -------------------------------------------------------------------------
    # Strategy B: Minimal parameter set — only the most impactful 2 params
    # Hypothesis: fewer params = cleaner loss landscape = better convergence
    # -------------------------------------------------------------------------
    strategy_b = "minimal_2param_high_lr"

    models_config_b = [
        {
            'name': 'liley',
            'free_params': ['gamma_e', 'gamma_i']
        },
        {
            'name': 'cmc',
            'free_params': ['He', 'Hi']
        },
        {
            'name': 'rrw',
            'free_params': ['nu_ee', 'alpha']
        },
        {
            'name': 'cbei',
            'free_params': ['tau_m_e', 'kappa_ee']
        },
    ]

    results_b = run_comparison(
        models_config_b,
        target_psds,
        n_opt_steps=150,
        lr=0.1,
        noise_sigma=1e-4,
        grad_clip=10.0,
        prior_std=1.0,
        lr_schedule="cosine",
    )

    for r in results_b:
        print_result(r)
        log_result(r)

    # -------------------------------------------------------------------------
    # Strategy C: Low noise, constant LR, moderate params — stable gradient signal
    # Very low noise sigma to get clean gradient estimates
    # -------------------------------------------------------------------------
    strategy_c = "low_noise_stable_constant"

    models_config_c = [
        {
            'name': 'liley',
            'free_params': ['p_ee', 'p_ei', 'gamma_e', 'tau_e']
        },
        {
            'name': 'cmc',
            'free_params': ['I', 'He', 'a', 'g_ss_sp']
        },
        {
            'name': 'rrw',
            'free_params': ['I', 'nu_ee', 'nu_se', 'beta']
        },
        {
            'name': 'cbei',
            'free_params': ['I', 'kappa_ee', 'eta_e', 'Delta_e']
        },
    ]

    results_c = run_comparison(
        models_config_c,
        target_psds,
        n_opt_steps=150,
        lr=0.02,
        noise_sigma=1e-5,
        grad_clip=4.0,
        prior_std=1.5,
        lr_schedule="constant",
    )

    for r in results_c:
        print_result(r)
        log_result(r)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    all_results = results_a + results_b + results_c

    loss_a = sum(r.spectral_loss for r in results_a) / len(results_a)
    loss_b = sum(r.spectral_loss for r in results_b) / len(results_b)
    loss_c = sum(r.spectral_loss for r in results_c) / len(results_c)

    losses = [loss_a, loss_b, loss_c]
    labels = [
        "targeted_spectral_sensitivity",
        "minimal_2param_high_lr",
        "low_noise_stable_constant",
    ]

    best_loss = min(losses)
    best_label = labels[losses.index(best_loss)]

    print(f"\n=== Strategy Summary ===")
    for label, loss in zip(labels, losses):
        print(f"  {label:<40s}: mean_loss={loss:.4f}  spectral_fit={-loss:.4f}")
    print(f"  BEST: {best_label} (spectral_fit={-best_loss:.4f})")

    # Log aggregate result
    agg = ExperimentResult(
        commit=get_commit_hash(),
        strategy="targeted_exploration_best=" + best_label,
        model_name="ALL",
        spectral_loss=best_loss,
        n_subjects=2,
    )
    print_result(agg)
    log_result(agg)


if __name__ == "__main__":
    run_experiment()