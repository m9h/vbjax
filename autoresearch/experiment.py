"""Baseline experiment: fit 4 models with default Adam optimizer.

This file is the MUTABLE target for AgentSciML evolutionary optimization.
Modify the fitting strategy, hyperparameters, parameter selection, etc.
"""

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
    """Baseline: Adam optimizer, 2 free params, 100 steps, lr=0.01."""

    # Generate ground truth
    target_psds = generate_synthetic_eeg(n_subjects=2, seed=0)

    # Model configurations: which params to fit per model
    models = [
        {'name': 'liley', 'free_params': ['p_ee', 'sigma_e']},
        {'name': 'cmc',   'free_params': ['I', 'g_ss_sp']},
        {'name': 'rrw',   'free_params': ['I', 'nu_ee']},
        {'name': 'cbei',  'free_params': ['I', 'kappa_ee']},
    ]

    # Fit all models
    results = run_comparison(
        models, target_psds,
        n_opt_steps=100,
        lr=0.01,
        noise_sigma=1e-3,
        grad_clip=10.0,
        prior_std=1.0,
        lr_schedule="constant",
        strategy="baseline_adam_2params",
    )

    # Aggregate: mean spectral loss across all fits
    total_loss = sum(r.spectral_loss for r in results) / len(results)

    # Print per-model results
    for r in results:
        print_result(r)
        log_result(r)

    # Print aggregate
    agg = ExperimentResult(
        commit=get_commit_hash(),
        strategy="baseline_adam_2params",
        model_name="ALL",
        spectral_loss=total_loss,
        n_subjects=2,
    )
    print_result(agg)


if __name__ == "__main__":
    run_experiment()
