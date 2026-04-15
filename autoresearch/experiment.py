"""Explore new parameter space regions: expanded free params, multi-start, warmup schedule,
and adaptive noise strategies.

Strategy 1: Expanded free parameter sets with moderate LR (more expressive models).
Strategy 2: Multi-start with different noise seeds to escape local minima.
Strategy 3: Warmup LR schedule with tight gradient clipping for stable convergence.
Strategy 4: High prior_std (relaxed regularization) + cosine schedule.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

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
    """Four strategies exploring new regions of the optimization landscape."""

    # Generate ground truth (2 subjects, fixed seed)
    target_psds = generate_synthetic_eeg(n_subjects=2, seed=0)

    # -----------------------------------------------------------------------
    # Strategy 1: Expanded free parameter sets — more expressive models
    # Each model gets 4 free params instead of 2, moderate LR + cosine
    # -----------------------------------------------------------------------
    expanded_models = [
        {'name': 'liley', 'free_params': ['p_ee', 'p_ei', 'sigma_e', 'sigma_i']},
        {'name': 'cmc',   'free_params': ['I', 'He', 'g_ss_sp', 'g_sp_ii']},
        {'name': 'rrw',   'free_params': ['I', 'nu_ee', 'nu_ei', 'gamma_e']},
        {'name': 'cbei',  'free_params': ['I', 'kappa_ee', 'kappa_ei', 'tau_m_e']},
    ]

    results1 = run_comparison(
        expanded_models, target_psds,
        n_opt_steps=150,
        lr=0.04,
        noise_sigma=2e-4,
        grad_clip=8.0,
        prior_std=1.0,
        lr_schedule='cosine',
        strategy='expanded_free_params',
    )
    for r in results1:
        print_result(r)
        log_result(r)

    # -----------------------------------------------------------------------
    # Strategy 2: Multi-start — run each model 3 times with different noise
    # seeds, keep best result per model
    # -----------------------------------------------------------------------
    multistart_models = [
        {'name': 'liley', 'free_params': ['p_ee', 'sigma_e', 'Gamma_e']},
        {'name': 'cmc',   'free_params': ['I', 'He', 'g_ss_sp']},
        {'name': 'rrw',   'free_params': ['I', 'nu_ee', 'nu_ei']},
        {'name': 'cbei',  'free_params': ['I', 'kappa_ee', 'kappa_ei']},
    ]

    best_multistart = {}
    for m in multistart_models:
        best_r = None
        for seed in [10, 20, 30]:
            r = fit_model(
                model_name=m['name'],
                free_param_names=m['free_params'],
                target_psd=target_psds,
                n_opt_steps=120,
                lr=0.05,
                noise_sigma=3e-4,
                grad_clip=10.0,
                prior_std=1.0,
                lr_schedule='cosine',
                noise_seed=seed,
            )
            # Override strategy label for logging
            r = ExperimentResult(
                commit=r.commit,
                strategy=f'multistart_seed{seed}',
                model_name=r.model_name,
                n_free_params=r.n_free_params,
                n_subjects=r.n_subjects,
                n_opt_steps=r.n_opt_steps,
                lr=r.lr,
                noise_sigma=r.noise_sigma,
                dt_s=r.dt_s,
                n_steps=r.n_steps,
                spectral_loss=r.spectral_loss,
                free_energy=r.free_energy,
                theta_map=r.theta_map,
                wall_time=r.wall_time,
                status=r.status,
            )
            print_result(r)
            log_result(r)
            if best_r is None or r.spectral_loss < best_r.spectral_loss:
                best_r = r
        best_multistart[m['name']] = best_r

    # -----------------------------------------------------------------------
    # Strategy 3: Warmup LR schedule with tight gradient clipping
    # Warmup helps avoid early divergence; tight clip stabilizes gradients
    # -----------------------------------------------------------------------
    warmup_models = [
        {'name': 'liley', 'free_params': ['p_ee', 'sigma_e']},
        {'name': 'cmc',   'free_params': ['I', 'g_ss_sp']},
        {'name': 'rrw',   'free_params': ['I', 'nu_ee']},
        {'name': 'cbei',  'free_params': ['I', 'kappa_ee']},
    ]

    results3 = run_comparison(
        warmup_models, target_psds,
        n_opt_steps=200,
        lr=0.07,
        noise_sigma=2e-4,
        grad_clip=5.0,
        prior_std=1.0,
        lr_schedule='warmup',
        strategy='warmup_tight_clip',
    )
    for r in results3:
        print_result(r)
        log_result(r)

    # -----------------------------------------------------------------------
    # Strategy 4: Relaxed regularization (high prior_std) + cosine schedule
    # Allows parameters to deviate more from defaults — wider search
    # -----------------------------------------------------------------------
    relaxed_models = [
        {'name': 'liley', 'free_params': ['p_ee', 'p_ei', 'sigma_e']},
        {'name': 'cmc',   'free_params': ['I', 'He', 'Hi']},
        {'name': 'rrw',   'free_params': ['I', 'nu_ee', 'nu_sr']},
        {'name': 'cbei',  'free_params': ['I', 'kappa_ee', 'eta_e']},
    ]

    results4 = run_comparison(
        relaxed_models, target_psds,
        n_opt_steps=175,
        lr=0.05,
        noise_sigma=1e-4,
        grad_clip=10.0,
        prior_std=2.0,
        lr_schedule='cosine',
        strategy='relaxed_prior_cosine',
    )
    for r in results4:
        print_result(r)
        log_result(r)

    # -----------------------------------------------------------------------
    # Aggregate: best result per model across all strategies
    # -----------------------------------------------------------------------
    all_results = results1 + list(best_multistart.values()) + results3 + results4

    best_per_model = {}
    for r in all_results:
        name = r.model_name
        if name not in best_per_model or r.spectral_loss < best_per_model[name].spectral_loss:
            best_per_model[name] = r

    if best_per_model:
        total_loss = sum(r.spectral_loss for r in best_per_model.values()) / len(best_per_model)
        agg = ExperimentResult(
            commit=get_commit_hash(),
            strategy='best_aggregate',
            model_name='ALL',
            spectral_loss=total_loss,
            n_subjects=2,
        )
        print_result(agg)
        log_result(agg)

        # Print summary
        print("\n=== Best results per model ===")
        for name, r in sorted(best_per_model.items()):
            print(f"  {name:8s}: spectral_loss={r.spectral_loss:.4f}  strategy={r.strategy}")
        print(f"  Mean spectral_loss: {total_loss:.4f}  (spectral_fit={-total_loss:.4f})")


if __name__ == "__main__":
    run_experiment()