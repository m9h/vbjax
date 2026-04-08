---
category: research
section: methodology
weight: 30
title: "Neural Mass Model Spectral Fitting Optimization"
status: draft
slide_summary: "Evolutionary optimization of spectral fitting strategies for neural mass models (Liley, CMC, RRW, CBEI) — optimizer hyperparameters, parameter selection, noise tuning, and initialization."
tags: [autoresearch, neural-mass, spectral-fitting, bayesian-model-comparison, eeg]
---

# Neural Mass Model Spectral Fitting Autoresearch

## Goal

Optimize the **spectral fitting pipeline** for whole-brain electrophysiological
models so that Bayesian model comparison (Stephan et al. 2009) produces
reliable, well-separated free energies across model families.

The current problem: fitting neural mass models to EEG power spectra via
gradient descent often gets stuck (parameters don't move, identical fits
across subjects). This loop evolves the fitting strategy itself.

## Research Questions

**Q1: Optimizer Configuration** — What combination of learning rate, schedule,
number of steps, and gradient clipping produces the lowest spectral loss
across all 4 model families?

**Q2: Parameter Selection** — Which subset of each model's parameters should
be free (estimated) vs fixed? More free params = more expressive but harder
to fit and higher complexity penalty.

**Q3: Initialization** — Do informed initial conditions (from linearization,
analytical fixed points) beat zero initialization?

**Q4: Noise & Simulation** — How do simulation length, dt, and diffusion
coefficient affect the loss landscape smoothness and gradient quality?

## The Loop

You are an autonomous research agent. Run this loop:

1. **Read** `results.tsv` to see what strategies have been tried
2. **Modify** `experiment.py` with a new fitting strategy
3. **Run**: the experiment fits 4 models to synthetic Liley EEG
4. **Extract**: `grep "^RESULT|"` lines with spectral_fit score
5. **Log** to `results.tsv`
6. Higher `spectral_fit` is better (negative mean spectral loss across models)

## Metric

```
spectral_fit = -mean_spectral_loss
```

Where `mean_spectral_loss` is the average log-spectral distance across
all 4 models after fitting to synthetic Liley-generated EEG. Lower loss
(higher spectral_fit) means better fitting strategy.

## Models

| Model | States | Key |
|-------|--------|-----|
| Liley (ground truth) | 14 | Conductance-based, long-range axonal |
| CMC | 8 | Laminar canonical microcircuit |
| RRW | 8 | Corticothalamic loop |
| CBEI | 8 | Next-gen exact mean-field E-I |

## Constraints

- Experiment must complete in < 5 minutes on RTX 2080 (8GB)
- Use BIC free energy approximation (no Hessian — too slow on this GPU)
- DO NOT modify prepare.py
- Each experiment should fit all 4 models to 2 synthetic subjects minimum
- Use deterministic seeds for reproducibility
