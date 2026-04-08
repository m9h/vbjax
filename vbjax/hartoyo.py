"""Hartoyo et al. (2019, 2020) EEG dataset loaders.

Loads resting-state and alpha-blocking EEG power spectra from the
Swinburne datasets for fitting with vbjax neural mass models.

References:
    Hartoyo et al. (2019) Parameter estimation and identifiability in a
        neural population model for electro-cortical activity. PLoS Comp Bio.
    Hartoyo et al. (2020) Inferring a simple mechanism for alpha-blocking
        by fitting a neural population model to EEG spectra. PLoS Comp Bio.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import scipy.io as sio


# Default data paths (relative to vbjax project root)
_PROJECT_ROOT = Path(__file__).parent.parent
DATA_2019 = _PROJECT_ROOT / "data" / "hartoyo2019" / "Hartoyo et al. (2019) code"
DATA_2020 = _PROJECT_ROOT / "data" / "hartoyo2020" / "Datasets"


def load_resting_spectra(data_dir: Path | None = None):
    """Load eyes-closed resting EEG spectra (82 subjects x 73 freq bins).

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the .mat files. Defaults to the 2019 dataset.

    Returns
    -------
    spectra : jnp.ndarray, shape (82, 73)
        Power spectral density for each subject.
    freqs : jnp.ndarray, shape (73,)
        Frequency axis (Hz), typically 2.0--19.75 Hz.
    """
    data_dir = data_dir or DATA_2019

    target = sio.loadmat(str(data_dir / "82x73_target_spectra.mat"))
    spectra = target["selected_ec_spectra"]  # (82, 73)

    eeg = sio.loadmat(str(data_dir / "EEGSpectra.mat"))
    freq = eeg["freq"].ravel()           # (80,)
    indx_f = eeg["indx_f"].ravel() - 1   # MATLAB 1-indexed → 0-indexed
    freqs = freq[indx_f]                  # (72,) — but spectra has 73 cols

    # The target spectra have 73 bins; freq indices give 72.
    # The extra bin is likely the DC or a padding — trim to match.
    n = min(len(freqs), spectra.shape[1])
    freqs = freqs[:n]
    spectra = spectra[:, :n]

    return jnp.array(spectra, dtype=jnp.float32), jnp.array(freqs, dtype=jnp.float32)


def load_alpha_blocking_spectra(data_dir: Path | None = None):
    """Load alpha-blocking spectra (82 subjects x 2 conditions x 73 freq bins).

    Condition 0 = eyes-closed, condition 1 = eyes-open.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing the 2020 .mat files.

    Returns
    -------
    spectra : jnp.ndarray, shape (82, 2, 73)
        Power spectral density: [..., 0] = eyes-closed, [..., 1] = eyes-open.
    freqs : jnp.ndarray, shape (72,)
        Frequency axis (Hz).
    """
    data_dir = data_dir or DATA_2020

    ab = sio.loadmat(str(data_dir / "82x2x73_alpha_blocking_spectra.mat"))
    spectra = ab["spectra_pairs"]  # (82, 2, 73)

    # Load freq grid from 2019 data (same dataset)
    eeg = sio.loadmat(str(DATA_2019 / "EEGSpectra.mat"))
    freq = eeg["freq"].ravel()
    indx_f = eeg["indx_f"].ravel() - 1
    freqs = freq[indx_f]

    n = min(len(freqs), spectra.shape[2])
    freqs = freqs[:n]
    spectra = spectra[:, :, :n]

    return jnp.array(spectra, dtype=jnp.float32), jnp.array(freqs, dtype=jnp.float32)


def load_hartoyo_params(data_dir: Path | None = None, year: int = 2019):
    """Load Hartoyo's best-fit Liley parameters for comparison.

    Parameters
    ----------
    data_dir : Path, optional
    year : int
        2019 for 24-param fits, 2020 for 32-param regularized fits.

    Returns
    -------
    params : np.ndarray
        Shape (82, 100, n_params) — 100 PSO particles per subject.
    """
    if year == 2019:
        data_dir = data_dir or DATA_2019
        bp = sio.loadmat(str(data_dir / "82x100x24_best_paramsets.mat"))
        return bp["paramset"]  # (82, 100, 24)
    elif year == 2020:
        data_dir = data_dir or DATA_2020
        bp = sio.loadmat(str(data_dir / "82x100x32_regularized_best_paramsets.mat"))
        return bp["paramset"]  # (82, 100, 32)
    else:
        raise ValueError(f"Unknown year: {year}")
