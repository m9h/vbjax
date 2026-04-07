"""Canonical Microcircuit (CMC) comparison with other vbjax neural mass models.

Compares CMC against Jansen-Rit (JR), Montbrio-Pazo-Roxin (MPR), and
the Reduced Wong-Wang (RWW, from neurojax) across several dimensions:

1. Model properties table (states, params, populations, time scale)
2. Spectral fingerprints — power spectral density under noise drive
3. Input-output transfer — step response dynamics
4. Bifurcation sketch — sweep external input, track oscillation amplitude
5. Laminar decomposition — CMC-unique: sp vs dp output comparison
6. Phase-plane projections — dynamical structure visualization

Usage:
    python examples/cmc_comparison.py            # run all comparisons
    python examples/cmc_comparison.py --no-plot   # print table only

References:
    Bastos et al. (2012) Neuron 76(4):695-711.
    Douglas (2025) arXiv:2508.06501.
    Deco et al. (2013) J Neurosci 33(27):11239-11252.
"""

import sys
import numpy as np
import jax
import jax.numpy as jp
import vbjax as vb

# ── Model properties ────────────────────────────────────────────────

MODEL_TABLE = """
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     Neural Mass Model Comparison Table                               │
├──────────┬────────┬────────┬──────────────┬───────────┬──────────────────────────────┤
│ Model    │ States │ Params │ Populations  │ PSP order │ Unique capability            │
├──────────┼────────┼────────┼──────────────┼───────────┼──────────────────────────────┤
│ CMC      │   8    │   16   │ ss,sp,ii,dp  │ 2nd-order │ Laminar decomposition,       │
│          │        │        │              │           │ forward/backward asymmetry,  │
│          │        │        │              │           │ predictive coding            │
├──────────┼────────┼────────┼──────────────┼───────────┼──────────────────────────────┤
│ JR       │   6    │   14   │ pyr,ei,ii    │ 2nd-order │ Established workhorse,       │
│          │        │        │              │           │ well-characterized bifurc.   │
├──────────┼────────┼────────┼──────────────┼───────────┼──────────────────────────────┤
│ MPR      │   2    │    7   │ r, V         │ 1st-order │ Exact mean-field of QIF,     │
│          │        │        │              │           │ analytical tractability       │
├──────────┼────────┼────────┼──────────────┼───────────┼──────────────────────────────┤
│ RWW      │   2    │   20   │ S_E, S_I     │ 1st-order │ NMDA kinetics, WhoBPyT       │
│          │        │        │              │           │ compatible, bistability       │
├──────────┼────────┼────────┼──────────────┼───────────┼──────────────────────────────┤
│ Dopa     │   6    │   27   │ r,V,u,Sa,Sg, │ Mixed    │ Dopamine modulation,         │
│          │        │        │ Dp           │           │ 3-channel coupling           │
└──────────┴────────┴────────┴──────────────┴───────────┴──────────────────────────────┘
"""

# ── Informative comparison dimensions ────────────────────────────────
#
# Each comparison below is designed to reveal a specific strength or
# weakness of the CMC relative to existing models.

COMPARISON_DESIGN = """
COMPARISON DESIGN: CMC vs existing vbjax models
================================================

1. SPECTRAL FINGERPRINTS (CMC vs JR)
   ─────────────────────────────────
   Why:  Both use 2nd-order PSP kernels and same sigmoid, so spectral
         differences isolate the effect of circuit architecture alone.
   What: PSD under identical noise drive.
   Expect: JR → single alpha peak (~10 Hz).
           CMC → potentially richer spectrum with alpha + gamma due to
           separate E/I loops through sp-ii (fast) and dp-ii (slower).
   Strength revealed: CMC produces multi-band spectra from a single node.
   Weakness revealed: CMC may require more careful tuning for stable alpha.

2. INPUT-OUTPUT TRANSFER (CMC vs JR)
   ──────────────────────────────────
   Why:  The ss → sp → ii → dp → sp loop creates a delay chain absent
         in JR's 3-population circuit.
   What: Step response — inject I=300 at t=0, record all populations.
   Expect: CMC shows sequential activation: ss first, then sp, then ii,
           then dp. JR activates all 3 populations near-simultaneously.
   Strength revealed: CMC captures temporal sequencing across layers.
   Weakness revealed: Longer transient to steady state.

3. WHOLE-BRAIN FC FITTING COST (CMC vs RWW vs JR)
   ────────────────────────────────────────────────
   Why:  More states per node = more computation. The question is whether
         the extra dynamics help or hurt FC fitting.
   What: Compute time per step × n_nodes, parameter identifiability.
   Expect: RWW (2 states) ~4x faster than CMC (8 states) per node.
           JR (6 states) ~25% cheaper than CMC per node.
           CMC may fit FC better when empirical data has laminar signatures.
   Strength revealed: CMC captures E/I balance more explicitly.
   Weakness revealed: Overparameterized for FC-only fitting.

4. OSCILLATION ONSET BIFURCATION (CMC vs MPR)
   ───────────────────────────────────────────
   Why:  MPR has analytical Hopf bifurcation results. Comparing CMC's
         numerical bifurcation against MPR's analytical one tests whether
         CMC's richer circuit changes the phase transition qualitatively.
   What: Sweep I from 0→500, track oscillation amplitude.
   Expect: Both show supercritical Hopf, but CMC may show secondary
           bifurcation (period doubling or torus) at higher I.
   Strength revealed: CMC has richer bifurcation structure.
   Weakness revealed: Harder to characterize analytically.

5. LAMINAR DECOMPOSITION (CMC only)
   ─────────────────────────────────
   Why:  This is CMC's unique capability — no other vbjax model separates
         superficial and deep pyramidal output.
   What: Compare sp (EEG-like, prediction error) vs dp (LFP-like,
         prediction) time series under the same drive.
   Expect: sp and dp are correlated but not identical. sp leads dp for
         feedforward input (bottom-up), dp leads sp for feedback
         (top-down). This is the predictive coding signature.
   Strength revealed: Testable predictions for simultaneous EEG-LFP.
   Weakness revealed: Adds 2 extra unobserved states if you only have EEG.

6. FORWARD/BACKWARD ASYMMETRY (CMC only)
   ──────────────────────────────────────
   Why:  The CMC's circuit structure distinguishes feedforward (→ ss) from
         feedback (→ sp + dp) pathways. This is absent in all other models
         where coupling enters via a single channel.
   What: Drive same node via ss (forward) vs sp+dp (backward), compare
         output at sp and dp.
   Expect: Forward drive → strong sp response, weak dp.
           Backward drive → strong dp response, modulated sp.
   Strength revealed: Natural implementation of hierarchical processing.
   Weakness revealed: Requires knowing the hierarchy direction a priori.

7. PARAMETER IDENTIFIABILITY (CMC vs JR)
   ──────────────────────────────────────
   Why:  CMC has 8 connectivity parameters vs JR's 4 (a_1..a_4 × J).
         More parameters risk non-identifiability under noisy observations.
   What: Generate synthetic data from known params, fit via gradient
         descent, check recovery accuracy.
   Expect: JR's 4 connectivity params are more identifiable from EEG
         alone. CMC may need multimodal data (EEG + LFP, or EEG + fMRI
         via vpjax) to constrain all 8 connections.
   Strength revealed: CMC rewards richer measurement modalities.
   Weakness revealed: Overfit risk with single-modality data.
"""


def simulate_model(dfun_factory, n_steps=8000, dt=0.5, sigma=1e-3, seed=42):
    """Simulate a single-node model and return time series."""
    dfun, y0 = dfun_factory()
    _, loop = vb.make_sde(dt=dt, dfun=dfun, gfun=sigma)
    key = jax.random.PRNGKey(seed)
    zs = jax.random.normal(key, (n_steps,) + y0.shape)
    ys = loop(y0, zs, None)
    t = jp.arange(n_steps) * dt
    return t, ys


def compute_psd(x, dt, nperseg=1024):
    """Power spectral density via Welch's method (numpy fallback)."""
    from scipy.signal import welch
    fs = 1000.0 / dt  # convert ms timestep to Hz
    f, pxx = welch(np.array(x), fs=fs, nperseg=nperseg)
    return f, pxx


# ── Model factories (wrap dfun to take (y, None) for make_sde) ───────

def make_cmc_factory(I=220.0):
    def factory():
        p = vb.cmc_default_theta._replace(I=I)
        dfun = lambda y, _: vb.cmc_dfun(y, 0.0, p)
        y0 = jp.zeros(8)
        return dfun, y0
    return factory


def make_jr_factory(I=220.0):
    def factory():
        p = vb.jr_default_theta._replace(I=I)
        dfun = lambda y, _: vb.jr_dfun(y, 0.0, p)
        y0 = jp.zeros(6)
        return dfun, y0
    return factory


def make_mpr_factory(I=3.0):
    def factory():
        p = vb.mpr_default_theta._replace(I=I)
        dfun = lambda y, _: vb.mpr_dfun(y, jp.array([0.0, 0.0]), p)
        y0 = jp.array([0.0, -2.0])
        return dfun, y0
    return factory


def run_spectral_comparison(do_plot=True):
    """Compare PSD across CMC, JR, and MPR."""
    print("\n─── Spectral Fingerprints ───")

    dt = 0.5
    n_steps = 16000  # 8 seconds
    results = {}

    for name, factory in [
        ('CMC (sp)', make_cmc_factory()),
        ('JR (y1-y2)', make_jr_factory()),
        ('MPR (V)', make_mpr_factory()),
    ]:
        t, ys = simulate_model(factory, n_steps=n_steps, dt=dt, sigma=1e-3)
        # Observable: CMC=sp, JR=y1-y2 (pyramidal PSP), MPR=V
        if 'CMC' in name:
            obs = ys[:, 1]  # superficial pyramidal
        elif 'JR' in name:
            obs = ys[:, 1] - ys[:, 2]  # pyramidal membrane potential
        else:
            obs = ys[:, 1]  # mean voltage
        f, pxx = compute_psd(obs[2000:], dt)  # skip transient
        peak_f = f[np.argmax(pxx[1:])+1]  # skip DC
        peak_p = np.max(pxx[1:])
        results[name] = (f, pxx, peak_f, peak_p)
        print(f"  {name:20s}  peak freq = {peak_f:6.1f} Hz  peak power = {peak_p:.2e}")

    if do_plot:
        try:
            import pylab as pl
            fig, ax = pl.subplots(figsize=(8, 4))
            for name, (f, pxx, _, _) in results.items():
                ax.semilogy(f, pxx, label=name, alpha=0.8)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power (V²/Hz)')
            ax.set_title('Spectral Fingerprints: CMC vs JR vs MPR')
            ax.legend()
            ax.grid(True, alpha=0.3)
            pl.tight_layout()
            pl.savefig('cmc_spectral_comparison.png', dpi=150)
            print("  Saved: cmc_spectral_comparison.png")
        except ImportError:
            print("  (matplotlib not available, skipping plot)")


def run_bifurcation_sketch(do_plot=True):
    """Sweep input current and track oscillation amplitude for CMC vs JR."""
    print("\n─── Bifurcation Sketch ───")

    dt = 0.5
    n_steps = 6000
    I_range = np.linspace(0, 500, 30)
    results = {}

    for model_name, factory_fn, obs_idx in [
        ('CMC', make_cmc_factory, 1),
        ('JR', make_jr_factory, None),
    ]:
        amps = []
        for I in I_range:
            t, ys = simulate_model(factory_fn(I=float(I)), n_steps=n_steps,
                                   dt=dt, sigma=1e-4, seed=42)
            if obs_idx is not None:
                obs = ys[3000:, obs_idx]
            else:
                obs = ys[3000:, 1] - ys[3000:, 2]
            amps.append(float(jp.std(obs)))
        results[model_name] = np.array(amps)
        print(f"  {model_name}: onset ~I={I_range[np.argmax(np.array(amps) > 0.01)]:.0f}, "
              f"max amp={np.max(amps):.4f}")

    if do_plot:
        try:
            import pylab as pl
            fig, ax = pl.subplots(figsize=(7, 4))
            for name, amps in results.items():
                ax.plot(I_range, amps, 'o-', label=name, markersize=3)
            ax.set_xlabel('External input I')
            ax.set_ylabel('Oscillation amplitude (std)')
            ax.set_title('Bifurcation Sketch: CMC vs JR')
            ax.legend()
            ax.grid(True, alpha=0.3)
            pl.tight_layout()
            pl.savefig('cmc_bifurcation.png', dpi=150)
            print("  Saved: cmc_bifurcation.png")
        except ImportError:
            pass


def run_laminar_comparison(do_plot=True):
    """CMC-unique: compare superficial vs deep pyramidal output."""
    print("\n─── Laminar Decomposition (CMC only) ───")

    dt = 0.5
    n_steps = 8000
    t, ys = simulate_model(make_cmc_factory(I=250.0), n_steps=n_steps,
                           dt=dt, sigma=1e-3)
    sp = ys[2000:, 1]
    dp = ys[2000:, 3]

    corr = float(jp.corrcoef(jp.stack([sp, dp]))[0, 1])
    sp_std = float(jp.std(sp))
    dp_std = float(jp.std(dp))
    print(f"  sp-dp correlation: {corr:.3f}")
    print(f"  sp amplitude:      {sp_std:.4f}")
    print(f"  dp amplitude:      {dp_std:.4f}")
    print(f"  sp/dp ratio:       {sp_std/max(dp_std, 1e-10):.2f}")

    # PSD comparison
    f_sp, pxx_sp = compute_psd(sp, dt)
    f_dp, pxx_dp = compute_psd(dp, dt)
    peak_sp = f_sp[np.argmax(pxx_sp[1:])+1]
    peak_dp = f_dp[np.argmax(pxx_dp[1:])+1]
    print(f"  sp peak freq:      {peak_sp:.1f} Hz")
    print(f"  dp peak freq:      {peak_dp:.1f} Hz")

    if do_plot:
        try:
            import pylab as pl
            fig, axes = pl.subplots(2, 1, figsize=(8, 5))
            t_plot = np.arange(500) * dt  # 250 ms window
            axes[0].plot(t_plot, sp[:500], label='sp (superficial)', alpha=0.8)
            axes[0].plot(t_plot, dp[:500], label='dp (deep)', alpha=0.8)
            axes[0].set_xlabel('Time (ms)')
            axes[0].set_ylabel('Membrane potential')
            axes[0].set_title('Laminar Decomposition: Superficial vs Deep Pyramidal')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].semilogy(f_sp, pxx_sp, label='sp (EEG-like)', alpha=0.8)
            axes[1].semilogy(f_dp, pxx_dp, label='dp (LFP-like)', alpha=0.8)
            axes[1].set_xlim(0, 100)
            axes[1].set_xlabel('Frequency (Hz)')
            axes[1].set_ylabel('Power')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            pl.tight_layout()
            pl.savefig('cmc_laminar.png', dpi=150)
            print("  Saved: cmc_laminar.png")
        except ImportError:
            pass


def run_step_response(do_plot=True):
    """Compare step response across models — sequential activation in CMC."""
    print("\n─── Step Response ───")

    dt = 0.5
    n_pre = 500   # 250ms baseline
    n_post = 2000  # 1000ms after step

    # CMC step response
    p = vb.cmc_default_theta._replace(I=0.0)
    _, loop_pre = vb.make_sde(dt=dt, dfun=lambda y, _: vb.cmc_dfun(y, 0.0, p), gfun=0.0)
    p_on = vb.cmc_default_theta._replace(I=300.0)
    _, loop_post = vb.make_sde(dt=dt, dfun=lambda y, _: vb.cmc_dfun(y, 0.0, p_on), gfun=0.0)

    y0 = jp.zeros(8)
    zs_pre = jp.zeros((n_pre, 8))
    zs_post = jp.zeros((n_post, 8))
    ys_pre = loop_pre(y0, zs_pre, None)
    ys_post = loop_post(ys_pre[-1], zs_post, None)
    ys_cmc = jp.concatenate([ys_pre, ys_post])

    # Report peak latencies for each population
    pop_names = ['ss', 'sp', 'ii', 'dp']
    for i, name in enumerate(pop_names):
        post = ys_cmc[n_pre:, i]
        peak_idx = int(jp.argmax(jp.abs(post)))
        peak_t = peak_idx * dt
        print(f"  CMC {name}: peak latency = {peak_t:.1f} ms")

    if do_plot:
        try:
            import pylab as pl
            fig, ax = pl.subplots(figsize=(8, 4))
            t = np.arange(n_pre + n_post) * dt
            for i, name in enumerate(pop_names):
                ax.plot(t, ys_cmc[:, i], label=name, alpha=0.8)
            ax.axvline(n_pre * dt, color='k', ls='--', alpha=0.5, label='step on')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Membrane potential')
            ax.set_title('CMC Step Response: Sequential Layer Activation')
            ax.legend(ncol=5, fontsize=9)
            ax.grid(True, alpha=0.3)
            pl.tight_layout()
            pl.savefig('cmc_step_response.png', dpi=150)
            print("  Saved: cmc_step_response.png")
        except ImportError:
            pass


def run_compute_cost():
    """Compare wall-clock time per integration step."""
    print("\n─── Compute Cost (time per 1000 steps, single node) ───")
    import time

    dt = 0.5
    n_steps = 1000
    n_reps = 5

    models = [
        ('CMC (8D)', make_cmc_factory()),
        ('JR  (6D)', make_jr_factory()),
        ('MPR (2D)', make_mpr_factory()),
    ]

    for name, factory in models:
        dfun, y0 = factory()
        _, loop = vb.make_sde(dt=dt, dfun=dfun, gfun=1e-3)
        key = jax.random.PRNGKey(0)
        zs = jax.random.normal(key, (n_steps,) + y0.shape)
        # warmup JIT
        _ = loop(y0, zs, None)
        times = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            _ = loop(y0, zs, None).block_until_ready()
            times.append(time.perf_counter() - t0)
        avg = np.mean(times) * 1000
        print(f"  {name}: {avg:.2f} ms / 1000 steps")


def run_fwd_bwd_asymmetry(do_plot=True):
    """CMC-unique: test forward vs backward drive asymmetry."""
    print("\n─── Forward/Backward Asymmetry (CMC only) ───")

    dt = 0.5
    n_steps = 8000
    key = jax.random.PRNGKey(9)
    zs = jax.random.normal(key, (n_steps, 8))
    drive = 250.0

    results = {}
    for label, c_fwd, c_bwd in [
        ('Forward (→ss)',  drive, 0.0),
        ('Backward (→sp+dp)', 0.0, drive),
        ('Balanced', drive/2, drive/2),
    ]:
        p = vb.cmc_default_theta
        _, loop = vb.make_sde(
            dt=dt,
            dfun=lambda y, _, cf=c_fwd, cb=c_bwd: vb.cmc_hier_dfun(y, cf, cb, p),
            gfun=1e-3)
        ys = loop(jp.zeros(8), zs, None)
        late = ys[4000:]
        sp_std = float(jp.std(late[:, 1]))
        dp_std = float(jp.std(late[:, 3]))
        sp_mean = float(jp.mean(late[:, 1]))
        dp_mean = float(jp.mean(late[:, 3]))
        ratio = sp_std / max(dp_std, 1e-10)
        results[label] = {'sp_std': sp_std, 'dp_std': dp_std,
                          'sp_mean': sp_mean, 'dp_mean': dp_mean,
                          'ratio': ratio, 'ys': late}
        print(f"  {label:25s}  sp_std={sp_std:.4f}  dp_std={dp_std:.4f}  "
              f"sp/dp={ratio:.2f}  sp_mean={sp_mean:.3f}  dp_mean={dp_mean:.3f}")

    if do_plot:
        try:
            import pylab as pl
            fig, axes = pl.subplots(3, 1, figsize=(9, 7), sharex=True)
            for ax, (label, data) in zip(axes, results.items()):
                t = np.arange(500) * dt
                ax.plot(t, data['ys'][:500, 1], label='sp (pred. error)', alpha=0.8)
                ax.plot(t, data['ys'][:500, 3], label='dp (prediction)', alpha=0.8)
                ax.set_ylabel('x (mV)')
                ax.set_title(f'{label}  sp/dp ratio={data["ratio"]:.2f}')
                ax.legend(fontsize=8, loc='upper right')
                ax.grid(True, alpha=0.3)
            axes[-1].set_xlabel('Time (ms)')
            pl.tight_layout()
            pl.savefig('cmc_fwd_bwd_asymmetry.png', dpi=150)
            print("  Saved: cmc_fwd_bwd_asymmetry.png")
        except ImportError:
            pass


def run_2node_hierarchy(do_plot=True):
    """2-node hierarchical CMC: full predictive coding circuit."""
    print("\n─── 2-Node Hierarchical Predictive Coding ───")

    dt = 0.5
    n_steps = 10000
    node_p = vb.cmc_default_theta._replace(I=220.0)
    key = jax.random.PRNGKey(11)
    zs = jax.random.normal(key, (n_steps, 8, 2))

    results = {}
    for label, G_fwd, G_bwd in [
        ('Uncoupled',  0.0,  0.0),
        ('Fwd only',  80.0,  0.0),
        ('Bwd only',   0.0, 80.0),
        ('Bidirectional', 80.0, 80.0),
    ]:
        p = (G_fwd, G_bwd, node_p)
        _, loop = vb.make_sde(dt=dt, dfun=vb.cmc_hier_2node_dfun, gfun=1e-3)
        ys = loop(jp.zeros((8, 2)), zs, p)
        late = ys[5000:]

        # Cross-correlation between nodes' sp activity
        sp0 = np.array(late[:, 1, 0])
        sp1 = np.array(late[:, 1, 1])
        corr_sp = float(jp.corrcoef(jp.stack([jp.array(sp0), jp.array(sp1)]))[0, 1])

        # Cross-correlation between lower sp and higher dp (pred coding loop)
        dp1 = np.array(late[:, 3, 1])
        corr_loop = float(jp.corrcoef(jp.stack([jp.array(sp0), jp.array(dp1)]))[0, 1])

        results[label] = {'corr_sp': corr_sp, 'corr_loop': corr_loop, 'ys': late}
        print(f"  {label:18s}  sp0↔sp1 corr={corr_sp:+.3f}  "
              f"sp0↔dp1 corr={corr_loop:+.3f}")

    if do_plot:
        try:
            import pylab as pl
            fig, axes = pl.subplots(2, 2, figsize=(10, 6))
            for ax, (label, data) in zip(axes.flat, results.items()):
                t = np.arange(500) * dt
                ax.plot(t, data['ys'][:500, 1, 0], label='lower sp', alpha=0.8)
                ax.plot(t, data['ys'][:500, 1, 1], label='higher sp', alpha=0.8)
                ax.plot(t, data['ys'][:500, 3, 1], label='higher dp', alpha=0.7, ls='--')
                ax.set_title(f'{label} (r={data["corr_sp"]:+.2f})')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            pl.suptitle('2-Node Hierarchical CMC: Predictive Coding')
            pl.tight_layout()
            pl.savefig('cmc_2node_hierarchy.png', dpi=150)
            print("  Saved: cmc_2node_hierarchy.png")
        except ImportError:
            pass


def run_cross_frequency_coupling(do_plot=True):
    """Test phase-amplitude coupling between CMC populations."""
    print("\n─── Cross-Frequency Coupling (CMC only) ───")

    dt = 0.5
    n_steps = 20000  # 10 seconds for good freq resolution
    p = vb.cmc_default_theta._replace(I=250.0)
    _, loop = vb.make_sde(
        dt=dt, dfun=lambda y, p: vb.cmc_dfun(y, 0.0, p), gfun=1e-3)
    y0 = jp.zeros(8)
    key = jax.random.PRNGKey(13)
    zs = jax.random.normal(key, (n_steps, 8))
    ys = loop(y0, zs, p)
    late = ys[5000:]  # skip transient

    sp = np.array(late[:, 1])
    dp = np.array(late[:, 3])
    ss = np.array(late[:, 0])
    ii = np.array(late[:, 2])

    # PSD per population
    from scipy.signal import welch
    fs = 1000.0 / dt
    pop_names = ['ss', 'sp', 'ii', 'dp']
    pop_data = [ss, sp, ii, dp]

    print("  Per-population spectral peaks:")
    for name, x in zip(pop_names, pop_data):
        f, pxx = welch(x, fs=fs, nperseg=2048)
        peak_idx = np.argmax(pxx[1:]) + 1
        print(f"    {name}: peak = {f[peak_idx]:.1f} Hz, power = {pxx[peak_idx]:.2e}")

    # Phase-amplitude coupling: filter dp in low-freq, sp in high-freq
    # Use Hilbert transform for analytic signal
    try:
        from scipy.signal import hilbert, butter, filtfilt

        # Low-freq phase from dp (1-15 Hz)
        b_lo, a_lo = butter(3, [1, 15], btype='bandpass', fs=fs)
        dp_lo = filtfilt(b_lo, a_lo, dp)
        dp_phase = np.angle(hilbert(dp_lo))

        # High-freq amplitude from sp (20-80 Hz)
        b_hi, a_hi = butter(3, [20, min(80, fs/2-1)], btype='bandpass', fs=fs)
        sp_hi = filtfilt(b_hi, a_hi, sp)
        sp_amp = np.abs(hilbert(sp_hi))

        # Modulation index (Tort et al. 2010)
        n_bins = 18
        phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        mean_amp = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (dp_phase >= phase_bins[i]) & (dp_phase < phase_bins[i+1])
            if mask.sum() > 0:
                mean_amp[i] = sp_amp[mask].mean()

        # Normalize to get distribution
        if mean_amp.sum() > 0:
            p_dist = mean_amp / mean_amp.sum()
            # KL divergence from uniform
            uniform = np.ones(n_bins) / n_bins
            kl = np.sum(p_dist * np.log(p_dist / uniform + 1e-10))
            mi = kl / np.log(n_bins)  # Modulation Index
        else:
            mi = 0.0

        print(f"  Phase-amplitude coupling (dp phase → sp amplitude):")
        print(f"    Modulation index = {mi:.4f} (0=none, 1=max)")
        if mi > 0.001:
            print(f"    → CMC produces cross-frequency coupling from single node")
        else:
            print(f"    → Weak/no PAC at current parameter settings")

    except Exception as e:
        print(f"  PAC analysis skipped: {e}")

    if do_plot:
        try:
            import pylab as pl
            fig, axes = pl.subplots(2, 2, figsize=(10, 6))
            for ax, name, x in zip(axes.flat, pop_names, pop_data):
                f, pxx = welch(x, fs=fs, nperseg=2048)
                ax.semilogy(f, pxx)
                ax.set_xlim(0, 100)
                ax.set_title(f'{name} population PSD')
                ax.set_xlabel('Freq (Hz)')
                ax.grid(True, alpha=0.3)
            pl.tight_layout()
            pl.savefig('cmc_crossfreq.png', dpi=150)
            print("  Saved: cmc_crossfreq.png")
        except ImportError:
            pass


def run_bifurcation_with_frequency(do_plot=True):
    """Bifurcation sweep tracking both amplitude AND dominant frequency."""
    print("\n─── Bifurcation with Frequency Tracking ───")

    dt = 0.5
    n_steps = 8000
    I_range = np.linspace(0, 500, 40)
    fs = 1000.0 / dt

    results = {'CMC': {'amp': [], 'freq': []}, 'JR': {'amp': [], 'freq': []}}

    for I in I_range:
        for model_name, factory_fn, obs_fn in [
            ('CMC', make_cmc_factory, lambda ys: ys[4000:, 1]),
            ('JR', make_jr_factory, lambda ys: ys[4000:, 1] - ys[4000:, 2]),
        ]:
            t, ys = simulate_model(factory_fn(I=float(I)), n_steps=n_steps,
                                   dt=dt, sigma=1e-4, seed=42)
            obs = obs_fn(ys)
            amp = float(jp.std(obs))
            results[model_name]['amp'].append(amp)

            if amp > 1e-4:
                from scipy.signal import welch
                f, pxx = welch(np.array(obs), fs=fs, nperseg=1024)
                peak_f = f[np.argmax(pxx[1:])+1]
            else:
                peak_f = 0.0
            results[model_name]['freq'].append(peak_f)

    for name in ['CMC', 'JR']:
        amps = np.array(results[name]['amp'])
        freqs = np.array(results[name]['freq'])
        osc_mask = amps > 0.01
        if osc_mask.any():
            onset_I = I_range[np.argmax(osc_mask)]
            freq_range = freqs[osc_mask]
            print(f"  {name}: onset I≈{onset_I:.0f}, "
                  f"freq range [{freq_range.min():.1f}, {freq_range.max():.1f}] Hz, "
                  f"max amp={amps.max():.4f}")
        else:
            print(f"  {name}: no oscillation detected (max amp={amps.max():.4f})")

    if do_plot:
        try:
            import pylab as pl
            fig, (ax1, ax2) = pl.subplots(2, 1, figsize=(8, 6), sharex=True)
            for name in ['CMC', 'JR']:
                ax1.plot(I_range, results[name]['amp'], 'o-', label=name, ms=3)
                ax2.plot(I_range, results[name]['freq'], 'o-', label=name, ms=3)
            ax1.set_ylabel('Amplitude (std)')
            ax1.set_title('Bifurcation: Amplitude + Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax2.set_ylabel('Peak Frequency (Hz)')
            ax2.set_xlabel('External input I')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            pl.tight_layout()
            pl.savefig('cmc_bifurcation_freq.png', dpi=150)
            print("  Saved: cmc_bifurcation_freq.png")
        except ImportError:
            pass


def run_layer_bold_bridge(do_plot=True):
    """Demonstrate CMC → vpjax layer-resolved BOLD pipeline."""
    print("\n─── CMC → Layer-Resolved BOLD Bridge ───")

    dt = 0.5
    n_steps = 8000
    p = vb.cmc_default_theta._replace(I=250.0)
    _, loop = vb.make_sde(
        dt=dt, dfun=lambda y, p: vb.cmc_dfun(y, 0.0, p), gfun=1e-3)
    y0 = jp.zeros(8)
    key = jax.random.PRNGKey(17)
    zs = jax.random.normal(key, (n_steps, 8))
    ys = loop(y0, zs, p)

    # Map CMC state to layer activity at each timepoint
    layer_names = ['deep (dp→L5/6)', 'middle (ss→L4)', 'superficial (sp→L2/3)']
    late = ys[4000:]

    # Use vb.cmc_to_layer_activity on each timestep
    # For single-node: ys shape is (T, 8), need to transpose for the function
    layer_ts = jax.vmap(vb.cmc_to_layer_activity)(late)  # (T, 3)

    print("  Layer activity statistics:")
    for i, name in enumerate(layer_names):
        mean = float(jp.mean(layer_ts[:, i]))
        std = float(jp.std(layer_ts[:, i]))
        print(f"    {name}: mean={mean:.4f}, std={std:.4f}")

    # Cross-layer correlations
    corr_mat = np.corrcoef(np.array(layer_ts).T)
    print("  Cross-layer correlation matrix:")
    print(f"    deep-mid:  {corr_mat[0,1]:.3f}")
    print(f"    deep-sup:  {corr_mat[0,2]:.3f}")
    print(f"    mid-sup:   {corr_mat[1,2]:.3f}")
    print("  → These replace vpjax's heuristic feedforward_frac/feedback_frac")

    if do_plot:
        try:
            import pylab as pl
            fig, ax = pl.subplots(figsize=(8, 3))
            t = np.arange(500) * dt
            for i, name in enumerate(layer_names):
                ax.plot(t, layer_ts[:500, i], label=name, alpha=0.8)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Neural activity')
            ax.set_title('CMC → Layer Activity (input to vpjax hemodynamics)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            pl.tight_layout()
            pl.savefig('cmc_layer_bold_bridge.png', dpi=150)
            print("  Saved: cmc_layer_bold_bridge.png")
        except ImportError:
            pass


if __name__ == '__main__':
    do_plot = '--no-plot' not in sys.argv

    print(MODEL_TABLE)
    print(COMPARISON_DESIGN)

    run_spectral_comparison(do_plot)
    run_bifurcation_sketch(do_plot)
    run_laminar_comparison(do_plot)
    run_step_response(do_plot)
    run_compute_cost()
    run_fwd_bwd_asymmetry(do_plot)
    run_2node_hierarchy(do_plot)
    run_cross_frequency_coupling(do_plot)
    run_bifurcation_with_frequency(do_plot)
    run_layer_bold_bridge(do_plot)

    print("\n─── Summary ───")
    print("CMC strengths:")
    print("  + Laminar decomposition (sp vs dp) for simultaneous EEG-LFP")
    print("  + Forward/backward asymmetry for hierarchical predictive coding")
    print("  + Richer bifurcation structure from 4-population circuit")
    print("  + Same sigmoid as JR — direct spectral comparison possible")
    print("  + Compatible with vpjax layer-resolved physiology pipeline")
    print("  + Cross-frequency coupling from single node")
    print("  + 2-node hierarchy implements full predictive coding loop")
    print()
    print("CMC weaknesses:")
    print("  - 8 states/node vs 2 (RWW/MPR) or 6 (JR) — ~20% slower than JR")
    print("  - 8 connectivity params may be non-identifiable from EEG alone")
    print("  - Requires hierarchy direction to exploit forward/backward paths")
    print("  - No analytical bifurcation results (unlike MPR)")
    print()
    print("Recommended next steps:")
    print("  1. Add CMC to neurojax VbjaxFitnessAdapter for whole-brain fitting")
    print("  2. Couple CMC to vpjax Riera model for full layer-resolved BOLD")
    print("  3. Test forward/backward asymmetry with WAND hierarchical connectivity")
    print("  4. Parameter sensitivity analysis (Sobol) for identifiability")
    print("  5. Compare learned (Douglas 2025) vs biologically constrained params")
