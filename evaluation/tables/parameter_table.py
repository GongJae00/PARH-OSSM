"""Rows of the Supplementary parameter table, read from the code that produced the reported runs.

Each value is read from a fresh PARH-OSSM head (the materializer builds the head without parameters and
the runs set no RESPYRE_PARH_* override; the pickle meta of the frozen runs carries the same values) or
from the materializer module, so the typeset table cannot drift from the code. The last column names the
constants or parameter fields in the code, which stay findable when lines move. The last block lists the
terms the code defines but
the reported configuration never applies, with the condition that keeps them off.

usage: parameter_table.py > supplementary_parameter_table.tex
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from estimate.filters.parh_ossm import oscillator_PARH_OSSM
import estimate.readout.multichannel_fusion as materializer

head = oscillator_PARH_OSSM()

WARMUP_FIELDS = ("max_g_h1", "max_g_h2", "prior_strength", "max_lag_sec")

# (block, symbol, meaning, source, names); source "head" reads a head constant, "params" a field of the
# oscillator parameters, "materializer" a module constant of the readout, and "warmup:<family>" the bounds the
# head itself derives for that family's warm-up fit (_obs_cal_family_prior takes the smallest of several caps).
ROWS = [
    ("State and process noise", r"$f_{\min}$, $f_{\max}$", "respiratory band (Hz)", "params", ("f_min", "f_max")),
    ("State and process noise", r"$\tau_{\mathrm{env}}$", "damping time constant of the oscillator (s)", "params", ("tau_env",)),
    ("State and process noise", r"$q_x$", "base process-noise variance", "params", ("qx",)),
    ("State and process noise", "", "process-noise scale of the first and second harmonic", "head", ("Q_HARMONIC1_SCALE", "Q_HARMONIC2_SCALE")),
    ("State and process noise", "", "process noise of the baseline position and velocity", "head", ("Q_BASELINE_POS", "Q_BASELINE_VEL")),
    ("State and process noise", "", "process noise of the residual position and velocity", "head", ("Q_RESIDUAL_POS", "Q_RESIDUAL_VEL")),
    ("State and process noise", "", "damping time constant of the residual (s)", "head", ("TAU_RESIDUAL_SEC",)),
    ("State and process noise", "", "gain of the frequency-change score in the harmonic noise; gain of the residual noise where the oscillator support is low", "head", ("Q_DYN_GAMMA", "Q_APER_GAMMA")),
    ("Measurement variance", "", "time constant of the innovation variance (s)", "head", ("TAU_R_SEC",)),
    ("Measurement variance", "", r"lower bound of $R^{\mathrm{self}}$ as a fraction of its initial value", "head", ("R_ANCHOR_FRAC",)),
    ("Measurement variance", "", "robust-scale factor and floor of the initial variance", "params", ("rv_mad_scale", "rv_floor")),
    ("Measurement variance", r"$\nu_t$", "range of the Student-$t$ degrees of freedom", "head", ("NU_MIN", "NU_MAX")),
    ("Measurement variance", "", "time constant of the kurtosis estimate (s); variational iterations", "head", ("TAU_KAPPA_SEC", "VB_ITERS")),
    ("Trust", "", "innovation and jump sensitivity of the frame quality; its floor", "head", ("Q_OBS_ROBUST_Z_SCALE", "Q_OBS_JUMP_SCALE", "Q_OBS_MIN")),
    ("Trust", "", "time constant (s), temperature and per-channel floor of the trust mixture", "head", ("DYNAMIC_MIXTURE_TAU_SEC", "DYNAMIC_MIXTURE_TEMPERATURE", "DYNAMIC_MIXTURE_MIN_WEIGHT")),
    ("Trust", "", "exponent of the relative-variance score; exponent of the rate-observability score", "head", ("DYNAMIC_MIXTURE_R_WEIGHT", "RATE_OBS_POWER")),
    ("Trust", "", "floors of the family context and of the global quality", "head", ("DYNAMIC_MIXTURE_CONTEXT_FLOOR", "DYNAMIC_MIXTURE_GLOBAL_QUALITY_FLOOR")),
    ("Trust", "", "family prior of the windowed regime context", "materializer", ("REGIME_GROUP_PRIOR",)),
    ("Trust", "", "floor of the windowed regime context", "materializer", ("REGIME_CHANNEL_FLOOR",)),
    ("Trust", "", "exponent of the role score in the family context", "head", ("TARGET_OBS_ROLE_POWER",)),
    ("Target observability and nuisance", r"$\lambda_n$", "variance inflation per unit nuisance score", "head", ("TARGET_OBS_NUISANCE_R_SCALE",)),
    ("Target observability and nuisance", "", "floor of the observability factor on frame quality and on the frequency-change score", "head", ("TARGET_OBS_TRUST_FLOOR", "TARGET_OBS_QDYN_FLOOR")),
    ("Target observability and nuisance", "", "residual identifiability gate: floor of its trust and observability factors, scale of its nuisance factor", "head", ("RESIDUAL_GUARD_TRUST_FLOOR", "RESIDUAL_GUARD_NUISANCE_SCALE")),
    ("Oscillator frequency", "", "update interval (s), confirmations needed, largest step (Hz)", "head", ("FREQ_UPDATE_INTERVAL_SEC", "FREQ_CONFIRM_COUNT", "FREQ_MAX_STEP_HZ")),
    ("Oscillator frequency", "", "window of the candidate-frequency median (s)", "head", ("FREQ_RESCUE_WINDOW_SEC",)),
    ("Oscillator frequency", "", "update toward a departing consensus: largest consensus s.d. (Hz), minimum helper support, minimum frequency-change score, minimum departure (Hz), confirmations needed, largest step (Hz)", "head", ("FREQ_RESCUE_HELPER_STD_MAX_HZ", "FREQ_RESCUE_MIN_SUPPORT", "FREQ_RESCUE_MIN_QDYN", "FREQ_RESCUE_MIN_MISMATCH_HZ", "FREQ_RESCUE_CONFIRM_COUNT", "FREQ_RESCUE_MAX_STEP_HZ")),
    ("Oscillator frequency", "", "helper frequency: window of the rate-observability scores (s), blend, largest step (Hz)", "head", ("RATE_OBS_WINDOW_SEC", "RATE_OBS_HELPER_BLEND", "RATE_OBS_HELPER_MAX_STEP_HZ")),
    ("Warm-up calibration", "", "warm-up window and skipped start (s)", "head", ("OBS_CAL_WARMUP_SEC", "OBS_CAL_SKIP_SEC")),
    ("Warm-up calibration", "", "ridge penalty of the gain fit", "head", ("OBS_CAL_RIDGE",)),
    ("Warm-up calibration", "", "acceptance: minimum fit correlation, maximum normalized RMSE", "head", ("OBS_CAL_MIN_FIT_CORR", "OBS_CAL_MAX_FIT_RMSE_NORM")),
    ("Warm-up calibration", "", "P1D: largest first- and second-harmonic gain, prior strength, largest lag (s)", "warmup:profile1d_quadratic", WARMUP_FIELDS),
    ("Rate readout", "", "family prior of the per-window readout score", "materializer", ("RATE_TIMING_GROUP_PRIOR",)),
    ("Rate readout", "", "OF helper blend of the state rate: weight, minimum helper support, minimum frequency-change score, minimum mismatch (Hz)", "head", ("OUTPUT_RATE_BLEND_ALPHA", "OUTPUT_RATE_MIN_SUPPORT", "OUTPUT_RATE_MIN_QDYN", "OUTPUT_RATE_MIN_MISMATCH_HZ")),
    ("Rate readout", "", "largest blend of the phase-anchored waveform readout", "head", ("PHASE_MORPH_MAX_BLEND",)),
    ("Rate readout", "", "phase bins of the waveform template", "head", ("PHASE_MORPH_BINS",)),
    ("Defined but not applied", r"$c_k$", "warm-up correction of the variance; requires fit correlation $\\geq$ 0.975 and fit RMSE $\\leq$ 0.20, which no observation met", "head", ("FAMILY_CONFIDENCE_R_SCALE",)),
    ("Defined but not applied", "", "trust floor under $c_k$; outside the mixture branch the method uses", "head", ("FAMILY_CONFIDENCE_PI_FLOOR",)),
    ("Defined but not applied", r"$\lambda_a$", "variance inflation per unit abstention; $a_{k,t}=0$ because the role-specific abstention prior is not built", "head", ("STATE_ROLE_ABSTAIN_R_SCALE",)),
    ("Defined but not applied", "", "residual noise added for observation content the oscillators leave unexplained; its gain is zero", "head", ("Q_APER_OBS_GAMMA",)),
]

SOURCES = {"head": head, "params": head.params, "materializer": materializer}

# Readout groups of the three observations of the bank, under their names in the paper
# (estimate/readout/reliability_priors.py maps each observation to its group); the code also
# defines priors for the groups of the observations the bank leaves out, which no reported run reads.
GROUP_NAMES = {"G_OF": "OF", "G_P1D_morph": "P1D", "G_Torso": "torso"}


def value_text(value):
    if isinstance(value, dict):
        return ", ".join(f"{GROUP_NAMES[group]} {weight:g}" for group, weight in value.items() if group in GROUP_NAMES)
    return f"{value:g}"


def code_names(names):
    # a break is allowed after each underscore, so long constant names wrap inside the column
    return ", ".join(rf"\texttt{{{name.replace('_', chr(92) + '_' + chr(92) + 'allowbreak{}')}}}" for name in names)


block = None
for row_block, symbol, meaning, source, names in ROWS:
    if row_block != block:
        block = row_block
        print(rf"\multicolumn{{4}}{{l}}{{\textit{{{block}}}}} \\")
    if source.startswith("warmup:"):
        family = source.split(":")[1]
        bounds = head._obs_cal_family_prior(family)
        values = " / ".join(value_text(bounds[name]) for name in names)
        location = code_names([f"_obs_cal_family_prior('{family}')"])
    else:
        values = " / ".join(value_text(getattr(SOURCES[source], name)) for name in names)
        location = code_names(names)
    print(rf"{symbol} & {meaning} & {values} & {location} \\")
