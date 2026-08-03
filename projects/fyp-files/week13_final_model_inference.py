"""
================================================================================
FINAL MODEL  --  CNN-BiLSTM-Attention + HistGradientBoosting fusion
                  Inference only. Self-contained, no external project files.
================================================================================

WHAT THIS IS
    The final production fault-classification model, incorporating the two
    findings the FYP log's Week 11 and Week 12 entries describe:

    Week 11 (generalisation directive)
        Trained POOLED across all three networks (ieee39 + ieee14 + ieee9),
        not on one system -- per the supervisor's directive that the
        priority is a genuinely generalised model, not a higher own-system
        number.

    Week 12 (architecture search + HGB finding)
        The ablation study showed CNN -> +BiLSTM -> +Attention is the
        correct trunk (BiLSTM +1.0 pt, Attention +1.9 pt; +MultiTask cost
        -2.9 pt, so classification heads only, no multi-task heads), and
        that a HistGradientBoosting baseline on the 47 engineered scalars
        matched or beat the neural candidates. This model is exactly that:
        the CNN-BiLSTM-Attention trunk fused with an HGB branch over the
        same 47 scalars, so it gets the trunk's noise-robustness AND the
        HGB branch's accuracy rather than choosing one or the other.

    NOT adopted: Week 12 also flagged a 1-pre+2-post-cycle window scoring
    higher in that week's sweep than the validated 0.5-pre+2-post default.
    Adopting it would mean re-harmonising every source simulation at a new
    window size without re-running the validation gate the current window
    already passed, so it was left as a flagged follow-up rather than
    silently changed underneath a "final" model. This artifact still uses
    the validated 0.5+2.0 cycle window.

TRAINED ON
    706 rows / 508 independent physical simulations, pooled across
    ieee39 + ieee14 + ieee9. Single production fit, seed 42, on ALL of the
    pooled corpus (not a held-out fold -- this is a deployment build).

    The rigorously CROSS-VALIDATED number for this exact architecture is
    the leave-one-network-out mean from the underlying hgb_fusion_v3 study:
    0.723 +/- 0.190 accuracy over 15 folds (5 seeds x 3 held-out networks).
    That is the number to quote for generalisation claims. This artifact is
    what gets shipped from that study.

EXPECTED INPUT
    Sampling rate      any; the record is resampled onto the common grid
    Common grid        1200 Hz  =  20 samples/cycle at 60 Hz
    Window             0.5 pre-fault + 2.0 post-fault cycles = 50 samples
    Channels           6, sending end, ordered [Va Vb Vc Ia Ib Ic]
    Units              irrelevant -- every feature is normalised by the
                       record's own healthy RMS, so SI and per-unit both work

OUTPUT
    One of 11 canonical classes: ab abc abcg abg ac acg ag bc bcg bg cg
    plus a per-class probability vector and the abc/abcg-merged label a
    relay would actually act on (a balanced 3-phase fault drives no
    zero-sequence current whether or not ground is involved, and a relay
    trips all three phases either way).

ARTIFACTS  (expected in the SAME DIRECTORY as this file)
    week13_final_model_seed42.weights.h5     Keras weights, 23,138 params
    week13_final_model_hgb_scalers.joblib    HGB model + scalers + config

ARCHITECTURE DIAGRAM
    week13_final_model_flowchart.png

TRAINING CODE (not included here -- this file is inference only)
    week13_final_model_train.py
================================================================================
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sstats
from scipy.signal import resample_poly

# ------------------------------------------------------------------ constants
F0_HZ = 60.0
TARGET_SPC = 20                 # 1200 Hz common grid
PRE_CYCLES = 0.5
POST_CYCLES = 2.0
JITTER = 4
WIN_LEN = int(round((PRE_CYCLES + POST_CYCLES) * TARGET_SPC))   # 50
STORE_LEN = WIN_LEN + 2 * JITTER                                # 58
PRE_SAMPLES = int(round(PRE_CYCLES * TARGET_SPC))               # 10

SEND_COLS = ["Va_A", "Va_B", "Va_C", "Ia_A", "Ia_B", "Ia_C"]
PU_COLS = ["Va_pu", "Vb_pu", "Vc_pu", "Ia_pu", "Ib_pu", "Ic_pu"]
CANONICAL_CLASSES = ["ab", "abc", "abcg", "abg", "ac", "acg", "ag",
                     "bc", "bcg", "bg", "cg"]
MERGE_FOR_REPORT = {"abcg": "abc"}
N_CLASSES = len(CANONICAL_CLASSES)
SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, f"week13_final_model_seed{SEED}.weights.h5")
BUNDLE_PATH = os.path.join(HERE, "week13_final_model_hgb_scalers.joblib")


# ============================================================ harmonisation
def resample_to_spc(x, spc_src, spc_dst):
    if spc_src == spc_dst:
        return x.copy()
    g = np.gcd(int(spc_dst), int(spc_src))
    return resample_poly(x, up=int(spc_dst) // g, down=int(spc_src) // g, axis=0)


def superimposed_lag(x, spc):
    out = np.zeros_like(x)
    out[spc:] = x[spc:] - x[:-spc]
    return out


def smooth_box(x, win):
    return np.convolve(x, np.ones(win) / win, mode="same")


def disturbance_metric(raw, spc):
    """Disturbance trace from the three superimposed currents. Current rather
    than voltage: a fault multiplies current but may drop voltage by only a
    third, so current gives a clearer inception edge across all Rf."""
    d = np.zeros(raw.shape[0])
    for k in (3, 4, 5):
        d += np.abs(superimposed_lag(raw[:, k], spc))
    return smooth_box(d, max(3, spc // 2))


def find_healthy_window(d, spc):
    win = min(2 * spc, len(d))
    csum = np.convolve(d, np.ones(win), mode="valid")
    start = int(np.argmin(csum))
    return start, start + win


def find_inception(d, spc):
    peak = float(d.max())
    if peak <= 0:
        return None, None
    lo, hi = find_healthy_window(d, spc)
    quiet = d[lo:hi]
    base = float(np.median(quiet))
    mad = float(np.median(np.abs(quiet - base))) + 1e-12
    thr = max(0.20 * peak, base + 8.0 * mad)
    above = d > thr
    debounce = max(2, spc // 4)
    for i in range(len(above) - debounce):
        if above[i] and above[i:i + debounce].all():
            return i, int(np.where(above)[0].max())
    return None, None


def fit_fundamental(x, idx, spc):
    w = 2.0 * np.pi / spc
    n = idx.astype(float)
    M = np.stack([np.cos(w * n), np.sin(w * n), np.ones_like(n)], axis=1)
    coef, *_ = np.linalg.lstsq(M, x[idx], rcond=None)
    return coef


def eval_fundamental(coef, n, spc):
    w = 2.0 * np.pi / spc
    return coef[0] * np.cos(w * n) + coef[1] * np.sin(w * n) + coef[2]


def dft_phasor(x, spc):
    n = np.arange(spc)
    c, s = np.cos(2 * np.pi * n / spc), np.sin(2 * np.pi * n / spc)
    re = np.convolve(x, c[::-1], mode="same") * 2.0 / spc
    im = -np.convolve(x, s[::-1], mode="same") * 2.0 / spc
    return re + 1j * im


def sequence_components(pa, pb, pc):
    a = np.exp(1j * 2 * np.pi / 3)
    return ((pa + a * pb + a * a * pc) / 3.0,
            (pa + a * a * pb + a * pc) / 3.0,
            (pa + pb + pc) / 3.0)


def clarke(xa, xb, xc):
    return ((2.0 / 3.0) * (xa - 0.5 * xb - 0.5 * xc),
            (1.0 / np.sqrt(3.0)) * (xb - xc),
            (xa + xb + xc) / 3.0)


def harmonise_raw(raw, fs, spc_dst=TARGET_SPC):
    """Raw 6-channel record -> a fixed-length, dimensionless window.

    1. resample onto the common samples-per-cycle grid (anti-aliased)
    2. detect fault inception from the superimposed current
    3. cut a fixed window around inception
    4. divide by the record's OWN healthy RMS -- this is what makes SI and
       per-unit records comparable without knowing either base
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 6:
        raise ValueError(f"expected (N,>=6) array, got {raw.shape}")
    raw = raw[:, :6]

    spc_src = int(round(fs / F0_HZ))
    x = resample_to_spc(raw, spc_src, spc_dst)
    spc, n_tot = spc_dst, x.shape[0]

    d = disturbance_metric(x, spc)
    onset, _ = find_inception(d, spc)
    detected = onset is not None
    if not detected:
        onset = n_tot // 2

    lo_h, hi_h = find_healthy_window(d, spc)
    idx_h = np.arange(lo_h, hi_h)

    v_rms = float(np.sqrt(np.mean(x[idx_h, 0:3] ** 2)))
    i_rms = float(np.sqrt(np.mean(x[idx_h, 3:6] ** 2)))
    v_base = v_rms if v_rms > 1e-12 else 1.0
    i_base = i_rms if i_rms > 1e-12 else 1.0

    n_all = np.arange(n_tot, dtype=float)
    ref = np.zeros_like(x)
    for ch in range(6):
        ref[:, ch] = eval_fundamental(
            fit_fundamental(x[:, ch], idx_h, spc), n_all, spc)

    start = onset - PRE_SAMPLES - JITTER
    stop = start + STORE_LEN
    pad_lo, pad_hi = max(0, -start), max(0, stop - n_tot)
    s0, s1 = max(0, start), min(n_tot, stop)
    win, ref_w = x[s0:s1], ref[s0:s1]
    if pad_lo or pad_hi:
        win = np.pad(win, ((pad_lo, pad_hi), (0, 0)), mode="edge")
        ref_w = np.pad(ref_w, ((pad_lo, pad_hi), (0, 0)), mode="edge")

    win, ref_w = win.astype(np.float64), ref_w.astype(np.float64)
    win[:, 0:3] /= v_base
    win[:, 3:6] /= i_base
    ref_w[:, 0:3] /= v_base
    ref_w[:, 3:6] /= i_base
    return win, ref_w, {"fault_detected": bool(detected), "onset_idx": int(onset),
                        "v_base": v_base, "i_base": i_base,
                        "edge_padded": bool(pad_lo or pad_hi)}


def centre_crop(win, ref):
    """58-sample stored window -> the 50-sample window the model expects."""
    return win[JITTER:JITTER + WIN_LEN], ref[JITTER:JITTER + WIN_LEN]


# ============================================================ features
def _ratio(num, den, cap=8.0):
    return np.clip(num / (np.abs(den) + 1e-6), -cap, cap)


def build_sequence_and_scalar_features(win, ref, spc=TARGET_SPC):
    """39 sequence channels (fed to the CNN branch) and 47 dimensionless
    scalars (fed to the HGB branch). None carries an absolute scale, a
    sampling rate, or a bus/line name -- each of those is a route back to
    dataset identity that the model would otherwise use as a shortcut."""
    win = np.asarray(win, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    va, vb, vc, ia, ib, ic = [win[:, k] for k in range(6)]
    ra, rb, rc, ria, rib, ric = [ref[:, k] for k in range(6)]

    dva, dvb, dvc = va - ra, vb - rb, vc - rc
    dia, dib, dic = ia - ria, ib - rib, ic - ric

    pv = [dft_phasor(s, spc) for s in (va, vb, vc)]
    pi = [dft_phasor(s, spc) for s in (ia, ib, ic)]
    v1, v2, v0 = sequence_components(*pv)
    i1, i2, i0 = sequence_components(*pi)
    v1m, v2m, v0m = np.abs(v1), np.abs(v2), np.abs(v0)
    i1m, i2m, i0m = np.abs(i1), np.abs(i2), np.abs(i0)

    val, vbe, _ = clarke(va, vb, vc)
    ial, ibe, _ = clarke(ia, ib, ic)
    dial, dibe, _ = clarke(dia, dib, dic)

    zap = [_ratio(np.abs(pv[k]), np.abs(pi[k]), cap=10.0) for k in range(3)]
    zab = _ratio(np.abs(pv[0] - pv[1]), np.abs(pi[0] - pi[1]), cap=10.0)
    zbc = _ratio(np.abs(pv[1] - pv[2]), np.abs(pi[1] - pi[2]), cap=10.0)
    zca = _ratio(np.abs(pv[2] - pv[0]), np.abs(pi[2] - pi[0]), cap=10.0)

    env = lambda s: np.sqrt(smooth_box(s ** 2, spc))
    p_inst = val * ial + vbe * ibe
    q_inst = vbe * ial - val * ibe

    seq = np.stack([
        va, vb, vc, ia, ib, ic,
        dva, dvb, dvc, dia, dib, dic,
        v1m, v2m, v0m, i1m, i2m, i0m,
        _ratio(v2m, v1m), _ratio(v0m, v1m),
        _ratio(i2m, i1m), _ratio(i0m, i1m),
        val, vbe, ial, ibe, dial, dibe,
        zap[0], zap[1], zap[2], zab, zbc, zca,
        env(dia), env(dib), env(dic), p_inst, q_inst,
    ], axis=1).astype(np.float32)

    tail = slice(-spc, None)
    scal = []
    for m, r in ((ia, ria), (ib, rib), (ic, ric)):
        scal.append(_ratio(np.sqrt(np.mean(m[tail] ** 2)),
                           np.sqrt(np.mean(r[tail] ** 2)), cap=20.0))
    for m, r in ((va, ra), (vb, rb), (vc, rc)):
        scal.append(_ratio(np.sqrt(np.mean(m[tail] ** 2)),
                           np.sqrt(np.mean(r[tail] ** 2)), cap=5.0))
    di_rms = np.array([np.sqrt(np.mean(s[tail] ** 2)) for s in (dia, dib, dic)])
    dv_rms = np.array([np.sqrt(np.mean(s[tail] ** 2)) for s in (dva, dvb, dvc)])
    scal.extend(list(di_rms / (di_rms.max() + 1e-9)))
    scal.extend(list(dv_rms / (dv_rms.max() + 1e-9)))
    i1f = float(np.median(i1m[tail])) + 1e-9
    v1f = float(np.median(v1m[tail])) + 1e-9
    scal.extend([float(np.median(i2m[tail])) / i1f,
                 float(np.median(i0m[tail])) / i1f,
                 float(np.median(v2m[tail])) / v1f,
                 float(np.median(v0m[tail])) / v1f])
    ang = lambda z: float(np.angle(np.mean(z[tail])))
    wrap = lambda a: float(np.arctan2(np.sin(a), np.cos(a)))
    a_i1, a_i2, a_i0 = ang(i1), ang(i2), ang(i0)
    scal.extend([wrap(a_i2 - a_i1), wrap(a_i0 - a_i1), wrap(a_i0 - a_i2)])
    ipp = np.array([np.sqrt(np.mean((ia - ib)[tail] ** 2)),
                    np.sqrt(np.mean((ib - ic)[tail] ** 2)),
                    np.sqrt(np.mean((ic - ia)[tail] ** 2))])
    scal.extend(list(ipp / (ipp.max() + 1e-9)))
    scal.extend([float(np.median(z[tail])) for z in zap])
    scal.extend([float(np.median(z[tail])) for z in (zab, zbc, zca)])
    try:
        import pywt
        for sig in (dial, dibe, val):
            e = np.array([float(np.sum(c ** 2))
                          for c in pywt.wavedec(sig, "db4", level=3)])
            scal.extend(list(e / (e.sum() + 1e-12)))
    except Exception:
        scal.extend([0.0] * 12)
    for sig in (dial, dibe):
        n = len(sig)
        mag = np.abs(np.fft.rfft(sig * np.hanning(n)))
        p = mag ** 2
        p = p / (p.sum() + 1e-12)
        scal.append(float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p))))
        scal.append(float(np.clip(mag[4:].max() / (mag[1:4].max() + 1e-9), 0, 5)))
    scal.extend([float(sstats.skew(dial)), float(sstats.kurtosis(dial)),
                 float(np.max(np.abs(dial)) / (np.sqrt(np.mean(dial ** 2)) + 1e-9))])

    return (np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(np.array(scal, dtype=np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0))


# ============================================================ model architecture
def _build_keras_model(seq_len, n_seq, n_hgb=N_CLASSES):
    """Reconstructs the exact fused architecture. Weights-only files need the
    architecture rebuilt before loading, same pattern used throughout this
    project so no extra model-serialisation format is required."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    C1, C2, HEADS, KD, LSTM, DENSE, FUSE = 8, 24, 2, 8, 16, 32, 32
    DROP_S, DROP_D = 0.20, 0.50

    seq_in = layers.Input(shape=(seq_len, n_seq), name="sequence_input")
    hgb_in = layers.Input(shape=(n_hgb,), name="hgb_input")

    br = [layers.Conv1D(C1, k, padding="same", activation="swish")(seq_in)
          for k in (3, 9, 17)]
    x = layers.Concatenate()(br)
    x = layers.BatchNormalization()(x)
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling1D()(x)
    s = layers.Dense(max(4, ch // 4), activation="swish")(s)
    s = layers.Dense(ch, activation="sigmoid")(s)
    x = layers.Multiply()([x, layers.Reshape((1, ch))(s)])
    x = layers.Conv1D(C2, 5, padding="same", activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.SpatialDropout1D(DROP_S)(x)

    x = layers.Bidirectional(layers.LSTM(LSTM, return_sequences=True,
                                         dropout=DROP_S))(x)
    attn = layers.MultiHeadAttention(num_heads=HEADS, key_dim=KD,
                                     name="mhsa")(x, x)
    x = layers.LayerNormalization()(layers.Add()([x, attn]))
    score = layers.Softmax(axis=1, name="attn_weights")(layers.Dense(1)(x))
    deep = layers.Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1),
                         output_shape=lambda sh: (sh[0][0], sh[0][2]),
                         name="deep_feature")([x, score])
    deep = layers.Dense(DENSE, activation="swish", name="deep_vector")(deep)

    h = layers.Dense(16, activation="swish", name="hgb_embed")(hgb_in)
    fused = layers.Concatenate(name="fusion_concat")([deep, h])
    fused = layers.Dense(FUSE, activation="swish", name="feature_fusion")(fused)
    fused = layers.Dropout(DROP_D)(fused)
    out = layers.Dense(N_CLASSES, activation="softmax", name="fault_type")(fused)
    return keras.Model([seq_in, hgb_in], out, name="cnn_bilstm_attn_hgb")


# ============================================================ the model
class FinalFusionModel:
    """Load once, predict many. Thread-safe for read-only use."""

    def __init__(self, weights_path=WEIGHTS_PATH, bundle_path=BUNDLE_PATH):
        import joblib
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(
                f"model bundle not found: {bundle_path}\n"
                f"It must sit in the same directory as this script.")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"model weights not found: {weights_path}\n"
                f"It must sit in the same directory as this script.")
        b = joblib.load(bundle_path)
        self.hgb = b["hgb_model"]
        self.seq_scaler = b["seq_scaler"]
        self.scal_scaler = b["scal_scaler"]
        self.classes = np.array(b["classes"])
        self.meta = {k: b[k] for k in
                     ("trained_on_rows", "trained_on_simulations",
                      "trained_on_networks", "lono_reference_accuracy",
                      "lono_reference_std", "internal_val_accuracy") if k in b}

        self.model = _build_keras_model(WIN_LEN, 39)
        # a subclassed/functional model needs one real call before
        # load_weights() will accept a checkpoint on some Keras versions
        _ = self.model([np.zeros((1, WIN_LEN, 39), np.float32),
                        np.zeros((1, N_CLASSES), np.float32)])
        self.model.load_weights(weights_path)

    def predict_from_window(self, win50, ref50):
        seq, scal = build_sequence_and_scalar_features(win50, ref50)
        Xs = self.seq_scaler.transform(seq)[None, ...].astype(np.float32)
        Xk = self.scal_scaler.transform(scal[None, :]).astype(np.float32)
        h = np.log(np.clip(self.hgb.predict_proba(Xk), 1e-7, 1.0)).astype(np.float32)
        p = self.model.predict([Xs, h], verbose=0)[0]
        k = int(np.argmax(p))
        lbl = str(self.classes[k])
        return {"fault_type": lbl,
                "fault_type_merged": MERGE_FOR_REPORT.get(lbl, lbl),
                "confidence": float(p[k]),
                "probabilities": {str(c): float(v)
                                  for c, v in zip(self.classes, p)}}

    def predict_from_raw(self, raw, fs):
        win, ref, info = harmonise_raw(raw, fs)
        w50, r50 = centre_crop(win, ref)
        out = self.predict_from_window(w50, r50)
        out["harmonisation"] = info
        return out

    def predict_from_file(self, path, fs=None):
        raw, fs_file = load_raw_record(path)
        return self.predict_from_raw(raw, fs if fs is not None else fs_file)


def load_raw_record(path):
    """Read a .parquet or MATLAB .mat fault record -> (N,6) array plus rate."""
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head == b"PAR1":
        df = pd.read_parquet(path)
        cols = set(df.columns)
        if set(SEND_COLS) <= cols:
            raw = df[SEND_COLS].to_numpy(np.float64)
        elif set(PU_COLS) <= cols:
            raw = df[PU_COLS].to_numpy(np.float64)
        else:
            num = df.select_dtypes(include=[np.number]).drop(
                columns=[c for c in df.columns
                         if str(c).lower().startswith(("time", "target", "fs"))],
                errors="ignore")
            if num.shape[1] < 6:
                raise ValueError(f"{path}: fewer than six numeric channels")
            raw = num.to_numpy(np.float64)[:, :6]
        fs = float(df["Fs_Hz"].iloc[0]) if "Fs_Hz" in cols else None
        if fs is None:
            raise ValueError(f"{path}: no Fs_Hz column, pass fs= explicitly")
        return raw, fs
    import scipy.io as sio
    d = sio.loadmat(path)
    if "faultData" in d:
        return np.asarray(d["faultData"], np.float64)[:, :6], None
    for k in d:
        if k.startswith("__"):
            continue
        v = np.asarray(d[k])
        if v.ndim == 2 and v.dtype.kind == "f" and v.shape[0] > 50:
            return v.astype(np.float64)[:, :6], None
    raise ValueError(f"{path}: no usable 2-D array inside")


# ============================================================ demo
if __name__ == "__main__":
    print("=" * 72)
    print("FINAL MODEL  --  CNN-BiLSTM-Attention + HistGradientBoosting fusion")
    print("=" * 72)

    clf = FinalFusionModel()
    print(f"loaded: {clf.meta}")
    print(f"classes: {list(clf.classes)}")
    print(f"expects: {TARGET_SPC} samples/cycle ({TARGET_SPC*F0_HZ:.0f} Hz), "
          f"{WIN_LEN}-sample window, 6 channels [Va Vb Vc Ia Ib Ic]\n")

    if len(sys.argv) > 1:
        path = sys.argv[1]
        fs = float(sys.argv[2]) if len(sys.argv) > 2 else None
        print(f"--- raw file: {os.path.basename(path)} ---")
        raw, fs_file = load_raw_record(path)
        fs = fs or fs_file
        if fs is None:
            print("  no rate in the file; pass it as the 2nd argument, e.g.")
            print("  py week13_final_model_inference.py <file.mat> 6000")
            sys.exit(1)
        print(f"  raw shape {raw.shape}, fs {fs:.0f} Hz")
        r = clf.predict_from_raw(raw, fs)
        print(f"  -> {r['fault_type']}  (merged {r['fault_type_merged']}), "
              f"confidence {r['confidence']:.3f}")
        print(f"     harmonisation: {r['harmonisation']}")
    else:
        print("Usage:")
        print("  py week13_final_model_inference.py <record.mat|.parquet> [rate_hz]")
        print("\nExpects, in this same folder:")
        print(f"  {os.path.basename(WEIGHTS_PATH)}")
        print(f"  {os.path.basename(BUNDLE_PATH)}")
