"""
Per-channel checks and the overview figure, computed in one streaming pass over each channel.

- flat: no change at all, or (except stimulation and device heart-rate channels, which are
  constant between pulses or beats) at least FLAT_FRACTION of the 1 s windows without any change;
- saturated: at least SAT_FRACTION of the samples sit at the channel's minimum or maximum in runs
  of 3 or more (a clipped amplifier or transducer); not for stimulation channels;
- noisy: for slow physiological waveforms (ECG, BP, respiration, PPG, temperature, pressures)
  a lag-1 autocorrelation below NOISY_AUTOCORR: neighbouring samples are less alike than a
  waveform sampled at this rate should be, i.e. the channel is mostly noise or mains hum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FLAT_FRACTION = 0.5
SAT_FRACTION = 0.01
NOISY_AUTOCORR = 0.5
N_BINS = 1500
ZOOM_S = 10.0
NO_FLAT_WINDOWS = {"stim", "hr"}
NO_SATURATION = {"stim"}
NOISY_TYPES = {"ecg", "bp", "resp", "ppg", "temp", "pressure"}


@dataclass
class ChannelStats:
    """Accumulates over chunks of stored values; physical value = stored * conversion + offset."""
    rate: float
    span: float                       # recording span (s) for the envelope bins
    zoom: tuple[float, float]
    n: int = 0
    n_nan: int = 0
    s1: float = 0.0
    s2: float = 0.0
    sxy: float = 0.0                  # sum x_i x_{i+1} within chunks
    sx_a: float = 0.0                 # sum x_i of those pairs
    sx_b: float = 0.0                 # sum x_{i+1}
    sxx_a: float = 0.0
    sxx_b: float = 0.0
    n_pairs: int = 0
    vmin: float = math.inf
    vmax: float = -math.inf
    n_at_min: int = 0
    n_at_max: int = 0
    win_total: int = 0
    win_flat: int = 0
    quant: float | None = None        # candidate step of float values (None: not quantised)
    quant_ok: bool = True
    env_min: np.ndarray = field(default=None)
    env_max: np.ndarray = field(default=None)
    zoom_t: list = field(default_factory=list)
    zoom_v: list = field(default_factory=list)

    def __post_init__(self):
        self.env_min = np.full(N_BINS, np.nan)
        self.env_max = np.full(N_BINS, np.nan)

    def add(self, x: np.ndarray, t0: float, times: np.ndarray | None, float_source: bool) -> None:
        """x: physical values (float64) of consecutive samples starting at time t0 (or at `times`)."""
        n = x.size
        if n == 0:
            return
        fin = np.isfinite(x)
        self.n += n
        self.n_nan += int(n - fin.sum())
        xf = x[fin]
        if xf.size:
            self.s1 += float(xf.sum())
            self.s2 += float(np.dot(xf, xf))
            lo, hi = float(xf.min()), float(xf.max())
            run = _runs3(x)
            if lo < self.vmin:
                self.vmin, self.n_at_min = lo, 0
            if hi > self.vmax:
                self.vmax, self.n_at_max = hi, 0
            self.n_at_min += int(np.count_nonzero(run & (x == self.vmin)))
            self.n_at_max += int(np.count_nonzero(run & (x == self.vmax)))
        if n > 1:
            a, b = x[:-1], x[1:]
            m = np.isfinite(a) & np.isfinite(b)
            a, b = a[m], b[m]
            self.sxy += float(np.dot(a, b))
            self.sx_a += float(a.sum())
            self.sx_b += float(b.sum())
            self.sxx_a += float(np.dot(a, a))
            self.sxx_b += float(np.dot(b, b))
            self.n_pairs += int(a.size)
        w = max(1, int(round(self.rate)))
        k = n // w
        if k:
            blk = x[: k * w].reshape(k, w)
            with np.errstate(invalid="ignore"):
                ok = np.isfinite(blk).all(axis=1)
                ptp = np.nanmax(blk, axis=1) - np.nanmin(blk, axis=1) if ok.any() else np.array([])
            if ok.any():
                self.win_total += int(ok.sum())
                self.win_flat += int(np.count_nonzero(ptp[ok] == 0))
        # envelope
        t = times if times is not None else t0 + np.arange(n) / self.rate
        if self.span > 0:
            bi = np.clip((t / self.span * N_BINS).astype(np.int64), 0, N_BINS - 1)
            starts = np.flatnonzero(np.r_[True, bi[1:] != bi[:-1]])
            with np.errstate(invalid="ignore"):
                mn = np.fmin.reduceat(x, starts)
                mx = np.fmax.reduceat(x, starts)
            ub = bi[starts]
            self.env_min[ub] = np.fmin(self.env_min[ub], mn)
            self.env_max[ub] = np.fmax(self.env_max[ub], mx)
        z0, z1 = self.zoom
        if t[0] <= z1 and t[-1] >= z0:
            sel = (t >= z0) & (t < z1)
            if sel.any():
                self.zoom_t.append(t[sel])
                self.zoom_v.append(x[sel])

    def check_quant(self, raw: np.ndarray) -> None:
        """Float values that are all integer multiples of one step (an ADC's resolution) can be stored as
        integers without loss; keeps the step while every chunk agrees."""
        if not self.quant_ok:
            return
        r = raw[np.isfinite(raw)] if raw.dtype.kind == "f" else raw
        if r.size != raw.size:
            self.quant_ok = False
            return
        if self.quant is None:
            # float32-stored sources read as float64 carry rounding noise below the ADC step: find the step on
            # values rounded to float32 precision
            u = np.unique(r[:200_000].astype(np.float32)).astype(np.float64)
            d = np.diff(u)
            d = d[d > 0]
            if d.size == 0:
                self.quant_ok = False
                return
            q = float(d.min())
            # refine by least squares on the integer multiples (the smallest difference carries rounding noise)
            for _ in range(2):
                k = np.round(u / q)
                if not np.any(k):
                    break
                q = float(np.dot(u, k) / np.dot(k, k))
            self.quant = q
        q = self.quant
        k = r / q
        # within 1/1000 of a step, plus 4 float32 ulps for values that went through float32
        if not np.all(np.abs(k - np.round(k)) < 1e-3 + 2.4e-7 * np.abs(k)) or np.max(np.abs(k), initial=0) > 2**31:
            self.quant_ok = False

    def int_dtype(self, raw_min: float, raw_max: float) -> np.dtype | None:
        if not (self.quant_ok and self.quant):
            return None
        lo, hi = raw_min / self.quant, raw_max / self.quant
        if -32768 <= lo and hi <= 32767:
            return np.dtype(np.int16)
        if -2**31 <= lo and hi <= 2**31 - 1:
            return np.dtype(np.int32)
        return None

    def autocorr(self) -> float:
        n = self.n_pairs
        if n < 3:
            return float("nan")
        cov = self.sxy / n - (self.sx_a / n) * (self.sx_b / n)
        va = self.sxx_a / n - (self.sx_a / n) ** 2
        vb = self.sxx_b / n - (self.sx_b / n) ** 2
        if va <= 0 or vb <= 0:
            return float("nan")
        return float(cov / math.sqrt(va * vb))

    def mean_sd(self) -> tuple[float, float]:
        m = self.n - self.n_nan
        if m == 0:
            return float("nan"), float("nan")
        mean = self.s1 / m
        return mean, math.sqrt(max(0.0, self.s2 / m - mean * mean))

    def status(self, ch_type: str) -> tuple[str, str, dict]:
        m = self.n - self.n_nan
        vals = {"n_samples": self.n, "nan_fraction": self.n_nan / self.n if self.n else float("nan"),
                "min": self.vmin if m else float("nan"), "max": self.vmax if m else float("nan")}
        mean, sd = self.mean_sd()
        vals["mean"], vals["sd"] = mean, sd
        flat_frac = self.win_flat / self.win_total if self.win_total else float("nan")
        sat = (self.n_at_min + self.n_at_max) / m if m else float("nan")
        ac = self.autocorr()
        vals.update(flat_fraction=flat_frac, saturated_fraction=sat, lag1_autocorrelation=ac)
        if m == 0:
            return "flat", "no samples (every value is missing)", vals
        if self.vmax == self.vmin:
            return "flat", f"constant at {self.vmin:g} over the whole recording", vals
        if ch_type not in NO_FLAT_WINDOWS and self.win_total and flat_frac >= FLAT_FRACTION:
            return "flat", f"{100 * flat_frac:.0f}% of 1 s windows without any change", vals
        if ch_type not in NO_SATURATION and sat >= SAT_FRACTION:
            return ("saturated", f"{100 * sat:.1f}% of samples stuck at the minimum ({self.vmin:g}) or maximum "
                    f"({self.vmax:g}) in runs of 3 or more: clipped amplifier or transducer?", vals)
        if ch_type in NOISY_TYPES and math.isfinite(ac) and ac < NOISY_AUTOCORR:
            return ("noisy", f"lag-1 autocorrelation {ac:.2f} (< {NOISY_AUTOCORR:g}): neighbouring samples barely "
                    "related, mostly noise or mains interference", vals)
        notes = []
        if vals["nan_fraction"] > 0:
            notes.append(f"{100 * vals['nan_fraction']:.1f}% of samples missing (gaps in the source)")
        return "good", "; ".join(notes), vals


def _runs3(x: np.ndarray) -> np.ndarray:
    """True where a sample equals both neighbours (inside a run of 3 or more equal values)."""
    r = np.zeros(x.size, dtype=bool)
    if x.size >= 3:
        eq = x[1:] == x[:-1]
        mid = eq[1:] & eq[:-1]
        r[1:-1] = mid
        r[:-2] |= mid
        r[2:] |= mid
    return r


def decimate_minmax(t: np.ndarray, v: np.ndarray, n_out: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    if v.size <= n_out:
        return t, v
    k = int(math.ceil(v.size / (n_out / 2)))
    m = v.size // k
    vv = v[: m * k].reshape(m, k)
    tt = t[: m * k].reshape(m, k)[:, 0]
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanmin(vv, axis=1), np.nanmax(vv, axis=1)
    return np.repeat(tt, 2), np.column_stack([lo, hi]).reshape(-1)


def overview_png(path, rid: str, rows: list[dict], span: float, zoom: tuple[float, float], segments_note: str) -> None:
    """One row per channel: the whole recording (min/max envelope) and a 10 s zoom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = rows[:16]
    h = max(2.5, 1.35 * len(shown) + 1.0)
    fig, axes = plt.subplots(len(shown), 2, figsize=(14, h), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 1.4]})
    bins_t = (np.arange(N_BINS) + 0.5) / N_BINS * span
    for i, r in enumerate(shown):
        st: ChannelStats = r["stats"]
        ax, az = axes[i]
        color = {"good": "#1f5fa8", "flat": "#b8860b", "saturated": "#c0392b", "noisy": "#8e44ad"}[r["status"]]
        ok = np.isfinite(st.env_min)
        if ok.any():
            ax.fill_between(bins_t, st.env_min, st.env_max, where=ok, color=color, linewidth=0.6, step="mid")
        ax.set_xlim(0, span if span > 0 else 1)
        lab = f"{r['channel']}\n{r['original'][:22]}\n[{r['unit']}]"
        ax.set_ylabel(lab, fontsize=7, rotation=0, ha="right", va="center")
        flag = "" if r["status"] == "good" else f"  {r['status'].upper()}: {r['status_description'][:90]}"
        ax.set_title(f"{r['channel']} ({r['type']}, typed by {r['typing']}){flag}", fontsize=8, loc="left",
                     color=("black" if r["status"] == "good" else color))
        ax.axvspan(zoom[0], zoom[1], color="#999999", alpha=0.15, linewidth=0)
        ax.tick_params(labelsize=7)
        if st.zoom_t:
            tz, vz = decimate_minmax(np.concatenate(st.zoom_t), np.concatenate(st.zoom_v))
            az.plot(tz, vz, color=color, linewidth=0.6)
        az.set_xlim(*zoom)
        az.tick_params(labelsize=7)
        if i < len(shown) - 1:
            ax.set_xticklabels([])
            az.set_xticklabels([])
    axes[-1][0].set_xlabel("time from recording start (s)", fontsize=8)
    axes[-1][1].set_xlabel(f"10 s zoom (s; shaded on the left)", fontsize=8)
    more = f" (first 16 of {len(rows)} channels)" if len(rows) > 16 else ""
    fig.suptitle(f"{rid}: channel overview{more}{segments_note}", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150)
    plt.close(fig)
