"""Generate docs/signal_validation.png from synthetic and Q.Lin windows."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mr_tae_fusion.data.qlin_loader import load_qlin_dataset
from mr_tae_fusion.data.pulse_generators import generate_pd_signal
from mr_tae_fusion.data.noise_generators import add_noise_at_snr, generate_composite_noise
from mr_tae_fusion.config import SignalConfig, NoiseConfig


def _psd_peak_hz(x: np.ndarray, fs: float = 1e6) -> float:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, d=1.0 / fs)
    p = np.abs(X) ** 2
    return float(f[np.argmax(p[1:]) + 1])


def main() -> None:
    out = PROJECT_ROOT / "docs" / "signal_validation.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    sig_cfg = SignalConfig(sample_rate=1e6, signal_length=2048)
    noise_cfg = NoiseConfig()
    t = np.arange(2048) / 1e6
    snrs = [-15, -5, 5]
    syn = []
    for snr in snrs:
        _, c, _ = generate_pd_signal("C", sig_cfg)
        n = generate_composite_noise(t, noise_cfg)
        y, _ = add_noise_at_snr(c, n, snr)
        syn.append(y.astype(np.float32))

    qlin = load_qlin_dataset(
        data_dir=str(PROJECT_ROOT / "data" / "raw" / "qlin"),
        window_size=2048,
        overlap=0.5,
        normalize=True,
        split=0.9,
    )
    Xr, yr = qlin["test"]
    # pick one from each first three classes if available
    real = []
    for cls in [0, 1, 2]:
        idx = np.where(yr == cls)[0]
        if idx.size:
            real.append(Xr[idx[0]])
        else:
            real.append(Xr[min(cls, len(Xr) - 1)])

    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    tt = np.arange(2048) / 1e6 * 1e6
    for i in range(3):
        ps = _psd_peak_hz(syn[i])
        axes[0, i].plot(tt, syn[i], lw=0.8)
        axes[0, i].set_title(f"Synthetic SNR {snrs[i]} dB | PSD {ps/1e3:.1f} kHz")
        axes[0, i].set_ylabel("amp")

        pr = _psd_peak_hz(real[i])
        axes[1, i].plot(tt, real[i], lw=0.8, color="tab:orange")
        axes[1, i].set_title(f"Q.Lin class {i} | PSD {pr/1e3:.1f} kHz")
        axes[1, i].set_ylabel("amp")
        axes[1, i].set_xlabel("time (us)")

    fig.suptitle("Signal morphology validation: synthetic vs real Q.Lin windows")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()

