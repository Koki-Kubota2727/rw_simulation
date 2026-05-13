import numpy as np

# --- Step 3: 入力信号の生成 ---
# RW 擾乱モデル: 各高調波 n で 6成分それぞれに振幅・位相を持つ正弦波
#   u_rw(t) = Σ_n  A_n ⊙ sin(n·Ω·t + φ_n)   (⊙: 成分ごとの積)
# A_n ∈ R^6, φ_n ∈ R^6  は各成分・各次数で独立に設定できる。

# 振幅係数（固定・再現性あり）
# 実際の RW では次数が上がると振幅が小さくなる傾向があるため 1/n でスケール
_rng_amp = np.random.default_rng(seed=7)

def _default_amplitudes(n_harmonics, component_scales=None):
    """各高調波 n の 6成分振幅行列 (n_harmonics×6) を返す（再現性固定）

    component_scales : array-like, shape (6,), optional
        [Fx, Fy, Fz, Mx, My, Mz] それぞれの振幅スケール。
        None なら全成分 1.0（デフォルト動作）。
    """
    if component_scales is None:
        component_scales = np.ones(6)
    component_scales = np.asarray(component_scales, dtype=float)  # (6,)
    rng = np.random.default_rng(seed=7)
    base = rng.uniform(0.5, 1.5, size=(n_harmonics, 6))
    decay = 1.0 / np.arange(1, n_harmonics + 1).reshape(-1, 1)
    return base * decay * component_scales   # shape: (n_harmonics, 6)

def _default_phases(n_harmonics):
    """各高調波 n の 6成分位相行列 (n_harmonics×6) [rad]"""
    rng = np.random.default_rng(seed=13)
    return rng.uniform(0, 2 * np.pi, size=(n_harmonics, 6))


def generate_u_rw(t, Omega, n_harmonics, amplitudes=None, phases=None, Q=None):
    """
    RW 擾乱の時間信号 u_rw(t) を生成する。

    Parameters
    ----------
    t : array, shape (N,)
        時間軸 [s]
    Omega : float
        RW 基本角速度 [rad/s]
    n_harmonics : int
        使用する高調波の最大次数
    amplitudes : array, shape (n_harmonics, 6), optional
        各高調波・各成分の振幅。None なら自動生成。
    phases : array, shape (n_harmonics, 6), optional
        各高調波・各成分の初期位相 [rad]。None なら自動生成。
    Q : float or None
        各高調波の品質係数。大きいほどピークが鋭い。
        None または inf で純正弦波（デフォルト）。
        半値幅 = f_n / Q  例: Q=50 → 20Hz成分の幅 0.4Hz

    Returns
    -------
    u_rw : ndarray, shape (N, 6)
        時系列入力信号 [N, 6]
    amplitudes : ndarray, shape (n_harmonics, 6)
    phases : ndarray, shape (n_harmonics, 6)
    """
    if amplitudes is None:
        amplitudes = _default_amplitudes(n_harmonics)
    if phases is None:
        phases = _default_phases(n_harmonics)

    N = len(t)
    dt = t[1] - t[0]
    u_rw = np.zeros((N, 6))
    use_narrowband = (Q is not None) and np.isfinite(Q) and (Q > 0)
    rng_q = np.random.default_rng(seed=99)  # 再現性のためシード固定

    for n in range(1, n_harmonics + 1):
        f_n = n * Omega / (2 * np.pi)   # 高調波周波数 [Hz]

        if not use_narrowband:
            # 純正弦波（Q=None のデフォルト）
            angle = n * Omega * t[:, None] + phases[n - 1]   # (N, 6)
            u_rw += amplitudes[n - 1] * np.sin(angle)
        else:
            # 複素エンベロープ法: 半値幅 = f_n/Q の狭帯域過程を IIR で生成
            # 相関時間 τ_c = Q / (π f_n)  →  IIR係数 α = exp(-dt/τ_c)
            tau_c = Q / (np.pi * f_n)
            alpha = np.exp(-dt / tau_c)
            sigma = np.sqrt((1 - alpha ** 2) / 2)   # 分散 1 に正規化

            # 複素ノイズを IIR フィルタに通して複素エンベロープ a(t) を得る
            cn = (rng_q.standard_normal((N, 6)) +
                  1j * rng_q.standard_normal((N, 6))) * sigma
            a = np.zeros((N, 6), dtype=complex)
            for i in range(1, N):
                a[i] = alpha * a[i - 1] + cn[i]

            # RMS を 1 に正規化してから amplitude を乗じる
            rms = np.sqrt(np.mean(np.abs(a) ** 2, axis=0))
            a /= np.where(rms > 0, rms, 1.0)

            # 搬送波を乗じて実部を取る: Re[a(t)·exp(j(n·Ω·t + φ_n))]
            carrier = np.exp(1j * (2 * np.pi * f_n * t[:, None] + phases[n - 1]))
            u_rw += amplitudes[n - 1] * np.real(a * carrier)

    return u_rw, amplitudes, phases


def fft_signal(u_rw, fs):
    """
    時間信号を FFT して周波数スペクトルを返す。

    Returns
    -------
    freq : ndarray, shape (N_freq,)  [Hz]
    U_rw : ndarray, shape (N_freq, 6)  複素スペクトル（rfft）
    """
    N = u_rw.shape[0]
    freq = np.fft.rfftfreq(N, d=1.0 / fs)
    U_rw = np.fft.rfft(u_rw, axis=0) / N   # 両側スペクトルを片側に正規化
    return freq, U_rw


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from main import t, fs, Omega, n_harmonics

    # ---- 成分ごとの振幅スケール [Fx, Fy, Fz, Mx, My, Mz] ----
    component_scales = np.array([1.0, 1.0, 0.1, 0.015, 0.015, 0.0015])

    # ---- Q値（ピークの鋭さ）: 大きいほど鋭い / None で純正弦波 ----
    # 半値幅 = f_n / Q  例: Q=30, f0=20Hz → 1次ピーク幅 ≈ 0.67 Hz
    Q = 60

    custom_amps = _default_amplitudes(n_harmonics, component_scales)
    u_rw, amps, phases = generate_u_rw(t, Omega, n_harmonics, amplitudes=custom_amps, Q=Q)
    freq, U_rw = fft_signal(u_rw, fs)

    dof_labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    units      = ["N",  "N",  "N",  "Nm", "Nm", "Nm"]
    f0 = Omega / (2 * np.pi)
    rpm = Omega / (2 * np.pi) * 60

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for k, ax in enumerate(axes.flat):
        ax.plot(freq, np.abs(U_rw[:, k]), color="steelblue", linewidth=0.8)
        for n in range(1, n_harmonics + 1):
            ax.axvline(n * f0, color="red", linestyle="--", linewidth=0.7,
                       label=f"n={n}" if k == 0 else None)
        ax.set_title(f"{dof_labels[k]}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(f"{dof_labels[k]} [{units[k]}]")
        ax.set_xlim([0, (n_harmonics + 1) * f0])
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=7)
    plt.suptitle(f"RW Disturbance Spectrum  (RW Speed = {rpm:.0f} rpm)", fontsize=12)
    plt.tight_layout()
    plt.savefig("u_rw_spectrum.png", dpi=120)
    print("saved: u_rw_spectrum.png")
    plt.show()
