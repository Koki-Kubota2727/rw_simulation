import numpy as np

# --- Step 1: 真の伝達行列 G_true(ω) の定義 ---
# 各自由度: 1自由度ばね-マス-ダンパー系
# H_k(ω) = 1 / (m_k*(jω)^2 + c_k*(jω) + k_k)
# ω_n = sqrt(k/m), ζ = c / (2*sqrt(k*m))

def _smd_params(m, zeta, f_n):
    """質量・減衰比・固有振動数からパラメータを計算"""
    omega_n = 2 * np.pi * f_n
    k = m * omega_n**2
    c = 2 * zeta * np.sqrt(k * m)
    return m, c, k

# 6自由度のパラメータ設定
# [質量 m (kg), 減衰比 ζ, 固有振動数 f_n (Hz)]
# 固有振動数を RW 高調波（20,40,...,120 Hz）の間に分散させる
_DOF_PARAMS = [
    _smd_params(m=2.0, zeta=0.02, f_n=15.0),   # DOF 0 (x1):  15 Hz
    _smd_params(m=2.0, zeta=0.02, f_n=30.0),   # DOF 1 (x2):  30 Hz
    _smd_params(m=1.5, zeta=0.03, f_n=50.0),   # DOF 2 (x3):  50 Hz
    _smd_params(m=1.5, zeta=0.03, f_n=70.0),   # DOF 3 (θx):  70 Hz
    _smd_params(m=1.0, zeta=0.02, f_n=95.0),   # DOF 4 (θy):  95 Hz
    _smd_params(m=1.0, zeta=0.02, f_n=115.0),  # DOF 5 (θz): 115 Hz
]

# フラットな非対角結合の強さ（対角成分 RMS に対する比率）
_EPSILON = 0.03

# 固定乱数で非対角フラット成分を生成（再現性確保）
_rng = np.random.default_rng(seed=42)
_G_off_template = (
    _rng.standard_normal((6, 6)) + 1j * _rng.standard_normal((6, 6))
)
np.fill_diagonal(_G_off_template, 0.0)

# 共振型クロストーク（一部の非対角成分に固有の共振ピークを追加）
# (row, col, zeta, f_n [Hz], amplitude_fraction)
#   amplitude_fraction: 参照する対角成分 G[row,row] の共振ピーク振幅に対する比率
_CROSSTALK_RESONANCES = [
    (0, 3, 0.03,  28.0, 0.20),   # x1 → θx:  28 Hz で共振
    (1, 4, 0.02,  45.0, 0.15),   # x2 → θy:  45 Hz で共振
    (2, 0, 0.03,  62.0, 0.18),   # x3 → x1:  62 Hz で共振
    (3, 5, 0.02,  85.0, 0.12),   # θx → θz:  85 Hz で共振
    (4, 1, 0.03, 105.0, 0.15),   # θy → x2: 105 Hz で共振
    (5, 2, 0.02,  35.0, 0.20),   # θz → x3:  35 Hz で共振
]


def _diagonal_peak(row):
    """G[row, row] の共振ピーク振幅の近似値 ≈ 1 / (2ζk)"""
    m, c, stiff = _DOF_PARAMS[row]
    omega_n = np.sqrt(stiff / m)
    return 1.0 / (c * omega_n)


def G_true(omega):
    """
    真の伝達行列 G(ω) を返す。

    Parameters
    ----------
    omega : array_like, shape (N_freq,)
        角周波数 [rad/s]

    Returns
    -------
    G : ndarray, shape (N_freq, 6, 6), dtype complex
        各周波数点での 6×6 伝達行列
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    N_freq = len(omega)
    G = np.zeros((N_freq, 6, 6), dtype=complex)

    jw = 1j * omega

    # 対角成分: 各自由度のばね-マス-ダンパー伝達関数
    for k, (m, c, stiff) in enumerate(_DOF_PARAMS):
        G[:, k, k] = 1.0 / (m * jw**2 + c * jw + stiff)

    # フラットな非対角結合
    diag_rms = np.sqrt(np.mean(np.abs(G[:, range(6), range(6)])**2))
    G += _EPSILON * diag_rms * _G_off_template[np.newaxis, :, :]

    # 共振型クロストーク
    for row, col, zeta, f_n, amp_frac in _CROSSTALK_RESONANCES:
        _, c_ct, k_ct = _smd_params(m=1.0, zeta=zeta, f_n=f_n)
        omega_n_ct = 2 * np.pi * f_n
        peak_ct = 1.0 / (c_ct * omega_n_ct)           # クロストーク項のピーク振幅
        scale = amp_frac * _diagonal_peak(row) / peak_ct
        G[:, row, col] += scale / (1.0 * jw**2 + c_ct * jw + k_ct)

    return G


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from main import omega, freq

    G = G_true(omega)

    labels = ["x1", "x2", "x3", "thx", "thy", "thz"]

    # クロストーク共振の (row, col) → f_n の対応を作っておく
    ct_map = {(r, c): f for r, c, _, f, _ in _CROSSTALK_RESONANCES}

    fig, axes = plt.subplots(6, 6, figsize=(20, 18))
    for row in range(6):
        for col in range(6):
            ax = axes[row, col]
            ax.semilogy(freq, np.abs(G[:, row, col]),
                        color="steelblue" if row == col else "darkorange",
                        linewidth=0.8)
            if row == col:
                m, c, stiff = _DOF_PARAMS[row]
                f_n = np.sqrt(stiff / m) / (2 * np.pi)
                ax.axvline(f_n, color="red", linestyle="--", linewidth=0.7)
            elif (row, col) in ct_map:
                ax.axvline(ct_map[(row, col)], color="green", linestyle="--", linewidth=0.7)
            ax.set_title(f"G[{labels[row]},{labels[col]}]", fontsize=7)
            ax.set_xlim([0, 150])
            ax.tick_params(labelsize=6)
            ax.grid(True, which="both", alpha=0.3)
            if row < 5:
                ax.set_xticklabels([])

            # 縦軸スケール
            amp = np.abs(G[:, row, col])
            if row == col or (row, col) in ct_map:
                # 対角・クロストーク共振: 全周波数のデータが見えるよう上下に余白
                vmin, vmax = np.min(amp), np.max(amp)
                if vmax > 0:
                    ax.set_ylim([vmin * 0.5, vmax * 2.0])
            else:
                # フラット非対角: メジアン振幅を中心に 1 decade 幅
                med = np.median(amp)
                if med > 0:
                    ax.set_ylim([med / np.sqrt(10), med * np.sqrt(10)])

    fig.text(0.5, 0.01, "Frequency [Hz]", ha="center", fontsize=11)
    fig.text(0.01, 0.5, "|G|", va="center", rotation="vertical", fontsize=11)
    plt.suptitle("G_true all components (amplitude)  [blue=diag, orange=off-diag, red=diag f_n, green=crosstalk f_n]",
                 fontsize=11)
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.97])
    plt.savefig("G_true_all.png", dpi=120)
    print("saved: G_true_all.png")
    plt.show()
