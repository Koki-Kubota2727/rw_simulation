import os
import numpy as np
import matplotlib.pyplot as plt

# --- Step 5: G(ω) の同定（逆問題） ---
# 各高調波周波数 ω_n = n·Ω において:
#   U_mat(ω_n) = [U_1(ω_n) | U_2(ω_n) | ... | U_6(ω_n)]  (6×6)
#   Y_mat(ω_n) = [Y_1(ω_n) | Y_2(ω_n) | ... | Y_6(ω_n)]  (6×6)
#   G_hat(ω_n) = Y_mat @ inv(U_mat)

INPUT_LABELS  = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]   # 列: 入力（力・モーメント）
OUTPUT_LABELS = ["x",  "y",  "z",  "θx", "θy", "θz"]   # 行: 出力（変位・角変位）


def identify_G(U_freq_list, Y_freq_list, freq, Omega, n_harmonics,
               n_avg_bins=5, bw_ratio=None):
    """
    高調波周波数における伝達行列 G_hat(ω_n) を同定する。

    Parameters
    ----------
    U_freq_list : list of (N_freq, 6) ndarray  各配置の実効入力スペクトル
    Y_freq_list : list of (N_freq, 6) ndarray  各配置の出力スペクトル
    freq        : ndarray, shape (N_freq,)      周波数軸 [Hz]
    Omega       : float                          RW 基本角速度 [rad/s]
    n_harmonics : int                            高調波次数の最大値
    n_avg_bins  : int, optional                  ビン平均するビン数（デフォルト=1、平均なし）
                                                 bw_ratio 未指定時に使用。
    bw_ratio    : float or None, optional        帯域幅を周波数の比率で指定（帯域 = fh × bw_ratio）。
                                                 設定した場合は n_avg_bins より優先。

    Returns
    -------
    G_hat   : ndarray, shape (n_harmonics, 6, 6)  同定結果
    cond_U  : ndarray, shape (n_harmonics,)        U_mat の条件数
    harm_freqs : ndarray, shape (n_harmonics,)     高調波周波数 [Hz]
    harm_idx   : ndarray, shape (n_harmonics,)     対応する周波数ビン番号
    """
    f0 = Omega / (2 * np.pi)
    df = freq[1] - freq[0]
    harm_freqs = np.array([n * f0 for n in range(1, n_harmonics + 1)])
    harm_idx = np.array([np.argmin(np.abs(freq - fh)) for fh in harm_freqs])

    G_hat = np.zeros((n_harmonics, 6, 6), dtype=complex)
    cond_U = np.zeros(n_harmonics)

    for ni, (fh, ki) in enumerate(zip(harm_freqs, harm_idx)):
        if bw_ratio is not None:
            n_bins = max(1, int(np.round(fh * bw_ratio / df)))
        else:
            n_bins = n_avg_bins

        half = n_bins // 2
        k_lo = max(0, ki - half)
        k_hi = min(len(freq), ki + half + 1)

        # 各ビンの 6×6 行列を横方向にスタック → (6, n_bins*6) の過決定系
        U_stacked = np.hstack([
            np.column_stack([U[k, :] for U in U_freq_list]) for k in range(k_lo, k_hi)
        ])
        Y_stacked = np.hstack([
            np.column_stack([Y[k, :] for Y in Y_freq_list]) for k in range(k_lo, k_hi)
        ])

        cond_U[ni] = np.linalg.cond(U_stacked)

        # G @ U_stacked = Y_stacked  ⟹  U_stacked.T @ G.T = Y_stacked.T
        G_T, _, _, _ = np.linalg.lstsq(U_stacked.T, Y_stacked.T, rcond=None)
        G_hat[ni] = G_T.T

    return G_hat, cond_U, harm_freqs, harm_idx


def compare_G(G_hat, G_true_vals, harm_freqs, cond_U,
              save_path="png/identification_result.png"):
    """
    G_hat と G_true を全 36 成分で比較する（振幅・位相・相対誤差）。

    Parameters
    ----------
    G_hat       : (n_harmonics, 6, 6) complex
    G_true_vals : (n_harmonics, 6, 6) complex  高調波周波数での G_true の値
    harm_freqs  : (n_harmonics,) [Hz]
    cond_U      : (n_harmonics,)  条件数
    """
    n_h = len(harm_freqs)
    from transfer_matrix import _CROSSTALK_RESONANCES
    ct_set = {(r, c) for r, c, *_ in _CROSSTALK_RESONANCES}

    # --- 36 成分の振幅比較（6×6 グリッド） ---
    # 左マージン（行ラベル用）・上マージン（列ラベル用）を確保
    fig, axes = plt.subplots(6, 6, figsize=(20, 18),
                             gridspec_kw={"hspace": 0.15, "wspace": 0.35})

    for row in range(6):
        for col in range(6):
            ax = axes[row, col]
            amp_true = np.abs(G_true_vals[:, row, col])
            amp_hat  = np.abs(G_hat[:, row, col])

            ax.semilogy(harm_freqs, amp_true, "o-", color="steelblue",
                        markersize=4, linewidth=1.2, label="G_true")
            ax.semilogy(harm_freqs, amp_hat,  "x--", color="tomato",
                        markersize=5, linewidth=1.2, label="G_hat")
            ax.tick_params(labelsize=7)
            ax.grid(True, which="both", alpha=0.3)

            # x 軸: 最下行のみ表示
            if row < 5:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Freq [Hz]", fontsize=7)

            # y 軸: 最左列のみ表示
            if col > 0:
                ax.set_yticklabels([])

            # 縦軸スケール
            if row == col or (row, col) in ct_set:
                vmin = min(np.min(amp_true[amp_true > 0] if amp_true.any() else [1e-10]),
                           np.min(amp_hat[amp_hat > 0]   if amp_hat.any()  else [1e-10]))
                vmax = max(np.max(amp_true), np.max(amp_hat))
                if vmax > 0:
                    ax.set_ylim([vmin * 0.3, vmax * 3.0])
            else:
                med = np.median(amp_true[amp_true > 0]) if amp_true.any() else 1e-7
                ax.set_ylim([med / 3, med * 3])

    # --- 列ヘッダー（入力 DOF）を最上行の上に配置 ---
    for col in range(6):
        axes[0, col].set_title(f"Input: {INPUT_LABELS[col]}", fontsize=9, pad=4)

    # --- 行ラベル（出力 DOF）を最左列の左に配置 ---
    for row in range(6):
        axes[row, 0].set_ylabel(f"Output: {OUTPUT_LABELS[row]}", fontsize=9, labelpad=4)

    # 凡例は左上セルのみ
    axes[0, 0].legend(fontsize=8, loc="lower left")

    fig.text(0.5, 0.005, "Frequency [Hz]", ha="center", fontsize=11)
    fig.text(0.005, 0.5, "|G|", va="center", rotation="vertical", fontsize=11)
    plt.suptitle("Identification result: |G_true| vs |G_hat|  —  rows: output DOF,  cols: input DOF",
                 fontsize=12, y=0.995)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"saved: {save_path}")
    plt.show()

    # --- フロベニウスノルム相対誤差 ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))

    # 各高調波の相対誤差
    rel_err = (np.linalg.norm(G_hat - G_true_vals, axis=(1, 2)) /
               np.linalg.norm(G_true_vals, axis=(1, 2)))
    axes2[0].semilogy(harm_freqs, rel_err, "o-", color="steelblue", markersize=5)
    axes2[0].set_xlabel("Frequency [Hz]")
    axes2[0].set_ylabel("Relative error  ||G_hat - G_true||_F / ||G_true||_F")
    axes2[0].set_title("Identification error per harmonic")
    axes2[0].grid(True, which="both", alpha=0.3)

    # 条件数
    axes2[1].semilogy(harm_freqs, cond_U, "s-", color="darkorange", markersize=5)
    axes2[1].set_xlabel("Frequency [Hz]")
    axes2[1].set_ylabel("cond(U_mat)")
    axes2[1].set_title("Condition number of U_mat at each harmonic")
    axes2[1].grid(True, which="both", alpha=0.3)

    plt.suptitle("Identification quality metrics", fontsize=12)
    plt.tight_layout()
    err_path = save_path.replace(".png", "_error.png")
    os.makedirs(os.path.dirname(err_path), exist_ok=True)
    plt.savefig(err_path, dpi=120)
    print(f"saved: {err_path}")
    plt.show()

    # --- 数値サマリ ---
    print("\n=== 同定精度サマリ ===")
    print(f"{'freq [Hz]':>10}  {'rel_err':>10}  {'cond(U)':>12}")
    for fh, err, cn in zip(harm_freqs, rel_err, cond_U):
        print(f"{fh:10.1f}  {err:10.4e}  {cn:12.2f}")
    print(f"\n全周波数平均相対誤差: {rel_err.mean():.4e}")


if __name__ == "__main__":
    from main import t, fs, freq, omega, Omega, n_harmonics
    from transfer_matrix import G_true
    from position_transform import T_LIST, CONFIG_LABELS
    from signal_gen import generate_u_rw, fft_signal
    from forward import compute_outputs

    # Step 3: 入力信号
    u_rw, _, _ = generate_u_rw(t, Omega, n_harmonics)
    _, U_rw_freq = fft_signal(u_rw, fs)

    # Step 1: G_true
    G = G_true(omega)

    # Step 4: 順問題
    U_freq_list, Y_freq_list, _ = compute_outputs(
        U_rw_freq, T_LIST, G, fs,
        add_noise=True,
        noise_snr_db=20.0,
    )

    # Step 5: 同定
    G_hat, cond_U, harm_freqs, harm_idx = identify_G(
        U_freq_list, Y_freq_list, freq, Omega, n_harmonics
    )

    # G_true の値を高調波周波数で評価
    G_true_vals = G[harm_idx]   # (n_harmonics, 6, 6)

    # Step 6 の準備: 比較・可視化
    compare_G(G_hat, G_true_vals, harm_freqs, cond_U)
