import numpy as np
import matplotlib.pyplot as plt

# --- Step 5: G(ω) の同定（逆問題） ---
# 各高調波周波数 ω_n = n·Ω において:
#   U_mat(ω_n) = [U_1(ω_n) | U_2(ω_n) | ... | U_6(ω_n)]  (6×6)
#   Y_mat(ω_n) = [Y_1(ω_n) | Y_2(ω_n) | ... | Y_6(ω_n)]  (6×6)
#   G_hat(ω_n) = Y_mat @ inv(U_mat)

DOF_LABELS = ["x1", "x2", "x3", "thx", "thy", "thz"]


def identify_G(U_freq_list, Y_freq_list, freq, Omega, n_harmonics):
    """
    高調波周波数における伝達行列 G_hat(ω_n) を同定する。

    Parameters
    ----------
    U_freq_list : list of (N_freq, 6) ndarray  各配置の実効入力スペクトル
    Y_freq_list : list of (N_freq, 6) ndarray  各配置の出力スペクトル
    freq        : ndarray, shape (N_freq,)      周波数軸 [Hz]
    Omega       : float                          RW 基本角速度 [rad/s]
    n_harmonics : int                            高調波次数の最大値

    Returns
    -------
    G_hat   : ndarray, shape (n_harmonics, 6, 6)  同定結果
    cond_U  : ndarray, shape (n_harmonics,)        U_mat の条件数
    harm_freqs : ndarray, shape (n_harmonics,)     高調波周波数 [Hz]
    harm_idx   : ndarray, shape (n_harmonics,)     対応する周波数ビン番号
    """
    f0 = Omega / (2 * np.pi)
    harm_freqs = np.array([n * f0 for n in range(1, n_harmonics + 1)])
    harm_idx = np.array([np.argmin(np.abs(freq - fh)) for fh in harm_freqs])

    G_hat = np.zeros((n_harmonics, 6, 6), dtype=complex)
    cond_U = np.zeros(n_harmonics)

    for ni, (fh, ki) in enumerate(zip(harm_freqs, harm_idx)):
        # 6×6 行列をスタック（各列が 1 配置の入力/出力ベクトル）
        U_mat = np.column_stack([U[ki, :] for U in U_freq_list])   # (6, 6)
        Y_mat = np.column_stack([Y[ki, :] for Y in Y_freq_list])   # (6, 6)

        cond_U[ni] = np.linalg.cond(U_mat)

        # G_hat = Y_mat @ inv(U_mat)
        # G_hat[ni] = Y_mat @ np.linalg.inv(U_mat)
        # 数値安定性を考慮して np.linalg.solve を使用
        # Y_mat = G_hat @ U_mat  ⟹  U_mat.T @ G_hat.T = Y_mat.T
        G_hat[ni] = np.linalg.solve(U_mat.T, Y_mat.T).T

    return G_hat, cond_U, harm_freqs, harm_idx


def compare_G(G_hat, G_true_vals, harm_freqs, cond_U,
              save_path="identification_result.png"):
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

    # --- 36 成分の振幅比較（6×6 グリッド） ---
    fig, axes = plt.subplots(6, 6, figsize=(20, 18))
    for row in range(6):
        for col in range(6):
            ax = axes[row, col]
            amp_true = np.abs(G_true_vals[:, row, col])
            amp_hat  = np.abs(G_hat[:, row, col])
            ax.semilogy(harm_freqs, amp_true, "o-", color="steelblue",
                        markersize=4, linewidth=1, label="G_true")
            ax.semilogy(harm_freqs, amp_hat,  "x--", color="tomato",
                        markersize=5, linewidth=1, label="G_hat")
            ax.set_title(f"G[{DOF_LABELS[row]},{DOF_LABELS[col]}]", fontsize=7)
            ax.tick_params(labelsize=5)
            ax.grid(True, which="both", alpha=0.3)
            if row < 5:
                ax.set_xticklabels([])

            # 縦軸スケール（transfer_matrix.py と同じ分類ルール）
            from transfer_matrix import _CROSSTALK_RESONANCES
            ct_set = {(r, c) for r, c, *_ in _CROSSTALK_RESONANCES}
            if row == col or (row, col) in ct_set:
                # 対角・クロストーク共振: G_true と G_hat 両方が見えるよう全点の範囲で設定
                vmin = min(np.min(amp_true), np.min(amp_hat))
                vmax = max(np.max(amp_true), np.max(amp_hat))
                if vmax > 0:
                    ax.set_ylim([vmin * 0.5, vmax * 2.0])
            else:
                # フラット非対角: メジアン振幅を中心に 1 decade 幅
                med = np.median(amp_true)
                if med > 0:
                    ax.set_ylim([med / np.sqrt(10), med * np.sqrt(10)])

    axes[0, 0].legend(fontsize=7)
    fig.text(0.5, 0.01, "Frequency [Hz]", ha="center", fontsize=11)
    fig.text(0.01, 0.5, "|G|", va="center", rotation="vertical", fontsize=11)
    plt.suptitle("Identification result: |G_true| vs |G_hat| at harmonic frequencies",
                 fontsize=11)
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.97])
    plt.savefig(save_path, dpi=120)
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
