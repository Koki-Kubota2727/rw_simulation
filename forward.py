import numpy as np
import matplotlib.pyplot as plt

# --- Step 4: 出力の計算（順問題） ---
# Y_i(ω) = G(ω) @ U_i(ω),   U_i(ω) = T_i @ U_rw(ω)

DOF_LABELS = ["x1", "x2", "x3", "thx", "thy", "thz"]


def compute_outputs(U_rw_freq, T_list, G_omega, fs,
                    add_noise=True, noise_snr_db=20.0):
    """
    各配置の実効入力・出力スペクトルと時間信号を計算する。

    Parameters
    ----------
    U_rw_freq : ndarray, shape (N_freq, 6)  複素スペクトル (rfft / N 正規化済み)
    T_list    : list of (6,6) ndarray       変換行列リスト（N_configs 個）
    G_omega   : ndarray, shape (N_freq,6,6) 真の伝達行列
    fs        : float                        サンプリング周波数 [Hz]
    add_noise : bool                         出力にノイズを加えるか
    noise_snr_db : float or array-like (6,)
        DOFごとの SNR [dB]。スカラーなら全成分共通。
        順序: [x1, x2, x3, thx, thy, thz]

    Returns
    -------
    U_freq_list : list of (N_freq, 6) ndarray  各配置の実効入力スペクトル
    Y_freq_list : list of (N_freq, 6) ndarray  各配置の出力スペクトル
    y_time_list : list of (N_time, 6) ndarray  各配置の出力時間信号
    """
    N_freq = U_rw_freq.shape[0]
    N_time = (N_freq - 1) * 2   # rfft の逆変換サンプル数

    snr_arr = np.broadcast_to(np.asarray(noise_snr_db, dtype=float), (6,)).copy()

    U_freq_list, Y_freq_list, y_time_list = [], [], []

    for T in T_list:
        # 実効入力: U_i(ω) = T @ U_rw(ω)  （各周波数で独立に適用）
        U_i = U_rw_freq @ T.T          # (N_freq, 6)

        # 出力: Y_i(ω) = G(ω) @ U_i(ω)
        Y_i = np.einsum("fij,fj->fi", G_omega, U_i)   # (N_freq, 6)

        # ノイズの加算（DOFごとに独立した SNR）
        if add_noise:
            signal_power = np.mean(np.abs(Y_i) ** 2, axis=0)          # (6,)
            noise_power  = signal_power / (10 ** (snr_arr / 10))       # (6,)
            rng = np.random.default_rng()
            noise = (rng.standard_normal(Y_i.shape) +
                     1j * rng.standard_normal(Y_i.shape)) * np.sqrt(noise_power / 2)
            Y_i = Y_i + noise

        # 時間信号に戻す（rfft の /N 正規化を戻してから irfft）
        y_i = np.fft.irfft(Y_i * N_time, axis=0)      # (N_time, 6)

        U_freq_list.append(U_i)
        Y_freq_list.append(Y_i)
        y_time_list.append(y_i)

    return U_freq_list, Y_freq_list, y_time_list


def plot_time_domain(t, y_time_list, config_labels, save_path="output_time.png",
                     t_start=0.0, t_end=1.0):
    """各配置の出力時間信号を描画（6成分 × N_configs）

    t_start, t_end : 表示する時間範囲 [s]
    """
    t_full = t[:len(y_time_list[0])]
    mask = (t_full >= t_start) & (t_full <= t_end)

    N_configs = len(y_time_list)
    fig, axes = plt.subplots(6, N_configs, figsize=(3 * N_configs, 10), sharex=True)

    for ci, (y, label) in enumerate(zip(y_time_list, config_labels)):
        for k in range(6):
            ax = axes[k, ci]
            ax.plot(t_full[mask], y[mask, k], linewidth=0.8, color="steelblue")
            if k == 0:
                ax.set_title(label, fontsize=9)
            if ci == 0:
                ax.set_ylabel(DOF_LABELS[k], fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3)
        axes[-1, ci].set_xlabel("t [s]", fontsize=8)

    plt.suptitle(f"Output time signals  y_i(t)  [{t_start:.2f} – {t_end:.2f} s]",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"saved: {save_path}")
    plt.show()


def plot_freq_domain(freq, U_freq_list, Y_freq_list, G_omega,
                     Omega, n_harmonics, config_labels,
                     save_path="output_freq.png"):
    """
    各配置の出力スペクトルと入力スペクトルを並べて描画。
    高調波位置に縦線を引き、G*U の掛け算効果を視覚化する。
    """
    N_configs = len(Y_freq_list)
    f0 = Omega / (2 * np.pi)
    harmonic_freqs = [n * f0 for n in range(1, n_harmonics + 1)]

    # 描画: 各配置ごとに 1 列、上段=入力 |U_i|、下段=出力 |Y_i|
    fig, axes = plt.subplots(2, N_configs, figsize=(3.2 * N_configs, 7))

    colors = plt.cm.tab10(np.linspace(0, 0.6, 6))

    for ci, (U_i, Y_i, label) in enumerate(zip(U_freq_list, Y_freq_list, config_labels)):
        ax_u = axes[0, ci]
        ax_y = axes[1, ci]

        for k in range(6):
            ax_u.semilogy(freq, np.abs(U_i[:, k]), color=colors[k],
                          linewidth=0.7, label=DOF_LABELS[k])
            ax_y.semilogy(freq, np.abs(Y_i[:, k]), color=colors[k],
                          linewidth=0.7, label=DOF_LABELS[k])

        for fh in harmonic_freqs:
            ax_u.axvline(fh, color="gray", linestyle="--", linewidth=0.5)
            ax_y.axvline(fh, color="gray", linestyle="--", linewidth=0.5)

        ax_u.set_title(f"{label}\n|U_i(f)|", fontsize=8)
        ax_y.set_title(f"{label}\n|Y_i(f)| = |G·U_i|", fontsize=8)

        for ax in [ax_u, ax_y]:
            ax.set_xlim([0, (n_harmonics + 1) * f0])
            ax.tick_params(labelsize=6)
            ax.grid(True, which="both", alpha=0.3)

        axes[1, ci].set_xlabel("Frequency [Hz]", fontsize=8)

    axes[0, 0].legend(fontsize=6, loc="upper right")
    axes[0, 0].set_ylabel("|U_i(f)|", fontsize=9)
    axes[1, 0].set_ylabel("|Y_i(f)|", fontsize=9)

    # G_true の共振周波数を下段に重ねる（対角成分ピーク位置）
    from transfer_matrix import _DOF_PARAMS
    for k, (m, c, stiff) in enumerate(_DOF_PARAMS):
        fn = np.sqrt(stiff / m) / (2 * np.pi)
        for ci in range(N_configs):
            axes[1, ci].axvline(fn, color=colors[k], linestyle=":",
                                linewidth=0.8)

    plt.suptitle("Forward problem: |U_i| → |G·U_i| = |Y_i|\n"
                 "(gray dashed: harmonics n·f0,  dotted: G_true resonances)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    from main import t, fs, freq, omega, Omega, n_harmonics
    from transfer_matrix import G_true
    from position_transform import T_LIST, CONFIG_LABELS
    from signal_gen import generate_u_rw, fft_signal

    # 入力信号の生成
    u_rw, _, _ = generate_u_rw(t, Omega, n_harmonics)
    _, U_rw_freq = fft_signal(u_rw, fs)

    # G_true の計算
    G = G_true(omega)

    # DOFごとの SNR [dB]: [x1, x2, x3, thx, thy, thz]
    # x3(Fz方向) と thz(Mz方向) は信号が弱くクロストーク影響が大きいため低SNR
    noise_snr_db = [20.0, 20.0, 15.0, 20.0, 20.0, 5.0]

    # 順問題
    U_freq_list, Y_freq_list, y_time_list = compute_outputs(
        U_rw_freq, T_LIST, G, fs,
        add_noise=True,
        noise_snr_db=noise_snr_db,
    )

    # プロット（t_start〜t_end の範囲だけ表示）
    plot_time_domain(t, y_time_list, CONFIG_LABELS, t_start=1.0, t_end=1.3)
    plot_freq_domain(freq, U_freq_list, Y_freq_list, G,
                     Omega, n_harmonics, CONFIG_LABELS)
