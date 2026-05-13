import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from position_transform import make_T

# --- 固定位置（最適化対象外） ---
R1_FIXED = np.array([0.0, 0.0, 0.0])
R2_FIXED = np.array([0.20, 0.0, 0.0])

# --- デフォルト u_local（静的不釣り合い優位: スピン軸方向に力） ---
U_LOCAL_DEFAULT = np.array([1., 0., 0., 0., 0., 0.])

# --- 設計変数レイアウト ---
# params[0:3]   : r_3 = (x, y, z)
# params[3:5]   : (θ_1, φ_1) → d_11  位置1・RW1
# params[5:7]   : (θ_2, φ_2) → d_12  位置1・RW2
# params[7:9]   : (θ_3, φ_3) → d_21  位置2・RW1
# params[9:11]  : (θ_4, φ_4) → d_22  位置2・RW2
# params[11:13] : (θ_5, φ_5) → d_31  位置3・RW1
# params[13:15] : (θ_6, φ_6) → d_32  位置3・RW2
N_PARAMS = 15

# 変数境界
#   r_3  : 衛星サイズに合わせて ±0.5 m 程度
#   θ    : [0, π]
#   φ    : [-π/2, π/2]  (d_x = sinθ cosφ ≥ 0 を保証)
BOUNDS = (
    [(-0.5,  0.5), (-0.5, 0.5), (-0.5, 0.5)]   # r_3
    + [(0, np.pi), (-np.pi / 2, np.pi / 2)] * 6  # 6 軸 × (θ, φ)
)


def spherical_to_unit(theta: float, phi: float) -> np.ndarray:
    """球面角 (θ, φ) → 単位ベクトル d。d_x = sinθ cosφ ≥ 0 が保証される。"""
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def unpack(params: np.ndarray):
    """
    15 次元パラメータベクトルを (r_3, directions) に展開する。

    Returns
    -------
    r3 : ndarray (3,)
    dirs : list of 6 ndarray (3,)  [d_11, d_12, d_21, d_22, d_31, d_32]
    """
    r3 = params[0:3].copy()
    dirs = [
        spherical_to_unit(params[3 + 2 * k], params[3 + 2 * k + 1])
        for k in range(6)
    ]
    return r3, dirs


def build_configs(params: np.ndarray):
    """
    params から 6 配置の (r_i, d_i) リストを生成する。

    配置順:
      0: (r1_fixed, d_11)
      1: (r1_fixed, d_12)
      2: (r2_fixed, d_21)
      3: (r2_fixed, d_22)
      4: (r3,       d_31)
      5: (r3,       d_32)
    """
    r3, dirs = unpack(params)
    return [
        (R1_FIXED, dirs[0]),
        (R1_FIXED, dirs[1]),
        (R2_FIXED, dirs[2]),
        (R2_FIXED, dirs[3]),
        (r3,       dirs[4]),
        (r3,       dirs[5]),
    ]


def build_U_mat(params: np.ndarray,
                u_local: np.ndarray = None) -> np.ndarray:
    """
    params から U_mat (6×6) を構築する。

    Parameters
    ----------
    params  : 15 次元設計変数ベクトル
    u_local : RW ローカル座標での外乱ベクトル (6,)。
              None のとき U_LOCAL_DEFAULT（静的不釣り合い優位）を使用。
    """
    if u_local is None:
        u_local = U_LOCAL_DEFAULT
    configs = build_configs(params)
    cols = [make_T(r, d) @ u_local for r, d in configs]
    return np.column_stack(cols)


def plot_configs_3d(params: np.ndarray,
                    u_local: np.ndarray = None,
                    save_path: str = None) -> None:
    """
    RW 取り付け配置（位置・スピン軸）を 3D 空間に描画する。

    Parameters
    ----------
    params    : 15 次元設計変数ベクトル
    u_local   : RW ローカル外乱ベクトル (6,)。指定するとランチベクトル
                (T_i @ u_local の力成分) も矢印で重ね描きする。
    save_path : 保存先パス。None のとき保存しない。
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    r3, dirs = unpack(params)
    positions = [R1_FIXED, R1_FIXED, R2_FIXED, R2_FIXED, r3, r3]
    configs = build_configs(params)

    pos_labels = ["pos1", "pos1", "pos2", "pos2", "pos3", "pos3"]
    rw_labels  = ["RW1-1", "RW1-2", "RW2-1", "RW2-2", "RW3-1", "RW3-2"]
    colors     = ["steelblue", "steelblue", "darkorange", "darkorange",
                  "forestgreen", "forestgreen"]
    pos_colors = {"pos1": "steelblue", "pos2": "darkorange", "pos3": "forestgreen"}

    # スピン軸矢印の長さ: 位置間距離の 20% 程度
    all_pos = np.array([R1_FIXED, R2_FIXED, r3])
    span = np.linalg.norm(all_pos.max(axis=0) - all_pos.min(axis=0))
    arrow_len = max(span * 0.2, 0.04)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # 取り付け位置のマーカー（重複なし）
    plotted = set()
    for i, (r, d) in enumerate(configs):
        pk = pos_labels[i]
        if pk not in plotted:
            ax.scatter(*r, color=pos_colors[pk], s=100, zorder=5,
                       label=f"{pk}: ({r[0]:.3f}, {r[1]:.3f}, {r[2]:.3f})")
            plotted.add(pk)

    # スピン軸の矢印
    for i, (r, d) in enumerate(configs):
        ax.quiver(*r, *(d * arrow_len),
                  color=colors[i], arrow_length_ratio=0.3, linewidth=2.0)
        ax.text(*(r + d * arrow_len * 1.3), rw_labels[i],
                fontsize=8, color=colors[i], ha="center")

    # u_local が与えられたとき: ランチベクトル（力成分のみ）を追加描画
    if u_local is not None:
        for i, (r, d) in enumerate(configs):
            T = make_T(r, d)
            wrench = T @ u_local
            force_dir = wrench[:3]
            fn = np.linalg.norm(force_dir)
            if fn > 1e-12:
                ax.quiver(*r, *(force_dir / fn * arrow_len * 0.7),
                          color=colors[i], linestyle="dashed",
                          arrow_length_ratio=0.4, linewidth=1.2, alpha=0.6)

    ax.scatter(0, 0, 0, color="black", s=60, marker="x", zorder=6, label="origin")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("RW mounting positions and spin-axis directions")
    ax.legend(fontsize=8, loc="upper left")

    # 軸範囲を全位置 + 矢印が収まるよう設定
    margin = arrow_len * 1.5
    lim_lo = all_pos.min(axis=0) - margin
    lim_hi = all_pos.max(axis=0) + margin
    ax.set_xlim(lim_lo[0], lim_hi[0])
    ax.set_ylim(lim_lo[1], lim_hi[1])
    ax.set_zlim(lim_lo[2], lim_hi[2])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"saved: {save_path}")
    plt.show()


# --- 動作確認 ---
if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    rng = np.random.default_rng(0)
    p0 = np.zeros(N_PARAMS)
    p0[0:3] = [0.05, 0.05, 0.05]
    p0[3:] = rng.uniform(
        [0, -np.pi / 2] * 6,
        [np.pi,  np.pi / 2] * 6,
    )

    r3, dirs = unpack(p0)
    print("=== Step 1: 骨格確認 ===")
    print(f"r1 = {R1_FIXED}")
    print(f"r2 = {R2_FIXED}")
    print(f"r3 = {r3}")
    for i, d in enumerate(dirs):
        print(f"d_{i+1} = {d}  |d|={np.linalg.norm(d):.6f}")

    # デフォルト u_local でのチェック
    U_default = build_U_mat(p0)
    print(f"\n--- u_local = default (static imbalance) ---")
    print(f"U_mat:\n{U_default}")
    sign, logdet = np.linalg.slogdet(U_default)
    print(f"rank={np.linalg.matrix_rank(U_default)}  log|det|={logdet:.4f}")

    # カスタム u_local でのチェック
    u_custom = np.array([1., 0., 0., 0., 1., 0.])
    u_custom = u_custom / np.linalg.norm(u_custom)
    U_custom = build_U_mat(p0, u_local=u_custom)
    print(f"\n--- u_local = {u_custom} ---")
    sign2, logdet2 = np.linalg.slogdet(U_custom)
    print(f"rank={np.linalg.matrix_rank(U_custom)}  log|det|={logdet2:.4f}")

    # 3D 描画
    plot_configs_3d(p0, u_local=U_LOCAL_DEFAULT,
                    save_path="d_optimize/result/step1_configs.png")
