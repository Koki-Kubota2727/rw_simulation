import numpy as np

# --- Step 2: 取り付け配置と変換行列 T_i の計算 ---
#
# 取り付け位置 3か所、向きの組み合わせで合計 6 配置（3+2+1）。
# 2か所では構造的にランク5が上限。3か所あれば位置差が2方向に広がり
# ランク6を達成できる（向きは軸方向のみで十分）。

# 取り付け位置 3か所 [m]
r_A = np.array([ 0.10,  0.05,  0.00])
r_B = np.array([-0.08,  0.00,  0.12])
r_C = np.array([ 0.05, -0.10,  0.08])

# 6 配置: (位置, スピン軸方向)
#   位置 A: x, y, z の 3 方向
#   位置 B: x, y の 2 方向
#   位置 C: z の 1 方向
CONFIGS_SPEC = [
    (r_A, np.array([1, 0, 0])),   # A-x
    (r_A, np.array([0, 1, 0])),   # A-y
    (r_A, np.array([0, 0, 1])),   # A-z
    (r_B, np.array([1, 0, 0])),   # B-x
    (r_B, np.array([0, 1, 0])),   # B-y
    (r_C, np.array([0, 0, 1])),   # C-z
]

CONFIG_LABELS = ["A-x", "A-y", "A-z", "B-x", "B-y", "C-z"]


def skew(r):
    """3次元ベクトル r の歪対称行列（クロス積行列）"""
    rx, ry, rz = r
    return np.array([[ 0,  -rz,  ry],
                     [ rz,   0, -rx],
                     [-ry,  rx,   0]], dtype=float)


def rotation_to_align(d):
    """
    RW スピン軸が方向 d を向くような回転行列 R を返す。
    R @ [1, 0, 0] = d / |d|  （Rodrigues の回転公式）
    """
    t = np.asarray(d, dtype=float)
    t = t / np.linalg.norm(t)
    x = np.array([1., 0., 0.])
    if np.allclose(t, x):
        return np.eye(3)
    v = np.cross(x, t)
    s = np.linalg.norm(v)
    c = float(np.dot(x, t))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 - c) / s**2


def make_T(r, d):
    """
    位置 r、スピン軸方向 d に対する 6×6 変換行列 T を返す。

        T = [ R            0 ]
            [ S(r) @ R    R ]
    """
    R = rotation_to_align(d)
    T = np.zeros((6, 6))
    T[:3, :3] = R
    T[3:, :3] = skew(r) @ R
    T[3:, 3:] = R
    return T


# 変換行列リスト（同定時のスタック順）
T_LIST = [make_T(r, d) for r, d in CONFIGS_SPEC]


def build_U_mat(u_rw):
    """U_rw ∈ C^6 に対して 6×6 のスタック行列 U_mat を返す"""
    return np.column_stack([T @ u_rw for T in T_LIST])


def plot_configs(save_path="configs_3d.png"):
    """取り付け位置と向きを3D空間上に矢印で描画する"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pos_colors = {"A": "steelblue", "B": "darkorange", "C": "forestgreen"}
    arrow_len = 0.05   # スピン軸矢印の長さ [m]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    plotted_positions = {}
    for label, (r, d) in zip(CONFIG_LABELS, CONFIGS_SPEC):
        pos_key = label[0]  # "A", "B", "C"
        color = pos_colors[pos_key]
        d_norm = np.array(d, dtype=float) / np.linalg.norm(d)

        # 取り付け位置（初回のみマーカー）
        if pos_key not in plotted_positions:
            ax.scatter(*r, color=color, s=80, zorder=5,
                       label=f"pos {pos_key}: {r}")
            plotted_positions[pos_key] = True

        # スピン軸の矢印
        ax.quiver(*r, *(d_norm * arrow_len),
                  color=color, arrow_length_ratio=0.3, linewidth=1.8)
        ax.text(*(r + d_norm * arrow_len * 1.2), label,
                fontsize=8, color=color, ha="center")

    # 衛星筐体の原点
    ax.scatter(0, 0, 0, color="black", s=60, marker="x", zorder=6, label="origin")

    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title("RW mounting positions and spin-axis directions")
    ax.legend(fontsize=8, loc="upper left")

    lim = 0.18
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    print("=== Step 2: 取り付け配置の確認 ===\n")
    for label, (r, d), T in zip(CONFIG_LABELS, CONFIGS_SPEC, T_LIST):
        print(f"[{label}]  r={r}  spin={d}")
        print(T)
        print()

    print("=== ランク・条件数の検証 (n=2000) ===")
    rng = np.random.default_rng(42)
    ranks, conds = [], []
    for _ in range(2000):
        u = rng.standard_normal(6) + 1j * rng.standard_normal(6)
        U = build_U_mat(u)
        ranks.append(np.linalg.matrix_rank(U))
        conds.append(np.linalg.cond(U))
    ranks, conds = np.array(ranks), np.array(conds)
    print(f"rank : min={ranks.min()}  max={ranks.max()}  "
          f"full-rank(6) rate = {(ranks == 6).mean()*100:.1f}%")
    print(f"cond : median={np.median(conds):.1f}  90th={np.percentile(conds, 90):.1f}")

    plot_configs()
