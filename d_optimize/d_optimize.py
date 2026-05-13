import numpy as np
import sys
import os
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from position_transform import make_T
_POS_LIM = 1.0 * 0.3   # 位置境界スケール [m]
# --- 固定位置（最適化対象外） ---
R1_FIXED = np.array([0.0, 0.0, 0.0])
R2_FIXED = np.array([_POS_LIM, 0.0, 0.0])

# --- デフォルト u_local（静的不釣り合い優位: スピン軸方向に力） ---
U_LOCAL_DEFAULT = np.array([1.0, 1.0, 0.1, 0.015, 0.015, 0.0015])

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
_POS_LIM = 1.0 * 0.3   # 位置境界スケール [m]
BOUNDS = (
    [(-_POS_LIM, _POS_LIM)] * 3                   # r_3
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
                u_local: np.ndarray | None = None) -> np.ndarray:
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


def objective(params: np.ndarray,
              u_local: np.ndarray | None = None) -> float:
    """
    D 最適基準の目的関数（最小化用）。

    Returns
    -------
    float
        -log|det(U_mat)|。U_mat がランク落ちのとき大きなペナルティを返す。
    """
    U = build_U_mat(params, u_local)
    sign, logdet = np.linalg.slogdet(U)
    if sign == 0:
        return 1e10
    return -logdet


def run_optimization(u_local: np.ndarray | None = None,
                     seed: int = 42,
                     popsize: int = 20,
                     maxiter: int = 600,
                     tol: float = 1e-4) -> tuple[np.ndarray, float]:
    """
    D 最適配置を求める。

    大域探索: differential_evolution（多峰性対応）
    局所精緻化: L-BFGS-B

    Parameters
    ----------
    u_local  : RW ローカル外乱ベクトル。None で U_LOCAL_DEFAULT を使用。
    seed     : 乱数シード（再現性）
    popsize  : DE の集団サイズ倍率（大きいほど精度↑・時間↑）
    maxiter  : DE の最大世代数
    tol      : DE の収束許容値

    Returns
    -------
    best_params : 最適パラメータ (15,)
    best_val    : 最小値 = -log|det(U_mat)|
    """
    from scipy.optimize import differential_evolution, minimize

    obj = partial(objective, u_local=u_local)

    print("--- 大域探索: differential_evolution ---")
    res_global = differential_evolution(
        obj,
        BOUNDS,
        seed=seed,
        popsize=popsize,
        maxiter=maxiter,
        tol=tol,
        workers=-1,
        updating="deferred",
        disp=True,
    )
    print(f"DE 終了: f={res_global.fun:.6f}  nfev={res_global.nfev}")

    print("\n--- 局所精緻化: L-BFGS-B ---")
    res_local = minimize(
        obj,
        res_global.x,
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 2000},
    )
    print(f"L-BFGS-B 終了: f={res_local.fun:.6f}  nit={res_local.nit}")

    best = res_local if res_local.fun < res_global.fun else res_global
    return best.x, best.fun


def plot_configs_3d(params: np.ndarray,
                    u_local: np.ndarray | None = None,
                    save_path: str | None = None) -> None:
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
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    r3, dirs = unpack(params)
    configs = build_configs(params)

    pos_labels = ["pos1", "pos1", "pos2", "pos2", "pos3", "pos3"]
    rw_labels  = ["RW1-1", "RW1-2", "RW2-1", "RW2-2", "RW3-1", "RW3-2"]
    colors     = ["steelblue", "steelblue", "darkorange", "darkorange",
                  "forestgreen", "forestgreen"]
    pos_colors = {"pos1": "steelblue", "pos2": "darkorange", "pos3": "forestgreen"}

    all_pos = np.array([R1_FIXED, R2_FIXED, r3])
    span = np.linalg.norm(all_pos.max(axis=0) - all_pos.min(axis=0))
    arrow_len = max(span * 0.2, 0.04)

    fig = plt.figure(figsize=(10, 7))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # 取り付け位置マーカー（凡例用）
    plotted = set()
    for i, (r, d) in enumerate(configs):
        pk = pos_labels[i]
        if pk not in plotted:
            ax.scatter(*r, color=pos_colors[pk], s=100, zorder=5)
            plotted.add(pk)

    # スピン軸（実線矢印）+ ラベル
    for i, (r, d) in enumerate(configs):
        ax.quiver(*r, *(d * arrow_len),
                  color=colors[i], arrow_length_ratio=0.3, linewidth=2.0)
        ax.text(*(r + d * arrow_len * 1.35), rw_labels[i],
                fontsize=8, color=colors[i], ha="center")

    # ランチベクトル（点線矢印）= T_i @ u_local の力成分
    if u_local is not None:
        for i, (r, d) in enumerate(configs):
            T = make_T(r, d)
            force_dir = (T @ u_local)[:3]
            fn = np.linalg.norm(force_dir)
            if fn > 1e-12:
                ax.quiver(*r, *(force_dir / fn * arrow_len * 0.7),
                          color=colors[i], linestyle="dashed",
                          arrow_length_ratio=0.4, linewidth=1.2, alpha=0.6)

    ax.scatter(0, 0, 0, color="black", s=60, marker="x", zorder=6)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("RW mounting positions and spin-axis directions")

    # --- 凡例をプロキシアーティストで手動構成 ---
    legend_handles = [
        mpatches.Patch(color="steelblue",   label="pos1 (r1_fixed)"),
        mpatches.Patch(color="darkorange",  label="pos2 (r2_fixed)"),
        mpatches.Patch(color="forestgreen", label="pos3 (optimized)"),
        mlines.Line2D([], [], color="gray", linewidth=2,
                      label="line: spin axis direction"),
    ]
    if u_local is not None:
        legend_handles.append(
            mlines.Line2D([], [], color="gray", linewidth=1.2,
                          linestyle="dashed", alpha=0.7,
                          label="dashed: launch vector (force components of T@u)")
        )
    legend_handles.append(
        mlines.Line2D([], [], color="black", marker="x", linestyle="None",
                      markersize=8, label="origin")
    )
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left")

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


def export_configs(params: np.ndarray,
                   logdet: float,
                   cond: float,
                   save_path: str = "d_optimize/result/optimal_configs.py") -> None:
    """
    最適配置を position_transform.py 互換の Python ファイルとして書き出す。
    既存シミュレーションフローに差し込める形（CONFIGS_SPEC / CONFIG_LABELS）。
    """
    r3, dirs = unpack(params)
    positions = [R1_FIXED, R1_FIXED, R2_FIXED, R2_FIXED, r3, r3]
    rw_labels = ["pos1-RW1", "pos1-RW2", "pos2-RW1", "pos2-RW2", "pos3-RW1", "pos3-RW2"]

    def arr(v: np.ndarray) -> str:
        return f"np.array([{v[0]:.8f}, {v[1]:.8f}, {v[2]:.8f}])"

    lines = [
        "import numpy as np",
        "",
        "# === D最適化結果 (d_optimize.py で自動生成) ===",
        f"# log|det(U_mat)| = {logdet:.6f}",
        f"# cond(U_mat)     = {cond:.2f}",
        f"# rank            = 6",
        "",
        f"r_1 = {arr(R1_FIXED)}",
        f"r_2 = {arr(R2_FIXED)}",
        f"r_3 = {arr(r3)}",
        "",
        "CONFIGS_SPEC = [",
    ]
    for label, r, d in zip(rw_labels, positions, dirs):
        r_name = "r_1" if np.allclose(r, R1_FIXED) else ("r_2" if np.allclose(r, R2_FIXED) else "r_3")
        lines.append(f"    ({r_name}, {arr(d)}),  # {label}")
    lines += [
        "]",
        "",
        f'CONFIG_LABELS = {rw_labels}',
    ]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved: {save_path}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("=== Step 3: D 最適配置の探索 ===\n")
    best_params, best_val = run_optimization()

    print(f"\n=== 最適解 ===")
    print(f"log|det(U_mat)| = {-best_val:.6f}")

    r3_opt, dirs_opt = unpack(best_params)
    print(f"\n取り付け位置:")
    print(f"  r1 = {R1_FIXED}")
    print(f"  r2 = {R2_FIXED}")
    print(f"  r3 = {r3_opt}")
    print(f"\nスピン軸方向:")
    labels = ["RW1-1", "RW1-2", "RW2-1", "RW2-2", "RW3-1", "RW3-2"]
    for label, d in zip(labels, dirs_opt):
        print(f"  {label}: {d}")

    U_opt = build_U_mat(best_params)
    _, logdet_opt = np.linalg.slogdet(U_opt)
    cond_opt = np.linalg.cond(U_opt)
    print(f"\nrank = {np.linalg.matrix_rank(U_opt)}")
    print(f"log|det| = {logdet_opt:.6f}")
    print(f"cond(U_mat) = {cond_opt:.2f}")

    os.makedirs("d_optimize/result", exist_ok=True)

    # エクスポート
    export_configs(best_params, logdet=logdet_opt, cond=cond_opt,
                   save_path="d_optimize/result/optimal_configs.py")

    # 3D 可視化
    plot_configs_3d(best_params, u_local=U_LOCAL_DEFAULT,
                    save_path="d_optimize/result/optimal_configs.png")
