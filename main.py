import numpy as np
import matplotlib.pyplot as plt

# --- Step 0: パラメータ設定 ---

# サンプリング
fs = 1000.0                    # サンプリング周波数 [Hz]
T  = 10.0                      # 計測時間 [s]
N  = int(fs * T)               # サンプル数

t  = np.linspace(0, T, N, endpoint=False)   # 時間軸 [s]

# 周波数軸（実FFT用）
freq  = np.fft.rfftfreq(N, d=1/fs)          # [Hz]
omega = 2 * np.pi * freq                    # [rad/s]

# RW 擾乱パラメータ
Omega = 2 * np.pi * 20.0       # RW 基本角速度 [rad/s]（例: 20 Hz 回転）
n_harmonics = 6                # 使用する高調波の最大次数

# 入出力の次元
N_DOF = 6                      # 入力・出力ともに 6 自由度

# 取り付け配置
r1 = np.array([ 0.10,  0.05,  0.00])   # 位置1 [m]
r2 = np.array([-0.08,  0.00,  0.12])   # 位置2 [m]

if __name__ == "__main__":
    print("=== Step 0: パラメータ確認 ===")
    print(f"  サンプリング周波数  fs = {fs} Hz")
    print(f"  計測時間            T  = {T} s")
    print(f"  サンプル数          N  = {N}")
    print(f"  周波数分解能        Δf = {freq[1]:.4f} Hz")
    print(f"  RW 基本周波数       f0 = {Omega / (2*np.pi):.1f} Hz")
    print(f"  高調波次数          n  = 1 ~ {n_harmonics}")
    print(f"  高調波周波数        ", [f"{Omega/(2*np.pi)*n:.1f} Hz" for n in range(1, n_harmonics+1)])
    print(f"  取り付け位置 r1 = {r1} m")
    print(f"  取り付け位置 r2 = {r2} m")
