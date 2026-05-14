#!/bin/bash
# =============================================================
# RW シミュレーション 実行スクリプト
#
# 実行手順:
#   Step 1: RW 配置の D 最適化 → result/optimal_configs.py 生成
#   Step 2: 変換行列の確認（optimal_configs.py を参照）
#   Step 3: シミュレーション本体
#
# 使い方:
#   bash run.sh          # Step 1 → 2 → 3 を順番に全実行
#   bash run.sh optimize # Step 1 のみ（配置最適化だけやり直す）
#   bash run.sh sim      # Step 2 → 3 のみ（最適化済みの配置で再実行）
#   bash run.sh all      # 全ファイル実行（signal_gen, transfer_matrix, forward も含む）
# =============================================================

set -e  # エラーが出たら即停止

PYTHON=python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

run_optimize() {
    echo "=========================================="
    echo "Step 1: D 最適配置の探索"
    echo "=========================================="
    $PYTHON d_optimize/d_optimize.py
    echo ""
}

run_sim() {
    echo "=========================================="
    echo "Step 2: 変換行列の確認"
    echo "=========================================="
    $PYTHON position_transform.py
    echo ""

    echo "=========================================="
    echo "Step 3: シミュレーション本体"
    echo "=========================================="
    $PYTHON identify.py
    echo ""
}

run_all() {
    echo "=========================================="
    echo "Step 1: D 最適配置の探索"
    echo "=========================================="
    $PYTHON d_optimize/d_optimize.py
    echo ""

    echo "=========================================="
    echo "Step 2: 変換行列の確認"
    echo "=========================================="
    $PYTHON position_transform.py
    echo ""

    echo "=========================================="
    echo "Step 3: 入力信号の生成"
    echo "=========================================="
    $PYTHON signal_gen.py
    echo ""

    echo "=========================================="
    echo "Step 4: 伝達行列の確認"
    echo "=========================================="
    $PYTHON transfer_matrix.py
    echo ""

    echo "=========================================="
    echo "Step 5: 順問題（出力計算）"
    echo "=========================================="
    $PYTHON forward.py
    echo ""

    echo "=========================================="
    echo "Step 6: シミュレーション本体"
    echo "=========================================="
    $PYTHON identify.py
    echo ""
}

case "${1:-}" in
    optimize)
        run_optimize
        ;;
    sim)
        run_sim
        ;;
    all)
        run_all
        ;;
    *)
        run_optimize
        run_sim
        ;;
esac

echo "完了"
