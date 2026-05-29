#!/usr/bin/env bash
# run_demo.sh —— 视频稳定化一键运行脚本
#
# 用法：
#   bash run_demo.sh [选项]
#
# 选项（均可省略，省略时使用括号内默认值）：
#   --video  <名称>   待处理视频名（不含扩展名）  默认: demo
#   --n_iter <次数>   稳定化迭代轮数              默认: 2
#   --skip   <帧数>   参考帧间隔                  默认: 4
#
# 示例：
#   bash run_demo.sh
#   bash run_demo.sh --n_iter 3 --skip 2
#   bash run_demo.sh --video myvideo
#
# 运行环境要求：
#   conda 环境 videostab（含 torch / opencv / cupy-cuda12x 等依赖）
#   详见 README.md 的环境安装章节

set -e  # 任意命令失败时立即退出

# ── 路径定位 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 默认参数 ──────────────────────────────────────────────────────────────────
VIDEO="demo"
N_ITER=2
SKIP=2

# ── 解析命令行参数 ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)  VIDEO="$2";  shift 2 ;;
        --n_iter) N_ITER="$2"; shift 2 ;;
        --skip)   SKIP="$2";   shift 2 ;;
        *)
            echo "未知参数: $1"
            echo "用法: bash run_demo.sh [--video demo] [--n_iter 2] [--skip 2]"
            exit 1 ;;
    esac
done

# ── 检查 conda 环境 ───────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "[错误] 未找到 conda 命令，请先安装 Miniconda/Anaconda"
    exit 1
fi

ENV_NAME="videostab"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "[错误] conda 环境 '${ENV_NAME}' 不存在，请参考 README.md 创建"
    exit 1
fi

# ── 检查视频文件 ──────────────────────────────────────────────────────────────
VIDEO_PATH="${SCRIPT_DIR}/demo/${VIDEO}.avi"
if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "[错误] 视频文件不存在: ${VIDEO_PATH}"
    echo "       请将待处理视频（.avi）放入 demo/ 目录后重试"
    exit 1
fi

# ── 打印运行参数 ──────────────────────────────────────────────────────────────
echo "============================================================"
echo " 视频稳定化推理"
echo "  视频      : ${VIDEO_PATH}"
echo "  迭代次数  : ${N_ITER}"
echo "  参考帧间隔: ${SKIP}"
echo "  conda 环境: ${ENV_NAME}"
echo "============================================================"

# ── 激活 conda 环境并执行推理 ─────────────────────────────────────────────────
# 使用 conda run 保证在非交互式 shell 中也能正确激活环境
conda run -n "$ENV_NAME" --no-capture-output \
    python "${SCRIPT_DIR}/run_demo.py" \
        --video  "$VIDEO"  \
        --n_iter "$N_ITER" \
        --skip   "$SKIP"
