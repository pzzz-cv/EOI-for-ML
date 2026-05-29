"""
run_demo.py —— 视频稳定化推理入口脚本

功能：
  1. 将 demo/ 目录下的视频解帧（如尚未解帧）
  2. 调用 main_raft.main() 执行多轮迭代稳定化
  3. 输出稳定化视频和质量评估指标

用法：
  python run_demo.py [--video demo] [--n_iter 2] [--skip 2]

参数说明：
  --video   指定处理哪个视频（不含扩展名），默认处理 demo/demo.avi
  --n_iter  稳定化迭代次数，默认 2
  --skip    帧插值时向前/向后取参考帧的间隔帧数，默认 2
"""

import os
import sys
# 确保项目根目录在 Python 路径中，保证各子模块可正常 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import torch
import argparse
import numpy as np
from src.config import Config

# 导入主推理模块，同时保留模块引用以便后续注入全局变量
import main_raft as _main_raft_module
from main_raft import main

# ──────────────────────────────────────────────────────────────────────────────
# metrics 替换：原版 utils/metrics.py 依赖外部数据集的 .pkl 文件（硬编码路径），
# 此处替换为独立版本，仅基于输出帧本身计算三项质量指标，无需额外数据。
# ──────────────────────────────────────────────────────────────────────────────
import utils.metrics as _metrics_module


def _metrics_standalone(original_dir, pred_dir, log_path):
    """
    独立版质量评估函数，计算三项视频稳定化指标：

    - CR（Cropping Ratio，裁剪比率）：
        稳定化后相对于原始帧的有效画幅比例，越接近 1 越好。
    - DV（Distortion Value，畸变值）：
        单应矩阵特征值之比，反映几何畸变程度，越接近 1 越好。
    - SS（Stability Score，稳定性分数）：
        基于相邻帧间相机运动轨迹的傅里叶频谱，
        低频能量占比越高说明运动越平稳，越接近 1 越好。

    参数：
        original_dir  原始输入帧目录（以 / 结尾）
        pred_dir      稳定化输出帧目录（以 / 结尾）
        log_path      指标写入的日志文件路径
    """
    # 获取预测帧列表（按文件名排序）
    image_paths = sorted([p for p in os.listdir(pred_dir) if p.endswith(".png")])

    # 使用 SIFT 特征点检测 + 暴力匹配计算帧间单应矩阵
    bf   = cv2.BFMatcher()
    sift = cv2.SIFT_create()
    MIN_MATCH_COUNT = 10   # 有效匹配点最低数量
    ratio           = 0.7  # Lowe's ratio test 阈值
    thresh          = 5.0  # RANSAC 重投影误差阈值

    CR_seq = []   # 逐帧裁剪比率序列
    DV_seq = []   # 逐帧畸变值序列
    Pt     = np.eye(3)   # 累积单应矩阵（用于计算绝对轨迹）
    P_seq  = []   # 累积单应矩阵序列（用于计算稳定性）

    for i in range(len(image_paths)):
        # 读取原始帧与对应稳定化帧（灰度）
        img1  = cv2.imread(original_dir + image_paths[i], 0)
        img1o = cv2.imread(pred_dir     + image_paths[i], 0)

        # 检测 SIFT 特征点并计算描述子
        kp1,  d1  = sift.detectAndCompute(img1,  None)
        kp1o, d1o = sift.detectAndCompute(img1o, None)
        if d1 is None or d1o is None:
            continue

        # 用 KNN 匹配后通过 ratio test 筛选优质匹配
        matches = bf.knnMatch(d1, d1o, k=2)
        good    = [m for m, n in matches if m.distance < ratio * n.distance]

        if len(good) > MIN_MATCH_COUNT:
            src = np.float32([kp1 [m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp1o[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            # 用 RANSAC 估计单应矩阵
            M, _ = cv2.findHomography(src, dst, cv2.RANSAC, thresh)
            if M is not None:
                # 计算裁剪比率（矩阵缩放因子的倒数）
                CR_seq.append(1 / max(np.sqrt(M[0, 1]**2 + M[0, 0]**2), 1e-6))
                # 计算畸变值（特征值之比）
                w, _ = np.linalg.eig(M[0:2, 0:2])
                w    = np.sort(np.abs(w))[::-1]
                DV_seq.append(w[1] / w[0] if w[0] > 0 else 0)

        # 计算相邻稳定化帧之间的单应矩阵（用于稳定性评估）
        if i + 1 < len(image_paths):
            img2o        = cv2.imread(pred_dir + image_paths[i + 1], 0)
            kp2o, d2o    = sift.detectAndCompute(img2o, None)
            if d1o is not None and d2o is not None:
                m2 = bf.knnMatch(d1o, d2o, k=2)
                g2 = [m for m, n in m2 if m.distance < ratio * n.distance]
                if len(g2) > MIN_MATCH_COUNT:
                    s2 = np.float32([kp1o[m.queryIdx].pt for m in g2]).reshape(-1, 1, 2)
                    d2 = np.float32([kp2o[m.trainIdx].pt for m in g2]).reshape(-1, 1, 2)
                    M2, _ = cv2.findHomography(s2, d2, cv2.RANSAC, thresh)
                    if M2 is not None:
                        # 累乘得到绝对轨迹
                        P_seq.append(np.matmul(Pt, M2))
                        Pt = np.matmul(Pt, M2)

        sys.stdout.write('\rMetrics frame: %d/%d' % (i + 1, len(image_paths)))
        sys.stdout.flush()
    print()

    if not CR_seq:
        print('[metrics] 无足够特征点，跳过指标计算')
        return

    # 提取绝对轨迹中的平移量和旋转角（用于 SS 计算）
    P_seq_t, P_seq_r = [], []
    for Mp in P_seq:
        P_seq_t.append(np.sqrt(Mp[0, 2]**2 + Mp[1, 2]**2))
        P_seq_r.append(np.arctan2(Mp[1, 0], Mp[0, 0]) * 180 / np.pi)

    def _stability_score(arr):
        """
        通过 FFT 计算稳定性分数：
        低频（前5个分量）能量占总能量比例越高，说明运动曲线越平滑。
        """
        if len(arr) < 2:
            return 0.0
        f = np.abs(np.fft.fft(arr))**2
        f = f[1:]                              # 去掉直流分量
        f = f[:len(f) // 2]                    # 取单边频谱
        return float(np.sum(f[:5]) / np.sum(f)) if np.sum(f) > 0 else 0.0

    SS_t = _stability_score(P_seq_t)   # 平移方向稳定性分数
    SS_r = _stability_score(P_seq_r)   # 旋转方向稳定性分数

    # 打印结果
    print('\n***Cropping ratio (Avg, Min):')
    print('{0:.4f} | {1:.4f}'.format(
        min(float(np.mean(CR_seq)), 1), min(float(np.min(CR_seq)), 1)))
    print('***Distortion value:')
    print('{0:.4f}'.format(abs(float(np.min(DV_seq)))))
    print('***Stability Score (Avg, Trans, Rot):')
    print('{0:.4f} | {1:.4f} | {2:.4f}'.format((SS_t + SS_r) / 2, SS_t, SS_r))

    # 写入日志文件
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write('CR: {0:.4f} | {1:.4f}\nDV: {2:.4f}\nSS: {3:.4f} | {4:.4f} | {5:.4f}\n'.format(
            min(float(np.mean(CR_seq)), 1), min(float(np.min(CR_seq)), 1),
            abs(float(np.min(DV_seq))), (SS_t + SS_r) / 2, SS_t, SS_r))


# 将 metrics 函数替换为独立版本（patch utils 模块和 main_raft 模块中的引用）
_metrics_module.metrics   = _metrics_standalone
_main_raft_module.metrics = _metrics_standalone

# ──────────────────────────────────────────────────────────────────────────────


def extract_frames(video_path, out_dir):
    """
    将视频文件解帧为 PNG 序列。

    参数：
        video_path  输入视频文件路径（.avi / .mp4 等）
        out_dir     输出帧目录，帧文件名格式为 00000.png, 00001.png, ...

    返回：
        解出的总帧数
    """
    os.makedirs(out_dir, exist_ok=True)
    cap   = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, '%05d.png' % count), frame)
        count += 1
    cap.release()
    return count


def build_iter0_symlinks(frame_dir, iter0_hat_dir):
    """
    为第 0 轮迭代的 f_hat 目录创建软链接，指向原始输入帧。
    main_raft.main() 在每轮迭代时从上一轮的 f_hat 目录读取输入，
    第 0 轮不存在上一轮输出，因此用软链接指向原始帧作为初始输入。

    参数：
        frame_dir      原始帧目录
        iter0_hat_dir  iter0/f_hat 目录路径
    """
    os.makedirs(iter0_hat_dir, exist_ok=True)
    for fname in os.listdir(frame_dir):
        src = os.path.join(frame_dir, fname)
        dst = os.path.join(iter0_hat_dir, fname)
        if not os.path.exists(dst):
            os.symlink(src, dst)


# ── 命令行参数解析 ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='视频稳定化推理脚本')
parser.add_argument('--video',  default='demo',
                    help='demo/ 目录下的视频名（不含扩展名），默认为 demo')
parser.add_argument('--n_iter', type=int, default=2,
                    help='稳定化迭代次数，默认 2')
parser.add_argument('--skip',   type=int, default=4,
                    help='参考帧间隔：每帧以前后第 skip 帧作为插值参考，默认 4')
cli_args = parser.parse_args()

# ── 构造推理参数 Namespace ──────────────────────────────────────────────────────
args = argparse.Namespace(
    edge_guide            = True,            # 是否使用边缘引导光流补全
    mode                  = 'edge',          # 运行模式
    outroot               = os.path.join(BASE, 'demo_output'),
    consistencyThres      = 5.0,             # 前后向光流一致性误差阈值
    alpha                 = 0.1,
    homography            = False,           # 是否用单应变换辅助光流计算
    device                = [0],
    flowmodel             = 'RAFT',          # 光流模型（固定使用 RAFT）
    raft_model            = os.path.join(BASE, 'weight/raft-things.pth'),
    small                 = False,           # 是否使用 RAFT-small 轻量版
    mixed_precision       = False,
    alternate_corr        = False,
    deepfill_model        = os.path.join(BASE, 'weight/imagenet_deepfill.pth'),
    edge_completion_model = os.path.join(BASE, 'weight/edge_completion.pth'),
    n_iter                = cli_args.n_iter, # 稳定化迭代次数
    skip                  = cli_args.skip,   # 参考帧间隔
    difrint_model         = os.path.join(BASE, 'weight/DIFNet2.pth'),
    path                  = os.path.join(BASE, 'checkpoints/ws_wg'),  # 流补全模型目录
    model                 = None,
    itera                 = 1,               # 流补全内部迭代次数
)
args.outroot_origin = args.outroot

# ── 加载流补全模型配置 ─────────────────────────────────────────────────────────
config_path = os.path.join(args.path, 'config_train.yml')
config      = Config(config_path)

# 自动选择 GPU / CPU
if torch.cuda.is_available():
    config.DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True   # 固定输入尺寸时可加速 cudnn
    gpu_name = torch.cuda.get_device_name(0)
else:
    config.DEVICE = torch.device("cpu")
    gpu_name = "CPU"

# ── 注入 main_raft 模块级全局变量（main() 内部直接引用这些变量）─────────────
model_name = 'ws_wg'
scene_name = 'demo'
_main_raft_module.model_name = model_name
_main_raft_module.scene_name = scene_name

# ── 确定待处理视频列表 ─────────────────────────────────────────────────────────
video_name   = cli_args.video                                    # 如 "demo"
video_path   = os.path.join(BASE, 'demo', video_name + '.avi')  # demo/demo.avi
frames_dir   = os.path.join(BASE, 'demo_frames', video_name)    # demo_frames/demo/

print("=" * 60)
print(f"视频稳定化推理  |  GPU: {gpu_name}")
print(f"视频: {video_path}")
print(f"迭代次数: {args.n_iter}  参考帧间隔: {args.skip}")
print("=" * 60)

# ── Step 1：解帧 ──────────────────────────────────────────────────────────────
if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
    print(f"\n[1/3] 解帧中：{video_path} -> {frames_dir}")
    n = extract_frames(video_path, frames_dir)
    print(f"      共解出 {n} 帧")
else:
    n = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    print(f"\n[1/3] 帧目录已存在，共 {n} 帧，跳过解帧")

# ── Step 2：准备 iter0 软链接目录 ─────────────────────────────────────────────
print(f"\n[2/3] 准备迭代初始目录...")
iter0_hat_dir = os.path.join(
    args.outroot, model_name + '_iter0', scene_name, video_name, 'f_hat')
build_iter0_symlinks(frames_dir, iter0_hat_dir)

# 设置推理所需路径
args.in_file = frames_dir
_main_raft_module.video_index = video_name

# ── Step 3：执行多轮稳定化推理 ────────────────────────────────────────────────
print(f"\n[3/3] 开始稳定化推理（共 {args.n_iter} 轮迭代）...\n")
main(args, config)

# ── 输出结果路径 ───────────────────────────────────────────────────────────────
# 注意：main() 执行后 args.outroot 会被修改为最后一轮的输出目录，
# 因此这里使用 args.outroot_origin 来构造正确的结果路径
result_dir = os.path.join(
    args.outroot_origin, f'{model_name}_iter{args.n_iter}', scene_name, video_name)
print(f"\n{'='*60}")
print(f"完成！最终稳定化视频：")
print(f"  {result_dir}/f_hat_30.avi  (30fps)")
print(f"  {result_dir}/f_hat_50.avi  (50fps)")
print(f"评估日志：{result_dir}/log.txt")
print(f"{'='*60}")
