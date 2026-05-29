# 视频稳像系统

基于深度学习的全帧视频稳定化方法，综合使用以下技术：

- **RAFT**：光流估计（Recurrent All-Pairs Field Transforms）
- **InpaintingModel**：GAN 光流修复（补全遮挡/运动边界处的不可信光流）
- **DIFNet**：深度迭代帧插值（Deep Iterative Frame Interpolation）

多轮迭代推理，逐步提升稳定性评分（SS），同时保持原始画幅（CR≈1）和低畸变（DV≈1）。

---

## 目录结构

```
.
├── demo/                      # 输入视频
│   └── demo.avi
├── demo_frames/               # 解帧后的 PNG 序列（运行后自动生成）
│   └── demo/
├── demo_output/               # 稳定化输出（运行后自动生成）
│   ├── ws_wg_iter1/demo/demo/
│   ├── ws_wg_iter2/demo/demo/
│   │   ├── f_hat/             # 稳定化帧序列
│   │   ├── f_hat_30.avi       # 30fps 输出视频
│   │   ├── f_hat_50.avi       # 50fps 输出视频
│   │   ├── comp_forward/      # 前向光流可视化
│   │   ├── comp_backward/     # 后向光流可视化
│   │   ├── edge_forward/      # 前向边缘图
│   │   ├── edge_backward/     # 后向边缘图
│   │   └── log.txt            # 质量评估指标
│   └── ...
├── checkpoints/ws_wg/         # 光流修复模型权重
├── weight/                    # 各网络预训练权重
│   ├── raft-things.pth        # RAFT 光流网络
│   ├── DIFNet2.pth            # 帧插值网络
│   ├── imagenet_deepfill.pth  # DeepFill 图像修复网络
│   └── edge_completion.pth    # 边缘补全网络
├── RAFT/                      # RAFT 光流网络模块
├── DIFRINT/                   # DIFNet 帧插值网络模块
├── src/                       # 光流修复模型（GAN）
├── utils/                     # 工具函数
├── networks.py                # 边缘生成网络定义
├── flow_completion.py         # 光流估计与修复核心逻辑
├── main_raft.py               # 多轮迭代稳定化推理
├── run_demo.py                # Python 推理入口脚本
└── run_demo.sh                # Bash 一键运行脚本
```

---

## 环境配置

### 1. 前置要求

- NVIDIA GPU（已在 RTX 4090 + CUDA 12.1 上验证）
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda

### 2. 创建 conda 环境

```bash
conda create -n videostab python=3.10 -y
conda activate videostab
```

### 3. 安装 PyTorch（CUDA 12.1）

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

> 如使用其他 CUDA 版本，请到 [pytorch.org](https://pytorch.org) 选取对应版本。

### 4. 安装其他依赖

```bash
pip install -r requirements.txt
```

> `cupy` 版本需与 CUDA 对应，如需修改请编辑 `requirements.txt` 中的 `cupy-cuda12x` 一行：
> - CUDA 11.x → `cupy-cuda11x`
> - CUDA 12.x → `cupy-cuda12x`

### 5. 验证安装

```bash
python -c "
import torch, cv2, cupy
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('OpenCV:', cv2.__version__)
print('CuPy:', cupy.__version__)
"
```

---

## 快速开始（Quick Start）

以 `demo/demo.avi` 为例，执行以下步骤即可完成视频稳定化：

### 方式一：Bash 脚本（推荐）

```bash
bash run_demo.sh
```

脚本会自动激活 conda 环境 `videostab`，无需手动 `conda activate`。

也可以传入参数：

```bash
bash run_demo.sh --n_iter 3 --skip 2
bash run_demo.sh --video myvideo --n_iter 2
```

### 方式二：直接运行 Python 脚本

```bash
conda activate videostab   # 或你的环境名
cd /path/to/github
python run_demo.py
```

运行过程输出示例：

```
============================================================
视频稳定化推理  |  GPU: NVIDIA GeForce RTX 4090
视频: /path/to/github/demo/demo.avi
迭代次数: 2  参考帧间隔: 4
============================================================

[1/3] 帧目录已存在，共 101 帧，跳过解帧

[2/3] 准备迭代初始目录...

[3/3] 开始稳定化推理（共 2 轮迭代）...

Iter: 1
Frame: 1/98 ... Frame: 98/98
Making video...  Computing metrics...

***Cropping ratio (Avg, Min): 1.0000 | 0.9878
***Distortion value: 0.9893
***Stability Score (Avg, Trans, Rot): 0.9302 | 0.8802 | 0.9803

Iter: 2  ...

============================================================
完成！最终稳定化视频：
  demo_output/ws_wg_iter2/demo/demo/f_hat_30.avi  (30fps)
  demo_output/ws_wg_iter2/demo/demo/f_hat_50.avi  (50fps)
评估日志：demo_output/ws_wg_iter2/demo/demo/log.txt
============================================================
```

### 查看结果

| 文件 | 说明 |
|------|------|
| `demo_output/ws_wg_iter2/demo/demo/f_hat_30.avi` | 最终稳定化视频（30fps） |
| `demo_output/ws_wg_iter2/demo/demo/f_hat_50.avi` | 最终稳定化视频（50fps） |
| `demo_output/ws_wg_iter2/demo/demo/log.txt` | CR / DV / SS 指标 |
| `demo_output/ws_wg_iter{1,2}/` | 各轮迭代的中间结果 |

---

## 运行参数说明

```bash
python run_demo.py [--video VIDEO] [--n_iter N] [--skip K]
# 或
bash run_demo.sh   [--video VIDEO] [--n_iter N] [--skip K]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | `demo` | `demo/` 目录下的视频名（不含扩展名） |
| `--n_iter` | `2` | 稳定化迭代轮数。越多稳定效果越好，但耗时线性增长；通常 2～3 轮即可收敛 |
| `--skip` | `2` | 参考帧间隔。每帧以其前/后第 `skip` 帧作为 DIFNet 插值的参考帧。数值越大稳定效果更强，但视频首尾各有 `skip` 帧无法处理 |

**示例：**

```bash
# 默认设置（2轮，skip=2）
python run_demo.py

# 高质量模式：3轮迭代
python run_demo.py --n_iter 3

# 快速测试：1轮
python run_demo.py --n_iter 1

# 处理自定义视频（需先将视频放入 demo/ 目录）
python run_demo.py --video myvideo --n_iter 2
```

---

## 质量评估指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| **CR**（Cropping Ratio） | 稳定化后有效画幅比例，越接近 1 代表裁剪越少 | → 1.0 |
| **DV**（Distortion Value） | 几何畸变程度，越接近 1 代表形变越小 | → 1.0 |
| **SS**（Stability Score） | 运动轨迹低频能量占比，越高代表画面越稳定 | → 1.0 |

**demo/demo.avi 实测结果（1280×720，101帧，skip=2）：**

| 迭代 | CR (Avg/Min) | DV | SS (Avg/Trans/Rot) |
|------|-------------|-----|-------------------|
| Iter 1 | 1.0000 / 0.9878 | 0.9893 | 0.9302 / 0.8802 / 0.9803 |
| Iter 2 | 1.0000 / 0.9874 | 0.9849 | 0.9172 / 0.8426 / 0.9918 |
| Iter 3 | 1.0000 / 0.9837 | 0.9774 | 0.9191 / 0.8575 / 0.9807 |

---

## 模型权重说明

所有权重文件位于 `weight/` 目录：

| 文件 | 对应模型 | 用途 |
|------|---------|------|
| `raft-things.pth` | RAFT | 光流估计 |
| `DIFNet2.pth` | DIFNet2 | 帧插值与融合 |
| `imagenet_deepfill.pth` | DeepFill | 图像级 inpainting（备用） |
| `edge_completion.pth` | EdgeGenerator | 边缘补全（备用） |

光流修复模型（InpaintingModel）权重位于 `checkpoints/ws_wg/`：

| 文件 | 说明 |
|------|------|
| `ws_wg_gen.pth` | 生成器权重 |
| `ws_wg_dis.pth` | 判别器权重 |
| `config_train.yml` | 模型训练配置 |

---

## 常见问题

**Q：运行时报 `No CUDA GPUs are available`**

检查 `main_raft.py` 第 9 行的 `CUDA_VISIBLE_DEVICES` 设置，确保 GPU 编号正确（默认为 `'0'`）。

**Q：`cupy` 安装失败**

确认 CUDA 版本后安装对应包：
```bash
nvidia-smi  # 查看 CUDA 版本
pip install cupy-cuda11x  # CUDA 11.x
pip install cupy-cuda12x  # CUDA 12.x
```

**Q：输出视频抖动仍然明显**

尝试增加迭代次数：`--n_iter 3`，或调小跳帧数：`--skip 2`。
