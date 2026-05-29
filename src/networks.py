"""
src/networks.py — 光流修复网络结构定义（InpaintingModel 使用）

与根目录 networks.py 的区别：
  InpaintGenerator 采用双编码器设计，分别编码光流和边缘/mask 信息，
  特征相加后经残差中间层和解码器输出修复后的光流。

包含以下网络：
  - BaseNetwork       : 所有网络的基类，提供权重初始化方法
  - InpaintGenerator  : 光流修复生成器（双编码器：flow_encoder + em_encoder）
  - EdgeGenerator     : 边缘补全生成器（encoder-residual-decoder，输入 3 通道，输出 1 通道边缘图）
  - Discriminator     : PatchGAN 判别器（5 层卷积，输出逐块真假概率）
  - ResnetBlock       : 残差卷积块（带膨胀卷积，用于 middle 特征变换）
  - spectral_norm     : 谱归一化包装函数
"""

import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    """所有网络的基类，提供统一的权重初始化接口。"""

    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        """
        初始化网络权重。
        init_type: 初始化方式，支持 normal | xavier | kaiming | orthogonal
        gain:      初始化缩放系数
        参考：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
        """

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class InpaintGenerator(BaseNetwork):
    """
    光流修复生成器（双编码器 + 残差中间层 + 解码器）。

    结构：
      flow_encoder : 编码原始光流（2 通道），2 → 64 → 128 → 256
      em_encoder   : 编码边缘图和 mask 的拼接（2 通道），2 → 64 → 128 → 256
      middle       : 8 个 ResnetBlock，对 flow_features + em_features 做特征融合
      decoder      : 256 → 128 → 64 → 2，反卷积上采样，输出修复后的光流

    输入：
      flow  : (B, 2, H, W) 原始光流（含遮挡区域）
      edge  : (B, 1, H, W) 边缘图
      mask  : (B, 1, H, W) 遮挡 mask（1 表示需修复区域）
    输出：
      output : (B, 2, H, W) 修复后的光流（经 tanh 归一化到 [-1, 1]）
    """

    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        # 光流编码器：提取光流的空间特征
        self.flow_encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(False),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(False),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(False)
        )

        # 边缘/mask 编码器：编码遮挡位置的结构信息（edge + mask 拼接为 2 通道）
        self.em_encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(False),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(False),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(False)
        )

        # 中间层：两路特征相加后通过残差块进行深层融合
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        # 解码器：上采样还原分辨率，输出 2 通道修复光流
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(False),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(False),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, flow, edge, mask):
        # 分别编码光流特征和边缘/mask 特征
        flow_features = self.flow_encoder(flow)
        em_features = self.em_encoder(torch.cat([edge, mask], dim=1).float())
        # 两路特征逐元素相加后送入残差中间层
        x1 = self.middle(flow_features + em_features)
        x2 = self.decoder(x1)
        # tanh 将输出归一化到 [-1, 1]
        output = torch.tanh(x2)
        return output


class EdgeGenerator(BaseNetwork):
    """
    边缘补全生成器（encoder-residual-decoder 架构，带谱归一化）。

    结构：
      encoder : 3 → 64 → 128 → 256，步长卷积下采样 2 次
      middle  : 8 个 ResnetBlock（膨胀率 2）
      decoder : 256 → 128 → 64 → 1，反卷积上采样 2 次，输出 1 通道边缘图
    输入：3 通道（灰度图 + 边缘图 + mask 拼接）
    输出：1 通道预测边缘图（经 sigmoid 归一化到 [0, 1]）
    """

    def __init__(self, residual_blocks=8, use_spectral_norm=True, init_weights=True):
        super(EdgeGenerator, self).__init__()

        # 编码器：带谱归一化的下采样特征提取
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(False),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(False),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(False)
        )

        # 中间层：残差块特征变换
        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)
        self.middle = nn.Sequential(*blocks)

        # 解码器：上采样还原分辨率，输出 1 通道边缘预测
        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(BaseNetwork):
    """
    PatchGAN 判别器（5 层卷积，逐块判断真假）。

    结构：conv1~conv4 逐步下采样（步长 2/1），conv5 输出逐块概率图
    输入：任意通道数（由 in_channels 指定）
    输出：
      outputs      : 逐块真假概率（use_sigmoid=True 时经 sigmoid 输出）
      feature_maps : [conv1, conv2, conv3, conv4, conv5] 各层特征（用于特征匹配损失）
    """

    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        # conv1: 步长 2 下采样，输出 64 通道
        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        # conv2: 步长 2 下采样，输出 128 通道
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        # conv3: 步长 2 下采样，输出 256 通道
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        # conv4: 步长 1，输出 512 通道
        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        # conv5: 步长 1，输出 1 通道逐块概率图
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm)
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)
        else:
            outputs = conv5

        # 返回预测结果和各层特征图（用于特征匹配损失）
        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    """
    残差卷积块（ResNet Block）。

    包含两个卷积层：第一个使用膨胀卷积扩大感受野，第二个使用标准卷积。
    输出为输入与卷积结果的逐元素相加（残差连接），末尾不加 ReLU。
    参考：http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(False),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        # 残差连接：输入直接加到卷积输出
        out = x + self.conv_block(x)
        return out


def spectral_norm(module, mode=True):
    """对卷积层应用谱归一化（Spectral Normalization），用于稳定 GAN 训练。"""
    if mode:
        # return nn.utils.parametrizations.spectral_norm(module)
        return nn.utils.spectral_norm(module)
    return module
