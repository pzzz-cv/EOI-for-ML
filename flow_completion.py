import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import cv2
import glob
import numpy as np
import torch
import copy
from PIL import Image
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
import torchvision.transforms.functional as F
import argparse
import math

from RAFT import raft_utils
from RAFT import RAFT
from thop import profile,clever_format

from utils.edge_screen import edge_screen
from utils.Poisson_blend import Poisson_blend
from utils.Poisson_blend_img import Poisson_blend_img

from networks import EdgeGenerator
try:
    from gmflow.evaluate import inference_on_dir
except ImportError:
    inference_on_dir = None

# from get_flowNN import get_flowNN
# from get_flowNN_gradient import get_flowNN_gradient
# from utils.common_utils import flow_edge
# from spatial_inpaint import spatial_inpaint


def create_dir(dir):
    """如果目录不存在则创建。"""
    if not os.path.exists(dir):
        os.makedirs(dir)

def gradient_mask(mask):
    """
    将掩码膨胀一个像素（向下和向右扩展），
    用于在计算流梯度时包含掩码边界像素。
    """
    gradient_mask = np.logical_or.reduce((mask,
        np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=bool)), axis=0),
        np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=bool)), axis=1)))

    return gradient_mask

def initialize_RAFT(args):
    """
    初始化 RAFT 光流模型并加载预训练权重。

    参数：
        args  包含 model（权重路径）等字段的命名空间

    返回：
        加载好权重、设置为 eval 模式的 RAFT 模型
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda')
    model.eval()

    return model

def homograpy(image1, image2):
    """
    用 ORB 特征点估计从 image2 到 image1 的单应矩阵，
    并将 image2 配准（warp）到 image1 坐标系下。

    返回：
        image2_registered  配准后的 image2（numpy uint8）
        H                  3x3 单应矩阵（float32）
    """
    image1 = np.array(image1[0].permute(1, 2, 0).cpu(), dtype=np.uint8)
    image2 = np.array(image2[0].permute(1, 2, 0).cpu(), dtype=np.uint8)

    imgH, imgW, _ = image1.shape

    src_pts, dst_pts = detect_points(image1, image2)

    H, mask = cv2.findHomography(dst_pts, src_pts,  cv2.RANSAC, 5.0)
    image2_registered = cv2.warpPerspective(image2, H, (imgW, imgH))

    return image2_registered, H.astype(np.float32)


def detect_points(im1, im2):
    """
    使用 ORB 检测特征点并通过暴力匹配找到两图像间的对应点对。

    参数：
        im1, im2  输入图像（BGR uint8）

    返回：
        points1, points2  对应特征点坐标数组（各 Nx2，float32），
                          若任意一张图像无特征点则返回 (None, None)
    """
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(2500)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return None, None

    # 暴力匹配，保留匹配距离最小的前 15%
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = list(matcher.match(descriptors1, descriptors2, None))
    matches.sort(key=lambda x: x.distance, reverse=False)

    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    return points1, points2

def to_tensor(img):
    """将 numpy 图像转换为归一化的 float32 张量（C x H x W）。"""
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t

def infer_flow(args, mode, filename, image1, image2, imgH, imgW, model, saving=False, homography=False):
    """
    推理两帧之间的光流。

    支持两种模式：
    - homography=False：直接用 RAFT / PWC 估计光流
    - homography=True ：先做单应配准，再估计光流，
                        然后将光流变换回原始坐标系，
                        以减少大位移时的估计误差

    参数：
        args        推理参数
        mode        'forward' 或 'backward'
        filename    帧序号字符串（用于保存文件名）
        image1/2    输入帧张量 [1,3,H,W]，像素值范围 [0,255]
        imgH/W      图像高宽
        model       已加载的光流模型
        saving      是否将光流可视化结果保存到磁盘
        homography  是否启用单应配准辅助

    返回：
        flow        光流张量（RAFT 输出格式）或 numpy 数组（单应模式）
    """

    if not homography:
        if args.flowmodel=='RAFT':
            _, flow = model(image1, image2, iters=20, test_mode=True)
        elif args.flowmodel=='PWC':
            temp_w = int(math.floor(math.ceil(image1.size(3) / 64.0) * 64.0)) # Due to Pyramid method?
            temp_h = int(math.floor(math.ceil(image1.size(2) / 64.0) * 64.0))

            temp_fr_1 = torch.nn.functional.interpolate(input=image1, size=(temp_h, temp_w), mode='nearest')
            temp_fr_2 = torch.nn.functional.interpolate(input=image2, size=(temp_h, temp_w), mode='nearest')

            flow = 20.0 * torch.nn.functional.interpolate(input=model(temp_fr_1/255, temp_fr_2/255), size=(image1.size(2), image1.size(3)), mode='bilinear', align_corners=False)
    else:
        image2_reg, H_BA = homograpy(image1, image2)
        image2_reg = torch.tensor(image2_reg).permute(2, 0, 1)[None].float().to('cuda')
        if args.flowmodel=='RAFT':
            _, flow = model(image1, image2, iters=20, test_mode=True)
        elif args.flowmodel=='PWC':
            flow = 20.0 * torch.nn.functional.interpolate(input=model(image1/255, image2/255), size=(image1.size(2), image1.size(3)), mode='bilinear', align_corners=False)
        flow = flow[0].permute(1, 2, 0).cpu().numpy()

        (fy, fx) = np.mgrid[0 : imgH, 0 : imgW].astype(np.float32)

        fxx = copy.deepcopy(fx) + flow[:, :, 0]
        fyy = copy.deepcopy(fy) + flow[:, :, 1]

        (fxxx, fyyy, fz) = np.linalg.inv(H_BA).dot(np.concatenate((fxx.reshape(1, -1),
                                                   fyy.reshape(1, -1),
                                                   np.ones_like(fyy).reshape(1, -1)), axis=0))
        fxxx, fyyy = fxxx / fz, fyyy / fz

        flow = np.concatenate((fxxx.reshape(imgH, imgW, 1) - fx.reshape(imgH, imgW, 1),
                               fyyy.reshape(imgH, imgW, 1) - fy.reshape(imgH, imgW, 1)), axis=2)

    if saving:
        flow_save = flow[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(raft_utils.flow_viz.flow_to_image(flow_save)).save(os.path.join(args.outroot, 'flow', mode + '_png', filename + '.png'))
        raft_utils.frame_utils.writeFlow(os.path.join(args.outroot, 'flow', mode + '_flo', filename + '.flo'), flow_save)

    return flow


def complete_flow(args, corrFlow, flow_mask, edge, index, mode):
    """
    使用泊松融合（Poisson blending）补全光流中不可信区域。

    在前后向光流一致性检验后，不一致区域（flow_mask=True）的光流值
    不可信，此函数利用边缘图引导的泊松融合对这些区域进行补全。

    参数：
        args        推理参数
        corrFlow    待补全的光流场 [H x W x 2]（numpy float32）
        flow_mask   不可信区域掩码 [H x W]（bool）
        edge        边缘图，用于引导泊松融合（可为 None）
        index       当前帧序号
        mode        'forward' 或 'backward'

    返回：
        compFlow    补全后的光流场 [H x W x 2]
    """
    if mode not in ['forward', 'backward', 'nonlocal_forward', 'nonlocal_backward']:
        raise NotImplementedError

    sh = corrFlow.shape
    imgH = sh[0]
    imgW = sh[1]

    # create_dir(os.path.join(args.outroot, 'flow_comp', mode + '_flo'))
    # create_dir(os.path.join(args.outroot, 'flow_comp', mode + '_png'))

    flow = corrFlow
    if edge is not None:
        # imgH x (imgW - 1 + 1) x 2
        gradient_x = np.concatenate((np.diff(flow, axis=1), np.zeros((imgH, 1, 2), dtype=np.float32)), axis=1)
        # (imgH - 1 + 1) x imgW x 2
        gradient_y = np.concatenate((np.diff(flow, axis=0), np.zeros((1, imgW, 2), dtype=np.float32)), axis=0)

        # concatenate gradient_x and gradient_y
        gradient = np.concatenate((gradient_x, gradient_y), axis=2)

        # We can trust the gradient outside of flow_mask_gradient_img
        # We assume the gradient within flow_mask_gradient_img is 0.
        flow_mask_gradient_img = gradient_mask(flow_mask)
        gradient[flow_mask_gradient_img, :] = 0

        # Complete the flow
        imgSrc_gy = gradient[:, :, 2 : 4]
        imgSrc_gy = imgSrc_gy[0 : imgH - 1, :, :]
        imgSrc_gx = gradient[:, :, 0 : 2]
        imgSrc_gx = imgSrc_gx[:, 0 : imgW - 1, :]


        
        gr_img_gray = (imgSrc_gx[:, :, 0] ** 2 + imgSrc_gx[:, :, 1] ** 2) ** 0.5
        gr_img_gray = gr_img_gray / gr_img_gray.max()
        flow_img = Image.fromarray(np.uint8(np.tile(gr_img_gray[..., None], (1,1,3))*255))
        
        compFlow = Poisson_blend(flow, imgSrc_gx, imgSrc_gy, flow_mask, edge)

    # Flow visualization.
    flow_img = raft_utils.flow_viz.flow_to_image(compFlow)
    flow_img = Image.fromarray(flow_img)
    flow_img.save('test.png')
    
    # Saves the flow and flow_img.
    # flow_img.save(os.path.join(args.outroot, 'flow_comp', mode + '_png', '%05d.png'%(index)))
    # raft_utils.frame_utils.writeFlow(os.path.join(args.outroot, 'flow_comp', mode + '_flo', '%05d.flo'%(index)), compFlow)

    return compFlow


def calculate_flow(args, model, fr1, fr2, i, saving=True, homography=True):
    """
    计算两帧之间的前向和后向光流，并创建保存目录。

    参数：
        args        推理参数（含 flowmodel、outroot 等）
        model       光流模型（RAFT / PWC / gmflow）
        fr1, fr2    相邻两帧张量 [1,3,H,W]
        i           当前帧序号（用于文件命名）
        saving      是否将光流保存为 .png 和 .flo 文件
        homography  是否启用单应配准辅助估计

    返回：
        flowF  前向光流（fr1 -> fr2）
        flowB  后向光流（fr2 -> fr1）
    """
    mode_list = ['forward', 'backward']

    _, _, imgH, imgW = fr1.shape

    create_dir(os.path.join(args.outroot, 'flow'))
    create_dir(os.path.join(args.outroot, 'flow', 'forward' + '_png'))
    create_dir(os.path.join(args.outroot, 'flow', 'backward' + '_png'))
    create_dir(os.path.join(args.outroot, 'flow', 'forward' + '_flo'))
    create_dir(os.path.join(args.outroot, 'flow', 'backward' + '_flo'))

    with torch.no_grad():
        if args.flowmodel == 'gmflow':
            flowF, flowB = inference_on_dir(model,
                    fr1, 
                    fr2,
                    output_path = 'test',
                    padding_factor=args.padding_factor,
                    inference_size=args.inference_size,
                    paired_data=args.dir_paired_data,
                    save_flo_flow=args.save_flo_flow,
                    attn_splits_list=args.attn_splits_list,
                    corr_radius_list=args.corr_radius_list,
                    prop_radius_list=args.prop_radius_list,
                    pred_bidir_flow=args.pred_bidir_flow,
                    fwd_bwd_consistency_check=args.fwd_bwd_consistency_check,
                    )
        else:
            flowF = infer_flow(args, 'forward', '%05d'%i, fr1, fr2, imgH, imgW, model, saving, homography=homography)
            flowB = infer_flow(args, 'backward', '%05d'%i, fr2, fr1, imgH, imgW, model, saving, homography=homography)

    return flowF, flowB

def infer(EdgeGenerator, flow_img_gray, edge, mask):
    """
    使用边缘补全网络（EdgeGenerator）对掩码区域内的边缘进行补全。

    参数：
        EdgeGenerator   预训练的边缘生成网络
        flow_img_gray   光流幅值的灰度图（numpy uint8，H x W）
        edge            Canny 边缘图（numpy float，H x W，值域 [0,1]）
        mask            不可信区域掩码（numpy bool，H x W）

    返回：
        edge_completed  补全后的二值边缘图（0 或 1，H x W）
    """

    # Add a pytorch dataloader
    flow_img_gray_tensor = to_tensor(flow_img_gray)[None, :, :].float().cuda()
    edge_tensor = to_tensor(edge)[None, :, :].float().cuda()
    mask_tensor = torch.from_numpy(mask.astype(np.float64))[None, None, :, :].float().cuda()

    # Complete the edges
    edges_masked = (edge_tensor * (1 - mask_tensor))
    images_masked = (flow_img_gray_tensor * (1 - mask_tensor)) + mask_tensor
    inputs = torch.cat((images_masked, edges_masked, mask_tensor), dim=1)
    with torch.no_grad():
        edges_completed = EdgeGenerator(inputs) # in: [grayscale(1) + edge(1) + mask(1)]
    edges_completed = edges_completed * mask_tensor + edge_tensor * (1 - mask_tensor)
    edge_completed = edges_completed[0, 0].data.cpu().numpy()
    edge_completed[edge_completed < 0.9] = 0
    edge_completed[edge_completed >= 0.9] = 1

    return edge_completed



def edge(img):
    """
    从 RGB 帧中提取梯度边缘图。

    将输入帧转为灰度图后，计算 x/y 方向梯度，
    合成梯度幅值并归一化到 [0, 1]。

    参数：
        img  输入帧张量 [1,3,H,W]，值域 [0,255]

    返回：
        edge  归一化边缘强度图（numpy float64，H x W）
    """
    img_gray = rgb2gray(np.array(img.squeeze().permute(1,2,0).cpu()))
    
    H, W = img_gray.shape

    img_gradient_x, img_gradient_y = np.gradient(img_gray)
    # (imgH - 1 + 1) x imgW x 2
    

    img_gradient = np.sqrt(np.power(img_gradient_x, 2) + np.power(img_gradient_y, 2))

    edge = img_gradient/np.max(img_gradient)
    return edge

@torch.no_grad()
def stableflow(args, config, flowmodel, frameF, frameB, flowF, flowB, maskF, maskB, index):
    """
    对不可信光流区域进行 GAN 修复（光流补全核心函数）。

    流程：
    1. 若设置了固定 INPUT_SIZE，将帧和光流双线性缩放至该尺寸
    2. 分别以对侧帧为参考，提取边缘图（梯度边缘）
    3. 用 InpaintingModel (GAN) 对掩码区域的前向/后向光流进行补全
    4. 将补全结果 resize 回原始分辨率
    5. 在补全区域用修复结果替换原光流（其余区域保持原值）

    参数：
        args        推理参数
        config      模型配置（含 INPUT_SIZE、DEVICE 等）
        flowmodel   已加载的光流修复网络（InpaintingModel）
        frameF      前帧张量 [1,3,H,W]
        frameB      后帧张量 [1,3,H,W]
        flowF       前向原始光流 [1,2,H,W]
        flowB       后向原始光流 [1,2,H,W]
        maskF       前向不可信掩码 [1,1,H,W]（1=需修复）
        maskB       后向不可信掩码 [1,1,H,W]
        index       当前帧序号（用于保存调试图像）

    返回：
        corrFlowF   修复后的前向光流 [1,2,H,W]
        corrFlowB   修复后的后向光流 [1,2,H,W]
    """
    #     mask_img = np.array(Image.open(filename).convert('L').resize((imgW, imgH)))
    #     mask.append(mask_img)

    #     # Dilate 15 pixel so that all known pixel is trustworthy
    #     flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=15)
    #     # Close the small holes inside the foreground objects
    #     flow_mask_img = cv2.morphologyEx(flow_mask_img.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
    #     flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(bool)
    #     flow_mask.append(flow_mask_img)

    #     img = np.array(Image.open(frame_list[index]).convert('L').resize((imgW, imgH)))
    #     images.append(img)
    #     index += 1

    # if args.edge_guide:
    #     # Edge completion model.
    #     edgegenerator = EdgeGenerator()
    #     EdgeComp_ckpt = torch.load(args.edge_completion_model)
    #     edgegenerator.load_state_dict(EdgeComp_ckpt['generator'])
    #     edgegenerator.cuda()
    #     edgegenerator.eval()


    #     flows, edges, masks = cuda(*items)

    #     inputs = (flows * (1 - masks)) #+ masks
    #     outputs = model(flows, edges, masks)
    #     outputs_merged = (outputs * masks) + (flows * (1 - masks))

    #     # Edge completion.
    #     FlowF_edge = edge_completion(args, frB,  edgegenerator, FlowF, maskF, index, 'forward')
    #     FlowB_edge = edge_completion(args, frF,  edgegenerator, FlowB, maskB, index, 'backward')
    #     # print('\nFinish edge completion.')

    #     # corrFlowF = complete_flow(args, FlowF, maskF,  FlowF_edge, index, 'forward')
    #     # corrFlowB = complete_flow(args, FlowB, maskB,  FlowB_edge, index, 'backward')
    #     corrFlowF = torch.zeros([2,360,640])
    #     corrFlowB = torch.zeros([2,360,640])

    # flow_img_grayF = (flowF[0, 0, :, :] ** 2 + flowF[0, 1, :, :] ** 2) ** 0.5
    # flow_img_grayF = (flow_img_grayF / flow_img_grayF.max()).cpu().numpy()
    # EdgeF = canny(flow_img_grayF, sigma= config.SIGMA, mask=(1 - maskF.squeeze()).cpu().numpy()>0.999)
    c, b, w, h = frameF.size()

    if config.INPUT_SIZE > 0:
        input_size = config.INPUT_SIZE
        frF = torch.nn.functional.interpolate(frameF, size=(input_size, input_size), mode='bilinear')
        frB = torch.nn.functional.interpolate(frameB, size=(input_size, input_size), mode='bilinear')
        mkF = torch.nn.functional.interpolate(maskF, size=(input_size, input_size), mode='bilinear')
        mkB = torch.nn.functional.interpolate(maskB, size=(input_size, input_size), mode='bilinear')
        flF = torch.nn.functional.interpolate(flowF, size=(input_size, input_size), mode='bilinear')/255.0
        flB = torch.nn.functional.interpolate(flowB, size=(input_size, input_size), mode='bilinear')/255.0

    EdgeF = edge(frB)
    edgeF = torch.tensor(EdgeF).cuda().unsqueeze(0).unsqueeze(0).float()

    outputF = flowmodel.forward(flF, edgeF, mkF)

    # flow_img_grayB = (flowB[0, 0, :, :] ** 2 + flowB[0, 1, :, :] ** 2) ** 0.5
    # flow_img_grayB = (flow_img_grayB / flow_img_grayB.max()).cpu().numpy()
    # EdgeB = canny(flow_img_grayB, sigma= config.SIGMA, mask=(1 - maskB.squeeze()).cpu().numpy()>0.999)
    EdgeB = edge(frF)
    edgeB = torch.tensor(EdgeB).cuda().unsqueeze(0).unsqueeze(0).float()
    

    # macs, params = profile(flowmodel, inputs=(flB, edgeB, mkB))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(macs, params)
    outputB = flowmodel.forward(flB, edgeB, mkB)
    
    if config.INPUT_SIZE > 0:
        EdgeF = cv2.resize(EdgeF, (h, w))
        EdgeB = cv2.resize(EdgeB, (h, w))
        outputF = torch.nn.functional.interpolate(255*outputF, size=(w, h), mode='bilinear')
        outputB = torch.nn.functional.interpolate(255*outputB, size=(w, h), mode='bilinear')

    os.makedirs(args.out_file+'/edge_forward', exist_ok=True)
    os.makedirs(args.out_file+'/edge_backward', exist_ok = True)
    Image.fromarray(255-np.uint8(np.concatenate((EdgeF[..., None],EdgeF[..., None],EdgeF[..., None]), axis= 2)*255)).save(args.out_file+'/edge_forward/%05d.png'%(index))
    Image.fromarray(255-np.uint8(np.concatenate((EdgeB[..., None],EdgeB[..., None],EdgeB[..., None]), axis= 2)*255)).save(args.out_file+'/edge_backward/%05d.png'%(index))

    outputs_mergedF = (outputF * maskF) + (flowF * (1 - maskF))
    outputs_mergedB = (outputB * maskB) + (flowB * (1 - maskB))

    corrFlowF = outputs_mergedF
    corrFlowB = outputs_mergedB

    return corrFlowF, corrFlowB

def consistCheck(flowF, flowB, consistencyThres):
    """
    计算前后向光流一致性，返回不一致像素的掩码。

    原理：将后向光流 flowB 的终点再经前向光流 flowF warp 回来，
    若偏差超过阈值则认为该点的光流不可信（遮挡/运动边界等区域）。

    参数：
        flowF           前向光流 [H x W x 2]（numpy）
        flowB           后向光流 [H x W x 2]（numpy）
        consistencyThres 一致性误差像素阈值

    返回：
        IsConsist  不一致区域掩码 [H x W]（bool），True 表示不可信
    """
    # |       y            |  |       v            |
    # |   x   *            |  |   u   *            |
    # |                    |  |                    |
    # |--------------------|  |--------------------|

    # sub: numPix * [y x t]

    imgH, imgW, _ = flowF.shape

    (fy, fx) = np.mgrid[0 : imgH, 0 : imgW].astype(np.float32)
    fxx = fx + flowB[:, :, 0]  # horizontal
    fyy = fy + flowB[:, :, 1]  # vertical

    u = (fxx + cv2.remap(flowF[:, :, 0], fxx, fyy, cv2.INTER_LINEAR) - fx)
    v = (fyy + cv2.remap(flowF[:, :, 1], fxx, fyy, cv2.INTER_LINEAR) - fy)

    BFdiff = (u ** 2 + v ** 2) ** 0.5
    IsConsist = BFdiff > consistencyThres

    return IsConsist#, np.stack((u, v), axis=2) 

def confident_field(FlowF, FlowB, consistencyThres):
    """
    基于前后向一致性检验，生成前向和后向光流的不可信区域掩码。

    参数：
        FlowF           前向光流 [H x W x 2]
        FlowB           后向光流 [H x W x 2]
        consistencyThres 一致性误差阈值（像素）

    返回：
        maskF  前向光流不可信掩码（uint8，0/1）
        maskB  后向光流不可信掩码（uint8，0/1）
    """
    inconsistF = consistCheck(FlowB, FlowF, consistencyThres)
    maskF = inconsistF
    # MaskedFlowF = FlowF(inconsistF)

    inconsistB = consistCheck(FlowF, FlowB, consistencyThres)
    maskB = inconsistB
    # MaskedFlowB = FlowB(inconsistB)


    return maskF.astype(np.uint8), maskB.astype(np.uint8)

def flow(args, config, flowmodel, inpaintingmodel, fr1, fr2, i, iteration=1):
    """
    完整的光流估计与修复流程（带边缘可视化保存）。

    流程：
    1. 用 RAFT/PWC 估计前向/后向原始光流
    2. 循环 iteration 次：
       a. 前后向一致性检验，生成不可信掩码
       b. 调用 stableflow 用 GAN 修复不可信区域
    3. 返回修复后的光流

    参数：
        args            推理参数
        config          模型配置
        flowmodel       光流估计网络（RAFT 等）
        inpaintingmodel 光流修复网络（InpaintingModel）
        fr1, fr2        相邻两帧张量 [1,3,H,W]
        i               当前帧序号
        iteration       光流修复迭代次数

    返回：
        corrFlowF  修复后的前向光流张量
        corrFlowB  修复后的后向光流张量
    """
    #w,h,2
    FlowF, FlowB = calculate_flow(args, flowmodel, fr1, fr2, i, saving=False, homography=args.homography)
    corrFlowF = FlowF
    corrFlowB = FlowB
    consistencyThres = args.consistencyThres
    for _ in range(iteration):    
        corrFlowF = corrFlowF[0].permute(1, 2, 0).cpu().numpy()
        corrFlowB = corrFlowB[0].permute(1, 2, 0).cpu().numpy()

        consistencyThres = consistencyThres*(_+1)
        MaskF, MaskB = confident_field(corrFlowF, corrFlowB, consistencyThres)



        # os.makedirs(args.out_file+'/mask_forward_iter%d'%_, exist_ok=True)
        # os.makedirs(args.out_file+'/mask_backward_iter%d'%_, exist_ok = True)
        # Image.fromarray(np.uint8(np.concatenate((MaskF[..., None],MaskF[..., None],MaskF[..., None]), axis= 2)*255)).save(args.out_file+'/mask_forward_iter%d/%05d.png'%(_,i))
        # Image.fromarray(np.uint8(np.concatenate((MaskB[..., None],MaskB[..., None],MaskB[..., None]), axis= 2)*255)).save(args.out_file+'/mask_backward_iter%d/%05d.png'%(_,i))

        maskF = torch.tensor(MaskF).unsqueeze(0).unsqueeze(0).cuda().float()
        maskB = torch.tensor(MaskB).unsqueeze(0).unsqueeze(0).cuda().float()
        corrFlowF = torch.tensor(corrFlowF).permute(2,0,1).unsqueeze(0).cuda()
        corrFlowB = torch.tensor(corrFlowB).permute(2,0,1).unsqueeze(0).cuda()
        
        corrFlowF, corrFlowB = stableflow(args, config, inpaintingmodel, fr1, fr2, corrFlowF, corrFlowB, maskF, maskB, index = i)
    # a = corrFlowF.cpu().squeeze().permute(1,2,0) - FlowF
    # b = corrFlowB.cpu().squeeze().permute(1,2,0) - FlowB
        # os.makedirs(args.out_file+'/comp_forward_iter%d'%_, exist_ok = True)
        # os.makedirs(args.out_file+'/comp_backward_iter%d'%_, exist_ok = True)
        # Image.fromarray(raft_utils.flow_viz.flow_to_image(np.array(corrFlowF.cpu().squeeze().permute(1, 2, 0)))).save(args.out_file+'/comp_forward_iter%d/%05d.png'%(_,i))
        # Image.fromarray(raft_utils.flow_viz.flow_to_image(np.array(corrFlowB.cpu().squeeze().permute(1, 2, 0)))).save(args.out_file+'/comp_backward_iter%d/%05d.png'%(_,i))
    return corrFlowF, corrFlowB

def flow_nosave(args, config, flowmodel, inpaintingmodel, fr1, fr2, i, iteration=1):
    """
    与 flow() 相同的光流估计与修复流程，但不保存边缘可视化图像。

    用于 main_raft.py 中计算插值帧到原始帧之间的中间光流，
    不需要保存调试可视化文件时调用此版本以节省 I/O。

    参数 / 返回值同 flow()。
    """
    #w,h,2
    FlowF, FlowB = calculate_flow(args, flowmodel, fr1, fr2, i, saving=False, homography=args.homography)
    corrFlowF = FlowF
    corrFlowB = FlowB
    consistencyThres = args.consistencyThres
    for _ in range(iteration):    
        corrFlowF = corrFlowF[0].permute(1, 2, 0).cpu().numpy()
        corrFlowB = corrFlowB[0].permute(1, 2, 0).cpu().numpy()

        consistencyThres = consistencyThres*(_+1)
        MaskF, MaskB = confident_field(corrFlowF, corrFlowB, consistencyThres)

        maskF = torch.tensor(MaskF).unsqueeze(0).unsqueeze(0).cuda().float()
        maskB = torch.tensor(MaskB).unsqueeze(0).unsqueeze(0).cuda().float()
        corrFlowF = torch.tensor(corrFlowF).permute(2,0,1).unsqueeze(0).cuda()
        corrFlowB = torch.tensor(corrFlowB).permute(2,0,1).unsqueeze(0).cuda()
        
        corrFlowF, corrFlowB = stableflow(args, config, inpaintingmodel, fr1, fr2, corrFlowF, corrFlowB, maskF, maskB, index = i)

    # a = corrFlowF.cpu().squeeze().permute(1,2,0) - FlowF
    # b = corrFlowB.cpu().squeeze().permute(1,2,0) - FlowB
    return corrFlowF, corrFlowB

def main(args):
    # Flow model.
    RAFT_model = initialize_RAFT(args)

    # Loads frames.
    filename_list = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))

    filename_list = sorted(filename_list)[:21]

    
    # Obtains imgH, imgW and nFrame.
    imgH, imgW = np.array(Image.open(filename_list[0])).shape[:2]
    nFrame = len(filename_list)

    video = []
    for filename in sorted(filename_list):
        video.append(torch.from_numpy(np.array(Image.open(filename)).astype(np.uint8)[..., :3]).permute(2, 0, 1).float())

    video = torch.stack(video, dim=0)
    video = video.cuda()

    # Calcutes the corrupted flow.

    for i in range(len(video)-1):
        fr1 = video[i, ...].unsqueeze(0)
        fr2 = video[i+1, ...].unsqueeze(0)
        FlowF, FlowB = calculate_flow(args, RAFT_model, fr1, fr2, i, homography= args.homography)
        maskF, maskB = confident_field(FlowF, FlowB, args.consistencyThres)
        corrFlowF, corrFlowB = stableflow(fr1, fr2, args, FlowF, FlowB, maskF, maskB, index = i)
        # Image.fromarray(raft_utils.flow_viz.flow_to_image(np.array(corrFlowF[i,...]))).save('testF.py')
        # Image.fromarray(raft_utils.flow_viz.flow_to_image(np.array(corrFlowB[i,...]))).save('testB.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # video completion
    parser.add_argument('--seamless', action='store_true', help='Whether operate in the gradient domain')
    parser.add_argument('--edge_guide', default='True', action='store_true', help='Whether use edge as guidance to complete flow')
    parser.add_argument('--mode', default='edge', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--path', default='./data/test/1', help="dataset for evaluation")
    parser.add_argument('--path_mask', default='./data/tennis_mask', help="mask for object removal")
    parser.add_argument('--outroot', default='./result/test_imageedge/', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=5.0, type=float, help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--Nonlocal', action='store_true', help='Whether use edge as guidance to complete flow')
    parser.add_argument('--homography', action='store_true', default=True, help='Whether use homography as guidance to compute flow')
    parser.add_argument('--device', action='store_true', default=[2])

    # RAFT
    parser.add_argument('--model', default='./weight/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # Deepfill
    parser.add_argument('--deepfill_model', default='./weight/imagenet_deepfill.pth', help="restore checkpoint")

    # Edge completion
    parser.add_argument('--edge_completion_model', default='./weight/edge_completion.pth', help="restore checkpoint")

    # extrapolation
    parser.add_argument('--H_scale', dest='H_scale', default=2, type=float, help='H extrapolation scale')
    parser.add_argument('--W_scale', dest='W_scale', default=2, type=float, help='W extrapolation scale')

    args = parser.parse_args()

    args.outroot = args.outroot + args.mode

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.device)
    main(args)