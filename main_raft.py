import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import cv2
import glob
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import copy
from PIL import Image
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
import torchvision.transforms.functional as F
import argparse

from RAFT import raft_utils
from RAFT import RAFT

import torch
import torch.nn as nn

import utils.region_fill as rf
from utils.Poisson_blend import Poisson_blend
from utils.Poisson_blend_img import Poisson_blend_img
from utils.common_utils import interp, BFconsistCheck, \
    FBconsistCheck, consistCheck, get_KeySourceFrame_flowNN
from utils.metrics import metrics
from utils.frame2vid import frame2vid
from itertools import islice

from shutil import copyfile

from DIFRINT.models.models import DIFNet2
from DIFRINT.models.pwcNet import PwcNet

from flow_completion import flow, flow_nosave
from src.models_gan import InpaintingModel
from src.config import Config
from src.utils_file import flow_viz



def main(args, config):
    """
    视频稳定化主推理函数。

    整体流程（循环 args.n_iter 轮）：
    ─────────────────────────────────────────────────────────────────
    每轮迭代对视频中每一帧执行：

    1. 读取当前帧（fi）及其前后第 skip 帧（frf / frl）
    2. 调用 flow()：
       a. RAFT 估计 frf<->frl 的前后向光流
       b. 前后向一致性检验，生成不可信掩码
       c. InpaintingModel 对不可信区域的光流进行 GAN 修复
    3. DIFNet.forward1()：
       利用修复后的光流对 frf/frl 做双向 warp 并融合，
       得到插值中间帧 I_int
    4. 调用 flow_nosave()：
       RAFT + InpaintingModel 估计 I_int<->fi 间的光流（IntFlowF）
    5. DIFNet.forward2()：
       用 IntFlowF 将 I_int warp 到 fi 坐标系，与 fi 融合，
       得到稳定化输出帧 f_hat
    6. 保存 f_hat 及光流可视化图

    每轮迭代结束后：
    - 将所有帧合成 AVI 视频（30fps / 50fps）
    - 调用 metrics() 计算 CR / DV / SS 指标

    注意：
    - model_name / scene_name / video_index 为模块级全局变量，
      需在调用前由外部设置（见 run_demo.py）

    参数：
        args    推理参数 Namespace（含模型路径、迭代次数等）
        config  流补全模型配置（Config 对象）
    """
    # ── 加载 DIFNet 帧插值网络 ──────────────────────────────────────────────
    DIFNet = DIFNet2().cuda()
    ckpt = torch.load(args.difrint_model)
    weitghts = {}
    for k, v in ckpt.items():
        # 兼容 DataParallel 保存的权重（去掉 'module.' 前缀）
        new_k = k.replace('module.', '') if 'module' in k else k
        weitghts[new_k] = v
        
    DIFNet.load_state_dict(weitghts,strict=False)
    DIFNet.eval()


    if args.flowmodel == 'RAFT':
        RAFTNet = torch.nn.DataParallel(RAFT(args)).cuda()
        RAFTNet.load_state_dict(torch.load(args.raft_model))
        RAFTNet = RAFTNet.module
        RAFTNet.eval()
        flowmodel = RAFTNet
    elif args.flowmodel == 'PWC':
        pwc = PwcNet().cuda()
        pwc.load_state_dict(torch.load('./DIFRINT/trained_models/sintel.pytorch'))
        pwc.eval()
        flowmodel = pwc
        pass

    

    inpaintingmodel = InpaintingModel(config).to(config.DEVICE)
    inpaintingmodel.load()
    inpaintingmodel.eval()

    for num_iter in range(0, args.n_iter):
        idx = 0
        print('\nIter: ' + str(num_iter+1))

        out_path = args.outroot_origin 
        model_out_path = os.path.join(out_path, model_name) + '_iter%d'%(num_iter+1)
        if os.path.exists(model_out_path) is False:
            os.mkdir(model_out_path) 
        scene_out_path = os.path.join(model_out_path, scene_name)
        if os.path.exists(scene_out_path) is False:
            os.mkdir(scene_out_path) 
        frame_out_path =  os.path.join(scene_out_path, video_index)
        if os.path.exists(frame_out_path) is False:
            os.mkdir(frame_out_path) 
        args.out_file = frame_out_path
        args.outroot = args.out_file

        input_path = os.path.join(out_path, model_name) + '_iter%d'%(num_iter)
        input_path = os.path.join(input_path, scene_name, video_index, 'f_hat')

        frameList = os.listdir(args.in_file)
        frameList.sort()

        os.makedirs(args.out_file + '/f_hat', exist_ok=True)
        # os.makedirs(args.out_file + '/I_int', exist_ok=True)
        # os.makedirs(args.out_file + '/w1', exist_ok=True)
        # os.makedirs(args.out_file + '/w2', exist_ok=True)
        # os.makedirs(args.out_file + '/w12', exist_ok=True)

        outpath = args.out_file + '/f_hat'
        copyfile(args.in_file +'/'+ frameList[0], outpath +'/'+ frameList[0])
        copyfile(args.in_file +'/'+ frameList[-1], outpath +'/'+ frameList[-1])
        copyfile(args.in_file +'/'+ frameList[-2], outpath +'/'+ frameList[-2])
        # outpath = args.out_file + '/I_int'
        # copyfile(args.in_file +'/'+ frameList[0], outpath +'/'+ frameList[0])
        # copyfile(args.in_file +'/'+ frameList[-1], outpath +'/'+ frameList[-1])
        # copyfile(args.in_file +'/'+ frameList[-2], outpath +'/'+ frameList[-2])


        for f in frameList[1:-2]:
            if f.endswith('.png'):
                if num_iter == 0:
                    src = args.in_file
                else:
                    src = input_path
                #end

                if idx < args.skip or idx >= (len(frameList)-1-args.skip):
                    skip = 1
                else:
                    skip = args.skip
                #end

                frf = torch.cuda.FloatTensor(np.array(Image.open(src +'/'+ f[:-9] + '%05d.png' % (int(f[-9:-4])-skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] ).cuda()

                #frf = torch.cuda.FloatTensor(np.array(Image.open(outpath +'/'+ f[:-9] + '%05d.png' % (int(f[-9:-4])-skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] ).cuda()
                #fr_g2 = torch.cuda.FloatTensor(np.array(Image.open(src + f)).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
                frl = torch.cuda.FloatTensor(np.array(Image.open(src +'/'+ f[:-9] + '%05d.png' % (int(f[-9:-4])+skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] ).cuda()

                #fr_o1 = torch.cuda.FloatTensor(np.array(Image.open(args.in_file + f[:-9] + '%05d.png' % (int(f[-9:-4])-skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
                fi = torch.cuda.FloatTensor(np.array(Image.open(args.in_file +'/'+ f)).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] ).cuda()
                #fr_o3 = torch.cuda.FloatTensor(np.array(Image.open(args.in_file + f[:-9] + '%05d.png' % (int(f[-9:-4])+skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)

                with torch.no_grad():
                    itera = args.itera 
                    corrFlowF, corrFlowB = flow(args, config, flowmodel, inpaintingmodel, frf, frl, idx, itera)
                    # idx += 1
                    # continue

                    I_int, w1, w2 = DIFNet.forward1(frf/255.0, frl/255.0, corrFlowB.cuda(), corrFlowF.cuda(), 0.5)

                    IntFlowF, _ = flow_nosave(args, config, flowmodel, inpaintingmodel, I_int*255.0, fi, idx, itera)
                     
                    fhat = DIFNet.forward2(I_int, fi/255.0, IntFlowF.cuda())

                # # Save image
                # img = Image.fromarray(np.uint8(fhat.cpu().squeeze().permute(1,2,0)*255))
                # img.save(args.out_file + f)

                # img = Image.fromarray(np.uint8(fhat.cpu().squeeze().permute(1,2,0)*255))
                # img.save(args.out_file + f)

                # sys.stdout.write('\rFrame: ' + str(idx) + '/' + str(len(frameList)-2))
                # sys.stdout.flush()

                os.makedirs(args.out_file+'/comp_forward', exist_ok = True)
                os.makedirs(args.out_file+'/comp_backward', exist_ok = True)
                Image.fromarray(raft_utils.flow_viz.flow_to_image(np.array(corrFlowF.cpu().squeeze().permute(1, 2, 0)))).save(args.out_file+'/comp_forward/%05d.png'%(idx))
                Image.fromarray(raft_utils.flow_viz.flow_to_image(np.array(corrFlowB.cpu().squeeze().permute(1, 2, 0)))).save(args.out_file+'/comp_backward/%05d.png'%(idx))

                idx += 1

                img_f_hat = Image.fromarray(np.uint8(fhat.cpu().squeeze().permute(1,2,0)*255))
                img_f_hat.save(args.out_file +'/f_hat/'+ f)
                # img_I_int = Image.fromarray(np.uint8(I_int.cpu().squeeze().permute(1,2,0)*255))
                # img_I_int.save(args.out_file +'/I_int/'+ f)
                # img_I_int = Image.fromarray(np.uint8(w1.cpu().squeeze().permute(1,2,0)*255))
                # img_I_int.save(args.out_file +'/w1/'+ f)
                # img_I_int = Image.fromarray(np.uint8(w2.cpu().squeeze().permute(1,2,0)*255))
                # img_I_int.save(args.out_file +'/w2/'+ f)
                # img_I_int = Image.fromarray(np.uint8((0.5*(w1+w2)).cpu().squeeze().permute(1,2,0)*255))
                # img_I_int.save(args.out_file +'/w12/'+ f)

                sys.stdout.write('\rFrame: ' + str(idx) + '/' + str(len(frameList)-1-args.skip))
                sys.stdout.flush()

        print('\nMaking video...')
        fr = 30
        frame2vid(src=args.out_file+'/f_hat', vidDir=args.out_file + '/f_hat_%d.avi'%fr, framerate = fr)
        # frame2vid(src=args.out_file+'/I_int', vidDir=args.out_file + '/I_int_%d.avi'%fr, framerate = fr)
        # frame2vid(src=args.out_file+'/w1', vidDir=args.out_file + '/w1_%d.avi'%fr, framerate = fr)
        # frame2vid(src=args.out_file+'/w2', vidDir=args.out_file + '/w2_%d.avi'%fr, framerate = fr)
        # frame2vid(src=args.out_file+'/w12', vidDir=args.out_file + '/w12_%d.avi'%fr, framerate = fr)

        fr = 50
        frame2vid(src=args.out_file+'/f_hat', vidDir=args.out_file + '/f_hat_%d.avi'%fr, framerate = fr)
        # frame2vid(src=args.out_file+'/I_int', vidDir=args.out_file + '/I_int__%d.avi'%fr, framerate = fr)

        ### Assess with metrics
        print('\nComputing metrics...')
        log_path = args.out_file + '/log.txt'
        metrics(original_dir=args.in_file+'/', pred_dir=args.out_file+'/f_hat/', log_path=log_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # video completion
    parser.add_argument('--edge_guide', default='True', action='store_true', help='Whether use edge as guidance to complete flow')
    parser.add_argument('--mode', default='edge', help="modes: object_removal / video_extrapolation")
    # parser.add_argument('--path', default='./data/test/1', help="dataset for evaluation")
    parser.add_argument('--outroot', default='/data1/pengzhan/work/flow_dif/output', help="output directory")
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=5.0, type=float, help='flow consistency error threshold')
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--homography', action='store_true', default=False, help='Whether use homography as guidance to compute flow')
    parser.add_argument('--device', action='store_true', default=[1])

    parser.add_argument('--flowmodel', default='RAFT' , action='store_true')

    # RAFT
    parser.add_argument('--raft_model', default='./weight/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # Deepfill
    parser.add_argument('--deepfill_model', default='./weight/imagenet_deepfill.pth', help="restore checkpoint")

    # Edge completion
    parser.add_argument('--edge_completion_model', default='./weight/edge_completion.pth', help="restore checkpoint")
    
    #DIFRINT
    parser.add_argument('--n_iter', type=int, default=3, help='number of stabilization interations')
    parser.add_argument('--skip', type=int, default=2, help='number of frame skips for interpolation')
    parser.add_argument('--difrint_model', default='./weight/DIFNet2.pth', help="restore checkpoint")

    #flowinpainting
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints/wos_wog', help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1])


    args = parser.parse_args()
    args.outroot = '/data1/pengzhan/work/flow_dif/output'
    args.outroot = '/data4/pengzhan/work/flow_dif/output'
    args.outroot_origin = args.outroot

    model_name = os.path.basename(args.path)
    # model_name = 'testing_ws_wg'
    model_name = 'wos_wog'
    # scene_name = 'Running'
    # video_index = '0'
    args.itera = 1

    # model_name = 'raft_flow_' + str(args.itera) 
    # model_list = ['flow_DIF_intersection','flow_DIF_intersection_edgeconnect','flow_DIF']
    # model_list = ['flow_DIF']
    # for model_name in model_list:
    scene_list = ['Regular', 'QuickRotation', 'Running', 'Crowd',  'Parallax', 'Zooming']
    scene_list1 = ['Running']
    scene_list2 = ['Crowd', 'Regular']
    scene_list3 = ['Parallax', 'Zooming']
    # scene_list = ['Regular']
    # scene_list =['Zooming','Regular','Parallax']
    startlist = [23, 29, 22, 23, 9, 28]
    startlist = [0, 0, 0, 0, 0, 0]
    for idx, scene_name in enumerate(scene_list1[::]):
        scene_path =os.path.join('/data3/zhaoweiyue/data/stable_video_dataset/all_video_results/results_frame', scene_name)
        # for video_index in range(3,4):
        start = startlist[idx]
        for video_index in range(start,50):
            video_index = str(video_index)
            video_path = os.path.join(scene_path,video_index + "_Input.mp4")
            if not os.path.exists(video_path):
                break
            frameList = os.listdir(video_path)
            frameList.sort()

            args.in_file = video_path
            args.model = model_name

            # out_path = args.outroot 

            # model_out_path = os.path.join(out_path, model_name) + '_iter%d'%(args.n_iter)
            # if os.path.exists(model_out_path) is False:
            #     os.mkdir(model_out_path) 
            # scene_out_path = os.path.join(model_out_path, scene_name)
            # if os.path.exists(scene_out_path) is False:
            #     os.mkdir(scene_out_path) 
            # frame_out_path =  os.path.join(scene_out_path, video_index)
            # if os.path.exists(frame_out_path) is False:
            #     os.mkdir(frame_out_path) 
            # args.out_file = frame_out_path
            # args.outroot = args.out_file



            config_path = os.path.join(args.path, 'config_train.yml')

            os.makedirs(args.path, exist_ok=True)
            config = Config(config_path)

            # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.device)



            # init device
            if torch.cuda.is_available():
                config.DEVICE = torch.device("cuda")
                torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
            else:
                config.DEVICE = torch.device("cpu")

            # frame2vid(src=args.out_file+'/f_hat', vidDir=args.out_file + '/50.avi', framerate=50)

            main(args, config)
        # fpath =os.path.dirname(args.out_file)+'/average.txt'

        # Cropping_Ratio = []
        # Distortion = []
        # Stability = []

        # with open(fpath, 'r') as file_to_read:
        #     for lines in islice(file_to_read,0,None):
        #         # if not lines:#空数组代表False
        #         #     break
        #         [name, CR, D, S]= lines.split()
        #         Cropping_Ratio.append([float(cr) for cr in CR.split('|') ])
        #         Distortion.append(float(D))
        #         Stability.append([float(s) for s in S.split('|') ])
        # file_to_read.close()
        
        # with open(fpath, 'a') as file_to_read:
        #     Cropping_Ratio = np.array(Cropping_Ratio).mean(axis=0)
        #     Distortion = np.array(Distortion).mean()
        #     Stability = np.array(Stability).mean(axis=0)
        #     input = ['average', '|'.join(str(Cropping_Ratio)[1:-1].split()), str(Distortion), '|'.join(str(Stability)[1:-1].split())]
        #     for index, num in enumerate(input):
        #     # self.file.write("{0:.6f}".format(num))
        #         file_to_read.write(num)
        #         file_to_read.write('\t')
        #     file_to_read.write('\n')
        #     file_to_read.flush
        # file_to_read.close()

        