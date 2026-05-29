import sys
sys.path.append('core')
from extractor import BasicEncoder
from corr import CorrBlock
from utils.utils import coords_grid

import cv2
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import sys
from shutil import copyfile

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.models import DIFNet, DIFNet2
from models.pwcNet import PwcNet
from metrics import metrics
from frame2vid import frame2vid

from PIL import Image
import numpy as np
import math
import pdb
import time



#python run_seq2.py --cuda --n_iter 3 --skip 2


def datagenerator(src, index, window_size=10, GT=True):
    size = (704, 384)

    frame_list = os.listdir(src)
    number = len(frame_list)
    frame_input = []

    # if GT: 
    #     # image = cv2.imread(stable_dir+stable_list[index])
    #     # image1 = cv2.resize(image, (640, 360), interpolation=cv2.INTER_CUBIC)
    #     # cv2.imwrite('test.jpg',image1)
    #     frame_GT = torch.tensor(cv2.resize(cv2.imread(stable_dir+stable_list[index]), size, interpolation=cv2.INTER_CUBIC)).permute(2,0,1).unsqueeze(0)
    for i in range(2*window_size+1):
        index_final = index-window_size+i
        if index_final < 0:
            frame_input.append(torch.tensor(cv2.resize(cv2.imread(src +'/'+ frame_list[0]), size, interpolation=cv2.INTER_CUBIC)).permute(2,0,1).unsqueeze(0))
            print('Warning: index=%d is less than zero, the first frame will supplant this frame!'%(int(index_final)))
        elif index_final >= number:
            frame_input.append(torch.tensor(cv2.resize(cv2.imread(src +'/'+ frame_list[-1]), size, interpolation=cv2.INTER_CUBIC)).permute(2,0,1).unsqueeze(0))
            print('Warning: index=%d is more than length of this video, the last frame will supplant this frame!'%(int(index_final)))
        else:
                frame_input.append(torch.tensor(cv2.resize(cv2.imread(src +'/'+ frame_list[index_final]), size, interpolation=cv2.INTER_CUBIC)).permute(2,0,1).unsqueeze(0))
    frame_prime = torch.tensor(cv2.resize(cv2.imread(src +'/'+ frame_list[index]), size, interpolation=cv2.INTER_CUBIC)).permute(2,0,1).unsqueeze(0).cuda()
    frame_ref = torch.cat(frame_input,dim=0).cuda()
    return frame_prime, frame_ref

class CORR(nn.Module):
    def __init__(self):
        super(CORR, self).__init__()

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 1
        self.corr_radius = 4

        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=0)   

    def forward(self, frame_prime, frame_ref):
        image1 = 2 * (frame_ref / 255.0) - 1.0#归一化到±1之间
        image2 = 2 * (frame_prime / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        fmap1, fmap2 = self.fnet([image1, image2])#(B,256,H,W), (1,256,H,W)    
        
        fmap1 = fmap1.float()#1，256， H/8, W/8
        fmap2 = fmap2.float()

        N, C, H, W = frame_ref.shape

        result = []
        for i in range(N):
            a = fmap1[i].unsqueeze(0)
            corr_fn = CorrBlock(fmap1[i].unsqueeze(0), fmap2, radius=self.corr_radius, num_levels=self.corr_levels)
            coords = coords_grid(1, H//8, W//8, device=frame_prime.device)
            corr = corr_fn(coords)
            result.append(corr.mean())

        index_sorted = np.argsort(result)
        # assert index_sorted[-1] == N//2
        for i in range(3,N//2+1):
            index1 = N//2+i
            if index1 in index_sorted[N//2+1:]:
                break
        for i in range(3,N//2+1):
            index2 = N//2-i
            if index2 in index_sorted[N//2+1:]:
                break
        return index1, index2

parser = argparse.ArgumentParser()
parser.add_argument('--model1', default='./trained_models/DIFNet2.pth') ####2
parser.add_argument('--model2', default='./trained_models/raft-things.pth') ####2
parser.add_argument('--in_file', default='./data/Stab_te_reg/10/')
parser.add_argument('--out_file', default='./output/OurStabReg2/10/')
parser.add_argument('--n_iter', type=int, default=1, help='number of stabilization interations')
parser.add_argument('--skip', type=int, default=2, help='number of frame skips for interpolation')
#parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
#parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
#parser.add_argument('--height', type=int, default=720, help='size of the data crop (squared assumed)')
#parser.add_argument('--width', type=int, default=1280, help='size of the data crop (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
	print("WARNING: You have a CUDA device, so you should probably run with --cuda")

##########################################################

### Networks
DIFNet = DIFNet2()
corr_module = CORR()

# Place Network in cuda memory
DIFNet.cuda()
corr_module.cuda()

### DataParallel
DIFNet = nn.DataParallel(DIFNet)
corr_module = nn.DataParallel(corr_module)
DIFNet.load_state_dict(torch.load(opt.model1))
DIFNet.eval()

save_model = torch.load(opt.model2)
model_dict = corr_module.state_dict()
state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
print(state_dict.keys())
model_dict.update(state_dict)
corr_module.load_state_dict(model_dict)
corr_module.eval()
##########################################################
i=0
while True:
    opt.in_file = '/data3/zhaoweiyue/data/stable_video_dataset/all_video_results/results_frame/Running/'

    # opt.out_file = './output/Running/%d_DIFRINT_plus_1_src.mp4/'%(i)
    opt.out_file = '/data4/pengzhan/work/DIFRINT_plus/output/Running/%d/'%(i)
    # opt.out_file = '/data4/pengzhan/work/DIFRINT_plus/output/OurStabReg2/new/Running/%d/'%(i)
    if os.path.exists(opt.out_file) is False:
        os.makedirs(opt.out_file)
    if os.path.exists(opt.in_file +'%d_Input.mp4'%(i)+'/') is False:
        break
    opt.in_file = opt.in_file+'%d_Input.mp4'%(i)+'/'

    # if i!=10:
    #     i+=1
    #     continue

    fpath = '/data4/pengzhan/work/DIFRINT_plus/xy/run/%d/log_divid_667.txt'%(i)
    fpath = '/data4/pengzhan/work/DIFRINT_plus/output/OurStabReg2/new/Running/%d_contrast/'%(i)

    # opt.in_file = '/data4/pengzhan/work/DIFRINT_plus/short/0/'
    # opt.out_file = '/data4/pengzhan/work/DIFRINT_plus/short/output/'

    frameList = os.listdir(opt.in_file)
    frameList.sort()

    if os.path.exists(opt.out_file):
        copyfile(opt.in_file + frameList[0], opt.out_file + frameList[0])
        copyfile(opt.in_file + frameList[-2], opt.out_file + frameList[-2])
        copyfile(opt.in_file + frameList[-1], opt.out_file + frameList[-1])
    else:
        os.makedirs(opt.out_file)
        copyfile(opt.in_file + frameList[0], opt.out_file + frameList[0])
        copyfile(opt.in_file + frameList[-2], opt.out_file + frameList[-2])
        copyfile(opt.in_file + frameList[-1], opt.out_file + frameList[-1])
    #end
    window_size = 10
    ## Generate output sequence
    for num_iter in range(opt.n_iter):
        idx = 1
        lam = (1,1)
        print('\nIter: ' + str(num_iter+1))
        for f in frameList[1:-1]:
            if f.endswith('.png'):
                if num_iter == 0:
                    src = opt.in_file
                else:
                    src = opt.out_file
                #end

                if idx < opt.skip or idx > (len(frameList)-1-opt.skip):
                    skip = 1
                else:
                    skip = opt.skip
                #end
                
                frame_prime, frame_ref = datagenerator(src, idx, window_size=window_size, GT=True)
                index1, index2 = corr_module(frame_prime, frame_ref)

                index1 = int(f[-9:-4])-window_size+index1
                index2 = int(f[-9:-4])-window_size+index2
                if index1 < 0:
                    index1 = 0
                elif index1 > len(frameList)-1:
                    index1 = len(frameList)-1

                if index2 < 0:
                    index2 = 0
                elif index2 > len(frameList)-1:
                    index2 = len(frameList)-1
                
                # fr_g1 = torch.cuda.FloatTensor(np.array(Image.open(src + f[:-9] + '%05d.png' % (int(f[-9:-4])-skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
                # #fr_g2 = torch.cuda.FloatTensor(np.array(Image.open(src + f)).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
                # fr_g3 = torch.cuda.FloatTensor(np.array(Image.open(src + f[:-9] + '%05d.png' % (int(f[-9:-4])+skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
                # fr_g4 = torch.cuda.FloatTensor(np.array(Image.open(src + f[:-9] + '%05d.png' % (int(f[-9:-4])+skip+1))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)

                # #fr_o1 = torch.cuda.FloatTensor(np.array(Image.open(opt.in_file + f[:-9] + '%05d.png' % (int(f[-9:-4])-skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
                # fr_o2 = torch.cuda.FloatTensor(np.array(Image.open(src + f)).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
                # #fr_o3 = torch.cuda.FloatTensor(np.array(Image.open(opt.in_file + f[:-9] + '%05d.png' % (int(f[-9:-4])+skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)

                fr_g1 = torch.cuda.FloatTensor(np.array(Image.open(opt.out_file + f[:-9] + '%05d.png' % (int(f[-9:-4])-skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
                fr_o2 = torch.cuda.FloatTensor(np.array(Image.open(src + f[:-9] + '%05d.png' % (int(f[-9:-4])-skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
                fr_g3 = torch.cuda.FloatTensor(np.array(Image.open(src + f)).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)
                fr_g4 = torch.cuda.FloatTensor(np.array(Image.open(src + f[:-9] + '%05d.png' % (int(f[-9:-4])+skip))).transpose(2, 0, 1).astype(np.float32)[None,:,:,:] / 255.0)

                with torch.no_grad():
                    if os.path.exists(fpath) is False:
                        os.makedirs(fpath)
                    fhat, I_int, lam = DIFNet(fr_g1, fr_g3, fr_o2, fr_g3, fr_g1, fr_g4, 0.5, fpath+f, lam = lam) # Notice 0.5

                # Save image
                img = Image.fromarray(np.uint8(fhat.cpu().squeeze().permute(1,2,0)*255))
                img.save(opt.out_file + f)

                sys.stdout.write('\rFrame: ' + str(idx) + '/' + str(len(frameList)-2))
                sys.stdout.flush()

                idx += 1
            #end
        #end
    #end

    ## Make video
    print('\nMaking video...')
    frame2vid(src=opt.out_file, vidDir=opt.out_file[:-1] + '.avi')

    ## Assess with metrics
    print('\nComputing metrics...')
    metrics(opt.in_file, opt.out_file)

    i+=1




