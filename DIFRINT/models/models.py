import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math
import pdb
from PIL import Image
from itertools import islice
import numpy as np
import math
from thop import profile
from thop import clever_format
import cv2
from skimage.color import rgb2gray
import os






#############################################################################################################

class UNet2(nn.Module):
	def __init__(self):
		super(UNet2, self).__init__()

		class Encoder(nn.Module):
			def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1):
				super(Encoder, self).__init__()

				self.seq = nn.Sequential(
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
					nn.ReLU()
				)
				self.GateConv = nn.Sequential(
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
					nn.Sigmoid()
				)

			def forward(self, x):
				return self.seq(x) * self.GateConv(x)

		class Decoder(nn.Module):
			def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1, tanh=False):
				super(Decoder, self).__init__()
				
				self.seq = nn.Sequential(
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, in_nc, kernel_size=k_size, stride=stride, padding=0),
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0)
				)

				if tanh:
					self.activ = nn.Tanh()
				else:
					self.activ = nn.ReLU()
				
				self.GateConv = nn.Sequential(
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, in_nc, kernel_size=k_size, stride=stride, padding=0),
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
					nn.Sigmoid()
				)

			def forward(self, x):
				s = self.seq(x)
				s = self.activ(s)
				return s * self.GateConv(x)


		self.enc0 = Encoder(16, 32, stride=1)
		self.enc1 = Encoder(32, 32, stride=2)
		self.enc2 = Encoder(32, 32, stride=2)
		self.enc3 = Encoder(32, 32, stride=2)

		self.dec0 = Decoder(32, 32, stride=1)
		# up-scaling + concat
		self.dec1 = Decoder(32+32, 32, stride=1)
		self.dec2 = Decoder(32+32, 32, stride=1)
		self.dec3 = Decoder(32+32, 32, stride=1)

		self.dec4 = Decoder(32, 3, stride=1, tanh=True)

	def forward(self, w1, w2, flo1, flo2, fr1, fr2):
		s0 = self.enc0(torch.cat([w1, w2, flo1, flo1, fr1, fr2],1).cuda())
		s1 = self.enc1(s0)
		s2 = self.enc2(s1)
		s3 = self.enc3(s2)

		s4 = self.dec0(s3)
		# up-scaling + concat
		s4 = F.interpolate(s4, scale_factor=2, mode='nearest')
		s5 = self.dec1(torch.cat([s4, s2],1).cuda())
		s5 = F.interpolate(s5, scale_factor=2, mode='nearest')
		s6 = self.dec2(torch.cat([s5, s1],1).cuda())
		s6 = F.interpolate(s6, scale_factor=2, mode='nearest')
		s7 = self.dec3(torch.cat([s6, s0],1).cuda())

		out = self.dec4(s7)
		return out


class DIFNet2(nn.Module):
	def __init__(self):
		super(DIFNet2, self).__init__()

		class Backward(torch.nn.Module):
			def __init__(self):
				super(Backward, self).__init__()
			# end

			def forward(self, tensorInput, tensorFlow, scale=1.0):
				if isinstance(scale,list):
					scale = torch.tensor(scale).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()

				if hasattr(self, 'tensorPartial') == False or self.tensorPartial.size(0) != tensorFlow.size(0) or self.tensorPartial.size(2) != tensorFlow.size(2) or self.tensorPartial.size(3) != tensorFlow.size(3):
					self.tensorPartial = torch.FloatTensor().resize_(tensorFlow.size(0), 1, tensorFlow.size(2), tensorFlow.size(3)).fill_(1.0).cuda()
				# end

				if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorFlow.size(0) or self.tensorGrid.size(2) != tensorFlow.size(2) or self.tensorGrid.size(3) != tensorFlow.size(3):
					tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
					tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

					self.tensorGrid = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
				# end
				#pdb.set_trace()
				tensorInput = torch.cat([ tensorInput, self.tensorPartial ], 1)
				tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

				tensorOutput = torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow*scale).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

				tensorMask = tensorOutput[:, -1:, :, :]; tensorMask[tensorMask > 0.999] = 1.0; tensorMask[tensorMask < 1.0] = 0.0

				return tensorOutput[:, :-1, :, :] * tensorMask

		# PWC
		# self.pwc = PwcNet()
		# self.pwc.load_state_dict(torch.load('/data4/pengzhan/work/DIFRINT-MAE/trained_models/sintel.pytorch'))
		# self.pwc.eval()


		# Warping layer
		self.warpLayer = Backward()
		self.warpLayer.eval()

		# UNets
		self.UNet2 = UNet2()
		self.ResNet2 = ResNet2()

	def warpFrame(self, fr_1, fr_2, flow, scale=1.0):
		with torch.no_grad():
			# temp_w = int(math.floor(math.ceil(fr_1.size(3) / 64.0) * 64.0)) # Due to Pyramid method?
			# temp_h = int(math.floor(math.ceil(fr_1.size(2) / 64.0) * 64.0))

			# temp_fr_1 = torch.nn.functional.interpolate(input=fr_1, size=(temp_h, temp_w), mode='nearest')
			# temp_fr_2 = torch.nn.functional.interpolate(input=fr_2, size=(temp_h, temp_w), mode='nearest')

			flo = torch.nn.functional.interpolate(input=torch.tensor(flow), size=(fr_1.size(2), fr_1.size(3)), mode='bilinear', align_corners=False)
			flo = flo.float().cuda()
			return self.warpLayer(fr_2, flo, scale)

	def forward1(self, frf, frl, flowlf, flowfl, scale):
		w1 = self.warpFrame(frl, frf, flowlf, scale)
		w2 = self.warpFrame(frf, frl, flowfl, scale)

		img = Image.fromarray(np.uint8(w1.cpu().squeeze().permute(1,2,0)*255))
		img.save('w1_test.jpg')

		# macs, params = profile(self.UNet2, inputs=(w1, w2, flowlf, flowfl, frf, frl))
		# macs, params = clever_format([macs, params], "%.3f")
		# print(macs, params)

		I_int = self.UNet2(w1, w2, flowlf, flowfl, frf, frl)
		return I_int, w1, w2
	
	def preprocess(self, frf, frl, flowlf, flowfl, scale):
		w1 = self.warpFrame(frl, frf, flowlf, scale)
		w2 = frl

		mask = rgb2gray(w2.cpu().squeeze().permute(1,2,0))>0
		mask = mask.astype(np.uint8)
		kernel = np.ones((2, 2), dtype=np.uint8)
		mask = torch.tensor(cv2.erode(mask, kernel, 1)).cuda().unsqueeze(2).permute(2,0,1).unsqueeze(0)
		w2 = w1*(1-mask) + w2*mask
		return w2

	def forward_test(self, frf, frl, flowlf, flowfl, scale):
		w1 = self.warpFrame(frl, frf, flowlf, scale)
		w2 = frl

		img = Image.fromarray(np.uint8(w1.cpu().squeeze().permute(1,2,0)*255))
		# img.save('test/w1_test.jpg')
		img = Image.fromarray(np.uint8(w2.cpu().squeeze().permute(1,2,0)*255))
		# img.save('test/w2_test.jpg')

		

		# macs, params = profile(self.UNet2, inputs=(w1, w2, flowlf, flowfl, frf, frl))
		# macs, params = clever_format([macs, params], "%.3f")
		# print(macs, params)

		I_int = self.UNet2(w1, w2, flowlf, 0*flowfl, frf, frl)

		# img = Image.fromarray(np.uint8(I_int.cpu().squeeze().permute(1,2,0)*255))
		# img.save('test/I_int.jpg')

		
		# mask = rgb2gray(np.uint8(w2.cpu().squeeze().permute(1,2,0)*255))==0
		
		# Image.fromarray(np.uint8(np.concatenate((mask[..., None],mask[..., None],mask[..., None]), axis= 2))).save('test/mask.png')

		# w1_ns = cv2.inpaint(w1, mask, 3, flags=cv2.INPAINT_NS)
		# w1_te = cv2.inpaint(w1, mask, 3, flags=cv2.INPAINT_TELEA)

		img = Image.fromarray(np.uint8(I_int.cpu().squeeze().permute(1,2,0)*255))
		# img.save('test/I_int_test.jpg')
		return I_int, w1, w2

	def forward2(self, I_int, fi, flow_int):
		f_int = self.warpFrame(I_int, fi, flow_int)

		# img = Image.fromarray(np.uint8(f_int.cpu().squeeze().permute(1,2,0)*255))
		# img.save('test1/f_int.jpg')

		fhat = self.ResNet2(I_int, f_int, flow_int, fi)

		# macs, params = profile(self.ResNet2, inputs=(I_int, f_int, flow_int, fi))
		# macs, params = clever_format([macs, params], "%.3f")
		# print(macs, params)

		return fhat




class ResNet2(nn.Module):
	def __init__(self):
		super(ResNet2, self).__init__()

		class ConvBlock(nn.Module):
			def __init__(self, in_ch, out_ch):
				super(ConvBlock, self).__init__()

				self.seq = nn.Sequential(
					nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
					nn.ReLU()
				)

				self.GateConv = nn.Sequential(
					nn.ReflectionPad2d(1),
					nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=0),
					nn.ReflectionPad2d(1),
					nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=0),
					nn.Sigmoid()
				)

			def forward(self, x):
				return self.seq(x) * self.GateConv(x)


		class ResBlock(nn.Module):
			def __init__(self, num_ch):
				super(ResBlock, self).__init__()

				self.seq = nn.Sequential(
					nn.Conv2d(num_ch, num_ch, kernel_size=1, stride=1, padding=0),
					nn.ReLU()
				)

				self.GateConv = nn.Sequential(
					nn.ReflectionPad2d(1),
					nn.Conv2d(num_ch, num_ch, kernel_size=3, stride=1, padding=0),
					nn.ReflectionPad2d(1),
					nn.Conv2d(num_ch, num_ch, kernel_size=3, stride=1, padding=0),
					nn.Sigmoid()
				)

			def forward(self, x):
				return self.seq(x) * self.GateConv(x) + x


		self.seq = nn.Sequential(
			ConvBlock(11, 32),
			ResBlock(32),
			ResBlock(32),
			ResBlock(32),
			ResBlock(32),
			ResBlock(32),
			ConvBlock(32, 3),
			nn.Tanh()
		)

	def forward(self, I_int, f_int, flo_int, f3):
		return self.seq(torch.cat([I_int, f_int, flo_int, f3],1).cuda())



#############################################################################################################
class UNetFlow(nn.Module):
	def __init__(self):
		super(UNetFlow, self).__init__()

		class Encoder(nn.Module):
			def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1):
				super(Encoder, self).__init__()

				self.seq = nn.Sequential(
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
					nn.LeakyReLU(0.2)
				)
				self.GateConv = nn.Sequential(
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
					nn.Sigmoid()
				)

			def forward(self, x):
				return self.seq(x) * self.GateConv(x)

		class Decoder(nn.Module):
			def __init__(self, in_nc, out_nc, stride, k_size=3, pad=1, tanh=False):
				super(Decoder, self).__init__()
				
				self.seq = nn.Sequential(
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0)
				)

				if tanh:
					self.activ = nn.Tanh()
				else:
					self.activ = nn.LeakyReLU(0.2)
				
				self.GateConv = nn.Sequential(
					nn.ReflectionPad2d(pad),
					nn.Conv2d(in_nc, out_nc, kernel_size=k_size, stride=stride, padding=0),
					nn.Sigmoid()
				)

			def forward(self, x):
				s = self.seq(x)
				s = self.activ(s)
				return s * self.GateConv(x)


		self.enc0 = Encoder(4, 32, stride=1)
		self.enc1 = Encoder(32, 32, stride=2)
		self.enc2 = Encoder(32, 32, stride=2)
		self.enc3 = Encoder(32, 32, stride=2)

		self.dec0 = Decoder(32, 32, stride=1)
		# up-scaling + concat
		self.dec1 = Decoder(32+32, 32, stride=1)
		self.dec2 = Decoder(32+32, 32, stride=1)
		self.dec3 = Decoder(32+32, 32, stride=1)

		self.dec4 = Decoder(32, 2, stride=1)

	def forward(self, x1, x2):
		s0 = self.enc0(torch.cat([x1, x2],1).cuda())
		s1 = self.enc1(s0)
		s2 = self.enc2(s1)
		s3 = self.enc3(s2)

		s4 = self.dec0(s3)
		# up-scaling + concat
		s4 = F.interpolate(s4, scale_factor=2, mode='nearest')
		s5 = self.dec1(torch.cat([s4, s2],1).cuda())
		s5 = F.interpolate(s5, scale_factor=2, mode='nearest')
		s6 = self.dec2(torch.cat([s5, s1],1).cuda())
		s6 = F.interpolate(s6, scale_factor=2, mode='nearest')
		s7 = self.dec3(torch.cat([s6, s0],1).cuda())

		out = self.dec4(s7)
		return out






############################ GAN Discriminator###############################

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

    def forward(self, fr_2):
        #x = torch.cat((fr_1, fr_2), 1)
        x = self.seq(fr_2)
        x = F.avg_pool2d(x, x.size()[2:]).view(-1, x.size()[0]).squeeze()
        out = torch.sigmoid(x)
        return out