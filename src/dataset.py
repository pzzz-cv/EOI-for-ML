import os
import glob
# import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
# import imageio.imread as imread
from imageio import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask
import cv2

def readFlow(fn):
# Code adapted from:
# http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

# WARNING: this will work on little-endian architectures (eg Intel x86) only!
# print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flow_flist, img_flist, mask_flist, edge_flist=None, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.flow_data = self.load_flist(flow_flist)
        self.image_data = self.load_flist(img_flist)
        self.edge_data = None #self.load_flist(edge_flist)
        self.mask_data = self.load_flist(mask_flist)

        self.input_size = config.INPUT_SIZE
        self.sigma = config.SIGMA
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS

        self.count = 0

        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random
        if config.MODE == 'test':
            self.mask = 6

    def __len__(self):
        return len(self.flow_data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.flow_data[index]
        return os.path.basename(name)


    def load_item(self, index):

        size = self.input_size

        # load image
        flow = readFlow(self.flow_data[index])
        img = imread(self.image_data[index])

        # load mask
        mask = self.load_mask(flow, index)

        # resize/crop if needed
        if size != 0:
            flow = self.resize(flow, size, size)
            img = self.resize(img, size, size)
            mask = self.resize(mask, size, size)

        # # load edge
        flow_img_gray = (flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2) ** 0.5
        flow_img_gray = flow_img_gray / flow_img_gray.max()
        # imgH, imgW, _= flow.shape
        # gradient_x = np.concatenate((np.diff(flow, axis=1), np.zeros((imgH, 1, 2), dtype=np.float32)), axis=1)
        # gradient_y = np.concatenate((np.diff(flow, axis=0), np.zeros((1, imgW, 2), dtype=np.float32)), axis=0)
        # flow_img_gray = (gradient_x[..., 0] ** 2 + gradient_y[..., 0] ** 2 + gradient_x[..., 1] ** 2 + gradient_y[..., 1] ** 2) ** 0.5
        # flow_img_gray = flow_img_gray / flow_img_gray.max()

        edge = canny(flow_img_gray, sigma=2, mask=(1 - mask).astype(bool))

        # img_gray = rgb2gray(img)
        # edge = self.load_edge(img_gray, index, mask)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            flow = flow[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]

            if np.random.binomial(1, 0.5) > 0:
                flow = flow[::-1, :, ...]
                edge = edge[::-1, :, ...]
                mask = mask[::-1, :, ...]

            if np.random.binomial(1, 0.5) > 0:
                flow = -flow[:, :, ...]
        flow = torch.tensor(np.ascontiguousarray(flow)/255.0).permute(2,0,1)
        edge = torch.tensor(np.ascontiguousarray(edge), dtype= torch.float32).unsqueeze(0)
        mask = torch.tensor(np.ascontiguousarray(mask)/255.0, dtype= torch.float32).unsqueeze(0)

        return flow, edge, mask

    def load_edge(self, img, index, mask):
        sigma = self.sigma

        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions
        mask = None if self.training else (1 - mask / 255).astype(np.bool)

        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float)

            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)

            return canny(img, sigma=sigma, mask=None).astype(np.float)

        # external
        else:
            imgh, imgw = img.shape[0:2]
            edge = imread(self.edge_data[index])
            edge = self.resize(edge, imgh, imgw)

            # non-max suppression
            if self.nms == 1:
                edge = edge * canny(img, sigma=sigma, mask=None)

            return edge

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            # mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        #img = scipy.misc.imresize(img, [height, width])
        img = cv2.resize(img, (width, height))

        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    with open(flist, 'r') as f:
                        flist = f.read().splitlines()
                    return flist
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


