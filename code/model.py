import PIL
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models, transforms
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from PartialConv2d import PartialConv2d as Pconv
from attention import SpatialAttentionGeneral
import torchvision.transforms as T
from utils.data import imagenet_preprocess, Resize
from miscc.config import cfg

from PIL import Image

def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        # print(vgg16)
        self.enc = nn.Sequential(*vgg16.features[:24])
        for param in (self.enc).parameters():
            param.requires_grad = False

    def forward(self, image):
        result = self.enc(image)
        return result

class GuidedGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_spectral_norm=True):
        super().__init__()
        # encoder
        self.e1_c = spectral_norm(Pconv(in_channels=input_nc, out_channels=ngf, kernel_size=4, stride=2, padding=1,\
            multi_channel = True, return_mask = False, bias = False), use_spectral_norm)

        self.e2_c = spectral_norm(Pconv(in_channels=ngf, out_channels=ngf*2, kernel_size=4, stride=2, padding=1,\
            multi_channel = True, return_mask = False, bias = False), use_spectral_norm)
        self.e2_norm = norm_layer(ngf * 2)

        self.e3_c = spectral_norm(Pconv(in_channels=ngf*2, out_channels=ngf*4, kernel_size=4, stride=2, padding=1,\
            multi_channel = True, return_mask = False, bias = False), use_spectral_norm)
        self.e3_norm = norm_layer(ngf * 4)

        self.e4_c = spectral_norm(Pconv(in_channels=ngf*4, out_channels=ngf*8, kernel_size=4, stride=2, padding=1,\
            multi_channel = True, return_mask = False, bias = False), use_spectral_norm)
        # self.e4_norm = norm_layer(ngf * 8)

        # attention layer
        self.attn = SpatialAttentionGeneral(idf=512, cdf=256)

        # decoder
        self.d1_c = spectral_norm(nn.ConvTranspose2d(in_channels=ngf*8, out_channels=ngf*4, kernel_size=4, stride=2, \
            padding=1,bias = False), use_spectral_norm)
        self.d1_norm = norm_layer(ngf * 4)

        self.d2_c = spectral_norm(nn.ConvTranspose2d(in_channels=ngf*8, out_channels=ngf*2, kernel_size=4, stride=2, \
            padding=1,bias = False), use_spectral_norm)
        self.d2_norm = norm_layer(ngf * 2)

        self.d3_c = spectral_norm(nn.ConvTranspose2d(in_channels=ngf*4, out_channels=ngf, kernel_size=4, stride=2, \
            padding=1,bias = False), use_spectral_norm)
        self.d3_norm = norm_layer(ngf)

        self.d4_c = spectral_norm(nn.ConvTranspose2d(in_channels=ngf*2, out_channels=output_nc, kernel_size=4, stride=2, \
            padding=1,bias = False), use_spectral_norm)
        

    def forward(self, image, condition):
        # encoder
        # No norm on the first layer
        e1 = self.e1_c(image)
        e2 = self.e2_norm(self.e2_c(F.leaky_relu_(e1, negative_slope=0.2)))
        e3 = self.e3_norm(self.e3_c(F.leaky_relu_(e2, negative_slope=0.2)))
        # No norm on the inner_most layer
        e4 = self.e4_c(F.leaky_relu_(e3, negative_slope=0.2))

        #print(e4.size()[1], condition.size()[1])
        #self.attn = SpatialAttentionGeneral(idf=(e4.size()[1]), cdf=(condition.size()[1]))
        a1, _ = self.attn(e4, condition)

        # decoder
        # No norm on the last layer
        d1 = self.d1_norm(self.d1_c(F.relu_(a1)))
        d2 = self.d2_norm(self.d2_c(F.relu_(torch.cat([d1, e3], dim=1)))) # torch.Size([10, 128, 64, 64])
        d3 = self.d3_norm(self.d3_c(F.relu_(torch.cat([d2, e2], dim=1)))) # torch.Size([10, 64, 128, 128])
        d4 = self.d4_c(F.relu_(torch.cat([d3, e1], 1)))
        d4 = torch.tanh(d4)

        return d4

# discriminator
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

# 将空间大小缩小一倍,h_out = h_in / 2
# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

# h_out = h_in / 16
# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img

class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            # h_out = h_in / 4
            nn.Sigmoid())

    def forward(self, h_code):
        output = self.outlogits(h_code)
        return output.view(-1)

# for 256 * 256 images
class Discriminator256(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.ndf = ndf
        self.img_code_s16 = encode_image_by_16times(self.ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.UNCOND_DNET = D_GET_LOGITS(self.ndf)
    
    def forward(self, image):
        # img； batch x 3 x 256 x 256
        x_code16 = self.img_code_s16(image)
        x_code8 = self.img_code_s32(x_code16)
        x_code4 = self.img_code_s64(x_code8)
        x_code4 = self.img_code_s64_1(x_code4)
        x_code4 = self.img_code_s64_2(x_code4)  # torch.Size([10, 512, 4, 4])
        return x_code4


class Discriminator_per_object(nn.Module):
    def __init__(self, ndf=32, normalize_image=False, image_size=(64, 64)):
        super().__init__()
        self.ndf = ndf
        self.normalize_image = normalize_image
        self.unloader = transforms.ToPILImage()
        self.set_image_size(image_size)
        
        # 不再使用vgg提取特征
        '''
        vgg16 = models.vgg16(pretrained=True)
        self.enc = nn.Sequential(*vgg16.features[:17]) # 3 x 64 x 64 -> 256 x 4 x 4
        for param in (self.enc).parameters():
            param.requires_grad = False
        '''
        self.enc = nn.Sequential(
                  spectral_norm(nn.Conv2d(3,64,4,2,1),True),
                  nn.LeakyReLU(0.02),
                  nn.BatchNorm2d(64),
                  spectral_norm(nn.Conv2d(64,128,4,2,1),True),
                  nn.LeakyReLU(0.02),
                  nn.BatchNorm2d(128),
                  spectral_norm(nn.Conv2d(128,256,4,2,1),True),
                  nn.LeakyReLU(0.02),
                  nn.BatchNorm2d(256),
                  spectral_norm(nn.Conv2d(256,256,4,2,1),True),
                  nn.LeakyReLU(0.02)
                )

        self.UNCOND_DNET = D_GET_LOGITS(self.ndf)
    
    def forward(self, img, boxes, obj_to_img):

        for i in range(len(obj_to_img)):
            img = img.cpu()
            per_img = self.unloader(img[int(obj_to_img[i])])
            x_s = int(256 * boxes[i][0])
            y_s = int(256 * boxes[i][1])
            x_e = int(256 * boxes[i][2])
            y_e = int(256 * boxes[i][3])
            img_patch = per_img.crop((x_s, y_s, x_e, y_e))
            image_patch = self.transform(img_patch.convert('RGB')) # 3 x 32 x 32
            image_patch = image_patch.unsqueeze(0)
            if i == 0:
                result_image_patch = image_patch
            else:
                result_image_patch = torch.cat((result_image_patch, image_patch), 0)
        
        # print("result_image_patch", result_image_patch.size())
        if cfg.TRAIN.FLAG:
            result_image_patch = result_image_patch.cuda()
        result_patch_feat = self.enc(result_image_patch)
        # print("result_patch_feat", result_patch_feat.size())

        obj_to_img_list = (obj_to_img.cpu().numpy()).tolist()
        start = 0
        end = 0
        return_list = []
        for i in range(img.size(0)):
            tmp_img_obj_num = obj_to_img_list.count(i)
            end = start + tmp_img_obj_num
            tmp_result_patch_feat = result_patch_feat[start:end] 
            start = end

            output = self.UNCOND_DNET(tmp_result_patch_feat)
            return_list.append(output)
        
        return return_list



    def set_image_size(self, image_size):
        # print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_image:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        




if __name__ == "__main__":  
    '''
    # test generator
    G1 = GuidedGenerator()
    input_t = torch.rand(10, 3, 256, 256)
    condition = torch.rand(10, 256, 16)
    output_t = G1(input_t, condition)
    print(output_t.size())

    input_t2 = torch.rand(2, 3, 256, 256)
    condition2 = torch.rand(2, 256, 16)
    output_t2 = G1(input_t2, condition2)
    print(output_t2.size())
    exit()
    '''

    '''
    # test VGG extractor
    vggmodel = VGGFeatureExtractor()
    input_vgg = torch.rand(10, 3, 64, 64)
    output_vgg = vggmodel(input_vgg)
    print("vgg, " , output_vgg.size())  # torch.Size([10, 512, 4, 4])

    # test discriminator
    D1 = Discriminator256()
    input_d = torch.rand(10, 3, 256, 256)
    output_d = D1(input_d)
    print(output_d.size())
    '''

    # test Discriminator_per_object
    D = Discriminator_per_object(ndf=32, normalize_image=False, image_size=(64, 64))
    img = torch.rand(4, 3, 256, 256)
    boxes = torch.tensor([[0.0808, 0.4849, 0.3311, 0.7973],
        [0.3480, 0.4989, 0.9505, 0.9860],
        [0.2143, 0.2078, 0.7130, 0.9844],
        [0.0000, 0.1865, 0.1938, 0.9888],
        [0.1146, 0.5506, 0.2242, 0.9685],
        [0.2688, 0.3190, 0.9986, 0.6336],
        [0.3390, 0.5411, 0.4613, 0.9370],
        [0.0000, 0.0619, 0.9983, 0.3616],
        [0.0077, 0.5934, 0.4704, 0.7353],
        [0.8495, 0.6721, 1.0000, 0.8262],
        [0.0034, 0.3363, 1.0000, 0.6390]])
    obj_to_img = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3])   
    print(type(img), type(boxes), type(obj_to_img)) 
    output = D(img, boxes, obj_to_img)
    print(output)