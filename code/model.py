import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models, transforms
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from PartialConv2d import PartialConv2d as Pconv
from attention import SpatialAttentionGeneral

from PIL import Image

def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        print(vgg16)
        self.enc = nn.Sequential(*vgg16.features[:17])
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
        # # No norm on the inner_most layer
        e4 = self.e4_c(F.leaky_relu_(e3, negative_slope=0.2))

        # attention layer
        self.attn = SpatialAttentionGeneral(idf=(e4.size()[1]), cdf=(condition.size()[1]))
        a1, _ = self.attn(e4, condition)

        # decoder
        # No norm on the last layer
        d1 = self.d1_norm(self.d1_c(F.relu_(a1)))
        d2 = self.d2_norm(self.d2_c(F.relu_(torch.cat([d1, e3], dim=1)))) # torch.Size([10, 128, 64, 64])
        d3 = self.d3_norm(self.d3_c(F.relu_(torch.cat([d2, e2], dim=1)))) # torch.Size([10, 64, 128, 128])
        d4 = self.d4_c(F.relu_(torch.cat([d3, e1], 1)))
        d4 = torch.tanh(d4)

        return d4

if __name__ == "__main__":
    # test generator
    G1 = GuidedGenerator()
    input_t = torch.rand(10, 3, 256, 256)
    condition = torch.rand(10, 256, 16)
    output_t = G1(input_t, condition)
    print(output_t.size())

    # test VGG extractor
    vggmodel = VGGFeatureExtractor()
    input_vgg = torch.rand(10, 3, 32, 32)
    output_vgg = vggmodel(input_vgg)
    print(output_vgg.size())
