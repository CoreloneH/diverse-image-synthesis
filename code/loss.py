import torch
import torch.nn as nn

import numpy as np

# 超参数
lambda_valid_loss = 1
lambda_hole_loss = 6


# discriminator
def discriminator_realfake_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels):
    # Forward
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())

    real_logits = netD.UNCOND_DNET(real_features)
    fake_logits = netD.UNCOND_DNET(fake_features)
    real_errD = nn.BCELoss()(real_logits, real_labels)
    fake_errD = nn.BCELoss()(fake_logits, fake_labels)
    errD = real_errD + fake_errD

    return errD

# generator
def generator_loss1(real_imgs, fake_imgs, mask=1):
    def l1_loss(real_imgs, fake_imgs, mask=1):
        return torch.mean(torch.mul(torch.abs(real_imgs - fake_imgs), mask))
    
    def style_loss(A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            _, c, w, h = A_feat.size()
            A_feat = A_feat.view(A_feat.size(0), A_feat.size(1), A_feat.size(2) * A_feat.size(3))
            B_feat = B_feat.view(B_feat.size(0), B_feat.size(1), B_feat.size(2) * B_feat.size(3))
            A_style = torch.matmul(A_feat, A_feat.transpose(2, 1))
            B_style = torch.matmul(B_feat, B_feat.transpose(2, 1))
            loss_value += torch.mean(torch.abs(A_style - B_style)/(c * w * h))
        return loss_value

    def preceptual_loss(A_feats, B_feats):
        assert len(A_feats) == len(B_feats), "the length of two input feature maps lists should be the same"
        loss_value = 0.0
        for i in range(len(A_feats)):
            A_feat = A_feats[i]
            B_feat = B_feats[i]
            loss_value += torch.mean(torch.abs(A_feat - B_feat))
        return loss_value

    valid_loss = l1_loss(real_imgs, fake_imgs, mask)
    hole_loss = l1_loss(real_imgs, fake_imgs, (1 - mask))


    errG = lambda_valid_loss * valid_loss + lambda_hole_loss * hole_loss


    return errG


def generator_loss2(netD1, fake_imgs, real_labels):
    fake_features = netD1(fake_imgs.detach())
    fake_logits = netD1.UNCOND_DNET(fake_features)
    errG = nn.BCELoss()(fake_logits, real_labels)
    # logs = 'errG_D1: %.2f ' % (errG.item())
    return errG






if __name__ == "__main__":
    # test discriminator_adversial_loss
    from model import Discriminator256
    D1 = Discriminator256()
    real_imgs = torch.rand(10, 3, 256, 256)
    fake_imgs = torch.rand(10, 3, 256, 256)
    real_labels = torch.ones(10)
    fake_labels = torch.zeros(10)
    output_errD = discriminator_realfake_loss(D1, real_imgs, fake_imgs, real_labels, fake_labels)
    print(output_errD)

    # test generator loss
    mask = torch.rand(10, 3, 256, 256)
    output_errG = generator_loss(real_imgs, fake_imgs, mask=mask)
    print(output_errG)