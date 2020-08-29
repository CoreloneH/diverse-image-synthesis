import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import os
import random
import time
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 

from miscc.utils import weights_init, load_params, copy_G_params, mkdir_p

from model import VGGFeatureExtractor, GuidedGenerator, Discriminator256, Discriminator_per_object
from dataset import save_image, batch_mask_image
from miscc.config import cfg
from loss import discriminator_realfake_loss, generator_loss1, generator_loss2


# 限制使用的GPU个数
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'

class CondGANTrainer(object):
    def __init__(self, output_dir, data_loader):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.loss_dir = os.path.join(output_dir, 'Loss')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)

        cudnn.benchmark = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def build_models(self):
        # vgg extractor
        vgg_extractor = VGGFeatureExtractor()

        # generator
        netG = GuidedGenerator()

        # discriminator
        netD1 = Discriminator256()
        netD2 = Discriminator_per_object()

        model_list = [netG, netD1, netD2]
        for model in model_list:
            model.apply(weights_init)
        model_list.append(vgg_extractor)

        epoch = 0

        if cfg.TRAIN.NET_G != '':
            state_dict = torch.load(cfg.TRAIN.NET_G , map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load G from: ', cfg.TRAIN.NET_G)
            istart = cfg.TRAIN.NET_G.rfind('_') + 1
            iend = cfg.TRAIN.NET_G.rfind('.')
            epoch = cfg.TRAIN.NET_G[istart:iend]
            epoch = int(epoch) + 1
            
            Gname = cfg.TRAIN.NET_G
            s_tmp = Gname[:Gname.rfind('/')]

            Dname1 = '%s/netD%d.pth' % (s_tmp, 1)
            print('Load D1 from: ', Dname)
            state_dict = torch.load(Dname1, map_location=lambda storage, loc: storage)
            netD1.load_state_dict(state_dict)

            Dname2 = '%s/netD%d.pth' % (s_tmp, 2)
            print('Load D2 from: ', Dname)
            state_dict = torch.load(Dname2, map_location=lambda storage, loc: storage)
            netD2.load_state_dict(state_dict)
        

        for model in model_list:
            model = nn.DataParallel(model)
            model.to(self.device)
        
        return [netG, netD1, netD2, epoch, vgg_extractor]

    def define_optimizers(self, netG, netD1, netD2):
        optimizerG = optim.Adam(netG.parameters(), lr=cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))
        
        optimizerD1 = optim.Adam(netG.parameters(), lr=cfg.TRAIN.DISCRIMINATOR1_LR, betas=(0.5, 0.999))
        optimizerD2 = optim.Adam(netG.parameters(), lr=cfg.TRAIN.DISCRIMINATOR2_LR, betas=(0.5, 0.999))

        return optimizerG, optimizerD1, optimizerD2
    
    def save_model(self, netG, avg_param_G, netD1, netD2, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        
        torch.save(netD1.state_dict(), '%s/netD1.pth' % (self.model_dir))
        torch.save(netD2.state_dict(), '%s/netD2.pth' % (self.model_dir))
        print('Save G/Ds models.')
    
    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    
    def save_img_results2(self, fake_imgs, gen_iterations, image_names, epoch):
        for i in range(len(fake_imgs)):
            name = (image_names[i].split('.'))[0]
            fullpath = '%s/fake_%s_%d_epoch_%d.png' % (self.image_dir, name, gen_iterations, epoch)
            save_image(fake_imgs[i], fullpath)

    def save_model(self, netG, avg_param_G, netD1, netD2, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        
        torch.save(netD1.state_dict(), '%s/netD1.pth' % (self.model_dir))
        torch.save(netD2.state_dict(), '%s/netD2.pth' % (self.model_dir))
        print('Save G/Ds models.')

    def train(self):
        netG, netD1, netD2, start_epoch, vgg_extractor = self.build_models()
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizerD1, optimizerD2 = self.define_optimizers(netG, netD1, netD2)

        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        errD1_list, errD2_list, errG_total_list = [], [], []
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            for i, batch_data in enumerate(self.data_loader):
                images, objs, boxes, obj_to_img, image_names = batch_data
                images = images.cuda()
                assert images.size()[0] == len(image_names)
                self.batch_size = images.size()[0]

                mask_imgs, mask_class, real_labels, masked_patch_images, total_mask = batch_mask_image(images, \
                    self.batch_size, objs, boxes, obj_to_img, image_names)

                mask_imgs = mask_imgs.cuda()
                mask_class = mask_class.cuda()
                masked_patch_images = masked_patch_images.cuda()
                total_mask = total_mask.cuda()
                # generate fake images
                conditions = vgg_extractor(masked_patch_images)
                conditions = conditions.view(self.batch_size, 256, -1)
                conditions = conditions.cuda()
                # print("conditions," , conditions.size())

                fake_imgs = netG(mask_imgs, conditions)

                # update D1
                D1_logs = ''
                netD1.zero_grad()
                errD1 = 0.
                real_D1_labels = torch.ones(self.batch_size).cuda()
                fake_D1_labels = torch.zeros(self.batch_size).cuda()
                errD1 = discriminator_realfake_loss(netD1, images, fake_imgs, real_D1_labels, fake_D1_labels)
                # backward and update parameters
                errD1.backward()
                optimizerD1.step()
                D1_logs += 'errD1: %.2f ' % (errD1.item())

                # update D2
                D2_logs = ''
                netD1.zero_grad()
                real_D2_labels_list = real_labels
                fake_D2_labels_list = netD2(images, boxes, obj_to_img)
                # calculate D2 loss
                errD2 = 0.
                for i in range(len(real_D2_labels_list)):
                    # print("real_D2_labels_list[i]", real_D2_labels_list[i], type(real_D2_labels_list[i]))
                    real_D2_labels = torch.from_numpy(np.array(real_D2_labels_list[i])).cuda()
                    real_D2_labels = real_D2_labels.float()
                    # print("fake_D2_labels_list[i], ", fake_D2_labels_list[i], type(fake_D2_labels_list[i]))
                    fake_D2_labels = fake_D2_labels_list[i]
                    #print("real_D2_labels, ", real_D2_labels, type(real_D2_labels))
                    #print("fake_D2_labels", fake_D2_labels, type(fake_D2_labels))
                    err_i = nn.BCELoss()(fake_D2_labels, real_D2_labels)
                    errD2 += err_i

                errD2.backward(retain_graph=True)
                optimizerD2.step()
                D2_logs += 'errD2: %.2f ' % (errD2.item())

                # update G
                # compute total loss for training G
                gen_iterations += 1
                # do not need to compute gradient for Ds
                # self.set_requires_grad_value(netsD, False)
                netG.zero_grad()
                errG_total = 0.
                errG_1 = cfg.TRAIN.LOSS.G_LAMBDA_OTHER * generator_loss1(images, fake_imgs, mask=total_mask)
                errG_2 = cfg.TRAIN.LOSS.G_LAMBDA_ADVERSIAL * generator_loss2(netD1, fake_imgs, real_D1_labels)
                errG_3 = 0.
                for i in range(len(real_D2_labels_list)):
                    real_D2_labels = torch.from_numpy(np.array(real_D2_labels_list[i])).cuda()
                    real_D2_labels = real_D2_labels.float()
                    fake_D2_labels = fake_D2_labels_list[i]
                    fake_G_labels = 1. - fake_D2_labels
                    err_i = nn.BCELoss()(fake_G_labels, real_D2_labels)
                    errG_3 += err_i
                errG_3 = errG_3 / len(real_D2_labels_list)
                errG_3 = cfg.TRAIN.LOSS.G_LAMBDA_PER_OBJ * errG_3
                errG_total = errG_1 + errG_2 + errG_3
                errG_total.backward()
                optimizerG.step()
                G_logs = "errG_1: %.2f, errG_2: %.2f, errG_3: %.2f," % (errG_1.item(), errG_2.item(), errG_3.item())

                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)
                if gen_iterations % 100 == 0:
                    print("gen_iterations: ", gen_iterations, D1_logs + ', ' + D2_logs + ',' + G_logs)
                
                # save images
                if gen_iterations % 500 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results2(fake_imgs, gen_iterations, image_names, epoch)
                    load_params(netG, backup_para)

            end_t = time.time()
            print('''[%d/%d][%d]
                  Loss_D1: %.2f , Loss_D2: %.2f, Loss_G: %.2f Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     errD1.item(), errD2.item(), errG_total.item(),
                     end_t - start_t))
            errD1_list.append(errD1.item())
            errD2_list.append(errD2.item())
            errG_total_list.append(errG_total.item())

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netD1, netD2, epoch)

        pic_name = "loss during training"
        pic_path = '%s/loss.jpg' % (self.loss_dir)
        draw_loss(errD1_list, errD2_list, errG_total_list, self.max_epoch - start_epoch, pic_name, pic_path)

        self.save_model(netG, avg_param_G, netD1, netD2, self.max_epoch)


def draw_loss(errD1_list, errD2_list, errG_total_list, max_epoch, name, path):
    x = range(max_epoch)
    plt.plot(x, errD1_list, marker='o', mec='r', mfc='w',label='errD1')
    plt.plot(x, errD2_list, marker='*', ms=10, label='errD2')
    plt.plot(x, errG_total_list, marker='+', mec='g', mfc='w',label='errG_total')

    plt.legend()  # 让图例生效
    # plt.xticks(x, names, rotation=1)
    
    plt.xlabel('epoch') #X轴标签
    plt.ylabel("loss") #Y轴标签
    plt.title(name) #标题
    plt.savefig(path, dpi = 900)





if __name__ == "__main__":
    # test draw
    errD1_list = [1.2, 2.3, 3.5, 3.6, 2.9]
    errD2_list = [2.2, 3.6, 0.5, 2.5, 1.9]
    errG_total_list = [1.3, 3.5, 6.9, 4.2, 4.22]
    max_epoch = 5
    name = "test"
    path = "test_draw.jpg"

    draw_loss(errD1_list, errD2_list, errG_total_list, max_epoch, name, path)