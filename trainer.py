import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import transforms

import os
import random
import time
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 
from collections import OrderedDict
from miscc.utils import weights_init, load_params, copy_G_params, mkdir_p
from miscc.config import cfg
import torchvision.transforms as T
from utils.data import imagenet_preprocess, Resize

from model import CondGenerator, VGGFeatureExtractor, VGG16FeatureExtractor2
from loss import generator_loss1
from detection import resnet50, yoloLoss
from dataset import get_dataloader

from dataset import save_image
from utils.visualizer import Visualizer
from utils.iter_counter import IterationCounter

# 限制使用的GPU个数
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,4,6'

# create tool for visualization
visualizer = Visualizer(cfg)

class CondGANTrainer(object):
    def __init__(self, output_dir, data_loader):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            # self.loss_dir = os.path.join(output_dir, 'Loss')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            # mkdir_p(self.loss_dir)

        cudnn.benchmark = True
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

        # create tool for counting iterations
        self.iter_counter = IterationCounter(cfg, len(data_loader))

    def build_models(self):
        # vgg extractor
        vgg_single = VGGFeatureExtractor()
        vgg_muti = VGG16FeatureExtractor2()

        # generator
        netG = CondGenerator()

        # discriminator
        netD_detection = resnet50()

        model_list = [netG, netD_detection]
        for model in model_list:
            model.apply(weights_init)

        epoch = 0

        '''
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
        '''
        
        
        return [netG, netD_detection, vgg_single, vgg_muti, epoch]

    def define_optimizers(self, netG, netD):
        optimizerG = optim.Adam(netG.parameters(), lr=cfg.TRAIN.GENERATOR_LR, betas=(0.5, 0.999))
        
        optimizerD = optim.Adam(netD.parameters(), lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))

        return optimizerG, optimizerD
    
    def save_model(self, netG, avg_param_G, netD, epoch):
        backup_para = copy_G_params(netG)
        load_params(netG, avg_param_G)
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.model_dir, epoch))
        load_params(netG, backup_para)
        
        torch.save(netD.state_dict(), '%s/netD_detection.pth' % (self.model_dir))
        # torch.save(netD2.state_dict(), '%s/netD2.pth' % (self.model_dir))
        print('Save G/Ds models.')
    
    def set_requires_grad_value(self, models_list, brequires):
        for i in range(len(models_list)):
            for p in models_list[i].parameters():
                p.requires_grad = brequires

    def save_img_results1(self, real_imgs, gen_iterations, image_names, epoch):
        for i in range(len(real_imgs)):
            name = (image_names[i].split('.'))[0]
            fullpath = '%s/real_%s_%d_epoch_%d.png' % (self.image_dir, name, gen_iterations, epoch)
            save_image(real_imgs[i], fullpath)

    def save_img_results2(self, fake_imgs, gen_iterations, image_names, epoch):
        for i in range(len(fake_imgs)):
            name = (image_names[i].split('.'))[0]
            fullpath = '%s/fake_%s_%d_epoch_%d.png' % (self.image_dir, name, gen_iterations, epoch)
            save_image(fake_imgs[i], fullpath)

    
    def train(self):
        netG, netD_detection, vgg_single, vgg_muti, start_epoch = self.build_models()
        netG = netG.cuda()
        netG = nn.DataParallel(netG, device_ids = cfg.DEVICE_IDS)
        netD_detection = netD_detection.cuda()
        netD_detection = nn.DataParallel(netD_detection, device_ids = cfg.DEVICE_IDS)
        vgg_single = vgg_single.cuda()
        vgg_single = nn.DataParallel(vgg_single, device_ids = cfg.DEVICE_IDS)
        vgg_muti = vgg_muti.cuda()
        vgg_muti = nn.DataParallel(vgg_muti, device_ids = cfg.DEVICE_IDS)
  
        
        avg_param_G = copy_G_params(netG)
        optimizerG, optimizerD = self.define_optimizers(netG, netD_detection)

        gen_iterations = 0
        
        discriminator_criterion = yoloLoss(4, 2, 5, 0.5)
        for epoch in range(start_epoch, self.max_epoch):
            self.iter_counter.record_epoch_start(epoch)
            for i, data in enumerate(self.data_loader, start=self.iter_counter.epoch_iter):
                self.iter_counter.record_one_iteration()
                error_dict = {}
                images, masked_images, mask_patchs, mask_obj_boxes, mask_obj_classes, image_names, \
                    masks, yolo_targets = data

                images = images.cuda()
                masked_images = masked_images.cuda()
                mask_patchs = mask_patchs.cuda()
                mask_obj_boxes = mask_obj_boxes.cuda()
                mask_obj_classes = mask_obj_classes.cuda()
                masks = masks.cuda()
                yolo_targets = yolo_targets.cuda()

                assert images.size()[0] == len(image_names)
                self.batch_size = images.size()[0]

                # generate fake images
                visual_feat = vgg_single(mask_patchs)
                fake_images = netG(masked_images, masks, visual_feat, mask_obj_classes)

                # update detection discriminator
                netD_detection.zero_grad()
                yolo_preds = netD_detection(fake_images)
                discriminator_loss = discriminator_criterion(yolo_preds, yolo_targets)
                discriminator_loss.backward(retain_graph=True)
                optimizerD.step()
                error_dict['errD'] = discriminator_loss.item()
                # D_logs = 'errD: %.2f ' % (discriminator_loss.item())

                # update generator
                real_feats = vgg_muti(images)
                fake_feats = vgg_muti(fake_images)
                errG1 = generator_loss1(images, fake_images, real_feats, fake_feats, mask=masks)
                errG2 = 0 * discriminator_loss
                errG = errG1 + errG2
                netG.zero_grad()
                errG.backward()
                optimizerG.step()
                # G_logs = "errG_adversial: %.2f, errG_other: %.2f" % (errG2.item(), errG1.item())
                error_dict['errG1'] = errG1.item()
                error_dict['errG2'] = errG2.item()
        

                for p, avg_p in zip(netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                if gen_iterations % 10 == 0:
                    cur_t = time.time()
                    usetime = cur_t - self.iter_counter.epoch_start_time
                    visualizer.print_current_errors(epoch, gen_iterations, error_dict, usetime)
                    # print("gen_iterations: ", gen_iterations, D_logs + ',' + G_logs)
                
                # save images
                if gen_iterations % 100 == 0:
                    backup_para = copy_G_params(netG)
                    load_params(netG, avg_param_G)
                    self.save_img_results2(fake_images[:10], gen_iterations, image_names[:10], epoch)
                    self.save_img_results1(images[:10], gen_iterations, image_names[:10], epoch)
                    load_params(netG, backup_para)
                
                gen_iterations += 1

                # Visualizations
                # visualizer.print_current_errors(epoch, self.iter_counter.epoch_iter, losses, self.iter_counter.time_per_iter)
                visualizer.plot_current_errors(error_dict, self.iter_counter.total_steps_so_far)

                visuals = OrderedDict([('raw_image', images),
                                       ('input_image(masked)', masked_images),
                                       ('generated_image', fake_images)])
                visualizer.display_current_results(visuals, epoch, self.iter_counter.total_steps_so_far)

            end_t = time.time()
            self.iter_counter.record_epoch_end()
            print('''[%d/%d][%d]Loss_D: %.2f \t Loss_G: %.2f \t Time: %.2fs'''
                  % (epoch, self.max_epoch, self.num_batches,
                     discriminator_loss.item(), errG.item(), end_t - self.iter_counter.epoch_start_time))

            if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:  # and epoch != 0:
                self.save_model(netG, avg_param_G, netD_detection, epoch)
                curtime = time.time()
                print('End of epoch [%d/%d] \t Total Time Taken: %d sec' %
                      (self.current_epoch, self.total_epochs, curtime - self.iter_counter.epoch_start_time))


        self.save_model(netG, avg_param_G, netD_detection, self.max_epoch)






if __name__ == "__main__":
    pass