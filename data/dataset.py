# coco2017地址：/home/zjj/data/coco

'''
homepage:https://cocodataset.org/#format-data

coco
    `-- |-- annotations
        |   |-- instances_train2017.json
        |   |-- instances_val2017.json
        |   |-- image_info_test-dev2017.json
        |   |-- image_info_test2017.json
        |--- train2017：图片名称为id
        |--- val2017
        `--- test2017

instancs_xxx.json:
dict_keys(['info', 'licenses', 'images', 'annotations', 'categories']

info{
"year": int, "version": str, "description": str, "contributor": str, "url": str, "date_created": datetime,
}

license{
"id": int, "name": str, "url": str,
}

image{
"id": int, "width": int, "height": int, "file_name": str, "license": int, "flickr_url": str, "coco_url": str, "date_captured": datetime,
}


annotation{
"id": int, "image_id": int, "category_id": int, "segmentation": RLE or [polygon], "area": float, "bbox": [x,y,width,height], "iscrowd": 0 or 1,
}
bbox:[左上角横坐标、左上角纵坐标、宽度、高度，单位为像素]

categories[{
"id": int, "name": str, "supercategory": str,
}]

'''

import os
import sys
import json
import PIL
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import trange
from collections import defaultdict
from torchvision import transforms

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utils.data import imagenet_preprocess, Resize

class CocoDataset(Dataset):
    def __init__(self, image_dir, instances_json, normalize_image=False, image_size=(64, 64),
                min_object_size=0.02, min_objects_per_image=3, max_objects_per_image=8, max_samples=None):
        super(Dataset, self).__init__()
        '''
        input:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a JSON file giving COCO annotations
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - normalize_image: If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        '''
        self.image_dir = image_dir
        self.normalize_image = normalize_image
        self.set_image_size(image_size)
        self.max_samples = max_samples

        with open(instances_json, 'r') as f:
            instances_data = json.load(f)
        
        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)
        
        self.vocab = {
            'object_name_to_idx': {},
            # 'pred_name_to_idx': {},
        }
        object_idx_to_name = {}
        all_instance_categories = []
        for category_data in instances_data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            # category_supercategory = category_data['supercategory']
            all_instance_categories.append(category_name)
            object_idx_to_name[category_id] = category_name
            self.vocab['object_name_to_idx'][category_name] = category_id
        
        # Add object data from instances
        self.image_id_to_objects = defaultdict(list)
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            lefttop_x, lefttop_y, w, h = object_data['bbox']
            W, H = self.image_id_to_size[image_id]
            box_area = (w * h) / (W * H)
            box_ok = box_area > min_object_size
            object_name = object_idx_to_name[object_data['category_id']]
            # category_ok = object_name in category_whitelist
            other_ok = object_name != 'other'
            if box_ok and other_ok:
                self.image_id_to_objects[image_id].append(object_data)
        
        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = 0

        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name
        self.num_objects = len(self.vocab['object_idx_to_name'])

        # Prune images that have too few or too many objects
        new_image_ids = []
        total_objs = 0
        for image_id in self.image_ids:
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids

    def set_image_size(self, image_size):
        print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_image:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size
    
    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):
            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def __len__(self):
        if self.max_samples is None:
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)

    def __getitem__(self, index):
        '''
        Get the pixels of an image, and its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box. 

        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        '''
        image_id = self.image_ids[index]

        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)
        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))
        
        H, W = self.image_size
        objs, boxes = [], []
        image_name = filename
        for object_data in self.image_id_to_objects[image_id]:
            objs.append(object_data['category_id'])
            x, y, w, h = object_data['bbox']
            x0 = x / WW
            y0 = y / HH
            x1 = (x + w) / WW
            y1 = (y + h) / HH
            boxes.append(torch.FloatTensor([x0, y0, x1, y1]))
        
        # shuffle objs
        O = len(objs)
        rand_idx = list(range(O))
        random.shuffle(rand_idx)

        objs = [objs[i] for i in rand_idx]
        boxes = [boxes[i] for i in rand_idx]

        objs = torch.LongTensor(objs)
        boxes = torch.stack(boxes, dim=0)

        return image, objs, boxes, image_name

def coco_collate_fn(batch):
    """
    Collate function to be used when wrapping CocoDataset in a DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving object categories
    - boxes: FloatTensor of shape (O, 4)
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    """

    all_imgs, all_objs, all_boxes, all_obj_to_img, all_imgs_name = [], [], [], [], []
    for i, (img, objs, boxes, image_name) in enumerate(batch):
        all_imgs.append(img[None])
        O = objs.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_obj_to_img.append(torch.LongTensor(O).fill_(i))
        all_imgs_name.append(image_name)

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_obj_to_img = torch.cat(all_obj_to_img)

    out = (all_imgs, all_objs, all_boxes, all_obj_to_img, all_imgs_name)

    return out




def get_dataloader(batch_size=10, COCO_DIR='/home/zjj/data/coco', shuffle_val=False):
    coco_train_image_dir = os.path.join(COCO_DIR, 'train2017')
    coco_val_image_dir = os.path.join(COCO_DIR, 'val2017')
    coco_train_instances_json = os.path.join(COCO_DIR, 'annotations/instances_train2017.json')
    coco_val_instances_json = os.path.join(COCO_DIR, 'annotations/instances_val2017.json')


    # build datasets
    train_dset_kwargs = {
        'image_dir': coco_train_image_dir,
        'instances_json': coco_train_instances_json,
        'normalize_image': False,
        'min_objects_per_image': 2, 
        'max_objects_per_image': 8,
        'image_size': (256, 256),
    }

    train_dset = CocoDataset(**train_dset_kwargs)
    # print(train_dset.__len__())
    # print(train_dset.__getitem__(0))
    print('Training dataset has %d images and %d objects' % (len(train_dset), train_dset.total_objects()))
    print('(%.2f objects per image)' % (float(train_dset.total_objects() / len(train_dset))))

    val_dset_kwargs = {
        'image_dir': coco_val_image_dir,
        'instances_json': coco_val_instances_json,
        'normalize_image': False,
        'min_objects_per_image': 2, 
        'max_objects_per_image': 8,
        'image_size': (256, 256),
    }
    val_dset = CocoDataset(**val_dset_kwargs)
    print('Validating dataset has %d images and %d objects' % (len(val_dset), val_dset.total_objects()))
    print('(%.2f objects per image)' % (float(val_dset.total_objects() / len(val_dset))))

    assert train_dset.vocab == val_dset.vocab

    # print(train_dset.vocab)
    vocab = json.loads(json.dumps(train_dset.vocab))


    # build dataloader
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 4,
        'shuffle': True,
        'collate_fn': coco_collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = shuffle_val
    loader_kwargs['num_workers'] = 1
    val_loader = DataLoader(val_dset, **loader_kwargs)

    return train_loader, val_loader



def testcode():
    with open('/home/zjj/data/coco/annotations/instances_val2017.json','r',encoding='utf8')as fp:
        instances_data = json.load(fp)
        print(instances_data.keys())
        for image_data in instances_data['images']:
            image_id = image_data['id']
            filename = image_data['file_name']
            width = image_data['width']
            height = image_data['height']
            break

        for category_data in instances_data['categories']:
            category_id = category_data['id']
            category_name = category_data['name']
            category_supercategory = category_data['supercategory']
            print(category_id, category_name, category_supercategory)
            break
        
        for object_data in instances_data['annotations']:
            image_id = object_data['image_id']
            lefttop_x, lefttop_y, w, h = object_data['bbox']
            # W, H = self.image_id_to_size[image_id]
            # box_area = (w * h) / (W * H)
            print(image_id, lefttop_x, lefttop_y, w, h)
            break


def save_image(tensor, name):
    # loader使用torchvision中自带的transforms函数
    loader = transforms.Compose([
        transforms.ToTensor()])  

    unloader = transforms.ToPILImage()

    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save(name)


def batch_mask_image(image, batch_size, boxes, obj_to_img, image_name):
    mask_images = []
    image_num = batch_size
    # 测试mask，生成一个mask，与图像做点乘
    # 获得当前某张img中包含的box,随机选择某个box进行mask
    obj2img_list = obj_to_img.cpu().numpy().tolist()
    for im in range(image_num):
        index1 = obj2img_list.index(im)
        index2 = obj2img_list[::-1].index(im)
        obj_list = obj2img_list[index1:-index2] if index2 != 0 else obj2img_list[index1:]
        box_i = np.random.randint(0, len(obj_list))
        mask = torch.ones(256, 256)
        order = index1 + box_i
        xs = int((image[im, :, :, :].size()[-2]) * boxes[order][0])
        xd = int((image[im, :, :, :].size()[-2]) * boxes[order][2])
        ys = int((image[im, :, :, :].size()[-1]) * boxes[order][1])
        yd = int((image[im, :, :, :].size()[-1]) * boxes[order][3])
        mask[ys:yd, xs:xd] = 0
        mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        mask = torch.repeat_interleave(mask, repeats=3, dim=1)
        mask_img = torch.mul(mask, image[im, :, :, :])
        mask_images.append(mask_img)
        imgid = image_name[im].split(".jpg")[0]
        mask_img_path = os.path.join("/home/zjj/diverse-image-synthesis/train_mask_image", imgid + "_mask_" + str(box_i)+".jpg")
        if not os.path.isfile(mask_img_path):
            save_image(mask_img, mask_img_path)
    mask_images = torch.cat(mask_images)
    return mask_images


if __name__ == "__main__":    
    # test reading data
    train_dataloader, val_dataloader = get_dataloader(batch_size=4)
    '''
    called set_image_size (256, 256)
    Training dataset has 77446 images and 273214 objects
    (3.53 objects per image)
    called set_image_size (256, 256)
    Validating dataset has 3228 images and 11461 objects
    (3.55 objects per image)
    '''
    for i, batch in enumerate(train_dataloader):
        image, objs, boxes, obj_to_img, image_name = batch
        # print(image)
        print(image.size()) # torch.Size([batchsize, 3, 256, 256])
        print("====================================================================")
        print(objs.size()) # torch.Size([?])
        print(boxes.size()) # torch.Size([?,4])
        print(obj_to_img.size()) # torch.Size([?])
        assert image.size()[0] == len(image_name)
        batch_size = image.size()[0]
        if batch_size > 1:
            for i in range(batch_size):
                raw_img_path = os.path.join("/home/zjj/diverse-image-synthesis/train_raw_image", image_name[i])
                if not os.path.isfile(raw_img_path):
                    save_image(image[i, :, :, :], raw_img_path)
        break
        # exit()
        mask_imgs = batch_mask_image(image, batch_size, boxes, obj_to_img, image_name)
        break