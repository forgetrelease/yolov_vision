from torchvision.datasets import VOCDetection,VOCSegmentation
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from config import *
import torch
from utils.boxs_util import *
import os
import numpy as np
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Dict, Any
import collections
def load_data(custom_collate,image_set='trainval',detect_cls=VOCDetection):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = detect_cls(DATA_ROOT, download=True, year='2007', image_set=image_set,transform=transform,transforms=None,target_transform=None)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn=custom_collate)
    return data_loader

class ImageLabelDataset(Dataset):
    def __init__(self,image_set='trainval'):
        super().__init__()
        self.img_files = []
        cache_path = os.path.join(DATA_ROOT, '{}.cache'.format(image_set))
        cache_path = Path(cache_path)
        try:
            cache = np.load(cache_path, allow_pickle=True).item()
        except Exception:
            cache = self.cacheLabels(cache_path,image_set=image_set)
        self.labels = list(cache.values())
        self.img_files = list(cache.keys())
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        
        
        
    def cacheLabels(self,cache_path,image_set='trainval'):
        def custom_collate(batch):
            imgs = []
            targets = []
            original_images = []
            for sample in batch:
                annotation = sample[1]['annotation']
                size = (int(annotation['size']['width']), int(annotation['size']['height']))
                boxs = []
                for obj in annotation['object']:
                    bbox = obj['bndbox']
                    xmin, ymin, xmax, ymax = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])
                    xmin, ymin, xmax, ymax = normalize_box((xmin, ymin, xmax, ymax), size, IMAGE_SIZE)
                    # boxs.append({'name': obj['name'], 'bbox': [xmin, ymin, xmax, ymax]})
                    boxs.append([xmin, ymin, xmax, ymax, OBJ_INDEX[obj['name']]])
                imgs.append(sample[0])
                targets.append(boxs)
                original_images.append(annotation['filename'])
                
            imgs = torch.stack(imgs,dim=0)
            return imgs, orignal_boxs_to_tensor(targets), original_images
        loader = load_data(custom_collate,image_set=image_set)
        x = {}
        for _, target, images in loader:
            for i in range(target.size(dim=0)):
                labels = target[i,: ,: ,:]
                image_name = images[i]
                x[image_name] = labels.numpy()
        try:
            np.save(cache_path, x)
            cache_path.with_suffix('.cache.npy').rename(cache_path)
        except Exception as e:
            print(f"保存缓存失败:{e}")
        return x
    
        
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, index):
        image_path = self.img_files[index]
        image_path = os.path.join(DATA_ROOT, 'VOCdevkit','VOC2007','JPEGImages', image_path)
        img = Image.open(image_path).convert('RGB')
        return self.transform(img),torch.from_numpy(self.labels[index])

class SegmentDataset(Dataset):
    def __init__(self,image_set='trainval'):
        super(SegmentDataset).__init__()
        self.img_files = []
        cache_path = os.path.join(DATA_ROOT, '{}.cache'.format(image_set+'_seg'))
        cache_path = Path(cache_path)
        try:
            cache = np.load(cache_path, allow_pickle=True).item()
        except Exception:
            cache = self.cacheLabels(cache_path)
        self.labels = list(cache.values())
        self.img_files = list(cache.keys())
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE[0]),
            # transforms.Pad(padding=(0,0,100,0), fill=0, padding_mode='constant'),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE[0]),
            # transforms.Pad(padding=(0,0,100,0), fill=0, padding_mode='constant'),
        ])
        
    def cacheLabels(self,cache_path, image_set='trainval'):
        save_dir = os.path.dirname(cache_path)
        image_save_dir = os.path.join(save_dir, 'images.cache')
        if not os.path.exists(image_save_dir):
            os.mkdir(image_save_dir)
        mask_save_dir = os.path.join(save_dir, 'masks.cache')
        if not os.path.exists(mask_save_dir):
            os.mkdir(mask_save_dir)
        def custom_collate(batch):
            imgs = []
            masks = []
            targets = []
            original_images = []
            for (image, mask) in batch:
                masks.append(mask)
                imgs.append(image)
                file_name=  os.path.basename(mask.filename)
                idx = file_name.index('.')
                file_name = file_name[:idx]
                original_images.append(file_name)
                boxs = annotation_filename(file_name)
                targets.append(boxs)
                
            imgs = torch.stack(imgs,dim=0)
            return imgs, orignal_boxs_to_tensor(targets), original_images
        loader = load_data(custom_collate=custom_collate,image_set=image_set,detect_cls=VOCSegmentation)
        x = {}
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE[0]),
            # transforms.Pad(padding=(0,0,100,0), fill=0, padding_mode='constant'),
        ])
        for _, target, images in loader:
            for i in range(target.size(dim=0)):
                labels = target[i,: ,: ,:]
                image_name = images[i]
                x[image_name] = labels.numpy()
                mask_path = os.path.join(DATA_ROOT, 'VOCdevkit','VOC2007','SegmentationClass', image_name + '.png')
                image_path = os.path.join(DATA_ROOT, 'VOCdevkit','VOC2007','JPEGImages', image_name + '.jpg')
                try:
                    img_save = Path(os.path.join(image_save_dir, image_name + '.npy'))
                    mask_save = Path(os.path.join(mask_save_dir, image_name + '.npy'))
                    if os.path.exists(img_save) and os.path.exists(mask_save):
                        continue
                    img = Image.open(image_path).convert('RGB')
                    img = resize_image_(img, transform_image)
                    mask = Image.open(mask_path).convert('RGBA')
                    mask =  resize_mask_image(mask)
                    np.save(img_save, img.numpy())
                    np.save(mask_save, mask.numpy())
                    img = Image.open(image_path).convert('RGB')
                    img = resize_image_(img, transform_image)
                    mask = Image.open(mask_path).convert('RGBA')
                    mask =  resize_mask_image(mask)
                
                    np.save(img_save, img.numpy())
                    np.save(mask_save, mask.numpy())
                    # img_save.with_suffix('.cache.npy').rename(img_save)
                    # mask_save.with_suffix('.cache.npy').rename(mask_save)
                except Exception as e:
                    print(f"保存缓存失败:{e}")
                    
                
        try:
            np.save(cache_path, x)
            cache_path.with_suffix('.cache.npy').rename(cache_path)
        except Exception as e:
            print(f"保存缓存失败:{e}")
        return x
    def resize_image(self, image,mask=False):
        if mask:
            image = self.mask_transform(image)
        else:
            image = self.transform(image)
        w, h = image.shape[-1], image.shape[-2]
        padding = (0,0,h-w ,0) if w < h else (0,0,0,w-h)
        transform = transforms.Compose([
            transforms.Pad(padding=padding, fill=0, padding_mode='constant'),
            transforms.Resize(IMAGE_SIZE),
        ])
        return transform(image)
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, index):
        image_name = self.img_files[index]
        # mask_path = os.path.join(DATA_ROOT, 'VOCdevkit','VOC2007','SegmentationClass', image_path + '.png')
        # image_path = os.path.join(DATA_ROOT, 'VOCdevkit','VOC2007','JPEGImages', image_path + '.jpg')
        # img = Image.open(image_path).convert('RGB')
        # mask = Image.open(mask_path).convert('RGBA')
        # mask =  resize_mask_image(mask)
        image_path = os.path.join(DATA_ROOT, 'images.cache', image_name + '.npy')
        mask_path = os.path.join(DATA_ROOT, 'masks.cache', image_name + '.npy')
        mask =  np.load(mask_path)
        image = np.load(image_path)
        return torch.from_numpy(image),torch.from_numpy(self.labels[index]),torch.from_numpy(mask)



def parse_voc_annotation(xml_path):
    return xml_to_dic(ET.parse(xml_path).getroot())

def xml_to_dic(node):
    voc_dict: Dict[str, Any] = {}
    children = list(node)
    if children:
        def_dic: Dict[str, Any] = collections.defaultdict(list)
        for dc in map(xml_to_dic, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == "annotation":
            def_dic["object"] = [def_dic["object"]]
        voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict

def annotation_filename(file_name):
    xml_file = os.path.join(DATA_ROOT, 'VOCdevkit','VOC2007','Annotations', file_name + '.xml')
    obj = parse_voc_annotation(xml_file)
    annotation = obj['annotation']
    size = (int(annotation['size']['width']), int(annotation['size']['height']))
    boxs = []
    for obj in annotation['object']:
        bbox = obj['bndbox']
        xmin, ymin, xmax, ymax = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])
        xmin, ymin, xmax, ymax = normalize_box((xmin, ymin, xmax, ymax), size, IMAGE_SIZE)
        # boxs.append({'name': obj['name'], 'bbox': [xmin, ymin, xmax, ymax]})
        boxs.append([xmin, ymin, xmax, ymax, OBJ_INDEX[obj['name']]])
    return boxs


def resize_mask_image(image):
    # 先分出类型
    masks = mask_channel_cls(image)
    mask_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE[0]),
    ])
    for key in masks.keys():
        cls_mask = masks[key]
        cls_channel = mask_transform(cls_mask)
        w, h = cls_channel.shape[-1], cls_channel.shape[-2]
        padding = (0,0,h-w ,0) if w < h else (0,0,0,w-h)
        transform = transforms.Compose([
            transforms.Pad(padding=padding, fill=0, padding_mode='constant'),
            transforms.Resize(IMAGE_SIZE),
        ])
        cls_channel = transform(cls_channel)
        cls_channel[cls_channel > 0]=int(key)
        masks[key] = cls_channel
    keys = list(masks.keys())
    if len(keys) == 0:
        return masks
    temp = masks[keys.pop(0)]
    if len(keys) == 0:
        return temp
    for key in keys:
        channel =  masks[key]
        for col in range(channel.size(dim=1)):
            for row in range(channel.size(dim=2)):
                if channel[0,col,row] == 0:
                    continue
                temp[0,col,row] = channel[0,col,row]

    return temp
def resize_image_(image,transform_image):
    image = transform_image(image)
    w, h = image.shape[-1], image.shape[-2]
    padding = (0,0,h-w ,0) if w < h else (0,0,0,w-h)
    transform = transforms.Compose([
        transforms.Pad(padding=padding, fill=0, padding_mode='constant'),
        transforms.Resize(IMAGE_SIZE),
    ])
    return transform(image)
