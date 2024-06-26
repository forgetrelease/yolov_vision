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
            os.makedirs(image_save_dir)
        mask_save_dir = os.path.join(save_dir, 'masks.cache')
        if not os.path.exists(mask_save_dir):
            os.makedirs(mask_save_dir)
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

def annotation_filename(file_name,resize=IMAGE_SIZE):
    xml_file = os.path.join(DATA_ROOT, 'VOCdevkit','VOC2007','Annotations', file_name + '.xml')
    obj = parse_voc_annotation(xml_file)
    annotation = obj['annotation']
    size = (int(annotation['size']['width']), int(annotation['size']['height']))
    boxs = []
    for obj in annotation['object']:
        bbox = obj['bndbox']
        xmin, ymin, xmax, ymax = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])
        xmin, ymin, xmax, ymax = normalize_box_mask_scale((xmin, ymin, xmax, ymax), size, IMAGE_SIZE)
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

############## 重构部分 ##############
def box_from_annotation(file_name):
    xml_file = os.path.join(DATA_ROOT, 'VOCdevkit','VOC2007','Annotations', file_name + '.xml')
    obj = parse_voc_annotation(xml_file)
    annotation = obj['annotation']
    size = (int(annotation['size']['width']), int(annotation['size']['height']))
    boxs = []
    for obj in annotation['object']:
        bbox = obj['bndbox']
        xmin, ymin, xmax, ymax = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])
        boxs.append([xmin, ymin, xmax, ymax, OBJ_INDEX[obj['name']]])
    return boxs, size
def annotation_from_box(image_size, boxs):
    w,h = image_size
    width, height = IMAGE_SIZE
    xy_rate =  width/w if w > h else height/h
    
    grid_w = width / 7
    grid_h = height / 7
    box_cell = {}
    rectangle_truth = torch.zeros((7, 7, 30))
    for box in boxs:
        xmin, ymin, xmax, ymax = box[:4]
        xmin, ymin, xmax, ymax = xmin*xy_rate, ymin*xy_rate, xmax*xy_rate, ymax*xy_rate
        centx = (xmin + xmax) / 2
        centy = (ymin + ymax) / 2
        col = int(centx // grid_w)
        row = int(centy // grid_h)
        if 0 <= col < 7 and 0 <= row < 7:
            box_idx = box_cell.get((row, col), 0)
            if box_idx >= 2:
                continue
            cls_idx = torch.zeros(20)
            cls_idx[box[4]] = 1.0
            rectangle_truth[row, col, :20] = cls_idx
            box_truth = ((centx - col * grid_w) / grid_w,
                    (centy - row * grid_h) / grid_h, 
                    (xmax - xmin)  / grid_w,
                    (ymax - ymin)  / grid_h,
                    1.0
            )
            start_idx = 20 + box_idx * 5
            rectangle_truth[row, col, start_idx:start_idx+5] = torch.tensor(box_truth)
            box_cell[(row, col)] = box_idx + 1
    return rectangle_truth
            
        
    
    
    
def resize_image_mask_target(image, target, mask=None, rgb_map=None):
    # 调整图片大小
    w, h = image.shape[-1], image.shape[-2]
    # w, h = target[0][5]
    padding = (0,0,h-w ,0) if w < h else (0,0,0,w-h)
    transform = transforms.Compose([
        # 首先填充成正方形
        transforms.Pad(padding=padding, fill=0, padding_mode='constant'),
        # resize 标准大小
        transforms.Resize(IMAGE_SIZE),
    ])
    # 最短边补充0到 IMAGE_SIZE[0]
    image = transform(image)
    if target:
        target = annotation_from_box((w,h), target)
    mask_data = None
    if mask is not None:
        transform_mask = transforms.Compose([
            transforms.Pad(padding=padding, fill=0, padding_mode='constant'),
        ])
        # 首先填充成正方形
        mask = transforms.ToTensor()(mask)
        mask = transform_mask(mask)
        channels = torch.zeros(rgb_map.shape[-1] ,mask.shape[-2],mask.shape[-1])
        # rgbs = torch.sum(mask, dim=0,keepdim=True) # 1, h, w
        for i in range(rgb_map.shape[-1]):
            rrgb = rgb_map[i]
            temp = torch.zeros_like(mask)
            temp[mask==rrgb] = 1.0
            channels[i,:,:] = temp[0, :, :]
        
        # 调整mask大小
        # 一个类一层
        scal_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
        ])
        channels = scal_transform(channels)
        cls_count = rgb_map.shape[-1]
        mask_data = torch.zeros(cls_count + 1 ,IMAGE_SIZE[-2],IMAGE_SIZE[-1])
        # 调整mask大小
        for i in range(channels.shape[0]):
            rgb = channels[i,:,:]
            temp = torch.zeros_like(rgb)
            temp[rgb>0] = rgb_map[i]
            mask_data[i,:,:] = temp
        confidence = torch.zeros(IMAGE_SIZE[-2],IMAGE_SIZE[-1])
        confidence_mask = torch.sum(mask_data, dim=0)
        confidence[confidence_mask>0] = 1.0
        mask_data[cls_count,:,:] = confidence
            
    return image, target, mask_data
def read_local_mask(mask_file):
    mask = Image.open(mask_file).convert('RGB')
    w,h = mask.size
    image_data = mask.load()
    chennales = {}
    for y in range(h):
        for x in range(w):
            r,g,b = image_data[x,y][:3]
            rgb = r + g + b
            if rgb in [0, 640]:
                continue
            key = str(rgb)
            idx = chennales.get(key, torch.zeros(w*h).reshape([1,h,w]))
            idx[0, y, x] = rgb
            chennales[key] = idx
    # temp = {}
    # for k in chennales.keys():
    #     key = float(k)/(255*3)
    #     temp[key] = chennales[k]
    # return temp
    return chennales
    
class DetectBase(Dataset):
    def __init__(self, source):
        super(DetectBase).__init__()
        self.image_files = []
        for root, folders, files in os.walk(source):
            for file in files:
                if file.endswith('.npy'):
                    self.image_files.append(os.path.join(root,file))
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            
        ])
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = np.load(image_file,allow_pickle=True).item()
        return self.format_ceche(image)
    def format_ceche(self, cache):
        return torch.from_numpy(cache)
    

class BoxDetect(DetectBase):
    def format_ceche(self, cache):
        image = cache['image']
        target = cache['target']
        return torch.from_numpy(image), torch.from_numpy(target)
    @classmethod
    def prepare_voc_data(self, save_dir, image_set='trainval'):
        save_dir = os.path.join(save_dir, 'box.cache', image_set)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        def custom_collate(batch):
            imgs = []
            targets = []
            original_images = []
            for (image, info) in batch:
                annotation = info['annotation']
                # if int(annotation['segmented']) == 0:
                #     continue
                imgs.append(image)
                file_name=  annotation['filename']
                idx = file_name.index('.')
                file_name = file_name[:idx]
                original_images.append(file_name)
                size = (int(annotation['size']['width']), int(annotation['size']['height']))
                boxs = []
                for obj in annotation['object']:
                    bbox = obj['bndbox']
                    xmin, ymin, xmax, ymax = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])
                    boxs.append([xmin, ymin, xmax, ymax, OBJ_INDEX[obj['name']], size])
               
                targets.append(boxs)
                
            # imgs = torch.stack(imgs,dim=0)
            return imgs, targets, original_images
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = VOCDetection(DATA_ROOT, download=True, year='2007', image_set=image_set,transform=transform,transforms=None,target_transform=None)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn=custom_collate)
        for images, target, image_names in data_loader:
            for i in range(len(target)):
                x = {}
                labels = target[i]
                image_name = image_names[i]
                image = images[i]
                # x[image_name] = labels.numpy()
                # mask_path = os.path.join(DATA_ROOT, 'VOCdevkit','VOC2007','SegmentationClass', image_name + '.png')
                # image_path = os.path.join(DATA_ROOT, 'VOCdevkit','VOC2007','JPEGImages', image_name + '.jpg')
                try:
                    img_save = Path(os.path.join(save_dir, image_name + '.npy'))
                    if os.path.exists(img_save):
                        continue
                    # img = Image.open(image_path).convert('RGB')
                    image, labels,_ = resize_image_mask_target(image=image, target=labels)
                    x['image'] = image.numpy()
                    x['target'] = labels.numpy()
                    np.save(img_save, x)
                except Exception as e:
                    print(f"保存缓存失败:{e}")
        return save_dir
    
class MaskDetect(DetectBase):
    def format_ceche(self, cache):
        image = cache['image']
        target = cache['target']
        mask = cache['mask']
        return torch.from_numpy(image), torch.from_numpy(target), torch.from_numpy(mask)
    @classmethod
    def prepare_voc_data(self, save_dir, image_set='trainval'):
        save_dir = os.path.join(save_dir, 'box-mask.cache', image_set)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        rgb_map_num = np.load("./rgbs.npy")
        rgb_map = torch.from_numpy(rgb_map_num)
        def custom_collate(batch):
            imgs = []
            targets = []
            original_images = []
            for (image, info) in batch:
                annotation = info['annotation']
                if int(annotation['segmented']) == 0:
                    continue
                imgs.append(image)
                file_name=  annotation['filename']
                idx = file_name.index('.')
                file_name = file_name[:idx]
                original_images.append(file_name)
                size = (int(annotation['size']['width']), int(annotation['size']['height']))
                boxs = []
                for obj in annotation['object']:
                    bbox = obj['bndbox']
                    xmin, ymin, xmax, ymax = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])
                    boxs.append([xmin, ymin, xmax, ymax, OBJ_INDEX[obj['name']], size])
               
                targets.append(boxs)
                
            # imgs = torch.stack(imgs,dim=0)
            return imgs, targets, original_images
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = VOCDetection(DATA_ROOT, download=True, year='2007', image_set=image_set,transform=transform,transforms=None,target_transform=None)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn=custom_collate)
        for images, target, image_names in data_loader:
            for i in range(len(target)):
                x = {}
                labels = target[i]
                image_name = image_names[i]
                image = images[i]
                # x[image_name] = labels.numpy()
                mask_path = os.path.join(DATA_ROOT, 'VOCdevkit','VOC2007','SegmentationClass', image_name + '.png')
                # image_path = os.path.join(DATA_ROOT, 'VOCdevkit','VOC2007','JPEGImages', image_name + '.jpg')
                try:
                    img_save = Path(os.path.join(save_dir, image_name + '.npy'))
                    if os.path.exists(img_save):
                        continue
                    
                    mask = Image.open(mask_path)
                    image, labels, masks = resize_image_mask_target(image=image, target=labels, mask=mask,rgb_map=rgb_map)
                    x['image'] = image.numpy()
                    x['target'] = labels.numpy()
                    x['mask'] = masks.numpy()
                    np.save(img_save, x)
                except Exception as e:
                    print(f"保存缓存失败:{e}")
        return save_dir
    
    