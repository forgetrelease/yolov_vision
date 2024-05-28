from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from config import *
import torch
from utils.boxs_util import *
import os
import numpy as np
from PIL import Image
from pathlib import Path

def load_data(image_set='trainval'):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    def custom_collate(batch):
        imgs = []
        targets = []
        original_images = []
        for sample in batch:
            annotation = sample[1]['annotation']
            if annotation['segmented'] == 0:
                continue
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
    dataset = VOCDetection(DATA_ROOT, download=True, year='2007', image_set=image_set,transform=transform,transforms=None,target_transform=None)
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
            cache = self.cacheLabels(cache_path)
        self.labels = list(cache.values())
        self.img_files = list(cache.keys())
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        
        
        
    def cacheLabels(self,cache_path):
        loader = load_data()
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
        
    