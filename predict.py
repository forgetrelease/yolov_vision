
import torch
from vision.model.models import ImageMaskNet, ImageResNet
from config import LEARNING_RATE,IMAGE_SIZE
from PIL import Image
from torchvision import transforms
from utils.boxs_util import single_image, plot_boxs, show_boxs
from utils.dataset import resize_image_, annotation_filename, resize_image_mask_target
import os
from utils.vision import show_box_masks
import sys

def pred_mask(images_path):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageMaskNet().to(device)
    
    try:
        model.load_state_dict(torch.load('/Users/chunsheng/Downloads/best-mask.pth', map_location=device))
    except Exception as e:
        print("加载模型出错", e)
        return
    img = Image.open(images_path).convert('RGB')
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE[0]),
        # transforms.Pad(padding=(0,0,100,0), fill=0, padding_mode='constant'),
    ])
    
    
    img = resize_image_(img,transform_image=transform_image)
    img = img.to(device)
    img = img.unsqueeze(0)
    
    
    results, masks = model(img)
    # print(results)
    # print(masks)
    masks[masks<0] = 0
    plot_boxs(img, results,save=True)
    # single_image(masks[0, :, :, :])
    
def pred_box(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageResNet().to(device)
    
    try:
        patams = torch.load('best.pth', map_location=device)
        model.load_state_dict(torch.load('final.pth', map_location=device))
    except Exception as e:
        print("加载模型出错", e)
        return
    img = Image.open(image_path).convert('RGB')
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(IMAGE_SIZE[0]),
        # transforms.Pad(padding=(0,0,100,0), fill=0, padding_mode='constant'),
    ])
    
    
    img, target, masks = resize_image_mask_target(transform_image(img), None, None)
    img = img.to(device)
    img = img.unsqueeze(0)
    
    
    results = model(img)
    image = show_box_masks(img[0,:,:,:], results[0,:,:,:])
    image.save('./results/00000.jpg')
    
    
if __name__ == '__main__':
    args = sys.argv
    img = "./data/VOCdevkit/VOC2007/JPEGImages/"
    if len(args) > 1:
        img = "./data/VOCdevkit/VOC2007/JPEGImages/" + args[1]
    if os.path.exists(img):
        pred_box(img)
    else:
    # 加载模型
        images_root = './data/VOCdevkit/VOC2007/JPEGImages'
        for root, folders, files in os.walk(images_root):
            for file in files:
                if file.endswith('.jpg'):
                    # print(file)
                    pred_box(os.path.join(root, file))
                