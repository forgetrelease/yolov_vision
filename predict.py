
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
from utils.dataset import BoxDetect, MaskDetect
from config import *
from tqdm import tqdm
from torch.utils.data import DataLoader

def pred_mask(images_path):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageMaskNet().to(device)
    
    try:
        model.load_state_dict(torch.load('/Users/chunsheng/Downloads/final-mask.pth', map_location=device))
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
    
def pred_box(image_path,model):
    image = Image.open(image_path).convert('RGB')
    transform_image = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    
    img, target, masks = resize_image_mask_target(transform_image(image), None, None)
    img = torch.unsqueeze(img,dim=0)
    
    with torch.no_grad():
        results = model(img)
    image = show_box_masks(image, results[0,:,:,:])
    return image
def predict(model):
    train_data_set = MaskDetect('./data/box-mask.cache/trainval')
    train_data_loader = DataLoader(
        train_data_set,
        batch_size=BATCH_SIZE,
        num_workers=1,
        persistent_workers=True,
        drop_last=True,
        shuffle=True
        )
    for image, target, mask in tqdm(train_data_loader, desc='Predict', leave=False):
        image = image.to(device)
        # target = target.to(device)
        with torch.no_grad():
            result, mask_result = model(image)
            for i in range(BATCH_SIZE):
                img = show_box_masks(image[i,:,:,:], result[i,:,:,:])
                img.show()
def predict_images(model, source):
    save_dir = os.path.dirname(DATA_ROOT)
    save_dir = os.path.join(save_dir, 'results')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir,  'tasks')
    idx = 0
    while os.path.exists(save_dir):
        save_dir = os.path.join(os.path.dirname(save_dir), f'tasks{idx}')
        idx += 1
    os.mkdir(save_dir)
    if os.path.isdir(source):
        for root, dirs, files in os.walk(source):
            for file in files:
                if file.endswith('.jpg'):
                    image = pred_box(os.path.join(root,file), model)
                    image.save(os.path.join(save_dir, file))
    if type(source) == list:
        for file in source:
            if os.path.isfile(file):
                image = pred_box(file, model)
                image.save(os.path.join(save_dir, os.path.basename(file)))        
    else:
        image = pred_box(source, model)
        image.save(os.path.join(save_dir, os.path.basename(source)))     
if __name__ == '__main__':
    args = sys.argv
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImageMaskNet().to(device)
    model.eval()
    try:
        model.load_state_dict(torch.load('/Users/chunsheng/Downloads/final-mask.pth', map_location=device))
    except Exception as e:
        print("加载模型出错", e)
        
    if len(args) > 1:
        predict_images(model, args[1])
    else:
        predict(model)
    
    
    
                