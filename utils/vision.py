
import torchvision.transforms as T
import torch
from config import *
from PIL import ImageDraw, ImageFont,Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.transforms as transforms

# 注意，0.0要保留，剔除白色1.0就可以了
# 因为后期对结果做softmax去过所有channel都是0，会出错
def parse_rgb_allImage(black_rgbs=[1.0]):
    all_rgbs = torch.Tensor(0)
    all_val=set()
    for root, dirs, files in os.walk('/Users/chunsheng/Desktop/workspace/pytorch_study/yolov_vision/data/VOCdevkit/VOC2007/SegmentationClass/'):
        for file in files:
            if file.endswith('.png'):
                image = Image.open(os.path.join(root, file)) #不要转RGB，否则1个channel变3个，image = image.convert('RGB')
                img = transforms.ToTensor()(image)
                rgbs = torch.sum(img, dim=0,keepdim=True)
                rgbs_list = rgbs.flatten()
                keys = torch.unique(rgbs_list.view(-1))
                all_rgbs = torch.cat((all_rgbs, keys), 0)
    unique_rgb = all_rgbs.unique()
    if len(black_rgbs) > 0:
        temp_rgb = unique_rgb.tolist()
        idx = 0
        mask  = torch.ones(unique_rgb.size(0), dtype=torch.bool)
        for rgb in temp_rgb:
            if rgb in black_rgbs:
                mask[idx] = False
            idx += 1
        unique_rgb = unique_rgb[mask]
    return unique_rgb

def show_box_masks(image, target, mask=None, color=(1,0,0)):
    width, height = IMAGE_SIZE
    grid_w = width / 7
    grid_h = height / 7
    
    all_boxs = []
    for col in range(7):
        for row in range(7):
            cls_idx = torch.argmax(target[row, col, :20]).item()
            for k in range(2):
                confidence = target[row, col, 20+k*5+4].item()
                if confidence > MIN_CONFIDENCE:
                    boxs = target[row, col, 20+k*5:20+(k+1)*5].tolist()
                    center_x, center_y, box_w, box_h, c = boxs
                    if box_w < 0 or box_h < 0:
                        continue
                    center_x = center_x * grid_w + col * grid_w
                    center_y = center_y * grid_h + row * grid_h
                    xmin = center_x - box_w * grid_w / 2
                    ymin = center_y - box_h * grid_h / 2
                    xmax = xmin + box_w * grid_w
                    ymax = ymin + box_h * grid_h
                    all_boxs.append([xmin, ymin, xmax, ymax, cls_idx, c])
                    
    hash_box = list(set([tuple(box) for box in all_boxs]))
    def class_of_index(idx):
        for k, v in OBJ_INDEX.items():
            if v == idx:
                return k
        return 'unknown'
    if type(image) == torch.Tensor:
        image = T.ToPILImage()(image)
    else:
        w, h = image.size
        width, height = IMAGE_SIZE
        xy_rate =  width/w if w > h else height/h
        temp_box = []
        for box in hash_box:
            xmin, ymin, xmax, ymax, cls_idx, c = box
            xmin, ymin, xmax, ymax = int(xmin/xy_rate), int(ymin/xy_rate), int(xmax/xy_rate), int(ymax/xy_rate)
            temp_box.append([xmin, ymin, xmax, ymax, cls_idx, c])
        hash_box = temp_box
        
    fill_color=(255, 0, 0)
    draw = ImageDraw.Draw(image)
    log_msg = []
    for box in hash_box:
        xmin, ymin, xmax, ymax, cls_idx, c = box
        text = f'{class_of_index(box[4])},{box[5]:.2f}'
        log_msg.append(f'{class_of_index(box[4])},{box[5]:.2f} at ({box[0]:.2f},{box[1]:.2f})-{box[2]:.2f},{box[3]:.2f}')
        draw.rectangle(box[:4], outline=fill_color)
        font=ImageFont.load_default()
        draw.text((box[0], box[1] - 11 if box[1] - 11 > 0 else box[1] + 1), text, fill=fill_color,font=font)
    del draw
    print(log_msg)
    if mask is not None:
        # mask 11,448,448
        rgb_map_num = np.load("./rgbs.npy")
        rgb_map = torch.from_numpy(rgb_map_num)#20
        count = rgb_map.shape[-1]
        rgb_map = rgb_map.unsqueeze(1).unsqueeze(1).repeat((1,IMAGE_SIZE[1],IMAGE_SIZE[0])).expand(count, IMAGE_SIZE[1],IMAGE_SIZE[0]) #11,448,448
        mask_image = torch.softmax(mask, dim=0)
        arg_max = torch.argmax(mask_image, dim=0)
        max_ones_idx = rgb_map.gather(dim=0,index=arg_max.unsqueeze(0).expand_as(rgb_map))
        # 这里所有值是一样的，所以是求平均而不是求和
        image_data = torch.mean(max_ones_idx, dim=0)
        mask_img = T.ToPILImage()(image_data).convert('RGBA')
        image = add_alphe_mask(image=image, mask=mask_img,color=color)
    image.show()
    return image

def add_alphe_mask(image, mask, color=(1,0,0)):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    image_data = image.load()
    mask_data = mask.load()
    w, h = mask.size
    for y in range(h):
        for x in range(w):
            r,g,b,a = mask_data[x,y][:4]
            r1,g1,b1,a1 = image_data[x,y][:4]
            rgb = r + g + b
            if rgb > 0:
                image_data[x,y] = (r1 + color[0] * 255,g1 +color[1] * 255 ,b1+color[2] * 255,a1)
            # print(rgb)
            
    return image
def save_loss_rate(data,save_path=None):
    # plt.figure()
    for key in data:
        y = data[key]
        x = [i+1 for i in range(len(y))]
        plt.plot(x, y, label=key)
        
    plt.title('Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss Rate')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)