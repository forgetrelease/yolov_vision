
import torchvision.transforms as T
import torch
from config import *
from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt

def show_box_masks(image, target, mask=None):
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
        
    color=(255, 0, 0)
    draw = ImageDraw.Draw(image)
    log_msg = []
    for box in hash_box:
        xmin, ymin, xmax, ymax, cls_idx, c = box
        text = f'{class_of_index(box[4])},{box[5]:.2f}'
        log_msg.append(f'{class_of_index(box[4])},{box[5]:.2f} at ({box[0]:.2f},{box[1]:.2f})-{box[2]:.2f},{box[3]:.2f}')
        draw.rectangle(box[:4], outline=color)
        font=ImageFont.load_default()
        draw.text((box[0], box[1] - 11 if box[1] - 11 > 0 else box[1] + 1), text, fill=color,font=font)
    del draw
    print(log_msg)
    # image.show()
    return image
def plot_box_masks(image, target, mask=None):
    pass
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