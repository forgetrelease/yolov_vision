
from PIL import ImageDraw, ImageFont, Image
import torchvision.transforms as T
from config import *
import torch

# (batch, 7, 7, 30) 
def get_iou(pred, target):
    pred_s, pred_e = coords(pred)
    target_s, target_e = coords(target)
    s = torch.max(
        pred_s,
        target_s,
    )
    e = torch.min(
        pred_e,
        target_e,
    )
    inn_sides = torch.clamp(e - s, min=0)
    inn = inn_sides[..., 0]*inn_sides[..., 1]
    
    pred_area = boxs_attr(pred, 2) * boxs_attr(pred, 3)
    pred_area = pred_area
    
    target_area = boxs_attr(target, 2) * boxs_attr(target, 3)
    target_area = target_area
    
    union = pred_area + target_area - inn
    zero_unions = (union == 0)
    union[zero_unions] = EPSILON
    inn[zero_unions] = 0.0
    return inn / union
    

# (center_x, center_y, w, h) -> (xmin, ymin, xmax, ymax)
def coords(data):
    w = boxs_attr(data,2)
    x = boxs_attr(data,0)
    x1 = x - w / 2
    x2 = x + w / 2
    
    h = boxs_attr(data,3)
    y = boxs_attr(data,1)
    y1 = y - h / 2
    y2 = y + h / 2
    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)
    
def boxs_attr(data, idx):
    start = 20 + idx
    return data[..., start::5]

def normalize_box(origin_points, origin_size, target_size):
    xmin, ymin, xmax, ymax = origin_points
    origin_w, origin_h = origin_size
    target_w, target_h = target_size
    xmin = int(target_w * xmin * 1.0 / origin_w)
    ymin = int(target_h * ymin * 1.0 / origin_h)
    xmax = int(target_w * xmax * 1.0 / origin_w)
    ymax = int(target_h * ymax * 1.0 / origin_h)
    return xmin, ymin, xmax, ymax
def normalize_box_mask_scale(origin_points, origin_size, target_size):
    xmin, ymin, xmax, ymax = origin_points
    origin_w, origin_h = origin_size
    target_w, target_h = target_size
    scale = (target_h * 1.0 / origin_h) if origin_w < origin_h else (1.0 * target_w / origin_w)
    xmin = int(scale * xmin)
    ymin = int(scale * ymin)
    xmax = int(scale * xmax)
    ymax = int(scale * ymax)
    return xmin, ymin, xmax, ymax

def show_boxs(image_data, boxss, color=(0, 255, 0),mask=None):
    for j in range(image_data.size(dim=0) ):
        image = T.ToPILImage()(image_data[j, :, :, :])
        draw = ImageDraw.Draw(image)
        boxs = boxss[j]
        for box in boxs:
            text = f'{class_of_index(box[4])}'
            draw.rectangle(box[:4], outline=color)
            draw.text((box[0], box[1] - 10), text, fill=color)
        if mask is not None:
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            m_t = mask[j, :, :, :]
            m = T.ToPILImage()(m_t)
            if m.mode != 'RGBA':
                m = m.convert('RGBA')
            r,g,b,a = m.split()
            a = a.point(lambda i : int(i * 0.5))
            # m.show()
            # m.save('/Users/chunsheng/Downloads/ttll.png')
            
            image = Image.alpha_composite(image, m)
        image.show()
def plot_boxs(data, label):
    width, height = IMAGE_SIZE
    grid_w = width / 7
    grid_h = height / 7
    all_bosx = []
    for i in range(data.size(dim=0)):
        image = data[i, :, :, :]
        # image = T.ToPILImage()(image)
        # image.show()
        pred_boxs = label[i, :, :, :]
        for col in range(7):
            for row in range(7):
                cls_idx = torch.argmax(pred_boxs[row, col, :20]).item()
                for k in range(2):
                    confidence = pred_boxs[row, col, 20+k*5+4].item()
                    if confidence > MIN_CONFIDENCE:
                        print(cls_idx, confidence)
                        boxs = pred_boxs[row, col, 20+k*5:20+(k+1)*5].tolist()
                        print(boxs)
                        center_x, center_y, w, h, _ = boxs
                        xcenter = center_x * grid_w + col * grid_w
                        ycenter =center_y* grid_h + row * grid_h
                        box_w = w * grid_w
                        box_h = h* grid_h
                        all_bosx.append([xcenter - box_w/2, ycenter-box_h/2, xcenter + box_w/2, ycenter + box_h/2, cls_idx])
    if len(all_bosx)>0:
        # boxs = torch.tensor(all_bosx)
        # all_bosx = 
        show_boxs(image_data=data, boxss=[all_bosx], color=(0, 255, 0))
                        
        
                        
# (xmin, ymin, xmax, ymax) -> 7 * 7 * 30
def orignal_boxs_to_tensor(batch_boxss):
    width, height = IMAGE_SIZE
    grid_w = width / 7
    grid_h = height / 7
    rectangle_truths = []
    for batch in range(len(batch_boxss)):
        boxss = batch_boxss[batch]
        rectangle_truth = torch.zeros((7, 7, 30))
        box_cell = {}
        for box in boxss:
            xmin, ymin, xmax, ymax = box[:4]
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            col = int(center_x // grid_w)
            row = int(center_y // grid_h)
            if 0 <= col < 7 and 0 <= row < 7:
                box_idx = box_cell.get((row, col), 0)
                if box_idx >= 2:
                    continue
                cls_idx = torch.zeros(20)
                cls_idx[box[4]] = 1.0
                rectangle_truth[row, col, :20] = cls_idx
                box_truth = ((center_x - col * grid_w) / grid_w,
                        (center_y - row * grid_h) / grid_h, 
                        (xmax - xmin)  / grid_w,
                        (ymax - ymin)  / grid_h,
                        1.0
                )
                start_idx = 20 + box_idx * 5
                rectangle_truth[row, col, start_idx:start_idx+5] = torch.tensor(box_truth)
                box_cell[(row, col)] = box_idx + 1
        rectangle_truths.append(rectangle_truth)
    return torch.stack(rectangle_truths)
    
# 7 * 7 * 30 -> (xmin, ymin, xmax, ymax)
def tensor_boxs_to_orignal(boxss_truths):
    width, height = IMAGE_SIZE
    grid_w = width / 7
    grid_h = height / 7
    all_boxss = []
    for batch in range(boxss_truths.size(0)):
        boxss_truth = boxss_truths[batch, :, :, :]
        boxss = []
        for col in range(7):
            for row in range(7):
                cls_idx = torch.argmax(boxss_truth[row, col, :20]).item()
                for k in range(2):
                    confidence = boxss_truth[row, col, 20+k*5+4]
                    if confidence > MIN_CONFIDENCE:
                        boxs = boxss_truth[row, col, 20+k*5:20+(k+1)*5].tolist()
                        center_x, center_y, box_w, box_h, c = boxs
                        center_x = center_x * grid_w + col * grid_w
                        center_y = center_y * grid_h + row * grid_h
                        xmin = center_x - box_w * grid_w / 2
                        ymin = center_y - box_h * grid_h / 2
                        xmax = xmin + box_w * grid_w
                        ymax = ymin + box_h * grid_h
                        boxss.append([xmin, ymin, xmax, ymax, cls_idx, c])
        all_boxss.append(boxss)               
    return all_boxss
def class_of_index(idx):
    for k, v in OBJ_INDEX.items():
        if v == idx:
            return k
    return 'unknown'

'''
temp = mask_cls(img=img)
flat_mask = temp.view(-1)
uniq_key = list(set(flat_mask.tolist()))
all_mask = []
for i in uniq_key:
    if i == 0:
        continue
    tt = torch.zeros_like(temp)
    tt[temp == i] = 1
    all_mask.append(tt)

for image in all_mask:
    t = T.ToPILImage()(image)
    t.show()
        '''
def mask_cls(img):
    w,h = img.size
    temp = torch.zeros(w*h).reshape([1,h,w])
    image_data = img.load()
    for y in range(h):
        for x in range(w):
            r,g,b = image_data[x,y][:3]
            rgb = r + g + b
            if rgb in [0, 640]:
                continue
            temp[0, y, x] = rgb
    return temp
def mask_channel_cls(img):
    w,h = img.size
    image_data = img.load()
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
    return chennales

def single_image(temp):
    flat_mask = temp.view(-1)
    uniq_key = list(set(flat_mask.tolist()))
    all_mask = []
    for i in uniq_key:
        if i == 0:
            continue
        tt = torch.zeros_like(temp)
        tt[temp == i] = 1
        all_mask.append(tt)

    for image in all_mask:
        t = T.ToPILImage()(image)
        t.show()