from utils.dataset import load_data, ImageLabelDataset, SegmentDataset, annotation_filename
from utils.boxs_util import *
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
import torch
import torch.nn.functional as F
if __name__ == "__main__":


    # 假设我们有一批4个样本的logits和对应的真实标签
    logits = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32)
    targets = torch.tensor([1, 0, 1, 0], dtype=torch.float32)

    # 计算二元交叉熵损失（带logits）
    loss = F.binary_cross_entropy_with_logits(input=logits, target=targets, reduction='mean')
    print(loss)
    # data_loader = load_data()
    # for image, target,images in data_loader:
    #     print(images)
    #     break
    
    # img = Image.open('/Users/chunsheng/Desktop/workspace/pytorch_study/yolov_vision/data/VOCdevkit/VOC2007/SegmentationClass/000129.png').convert('RGB')
    # masks = resize_mask_image(img)
    # single_image(masks)
    # data_set = SegmentDataset()
    # loader = DataLoader(data_set, batch_size=2, shuffle=True)
    # for image, label,mask in loader:
    #     # print(mask)
    #     # print(label)
    #     single_image(mask[0, :, : ,:])
    #     show_boxs(image, tensor_boxs_to_orignal(label),mask=mask)
    #     break
    
    
    a = [4,5,4,6,1.0,6,5,8,6,1.0]
    b = [5,7,6,8,1.0,7,6,6,4,1.0,]
    a = torch.tensor(a).reshape(1,1,1,10)
    b = torch.tensor(b).reshape(1,1,1,10)
    # (2,2), (2,2); (6,8),(10,8)
    pred_s, pred_e = coords(a)
    print(pred_s, pred_e)
    # (2,3), (4,4); (8,11),(10,8)
    target_s, target_e = coords(b)
    print(target_s, target_e)
    # aa= pred_s.unsqueeze(4)
    # print(aa.expand(-1, -1, -1, 2, 2, 2))
    # ss=target_s.unsqueeze(3)
    # print(ss.expand(-1, -1, -1, 2, 2, 2))
    # (3,3)
    s = torch.max(
        pred_s,
        target_s,
    )
    # (5,5)
    e = torch.min(
        pred_e,
        target_e,
    )
    # (2,3),(4,4); (6,8),(10,8)
    print(s, e)
    inn_sides = torch.clamp(e - s, min=0)
    # (4,5);(6,4)
    print(inn_sides)
    inn = inn_sides[..., 0]*inn_sides[..., 1]
    print(inn)
    pred_area = boxs_attr(a,2) * boxs_attr(a,3)
    print(pred_area)
    target_area = boxs_attr(b,2) * boxs_attr(b,3)
    print(target_area)
    uniob = pred_area + target_area - inn
    print(uniob)
    zero = (uniob == 0.0)
    print(zero)
    
    
    
    