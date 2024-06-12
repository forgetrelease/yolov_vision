import torch.nn as nn
from utils.boxs_util import *
from torch.nn import functional as F
from config import *
import numpy as np

def mse_loss(a, b):
    flt_a = torch.flatten(a, end_dim=-2)
    flt_b = torch.flatten(b, end_dim=-2).expand_as(flt_a)
    return F.mse_loss(flt_a, flt_b, reduction='sum')
    
################ 重新写一个损失函数 ###############################
def box_attr(data, idx):
    start = 20+idx
    return data[..., start::5]
def box_coord(data):
    centerx = box_attr(data, 0)
    centery = box_attr(data, 1)
    width = box_attr(data, 2)
    height = box_attr(data, 3)
    xmin = centerx - width / 2
    ymin = centery - height / 2
    xmax = centerx + width / 2
    ymax = centery + height / 2
    return xmin, ymin, xmax, ymax
    
class BoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.coord_loss = 5
        self.noobj_loss = 0.5
    # 返回计算相交区域占比，BATCH_SIZE, 7, 7, 2
    def iou(self,input, target):
        p_xmin, p_ymin, p_xmax, p_ymax = box_coord(input)
        t_xmin, t_ymin, t_xmax, t_ymax = box_coord(target)
        # 计算相交区域的左上角和右下角坐标
        union_xmin = torch.max(p_xmin,t_xmin)
        union_ymin = torch.max(p_ymin,t_ymin)
        union_xmax = torch.min(p_xmax,t_xmax)
        union_ymax = torch.min(p_ymax,t_ymax)
        inn = torch.clamp(union_xmax - union_xmin, min=0) * torch.clamp(union_ymax - union_ymin, min=0)
        all_area = box_attr(input, 2) * box_attr(input, 3) + box_attr(target, 2) * box_attr(target, 3) - inn
        zero_union = (all_area == 0)
        all_area[zero_union] = EPSILON         #预测的相交区域可能不为0，所以这个值不能为0
        inn[zero_union] = 0
        return inn / all_area
    
    # x y w h 损失
    # 相交区域占比
    # 类和非类相似
    # 
    def forward(self, input, target):
        iou = self.iou(input, target)   ## b, 7, 7, 2
        max_iou = torch.max(iou, dim=-1)[0] ## b, 7, 7, 1
        max_iou = torch.unsqueeze(max_iou, -1)  ## b, 7, 7, 1
        # 相似度大于0的索引，
        box_mask = box_attr(target, 4) > 0  ## b, 7, 7, 2
        input_template = box_attr(input, 4) > 0  ## b, 7, 7, 2
        # iou最大的索引
        obj_idx = box_mask[..., 0:1] # b, 7, 7, 1 #因为一个grid预测2个框，只要第一个框的相似度大于0，这个索引就有效，所以需要取第一个就可以了
        no_obj_idx = ~obj_idx       #去反
        # iou最大的索引设置,b,7,7,2
        resp = torch.zeros_like(input_template).scatter_(
            -1,
            torch.argmax(max_iou, dim=-1, keepdim=True),
            value=1
        )
        # 相似度大于0，且iou最大的索引, b,7,7,2
        obj_ij = obj_idx * resp
        # 只计算相似度大于0的索引
        class_loss = mse_loss(
            obj_idx * input[..., :20],
            obj_idx * target[..., :20],
        )
        input_confidence = box_attr(input, 4)
        # 只计算相似度大于0切的索引
        confidence_loss = mse_loss(
            obj_ij * input_confidence,
            obj_ij * torch.ones_like(input_confidence)
        )
        no_confidence_loss = mse_loss(
            no_obj_idx * input_confidence,
            torch.zeros_like(input_confidence)
        )
        x_loss = mse_loss(
            obj_ij * box_attr(input, 0),
            obj_ij * box_attr(target, 0)
        )
        y_loss = mse_loss(
            obj_ij * box_attr(input, 1),
            obj_ij * box_attr(target, 1)
        )
        input_width = box_attr(input, 2)
        w_loss = mse_loss(
            obj_ij * torch.sign(input_width)*torch.sqrt(torch.abs(input_width))+EPSILON,
            obj_ij * torch.sqrt(box_attr(target, 2))
        )
        input_height = box_attr(input, 3)
        h_loss = mse_loss(
            obj_ij * torch.sign(input_height)*torch.sqrt(torch.abs(input_height))+EPSILON,
            obj_ij * torch.sqrt(box_attr(target, 3))
        )
        print(f"class_loss:{class_loss.item()}, confidence_loss:{confidence_loss.item()}, no_confidence_loss:{no_confidence_loss.item()}, x_loss:{x_loss.item()}, y_loss:{y_loss.item()}, w_loss:{w_loss.item()}, h_loss:{h_loss.item()}")
        total_loss = class_loss \
                    + confidence_loss \
                    + self.noobj_loss * no_confidence_loss \
                    + self.coord_loss *(x_loss + y_loss + w_loss + h_loss)
        
        return total_loss / BATCH_SIZE


class SquaredMaskLoss(nn.Module):
    def __init__(self,device=None,only_box=True):
        super().__init__()
        self.coord_loss = 5
        self.noobj_loss = 0.5
        self.mask_loss = 0.001
        rgb_map_num = np.load("./rgbs.npy")
        self.rgb_map = torch.from_numpy(rgb_map_num)#20
        self.only_box = only_box
        if device is not None:
            self.rgb_map = self.rgb_map.to(device)
    # 返回计算相交区域占比，BATCH_SIZE, 7, 7, 2
    def iou(self,input, target):
        p_xmin, p_ymin, p_xmax, p_ymax = box_coord(input)
        t_xmin, t_ymin, t_xmax, t_ymax = box_coord(target)
        # 计算相交区域的左上角和右下角坐标
        union_xmin = torch.max(p_xmin,t_xmin)
        union_ymin = torch.max(p_ymin,t_ymin)
        union_xmax = torch.min(p_xmax,t_xmax)
        union_ymax = torch.min(p_ymax,t_ymax)
        inn = torch.clamp(union_xmax - union_xmin, min=0) * torch.clamp(union_ymax - union_ymin, min=0)
        all_area = box_attr(input, 2) * box_attr(input, 3) + box_attr(target, 2) * box_attr(target, 3) - inn
        zero_union = (all_area == 0)
        all_area[zero_union] = EPSILON         #预测的相交区域可能不为0，所以这个值不能为0
        inn[zero_union] = 0
        return inn / all_area
    # x y w h 损失
    # 相交区域占比
    # 类和非类相似
    # 
    def forward(self, input, target, mask_input, mask_target):
        if self.only_box == False:
            # mask loss
            # mask_input b,21,448,448 mask_target b,1,448,448
            cls_count = self.rgb_map.shape[-1]
            rgb_map = self.rgb_map.unsqueeze(1).unsqueeze(1).repeat((1,IMAGE_SIZE[1],IMAGE_SIZE[0])).expand(cls_count, IMAGE_SIZE[1],IMAGE_SIZE[0]) #21,448,448
            rgb_map = rgb_map.unsqueeze(0).repeat((BATCH_SIZE,1,1,1)).expand(BATCH_SIZE,cls_count,IMAGE_SIZE[1],IMAGE_SIZE[0]) #21,448,448 > 32,21,448,448
            '''
            在您提供的forward函数中，您似乎在尝试计算一个基于掩码（mask）的损失。在PyTorch中，当您使用.gather()方法或者类似的操作
            （如.argmax()）时，返回的通常是不可导的张量（即requires_grad=False），因为这些操作本质上不是可导的。然而，由于您正在计
            算损失，您可能希望这些操作是可导的，以便可以反向传播梯度。
            '''
            # mask_target > b,21,448,448
            mask_target = torch.mean(mask_target, dim=1,keepdim=True)
            # mask_input > b,21,448,448
            
            
            # mask_input_max = torch.softmax(mask_input, dim=1)
            # mask_input_max_arg = torch.argmax(mask_input_max, dim=1)
            # mask_input_cls = rgb_map.gather(dim=1,index=mask_input_max_arg.unsqueeze(1).expand_as(rgb_map))
            # mask_input_image = torch.mean(mask_input_cls, dim=1)
            
            # mask_target_max = torch.softmax(mask_target, dim=1)
            # mask_target_max_arg = torch.argmax(mask_target_max, dim=1)
            # mask_target_cls = rgb_map.gather(dim=1,index=mask_target_max_arg.unsqueeze(1).expand_as(rgb_map))
            # mask_target_image = torch.sum(mask_target_cls, dim=1)
            
            # mask_loss = F.binary_cross_entropy_with_logits(input=mask_input_image, target=mask_target_image, reduction='mean')
            mask_loss = F.binary_cross_entropy_with_logits(input=mask_input, target=mask_target, reduction='mean')
            # mask_loss = F.mse_loss(input=mask_input, target=mask_target, reduction='mean')
            return self.mask_loss * mask_loss / BATCH_SIZE
        iou = self.iou(input, target)   ## b, 7, 7, 2
        max_iou = torch.max(iou, dim=-1)[0] ## b, 7, 7, 1
        max_iou = torch.unsqueeze(max_iou, -1)  ## b, 7, 7, 1
        # 相似度大于0的索引，
        box_mask = box_attr(target, 4) > 0  ## b, 7, 7, 2
        input_template = box_attr(input, 4) > 0  ## b, 7, 7, 2
        # iou最大的索引
        obj_idx = box_mask[..., 0:1] # b, 7, 7, 1 #因为一个grid预测2个框，只要第一个框的相似度大于0，这个索引就有效，所以需要取第一个就可以了
        no_obj_idx = ~obj_idx       #去反
        # iou最大的索引设置,b,7,7,2
        resp = torch.zeros_like(input_template).scatter_(
            -1,
            torch.argmax(max_iou, dim=-1, keepdim=True),
            value=1
        )
        # 相似度大于0，且iou最大的索引, b,7,7,2
        obj_ij = obj_idx * resp
        # 只计算相似度大于0的索引
        class_loss = mse_loss(
            obj_idx * input[..., :20],
            obj_idx * target[..., :20],
        )
        input_confidence = box_attr(input, 4)
        # 只计算相似度大于0切的索引
        confidence_loss = mse_loss(
            obj_ij * input_confidence,
            obj_ij * torch.ones_like(input_confidence)
        )
        no_confidence_loss = mse_loss(
            no_obj_idx * input_confidence,
            torch.zeros_like(input_confidence)
        )
        x_loss = mse_loss(
            obj_ij * box_attr(input, 0),
            obj_ij * box_attr(target, 0)
        )
        y_loss = mse_loss(
            obj_ij * box_attr(input, 1),
            obj_ij * box_attr(target, 1)
        )
        input_width = box_attr(input, 2)
        w_loss = mse_loss(
            obj_ij * torch.sign(input_width)*torch.sqrt(torch.abs(input_width))+EPSILON,
            obj_ij * torch.sqrt(box_attr(target, 2))
        )
        input_height = box_attr(input, 3)
        h_loss = mse_loss(
            obj_ij * torch.sign(input_height)*torch.sqrt(torch.abs(input_height))+EPSILON,
            obj_ij * torch.sqrt(box_attr(target, 3))
        )
        
        
        # print(f"class_loss:{class_loss.item()}, confidence_loss:{confidence_loss.item()}, no_confidence_loss:{no_confidence_loss.item()}, x_loss:{x_loss.item()}, y_loss:{y_loss.item()}, w_loss:{w_loss.item()}, h_loss:{h_loss.item()}, mask_loss:{mask_loss.item()}")
        total_loss = class_loss \
                    + confidence_loss \
                    + self.noobj_loss * no_confidence_loss \
                    + self.coord_loss *(x_loss + y_loss + w_loss + h_loss) \
                    
        return total_loss / BATCH_SIZE