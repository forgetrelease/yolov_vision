import torch.nn as nn
from utils.boxs_util import *
from torch.nn import functional as F
from config import *

class SquaredLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.coord_loss = 5
        self.noobj_loss = 0.5
        
    def forward(self, input, target):
        iou = get_iou(input, target)
        max_iou = torch.max(iou,dim=-1)[0]
        # 概率大于0的
        bbox_mask = boxs_attr(target,4) > 0.0
        pre_temp = boxs_attr(input,4) > 0.0
        obj_i = bbox_mask[..., :]
        resp = torch.zeros_like(pre_temp).scatter_(
            -1,
            torch.argmax(iou, dim=-1, keepdim=True),
            value=1
        )
        obj_ij = obj_i * resp
        noobj_ij = ~obj_ij
        x_los = mse_loss(
            obj_ij * boxs_attr(input, 0),
            obj_ij * boxs_attr(target, 0)
        )
        y_los = mse_loss(
            obj_ij * boxs_attr(input, 1),
            obj_ij * boxs_attr(target, 1)
        )
        xy_los = x_los + y_los
        w_los = mse_loss(
            obj_ij * torch.sign(boxs_attr(input, 2)) * torch.sqrt(torch.abs(boxs_attr(input, 2)) + EPSILON),
            obj_ij * torch.sqrt(boxs_attr(target, 2))
        )
        h_los = mse_loss(
            obj_ij * torch.sign(boxs_attr(input, 3)) * torch.sqrt(torch.abs(boxs_attr(input, 3)) + EPSILON),
            obj_ij * torch.sqrt(boxs_attr(target, 3))
        )
        area_los = w_los + h_los
        obj_loss = mse_loss(
            obj_ij * boxs_attr(input, 4),
            obj_ij * torch.ones_like(boxs_attr(input, 4))
        )
        boobj_loss = mse_loss(
            noobj_ij * boxs_attr(input, 4),
            torch.zeros_like(boxs_attr(input, 4))
        )
        cls_loss = mse_loss(
            input[..., :20],
            target[..., :20]
        )
        
        total = self.coord_loss * (xy_los + area_los) + obj_loss + self.noobj_loss * boobj_loss + cls_loss
        return total / BATCH_SIZE
def mse_loss(a, b):
    flt_a = torch.flatten(a, end_dim=-2)
    flt_b = torch.flatten(b, end_dim=-2).expand_as(flt_a)
    return F.mse_loss(flt_a, flt_b, reduction='sum')

class SquaredMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.coord_loss = 5
        self.noobj_loss = 0.5
        self.mask_loss = 0.001
    
    
    def forward(self, input, target, pred_mask, target_mask):
        iou = get_iou(input, target)
        max_iou = torch.max(iou,dim=-1)[0]
        # 概率大于0的
        bbox_mask = boxs_attr(target,4) > 0.0
        pre_temp = boxs_attr(input,4) > 0.0
        obj_i = bbox_mask[..., :]
        resp = torch.zeros_like(pre_temp).scatter_(
            -1,
            torch.argmax(iou, dim=-1, keepdim=True),
            value=1
        )
        obj_ij = obj_i * resp
        noobj_ij = ~obj_ij
        x_los = mse_loss(
            obj_ij * boxs_attr(input, 0),
            obj_ij * boxs_attr(target, 0)
        )
        y_los = mse_loss(
            obj_ij * boxs_attr(input, 1),
            obj_ij * boxs_attr(target, 1)
        )
        xy_los = x_los + y_los
        w_los = mse_loss(
            obj_ij * torch.sign(boxs_attr(input, 2)) * torch.sqrt(torch.abs(boxs_attr(input, 2)) + EPSILON),
            obj_ij * torch.sqrt(boxs_attr(target, 2))
        )
        h_los = mse_loss(
            obj_ij * torch.sign(boxs_attr(input, 3)) * torch.sqrt(torch.abs(boxs_attr(input, 3)) + EPSILON),
            obj_ij * torch.sqrt(boxs_attr(target, 3))
        )
        area_los = w_los + h_los
        obj_loss = mse_loss(
            obj_ij * boxs_attr(input, 4),
            obj_ij * torch.ones_like(boxs_attr(input, 4))
        )
        boobj_loss = mse_loss(
            noobj_ij * boxs_attr(input, 4),
            torch.zeros_like(boxs_attr(input, 4))
        )
        cls_loss = mse_loss(
            input[..., :20],
            target[..., :20]
        )
        # temp = torch.zeros_like(pred_mask)
        # temp[pred_mask==target_mask] = 1
        mask_loss = F.binary_cross_entropy_with_logits(input=pred_mask, target=target_mask, reduction='mean')
        total = self.coord_loss * (xy_los + area_los) + obj_loss + self.noobj_loss * boobj_loss + cls_loss + self.mask_loss * mask_loss
        return total / BATCH_SIZE
    
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