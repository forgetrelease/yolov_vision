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
        temp = torch.zeros_like(pred_mask)
        temp[pred_mask==target_mask] = 1
        mask_loss = F.binary_cross_entropy_with_logits(input=temp, target=torch.zeros_like(pred_mask), reduction='mean')
        total = self.coord_loss * (xy_los + area_los) + obj_loss + self.noobj_loss * boobj_loss + cls_loss + self.mask_loss * mask_loss
        return total / BATCH_SIZE