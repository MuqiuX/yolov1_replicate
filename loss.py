from torch import nn
from torchvision.ops import box_iou
from torch.nn import functional as F
import torch

class LossModul(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        '''
        __init__ 的 Docstring
        
        :param self: 说明
        :param S: 说明
        :param B: 说明
        :param C: 说明
        :param lambda_coord: 说明
        :param lambda_noobj: 说明
        '''
        
        super().__init__()
        self.S=S
         
        self.B=B
        self.C=C
        self.lambda_coord=lambda_coord
        self.lambda_noobj=lambda_noobj
        
    def forward(self, predicts, targets):
        '''
        forward 的 Docstring
        
        :param self: 说明
        :param predicts: [7,7,30] -> (x,y,w,h,c,x,y,w,h,class...x20)
        :param targets: [7,7,30] -> (x,y,w,h,c,x,y,w,h,class...x20)
        '''
        
        # 预测值分离 -> [S, S, 5]
        pre_bbox1 = predicts[..., :5]
        pre_bbox2 = predicts[..., 5:10]
        pre_cls = predicts[..., 10:]
        
        # 目标值分离 -> [S, S, 5]
        tar_bbox1 = targets[..., :5]
        tar_bbox2 = targets[..., 5:10]
        tar_cls = targets[..., 10:]
        
        # 获取掩码
        obj_mask = targets[..., 4] > 0
        noobj_mask = ~obj_mask
        
        # 计算iou
        pre_box1_abs = torch.zeros_like(pre_bbox1[..., :4]) # [S, S, 4]
        
        pre_box1_abs[..., 0] = pre_bbox1[..., 0] - pre_bbox1[..., 2] / 2  # x1
        pre_box1_abs[..., 1] = pre_bbox1[..., 1] - pre_bbox1[..., 3] / 2  # y1
        pre_box1_abs[..., 2] = pre_bbox1[..., 0] + pre_bbox1[..., 2] / 2  # x2
        pre_box1_abs[..., 3] = pre_bbox1[..., 1] + pre_bbox1[..., 3] / 2  # y2
        
        pre_box2_abs = torch.zeros_like(pre_bbox2[..., :4]) # [S, S, 4]
        
        pre_box2_abs[..., 0] = pre_bbox2[..., 0] - pre_bbox2[..., 2] / 2  # x1
        pre_box2_abs[..., 1] = pre_bbox2[..., 1] - pre_bbox2[..., 3] / 2  # y1
        pre_box2_abs[..., 2] = pre_bbox2[..., 0] + pre_bbox2[..., 2] / 2  # x2
        pre_box2_abs[..., 3] = pre_bbox2[..., 1] + pre_bbox2[..., 3] / 2  # y2
        
        tar_box_abs = torch.zeros_like(tar_bbox1[..., :4])
        
        tar_box_abs[..., 0] = tar_bbox1[..., 0] - tar_bbox1[..., 2] / 2  # x1
        tar_box_abs[..., 1] = tar_bbox1[..., 1] - tar_bbox1[..., 3] / 2  # y1
        tar_box_abs[..., 2] = tar_bbox1[..., 0] + tar_bbox1[..., 2] / 2  # x2
        tar_box_abs[..., 3] = tar_bbox1[..., 1] + tar_bbox1[..., 3] / 2  # y2
        
        # 展平再计算iou -> [batch * S * S, batch * S * S]
        iou1 = box_iou(pre_box1_abs.view(-1, 4), tar_box_abs.view(-1, 4))
        iou2 = box_iou(pre_box2_abs.view(-1, 4), tar_box_abs.view(-1, 4))
        
        # 取对角元素，重塑 -> [batch, S, S]
        iou1 = iou1.diag().view(-1, self.S, self.S)
        iou2 = iou2.diag().view(-1, self.S, self.S)
        
        responsible_mask = iou1 > iou2
        
        # ========================
        # 1. 定位损失
        # ========================
        if obj_mask.any():
            loss_xy = torch.zeros(1, device=predicts.device)
            
            mask1 = obj_mask & responsible_mask # grid cell 包含目标且box1交并比最大
            loss_xy += F.mse_loss(
                pre_bbox1[mask1][..., :2],
                tar_bbox1[mask1][..., :2],
                reduction='sum'
            )
            
            mask2 = obj_mask & (~responsible_mask) # grid cell 包含目标且box2交并比最大
            loss_xy += F.mse_loss(
                pre_bbox2[mask2][..., :2], 
                tar_bbox1[mask2][..., :2], 
                reduction='sum'
            )
            
            loss_wh = torch.zeros(1, device=predicts.device)

            mask1 = obj_mask & responsible_mask
            loss_wh += F.mse_loss(
                torch.sqrt(torch.clamp(pre_bbox1[mask1][..., 2:4], min=1e-6)),
                torch.sqrt(torch.clamp(tar_bbox1[mask1][..., 2:4], min=1e-6)),
                reduction='sum'
            )
            
            mask2 = obj_mask & (~responsible_mask)
            loss_wh += F.mse_loss(
                torch.sqrt(torch.clamp(pre_bbox2[mask2][..., 2:4], min=1e-6)),
                torch.sqrt(torch.clamp(tar_bbox1[mask2][..., 2:4], min=1e-6)),
                reduction='sum'
            )
            
            # 最终定位损失
            coord_loss = self.lambda_coord * (loss_xy + loss_wh)
        else:
            coord_loss = torch.tensor(0.0, device=predicts.device)
        
        # ========================
        # 2. 置信度损失
        # ========================
        
        # 有物体置信度损失
        if obj_mask.any():
            conf_obj_loss = torch.zeros(1, device=predicts.device)
            mask1 = obj_mask & responsible_mask
            mask2 = obj_mask & (~responsible_mask)
            
            if mask1.any():
                conf_obj_loss += F.mse_loss(
                    pre_bbox1[mask1][..., 4],
                    iou1[mask1],
                    reduction='sum'
                )
            if mask2.any():
                conf_obj_loss += F.mse_loss(
                    pre_bbox2[mask2][..., 4],
                    iou2[mask2],
                    reduction='sum'
                )
        else:
            conf_obj_loss = torch.tensor(0.0, device=predicts.device)
            
        # 无物体置信度损失
        if noobj_mask.any():
            conf_noobj_loss = F.mse_loss(
                pre_bbox1[noobj_mask][..., 4],
                torch.zeros_like(pre_bbox1[noobj_mask][..., 4], device=predicts.device),
                reduction='sum'
            ) + F.mse_loss(
                pre_bbox2[noobj_mask][..., 4],
                torch.zeros_like(pre_bbox2[noobj_mask][..., 4], device=predicts.device),
                reduction='sum'
            )
            conf_noobj_loss *= self.lambda_noobj
        else:
            conf_noobj_loss = torch.tensor(0.0, device=predicts.device)
            
        # 最终置信度损失
        conf_loss = conf_obj_loss + conf_noobj_loss
        
        
        # ========================
        # 3. 分类损失
        # ========================
        if obj_mask.any():
            cls_loss = F.mse_loss(
                pre_cls[obj_mask],
                tar_cls[obj_mask],
                reduction='sum'
            )
        else:
            cls_loss = torch.tensor(0.0, device=predicts.device)
            
        # 总损失
        total_loss = coord_loss + conf_loss + cls_loss
        
        return total_loss
        
if __name__ == '__main__':
    lossfc = LossModul()
    
    pre = torch.rand(20, 7, 7, 30)
    tar = pre = torch.rand(20, 7, 7, 30)
    
    result = lossfc(pre, tar)
    
    print(result)
        