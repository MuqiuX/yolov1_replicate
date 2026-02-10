from torch import nn
from torchvision.ops import box_iou
from torch.nn import functional as F
import torch

class LossModul(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        '''
        __init__ 的 Docstring
        
        :param S: 网格边长
        :param B: 每个grid cell 预测的bounding box数
        :param C: 预测类别数量
        :param lambda_coord: 宽高损失系数
        :param lambda_noobj: 无目标置信度损失系数
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
        x和y表示的是中心点相对于对应grid cell的比例坐标
        :param predicts: [batch, 7,7,30] -> (x,y,w,h,c,x,y,w,h,class...x20)
        :param targets: [batch, 7,7,30] -> (x,y,w,h,c,x,y,w,h,class...x20)
        对于targets来说, grid cell如果存在目标两个bbox的值完全相等，否则都为0
        '''
        
        # 预测值分离
        pre_bbox1 = predicts[..., :5]
        pre_bbox2 = predicts[..., 5:10]
        pre_cls = predicts[..., 10:]
        
        # 目标值分离, 由于目标的每个grid cell的两个bbox都相同，所以只取一个
        tar_bbox = targets[..., :5]
        tar_cls = targets[..., 10:]
        
        # 获取存在物体掩码和不存在物体掩码
        # 由于只要存在目标该grid cell的bbox置信度都为1否则为0，所以直接使用>0判断
        obj_mask = targets[..., 4] > 0
        noobj_mask = ~obj_mask
        
        # 计算iou，由于bbox的x，y表示的是相对于grid cell的左上角的坐标
        # 所以需要得到绝对坐标， 采用的方法是将每个gird cell的长度都当作1来计算
        
        # 创建表示0 - 7的网格
        grid_x = torch.arange(self.S).view(1, -1).repeat(self.S, 1).float()
        grid_y = torch.arange(self.S).view(-1, 1).repeat(1, self.S).float()
        
        # 得到比例
        ratio = 1.0 / float(self.S)
        
        # tips: 不同维度的tensor相加会升维
        # [batch, S, S] + [S, S] = ... + [S, S].unsqueeze(0).repeat(batch, 1, 1)
        
        # 预测框1绝对坐标
        pre_bbox1_abs = torch.zeros_like(pre_bbox1[..., :4])
        pre_bbox1_abs[..., 0] = (pre_bbox1[..., 0] + grid_x) * ratio
        pre_bbox1_abs[..., 1] = (pre_bbox1[..., 1] + grid_y) * ratio
        pre_bbox1_abs[..., 2] = pre_bbox1[..., 2]
        pre_bbox1_abs[..., 3] = pre_bbox1[..., 3]
        
        # 预测框2绝对坐标
        pre_bbox2_abs = torch.zeros_like(pre_bbox2[..., :4])
        pre_bbox2_abs[..., 0] = (pre_bbox2[..., 0] + grid_x) * ratio
        pre_bbox2_abs[..., 1] = (pre_bbox2[..., 1] + grid_y) * ratio
        pre_bbox2_abs[..., 2] = pre_bbox2[..., 2]
        pre_bbox2_abs[..., 3] = pre_bbox2[..., 3]
        
        # 目标框绝对坐标
        tar_bbox_abs = torch.zeros_like(tar_bbox[..., :4])
        tar_bbox_abs[..., 0] = (tar_bbox[..., 0] + grid_x) * ratio
        tar_bbox_abs[..., 1] = (tar_bbox[..., 1] + grid_y) * ratio
        tar_bbox_abs[..., 2] = tar_bbox[..., 2]
        tar_bbox_abs[..., 3] = tar_bbox[..., 3]
        
        def cxcywh_to_xyxy(boxes):
            """将 (cx, cy, w, h) 转换为 (x1, y1, x2, y2)"""
            xyxy = torch.zeros_like(boxes)
            xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
            xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
            xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
            xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2
            return xyxy
        
        pre_bbox1_xyxy = cxcywh_to_xyxy(pre_bbox1_abs)
        pre_bbox2_xyxy = cxcywh_to_xyxy(pre_bbox2_abs)
        tar_bbox_xyxy = cxcywh_to_xyxy(tar_bbox_abs)
        
        # 展平再计算iou -> [batch * S * S, batch * S * S]
        iou1 = box_iou(pre_bbox1_xyxy.view(-1, 4), tar_bbox_xyxy.view(-1, 4))
        iou2 = box_iou(pre_bbox2_xyxy.view(-1, 4), tar_bbox_xyxy.view(-1, 4))
        
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
                tar_bbox[mask1][..., :2],
                reduction='sum'
            )
            
            mask2 = obj_mask & (~responsible_mask) # grid cell 包含目标且box2交并比最大
            loss_xy += F.mse_loss(
                pre_bbox2[mask2][..., :2], 
                tar_bbox[mask2][..., :2], 
                reduction='sum'
            )
            
            loss_wh = torch.zeros(1, device=predicts.device)

            mask1 = obj_mask & responsible_mask
            loss_wh += F.mse_loss(
                torch.sqrt(torch.clamp(pre_bbox1[mask1][..., 2:4], min=1e-6)),
                torch.sqrt(torch.clamp(tar_bbox[mask1][..., 2:4], min=1e-6)),
                reduction='sum'
            )
            
            mask2 = obj_mask & (~responsible_mask)
            loss_wh += F.mse_loss(
                torch.sqrt(torch.clamp(pre_bbox2[mask2][..., 2:4], min=1e-6)),
                torch.sqrt(torch.clamp(tar_bbox[mask2][..., 2:4], min=1e-6)),
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
        
        return total_loss.float()

def create_loss_fn(args):
    fn = LossModul()
    
    return fn