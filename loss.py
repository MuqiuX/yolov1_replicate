from torch import nn
from torchvision.ops import box_iou
from torch.nn import functional as F
import torch

class LossModul(nn.Module):
    def __init__(self, S: int=7, B: int=2, C: int=20, lambda_coord: float=5, lambda_noobj: float=0.5):
        super().__init__()
        
        self.S=S
        self.B=B
        self.C=C
        self.lambda_coord=lambda_coord
        self.lambda_noobj=lambda_noobj
        
    def forward(self, predicts, targets):
        # ===============================================================
        # 预测值和目标值分离
        # ===============================================================

        # 分离出两个bbox和类别预测，方便使用
        pre_bbox1 = predicts[..., :5]
        pre_bbox2 = predicts[..., 5:10]
        pre_cls = predicts[..., 10:]
        
        # 由于目标的每个grid cell的两个bbox都相同，所以只取一个
        tar_bbox = targets[..., :5]
        tar_cls = targets[..., 10:]
        
        # ===============================================================
        # 获取存在物体掩码, 不存在物体掩码，bbox1iou > bbox2iou掩码
        # ===============================================================

        # 由于只要存在目标该grid cell的bbox置信度都为1否则为0，所以直接使用>0判断
        obj_mask = targets[..., 4] > 0
        noobj_mask = ~obj_mask
        
        iou1, iou2 = self._get_iou(pre_bbox1,pre_bbox2,tar_bbox)
        
        responsible_mask = iou1 > iou2
        
        # ===============================================================
        # 定位损失
        # ===============================================================
        if obj_mask.any():
            # 定位损失
            loss_xy = torch.zeros(1, device=predicts.device)
            
            # grid cell 包含目标且box1交并比最大
            mask1 = obj_mask & responsible_mask
            
            loss_xy += F.mse_loss(
                pre_bbox1[mask1][..., :2],
                tar_bbox[mask1][..., :2],
                reduction='sum'
            )
            
            # grid cell 包含目标且box2交并比最大
            mask2 = obj_mask & (~responsible_mask)
            
            loss_xy += F.mse_loss(
                pre_bbox2[mask2][..., :2], 
                tar_bbox[mask2][..., :2], 
                reduction='sum'
            )
            
            # 宽高损失
            loss_wh = torch.zeros(1, device=predicts.device)

            mask1 = obj_mask & responsible_mask
            
            loss_wh += F.mse_loss(
                torch.sqrt(torch.clamp(pre_bbox1[mask1][..., 2:4], min=1e-4)),
                torch.sqrt(torch.clamp(tar_bbox[mask1][..., 2:4], min=1e-4)),
                reduction='sum'
            )
            
            mask2 = obj_mask & (~responsible_mask)
            
            loss_wh += F.mse_loss(
                torch.sqrt(torch.clamp(pre_bbox2[mask2][..., 2:4], min=1e-4)),
                torch.sqrt(torch.clamp(tar_bbox[mask2][..., 2:4], min=1e-4)),
                reduction='sum'
            )
            
            # 最终定位损失
            coord_loss = self.lambda_coord * (loss_xy + loss_wh)
        else:
            coord_loss = torch.tensor(0.0, device=predicts.device)
        
        # ===============================================================
        # 置信度损失
        # ===============================================================
        
        # 有物体置信度损失
        if obj_mask.any():
            # 置信度损失
            conf_obj_loss = torch.zeros(1, device=predicts.device)
            
            mask1 = obj_mask & responsible_mask
            
            if mask1.any():
                conf_obj_loss += F.mse_loss(
                    pre_bbox1[mask1][..., 4],
                    iou1[mask1],
                    reduction='sum'
                )
                
            mask2 = obj_mask & (~responsible_mask)
                
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
        
        
        # ===============================================================
        # 分类损失
        # ===============================================================
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
    
    def _get_iou(
        self,
        pre_bbox1: torch.Tensor,
        pre_bbox2: torch.Tensor,
        tar_bbox: torch.Tensor,
    ) -> torch.Tensor:
        # ===============================================================
        # 计算iou， 获取置信度大小比较掩码
        # ===============================================================
        
        # 由于bbox的x，y表示的是相对于grid cell的左上角的坐标
        # 所以需要得到绝对坐标
        # 采用的方法是将每个gird cell的长度都当作1来计算
        # 原坐标加上下面创建的网格后
        # 再除以长度S即可得到全局的归一化坐标
        
        # 分别创建x 和 y 方向上的 0 - 7的网格
        device = pre_bbox1.device  # 获取输入张量的设备
        grid_x = torch.arange(self.S, device=device).view(1, -1).repeat(self.S, 1).float()
        grid_y = torch.arange(self.S, device=device).view(-1, 1).repeat(1, self.S).float()
        
        ratio = 1.0 / float(self.S)
        
        # 预测框1绝对坐标
        pre_bbox1_abs = torch.zeros_like(pre_bbox1[..., :4])
        
        pre_bbox1_abs[..., 0] = (pre_bbox1[..., 0] + grid_x) * ratio # cx
        pre_bbox1_abs[..., 1] = (pre_bbox1[..., 1] + grid_y) * ratio # cy
        pre_bbox1_abs[..., 2] = pre_bbox1[..., 2]                    # w
        pre_bbox1_abs[..., 3] = pre_bbox1[..., 3]                    # h
        
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
        
        return iou1, iou2

def create_loss_fn(args: dict) -> nn.Module:
    """创建损失模型

    Args:
        args (dict): 配置参数字典，包含以下字段：
            - S (int): 网格大小，默认 7
            - B (int): 每个网格的边界框数，默认 2  
            - C (int): 类别数，默认 20
            - lambda_coord (float): 坐标损失权重，默认 5.0
            - lambda_noobj (float): 无目标损失权重，默认 0.5

    Returns:
        nn.Module: 创建的损失函数模型
    """
    
    fn = LossModul(
        S=args['S'],
        B=args['B'],
        C=args['C'],
        lambda_coord=args['lambda_coord'],
        lambda_noobj=args['lambda_noobj']
    )
    
    return fn