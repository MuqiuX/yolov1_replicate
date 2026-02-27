from torch import nn
import numpy as np
import torch
from torch import Tensor
import albumentations as A
import cv2

class ToV1Size(nn.Module):
    def __init__(self, S: int=7, B: int=2, C: int=20):
        super().__init__()
        
        self.v1size = 448
        self.S = S
        self.B = B
        self.C = C
        
    def forward(self, image: np.ndarray, label: np.ndarray):
        
        # 对输入标注进行拆分
        # 对坐标部分赋值
        classes = label[..., 0].astype(int)
        bboxes = label[..., 1:]
        
        transform = A.Compose([
            A.Resize(
                height=self.v1size,
                width=self.v1size,
                interpolation=cv2.INTER_CUBIC,
                area_for_downscale='image',
                ),
            A.HorizontalFlip(p=0.5)
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['bbox_class'],
        ))
        
        result = transform(
            image=image,
            bboxes=bboxes,
            bbox_class=classes
        )
        
        resized_bboxes = result['bboxes']
        resized_bbox_class = result['bbox_class']
        
    
class ToV1Label(nn.Module):
    def __init__(self, S: int=7, B: int=2, C: int=20):
        super().__init__()
        
        self.S = S
        self.B = B
        self.C = C
        
    def forward(self, label: list[list]) -> Tensor:
        
        label = np.array(label)
        
        # 对输入标注进行拆分
        # 对坐标部分赋值
        classes = label[..., 0].astype(int)
        bboxes = label[..., 1:]
        
        # 首先将原来的中心坐标乘S，归一化为[0, S]
        # 向下取整，获取在S*S网格下的格点坐标
        # 利用坐标-格点坐标得到各自对于所在网格的相对坐标
        bboxes[..., :2] *= self.S
        grid_coord = bboxes[..., :2].astype(int)
        bboxes[..., :2] -= grid_coord
        
        target_label = np.zeros((self.S, self.S, 5 * self.B + self.C))
        
        # 遍历所有bbox，为目标标注赋值
        for index, coord in enumerate(grid_coord.tolist()):
            grid_cell_label = np.zeros(5 * self.B + self.C)
            
            # 赋值类别
            class_id = classes[index]
            grid_cell_label[5 * self.B + class_id] = 1
            
            # 赋值bbox和置信度
            for i in range(self.B):
                grid_cell_label[i * 5:(i + 1) * 5 - 1] = bboxes[index]
                grid_cell_label[(i + 1) * 5 - 1] = 1
                
            target_label[coord[0], coord[1], :] = grid_cell_label
            
        return torch.from_numpy(target_label)