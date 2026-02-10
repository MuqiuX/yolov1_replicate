from torch import nn
from PIL import Image
import numpy as np
import torch

class ToRequired(nn.Module):
    '''
    将图片和yolo格式的标注转化成，yolov1所需要的图片和标注
    '''
    def __init__(self, input_image_size=448, S=7, B=2, C=20, transform=None, target_transform=None):
        super().__init__()
        
        self.S = S
        self.B = B
        self.C = C
        self.input_image_size = input_image_size
        
        self.transform = transform
        self.target_transform = target_transform
        
        
    def forward(self, image: Image, labels: list):
        '''
        对于图片的转化，将原图片缩放至宽或者高
        :param image: 说明
        :type image: Image
        :param labels: 说明
        :type labels: list
        '''
        # 拆分labels
        np_labels = np.array(labels)
        boxes = np_labels[..., 1:5]
        classes = np_labels[..., 0].astype(int)
        
        # 获取缩放比例,分别用宽高计算缩放比例, 取较小的，作为全图缩放比例
        w, h = image.size
        scale = min(self.input_image_size / w, self.input_image_size / h)
        
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        
        # 对图片进行缩放, 创建一张符合目标大小的新图片
        # 将原图片覆盖到新图片左上角
        resized_image = image.resize((scaled_w, scaled_h), Image.BILINEAR)
        target_image = Image.new('RGB', (self.input_image_size, self.input_image_size), (114, 114, 114))
        target_image.paste(resized_image)
        
        # 将坐标和宽高缩放
        # after_scale = now*(w | h)*scale/input_image_size
        #             = now*(w | h)*input_image_size/(max(w, h)*input_image_size)
        #             = now*(w | h)/max(w, h)
        max_w_h = max(w, h)
        w_scale = float(w) / max_w_h
        h_scale = float(h) / max_w_h
        
        scaled_boxes = boxes
        scaled_boxes[..., 0] *= w_scale
        scaled_boxes[..., 1] *= h_scale
        scaled_boxes[..., 2] *= w_scale
        scaled_boxes[..., 3] *= h_scale
        
        # 获取到在S x S 网格下的坐标
        # 并将scaled_labels 中的中心点坐标，转化成grid cell 的相对坐标
        scaled_boxes[..., :2] *= self.S
        grid_coord = scaled_boxes[..., :2].astype(int)
        scaled_boxes[..., :2] -= grid_coord
        
        # 用0填充目标标注
        target_label = np.zeros((self.S, self.S, 5 * self.B + self.C))
        
        # 遍历所有标注，为目标标注赋值
        for index, coord in enumerate(grid_coord.tolist()):
            label = np.zeros(5 * self.B + self.C)
            
            # 赋值类别
            class_id = classes[index]
            label[5 * self.B + class_id] = 1
            
            # 赋值bbox和置信度
            for i in range(self.B):
                label[i * 5:(i + 1) * 5 - 1] = scaled_boxes[index]
                label[(i + 1) * 5 - 1] = 1
                
            target_label[coord[0], coord[1], :] = label
    
        if self.transform:
            target_image = self.transform(target_image)
        if self.target_transform:
            target_label = self.target_transform(target_label)
        return target_image, target_label

class ArrayToTensor(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, arr):
        return torch.from_numpy(arr)
        

# 0 1 2 3 4 5 6 7
if __name__ == '__main__':
    img = Image.open(r'D:\longtime\dataset\yolov\images\new1.jpg')
    model = ToRequired()
    l = model(img, np.random.rand(2, 5))
    print(l)