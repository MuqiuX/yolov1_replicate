from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from typing import Literal
from utils import voc_to_yolo
import cv2
from transforms import ToV1Label, ToV1Size

class YOLODataset(Dataset):
    """加载voc数据集，并且转化成yolo标注格式

    Args:
        label_path (str): 标注文件夹路径
        image_path (str): 图片文件夹路径
        type (Literal[&#39;val&#39;, &#39;train&#39;]): 类型
        classes (list[str]): 数据集对象列表
        transform (_type_, optional): 图片转化. Defaults to None.
        target_transform (_type_, optional): 标注转化. Defaults to None.
    """

    def __init__(
        self,
        label_path: str,
        image_path: str,
        image_set_path: str,
        type: Literal['val', 'train'],
        classes: list[str]
    ):
        self.label_path = label_path
        self.image_path = image_path
        
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        self.type = type
        
        self.transform = ToV1Size()
        self.target_transform = ToV1Label()

        self.names = []
        self.load_names(image_set_path)
    
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        
        img_file = os.path.join(self.image_path, self.names[idx] + '.jpg')
        label_file = os.path.join(self.label_path, self.names[idx] + '.xml')
        
        image = cv2.imread(img_file)
        label = voc_to_yolo(label_file, self.classes)
        
        if self.transform != None:
            image, label = self.transform(
                image=image,
                label=label
            )
        
        if self.target_transform != None:
            label = self.target_transform(
                label=label
            )
        
        return image.float(), label.float()
    
    def load_names(self, image_set: str):
        
        name_list_file = os.path.join(image_set, f'{self.type}.txt')
        
        if not os.path.exists(name_list_file):
            raise FileNotFoundError(f'数据集文件不存在: {name_list_file}')
        
        self.names.clear()
        
        with open(name_list_file, 'r', encoding='utf-8') as f:
            self.names.extend(
                line.strip()
                for line in f
                if line.strip()
            )
    
def get_dataloader(args: dict):
    train_dataset = YOLODataset(
        image_path=args['image_dir'],
        image_set_path=args['image_set_dir'],
        label_path=args['ann_dir'],
        classes=args['class_names'],
        type='train'
    )
    
    val_dataset = YOLODataset(
        image_path=args['image_dir'],
        image_set_path=args['image_set_dir'],
        label_path=args['ann_dir'],
        classes=args['class_names'],
        type='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        shuffle=True,
        num_workers=args['num_workers'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['num_workers'],
    )
    
    return train_loader, val_loader