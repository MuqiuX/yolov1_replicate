from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
from transforms import ToRequired
from typing import Literal
import albumentations as A
from utils import voc_to_yolo

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
        type: Literal['val', 'train'],
        classes: list[str],
        transform=None,
        target_transform=None
    ):
        self.label_path = label_path
        self.image_path = image_path
        
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        self.type = type

        # 加载训练数据集列表
        self.names = []
        self.load_names()
        
        self.to_yolov1_annotation_model = ToRequired(transform=transform, target_transform=target_transform)
    
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        # 加载图片和标注
        img_path = os.path.join(self.images_dir, self.names[idx] + '.jpg')
        label_path = os.path.join(self.annotations_dir, self.names[idx] + '.xml')
        
        image = Image.open(img_path).convert('RGB')
        labels = self.parse_xml(label_path)
        
        transform = A.Compose([
            A.Resize(height=512, width=512),
            A.HorizontalFlip(p=0.5)
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['bbox_class'],
            clip=True
        ))
        
        transform()
        
        f_image, f_target = self.to_yolov1_annotation_model(image, labels)
        
        return f_image, f_target.float()
    
    def load_names(self, path: str = None):
        '''
        load_train_names 的 Docstring
            
        :param self: 说明
        :param path: 训练数据集文件， 如果没有指定则使用默认
        :type path: str
        '''
        if path is None or (path and not path.endswith('.txt')):
            name_list_file = os.path.join(self.root, 'ImageSets', 'Main', f'{self.type}.txt')
        else:
            name_list_file = path
        
        if not os.path.exists(name_list_file):
            raise FileNotFoundError(f'训练数据集文件不存在: {name_list_file}')
        
        self.names.clear()
        
        with open(name_list_file, 'r', encoding='utf-8') as f:
            self.names.extend(
                line.strip()
                for line in f
                if line.strip()
            )
    
def get_dataloader(args):
    train_dataset = YOLODataset(
        voc_root=args['voc_root'],
        classes=args['classes'],
        type='train',
        transform=args['transform'],
        target_transform=args['target_transform']
    )
    
    val_dataset = YOLODataset(
        voc_root=args['voc_root'],
        classes=args['classes'],
        type='val',
        transform=args['transform'],
        target_transform=args['target_transform']
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