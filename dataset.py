from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import xml.etree.ElementTree as ET
from transforms import ToRequired
from typing import Literal

class YOLODataset(Dataset):

    def __init__(self, voc_root: str, type: Literal['val', 'train'], classes: list[str], transform=None, target_transform=None):
        '''
        asdawda
        
        :param root: 指向voc数据集Annotations等内容所在的文件夹
        :param classes: 类别onehot映射
        :param transform: 图片转化
        :param label_transform: 标注转化
        '''
        self.root = voc_root # voc 数据集路径
        self.annotations_dir = os.path.join(voc_root, 'Annotations') # 标注文件夹
        self.images_dir = os.path.join(voc_root, 'JPEGImages') # 图片文件夹
        
        self.classes = classes # 类别onehot映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        self.type = type

        self.names = [] # 训练数据集列表
        self.load_names() # 加载训练数据集列表
        
        self.to_yolov1_annotation_model = ToRequired(transform=transform, target_transform=target_transform)
        
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
        
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.names[idx] + '.jpg')
        label_path = os.path.join(self.annotations_dir, self.names[idx] + '.xml')
        
        image = Image.open(img_path).convert('RGB')
        
        target = self.parse_annotation_to_yolo(label_path)
        
        f_image, f_target = self.to_yolov1_annotation_model(image, target)
        
        return f_image, f_target.float()
    
    def parse_annotation_to_yolo(self, path: str):
        
        if not os.path.exists(path):
            raise FileNotFoundError(f'标注文件不存在: {path}')
        
        tree = ET.parse(path)
        root = tree.getroot()
        
        # 获取图片尺寸
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # yolo标注
        annotations = []
        
        # 遍历所有的object
        for obj in root.findall('object'):
            # 获取对象名字
            class_name = obj.find('name').text.lower()
            # 如果在对象列表中不存在直接跳过该object
            if class_name not in self.class_to_idx:
                continue
            
            # 获取索引
            class_idx = self.class_to_idx[class_name]
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 获取中心点归一化坐标
            x_center = (xmin + xmax) / float(2 * width)
            y_center = (ymin + ymax) / float(2 * height)
            
            # 获取标注框，归一化宽高
            box_w = (xmax - xmin) / float(width)
            box_h = (ymax - ymin) / float(height)
            
            annotations.append([class_idx, x_center, y_center, box_w, box_h])
        
        return annotations
    
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