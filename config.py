import os
from dataclasses import dataclass

@dataclass
class YOLOv1Config:
    """YOLOv1配置类"""
    # 数据配置
    data_dir: str = r'.\data\VOCdevkit2007\VOC2007'
    train_ann: str = r'.\data\VOCdevkit2007\VOC2007\ImageSets\Main\train.json'
    val_ann: str = r'.\data\VOCdevkit2007\VOC2007\ImageSets\Main\val.json'
    test_ann: str = 'test.json'
    
    # 模型配置
    S: int = 7
    B: int = 2
    C: int = 20
    
    # 训练配置
    epochs: int = 135
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 5e-4
    momentum: float = 0.9
    
    # 损失权重
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5
    lambda_class: float = 1.0
    
    # 设备配置
    device: str = 'cpu'
    num_workers: int = 4
    pin_memory: bool = True
    
    # 保存配置
    save_dir: str = './checkpoints'
    log_dir: str = './logs'
    save_interval: int = 10
    eval_interval: int = 5
    
    # 数据增强
    image_size: tuple = (448, 448)
    use_hflip: bool = True
    use_color_jitter: bool = True
    use_random_crop: bool = True
    
    # 其他
    seed: int = 42
    debug: bool = False
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建目录
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设置类别名称（Pascal VOC）
        self.class_names = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

# 创建不同的配置
def get_train_config():
    """获取训练配置"""
    return YOLOv1Config()