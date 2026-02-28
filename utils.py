import xml.etree.ElementTree as ET
import os
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_with_albumentations(
    image: np.ndarray,
    bboxes: np.ndarray,
    class_labels: np.ndarray,
    classes: np.ndarray
    ):
    
    # 空转化， 裁剪
    visualize = A.Compose([
        A.NoOp()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    result = visualize(image=image, bboxes=bboxes, class_labels=class_labels)
    
    fig, ax = plt.subplots(1, figsize=(5, 5))
    
    ax.imshow(result['image'])
    
    for bbox, label in zip(result['bboxes'], result['class_labels']):
        x_center, y_center, width, height = bbox
        h, w = image.shape[:2]
        
        x_min = (x_center - width / 2) * w
        y_min = (y_center - height / 2) * h
        x_max = width * w
        y_max = height * h
        
        rect = plt.Rectangle((x_min, y_min), x_max, y_max, 
                            fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_min, y_min, classes[int(label)], 
               color='red', fontsize=12, weight='bold')
    
    plt.axis('off')
    plt.show()

def voc_to_yolo(xml_file: str, classes: list[str]) -> np.ndarray:
    """读取voc数据集标注文件，转化成yolo标注格式返回

    Args:
        xml_file (str): voc标注文件路径
        classes (list[str]): 数据集对象列表

    Raises:
        FileNotFoundError: 指定标注文件不存在

    Returns:
        list[list]: 返回一个数据集列表
    """
    
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f'标注文件不存在: {xml_file}')
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 获取图片尺寸
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # 类别标注和bbox标注（pascal voc）
    annotations = []
    
    # 遍历所有的object
    for obj in root.findall('object'):
        # 获取对象名字, 如果在对象列表中不存在直接跳过该object
        class_name = obj.find('name').text.lower()
        if class_name not in classes:
            continue
        
        class_idx = classes.index(class_name)
        
        # 获取标注框，将标注转化成yolo格式
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        x_center = (xmin + xmax) / float(2 * width)
        y_center = (ymin + ymax) / float(2 * height)
        
        box_w = (xmax - xmin) / float(width)
        box_h = (ymax - ymin) / float(height)
        
        annotations.append([class_idx, x_center, y_center, box_w, box_h])
    
    return np.array(annotations)