import xml.etree.ElementTree as ET
import os

def voc_to_yolo(xml_file: str, classes: list[str]) -> list[list]:
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
    
    return annotations