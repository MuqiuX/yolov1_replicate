import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import voc_to_yolo, visualize_with_albumentations
from transforms import ToV1Size

image_path = r'D:\longtime\yolov1_reproduce\data\VOCdevkit2007\VOC2007\JPEGImages\000005.jpg'
label_path = r'D:\longtime\yolov1_reproduce\data\VOCdevkit2007\VOC2007\Annotations\000005.xml'

image = cv2.imread(image_path)
label = voc_to_yolo(label_path, classes=[
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ])

tr = ToV1Size()

image, label = tr(image=image, label=label)

visualize_with_albumentations(
    image=image,
    bboxes=label[..., 1:],
    class_labels=label[..., 0],
    classes=np.array([
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ])
)

print('')
