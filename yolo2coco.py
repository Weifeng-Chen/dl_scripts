"""
YOLO 格式的数据集转化为 COCO 格式的数据集
--root_path 输入根路径
"""

import os
import cv2
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser("ROOT SETTING")
parser.add_argument('--root_path',type=str,default='coco', help="root path of images and labels")
arg = parser.parse_args()

# 默认划分比例为 8:1:1。 第一个划分点在8/10处，第二个在9/10。
VAL_SPLIT_POINT = 4/5
TEST_SPLIT_POINT = 9/10

root_path = arg.root_path
print(root_path)

# 原始标签路径
originLabelsDir = os.path.join(root_path, 'labels')                                        
# 原始标签对应的图片路径
originImagesDir = os.path.join(root_path, 'images')

# dataset用于保存所有数据的图片信息和标注信息
train_dataset = {'categories': [], 'annotations': [], 'images': []}
val_dataset = {'categories': [], 'annotations': [], 'images': []}
test_dataset = {'categories': [], 'annotations': [], 'images': []}

# 打开类别标签
with open(os.path.join(root_path, 'classes.txt')) as f:
    classes = f.read().strip().split()

# 建立类别标签和数字id的对应关系
for i, cls in enumerate(classes, 1):
    train_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    val_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    test_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

# 读取images文件夹的图片名称
indexes = os.listdir(originImagesDir)
# ---------------接着将，以上数据转换为COCO所需要的格式---------------
for k, index in enumerate(tqdm(indexes)):
    txtFile = index.replace('images','txt').replace('jpg','txt')

    # 用opencv读取图片，得到图像的宽和高
    im = cv2.imread(os.path.join(root_path, 'images/') + index)
    height, width, _ = im.shape

    # 切换dataset的引用对象，从而划分数据集
    if k+1 > round(len(indexes)*VAL_SPLIT_POINT):
        if k+1 > round(len(indexes)*TEST_SPLIT_POINT):
            dataset = test_dataset
        else:
            dataset = val_dataset
    else:
        dataset = train_dataset

    # 添加图像的信息到dataset中
    dataset['images'].append({'file_name': index,
                                'id': k,
                                'width': width,
                                'height': height})

    with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
        labelList = fr.readlines()
        for label in labelList:
            label = label.strip().split()
            x = float(label[1])
            y = float(label[2])
            w = float(label[3])
            h = float(label[4])

            # convert x,y,w,h to x1,y1,x2,y2
            imagePath = os.path.join(originImagesDir,
                                        txtFile.replace('txt', 'jpg'))
            image = cv2.imread(imagePath)
            H, W, _ = image.shape
            x1 = (x - w / 2) * W
            y1 = (y - h / 2) * H
            x2 = (x + w / 2) * W
            y2 = (y + h / 2) * H
            # 为了与coco标签方式对，标签序号从1开始计算
            cls_id = int(label[0]) + 1        
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': int(cls_id),
                'id': i,
                'image_id': k,
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })

# 保存结果的文件夹
folder = os.path.join(root_path, 'annotations')
if not os.path.exists(folder):
    os.makedirs(folder)
for phase in ['train','val','test']:
    json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
    with open(json_name, 'w') as f:
        if phase == 'train':
            json.dump(train_dataset, f)
        if phase == 'val':
            json.dump(val_dataset, f)
        if phase == 'test':
            json.dump(test_dataset, f)
