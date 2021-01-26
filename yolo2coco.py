"""
YOLO 格式的数据集转化为 COCO 格式的数据集
--root_path 输入根路径
--save_name 保存文件的名字(没有random_split时使用)
"""

import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path',type=str, help="root path of images and labels")
parser.add_argument('--random_split',action='store_true', help="random split the dataset, default ratio is 8:1:1")
parser.add_argument('--save_name',type=str,default='train.json',action='store_true', help="if not split the dataset, specify where to save the result")
arg = parser.parse_args()


def train_test_val_split(img_paths,ratio_train=0.8,ratio_test=0.1,ratio_val=0.1):
    # 这里可以修改数据集划分的比例。
    assert int(ratio_train+ratio_test+ratio_val) == 1
    train_img, middle_img = train_test_split(img_paths,test_size=1-ratio_train, random_state=233)
    ratio=ratio_val/(1-ratio_train)
    val_img, test_img  =train_test_split(middle_img,test_size=ratio, random_state=233)
    print("NUMS of train:val:test = {}:{}:{}".format(len(train_img), len(val_img), len(test_img)))
    return train_img, val_img, test_img


def yolo2coco(root_path, random_split):
    # 标签路径
    originLabelsDir = os.path.join(root_path, 'labels')                                        
    # 图片路径
    originImagesDir = os.path.join(root_path, 'images')
    # 类别
    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = f.read().strip().split()
    # 读取images文件夹的图片名称
    indexes = os.listdir(originImagesDir)

    if random_split:
        # 用于保存所有数据的图片信息和标注信息
        train_dataset = {'categories': [], 'annotations': [], 'images': []}
        val_dataset = {'categories': [], 'annotations': [], 'images': []}
        test_dataset = {'categories': [], 'annotations': [], 'images': []}

        # 建立类别标签和数字id的对应关系, COCO的类别id从1开始。
        for i, cls in enumerate(classes, 1):
            train_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            val_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
            test_dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

        train_img, val_img, test_img = train_test_val_split(indexes,0.8,0.1,0.1)
    else:
        dataset = {'categories': [], 'annotations': [], 'images': []}
        for i, cls in enumerate(classes, 1):
            dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})
    
    # annotations id
    ann_id_cnt = 0

    # ---开始转换数据---
    for k, index in enumerate(tqdm(indexes)):
        txtFile = index.replace('images','txt').replace('.jpg','.txt').replace('.png','.txt')

        # 读取图片，得到图像的宽和高
        im = cv2.imread(os.path.join(root_path, 'images/') + index)
        height, width, _ = im.shape

        if random_split:
            # 切换dataset的引用对象，从而划分数据集
                if index in train_img:
                    dataset = train_dataset
                elif index in val_img:
                    dataset = val_dataset
                elif index in test_img:
                    dataset = test_dataset

        # 添加图像的信息到dataset中
        dataset['images'].append({'file_name': index,
                                    'id': k,
                                    'width': width,
                                    'height': height})

        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            # 如没标签，跳过，只保留图片信息。
            continue

        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                # imagePath = os.path.join(originImagesDir,
                                            # txtFile.replace('txt', 'jpg'))
                # image = cv2.imread(imagePath)
                # H, W, _ = image.shape
                H, W, _ = im.shape
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
                    'id': ann_id_cnt,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果
    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if random_split:
        for phase in ['train','val','test']:
            json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
            with open(json_name, 'w') as f:
                if phase == 'train':
                    json.dump(train_dataset, f)
                elif phase == 'val':
                    json.dump(val_dataset, f)
                elif phase == 'test':
                    json.dump(test_dataset, f)
            print('Save annotation to {}'.format(json_name))
    else:
        json_name = os.path.join(root_path, 'annotations/{}'.format(arg.save_name))
        with open(json_name, 'w') as f:
            json.dump(dataset, f)
            print('Save annotation to {}'.format(json_name))

if __name__ == "__main__":
    root_path = arg.root_path
    assert os.path.exists(root_path)
    random_split = arg.random_split
    print("Loading data from ",root_path,"\nWhether to split the data:",random_split)
    yolo2coco(root_path,random_split)