"""
计算数据集图片标准差和均值。
--file_path是图片地址
--step是选择图片的间隔，如间隔为10，则只计算1/10
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import argparse

parser = argparse.ArgumentParser("ROOT SETTING")
parser.add_argument('--file_path',type=str,default='data/pedestrian/images' , help="root path of images and labels")
parser.add_argument('--step',type=int,default=1 , help="whether to pick interval images")

arg = parser.parse_args()

filepath = arg.file_path  # 数据集目录
STEP = arg.step
pathDir = os.listdir(filepath)
 

means = [0 for i in range(3)]
stds = [0 for i in range(3)]
cnt = 0
for idx in tqdm(range(0, len(pathDir), STEP)):
    cnt+=1
    filename = pathDir[idx]
    img = cv2.imread(os.path.join(filepath, filename)) 
    img = img /255.0
    b, g, r = cv2.split(img)
    means[0] += np.mean(r)
    means[1] += np.mean(g)
    means[2] += np.mean(b)
means = np.array(means) / cnt

# std要另外算，计算减去的均值是所有图片的均值，而不是某张图片的均值。
for idx in tqdm(range(0, len(pathDir), STEP)):
    filename = pathDir[idx]
    img = cv2.imread(os.path.join(filepath, filename)) 
    img = img /255.0
    b, g, r = cv2.split(img)
    stds[0] += np.mean((r - means[0]) ** 2)
    stds[1] += np.mean((g - means[1]) ** 2)
    stds[2] += np.mean((b - means[2]) ** 2)
stds = np.sqrt(np.array(stds) / cnt)

print("RGB MEAN:",means,"RBG STD:",stds) 
