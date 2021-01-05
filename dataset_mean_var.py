"""
计算数据集图片标准差和均值。
为了缩短计算时间，只挑选1/10进行计算。
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2

filepath = 'CenterNet/data/pedestrian/images'  # 数据集目录
pathDir = os.listdir(filepath)
 
R_channel = 0
G_channel = 0
B_channel = 0
cnt = 0
for idx in tqdm(range(0, len(pathDir), 10)):
    cnt+=1
    filename = pathDir[idx]
    
    img = cv2.imread(os.path.join(filepath, filename)) 
    img = cv2.resize(img,(512,512)) /255.0
    # print(np.max(img))
    b, g, r = cv2.split(img)

    R_channel = R_channel + np.sum(r)
    G_channel = G_channel + np.sum(g)
    B_channel = B_channel + np.sum(b)

num = cnt * 512 * 512  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

print(R_mean,G_mean,B_mean) 
R_channel = 0
G_channel = 0
B_channel = 0
for idx in tqdm(range(0, len(pathDir), 10)):
    filename = pathDir[idx]
    img = cv2.imread(os.path.join(filepath, filename)) 
    img = cv2.resize(img,(512,512)) /255.0
    b, g, r = cv2.split(img)
    
    R_channel = R_channel + np.sum((r - R_mean) ** 2)
    G_channel = G_channel + np.sum((g - G_mean) ** 2)
    B_channel = B_channel + np.sum((b - B_mean) ** 2)
 
R_var = np.sqrt(R_channel / num)
G_var = np.sqrt(G_channel / num)
B_var = np.sqrt(B_channel / num)
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))