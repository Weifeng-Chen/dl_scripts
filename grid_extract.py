"""
输入
    annotation的文件list
    参数1： w,h 网格的数量
    参数2： 指定网格区间

输出： 
    1. 有在slices区间的标签文件的路径，output.txt
    2. 网络伪彩图。

"""
import cv2
import argparse
import time
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--num_w',type=int,default=5, help="宽边网格数")
parser.add_argument('--num_h',type=int,default=5, help="高边网格数")
parser.add_argument('--slices',type=str,default="0 1 0 2", help="按[r1,r2) , [c1,c2) 给定行,列范围。左上角为起点,输入4个数字，空格为分隔符")
parser.add_argument('--dirs',type=str,default="", help="读文件夹，暂时空着")

arg = parser.parse_args()

num_class = 3

def save_gird_heatmap(anns,num_w,num_h):
    # 输出频率分布图
    factor_w = 1/num_w
    factor_h = 1/num_h

    if not os.path.exists("./output_hm"):
        os.mkdir("./output_hm")
    
    for ann_path in tqdm(anns):
        cls_dict = {}
        for i in range(num_class):
            cls_dict[i] = []
        rows = []
        # print(grid_matrix)
        with open(ann_path) as f:
            rows = f.readlines()
        for row in rows:
            row = row.split()
            cls, center_x, center_y = int(row[0]), float(row[1]), float(row[2])
            cls_dict[cls].append([center_x,center_y])
        # print(cls_dict)
        plt.figure(figsize=(9,3))
        # fig, ax = plt.subplots(1,3)
        # fig.suptitle(ann_path)
        for cls in range(num_class):
            cls_rows = cls_dict[cls]

            if not cls_rows:
                continue
            cls_rows = np.array(cls_rows)
            # print(cls_rows)   # 所有中心点的坐标
            grid_matrix = np.zeros((num_h,num_w)) # 起点是左上角。行数=h，列数=w
            
            cls_rows[:,0] //= factor_w
            cls_rows[:,1] //= factor_h
            cls_rows = cls_rows.astype(int)  # 转换为在网格的位置
            # print(cls_rows)
            for (x,y) in cls_rows:
                # 投票
                grid_matrix[y, x] += 1  # 这里要注意坐标，坐标(x,y)的位置应该是在 第y行, 第x列
            # print(grid_matrix)
            
            plt.subplot(1,3,cls+1)
            plt.imshow(grid_matrix)
            plt.axis('off') 
            plt.colorbar()
            plt.title('class={}'.format(cls))
            
        plt.suptitle(ann_path) # 总标题
        plt.tight_layout()
        # plt.show() 
        plt.savefig(os.path.join("./output_hm","{}.jpg".format(time.time())))
        

def save_gird_output(anns,num_w, num_h, slice):
    # 切片结果输出
    factor_w = 1/num_w
    factor_h = 1/num_h
    row1,row2, cow1,cow2 = slice

    f_write = open('output.txt','w')
    for ann_path in anns:
        rows = []
        with open(ann_path) as f:
            rows = f.readlines()
        for row in rows:
            row = row.split()
            center_x, center_y = float(row[1]), float(row[2])
        
            belong_y = int(center_x // factor_w)  # 列位置。行列与坐标是反过来的。
            belong_x = int(center_y // factor_h) # 行位置。

            # print("=========>",center_x, center_y , belong_x, belong_y)
            if belong_x < row2 and belong_x >= row1 and belong_y < cow2 and belong_y >= cow1:
                f_write.write(ann_path)
                f_write.write("\n")
                break
                # print("hit")
    f_write.close()
        

if __name__ == "__main__":
    
    
    anns_list = []  # 此处读取所有标签的文件名
    anns_list =  glob(os.path.join(r'C:\Users\winner\Desktop\DL_tools\labels','*.txt')) # 我自己测试用的（最好用绝对地址来读取）。


    num_w, num_h, slices = arg.num_w,arg.num_h,arg.slices
    slice = list(map(int,slices.split(" ")))  # r1 r2 c1 c2 第r1行到第r2行；第c1列到c2列。
    save_gird_heatmap(anns_list,num_w, num_h )
    save_gird_output(anns_list,num_w, num_h,slice)
