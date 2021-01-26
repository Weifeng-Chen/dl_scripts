import cv2
import os
from glob import glob
import random
import matplotlib.pyplot as plt 
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root_path',type=str,default=r'C:\Users\winner\Desktop\DL_tools\cart_yolov5l\round1', help="root path of images and labels")
arg = parser.parse_args()

colorlist = [
    # bgr
    (0,0,255),
    (0,128,255),
    (0,255,0),
    (255,0,0),
]

def plot_bbox(img_path, gt=None ,pred=None, cls2label=None, line_thickness=None):
   
    # Plots one bounding box on image img
    # print(img_path)
    img = cv2.imread(img_path)
    height, width,_ = img.shape
    # print(width,height)
    
    tl = line_thickness or round(0.002 * (width + height) / 2) + 1  # line/font thickness
    font = cv2.FONT_HERSHEY_SIMPLEX

    if gt:
        with open(gt,'r') as f:
            annotations = f.readlines()
            # print(annotations)    
            for ann in annotations:
                ann = list(map(float,ann.split()))
                ann[0] = int(ann[0])
                # print(ann)
                cls,x,y,w,h = ann
                color = colorlist[cls]
                c1, c2 = (int((x-w/2)*width),int((y-h/2)*height)), (int((x+w/2)*width), int((y+h/2)*height))
                cv2.rectangle(img, c1, c2, color, thickness=tl*2, lineType=cv2.LINE_AA)
                # 画出mask
                # zeros = np.zeros((img.shape), dtype=np.uint8)
                # gt_mask = cv2.rectangle(zeros, c1, c2,color=(0,0,255), thickness=-1) #thickness=-1 表示矩形框内颜色填充
                
                # cls label
                # tf = max(tl - 1, 1)  # font thickness
                # t_size = cv2.getTextSize(cls2label[cls], 0, fontScale=tl / 3, thickness=tf)[0]
                # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                # cv2.rectangle(gt_mask, c1, c2, color=(0,0,255), thickness=-1)  # filled
                # cv2.putText(gt_mask, cls2label[cls], (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

                # img = cv2.addWeighted(img, 1, gt_mask, 0.3, 0)
                # cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                 
    if pred:
        with open(pred,'r') as f:
            annotations = f.readlines()
            # print(annotations)    
            for ann in annotations:
                ann = list(map(float,ann.split()))
                ann[0] = int(ann[0])
                # print(ann)
                cls,x,y,w,h,conf = ann
                color = colorlist[cls+2]

                c1, c2 = (int((x-w/2)*width),int((y-h/2)*height)), (int((x+w/2)*width), int((y+h/2)*height))
                cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

                # # cls label
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(cls2label[cls], 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, str(round(conf,2)), (c1[0], c1[1] - 2), 0, tl / 4, color, thickness=tf, lineType=cv2.LINE_AA)

    
    cv2.imwrite(img_path.replace('images','outputs'),img)
    
if __name__ == "__main__":

    root_path = arg.root_path
    img_dir = os.path.join(root_path,'images')
    GT_dir = os.path.join(root_path,'labels')
    pred_dir = os.path.join(root_path,'preds')
    out_dir = os.path.join(root_path,'outputs')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    for each_img in tqdm(glob(os.path.join(img_dir,"*"))):
        gt = None
        pred = None
        if os.path.exists(each_img.replace('jpg','txt').replace('images','labels')):
            gt = each_img.replace('jpg','txt').replace('images','labels')
        if os.path.exists(each_img.replace('jpg','txt').replace('images','preds')):
            pred = each_img.replace('jpg','txt').replace('images','preds')
        plot_bbox(each_img,gt,pred,cls2label={0:'head',1:'body'})
        