import cv2
import os
from glob import glob
import random
import matplotlib.pyplot as plt 
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root_path',type=str , help="which should include ./images and ./labels and classes.txt")
parser.add_argument('--dt_path',type=str ,help="yolo format results of detection, include ./labels")
parser.add_argument('--conf' , type=float ,default=0.5, help="visulization conf thres")
arg = parser.parse_args()

colorlist = [
    # bgr
    (0,0,255),
    (0,128,255),
    (0,255,0),
    (255,0,0),
]

def plot_bbox(img_path, img_dir, out_dir, gt=None ,dt=None, cls2label=None, line_thickness=None):
    img = cv2.imread(os.path.join(img_dir, img_path))
    height, width,_ = img.shape
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
                 
    if dt:
        with open(dt,'r') as f:
            annotations = f.readlines()
            # print(annotations)    
            for ann in annotations:
                ann = list(map(float,ann.split()))
                ann[0] = int(ann[0])
                # print(ann)
                if len(ann) == 6:
                    cls,x,y,w,h,conf = ann
                    if conf < arg.conf:
                        # thres = 0.5
                        continue
                elif len(ann) == 5:
                    cls,x,y,w,h = ann
                color = colorlist[cls+2]

                c1, c2 = (int((x-w/2)*width),int((y-h/2)*height)), (int((x+w/2)*width), int((y+h/2)*height))
                cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

                # # cls label
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(cls2label[cls], 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                if len(ann) == 6:
                    cv2.putText(img, str(round(conf,2)), (c1[0], c1[1] - 2), 0, tl / 4, color, thickness=tf, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir,img_path),img)
    
if __name__ == "__main__":

    root_path = arg.root_path
    pred_path = arg.dt_path
    img_dir = os.path.join(root_path,'images')
    GT_dir = os.path.join(root_path,'labels')
    DT_dir = os.path.join(pred_path)
    out_dir = os.path.join(root_path,'outputs')
    cls_dir = os.path.join(root_path,'classes.txt')
    cls_dict = {}

    if not os.path.exists(img_dir):
        raise Exception("image dir {} do not exist!".format(img_dir))
    if not os.path.exists(cls_dir):
        raise Exception("class dir {} do not exist!".format(cls_dir))
    else:
        with open(cls_dir,'r') as f:
            classes = f.readlines()
            for i in range(len(classes)):
                cls_dict[i] = classes[i].strip()
            print(cls_dict)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    for each_img in tqdm(os.listdir(img_dir)):
        gt = None
        dt = None
        if os.path.exists(os.path.join(GT_dir,each_img.replace('jpg','txt'))):
            gt = os.path.join(GT_dir,each_img.replace('jpg','txt'))
        if os.path.exists(os.path.join(DT_dir,each_img.replace('jpg','txt'))):
            dt = os.path.join(DT_dir,each_img.replace('jpg','txt'))
        
        plot_bbox(each_img,img_dir, out_dir, gt, dt, cls2label=cls_dict)
        