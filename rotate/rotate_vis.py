import os
import cv2
import numpy as np


def plot_rotate_bbox(image, pts):
    """
    input:
        image -> read from cv2
        pts -> [cls,xy0,xy1,xy2,xy3]
    output:
        image with rotated bbox 
    """
    LINE_WIDTH = 2
    image = image.copy()
    h,w,_ = image.shape
    # print(h,w)
    
    # 结果绘图
    for pt in pts:
        draw_color = (0, 0, 0)
        pt = [eval(xy) for xy in pt]
        cls = pt[0]
        
        xy = np.array(pt[1:])
        xy[0:7:2] *= w
        xy[1:8:2] *= h
        # print(cls)
        if cls == 0:
            draw_color = (255, 0, 0)
        elif cls == 1:
            draw_color = (128, 0, 128)
        elif cls == 2:
            draw_color = (0, 255, 0)
        
        # print(xy)
        xy = list(map(int,xy))
        for i in range(0,5,2):
            cv2.line(image, (xy[i],xy[i+1]), (xy[i+2],xy[i+3]), draw_color, LINE_WIDTH)
        cv2.line(image, (xy[6],xy[7]), (xy[0],xy[1]), draw_color, LINE_WIDTH)
    return image

if __name__ =="__main__":
    img_path = '/home/winner/chenwf/rotate-yolov5/data/pedestrian/sample/images/IP9_192.168.100.199_192.168.100.199_20201219075953_20201219115357_1_4G_248695_a10.jpg'
    label_path = img_path.replace('images','labels').replace('.jpg','.txt')

    image = cv2.imread(img_path)
    with open(label_path,'r') as f:
        pts = f.readlines()
        pts = [each.split() for each in pts]

        labelled_img = plot_rotate_bbox(image,pts)
        cv2.imwrite('vis.jpg', labelled_img)



