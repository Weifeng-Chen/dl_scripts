"""
yolo格式数据，裁剪图像中心区域，生成一批新数据。
"""

import cv2
import os 
from tqdm import tqdm


def plot_bbox(img, gt=None ,line_thickness=None):
    # 可视化测试
    colorlist = []
    # 5^3种颜色。
    for i in range(30,256,50):
        for j in range(40,256,50):
            for k in range(50,256,50):
                colorlist.append((i,j,k))

    height, width,_ = img.shape
    tl = line_thickness or round(0.002 * (width + height) / 2) + 1  # line/font thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    tf = max(tl - 1, 1)  # font thickness
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
    return img

def parse_label(gt, crop_ratio=0.25,):
    scale_ratio = 1/(1-crop_ratio*2)
    out_str = ''
    with open(gt,'r') as f:
        annotations = f.readlines()
        # print(annotations)    
        for ann in annotations:
            ann = list(map(float,ann.split()))
            # print(ann)
            if crop_ratio < ann[1] < 1-crop_ratio and crop_ratio < ann[2] < 1-crop_ratio:
                # center point in the specified area
                # print(ann)
                out_l = [int(ann[0]), ann[1]-crop_ratio, ann[2]-crop_ratio, ann[3], ann[4]]
                out_l[1:] = [out*scale_ratio for out in out_l[1:]]
                out_l = list(map(str,out_l))
                out_str += ' '.join(out_l) +'\n'
    return out_str
    


if __name__ == '__main__':
    origin_root_dir = '/home/winner/chenwf/yolov5/data/pedestrian/train'
    save_dir = '/home/winner/chenwf/yolov5/data/pedestrian/train_crop'
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir,'images'))
        os.makedirs(os.path.join(save_dir,'labels'))

    img_dir = os.path.join(origin_root_dir,'images')
    label_dir = os.path.join(origin_root_dir,'labels')
    img_names = os.listdir(img_dir)
    # crop ratio
    crop_ratio = 0.25

    for img_name in tqdm(img_names):
        label_name = img_name.replace('jpg','txt')
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, label_name)

        img = cv2.imread(img_path)
        height,width,_ = img.shape
        out_str = parse_label(label_path, crop_ratio)

        if out_str:
            # 空样本就不添加了
            with open(os.path.join(save_dir, 'labels', label_name),'w') as f:
                # write
                f.write(out_str)
            # crop
            crop_img = img[int(height*crop_ratio):height-int(height*crop_ratio),int(width*crop_ratio):width-int(width*crop_ratio),:]
            # plot_bbox(crop_img, os.path.join(save_dir, 'labels', label_name)) # visualize the bbox
            cv2.imwrite(os.path.join(save_dir, 'images', img_name),crop_img)

        
    
