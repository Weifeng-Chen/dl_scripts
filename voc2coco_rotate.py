"""
VOC 格式的数据集转化为 COCO 格式的数据集。（旋转） 生成的旋转框是以中心点顺时针旋转的角度（0~pi）
--xml_path 输入根路径
--save_path 保存文件的名字(没有random_split时使用)
"""

import json
import cv2
import numpy as np
import glob
import PIL.Image
import os,sys
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xml_path', default='./voc_ro',type=str, help="coco format(json)")
parser.add_argument('--save_path', default='train.json', type=str, help="save to a json file")
arg = parser.parse_args()

class PascalVOC2coco(object):
    def __init__(self, xml=[], save_json_path='./new.json'):
        '''
        :param xml: 所有Pascal VOC的xml文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.xml = xml
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.ob = []
        self.save_json()
 
    def data_transfer(self):
        for num, json_file in enumerate(self.xml):
            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d' % (
                num + 1, len(self.xml)))
            sys.stdout.flush()
            self.json_file = json_file
            self.num = num 
            path = os.path.dirname(self.json_file)
            path = os.path.dirname(path)

            with open(json_file, 'r', encoding='UTF-8') as fp:
                flag = 0
                for p in fp:
                    f_name = 1
                    if 'filename' in p:
                        self.filen_ame = p.split('>')[1].split('<')[0]
                        f_name = 0
                        self.path = os.path.join(path, 'SegmentationObject', self.filen_ame.split('.')[0] + '.png')
                    if 'width' in p:
                        self.width = int(p.split('>')[1].split('<')[0])
                    if 'height' in p:
                        self.height = int(p.split('>')[1].split('<')[0])
                        self.images.append(self.image())
                    if flag == 1:
                        self.supercategory = self.ob[0]
                        if self.supercategory not in self.label:
                            self.categories.append(self.categorie())
                            self.label.append(self.supercategory)
                        x1 = float(self.ob[1])
                        y1 = float(self.ob[2])
                        w = float(self.ob[3])
                        h = float(self.ob[4])
                        # angle = float(self.ob[5])*180/math.pi
                        radian = float(self.ob[5])
                        # 把2pi的转化为pi的。
                        angle = radian if radian < math.pi else radian-math.pi
                        
                        self.rectangle = [x1, y1, x1+w, y1+w]
                        self.bbox = [x1, y1, w, h, angle]  # COCO 对应格式[x,y,w,h]
 
                        self.annotations.append(self.annotation())
                        self.annID += 1
                        self.ob = []
                        flag = 0
                    elif f_name == 1:
                        key = p.split('>')[0].split('<')[1]
                        print(key)
                        if key == 'name':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'cx':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'cy':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'w':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'h':
                            self.ob.append(p.split('>')[1].split('<')[0])
                        if key == 'angle':
                            self.ob.append(p.split('>')[1].split('<')[0])
                            flag = 1
 
 
        sys.stdout.write('\n')
        sys.stdout.flush()
 
    def image(self):
        image = {}
        image['height'] = self.height
        image['width'] = self.width
        image['id'] = self.num + 1
        image['file_name'] = self.filen_ame
        return image
 
    def categorie(self):
        categorie = {}
        categorie['supercategory'] = self.supercategory
        categorie['id'] = len(self.label) + 1  # 0 默认为背景
        categorie['name'] = self.supercategory
        return categorie
 
    def annotation(self):
        annotation = {}

        annotation['segmentation'] = [list(map(float, self.getsegmentation()))]
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.num + 1
        annotation['area'] = 1920 * 1080             # 这个标注其实也没啥用-。-
        annotation['rbbox'] = self.bbox
        annotation['category_id'] = self.getcatid(self.supercategory)
        annotation['id'] = self.annID
        return annotation
 
    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1
 
    def getsegmentation(self):
 
        try:
            mask_1 = cv2.imread(self.path, 0)
            mask = np.zeros_like(mask_1, np.uint8)
            rectangle = self.rectangle
            mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = mask_1[rectangle[1]:rectangle[3],
                                                                         rectangle[0]:rectangle[2]]
 
            # 计算矩形中点像素值
            mean_x = (rectangle[0] + rectangle[2]) // 2
            mean_y = (rectangle[1] + rectangle[3]) // 2
 
            end = min((mask.shape[1], int(rectangle[2]) + 1))
            start = max((0, int(rectangle[0]) - 1))
 
            flag = True
            for i in range(mean_x, end):
                x_ = i
                y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:  # 0 对应背景 220对应边界线
                    mask = (mask == pixels).astype(np.uint8)
                    flag = False
                    break
            if flag:
                for i in range(mean_x, start, -1):
                    x_ = i
                    y_ = mean_y
                    pixels = mask_1[y_, x_]
                    if pixels != 0 and pixels != 220:
                        mask = (mask == pixels).astype(np.uint8)
                        break
            self.mask = mask
 
            return self.mask2polygons()
 
        except:
            return [0]
 
    def mask2polygons(self):
        contours = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找到轮廓线
        bbox=[]
        for cont in contours[1]:
            [bbox.append(i) for i in list(cont.flatten())]
            # map(bbox.append,list(cont.flatten()))
        return bbox # list(contours[1][0].flatten())
 
  
    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco
 
    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # indent=4 更加美观显示
 
if __name__ == '__main__':
    xml_path = arg.xml_path
    xml_file = glob.glob(os.path.join(xml_path,'*.xml'))
    PascalVOC2coco(xml_file, arg.save_path)