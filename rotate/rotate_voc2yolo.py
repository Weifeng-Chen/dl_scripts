"""
5点定义的（相比4点多了一个中心位置，可以互转） Bounding Box 转为 类YOLO格式的标注(x,y,w,h,angle)
angle的定义：0~2pi, 不旋转的角度为0, 顺时针旋转为正,最多旋转一圈2pi. （角度会区分方向）
与 https://github.com/cgvict/roLabelImg 定义的一样。
"""

import os 
import cv2
import matplotlib.pyplot as plt
from pdv_labels_xml import PDVLabel, PDVObjHead, PDVObjHbody, PDVObjFbody
import numpy as np
from tqdm import tqdm

"""
[in] label_data, 标注数据，字典类型
    pts，顶点数据，numpy数据类型，顺序为[[[cx, cy], [tl_x, tl_y], [tr_x, tr_y], [br_x, br_y], [bl_x, bl_y]],...],shape 为 [目标数量,5,2]
    cat, 类别数据，numpy数据类型，0：人头；1：头肩；2：全身。shape 为[目标数量,]
"""

def convert_raw_label_to_vertex_pts(image, raw_object_dict):
    label_data = {}
    valid_pts = []
    valid_cat = []
    img_h, img_w, img_c = image.shape
    for object_name, object_list in raw_object_dict.items():
        if object_name == "head" or object_name == "hbody":
            for l_obj in object_list:
                cx = img_w*float(l_obj["cx"])
                cy = img_h*float(l_obj["cy"])
                w = img_w*float(l_obj["w"])
                h = img_h*float(l_obj["h"])
                tl_x = max(0, cx-w/2)
                tl_y = max(0, cy-h/2)
                tr_x = min(img_w-1, tl_x+w)
                tr_y = max(0, cy-h/2)
                br_x = min(img_w-1, tl_x+w)
                br_y = min(img_h-1, tl_y+h)
                bl_x = max(0, cx-w/2)
                bl_y = min(img_h-1, tl_y+h)

                valid_pts.append([[cx, cy], [tl_x, tl_y], [tr_x, tr_y], [
                    br_x, br_y], [bl_x, bl_y]])
                if object_name == "head":
                    valid_cat.append(0)
                elif object_name == "hbody":
                    valid_cat.append(1)

        elif object_name == "fbody":
            for l_obj in object_list:
                cx = img_w*float(l_obj["cx"])
                cy = img_h*float(l_obj["cy"])
                tl_x = img_w*float(l_obj["tlx"])
                tl_y = img_h*float(l_obj["tly"])
                tr_x = img_w*float(l_obj["trx"])
                tr_y = img_h*float(l_obj["try"])
                br_x = img_w*float(l_obj["brx"])
                br_y = img_h*float(l_obj["bry"])
                bl_x = img_w*float(l_obj["blx"])
                bl_y = img_h*float(l_obj["bly"])

                valid_pts.append([[cx, cy], [tl_x, tl_y], [tr_x, tr_y], [
                    br_x, br_y], [bl_x, bl_y]])
                valid_cat.append(2)
    label_data["pts"] = np.asarray(valid_pts, np.float32)
    label_data["cat"] = np.asarray(valid_cat, np.float32)
    return label_data


def convert_vertex_pts_to_cxywha(label_data):
    """
    将标注数据转换成 cx,cy,w,h,angle 格式

    [in] label_data, 输入的标注数据，经由 convert_raw_label_to_vertec_pts 得到
    [out] label_data, 输出的标注数据，多了cxywha字段的数据。 原来的坐标是从左上角->右上角->右下角->左下角定义的。
    """
    """
    w = math.sqrt((p0x-p1x) ** 2 + (p0y-p1y) ** 2)
    h = math.sqrt((p2x-p1x) ** 2 + (p2y-p1y) ** 2)
    vec_x=(p0x-cx)-(p3x-cx)
    vec_y=(cy-p0y)-(cy-p3y)
    # 坐标系问题，图像坐标系在左上角，所以要反过来减。求的夹角是与(0,1)的夹角，反过来减的话相当于求(0,-1)的夹角，也就是我们正常坐标系的(0,1)。
    angle=math.acos(
        (0*vec_x+1*vec_y)
        /
        ( ( math.sqrt(0*0+1*1) * math.sqrt(vec_x*vec_x+vec_y*vec_y)+0.000001 ) )
        )
    if (p0x+p1x)/2<cx:
        angle=2*math.pi-angle
    """
    if label_data["pts"].shape[0]==0:
        label_data["label_cxywha"]=label_data["pts"].copy()
        return label_data

    raw_pts = label_data["pts"].copy()
    w = np.sqrt(np.square(
        raw_pts[:, 1, 0]-raw_pts[:, 2, 0])+np.square(raw_pts[:, 1, 1]-raw_pts[:, 2, 1]))
    h = np.sqrt(np.square(
        raw_pts[:, 3, 0]-raw_pts[:, 2, 0])+np.square(raw_pts[:, 3, 1]-raw_pts[:, 2, 1]))
    vec_x = raw_pts[:, 1, 0]-raw_pts[:, 4, 0]
    vec_y = raw_pts[:, 4, 1]-raw_pts[:, 1, 1]
    angle = np.arccos(
        vec_y/(np.sqrt(np.square(vec_x)+np.square(vec_y))+0.000001))

    top_edge_center = raw_pts[:, 1, 0]+raw_pts[:, 2, 0]
    top_edge_center = top_edge_center/2

    angle_convert_idx = (raw_pts[:, 0, 0] > top_edge_center) & (label_data["cat"] >= 1.9) # 类0和类1 不转换,这里写的不优雅。
    angle[angle_convert_idx] = 2*np.pi-angle[angle_convert_idx] # 2pi 的角度定义方法。

    new_pts = np.concatenate(
        (raw_pts[:, 0, 0].reshape(-1, 1), raw_pts[:, 0, 1].reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1), angle.reshape(-1, 1)), axis=1)
    label_data["label_cxywha"] = new_pts
    return label_data

def coordinate_norm(img, label):
    # 坐标归一化，将坐标转换成YOLO的相对坐标。 pixel -> norm
    
    norm_label = label.copy()
    h,w,_ = img.shape
    # print(norm_label[:,:,0])
    norm_label[:,:,0] /= w
    norm_label[:,:,1] /= h
    
    return norm_label

if __name__ == '__main__':

    # xml_path = '/home/winner/chenwf/rotate-yolov5/data/pedestrian/sample/xml_labels'
    # img_path = '/home/winner/chenwf/rotate-yolov5/data/pedestrian/sample/images'
    # output_path = '/home/winner/chenwf/rotate-yolov5/data/pedestrian/sample/labels'

    xml_path = '/home/winner/chenwf/rotate-yolov5/data/pedestrian/transformed/xml_labels'
    img_path = '/home/winner/chenwf/rotate-yolov5/data/pedestrian/transformed/images'
    output_path = '/home/winner/chenwf/rotate-yolov5/data/pedestrian/transformed/labels'

    for xml in tqdm(os.listdir(xml_path)):
        output_name = os.path.join(output_path, xml.replace('xml','txt'))
        pdv_label = PDVLabel()
        pdv_label.load_xml(os.path.join(xml_path, xml))
        image = cv2.imread(os.path.join(img_path, xml.replace('xml','jpg')))

        raw_object_dict = pdv_label.get_objects()
        # print(raw_object_dict)
        label = convert_raw_label_to_vertex_pts(image, raw_object_dict)
        # convert_vertex_pts_to_cxywha(label)


        if len(label['pts']) > 0:
            norm_label = coordinate_norm(image, label['pts'][:,1:,:])
            # print(norm_label)
        # write
        output_annot = []
        for cls,each_norm in zip(label['cat'], norm_label):
            # tl,tr,br,bl (t->top,l->left,b->buttom,r->right)
            # print(each_norm)
            # print(each_norm.flatten())
            output_annot.append(str(int(cls)) + " " + " ".join(map(str,each_norm.ravel())))  # .ravel()一维化
        # print(output_annot)
        output = '\n'.join(output_annot)    # 转换结束

        with open(output_name,'w') as f:
            f.write(output)
        
