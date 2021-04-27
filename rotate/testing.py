import os 
import cv2
import matplotlib.pyplot as plt
from image_affine_augment import convert_raw_label_to_vertex_pts, convert_vertex_pts_to_cxywha
from pdv_labels_xml import PDVLabel, PDVObjHead, PDVObjHbody, PDVObjFbody
import numpy as np

image = cv2.imread("./Round_0-part2-review_192.168.16.27_image1_a00.jpg")
pdv_label = PDVLabel()
pdv_label.load_xml("./Round_0-part2-review_192.168.16.27_image1_a00.xml")
raw_object_dict = pdv_label.get_objects()


label = convert_raw_label_to_vertex_pts(image, raw_object_dict)
print(label)
convert_vertex_pts_to_cxywha(label)
print(label)

# # 结果绘图
pts_int32 = label["pts"].astype(np.int32)
draw_obj_num = pts_int32.shape[0]
LINE_WIDTH = 2

for i in range(draw_obj_num):
    draw_color = (0, 0, 0)
    if label["cat"][i] == 0:
        draw_color = (255, 0, 0)
    elif label["cat"][i] == 1:
        draw_color = (128, 0, 128)
    elif label["cat"][i] == 2:
        draw_color = (0, 255, 0)

    cv2.line(image, tuple(pts_int32[i, 1]),
                tuple(pts_int32[i, 2]), draw_color, LINE_WIDTH)
    cv2.line(image, tuple(pts_int32[i, 1]),
                tuple(pts_int32[i, 4]), draw_color, LINE_WIDTH)
    cv2.line(image, tuple(pts_int32[i, 3]),
                tuple(pts_int32[i, 2]), draw_color, LINE_WIDTH)
    cv2.line(image, tuple(pts_int32[i, 3]),
                tuple(pts_int32[i, 4]), draw_color, LINE_WIDTH)

cv2.imshow("vis", image)
cv2.waitKey()