import cv2
import numpy as np
from glob import glob
import os 
from tqdm import tqdm

path1 = r'F:\data\1314yolov5x'
path2 = r'C:\Users\winner\Desktop\outputs'
if not os.path.exists("cat"):
    os.mkdir("cat")

for img in tqdm(glob(os.path.join(path1,'*.jpg'))):
    img1 = cv2.imread(img)
    img2 = cv2.imread(img.replace(path1,path2))
    # print(img1.shape,img2.shape)
    img1 = cv2.resize(img1, (960, 800))
    img2 = cv2.resize(img2, (960, 800))
 
    image = np.concatenate([img1, img2], axis=1)
    cv2.imwrite(os.path.join("cat",img.split(os.sep)[-1]),image)
    # image = np.vstack((img1, img2))
    # cv2.imshow("test", image)
    # cv2.waitKey(0)