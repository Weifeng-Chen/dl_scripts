import cv2
import numpy as np
from glob import glob
import os 
from tqdm import tqdm

def cv_imread(file_path):
    # 中文路径读取
    root_dir, file_name = os.path.split(file_path)
    pwd = os.getcwd()
    if root_dir:
        os.chdir(root_dir)
    cv_img = cv2.imread(file_name)
    os.chdir(pwd)
    return cv_img

def cv_write(file_path, image):
    # 中文路径读取
    root_dir, file_name = os.path.split(file_path)
    pwd = os.getcwd()
    if root_dir:
        os.chdir(root_dir)
    cv_img = cv2.imwrite(file_name,image)
    os.chdir(pwd)

### 单个文件夹下多个目录
root1 = r'C:\Users\winner\Desktop\pdv_scratchv2_70w'
root2 = r'C:\Users\winner\Desktop\pvd_s_0.5_v2'
path1s = os.listdir(root1)
path2s =  os.listdir(root2)

for (p1,p2) in zip(path1s,path2s):
    path1 = os.path.join(root1,p1)
    path2 = os.path.join(root2,p2)
    save_dir = os.path.join('75w_result',p1)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        continue
    for img in tqdm(glob(os.path.join(path1,'*.jpg'))):
        img1 = cv_imread(img)
        img2 = cv_imread(img.replace(path1,path2))
        # print(img1.shape,img2.shape)
        img1 = cv2.resize(img1, (960, 600))
        img2 = cv2.resize(img2, (960, 600))
            
        image = np.concatenate([img1, img2], axis=1)
        cv_write(os.path.join(save_dir,img.split(os.sep)[-1]),image)


### 单个文件夹

# path1 = r'C:\Users\winner\Desktop\pdv_scratchv2_70w'
# path2 = r'C:\Users\winner\Desktop\pvd_s_0.5_v2'
# save_dir = "75w_result"
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
    
# for img in tqdm(glob(os.path.join(path1,'*.jpg'))):
#     img1 = cv2.imread(img)
#     img2 = cv2.imread(img.replace(path1,path2))
#     # print(img1.shape,img2.shape)
#     img1 = cv2.resize(img1, (960, 600))
#     img2 = cv2.resize(img2, (960, 600))
 
#     image = np.concatenate([img1, img2], axis=1)
#     cv2.imwrite(os.path.join(save_dir,img.split(os.sep)[-1]),image)
