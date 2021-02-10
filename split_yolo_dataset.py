from sklearn.model_selection import train_test_split
import os
import shutil
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path',type=str,default='yolo_data', help="root path of images and labels")
arg = parser.parse_args()


def train_test_val_split(img_paths,ratio_train,ratio_test,ratio_val):
    assert int(ratio_train+ratio_test+ratio_val) == 1

    phase = ['train', 'val', 'test']
    # 不同环境下面，listdir读取的顺序不一定相同的！
    train_img, middle_img = train_test_split(img_paths,test_size=1-ratio_train, random_state=233)
    ratio=ratio_val/(1-ratio_train)
    val_img, test_img  =train_test_split(middle_img,test_size=ratio, random_state=233)

    print("nums of train:val:test = {}:{}:{}".format(len(train_img), len(val_img), len(test_img)))
    p2path = {'train':train_img,'val':val_img,'test':test_img}
    for p in phase:
        dst_path = os.path.join(root_path,p)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
            os.mkdir(os.path.join(dst_path,'images'))
            os.mkdir(os.path.join(dst_path,'labels'))
        for img_name in tqdm(p2path[p]):
            shutil.copy(os.path.join(root_path, 'images', img_name), os.path.join(dst_path,'images'))
            if os.path.exists(os.path.join(root_path, 'labels', img_name.replace('jpg','txt'))):
                shutil.copy(os.path.join(root_path, 'labels', img_name.replace('jpg','txt')), os.path.join(dst_path,'labels'))

    return train_img, val_img, test_img

if __name__ == '__main__':
    root_path = arg.root_path
    img_paths = os.listdir(os.path.join(root_path, 'images'))
    label_paths = os.listdir(os.path.join(root_path, 'labels'))
    train_test_val_split(img_paths,0.8,0.1,0.1)
