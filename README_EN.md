
## Useful tools for computer vision/deep learning.

<h4 align="center">
    <p><a href="https://github.com/Weifeng-Chen/DL_tools">简体中文</a> | <b>English</b><p>
</h4>

# Usage
## yolo2coco.py

Transform YOLO format dataset to COCO format.`$ROOT_PATH` is the root directory.
**Please organize your files according to the following**:

```bash
└── $ROOT_PATH

  ├── classes.txt

  ├── images

  └──labels
```

- `classes.txt` , the statement of class. one class per line.

-  `images` , the directory should contain all the images you want to train(support `png` and `jpg`)

- `labels` , the directory should contain all the lables(**Same name** as the images, txt format)


Run `python yolo2coco.py --root_dir $ROOT_PATH ` ，and you will see a dir: `annotations`.

**About the argument**
- `--root_path` path of `$ROOT_PATH`
- `--random_split` whether to randomly split the datasete. If store ture, dir `annotations` will include `train.json` `val.json` `test.json` （split to 8:1:1）
- `--save_path` save name of output,default is `train.json`


## coco2yolo.py
Read the label in JSON format of coco dataset and output the label for Yolo training.
**It should be noted that the categories ID in the official dataset of coco2017 is not continuous**, which will cause problems when Yolo reads it, so it needs to be remapped. This script will map the class id from 0 to 79(If it is your own dataset, it will be remapped.)

Run: `python coco2yolo.py --json_path $JSON_FILE_PATH --save_path $LABEL_SAVE_PATH`

- `$JSON_FILE_PATH`是json文件的地址。
- `$JSON_FILE_PATH`是输出目录（默认为工作目录下的`./labels`目录。

