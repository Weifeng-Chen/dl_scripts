# Tools

Some useful tools for computer vision/deep learning.

# Usage

### yolo2coco.py

将数据集修改成YOLO格式。`$ROOT_PATH`是根目录（默认目录为当前目录下面的`yolo_data`文件夹，也可以直接创建一个），文件结构如下：

```bash
└── $ROOT_PATH

  ├── classes.txt

  ├── images

  └──labels
```

- `classes.txt` 是类的声明，一行一类。

-  `images` 目录包含所有图片 (format:`jpg` )

- `labels` 目录包含所有标签(与图片同名，format: `txt`)

配置好后，执行：`python yolo2coco.py --root_path $ROOT_PATH --random_split` ，然后你就能看见生成的 `annotations`, 包括 ``train.json` `val.json` `test.json` （默认随机划分成8:1:1）

- --root_path 输入根目录$ROOT_PATH的位置。
- --random_split 为划分参数，如果没有这个参数则只保存`train.json`文件

### dataset_mean_var.py

执行：`python yolo2coco.py --file_path $IMAGE_PATH --step $INTERVAL`

- --file_path 输入图片地址。

- --step 可选，默认为1，选择图片的间隔，如间隔为10，则只计算1/10。

### split_yolo_dataset.py

随机划分数据集（yolo数据格式的划分），按train:val:test = 8:1:1保存。
执行：`python split_dataset_yolo.py --root_path $ROOT_PATH`

