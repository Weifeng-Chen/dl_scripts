# tools

Some useful tools for deep learning implement.

# Usage

### yolo2coco.py

将数据集修改成YOLO格式。$ROOT_PATH是根目录，文件结构如下：

```bash
└── $ROOT_PATH

  ├── classes.txt

  ├── images

  └──labels
```

- `classes.txt` 是类的声明，一行一类。

-  `images` 目录包含所有图片 (format:`jpg` )

- `labels` 目录包含所有标签(与图片同名，format: `txt`)

配置好后，执行：`python yolo2coco.py --root_path $ROOT_PATH`

，然后你就能看见生成的 `annotations`, 包括 ``train.json` `val.json` `test.json` （默认被分为 **8: 1: 1**）

- --root_path 输入根目录$ROOT_PATH的位置。

### dataset_mean_var.py

执行：`python yolo2coco.py --file_path $IMAGE_PATH --step $PICK_INTERVAL`

- --file_path是图片地址。

- --step是选择图片的间隔，如间隔为10，则只计算1/10。

