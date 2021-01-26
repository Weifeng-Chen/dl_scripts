# Tools

Some useful tools for computer vision/deep learning.
（still updating......）

# Usage

### yolo2coco.py

将yolo格式数据集修改成coco格式。`$ROOT_PATH`是根目录，需要按下面的形式组织数据：

```bash
└── $ROOT_PATH

  ├── classes.txt

  ├── images

  └──labels
```

- `classes.txt` 是类的声明，一行一类。

-  `images` 目录包含所有图片 (目前支持`png`和`jpg`格式数据)

- `labels` 目录包含所有标签(与图片**同名**的`txt`格式数据)

配置好后，执行：`python yolo2coco.py --root_path $ROOT_PATH --random_split` ，然后你就能看见生成的 `annotations`,文件夹，包含 ``train.json` `val.json` `test.json` （默认随机划分成8:1:1），如果不想划分数据集，则不要输入`random_split`这个参数。

- `--root_path` 输入根目录$ROOT_PATH的位置。
- `--random_split`  为划分参数，如果没有这个参数则只保存`train.json`文件
- -`-save_name` 如果不进行随机划分，可利用此参数指定输出文件的名字，默认保存为`train.json`

### dataset_mean_var.py

执行：`python yolo2coco.py --file_path $IMAGE_PATH --step $INTERVAL`

- --file_path 输入图片地址。

- --step 可选，默认为1，选择图片的间隔，如间隔为10，则只计算1/10。

### split_yolo_dataset.py

随机划分数据集（yolo数据格式的划分），按`train:val:test = 8:1:1`保存。

执行：`python split_dataset_yolo.py --root_path $ROOT_PATH`

### vis_yolo_gt_pred.py

同时把GT和预测结果可视化在同一张图中。文件目录：

```bash
└── $ROOT_PATH

  ├── preds

  ├── images

  └── labels
```

执行：`python vis_yolo_gt_pred.py --root_path $ROOT_PATH`后生成在outputs中。（`preds`和`labels`文件夹都是可选的，没有的话就不画。）

### coco_eval.py

评估生成的结果，针对**yolov5**生成的检测结果（test中的`--save-json`参数，会生成`best_predictions.json`)，但是这个不适应cocoapi，需要用脚本来修改适应。执行：

`python coco_eval.py --gt $GT_PATH --dt $DT_PATH --yolov5`

- `--gt` json格式，用于指定测试集的结果，如果没有，可以利用前面的`yolo2coco.py`进行转换。
- `--dt` 同样检测网络生成的预测，使用cocoapi中`loadRes`来加载，所以需要有相应格式的检测结果。
- `--yolov5` 将官方代码中生成的结果转换成适配cocoapi的结果。

### modify_yolo_cls.py

用于修改yolo数据集中的类别。自用。

