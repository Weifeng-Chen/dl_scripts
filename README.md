<h2 align="center">
Some Scripts For DEEP LEARNING

# 1. detection 
## yolo2coco.py
将yolo格式数据集修改成coco格式。`$ROOT_PATH`是根目录，需要按下面的形式组织数据：

```bash
└── $ROOT_PATH

  ├── classes.txt

  ├── images

  └──labels
```

- `classes.txt` 是类的声明，一行一类。

- `images` 目录包含所有图片 (目前支持`png`和`jpg`格式数据)

- `labels` 目录包含所有标签(与图片**同名**的`txt`格式数据)

配置好文件夹后，执行：`python yolo2coco.py --root_dir $ROOT_PATH ` ，然后就能看见生成的 `annotations` 文件夹。

**参数说明**
- `--root_path` 输入根目录`$ROOT_PATH`的位置。
- `--save_path` 如果不进行数据集划分，可利用此参数指定输出文件的名字，默认保存为`train.json`
- `--random_split`  随机划分参数，若指定`--random_split`参数，则输出在`annotations`文件夹下包含 `train.json` `val.json` `test.json` （默认随机划分成8:1:1）
- `--split_by_file` 自定义数据集划分，若指定`--split_by_file`参数，则输出在`annotations`文件夹 `train.json` `val.json` `test.json`。需要在`$ROOT_PATH`文件下有 `./train.txt ./val.txt ./test.txt` ，可以这3个文件来定义训练集、验证集、测试集。**注意**， 这里里面填写的应是图片文件名字，而不是图片的绝对地址。（在line 43也自行可以修改一下读取方式，为了方便起见，不推荐把图片放在不同位置） 


## coco2yolo.py

读入coco数据集json格式的标注，输出可供yolo训练的标签。

**需要注意的是，COCO2017官方的数据集中categories id 是不连续的**，这在yolo读取的时候会出问题，所以需要重新映射一下，这个代码会按id从小到大映射到0~79之间。（如果是自己的数据集，也会重新映射）

执行：`python coco2yolo.py --json_path $JSON_FILE_PATH --save_path $LABEL_SAVE_PATH`

- `$JSON_FILE_PATH`是json文件的地址。
- `$JSON_FILE_PATH`是输出目录（默认为工作目录下的`./labels`目录。


## zeroshot_retrieval_evaluation.ipynb
- 检索topN的计算，支持一对多检索。（一张图对应有多个captions）

## vis_yolo_gt_dt.py
同时把GT和预测结果可视化在同一张图中。`$DT_DIR`是预测结果标签地址，必须是和GT同名的标签。`$ROOT_PATH`文件目录：

```bash
└── $ROOT_PATH

  ├── classes.txt

  ├── images

  └── labels
```

执行：`python vis_yolo_gt_dt.py --root $ROOT_PATH --dt $DT_DIR`后生成在`outputs`文件夹中。

- `classes.txt`和`images`必须有。
- `labels`可以没有，那样就只展示`$DT_DIR`预测结果。
- `$DT_DIR` 若没有输入，则只展示标签结果。

## coco_eval.py

评估生成的结果，针对**yolov5**生成的检测结果（test中的`--save-json`参数，会生成`best_predictions.json`)，但是这个不适应cocoapi，需要用脚本来修改适应。执行：

`python coco_eval.py --gt $GT_PATH --dt $DT_PATH --yolov5`

- `--gt` json格式，用于指定测试集的结果，如果没有，可以利用前面的`yolo2coco.py`进行转换。
- `--dt` 同样检测网络生成的预测，使用cocoapi中`loadRes`来加载，所以需要有相应格式的检测结果。
- `--yolov5` 将官方代码中生成的结果转换成适配cocoapi的结果。

# 2. text-image
## zeroshot_retrieval_evalution.ipynb
检索模型的评估指标。（topK召回率），支持多对多的情况。（比如一个文本匹配多张图片）
## fid_clip_score
用于画text2image的 FID-CLIP Score曲线图。