# tools

Some useful tools for deep learning implement.



# Usage

#### yolo2coco.py

Transform `yolo` dataset format into `coco` dataset format. **you need to modify your dataset to suit this script. **

`$ROOT_PATH` is the path to put your data. It should be like the following tree:

└── $ROOT_PATH
    ├── classes.txt
    ├── images
    └── labels

- `classes.txt` contains all classes. One class per line.

- Directory `images` contains all images for train, valid, test. (format:`jpg` )

- Directory `labels` contains all labels. Each label has the same name as the image.(format: `txt`)

**RUN**：

`python yolo2coco.py --root_path $ROOT_PATH`

Then, you will get a new directory `annotations`, which include ``train.json` `val.json` `test.json` （the dataset was split into **8: 1: 1**）

#### dataset_mean_var.py

set the `filepath` and run directly.

