import json
import argparse
from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval 
import os
import time

def transform_yolov5_result(result, filename2id):
    f = open(result ,'r',encoding='utf-8')
    dts = json.load(f)
    output_dts = []
    for dt in dts:
        dt['image_id'] = filename2id[dt['image_id']+'.jpg']
        dt['category_id'] # id对应好，coco格式和yolo格式的category_id可能不同。
        output_dts.append(dt)
    with open('temp.json', 'w') as f:
        json.dump(output_dts, f)

def coco_evaluate(gt_path, dt_path, yolov5_flag):
    cocoGt = COCO(gt_path)
    imgIds = cocoGt.getImgIds()
    gts = cocoGt.loadImgs(imgIds)
    filename2id = {}

    for gt in gts:
        filename2id[gt['file_name']] = gt['id']
    print("NUM OF TEST IMAGES: ",len(filename2id))

    if yolov5_flag:
        transform_yolov5_result(dt_path, filename2id)
        cocoDt = cocoGt.loadRes('temp.json')
    else:
        cocoDt = cocoGt.loadRes(dt_path)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    if yolov5_flag:
        os.remove('temp.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, help="Assign the groud true path.", default=None)
    parser.add_argument("--dt", type=str, help="Assign the detection result path.", default=None)
    parser.add_argument("--yolov5",action='store_true',help="fix yolov5 output bug", default=None)

    args = parser.parse_args()
    gt_path = args.gt
    dt_path = args.dt
    if args.yolov5:
        coco_evaluate(gt_path, dt_path, True)
    else:
        coco_evaluate(gt_path, dt_path, False)
    