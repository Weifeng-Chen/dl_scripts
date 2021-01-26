import json
import argparse
from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval 
import os

parser = argparse.ArgumentParser()
parser.add_argument("--gt", type=str, help="Assign the groud true path.", default=None)
parser.add_argument("--dt", type=str, help="Assign the detection result path.", default=None)
parser.add_argument("--yolov5",action='store_true',help="for yolov5", default=None)

args = parser.parse_args()

def transform_yolov5_result(result):
    f = open(result ,'r',encoding='utf-8')
    dts = json.load(f)
    output_dts = []
    for dt in dts:
        dt['image_id'] = filename2id[dt['image_id']+'.jpg']
        dt['category_id'] += 1 # 标号对应好
        output_dts.append(dt)
    with open('temp.json', 'w') as f:
        json.dump(output_dts, f)

if __name__ == '__main__':
    cocoGt = COCO(args.gt)
    imgIds = cocoGt.getImgIds()
    gts = cocoGt.loadImgs(imgIds)
    filename2id = {}
    for gt in gts:
        filename2id[gt['file_name']] = gt['id']
    print("NUM OF TEST IMAGES: ",len(filename2id))
    if args.yolov5:
        transform_yolov5_result(args.dt)
        cocoDt = cocoGt.loadRes('temp.json')
    else:
        cocoDt = cocoGt.loadRes(args.dt)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    if args.yolov5:
        os.remove('temp.json')