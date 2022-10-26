from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import json
import torch
from transformers import BertModel
import open_clip
import numpy as np
from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm
import argparse
import torch
# NOTE 加速读取数据，直接用原版的，在外部使用并行读取策略。30min->3min
class CsvDataset(Dataset):
    def __init__(self, input_filename, input_root, img_key, caption_key, transforms=None, thres=0.2, sep="\t"):
        # logging.debug(f'Loading csv data from {input_filename}.')
        print(f'Loading csv data from {input_filename}.')
        self.images = []
        self.captions = []

        if input_filename.endswith('.csv'):
            # print(f"Load Data from{input_filename}")
            df = pd.read_csv(input_filename, index_col=0)
            df = df[df['used'] == 1]
            df = df[df['score']>thres]
            self.images.extend(df[img_key].tolist())
            self.captions.extend(df[caption_key].tolist())
        
        # NOTE 中文的tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        self.context_length = 77
        self.root = input_root
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        image = self.transforms(Image.open( os.path.join(self.root, img_path ))) 
        text = self.tokenizer(str(self.captions[idx]), max_length=self.context_length, padding='max_length', truncation=True, return_tensors='pt')['input_ids'][0]
        return image, text


def process_pool_read_csv_dataset(input_root, input_filename, thres=0.20):
    # here input_filename is a directory containing a CSV file
    all_csvs = os.listdir(input_filename)

    csv_with_score = [each for each in all_csvs if 'score' in each ]
    all_datasets = []
    res = []        
    p = ProcessPoolExecutor(max_workers=24)
    for i in range(len(csv_with_score)):
        each_csv_path = os.path.join(input_filename, csv_with_score[i])
        print(i, each_csv_path)
        res.append(p.submit(CsvDataset, each_csv_path, input_root, img_key="name", caption_key="caption", thres=thres))
    p.shutdown()
    for future in res:
        all_datasets.append(future.result())
    dataset = ConcatDataset(all_datasets)
    return dataset


tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese", model_max_length=512)
input_filename = '/cognitive_comp/chenweifeng/project/dataset/wukong/release'   # 这里存的是csv标注地址
input_root = '/cognitive_comp/chenweifeng/project/dataset/wukong/images'
dataset = process_pool_read_csv_dataset(input_root, input_filename, thres=0.20)

print(len(dataset))