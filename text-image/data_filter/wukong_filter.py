# %%
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


parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--part",
    type=int,
    default=0,
    required=True,
)
args = parser.parse_args()


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, input_root, tokenizer, img_key, caption_key, sep="\t"):
        # logging.debug(f'Loading csv data from {input_filename}.')
        print(f'Loading csv data from {input_filename}.')
        self.images = []
        self.captions = []
        if input_filename.endswith('.csv'):
            df = pd.read_csv(input_filename, index_col=0)
            df = df[df['used'] == 1]
            self.images.extend(df[img_key].tolist())
            self.captions.extend(df[caption_key].tolist())
        # NOTE 中文的tokenizer
        self.tokenizer = tokenizer
        self.context_length = 77
        self.root = input_root
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])
        image = self.transforms(Image.open( os.path.join(self.root, img_path ))) 
        text = self.tokenizer(str(self.captions[idx]), max_length=self.context_length, padding='max_length', truncation=True, return_tensors='pt')['input_ids'][0]
        return image, text, img_path


text_encoder = BertModel.from_pretrained("IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese").eval().cuda()
clip_model, _, processor = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
clip_model = clip_model.eval().cuda()
text_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese")


input_filename = './project/dataset/wukong/release'
preprocess_fn = processor
input_root = './project/dataset/wukong/images'
tokenizer = text_tokenizer
all_csvs = sorted(os.listdir(input_filename))

for i in range(len(all_csvs)*args.part//5, len(all_csvs)*(args.part+1)//5):
    # 分成5part
    each_csv_path = os.path.join(input_filename, all_csvs[i])
    dataset = CsvDataset(each_csv_path, preprocess_fn, input_root, tokenizer, img_key="name", caption_key="caption")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    
    df = pd.read_csv(each_csv_path, index_col=0)
    df = df[df['used'] == 1]
    scores = []
    for iii, (image, text, image_path) in enumerate(tqdm(dataloader)):
        # print(image.shape, text.shape)
        with torch.no_grad():
            image = image.cuda()
            text = text.cuda()
            # print(image.shape, text.shape)
            image_features = clip_model.encode_image(image)
            text_features = text_encoder(text)[1]

            # print(image_features.shape, text_features.shape)
            # 归一化
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            score_each_pair =  image_features @ text_features.t()

            scores.extend(torch.diagonal(score_each_pair).detach().cpu().numpy())
            # break
    df['score'] = scores
    df.to_csv( each_csv_path.replace(all_csvs[i], 'score'+all_csvs[i]) , index=False)
    print('saving score to', each_csv_path.replace(all_csvs[i], 'score'+all_csvs[i]) )
   