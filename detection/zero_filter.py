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

# 添加Json数据集读取，主要是针对Zero23m数据集。
class JsonDataset(Dataset):
    def __init__(self, foloder_name, tokenizer, image_transform = None, size=512, center_crop=False):
        print(f'Loading folder data from {foloder_name}.')
        self.image_paths = []
        self.caption_paths = []
        self.tokenizer = tokenizer
        
        # 这里都存的是地址，避免初始化时间过多。
        for each_file in os.listdir(foloder_name):
            if each_file.endswith('.jpg'):
                self.image_paths.append(os.path.join(foloder_name, each_file))
                self.caption_paths.append(os.path.join(foloder_name, each_file.replace('.jpg', '.json')))
        self.image_transforms = image_transform
        print('Done loading data. Len of images:', len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        caption_path = str(self.caption_paths[idx])
        example = {}

        instance_image = Image.open(img_path)
        instance_image = self.image_transforms(instance_image)
        with open(caption_path, 'r') as f:
            caption = json.load(f)['caption']
            if not caption:
                caption = '无'
            input_id = self.tokenizer(
                caption, return_tensors='pt', padding='max_length',truncation=True, max_length=77).input_ids[0]
        return instance_image, input_id, img_path


# %%
text_encoder = BertModel.from_pretrained("IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese").eval().cuda()
clip_model, _, processor = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
clip_model = clip_model.eval().cuda()
text_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese")

# %%
import pandas as pd

# %%
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese", model_max_length=512)
root_path = "/cognitive_comp/chenweifeng/zero23m_cwf"

all_folders = sorted(os.listdir(root_path))

for i in range(len(tqdm(all_folders))):    
    each_folder_path = os.path.join(root_path, all_folders[i])
    dataset = JsonDataset(each_folder_path, tokenizer, image_transform=processor)
    # print(dataset[0])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
    # score_pd = pd.DataFrame(columns=['image_path', 'score'])

    image_paths = []
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

            image_paths.extend(image_path)
            scores.extend(torch.diagonal(score_each_pair).detach().cpu().numpy())
            # break
    score_pd = pd.DataFrame({'image_path': image_paths, 'score': scores})
    score_pd.to_csv( os.path.join(root_path, all_folders[i], 'score.csv') , index=False)
    print('saving score to', os.path.join(root_path, all_folders[i], 'score.csv'))
   