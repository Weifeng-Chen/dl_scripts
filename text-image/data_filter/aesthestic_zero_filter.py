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
import pytorch_lightning as pl
import torch.nn as nn
import argparse

# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


amodel = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
s = torch.load("/cognitive_comp/chenweifeng/project/dl_scripts/text-image/data_filter/improved-aesthetic-predictor/ava+logos-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
amodel.load_state_dict(s)
amodel.to("cuda")
amodel.eval()

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--part",
    type=int,
    default=0,
    required=True,
)
args = parser.parse_args()

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

for i in range(len(all_folders)*args.part//4, len(all_folders)*(args.part+1)//4):
    each_folder_path = os.path.join(root_path, all_folders[i])
    dataset = JsonDataset(each_folder_path, tokenizer, image_transform=processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=False)
    # score_pd = pd.DataFrame(columns=['image_path', 'score'])

    image_paths = []
    scores = []
    for iii, (image, text, image_path) in enumerate(tqdm(dataloader)):
        # print(image.shape, text.shape)
        with torch.no_grad():
            image = image.cuda()
            # print(image.shape, text.shape)
            image_features = clip_model.encode_image(image)
            # 归一化
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            score = amodel(image_features,)
            image_paths.extend(image_path)

            # print(score.shape)
            scores.extend(score[:, 0].detach().cpu().numpy())
            # break
    score_pd = pd.DataFrame({'image_path': image_paths, 'aesthestic_score': scores})
    score_pd.to_csv( os.path.join(root_path, all_folders[i], 'aesthestic_score.csv') , index=False)
    print('saving score to', os.path.join(root_path, all_folders[i], 'aesthestic_score.csv'))
   