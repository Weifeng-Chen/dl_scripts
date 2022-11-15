# %%
from torch.utils.data import Dataset, ConcatDataset
import pandas as pd
from tqdm import tqdm
import argparse
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms as T
from transformers import BertTokenizer
import os
from PIL import Image


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
        image = self.transforms(Image.open( os.path.join(self.root, img_path )).convert('RGB')) 
        text = self.tokenizer(str(self.captions[idx]), max_length=self.context_length, padding='max_length', truncation=True, return_tensors='pt')['input_ids'][0]
        return image, text, img_path

# model definition
model = timm.create_model(
        'efficientnet_b3a', pretrained=True, num_classes=2)

model.classifier = nn.Sequential(
    # 1536 is the orginal in_features
    nn.Linear(in_features=1536, out_features=625),
    nn.ReLU(),  # ReLu to be the activation function
    nn.Dropout(p=0.3),
    nn.Linear(in_features=625, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=2),
)
state_dict = torch.load('/cognitive_comp/chenweifeng/project/dl_scripts/text-image/data_filter/LAION-5B-WatermarkDetection/models/watermark_model_v1.pt')
model.load_state_dict(state_dict)
model.eval()
model.cuda()

processor = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

text_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese")
input_filename = '/cognitive_comp/chenweifeng/project/dataset/wukong/release'
preprocess_fn = processor
input_root = '/cognitive_comp/chenweifeng/project/dataset/wukong/images'
tokenizer = text_tokenizer

all_releases = os.listdir(input_filename)
all_csvs = []
for each in all_releases:
    if 'score' not in each:
        all_csvs.append(each)
all_csvs = sorted(all_csvs)

for i in range(len(all_csvs)*args.part//4, len(all_csvs)*(args.part+1)//4):
    
    # 分成4part
    each_csv_path = os.path.join(input_filename, all_csvs[i])
    if os.path.exists(each_csv_path.replace(all_csvs[i], 'watermark_score' + all_csvs[i])):
        continue
    dataset = CsvDataset(each_csv_path, preprocess_fn, input_root, tokenizer, img_key="name", caption_key="caption")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=False)
    
    df = pd.read_csv(each_csv_path, index_col=0)
    df = df[df['used'] == 1]
    scores = []
    for iii, (image, text, image_path) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            image = image.cuda()
            pred = model(image)
            is_watermark_outputs = F.softmax(pred, dim=1)[:,0].detach().cpu().numpy().tolist()
            scores.extend(is_watermark_outputs)

    df['watermark_score'] = scores
    df.to_csv( each_csv_path.replace(all_csvs[i], 'watermark_score' + all_csvs[i]) , index=False)
    print('saving score to', each_csv_path.replace(all_csvs[i], 'watermark_score' + all_csvs[i]))
