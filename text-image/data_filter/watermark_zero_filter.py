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
import json

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

        instance_image = Image.open(img_path).convert('RGB')
        instance_image = self.image_transforms(instance_image)
        with open(caption_path, 'r') as f:
            caption = json.load(f)['caption']
            if not caption:
                caption = '无'
            input_id = self.tokenizer(
                caption, return_tensors='pt', padding='max_length',truncation=True, max_length=77).input_ids[0]
        return instance_image, input_id, img_path


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
preprocess_fn = processor
root_path = "/cognitive_comp/chenweifeng/zero23m_cwf"
all_folders = sorted(os.listdir(root_path))


for i in range(len(all_folders)*args.part//4, len(all_folders)*(args.part+1)//4):
    each_folder_path = os.path.join(root_path, all_folders[i])
    dataset = JsonDataset(each_folder_path, text_tokenizer, image_transform=processor)
    # print(dataset[0])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=False)
    # score_pd = pd.DataFrame(columns=['image_path', 'score'])

    image_paths = []
    scores = []
    for iii, (image, text, image_path) in enumerate(tqdm(dataloader)):
        # print(image.shape, text.shape)
        with torch.no_grad():
            image = image.cuda()
            pred = model(image)
            is_watermark_outputs = F.softmax(pred, dim=1)[:,0].detach().cpu().numpy().tolist()
            image_paths.extend(image_path)
            scores.extend(is_watermark_outputs)

    score_pd = pd.DataFrame({'image_path': image_paths, 'watermark_score': scores})
    score_pd.to_csv( os.path.join(root_path, all_folders[i], 'watermark_score.csv') , index=False)
    print('saving score to', os.path.join(root_path, all_folders[i], 'watermark_score.csv'))
