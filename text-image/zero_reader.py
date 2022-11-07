from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from transformers import  BertTokenizer

# 添加Txt数据集读取，主要是针对Zero23m数据集。
class TxtDataset(Dataset):
    def __init__(self, foloder_name, tokenizer, thres=0.2, size=512, center_crop=False):
        print(f'Loading folder data from {foloder_name}.')
        self.image_paths = []
        self.tokenizer = tokenizer
        score_data = pd.read_csv(os.path.join(foloder_name, 'score.csv'))
        img_path2score = {score_data['image_path'][i]: score_data['score'][i] for i in range(len(score_data))}
        # print(img_path2score)
        # 这里都存的是地址，避免初始化时间过多。
        for each_file in os.listdir(foloder_name):
            if each_file.endswith('.jpg'):
                if img_path2score[os.path.join(foloder_name, each_file)] > thres:
                    self.image_paths.append(os.path.join(foloder_name, each_file))
                # self.image_paths.append(os.path.join(foloder_name, each_file))

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        print('Done loading data. Len of images:', len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        example = {}
        instance_image = Image.open(img_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        caption_path = img_path.replace('.jpg', '.txt') # 图片名称和文本名称一致。
        with open(caption_path, 'r') as f:
            caption = f.read()
            example["instance_prompt_ids"] = self.tokenizer(
                caption,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
        return example


def process_pool_read_txt_dataset(input_root=None, tokenizer=None, thres=0.2):
    root_path = input_root
    p = ProcessPoolExecutor(max_workers=24) 
    # 此处输入为文件夹。
    all_folders = os.listdir(root_path)
    all_datasets = []
    res = []
    
    for i in range(len(all_folders)):
        each_folder_path = os.path.join(root_path, all_folders[i])
        # print(i, each_folder_path)
        res.append(p.submit(TxtDataset, each_folder_path, tokenizer, thres))    # thres是CLIP分数的阈值
    p.shutdown()
    for future in res:
        all_datasets.append(future.result())
    dataset = ConcatDataset(all_datasets)
    return dataset


root_path = "/cognitive_comp/chenweifeng/zero23m_cwf"
tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese", model_max_length=512)
dataset = process_pool_read_txt_dataset(root_path, tokenizer, thres=0.22)
print(len(dataset))