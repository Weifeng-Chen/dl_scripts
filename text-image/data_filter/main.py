from model import FilterSystem
from dataset import TxtDataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torch


def sub_process(filter_model, each_folder_path):
    each_dataset = TxtDataset(each_folder_path)
    each_dataloader = DataLoader(each_dataset, batch_size=8, shuffle=False, num_workers=8)
    image_paths = []
    aes_scores = []
    clip_scores = []
    watermark_scores = []
    for iii, (batch_image_paths, texts,) in enumerate(tqdm(each_dataloader)):
        images =  [Image.open(each_image_path).convert("RGB") for each_image_path in batch_image_paths]
        image_paths.extend(batch_image_paths)

        image_features = filter_model.get_image_feature(images,)  # 计算图片特征，传入图片列表，一般而言，可以在数据库保存这个东西，用于响应文本query
        aes_score = filter_model.get_aesthetics_score(image_features)  # 计算美学分数，传入图片特征，一般而言，可以在数据库保存这个东西，用于响应文本query
        aes_scores.extend(aes_score)

        text_features = filter_model.get_text_feature(list(texts)) # 计算文本特征，传入文本列表
        clip_score = filter_model.calculate_clip_score(image_features, text_features)  # 计算相似度
        clip_scores.extend(torch.diagonal(clip_score).detach().cpu().numpy())  # 需要取对角线，只需要自己和对应文本的相似度

        watermark_score = filter_model.get_watermark_score(images)  # 计算水印分数，传入图片路径列表
        watermark_scores.extend(watermark_score)
        
    score_pd = pd.DataFrame({'image_path': image_paths, 'aes_score': aes_scores, 'clip_score': clip_scores, 'watermark_score': watermark_scores})
    score_pd.to_csv(os.path.join(each_folder_path, 'score.csv'), index=False)
    print('save score.csv in {}'.format(each_folder_path), '\n', '-'*20)

if __name__ == '__main__':
    # data setting
    root_path = "/cognitive_comp/chenweifeng/project/dataset/laion_chinese_cwf/image_part01"
    all_folders = sorted(os.listdir(root_path))

    # model setting
    filter_model = FilterSystem()
    filter_model.init_clip_model()
    filter_model.init_aesthetics_model()
    filter_model.init_watermark_model()

    for each_folder in all_folders:
        each_folder_path = os.path.join(root_path, each_folder)
        sub_process(filter_model, each_folder_path)