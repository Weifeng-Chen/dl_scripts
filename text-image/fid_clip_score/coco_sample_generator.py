from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import os
from diffusers import StableDiffusionPipeline
from argparse import ArgumentParser
from tqdm import tqdm
from multiprocessing import Process

parser = ArgumentParser()
parser.add_argument('--coco_path', type=str, default='../dataset/coco')
parser.add_argument('--coco_cache_file', type=str, default='../dataset/coco/subset.parquet')
parser.add_argument('--output_path', type=str, default='./output')
parser.add_argument('--model_path', type=str, default='../pretrained_models/stable-diffusion-v1-4')
parser.add_argument('--sample_step', type=int, default=20)
parser.add_argument('--guidance_scale', type=float, default=1.5)
parser.add_argument('--batch_size', type=int, default=2)
args = parser.parse_args()


class COCOCaptionSubset(Dataset):
    def __init__(self, path, transform=None):
        self.df = pd.read_parquet(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row['file_name'], row['caption']

def save_images(images, image_paths, output_path):
    for i, image_path in enumerate(image_paths):
        image_path = image_path.replace('/', '_')
        image_path = os.path.join(output_path, image_path)
        images[i].save(image_path)

if __name__ == '__main__':
    # testing 
    coco_path = args.coco_path
    # coco_cache_file = f'{coco_path}/subset.parquet'     # sampled subsets
    cocosubset = COCOCaptionSubset(args.coco_cache_file)
    cocosubsetloader = DataLoader(cocosubset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # load the t2i model
    stable_diffusion = StableDiffusionPipeline.from_pretrained(args.model_path, requires_safety_checker=False).to('cuda')   

    sample_step = args.sample_step
    guidance_scale = args.guidance_scale


    output_path = os.path.join(
        args.output_path,
        f'./gs{guidance_scale}_ss{sample_step}'
    ) 
    os.makedirs(output_path, exist_ok=True)

    for i, (image_paths, captions) in enumerate(tqdm(cocosubsetloader)):
        outputs = stable_diffusion(list(captions), num_inference_steps=sample_step, guidance_scale=guidance_scale).images
        p = Process(target=save_images, args=(outputs, image_paths, output_path))
        p.start()