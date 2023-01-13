from torch.utils.data import Dataset, DataLoader
import os 
from PIL import Image
import json
from tqdm import tqdm


class TxtDataset(Dataset):
    def __init__(self, foloder_name):
        print(f'Loading folder data from {foloder_name}.')
        self.image_paths = []
        self.caption_paths = []
        for each_file in os.listdir(foloder_name):
            if each_file.endswith('.jpg'):
                self.image_paths.append(os.path.join(foloder_name, each_file))
                self.caption_paths.append(os.path.join(foloder_name, each_file.replace('.jpg', '.txt')))
        print('Done loading data. Len of images:', len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        caption_path = str(self.caption_paths[idx])
        
        with open(caption_path, 'r') as f:
            caption = f.read()
            if not caption:
                caption = '无'
        return img_path, caption
    

if __name__ == '__main__':
    root_path = "./zero23m"
    all_folders = sorted(os.listdir(root_path))

    for each_folder in all_folders:
        each_folder_path = os.path.join(root_path, each_folder)
        each_dataset = JsonDataset(each_folder_path)
        each_dataloader = DataLoader(each_dataset, batch_size=3, shuffle=False, num_workers=0)

        for iii, (image_path, text,) in enumerate(tqdm(each_dataloader)):
            print(image_path, text,)