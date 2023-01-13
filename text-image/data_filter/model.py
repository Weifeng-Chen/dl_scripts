import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
import timm
from torchvision import transforms as T
import open_clip
import torch
from transformers import BertModel, BertTokenizer
from PIL import Image

class AestheticsMLP(pl.LightningModule):
    # 美学判别器是基于CLIP的基础上接了一个MLP
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


class WaterMarkModel(nn.Module):
    def __init__(self, model_path='./watermark_model_v1.pt'):
        super(WaterMarkModel, self).__init__()
        # model definition
        self.model = timm.create_model(
                'efficientnet_b3a', pretrained=True, num_classes=2)

        self.model.classifier = nn.Sequential(
            # 1536 is the orginal in_features
            nn.Linear(in_features=1536, out_features=625),
            nn.ReLU(),  # ReLu to be the activation function
            nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2),
        )
        self.model.load_state_dict(torch.load(model_path))
    def forward(self, x):
        return self.model(x)


class FilterSystem:
    def __init__(
                    self, 
                    clip_model_path="IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese",
                    aesthetics_model_path="./ava+logos-l14-linearMSE.pth",
                    watermark_model_path="./watermark_model_v1.pt"
                ):
        self.clip_model_path = clip_model_path
        self.aesthetics_model_path = aesthetics_model_path
        self.watermark_model_path = watermark_model_path

    def init_clip_model(self, ):
        # 此处初始化clip模型，返回模型、tokenizer、processor
        text_encoder = BertModel.from_pretrained(self.clip_model_path).eval().cuda()
        text_tokenizer = BertTokenizer.from_pretrained(self.clip_model_path)
        clip_model, _, processor = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        clip_model = clip_model.eval().cuda()
        self.text_encoder, self.text_tokenizer, self.clip_model, self.processor = text_encoder, text_tokenizer, clip_model, processor
        print("clip model loaded")
        return None

    def init_aesthetics_model(self, ):
        # 此处初始化美学模型
        self.aesthetics_model = AestheticsMLP(768)
        self.aesthetics_model.load_state_dict(torch.load(self.aesthetics_model_path))
        self.aesthetics_model.eval().cuda()
        print("aesthetics model loaded")
        return None

    def init_watermark_model(self, ):
        self.watermark_model = WaterMarkModel(self.watermark_model_path)
        self.watermark_model.eval().cuda()
        self.watermark_processor =  T.Compose([
                                                T.Resize((256, 256)),
                                                T.ToTensor(),
                                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])
        print("watermark model loaded")
        return None

    def get_image_feature(self, images):
        # 此处返回图像的特征向量
        if isinstance(images, list):
            images = torch.stack([self.processor(image) for image in images]).cuda()
        elif isinstance(images, torch.Tensor):
            images = images.cuda()

        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=1, keepdim=True)
        return image_features
    
    def get_text_feature(self, text):
        # 此处返回文本的特征向量
        if isinstance(text, list) or isinstance(text, str):
            text = self.text_tokenizer(text, return_tensors='pt', padding=True)['input_ids'].cuda()
        elif isinstance(text, torch.Tensor):
            text = text.cuda()

        with torch.no_grad():
            text_features = self.text_encoder(text)[1]
            text_features /= text_features.norm(dim=1, keepdim=True)
        return text_features

    def calculate_clip_score(self, features1, features2):
        # 此处2个特征向量的相似度，输入可以是 图片+文本、文本+文本、图片+图片。
        # 返回的是相似度矩阵，维度为 f1.shape[0] * f2.shape[0]
        score_matrix =  features1 @ features2.t()
        return score_matrix

    def get_aesthetics_score(self, features):
        # 此处返回美学分数，传入的是CLIP的feature, 先计算get_image_feature在传入此函数~(模型是ViT-L-14)
        with torch.no_grad():
            scores = self.aesthetics_model(features)
            scores = scores[:, 0].detach().cpu().numpy()
        return scores
    
    def get_watermark_score(self, images):
        if isinstance(images, list):
            images = torch.stack([self.watermark_processor(image) for image in images]).cuda()
        elif isinstance(images, torch.Tensor):
            images = images.cuda()
        with torch.no_grad():
            pred = self.watermark_model(images)
            watermark_scores = F.softmax(pred, dim=1)[:,0].detach().cpu().numpy()

        return watermark_scores