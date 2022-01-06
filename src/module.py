import random

import einops
import nltk
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from torchvision.transforms import Compose, ToTensor, Lambda, RandomCrop

from src.loss import minimize_maximum_cosine
from src.model import ResNetFeature, TextGRU, ImageTrans
from src.vocab import vocab, padding_len, padding_idx


class COCOSystem(pl.LightningModule):
    def __init__(self, latent_dim, text_embed_dim, batch_size):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be even"
        self.save_hyperparameters()
        self.caption_per_img = 5

        self.resnet = ResNetFeature()
        self.linear = torch.nn.Linear(ResNetFeature.sequence_dim, latent_dim)
        self.image_trans = ImageTrans(latent_dim)

        self.gru = TextGRU(text_embed_dim, latent_dim // 2)

        self.loss = minimize_maximum_cosine

    def training_step(self, batch, batch_idx):
        img, text = batch

        img = self.resnet(img)
        img = self.linear(img)
        img = self.image_trans(img)

        text = einops.rearrange(text, 'b g l -> (b g) l')
        text = self.gru(text)
        text = einops.rearrange(text, '(b g) l -> b g l', g=self.caption_per_img)

        loss = self.loss(img, text)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        coco_train = CocoCaptions(root='D:/dataset/coco2014/train2014',
                                  annFile='D:/dataset/coco2014/annotations_trainval2014/annotations/'
                                          'captions_train2014.json',
                                  transform=Compose([RandomCrop(224, pad_if_needed=True), ToTensor()]),
                                  target_transform=Lambda(self.word2idx))
        return DataLoader(coco_train, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        coco_val = CocoCaptions(root='D:/dataset/coco2014/val2014',
                                annFile='D:/dataset/coco2014/annotations_trainval2014/annotations/'
                                        'captions_val2014.json',
                                transform=Compose([RandomCrop(224, pad_if_needed=True), ToTensor()]),
                                target_transform=Lambda(self.word2idx))
        return DataLoader(coco_val, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8)

    def test_dataloader(self):
        coco_test = CocoCaptions(root='D:/dataset/coco2014/test2014',
                                 annFile='D:/dataset/coco2014/image_info_test2014/annotations/'
                                         'image_info_test2014.json',
                                 transform=Compose([RandomCrop(224, pad_if_needed=True), ToTensor()]))
        return DataLoader(coco_test, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8)

    @staticmethod
    def word2idx(words):
        tensor_list = []
        for word in words:
            tokenize = nltk.tokenize.word_tokenize(word.lower())
            start_end = ['<start>'] + tokenize + ['<end>']
            idx_list = [vocab(token) for token in start_end]
            padding_count = (padding_len - len(idx_list))
            assert padding_count >= 0, f'Exceeded maximum padding length: {len(idx_list)} > {padding_len}'
            idx_list += [padding_idx] * padding_count
            tensor = torch.LongTensor(idx_list)
            tensor_list.append(tensor)
        tensor_list = random.sample(tensor_list, 5)
        sequence = torch.stack(tensor_list)
        return sequence
