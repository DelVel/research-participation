import random

import einops
import nltk
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from torchvision.transforms import Compose, ToTensor, Lambda, RandomCrop

from src.dataset import train_root, val_root, train_caption, val_caption, test_root
from src.loss import minimize_maximum_cosine, similarity_criteria
from src.model import ResNetFeature, TextGRU, ImageTrans
from src.vocab import vocab, padding_len, padding_idx, start_token, end_token


class COCOSystem(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("COCOSystem")
        parser.add_argument("--num_worker", type=int, default=4)
        parser.add_argument("--persistent_workers", type=bool, default=False)
        parser.add_argument("--pin_memory", type=bool, default=False)

        parser = parent_parser.add_argument_group("COCOModel")
        parser.add_argument('--latent_dim', type=int, default=256)
        parser.add_argument('--text_embed_dim', type=int, default=256)
        parser.add_argument('--batch_size', type=int, default=32)

        parser = parent_parser.add_argument_group("ResNetConfig")
        parser.add_argument('--pretrained_resnet', type=bool, default=True)
        return parent_parser

    # noinspection PyUnusedLocal
    def __init__(self, latent_dim, text_embed_dim, batch_size, pretrained_resnet, num_worker, persistent_workers,
                 pin_memory):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be even"
        self.save_hyperparameters()

        self.resnet = ResNetFeature(pretrained_resnet)
        self.linear = torch.nn.Identity() if ResNetFeature.sequence_dim == latent_dim else torch.nn.Linear(
            ResNetFeature.sequence_dim, latent_dim)
        self.image_trans = ImageTrans(latent_dim)

        self.gru = TextGRU(text_embed_dim, latent_dim // 2)

        self.loss = minimize_maximum_cosine

    def forward(self, img, text):
        img = self.resnet(img)
        img = self.linear(img)
        img = self.image_trans(img)
        g_dim = text.shape[1]
        text = einops.rearrange(text, 'b g l -> (b g) l')
        text = self.gru(text)
        text = einops.rearrange(text, '(b g) l -> b g l', g=g_dim)
        return img, text

    def training_step(self, batch, batch_idx):
        img, text = batch
        img, text = self.forward(img, text)
        loss = self.loss(img, text)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, text = batch
        img, text = self.forward(img, text)
        loss = self.loss(img, text)
        self.log("val_loss", loss)
        return img, text

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')

    def get_dataloader(self, stage):
        ann_file, root, shuffle = self.get_stage_dataloader_param(stage)
        coco_val = CocoCaptions(
            root=root,
            annFile=ann_file,
            transform=Compose([RandomCrop(224, pad_if_needed=True), ToTensor()]),
            target_transform=Lambda(self.word2idx)
        )
        return DataLoader(
            coco_val,
            shuffle=shuffle,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_worker,
            persistent_workers=self.hparams.persistent_workers,
            pin_memory=self.hparams.pin_memory
        )

    def predict_dataloader(self):
        # Intentionally empty; No prediction for this model
        pass

    @staticmethod
    def get_stage_dataloader_param(stage):
        if stage == 'train':
            root = train_root
            ann_file = train_caption
            shuffle = True
        elif stage == 'val':
            root = val_root
            ann_file = val_caption
            shuffle = False
        elif stage == 'test':
            root = test_root
            ann_file = None
            shuffle = False
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        return ann_file, root, shuffle

    @staticmethod
    def word2idx(words):
        tensor_list = []
        for word in words:
            tokenize = nltk.tokenize.word_tokenize(word.lower())
            start_end = [start_token] + tokenize + [end_token]
            idx_list = [vocab(token) for token in start_end]
            padding_count = (padding_len - len(idx_list))
            assert padding_count >= 0, f'Exceeded maximum padding length: {len(idx_list)} > {padding_len}'
            idx_list += [padding_idx] * padding_count
            tensor = torch.LongTensor(idx_list)
            tensor_list.append(tensor)
        tensor_list = random.sample(tensor_list, 5)
        sequence = torch.stack(tensor_list)
        return sequence
