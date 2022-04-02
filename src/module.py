#  Copyright 2022 Taegyu Park
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import random

import einops
import nltk
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from torchvision.transforms import Compose, ToTensor, Lambda, RandomCrop

from src.dataset import train_root, val_root, train_caption, val_caption, \
    test_root
from src.loss import ContrastiveLoss
from src.model import TextGRU, ImageTrans
from src.vocab import vocab, padding_len, padding_idx, start_token, end_token


class COCOSystem(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("COCOSystem")
        parser.add_argument("--num_worker", type=int, default=4)
        parser.add_argument("--persistent_workers", action="store_true")
        parser.add_argument("--pin_memory", action="store_true")
        parser.add_argument('--batch_size', type=int, default=32)

        parser = parent_parser.add_argument_group("COCOModel")
        parser.add_argument('--latent_dim', type=int, default=256)

        TextGRU.add_module_specific_args(parent_parser)
        ImageTrans.add_module_specific_args(parent_parser)
        ContrastiveLoss.add_module_specific_args(parent_parser)

        return parent_parser

    # noinspection PyUnusedLocal
    def __init__(self, *,
                 num_worker,
                 persistent_workers,
                 pin_memory,
                 batch_size,

                 latent_dim,

                 pretrained_resnet,
                 z_per_img,
                 trans_dim,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dim_feedforward,
                 trans_dropout,
                 layer_norm_eps,
                 norm_first,

                 text_embed_dim,
                 gru_hidden_dim,
                 gru_num_layers,
                 gru_dropout,

                 temperature
                 ):
        super().__init__()
        assert latent_dim % 2 == 0, "latent_dim must be even"
        self.save_hyperparameters()

        self.image_trans = ImageTrans(
            out_dim=latent_dim,

            pretrained=pretrained_resnet,
            zpi=z_per_img,
            trans_dim=trans_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=trans_dropout,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first
        )
        self.gru = TextGRU(
            out_dim=latent_dim,

            text_embed_dim=text_embed_dim,
            gru_hidden_size=gru_hidden_dim,
            gru_layers=gru_num_layers,
            dropout=gru_dropout,
        )

        self.loss = ContrastiveLoss(temperature)

    def forward(self, img, text):
        img = self.image_trans(img)
        text = self._process_text(text)
        return img, text

    def _process_text(self, text):
        g_dim = text.shape[1]
        text = einops.rearrange(text, 'b g l -> (b g) l')
        text = self.gru(text)
        text = einops.rearrange(text, '(b g) l -> b g l', g=g_dim)
        return text

    def training_step(self, batch, batch_idx):
        loss = self._loss_of_batch(batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._loss_of_batch(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
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
            transform=Compose(
                [RandomCrop(224, pad_if_needed=True), ToTensor()]),
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

    def _loss_of_batch(self, batch):
        img, text = batch
        img, text = self.forward(img, text)
        loss = self.loss(img, text)
        return loss

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
            assert padding_count >= 0, f'Exceeded maximum padding length: ' \
                                       f'{len(idx_list)} > {padding_len}'
            idx_list += [padding_idx] * padding_count
            tensor = torch.LongTensor(idx_list)
            tensor_list.append(tensor)
        tensor_list = random.sample(tensor_list, 5)
        sequence = torch.stack(tensor_list)
        return sequence
