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

import nltk
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from torchvision.transforms import Compose, RandomCrop, ToTensor, Lambda

from src.datamodule.dataset import train_caption, train_root, val_caption, val_root, \
    test_root
from src.datamodule.vocab import start_token, end_token, vocab, padding_len, padding_idx


class COCODatasetSystem(pl.LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("COCOSystem")
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument("--num_worker", type=int, default=4)
        parser.add_argument("--persistent_workers", action="store_true")
        parser.add_argument("--pin_memory", action="store_true")

        return parent_parser

    def get_dataloader(self, stage):
        ann_file, root, shuffle = self.get_stage_dataloader_param(stage)
        dataset = CocoCaptions(
            root=root,
            annFile=ann_file,
            transform=Compose(
                [RandomCrop(224, pad_if_needed=True), ToTensor()]),
            target_transform=Lambda(self.word2idx)
        )
        parser = self.hparams.parser
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=parser.batch_size,
            num_workers=parser.num_worker,
            persistent_workers=parser.persistent_workers,
            pin_memory=parser.pin_memory
        )

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('val')

    def test_dataloader(self):
        return self.get_dataloader('test')

    def predict_dataloader(self):
        # Intentionally empty; No prediction for this model
        pass

    @staticmethod
    def get_stage_dataloader_param(stage):
        if stage == 'train':
            ann_file = train_caption
            root = train_root
            shuffle = True
        elif stage == 'val':
            ann_file = val_caption
            root = val_root
            shuffle = False
        elif stage == 'test':
            ann_file = None
            root = test_root
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
