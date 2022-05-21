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

from abc import abstractmethod, ABCMeta

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions

dataset_root = './dataset/coco2014'
train_val_annotations_root = f'{dataset_root}/annotations_trainval2014' \
                             f'/annotations'
train_caption = f'{train_val_annotations_root}/captions_train2014.json'
val_caption = f'{train_val_annotations_root}/captions_val2014.json'
test_info = f'{dataset_root}/image_info_test2014/annotations' \
            f'/image_info_test2014.json'
train_root = f'{dataset_root}/train2014'
val_root = f'{dataset_root}/val2014'
test_root = f'{dataset_root}/test2014'


class COCODatasetSystem(pl.LightningModule, metaclass=ABCMeta):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("COCODatasetSystem")
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument("--num_worker", type=int, default=4)
        parser.add_argument("--no_persistent_workers", action="store_false")
        parser.add_argument("--no_pin_memory", action="store_false")

        return parent_parser

    def get_dataloader(self, stage):
        parser = self.hparams.parser
        ann_file, root, shuffle = self.get_stage_dataloader_param(stage)
        dataset = CocoCaptions(
            root=root,
            annFile=ann_file,
            transform=self.get_image_transform(),
            target_transform=self.get_text_transform()
        )
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=parser.batch_size,
            num_workers=parser.num_worker,
            persistent_workers=parser.no_persistent_workers,
            pin_memory=parser.no_pin_memory
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

    @abstractmethod
    def get_image_transform(self):
        pass

    @abstractmethod
    def get_text_transform(self):
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
