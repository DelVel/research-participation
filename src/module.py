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

import einops
import torch

from src.datamodule import COCODatasetSystem
from src.loss import ContrastiveLoss
from src.model import ImageTrans
from src.model import TextGRU


class COCOSystem(COCODatasetSystem):
    @staticmethod
    def add_module_specific_args(parent_parser):
        super().add_module_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("COCOModel")
        parser.add_argument('--latent_dim', type=int, default=256)

        TextGRU.add_module_specific_args(parent_parser)
        ImageTrans.add_module_specific_args(parent_parser)
        ContrastiveLoss.add_module_specific_args(parent_parser)

        return parent_parser

    # noinspection PyUnusedLocal
    def __init__(self, parser):
        super().__init__()
        latent_dim = parser.latent_dim
        assert latent_dim % 2 == 0, "latent_dim must be even"
        self.save_hyperparameters()

        self.image_trans = ImageTrans(parser, out_dim=latent_dim)
        self.gru = TextGRU(parser, out_dim=latent_dim)

        self.loss = ContrastiveLoss(parser)

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

    def _loss_of_batch(self, batch):
        img, text = batch
        img, text = self.forward(img, text)
        loss = self.loss(img, text)
        return loss
