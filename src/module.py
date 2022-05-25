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
import time

import torch
from tqdm import tqdm

from src.datamodule import COCODatasetSystem
from src.functional import CosineAnnealingWarmUpRestarts
from src.loss import PVSELoss
from src.model.pienet import PIEText, PIEImage
from src.similarity import ChamferSimilarity


class COCOSystem(COCODatasetSystem):
    image_model_cls = PIEImage
    text_model_cls = PIEText
    similarity_cls = ChamferSimilarity
    loss_func_cls = PVSELoss

    @classmethod
    def get_run_name(cls):
        image_model_name = cls.image_model_cls.__name__
        text_model_name = cls.text_model_cls.__name__
        similarity_name = cls.similarity_cls.__name__
        loss_func_name = cls.loss_func_cls.__name__
        rtime = time.strftime("%Y%m%d-%H%M%S")
        return f"{image_model_name}-{text_model_name}-{similarity_name}" \
               f"-{loss_func_name}-{rtime}"

    @classmethod
    def add_module_specific_args(cls, parent_parser):
        COCODatasetSystem.add_module_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("COCOModel")
        parser.add_argument('--latent_dim', type=int, default=768)
        cls.image_model_cls.add_module_specific_args(parent_parser)
        cls.text_model_cls.add_module_specific_args(parent_parser)
        cls.similarity_cls.add_module_specific_args(parent_parser)
        cls.loss_func_cls.add_module_specific_args(parent_parser)
        return parent_parser

    def get_image_transform(self):
        return self.img_model.get_transform()

    def get_text_transform(self):
        return self.txt_model.get_transform()

    def get_image_collate(self, img):
        return self.img_model.get_collate(img)

    def get_text_collate(self, txt):
        return self.txt_model.get_collate(txt)

    # noinspection PyUnusedLocal
    def __init__(self, parser):
        super().__init__()
        latent_dim = parser.latent_dim
        assert latent_dim % 2 == 0, "latent_dim must be even"
        self.save_hyperparameters()

        self.img_model = self.image_model_cls(parser, out_dim=latent_dim)
        self.txt_model = self.text_model_cls(parser, out_dim=latent_dim)
        self.similarity = self.similarity_cls(parser)
        self.loss = self.loss_func_cls(parser, self.similarity)

    def training_step(self, batch, batch_idx):
        loss = self._loss_of_batch(batch)
        self.log("loss", loss)
        self.log("TL", self.loss.dict['triplet_loss'], prog_bar=True)
        self.log("DL", self.loss.dict['div_loss'], prog_bar=True)
        self.log("ML", self.loss.dict['mmd_loss'], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, text = batch
        img, text = self.forward(img, text)
        loss = self.loss(img, text, self.img_model.stash, self.txt_model.stash)
        self.log("val_loss", loss)
        return img.detach().to(device='cpu',
                               dtype=torch.float32), text.detach().to(
            device='cpu', dtype=torch.float32)

    def validation_epoch_end(self, outputs):
        imgs, txts = zip(*outputs)
        self._rank(imgs, txts, 'i2t')
        self._rank(txts, imgs, 't2i')

    def _rank(self, mod1, mod2, mod1_to_mod2):
        mod2 = torch.cat(mod2, dim=0)
        acc = 0
        res = [0, 0, 0]
        desc = f"{mod1_to_mod2}{random.randint(0, 100)}"
        for mod1_ in tqdm(mod1, desc=desc):
            sim = self.similarity(mod1_, mod2)
            end = acc + mod1_.shape[0]
            ind = torch.arange(acc, end).unsqueeze_(1)
            acc = end
            topk = sim.topk(10, dim=1)
            top10_ind = topk.indices
            res[2] += self._eq_tensor(top10_ind, ind)
            topk = topk.values.topk(5, dim=1)
            top5_ind = topk.indices
            res[1] += self._eq_tensor(top5_ind, ind)
            topk = topk.values.topk(1, dim=1)
            top1_ind = topk.indices
            res[0] += self._eq_tensor(top1_ind, ind)
        self.log(f"{mod1_to_mod2}_top1", res[0] / acc)
        self.log(f"{mod1_to_mod2}_top5", res[1] / acc)
        self.log(f"{mod1_to_mod2}_top10", res[2] / acc)

    @staticmethod
    def _eq_tensor(target, eq):
        # noinspection PyUnresolvedReferences
        return (target == eq).any(dim=1).sum().item()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-10)
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, t_0=50, t_mult=2, eta_max=0.1, t_up=10, gamma=0.5)
        return [optimizer], [scheduler]

    def forward(self, img, text):
        img = self.img_model(img)
        text = self.txt_model(text)
        return img, text

    def _loss_of_batch(self, batch):
        img, text = batch
        img, text = self.forward(img, text)
        loss = self.loss(img, text, self.img_model.stash, self.txt_model.stash)
        return loss
