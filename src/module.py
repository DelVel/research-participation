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

import einops
import torch
from einops import rearrange
from tqdm import tqdm

from src.datamodule import COCODatasetSystem
from src.loss import TripletLoss
from src.model import ImageTrans, TextGRU
from src.similarity import ChamferSimilarity


class COCOSystem(COCODatasetSystem):
    image_model_cls = ImageTrans
    text_model_cls = TextGRU
    similarity_cls = ChamferSimilarity
    loss_func_cls = TripletLoss

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
        parser.add_argument('--latent_dim', type=int, default=256)
        cls.image_model_cls.add_module_specific_args(parent_parser)
        cls.text_model_cls.add_module_specific_args(parent_parser)
        cls.similarity_cls.add_module_specific_args(parent_parser)
        cls.loss_func_cls.add_module_specific_args(parent_parser)
        return parent_parser

    def get_image_transform(self):
        return self.img_model.get_transform()

    def get_text_transform(self):
        return self.txt_model.get_transform()

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

    def forward(self, img, text):
        img = self.img_model(img)
        text = self._process_text(text)
        return img, text

    def _process_text(self, text):
        g_dim = text.shape[1]
        text = einops.rearrange(text, 'b g l -> (b g) l')
        text = self.txt_model(text)
        text = einops.rearrange(text, '(b g) l -> b g l', g=g_dim)
        return text

    def training_step(self, batch, batch_idx):
        loss = self._loss_of_batch(batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, text = batch
        img, text = self.forward(img, text)
        loss = self.loss(img, text)
        self.log("val_loss", loss)
        return img.detach().cpu(), text.detach().cpu()

    def validation_epoch_end(self, outputs):
        imgs, txts = zip(*outputs)
        self._rank_i2t(imgs, txts)
        self._rank_t2i(imgs, txts)

    def _rank_i2t(self, imgs, txts):
        txt = torch.cat(txts, dim=0)
        g = txt.shape[1]
        txt = rearrange(txt, 'b g ... -> (b g) 1 ...')
        acc = 0
        res = [0, 0, 0]
        for img in tqdm(imgs, desc=f"i2t{random.randint(0, 100)}"):
            sim = self.similarity(img, txt)
            end = acc + img.shape[0]
            ind_start = torch.arange(acc, end).unsqueeze_(1)
            ind_end = ind_start + g
            acc = end
            top1_ind = sim.topk(1, dim=1).indices
            res[0] += self._rank_tensor(ind_start, top1_ind, ind_end)
            top5_ind = sim.topk(5, dim=1).indices
            res[1] += self._rank_tensor(ind_start, top5_ind, ind_end)
            top10_ind = sim.topk(10, dim=1).indices
            res[2] += self._rank_tensor(ind_start, top10_ind, ind_end)
        self.log("i2t_top1", res[0] / acc)
        self.log("i2t_top5", res[1] / acc)
        self.log("i2t_top10", res[2] / acc)

    @staticmethod
    def _rank_tensor(ge, target, lt):
        ge_res = ge <= target
        lt_res = target < lt
        # noinspection PyUnresolvedReferences
        return ge_res.logical_and_(lt_res).any(dim=1).sum().item()

    def _rank_t2i(self, imgs, txts):
        img = torch.cat(imgs, dim=0)
        acc = 0
        res = [0, 0, 0]
        for txt in tqdm(txts, desc=f"t2i{random.randint(0, 100)}"):
            g = txt.shape[1]
            txt = rearrange(txt, 'b g ... -> (b g) 1 ...')
            sim = self.similarity(txt, img)
            end = acc + txt.shape[0]
            ind = torch.arange(acc, end) \
                .div_(g, rounding_mode='trunc') \
                .unsqueeze_(1)
            acc = end
            top1_ind = sim.topk(1, dim=1).indices
            res[0] += self._eq_tensor(top1_ind, ind)
            top5_ind = sim.topk(5, dim=1).indices
            res[1] += self._eq_tensor(top5_ind, ind)
            top10_ind = sim.topk(10, dim=1).indices
            res[2] += self._eq_tensor(top10_ind, ind)
        self.log("t2i_top1", res[0] / acc)
        self.log("t2i_top5", res[1] / acc)
        self.log("t2i_top10", res[2] / acc)

    @staticmethod
    def _eq_tensor(target, eq):
        # noinspection PyUnresolvedReferences
        return (target == eq).any(dim=1).sum().item()

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters())
        return optimizer

    def _loss_of_batch(self, batch):
        img, text = batch
        img, text = self.forward(img, text)
        loss = self.loss(img, text)
        return loss
