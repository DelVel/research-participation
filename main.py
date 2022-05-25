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

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from src import COCOSystem


def main():
    seed_everything(42, workers=True)
    args = init_args()
    override = {}
    if args.strategy == 'ddp':
        override['strategy'] = DDPPlugin(find_unused_parameters=False)
    if args.logger:
        wandb_logger = WandbLogger(project='coco-system',
                                   name=COCOSystem.get_run_name())
        override['logger'] = wandb_logger
    trainer = pl.Trainer.from_argparse_args(args, **override)
    model = COCOSystem(args)
    trainer.fit(model)


def init_args():
    parser = ArgumentParser(description='COCO System')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = COCOSystem.add_module_specific_args(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
