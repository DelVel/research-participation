from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from src.module import COCOSystem


def main():
    args = init_args()
    override = {}
    if args.strategy == 'ddp':
        override['strategy'] = DDPPlugin(find_unused_parameters=False)
    if args.logger:
        override['logger'] = WandbLogger(project='coco-system')
    trainer = pl.Trainer.from_argparse_args(
        args,
        **override
    )
    model = COCOSystem(args)
    trainer.fit(model)


def init_args():
    parser = ArgumentParser(description='COCO System')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = COCOSystem.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
