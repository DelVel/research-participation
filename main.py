from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from src.module import COCOSystem


def main():
    args = init_args()
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[EarlyStopping(monitor="R@1")]
    )
    model = COCOSystem(
        latent_dim=args.latent_dim,
        text_embed_dim=args.text_embed_dim,
        batch_size=args.batch_size,
        pretrained_resnet=args.pretrained_resnet,
        num_worker=args.num_worker,
        persistent_workers=args.persistent_workers
    )
    trainer.fit(model)


def init_args():
    parser = ArgumentParser(description='COCO System')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = COCOSystem.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
