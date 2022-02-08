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
    model = COCOSystem(
        num_worker=args.num_worker,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory,
        batch_size=args.batch_size,

        latent_dim=args.latent_dim,

        pretrained_resnet=args.no_pretrained_resnet,
        z_per_img=args.z_per_img,
        trans_dim=args.trans_dim,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        trans_dropout=args.trans_dropout,
        layer_norm_eps=args.layer_norm_eps,
        norm_first=args.norm_first,

        text_embed_dim=args.text_embed_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_num_layers=args.gru_num_layers,
        gru_dropout=args.gru_dropout,

        temperature=args.temperature,
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
