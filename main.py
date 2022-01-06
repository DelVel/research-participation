import pytorch_lightning as pl

from src.module import COCOSystem


def main():
    trainer = pl.Trainer(
        gpus=1,
        auto_scale_batch_size="binsearch"
    )
    model = COCOSystem(
        latent_dim=128,
        text_embed_dim=64,
        batch_size=4,
    )
    # trainer.tune(model)
    trainer.fit(model)


if __name__ == '__main__':
    main()
