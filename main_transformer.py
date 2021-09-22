# pip install git+https://github.com/PytorchLightning/lightning-bolts.git@master --upgrade


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy

from .modules import VisionTransformer


PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)


# https://medium.com/pytorch/training-compact-transformers-from-scratch-in-30-minutes-with-pytorch-ff5c21668ed5


def create_model():

    patch_size = 4
    H, W = 32, 32

    assert H % patch_size == 0
    assert W % patch_size == 0
    seqlen = int(H / patch_size * W / patch_size)
    # print(f"Sequence length: {seqlen}")

    model = VisionTransformer(
        dmodel=128,
        # EncoderBlocks
        h=8,
        hidden_mult=4,
        dropout=0.0,
        L=8,
        # PatchEmbeddings
        patch_size=patch_size,
        in_channels=3,
        # ClassificationHead
        n_hidden=1024,
        n_classes=10,
    )

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Parameters: {params / 1e6 :.3f}M")

    return model


class LitVisionTransformer(LightningModule):
    def __init__(self, lr=8e-4):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                self.hparams.lr * 10,
                div_factor=80,
                final_div_factor=1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


if __name__ == "__main__":

    seed_everything(42)

    model = LitVisionTransformer()
    model.datamodule = cifar10_dm

    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=30,
        gpus=AVAIL_GPUS,
        logger=TensorBoardLogger("lightning_logs/", name="vit"),
        callbacks=[LearningRateMonitor(logging_interval="step")],
    )

    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)

    """
    DATALOADER:0 TEST RESULTS

    """
