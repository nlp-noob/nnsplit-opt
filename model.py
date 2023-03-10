from pathlib import Path
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from sklearn.model_selection import train_test_split
from torch.utils import data
from utils.dataset import SplitDataset
from argparse import ArgumentParser
from utils.utils import postprocess
import json


class Network(pl.LightningModule):
    ONNX_NAME = "model.onnx"

    def __init__(self, text_dataset, labeler, hparams):
        super().__init__()
        self.text_dataset = text_dataset
        self.labeler = labeler
        self.save_hyperparameters(hparams)

        self.embedding = nn.Embedding(256, 64)
        self.downsample = nn.Conv1d(64, 64, kernel_size=2, stride=2)

        self.lstm1 = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)

        self.out = nn.Linear(256, 2 * len(hparams.predict_indices))

    def prepare_data(self):
        dataset = SplitDataset(
            self.text_dataset,
            self.labeler,
            500,
            800,
            20,
            return_indices=self.hparams.predict_indices,
        )

        train_indices, valid_indices = train_test_split(
            np.arange(len(dataset)), test_size=self.hparams.test_size, random_state=1234
        )
        self.train_dataset = data.Subset(dataset, train_indices)
        self.valid_dataset = data.Subset(dataset, valid_indices)

    def forward(self, x):
        input_length = x.shape[1]

        h = self.embedding(x.long())
        h = self.downsample(h.permute(0, 2, 1)).permute(0, 2, 1)

        h, _ = self.lstm1(h)
        h, _ = self.lstm2(h)
        h, _ = self.lstm3(h)

        h = self.out(h).reshape(-1, input_length, len(self.hparams.predict_indices))
        return h

    def loss(self, y_hat, y):
        weight = (
            torch.tensor(self.hparams.level_weights)
            .view((1, 1, len(self.hparams.level_weights)))
            .to(y_hat.device)
        )

        return F.binary_cross_entropy_with_logits(
            y_hat, y.float(), pos_weight=torch.tensor(10.0), weight=weight
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        val_loss = self.loss(y_hat, y)

        threshold = 0.5
        n_labels = y.shape[-1]

        y_flat = y.view((-1, n_labels))
        pred_flat = y_hat.view((-1, n_labels)) > threshold

        tp = ((pred_flat == 1) & (y_flat == 1)).sum(dim=0)
        fp = ((pred_flat == 1) & (y_flat == 0)).sum(dim=0)
        fn = ((pred_flat == 0) & (y_flat == 1)).sum(dim=0)

        return {"val_loss": val_loss, "tp": tp, "fp": fp, "fn": fn}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tp = torch.stack([x["tp"] for x in outputs]).sum(dim=0)
        fp = torch.stack([x["fp"] for x in outputs]).sum(dim=0)
        fn = torch.stack([x["fn"] for x in outputs]).sum(dim=0)

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        print()

        for i in range(len(f1)):
            print(
                f"f1={f1[i]:.3f}\tprecision={precision[i]:.3f}\trecall={recall[i]:.3f}"
            )

        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        adam = torch.optim.AdamW(self.parameters())

        return [adam], []

    def train_dataloader(self):
        # define 1 epoch = n random samples from train data
        # multiprocessing with spacy leaks memory so could go OOM without a sample limit
        # reload_dataloaders_every_epoch must be True in trainer
        # so that memory is cleaned up after each epoch

        epoch_indices = np.random.choice(
            np.arange(len(self.train_dataset)), self.hparams.train_size
        )
        epoch_sample = data.Subset(self.train_dataset, epoch_indices)

        return data.DataLoader(
            epoch_sample,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=6,
            collate_fn=SplitDataset.collate_fn,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.valid_dataset,
            batch_size=256,
            shuffle=False,
            num_workers=6,
            collate_fn=SplitDataset.collate_fn,
        )

    def store(self, directory, metadata):
        store_directory = Path(directory)
        store_directory.mkdir(exist_ok=True, parents=True)

        sample = torch.zeros([1, 100], dtype=torch.uint8)
        model_path = store_directory / self.ONNX_NAME

        torch.onnx.export(
            self.float().cpu(),
            sample.cpu(),
            model_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 1: "length"},
                "output": {0: "batch", 1: "length"},
            },
        )
        postprocess(
            model_path,
            metadata,
        )

    @staticmethod
    def get_parser():
        parser = ArgumentParser()
        parser.add_argument(
            "--test_size", type=int, help="Number of samples for test set."
        )
        parser.add_argument(
            "--train_size",
            type=int,
            help="Number of samples to train on for one epoch. "
            "Will be sampled without replacement from the text dataset.",
        )
        parser.add_argument(
            "--predict_indices",
            nargs="+",
            type=int,
            default=[],
            help="Which levels of the splits to predict.",
        )
        parser.add_argument(
            "--level_weights",
            nargs="+",
            type=float,
            default=[],
            help="Determines how much each level contributes to the loss. Must have the same length as the indices to predict.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
        )
        parser = Trainer.add_argparse_args(parser)

        parser.set_defaults(
            train_size=1_000_000,
            test_size=50_000,
            batch_size=128,
            max_epochs=1,
            reload_dataloaders_every_epoch=True,
        )

        return parser
