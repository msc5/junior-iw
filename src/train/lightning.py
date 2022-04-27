
import io
import os
import torch
import torch.nn as nn
import json
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from torch.utils.tensorboard import SummaryWriter

from ..analysis.plots import plot_seqs, plot_to_tensor, plot_loss_over_seq

GLOBAL_METRICS = {
    'Metrics': {
        'sequence':
            ['Multiline', ['sequence/loss']],
        'loss':
            ['Multiline', ['loss/train', 'loss/val', 'loss/test']],
        'output_range':
            ['Multiline', ['output_range/max', 'output_range/min']]
    }
}


class Lightning (pl.LightningModule):

    def __init__(
            self,
            opts: object,
            model: nn.Module = None,
            loaders: object = None,
    ):
        super(Lightning, self).__init__()

        self.save_hyperparameters(ignore=['model', 'loaders'])
        self.opts = opts
        self.model = model
        self.loaders = loaders
        self.total_steps = {k: len(v) for (k, v) in loaders.items()}

        # Initialize Criterion
        if opts['criterion'] == 'MSELoss':
            self.criterion = torch.nn.MSELoss()
        elif opts['criterion'] == 'CrossEntropyLoss':
            self.criterion = torch.nn.CrossEntropyLoss()

        self.steps = {'train': 0, 'test': 0, 'val': 0}
        self.seq_losses = self.avg_losses = 0

    def make_label(self):
        epoch, step = self.current_epoch, self.get_step()
        model, dataset = self.opts['model'], self.opts['dataset']
        return f'{model}_{dataset}_epoch_{epoch}_step_{step}'

    def make_image(self, x, y, output):
        # (batch_size, seq_len, img_chan, img_h, img_w)
        truth = torch.cat([x, y], dim=1)[0]
        prediction = torch.cat([x, output], dim=1)[0]
        difference = torch.cat([
            torch.zeros(x.shape).to(self.device),
            (y - output).abs()
        ], dim=1)[0]
        combined = torch.cat([truth, prediction, difference])
        return make_grid(combined, nrow=self.opts['seq_len'])

    def plot_seq_loss(self, losses):
        fig = plot_loss_over_seq(losses)
        return plot_to_tensor(fig)

    def plot_pred(self, x, y, output):
        fig = plot_seqs(x, y, output)
        return plot_to_tensor(fig)

    def forward(self, x):
        return self.model(x, self.opts['fut_len'])

    def fit(self):
        name = f'{self.opts["model"]}_{self.opts["dataset"]}'
        results_dir = f'{self.opts["results_dir"]}/train'
        logger = TensorBoardLogger(
            results_dir, name=name, version=self.opts['task_id'])
        logger.experiment.add_custom_scalars(GLOBAL_METRICS)
        checkpoint = ModelCheckpoint(
            every_n_train_steps=(self.total_steps['train'] // 2))
        progress_theme = RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        )
        progress_bar = RichProgressBar(leave=True, theme=progress_theme)
        trainer = pl.Trainer(
            logger=logger,
            accelerator=self.opts['device'],
            devices=1,
            max_epochs=self.opts['max_epochs'],
            callbacks=[checkpoint, progress_bar],
            limit_val_batches=self.opts['n_val_batches'],
            val_check_interval=self.opts['val_interval'])
        trainer.fit(self, ckpt_path=self.opts['checkpoint_path'])

    def test(self):
        name = f'{self.opts["model"]}_{self.opts["dataset"]}'
        results_dir = f'{self.opts["results_dir"]}/test'
        logger = TensorBoardLogger(
            results_dir, name=name, version=self.opts['task_id'])
        logger.experiment.add_custom_scalars(GLOBAL_METRICS)
        trainer = pl.Trainer(
            logger=logger,
            accelerator=self.opts['device'],
            devices=1,
            max_epochs=self.opts['max_epochs'],
            limit_test_batches=self.opts['n_test_batches'])
        trainer.test(self.model, dataloaders=self.test_dataloader())

    def save(self, name: str = 'checkpoint'):
        ckpt_dir = f'{self.logger.log_dir}/checkpoints/{name}.ckpt'
        self.trainer.save_checkpoint(ckpt_dir)

    def get_step(self):
        return self.steps['train'] + self.steps['test'] + self.steps['val']

    def training_step(self, batch, i):
        self.steps['train'] += 1
        inp_len = self.opts['seq_len'] - self.opts['fut_len']
        x, y = batch[:, :inp_len], batch[:, inp_len:]
        output = self.forward(x)
        loss = self.criterion(output.squeeze(), y.squeeze())
        writer, step = self.logger.experiment, self.get_step()
        writer.add_scalar('loss/train', loss, step)
        writer.add_scalar('output_range/max', output.max(), step)
        writer.add_scalar('output_range/min', output.min(), step)
        return {'loss': loss}

    def validation_step(self, batch, i):
        self.steps['val'] += 1
        inp_len = self.opts['seq_len'] - self.opts['fut_len']
        x, y = batch[:, :inp_len], batch[:, inp_len:]
        output = self.forward(x)
        loss = self.criterion(output.squeeze(), y.squeeze())
        writer, step = self.logger.experiment, self.get_step()
        if self.opts['dataset'] in {'MovingMNIST', 'KTH', 'BAIR'}:
            img_pred = self.make_image(x, y, output)
        else:
            img_pred = self.plot_pred(x, y, output)
        label = self.make_label()
        writer.add_image(f'val_{label}', img_pred, step)
        writer.add_scalar('loss/val', loss, step)
        return {'loss': loss}

    def test_step(self, batch, i):
        self.steps['test'] += 1
        inp_len = self.opts['seq_len'] - self.opts['fut_len']
        x, y = batch[:, :inp_len], batch[:, inp_len:]
        output = self.forward(x)
        loss = self.criterion(output.squeeze(), y.squeeze())
        self.add_seq_loss(output, y)
        self.add_avg_loss(loss)
        if not self.opts['no_images']:
            writer, step = self.logger.experiment, self.get_step()
            if self.opts['dataset'] in {'MovingMNIST', 'KTH', 'BAIR'}:
                img_pred = self.make_image(x, y, output)
            else:
                img_pred = self.plot_pred(x, y, output)
            label = self.make_label()
            writer.add_image(f'test_{label}_prediction', img_pred, step)
        return {'loss': loss}

    def add_avg_loss(self, loss):
        self.avg_losses += loss

    def add_seq_loss(self, output, y):
        pred_len = y.shape[1]
        self.seq_losses += torch.tensor(
            [self.criterion(output[:, i], y[:, i]) for i in range(pred_len)])

    def on_test_end(self):
        avg_loss = self.avg_losses / self.steps['test']
        avg_seq_losses = self.seq_losses / self.steps['test']
        writer = self.logger.experiment
        for i, loss in enumerate(avg_seq_losses):
            writer.add_scalar('sequence/loss', loss, i)
        writer.add_scalar('loss/test', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.opts['learning_rate'],
            betas=(0.9, 0.98))
        return optimizer

    def train_dataloader(self):
        return self.loaders['train']

    def val_dataloader(self):
        return self.loaders['val']

    def test_dataloader(self):
        return self.loaders['test']


if __name__ == "__main__":

    # batch_size = 4
    # model = ConvLSTMSeq2Seq(64, (1, 64, 64), 1)
    # opts = {
    #     'batch_size': batch_size,
    #     'learning_rate': 0.001,
    #     'epochs': 300,
    # }
    # lightning = VideoPredictionLightning(model, opts)
    # lightning.fit()

    batch_size = 20
    model = LSTMSeq2Seq(10, 64, 1)
    opts = {
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'epochs': 300,
        'device': 'gpu'
    }
    lightning = SequencePredictionLightning(model, opts)
    lightning.fit()
