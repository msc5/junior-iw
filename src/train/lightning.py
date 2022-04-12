
import io
import os
import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

from ..analysis.plots import plot_seqs, plot_to_tensor

GLOBAL_METRICS = {
    'Metrics': {
        'loss':
        ['Multiline', ['loss/train', 'loss/validation']],
        'output_range':
        ['Multiline', ['output_range/max', 'output_range/min']]
    }
}


class Lightning (pl.LightningModule):

    def __init__(self, model, loader, opts: object):
        super(Lightning, self).__init__()

        self.model = model
        self.loader = loader

        self.opts = opts

        # Initialize Criterion
        if opts['criterion'] == 'MSELoss':
            self.criterion = torch.nn.MSELoss()
        elif opts['criterion'] == 'CrossEntropyLoss':
            self.criterion = torch.nn.CrossEntropyLoss()

    def make_label(self):
        epoch, step = self.current_epoch, self.global_step
        model, dataset = self.opts['model'], self.opts['dataset']
        return f'{model}_{dataset}_epoch_{epoch}_step_{step}'

    def make_image(self, x, y, output):
        # (batch_size, seq_len, img_chan, img_h, img_w)
        truth = torch.cat([x, y], dim=1)[0]
        prediction = torch.cat([x, output], dim=1)[0]
        combined = torch.cat([truth, prediction])
        image = make_grid(combined, nrow=self.opts['seq_len'])
        return image, self.make_label()

    def make_plot(self, x, y, output):
        fig = plot_seqs(x, y, output)
        image = plot_to_tensor(fig)
        return image, self.make_label()

    def forward(self, x):
        if self.opts['model'] in ['FutureGAN']:
            return self.model(x)
        else:
            return self.model(x, self.opts['fut_len'])

    def fit(self):
        name = f'{self.opts["model"]}_{self.opts["dataset"]}'
        logger = TensorBoardLogger('tensorboard', name=name)
        logger.experiment.add_custom_scalars(GLOBAL_METRICS)
        trainer = pl.Trainer(
            logger=logger,
            accelerator=self.opts['device'],
            devices=1,
            max_epochs=self.opts['max_epochs'])
        trainer.fit(self)

    def training_step(self, batch, i):
        inp_len = self.opts['seq_len'] - self.opts['fut_len']
        x, y = batch[:, :inp_len], batch[:, inp_len:]
        output = self.forward(x)
        loss = self.criterion(output.squeeze(), y.squeeze())
        writer = self.logger.experiment
        step = self.global_step
        if step % self.opts['image_interval'] == 0:
            if self.opts['dataset'] in {'MovingMNIST', 'KTH', 'BAIR'}:
                image, label = self.make_image(x, y, output)
            else:
                image, label = self.make_plot(x, y, output)
            writer.add_image(label, image, step)
        logs = {
            'loss': {'train': loss},
            'output_range': {
                'max': output.max(),
                'min': output.min()
            }
        }
        for key, val in logs.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    writer.add_scalar(f'{key}/{k}', v, step)
            else:
                writer.add_scalar(key, val, step)
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.opts['learning_rate'],
            betas=(0.9, 0.98))
        return optimizer

    def train_dataloader(self):
        return self.loader


class SequencePredictionLightning (pl.LightningModule):

    def __init__(self, model, opts: object):
        super(SequencePredictionLightning, self).__init__()

        self.model = model

        self.dev = opts.get('device', 'cpu')
        self.epochs = opts.get('epochs', 300)
        self.batch_size = opts.get('batch_size', 10)
        self.lr = opts.get('learning_rate', 0.001)
        self.image_interval = opts.get('image_interval', 500)

        self.criterion = torch.nn.MSELoss()
        self.seq_len = self.fut_len = 20

        self.num_workers = 4

    def make_plot(self, x, y, output):
        fig = plot_seqs(x, y, output)
        image = plot_to_tensor(fig)
        label = f'sequence_epoch_{self.current_epoch}_step_{self.global_step}'
        return image, label

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x, self.fut_len)

    def fit(self):
        logger = TensorBoardLogger('tensorboard', name='LSTM')
        logger.experiment.add_custom_scalars(GLOBAL_METRICS)
        trainer = pl.Trainer(
            logger=logger,
            accelerator=self.dev,
            devices=1,
            max_epochs=self.epochs)
        trainer.fit(self)

    def training_step(self, batch, i):
        x = batch[:, :self.seq_len].unsqueeze(2)
        y = batch[:, self.seq_len:].unsqueeze(2)
        output = self.forward(x)
        loss = self.criterion(output, y)
        writer, step = self.logger.experiment, self.global_step
        if step % self.image_interval == 0:
            image, label = self.make_plot(x, y, output)
            writer.add_image(label, image, step)
        writer.add_scalar('loss/train', loss, step)
        logs = {'loss': {'train': loss}}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.98))
        return optimizer

    def train_dataloader(self):
        data = GeneratedSins(self.seq_len + self.fut_len)
        # data = GeneratedNoise(self.seq_len + self.fut_len)
        loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        return loader


class VideoPredictionLightning (pl.LightningModule):

    def __init__(self, model, opts: object):
        super(VideoPredictionLightning, self).__init__()

        self.model = model
        self.dev = opts.get('device', 'cpu')

        self.criterion = torch.nn.MSELoss()
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.epochs = opts.get('epochs', 300)
        self.batch_size = opts.get('batch_size', 10)
        self.lr = opts.get('learning_rate', 0.001)
        self.seq_len = 10
        self.fut_len = 10

        self.image_interval = 500
        self.num_workers = 2

    def make_image(self, x, y, output):
        # (batch_size, seq_len, img_chan, img_h, img_w)
        truth = torch.cat([x, y], dim=1)[0]
        prediction = torch.cat([x, output], dim=1)[0]
        combined = torch.cat([truth, prediction])
        image = make_grid(combined, nrow=self.seq_len + self.fut_len)
        label = f'epoch_{self.current_epoch}_step_{self.global_step}'
        return image, label

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x, self.fut_len)

    def fit(self):
        logger = TensorBoardLogger('tensorboard', name='ConvLSTM')
        logger.experiment.add_custom_scalars(GLOBAL_METRICS)
        trainer = pl.Trainer(
            logger=logger,
            accelerator=self.dev,
            devices=1,
            # val_check_interval=50,
            max_epochs=self.epochs)
        trainer.fit(self)

    def training_step(self, batch, i):
        # print(batch.shape)
        x = batch[:, :self.seq_len].permute(0, 1, 4, 2, 3)
        y = batch[:, self.seq_len:].permute(0, 1, 4, 2, 3)
        output = self.forward(x)
        loss = self.criterion(output.squeeze(), y.squeeze())
        writer = self.logger.experiment
        step = self.global_step
        if step % self.image_interval == 0:
            image, label = self.make_image(x, y, output)
            writer.add_image(label, image, step)
        logs = {
            'loss': {'train': loss},
            'output_range': {
                'max': output.max(),
                'min': output.min()
            }
        }
        for key, val in logs.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    writer.add_scalar(f'{key}/{k}', v, step)
            else:
                writer.add_scalar(key, val, step)
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.98))
        return optimizer

    def train_dataloader(self):

        # data = KTH(
        #     seq_len=(self.seq_len + self.fut_len),
        #     train=True
        # )
        # loader = torch.utils.data.DataLoader(
        #     dataset=data,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=self.num_workers
        # )
        # return loader

        data = MovingMNIST(
            train=True,
            seq_len=self.seq_len + self.fut_len,
            image_size=64,
            deterministic=True,
            num_digits=2
        )
        loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        return loader


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
