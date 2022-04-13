
import io
import os
import torch
import json
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
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

        self.total_steps = len(self.loader)

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
        log_dir = 'results'
        logger = TensorBoardLogger(log_dir, name=name)
        logger.experiment.add_custom_scalars(GLOBAL_METRICS)
        checkpoint = ModelCheckpoint(
            every_n_train_steps=(self.total_steps // 2))
        trainer = pl.Trainer(
            logger=logger,
            accelerator=self.opts['device'],
            devices=1,
            max_epochs=self.opts['max_epochs'],
            callbacks=[checkpoint])

        logger_path = f'{log_dir}/{name}/version_{logger.version}/opts.json'
        with open(logger_path, 'w', encoding='utf-8') as file:
            json.dump(self.opts, file)
        trainer.fit(self, ckpt_path=self.opts['checkpoint_path'])

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
