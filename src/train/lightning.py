
import os
import torch
import pytorch_lightning as pl

from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from ..data.dataloaders import MovingMNIST
from ..arch.convlstm import ConvLSTMSeq2Seq


class VideoPredictionLightning (pl.LightningModule):

    def __init__(
            self,
            model,
            opts: object,
    ):
        super(VideoPredictionLightning, self).__init__()

        self.model = model
        self.path = os.path.join(os.getcwd(), 'results')

        self.criterion = torch.nn.MSELoss()
        self.epochs = opts.get('epochs', 300)
        self.batch_size = opts.get('batch_size', 10)
        self.lr = opts.get('learning_rate', 0.001)
        self.seq_len = 10
        self.fut_len = 10

        self.image_interval = 50

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
        trainer = pl.Trainer(
            logger=logger,
            accelerator='gpu',
            devices=1,
            max_epochs=self.epochs)
        trainer.fit(self)

    def training_step(self, batch, i):
        x = batch[:, :self.seq_len]
        y = batch[:, self.seq_len:]
        output = self.forward(x)
        loss = self.criterion(output.squeeze(), y.squeeze())
        image, label = self.make_image(x, y, output)
        experiment = self.logger.experiment
        if i % self.image_interval == 0:
            experiment.add_image(label, image, i)
        logs = {
            'train_loss': loss,
            'output_range': {
                'output_max': output.max(),
                'output_min': output.min()
            }
        }
        for key, val in logs.items():
            if isinstance(val, dict):
                experiment.add_scalars(key, val, i)
            else:
                experiment.add_scalar(key, val, i)
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        path = os.path.join(
            os.getcwd(),
            'datasets',
            'MovingMNIST',
            'mnist_test_seq.npy'
        )
        dataset = MovingMNIST(path, train=True)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )
        return loader


if __name__ == "__main__":

    batch_size = 20

    model = ConvLSTMSeq2Seq(64, (1, 64, 64), 2)
    opts = {
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'epochs': 300,
    }
    lightning = VideoPredictionLightning(model, opts)
    lightning.fit()
