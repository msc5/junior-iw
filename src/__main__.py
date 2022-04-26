
import os
import argparse

from torch.utils.data import DataLoader

from .train.lightning import Lightning

KTH_CLASSES = ['boxing', 'handclapping', 'handwaving',
               'jogging', 'running', 'walking']

MODELS = ['ConvLSTM', 'ConvLSTM_REF', 'LSTM']
DATASETS = [
    'GeneratedSins',
    'GeneratedNoise',
    'Stocks',
    'MovingMNIST',
    'KTH',
    'BAIR'
]

OPTS = {
    'model': {
        'description': 'Model architecture'
    },
    'dataset': {
        'description': 'Dataset'
    },
    'device': {
        'description': 'Device to use',
        'default': 'gpu',
        'choices': ['gpu', 'cpu'],
    },
    'num_workers': {
        'description': 'Number of dataloader workers',
        'default': 4,
        'type': int,
    },
    'num_layers': {
        'description': 'Number of specified model architecture layers',
        'default': 1,
        'type': int,
    },
    'seq_len': {
        'description': 'Total Length of sequence to predict',
        'default': 20,
        'type': int,
    },
    'fut_len': {
        'description': 'Length of predicted sequence',
        'default': 10,
        'type': int,
    },
    'batch_size': {
        'description': 'Size of data batches for each step',
        'default': 4,
        'type': int,
    },
    'n_val_batches': {
        'description': 'Maximum number of batches for validation loop',
        'default': 10,
        'type': int,
    },
    'n_test_batches': {
        'description': 'Maximum number of batches for testing loop',
        'default': 50,
        'type': int,
    },
    'val_interval': {
        'description': 'Fraction of train batches to validate between',
        'default': 0.25,
        'type': float
    },
    'shuffle': {
        'description': 'Whether to shuffle data in dataloader',
        'default': True,
        'type': bool,
    },
    'learning_rate': {
        'description': 'Learning rate of optimizer',
        'default': 0.001,
        'type': float,
    },
    'max_epochs': {
        'description': 'Maximum number of epochs to train/test',
        'default': 300,
        'type': int,
    },
    'criterion': {
        'description': 'Loss function for training',
        'default': 'MSELoss',
        'choices': ['MSELoss'],
    },
    'image_interval': {
        'description': 'How many steps between image generation',
        'default': 500,
        'type': int,
    },
    'kth_classes': {
        'description': 'Which classes to use in the KTH dataset training',
        'default': KTH_CLASSES,
        'nargs': '+',
    },
    'checkpoint_path': {
        'description': 'Path of model checkpoint to resume training or test',
        'default': None,
    },
    'task_id': {
        'description': 'Task ID for slurm scheduler array jobs',
        'default': None,
    },
    'results_dir': {
        'description': 'Directory to log results',
        'default': 'results',
    },
    'mmnist_num_digits': {
        'description': 'Number of digits to use in the MovingMNIST dataset',
        'default': 2,
        'type': int,
    },
    'no_images': {
        'description': 'Set this value True to negate the creation of any images',
        'default': False,
        'type': bool,
    }
}


def add_args(parser):
    for key, val in OPTS.items():
        parser.add_argument(
            f'--{key}',
            help=val.get('help', None),
            choices=val.get('choices', None),
            type=val.get('type', None),
            nargs=val.get('nargs', None),
        )


parser = argparse.ArgumentParser(
    prog='video_prediction',
    description='Matthew Coleman Junior IW Video Prediction'
)

subparsers = parser.add_subparsers(
    help='Test or Train a Model', dest='command')

train_parser = subparsers.add_parser('train', help='Train a Model')
train_parser.add_argument('model', choices=MODELS, help='Specify Model')
train_parser.add_argument('dataset', choices=DATASETS, help='Specify Dataset')
add_args(train_parser)

test_parser = subparsers.add_parser('test', help='Test a Model')
test_parser.add_argument('model', choices=MODELS, help='Specify Model')
test_parser.add_argument('dataset', choices=DATASETS, help='Specify Dataset')
add_args(test_parser)


if __name__ == "__main__":

    from rich import print

    opts = {k: v if v is not None else OPTS[k]['default']
            for (k, v) in vars(parser.parse_args()).items()}
    opts['fut_len'] = opts['seq_len'] // 2

    print(opts)

    n_columns = 80
    print('-' * n_columns)
    print(f'{(opts["command"] + "ing").capitalize():>20}')
    print(f'{"Model":>20} : {opts["model"]:<20}')
    print(f'{"Dataset":>20} : {opts["dataset"]:<20}')
    print('-' * n_columns)

    # Initialize Dataset and DataLoader
    N_train, N_test = 26000, 3500
    if opts['dataset'] == 'GeneratedSins':
        from .data.generators import GeneratedSins
        train_dataset = GeneratedSins(opts['seq_len'], N_train)
        test_dataset = GeneratedSins(opts['seq_len'], N_test)
        opts['inp_size'] = 1
    elif opts['dataset'] == 'GeneratedNoise':
        from .data.generators import GeneratedNoise
        train_dataset = GeneratedNoise(opts['seq_len'], N_train)
        test_dataset = GeneratedNoise(opts['seq_len'], N_test)
        opts['inp_size'] = 1
    elif opts['dataset'] == 'Stocks':
        from .data.datasets.Stocks.Stocks import Stocks
        train_dataset = Stocks(seq_len=opts['seq_len'], split='train')
        test_dataset = Stocks(seq_len=opts['seq_len'], split='test')
        print(len(train_dataset))
        opts['inp_size'] = 1
    elif opts['dataset'] == 'MovingMNIST':
        from .data.datasets.MovingMNIST.MovingMNIST import MovingMNIST
        train_dataset = MovingMNIST(
            train=True,
            data_root='src/data/datasets/MovingMNIST',
            seq_len=opts['seq_len'],
            image_size=64,
            deterministic=True,
            num_digits=opts['mmnist_num_digits'])
        test_dataset = MovingMNIST(
            train=False,
            data_root='src/data/datasets/MovingMNIST',
            seq_len=opts['seq_len'],
            image_size=64,
            deterministic=True,
            num_digits=opts['mmnist_num_digits'])
        opts['inp_chan'] = 1
    elif opts['dataset'] == 'KTH':
        from .data.datasets.KTH.KTH import KTH
        train_dataset = KTH.make_dataset(
            data_dir='src/data/datasets/KTH/raw',
            nx=64,
            seq_len=opts['seq_len'],
            train=True,
            classes=opts['kth_classes'])
        test_dataset = KTH.make_dataset(
            data_dir='src/data/datasets/KTH/raw',
            nx=64,
            seq_len=opts['seq_len'],
            train=False,
            classes=opts['kth_classes'])
        opts['inp_chan'] = 1
    elif opts['dataset'] == 'BAIR':
        from .data.datasets.BAIR.BAIR import BAIR
        train_dataset = BAIR('src/data/datasets/BAIR/raw', train=True)
        test_dataset = BAIR('src/data/datasets/BAIR/raw', train=False)
        opts['inp_chan'] = 3
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opts['batch_size'],
        shuffle=opts['shuffle'],
        num_workers=opts['num_workers'])
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=opts['batch_size'],
        shuffle=False,
        num_workers=opts['num_workers']
    )
    loaders = {
        'train': train_loader,
        'test': test_loader,
        'val': test_loader,
    }

    # Initialize Model for Training
    if opts['model'] == 'ConvLSTM':
        from .arch.convlstm import ConvLSTMSeq2Seq as ConvLSTM
        model = ConvLSTM(opts['inp_chan'], 64, opts['num_layers'])
    elif opts['model'] == 'ConvLSTM_REF':
        from .arch.convlstm_ref import EncoderDecoderConvLSTM as ConvLSTM_REF
        model = ConvLSTM_REF(64, opts['inp_chan'])
    elif opts['model'] == 'LSTM':
        from .arch.lstm import LSTMSeq2Seq as LSTM
        model = LSTM(opts['inp_size'], 64)
    lightning = Lightning(opts, model, loaders)
    if opts['command'] == 'test':
        # Reload model for testing
        model = lightning.load_from_checkpoint(
            opts['checkpoint_path'], model=model, loaders=loaders, opts=opts)
        lightning.model = model

    # Initiate Training or Testing
    if opts['command'] == 'train':
        lightning.fit()
        lightning.save('final')
    elif opts['command'] == 'test':
        lightning.test()
