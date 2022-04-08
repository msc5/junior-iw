
import os
import argparse

from torch.utils.data import DataLoader

from .train.lightning import Lightning

MODELS = ['ConvLSTM', 'ConvLSTM_REF', 'LSTM']
DATASETS = ['GeneratedSins', 'GeneratedNoise', 'MovingMNIST', 'KTH', 'BAIR']


def add_args(parser):
    parser.add_argument('--device', choices=['gpu', 'cpu'], help='Device')
    parser.add_argument('--num_workers', type=int,
                        help='Number of DataLoader workers')
    parser.add_argument('--num_layers',
                        type=int, help='Number of ConvLSTM layers')
    parser.add_argument('--seq_len', type=int, help='Total length of sequence')
    parser.add_argument('--fut_len', type=int, help='Total length to predict')
    parser.add_argument('--batch_size', type=int, help='Batch Size')
    parser.add_argument('--shuffle', type=bool, help='Shuffle Dataset')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--max_epochs', type=int,
                        help='Maximum amount of epochs')
    parser.add_argument('--max_time', help='Maximum amount of time (s)')
    parser.add_argument('--criterion', help='Loss function')
    parser.add_argument('--image_interval',
                        help='Number of steps to make images')


parser = argparse.ArgumentParser(
    prog='video_prediction',
    description='Matthew Coleman Junior IW Video Prediction'
)
add_args(parser)

subparsers = parser.add_subparsers(help='Test or Train a Model')

train_parser = subparsers.add_parser('train', help='Train a Model')
train_parser.add_argument('model', choices=MODELS, help='Specify Model')
train_parser.add_argument('dataset', choices=DATASETS, help='Specify Dataset')
add_args(train_parser)

test_parser = subparsers.add_parser('test', help='Test a Model')
test_parser.add_argument('model', choices=MODELS, help='Specify Model')
test_parser.add_argument('dataset', choices=DATASETS, help='Specify Dataset')
add_args(test_parser)

if __name__ == "__main__":

    args = {k: v for (k, v) in
            vars(parser.parse_args()).items() if v is not None}

    n_columns = 80
    print('-' * n_columns)
    print(f'{"Model":>20} : {args.get("model"):<20}')
    print(f'{"Dataset":>20} : {args.get("dataset"):<20}')
    print('-' * n_columns)

    opts = {
        'model': args.get('model'),
        'dataset': args.get('dataset'),
        'device': args.get('device', 'gpu'),
        'num_workers': args.get('num_workers', 4),
        'num_layers': args.get('num_layers', 1),
        'seq_len': args.get('seq_len', 20),
        'fut_len': args.get('fut_len', 10),
        'batch_size': args.get('batch_size', 4),
        'shuffle': args.get('shuffle', True),
        'learning_rate': args.get('learning_rate', 0.001),
        'max_epochs': args.get('max_epochs', 300),
        'criterion': args.get('criterion', 'MSELoss'),
        'image_interval': args.get('image_interval', 500),
    }

    # Initialize Model
    if opts['model'] == 'ConvLSTM':
        from .arch.convlstm import ConvLSTMSeq2Seq as ConvLSTM
        model = ConvLSTM(1, 64, opts['num_layers'])
    elif opts['model'] == 'ConvLSTM_REF':
        from .arch.convlstm_ref import EncoderDecoderConvLSTM as ConvLSTM_REF
        model = ConvLSTM_REF(64, 1)
    elif opts['model'] == 'LSTM':
        from .arch.lstm import LSTMSeq2Seq as LSTM
        model = LSTM(1, 64)

    # Initialize Dataset and DataLoader
    if opts['dataset'] == 'GeneratedSins':
        from .data.generators import GeneratedSins
        dataset = GeneratedSins(opts['seq_len'])
    elif opts['dataset'] == 'GeneratedNoise':
        from .data.generators import GeneratedNoise
        dataset = GeneratedNoise(opts['seq_len'])
    elif opts['dataset'] == 'MovingMNIST':
        from .data.datasets.MovingMNIST.MovingMNIST import MovingMNIST
        dataset = MovingMNIST(
            train=True,
            seq_len=opts['seq_len'],
            image_size=64,
            deterministic=True,
            num_digits=2
        )
    elif opts['dataset'] == 'KTH':
        from .data.datasets.KTH.KTH import KTH
        dataset = KTH.make_dataset(
            'src/data/datasets/KTH/raw',
            64,
            (opts['seq_len']),
            True
        )
    elif opts['dataset'] == 'BAIR':
        from .data.datasets.BAIR.BAIR import BAIR
        dataset = BAIR('src/data/datasets/BAIR/raw')
    loader = DataLoader(
        dataset=dataset,
        batch_size=opts['batch_size'],
        shuffle=True,
        num_workers=opts['num_workers']
    )

    # Start Training
    lightning = Lightning(model, loader, opts)
    lightning.fit()
