
import os

from torch.utils.data import DataLoader

from .util.parsers import main_parser
from .train.lightning import Lightning


if __name__ == "__main__":

    from rich import print

    opts = main_parser()
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
        model = ConvLSTM(
            opts['inp_chan'],
            opts['hid_size'],
            opts['num_layers'])
    elif opts['model'] == 'ConvLSTM_REF':
        from .arch.convlstm_ref import EncoderDecoderConvLSTM as ConvLSTM_REF
        model = ConvLSTM_REF(opts['hid_size'], opts['inp_chan'])
    elif opts['model'] == 'LSTM':
        from .arch.lstm import LSTMSeq2Seq as LSTM
        model = LSTM(opts['inp_size'], opts['hid_size'])
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
