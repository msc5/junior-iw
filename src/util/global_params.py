
MODELS = [
    'ConvLSTM',
    'ConvLSTM_REF',
    'LSTM'
]

DATASETS = [
    'GeneratedSins',
    'GeneratedNoise',
    'Stocks',
    'MovingMNIST',
    'KTH',
    'BAIR'
]

KTH_CLASSES = [
    'boxing',
    'handclapping',
    'handwaving',
    'jogging',
    'running',
    'walking'
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
        'default': 0.5,
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
    },
    'hid_size': {
        'description': 'Hidden size or channel dimension',
        'default': 64,
        'type': int
    }
}
