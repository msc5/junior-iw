
import argparse

from .global_params import MODELS, DATASETS, KTH_CLASSES, OPTS


def add_args(parser):
    for key, val in OPTS.items():
        parser.add_argument(
            f'--{key}',
            help=val.get('help', None),
            choices=val.get('choices', None),
            type=val.get('type', None),
            nargs=val.get('nargs', None),
        )


def main_parser():

    parser = argparse.ArgumentParser(
        prog='video_prediction',
        description='Matthew Coleman Junior IW Video Prediction'
    )

    subparsers = parser.add_subparsers(
        help='Test or Train a Model', dest='command')

    train_parser = subparsers.add_parser('train', help='Train a Model')
    train_parser.add_argument('model',
                              choices=MODELS, help='Specify Model')
    train_parser.add_argument('dataset',
                              choices=DATASETS, help='Specify Dataset')
    add_args(train_parser)

    test_parser = subparsers.add_parser('test', help='Test a Model')
    test_parser.add_argument('model',
                             choices=MODELS, help='Specify Model')
    test_parser.add_argument('dataset',
                             choices=DATASETS, help='Specify Dataset')
    add_args(test_parser)

    opts = {k: v if v is not None else OPTS[k]['default']
            for (k, v) in vars(parser.parse_args()).items()}
    return opts
