
import torch

import multiprocessing as mp

from .train.lightning import VideoPredictionLightning
from .train.tensorboard import start_tensorboard

from .arch.convlstm import ConvLSTMSeq2Seq as ConvLSTM
from .arch.convlstm_ref import EncoderDecoderConvLSTM as ConvLSTM_REF

if __name__ == "__main__":

    batch_size = 4

    model = ConvLSTM(1, 64, 2)
    # model = ConvLSTM_REF(64, 1)

    opts = {
        # 'device': 'gpu',
        'device': 'cpu',
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'epochs': 300,
    }
    lightning = VideoPredictionLightning(model, opts)

    mp.set_start_method('spawn')
    p1 = mp.Process(target=lightning.fit)
    p2 = mp.Process(target=start_tensorboard)
    p2.start()
    p1.start()
    p1.join()
    p2.join()
