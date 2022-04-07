
import torch

import multiprocessing as mp

from .train.lightning import VideoPredictionLightning
from .train.lightning import SequencePredictionLightning
from .train.tensorboard import start_tensorboard

from .arch.lstm import LSTMSeq2Seq as LSTM
from .arch.convlstm import ConvLSTMSeq2Seq as ConvLSTM
from .arch.convlstm_ref import EncoderDecoderConvLSTM as ConvLSTM_REF

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Video Prediction
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Sequence Prediction
    # -------------------------------------------------------------------------

    # batch_size = 20
    # model = LSTM(1, 64, 4)
    # opts = {
    #     'device': 'gpu',
    #     # 'device': 'cpu',
    #     'batch_size': batch_size,
    #     'learning_rate': 0.001,
    #     'epochs': 300,
    # }
    # lightning = SequencePredictionLightning(model, opts)

    # -------------------------------------------------------------------------
    # Start Tensorboard and Training
    # -------------------------------------------------------------------------

    mp.set_start_method('spawn')
    p1 = mp.Process(target=lightning.fit)
    p2 = mp.Process(target=start_tensorboard)
    p2.start()
    p1.start()
    p1.join()
    p2.join()
