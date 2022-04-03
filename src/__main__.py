
# from multiprocessing import Process
import multiprocessing as mp

from .arch.convlstm import ConvLSTMSeq2Seq
from .train.lightning import VideoPredictionLightning
from .train.tensorboard import start_tensorboard

from .arch.convlstm_ref import EncoderDecoderConvLSTM

if __name__ == "__main__":
    mp.set_start_method('spawn')

    batch_size = 10

    model = ConvLSTMSeq2Seq(64, (1, 64, 64), 2)
    # model = EncoderDecoderConvLSTM(64, 1)
    opts = {
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'epochs': 300,
    }
    lightning = VideoPredictionLightning(model, opts)

    p1 = mp.Process(target=lightning.fit)
    p2 = mp.Process(target=start_tensorboard)
    p2.start()
    p1.start()
    p1.join()
    p2.join()
