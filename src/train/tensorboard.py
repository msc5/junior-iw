
import os
import time
from tensorboard import program


def start_tensorboard():

    program.logger.setLevel('FATAL')
    path = os.path.join(os.getcwd(), 'tensorboard')

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', path])
    url = tb.launch()

    print('-' * 80)
    print(f'Starting Tensorboard at {url}')
    print('-' * 80)

    while True:
        time.sleep(2)


if __name__ == "__main__":

    start_tensorboard()
