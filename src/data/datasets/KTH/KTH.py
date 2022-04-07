

from .KTH_dataset import KTH as kth_dataset


def KTH(seq_len: int, train: bool):
    dirpath = os.path.dirname(os.path.abspath(__file__))
    return kth_dataset.make_dataset('src/data/datasets/KTH/raw', 64, 20, True)


if __name__ == "__main__":

    kth = KTH(20, True)
