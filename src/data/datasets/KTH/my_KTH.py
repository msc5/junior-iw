
import socket
import os
import numpy as np

from torchvision import datasets, transforms
from torchvision.io import read_video
from torch.utils.data import Dataset

# Check out https://github.com/edouardelasalles/srvp for ready-made
# dataloaders :)


class KTH (Dataset):

    def __init__(self, train, seq_len: int):
        self.path = os.path.join(os.getcwd(), 'src', 'data', 'datasets', 'KTH')
        self.all_path = os.path.join(self.path, 'raw', 'all')
        self.seq_len = seq_len

        self.download()

        print(len(self))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        avi = self.files[index]
        video = read_video(os.path.join(self.all_path, avi))
        return video.permute(0, 3, 1, 2)

    def download(self):
        import wget
        import concurrent.futures
        import zipfile
        from functools import partial

        website = 'http://www.nada.kth.se/cvap/actions/'
        names = [
            'boxing.zip',
            'handclapping.zip',
            'handwaving.zip',
            'jogging.zip',
            'running.zip',
            'walking.zip',
            '00sequences.txt'
        ]

        folder_path = os.path.join(self.path, 'raw')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if not os.path.exists(self.all_path):
            os.makedirs(self.all_path)

        def download_file(name):
            try:
                path = os.path.join(folder_path, name)
                url = f'{website}{name}'
                print(f'Downloading {name} from {url} into {path}')
                wget.download(url, path)
            except Exception as e:
                print(e)

        def unzip_file(name):
            path = os.path.join(folder_path, name)
            with zipfile.ZipFile(path, 'r') as zip:
                zip.extractall(self.all_path)

        to_download = [name for name in names if not os.path.exists(
            os.path.join(self.path, 'raw', name))]
        with concurrent.futures.ThreadPoolExecutor() as exector:
            exector.map(download_file, to_download)
        with concurrent.futures.ThreadPoolExecutor() as exector:
            exector.map(unzip_file, to_download)

        self.files = os.listdir(self.all_path)

    def get_data(self):
        for avi in self.files:
            video, _, _ = read_video(os.path.join(self.all_path, avi))
            data += [video]
        return torch.stack(data)


if __name__ == "__main__":

    dataset = KTH(train=True, seq_len=20)
    # dataset.download()
