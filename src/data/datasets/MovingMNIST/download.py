
import os
from torchvision import datasets, transforms

if __name__ == "__main__":
    datasets.MNIST(
        os.path.dirname(os.path.abspath(__file__)),
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(64),
             transforms.ToTensor()]))
