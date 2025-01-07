import torch

from src.model.cnn import Convolutional
from src.model.train import Model
from src.preprocessing.dataloader import Dataloader


def main(lr: float = 0.001, momentum: float = 0.9) -> None:
    dataloader = Dataloader(workers=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Convolutional()
    model_instance = Model(model, device, dataloader, optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum))
    model_instance.train_model()
    model_instance.test_model()


if __name__ == "__main__":
    main()
