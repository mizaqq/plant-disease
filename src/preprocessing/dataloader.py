from pathlib import Path

import torch
from torchvision import datasets, transforms


class Dataloader:
    def __init__(
        self,
        data_path: Path = Path(__file__).parent.parent.parent.joinpath("data").joinpath("leaves"),
        batch_size: int = 32,
        size: int = 255,
        crop: int = 224,
        split: float = 0.8,
        workers: int = 15,
    ) -> None:
        self.data_path = data_path
        self.batch_size = batch_size
        self.transformer = self.get_transformer(size, crop)
        self.dataset = self.get_dataset()
        self.train_data, self.test_data = self.split_data(split)
        self.workers = workers
        self.train_loader = self.get_loader(self.train_data, shuffle=True)
        self.test_loader = self.get_loader(self.test_data, shuffle=False)

    def get_transformer(self, size: int, crop: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
            ]
        )

    def get_dataset(self) -> datasets.ImageFolder:
        return datasets.ImageFolder(self.data_path, transform=self.transformer)

    def get_loader(self, dataset: torch.utils.data.Dataset, shuffle: bool) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.workers
        )

    def split_data(self, train_split: float) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        train_size = int(train_split * len(self.dataset))
        test_size = len(self.dataset) - train_size
        return torch.utils.data.random_split(
            self.dataset,
            [train_size, test_size],
        )
