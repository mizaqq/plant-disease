from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms
from typing import Any, Tuple


class FRCNNImageFolder(datasets.ImageFolder):
    def __init__(self, root: Path, label_dir: Path = None, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.label_dir = label_dir

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, self.find_annotation(Path(path))

    def find_annotation(self, path: Path) -> Path:
        targets = []
        label_path = self.label_dir.joinpath(path.stem).with_suffix(".txt")
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    values = [float(x) for x in line.strip().split()]
                    label, cx, cy, w, h = values
                    x_min = cx - (w / 2)
                    y_min = cy - (h / 2)
                    x_max = cx + (w / 2)
                    y_max = cy + (h / 2)
                    targets.append([x_min, y_min, x_max, y_max, label])

        if targets:
            targets = torch.tensor(targets, dtype=torch.float32)
            boxes = targets[:, :4]
            labels = targets[:, 4].long()
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        return {"boxes": boxes, "labels": labels}


class Dataloader:
    def __init__(
        self,
        data_path: Path = Path(__file__).parent.parent.parent.joinpath("data").joinpath("leaves"),
        annotations_path: Path = None,
        batch_size: int = 32,
        size: int = 255,
        crop: int = 224,
        split: float = 0.8,
        workers: int = 15,
    ) -> None:
        self.data_path = data_path
        self.annotations_path = annotations_path
        self.batch_size = batch_size
        self.transformer = self.get_transformer(size, crop)
        self.dataset = self.get_dataset_with_annotations() if annotations_path is not None else self.get_dataset()
        self.train_data, self.test_data = self.split_data(split)
        self.workers = workers
        self.train_loader = self.get_loader(self.train_data, shuffle=True)
        self.test_loader = self.get_loader(self.test_data, shuffle=False)

    @staticmethod
    def get_transformer(size: int, crop: int) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
            ]
        )

    def get_dataset(self) -> datasets.ImageFolder:
        return datasets.ImageFolder(self.data_path, transform=self.transformer)

    def get_loader(
        self, dataset: torch.utils.data.Dataset, shuffle: bool, collate_fn=None
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.workers, collate_fn=collate_fn
        )

    def split_data(
        self, train_split: float, seed: int = 42
    ) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        train_size = int(train_split * len(self.dataset))
        test_size = len(self.dataset) - train_size
        return torch.utils.data.random_split(self.dataset, [train_size, test_size], torch.Generator().manual_seed(seed))

    def get_dataset_with_annotations(self) -> datasets.ImageFolder:
        return FRCNNImageFolder(self.data_path, self.annotations_path, transform=self.transformer)

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)  # ✅ Unzips batch into two lists
        return list(images), list(targets)
