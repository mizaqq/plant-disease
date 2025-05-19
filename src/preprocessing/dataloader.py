from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image
from torchvision import datasets, transforms


class Dataloader:
    def __init__(
        self,
        data_path: Path = Path(__file__).parent.parent.parent.joinpath("data").joinpath("leaves"),
        annotations_path: Path = None,
        batch_size: int = 32,
        size: int = 255,
        crop: int = 224,
        train_split: float = 0.75,
        val_split: float = 0.15,
        workers: int = 15,
    ) -> None:
        self.data_path = data_path
        self.annotations_path = annotations_path
        self.batch_size = batch_size
        self.transformer = self.get_transformer(size, crop)
        self.dataset = self.get_dataset_with_annotations() if annotations_path is not None else self.get_dataset()
        self.train_data, self.validation_data, self.test_data = self.split_data(train_split, val_split)
        self.workers = workers
        self.train_loader = self.get_loader(self.train_data, batch_size, workers, shuffle=True)
        self.validation_loader = self.get_loader(self.validation_data, batch_size, workers, shuffle=False)
        self.test_loader = self.get_loader(self.test_data, batch_size, workers, shuffle=False)

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

    @staticmethod
    def get_loader(
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        workers: int,
        shuffle: bool,
        collate_fn: Optional[Callable] = None,
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate_fn
        )

    def split_data(
        self, train_split: float, val_split: float, seed: int = 42
    ) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
        return torch.utils.data.random_split(
            self.dataset, [train_split, val_split, 1 - train_split - val_split], torch.Generator().manual_seed(seed)
        )

    def get_dataset_with_annotations(self) -> datasets.ImageFolder:
        return FRCNNImageFolder(self.data_path, self.annotations_path, transform=self.transformer)

    @staticmethod
    def collate_fn(batch: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        images, targets = [], []
        for img, target in batch:
            if target is None:
                continue
            images.append(img)
            targets.append(target)
        return images, targets


class FRCNNImageFolder(datasets.ImageFolder):
    def __init__(
        self,
        root: Path,
        label_dir: Optional[Path] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.label_dir = label_dir

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        annotation = self.find_annotation(Path(path), target)
        return sample, annotation

    def find_annotation(self, path: Path, target: int) -> Optional[dict]:
        targets = []
        if self.label_dir is not None:
            label_path = self.label_dir.joinpath(path.stem).with_suffix(".txt")
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    values = [float(x) for x in line.strip().split()]
                    _, cx, cy, w, h = values
                    x_min = int((cx - w / 2) * 244)
                    y_min = int((cy - h / 2) * 244)
                    x_max = int((cx + w / 2) * 244)
                    y_max = int((cy + h / 2) * 244)

                    targets.append([x_min, y_min, x_max, y_max, target])

        if targets:
            new_targets = torch.tensor(targets, dtype=torch.float32)
            boxes = new_targets[:, :4]
            labels = new_targets[:, 4].long()
        else:
            return None

        return {"boxes": boxes, "labels": labels}


class LabelStudioDataset(torch.utils.data.Dataset):
    def __init__(self, data: list, transform: transforms = None) -> None:
        self.image_paths = [d['image'] for d in data]
        self.labels = [d['label'] for d in data]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, dict]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        target = torch.tensor(label, dtype=torch.float32)
        boxes = target[:4].view(1, 4)
        labels = target[4].view(1).long()
        return image, {"boxes": boxes, "labels": labels}
