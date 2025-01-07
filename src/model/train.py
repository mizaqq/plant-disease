import torch
from tqdm import tqdm

from src.preprocessing.dataloader import Dataloader


class Model:
    def __init__(self, model: torch.nn.Module, device: torch.device, data_loader: Dataloader, optimizer: torch, loss_fn:torch=torch.nn.CrossEntropyLoss()) -> None:
        self.model = model
        self.device = device
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    def train_model(self, epochs: int = 10) -> None:
        self.model.to(self.device)

        running_loss = 0.0
        count = 1
        for epoch in tqdm(range(epochs), total=epochs):
            count = 0
            running_loss = 0.0
            for i, data in enumerate(self.data_loader.train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                count += 1
            print(f'Running loss for epoch {epoch+1}: {running_loss/(count)}')
        print('Finished Training')

    def test_model(self) -> None:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.data_loader.test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the test images: {100 * correct / total}%')
