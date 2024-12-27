import torch
from tqdm import tqdm

from src.preprocessing.dataloader import Dataloader
from matplotlib import pyplot as plt
import cv2


class Model:
    def __init__(
        self, model: torch.nn.Module, device: torch.device, data_loader: Dataloader
    ) -> None:
        self.model = model
        self.device = device
        self.data_loader = data_loader

    def train_model(
        self, lr: float = 0.001, momentum: float = 0.9, epochs: int = 10
    ) -> None:
        self.model.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        running_loss = 0.0
        count = 1
        for epoch in tqdm(range(epochs), total=epochs):
            count = 0
            running_loss = 0.0
            for i, data in enumerate(self.data_loader.train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                count += 1
            print(f"Running loss for epoch {epoch+1}: {running_loss/(count)}")
        print("Finished Training")

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
                #self.show_result(images[0], labels[0], predicted[0])
        print(f"Accuracy of the network on the test images: {100 * correct / total}%")

    def show_result(self, fig, label, predicted) -> None:
        fig = fig.permute(1, 2, 0).cpu().numpy()
        print(f"Actual: {label}, Predicted: {predicted}")
        cv2.imshow(f"image: {predicted},label: {label}", fig)
        cv2.waitKey(0)
