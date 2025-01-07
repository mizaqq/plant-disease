import cv2
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

from src.preprocessing.dataloader import Dataloader


class Model:
    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: Dataloader,
        optimizer: torch,
        loss_fn: torch = torch.nn.CrossEntropyLoss(),
        device: torch.device = None,
    ) -> None:
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
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
            print(f"Running loss for epoch {epoch+1}: {running_loss/(count)}")
        print("Finished Training")

    def test_model(self) -> tuple[list, list]:
        labels_list = []
        predicted_labels = []
        with torch.no_grad():
            for data in self.data_loader.test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                labels_list.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())
        return labels_list, predicted_labels

    @staticmethod
    def calculate_metrics(
        labels_list: list, predicted_labels: list, metrics: list = ['accuracy', 'precision', 'recall', 'f1']
    ) -> None:
        if 'accuracy' in metrics:
            print(f'Accuracy: {accuracy_score(labels_list, predicted_labels)}')
        if 'precision' in metrics:
            print(f'Precision: {precision_score(labels_list, predicted_labels, average="weighted")}')
        if 'recall' in metrics:
            print(f'Recall: {recall_score(labels_list, predicted_labels, average="weighted")}')
        if 'f1' in metrics:
            print(f'F1 Score: {f1_score(labels_list, predicted_labels, average="weighted")}')
        if 'classification_report' in metrics:
            print(f'Classification Report: {classification_report(labels_list, predicted_labels, average="weighted")}')

    @staticmethod
    def show_result(fig: torch.Tensor, label: list, predicted: list) -> None:
        fig = fig.permute(1, 2, 0).cpu().numpy()
        print(f"Actual: {label}, Predicted: {predicted}")
        cv2.imshow(f"image: {predicted},label: {label}", fig)
        cv2.waitKey(0)
