import torch
from tqdm import tqdm

from src.preprocessing.dataloader import Dataloader
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

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
    def calculate_metrics(labels_list, predicted_labels, metrics=['accuracy', 'precision', 'recall', 'f1']):
        for metric in metrics:
            if metric == 'accuracy':
                print(f'Accuracy: {accuracy_score(labels_list, predicted_labels)}')
            elif metric == 'precision':
                print(f'Precision: {precision_score(labels_list, predicted_labels, average="weighted")}')
            elif metric == 'recall':
                print(f'Recall: {recall_score(labels_list, predicted_labels, average="weighted")}')
            elif metric == 'f1':
                print(f'F1 Score: {f1_score(labels_list, predicted_labels, average="weighted")}')
            elif metric == 'classification_report':
                print(f'Classification Report: {classification_report(labels_list, predicted_labels, average="weighted")}')
        

