import lightning as L
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from preprocessing.dataloader import Dataloader


class LightningModule(L.LightningModule):
    def __init__(
        self,
        model: torch,
        optimizer: torch,
        epochs: int = 10,
        metrics: list = ['accuracy', 'precision', 'recall', 'f1'],
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.metrics = metrics

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.optimizer
        return optimizer

    def train_model_lightning(self, data_loader: Dataloader) -> tuple[torch.nn.Module, dict]:
        trainer = L.Trainer(max_epochs=self.epochs)
        trainer.fit(self, data_loader.train_loader)
        test_result = trainer.test(self, dataloaders=data_loader.test_loader, verbose=False)
        return self.model, test_result

    @staticmethod
    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        images, labels = batch
        preds = self.model(images).argmax(dim=-1)
        if 'accuracy' in self.metrics:
            self.log('Accuracy', accuracy_score(labels, preds))
        if 'precision' in self.metrics:
            self.log('Precision', precision_score(labels, preds, average="weighted"))
        if 'recall' in self.metrics:
            self.log('Recall', recall_score(labels, preds, average="weighted"))
        if 'f1' in self.metrics:
            self.log('F1', f1_score(labels, preds, average="weighted"))
        if 'classification_report' in self.metrics:
            self.log('classification_report', classification_report(labels, preds))
