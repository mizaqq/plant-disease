import torch.nn.functional as F
import lightning as L
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


class LightningModule(L.LightningModule):
    def __init__(self, model, optimizer, epochs=10, metrics: list = ['accuracy', 'precision', 'recall', 'f1']) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.metrics = metrics

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx) -> None:
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

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

    def train_model_lightning(self, data_loader):
        trainer = L.Trainer(max_epochs=self.epochs)
        trainer.fit(self, data_loader.train_loader)
        test_result = trainer.test(self, dataloaders=data_loader.test_loader, verbose=False)
        return self.model, test_result
