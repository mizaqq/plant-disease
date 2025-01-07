import torch.nn.functional as F
import lightning as L
import torch


class LightningModule(L.LightningModule):
    def __init__(self, model, optimizer, epochs=10) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs

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
        acc = (labels == preds).float().mean()
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

    def train_model_lightning(self, data_loader):
        trainer = L.Trainer(max_epochs=self.epochs)
        trainer.fit(self, data_loader.train_loader)
        test_result = trainer.test(self, dataloaders=data_loader.test_loader, verbose=False)
        result = {"test": test_result[0]["test_acc"]}
        return self.model, result
