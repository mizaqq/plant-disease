import lightning as L
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torchvision.ops import box_iou
from src.preprocessing.dataloader import Dataloader


class LightningModule(L.LightningModule):
    def __init__(
        self,
        model: torch,
        optimizer: torch,
        epochs: int = 10,
        metrics: list = [],
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


class FasterCNNLightning(LightningModule):
    def __init__(
        self,
        model: torch,
        optimizer: torch,
        epochs: int = 10,
        metrics: list = [],
    ) -> None:
        super().__init__(model, optimizer, epochs, metrics)
        self.model.roi_heads.score_thresh = 0.01

    def training_step(self, batch, batch_idx):
        if len(batch[0]) == 0:
            return None
        else:
            x, y = batch
        loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values())
        return loss

    def test_step(self, batch, batch_idx):
        if len(batch[0]) == 0:
            return None
        else:
            images, labels = batch
        self.model.eval()
        preds = self.model(images)
        calculate_preds = FasterCNNLightning.calculate_preds(preds, labels)
        self.log('IoU', calculate_preds[0])
        self.log('MSE', calculate_preds[1])

    @staticmethod
    def compute_iou(pred_boxes, gt_boxes):
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return torch.tensor(0.0)
        return box_iou(pred_boxes, gt_boxes)

    @staticmethod
    def calculate_preds(preds, labels):
        iou_scores = []
        mse_losses = []

        for pred, target in zip(preds, labels):
            pred_boxes = pred['boxes'].detach().cpu()
            pred_scores = pred['scores'].detach().cpu()

            gt_boxes = target['boxes'].detach().cpu()

            iou = FasterCNNLightning.compute_iou(pred_boxes, gt_boxes)
            iou_scores.append(iou.mean().item())

            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                best_pred_idx = pred_scores.argmax()
                best_pred_box = pred_boxes[best_pred_idx].unsqueeze(0)

                mse_loss = F.mse_loss(best_pred_box.float(), gt_boxes.float())
                mse_losses.append(mse_loss.item())

        avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
        avg_mse = sum(mse_losses) / len(mse_losses) if mse_losses else 0
        return avg_iou, avg_mse
