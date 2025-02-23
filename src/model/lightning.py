from typing import Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torchvision.ops import box_iou

from src.preprocessing.dataloader import Dataloader
from src.utils.mlflow import MLFlowRunManager
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection import _utils as det_utils


class LightningModule(L.LightningModule):
    def __init__(
        self,
        model: torch,
        optimizer: torch,
        mlflow: Optional[MLFlowRunManager] = None,
        epochs: int = 10,
        metrics: list = [],
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.metrics = metrics
        self.mlflow = mlflow

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.optimizer
        return optimizer

    def train_model_lightning(
        self, train_loader: Dataloader, validation_loader: Dataloader, test_loader: Dataloader = None
    ) -> tuple[torch.nn.Module, dict]:
        trainer = L.Trainer(max_epochs=self.epochs, logger=self.mlflow)
        trainer.fit(self, train_loader, val_dataloaders=validation_loader)
        test_result = {}
        if test_loader is not None:
            test_result = trainer.test(self, dataloaders=test_loader, verbose=False)
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
        mlflow: Optional[MLFlowRunManager] = None,
        epochs: int = 10,
    ) -> None:
        super().__init__(model, optimizer, mlflow, epochs, [])
        self.model.roi_heads.score_thresh = 0.01

    def training_step(self, batch: torch.tensor, batch_idx: int) -> Optional[float]:
        if len(batch[0]) == 0:
            return None
        else:
            x, y = batch
        loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.tensor, batch_idx: int) -> None:
        if len(batch[0]) == 0:
            return None
        else:
            x, y = batch
        transformed_images = self.model.transform(x)[0]
        features = self.model.backbone(transformed_images.tensors)
        self.model.rpn.train()
        self.model.roi_heads.train()
        proposals, proposals_losses = self.model.rpn(transformed_images, features, y)
        detections, detector_losses = self.model.roi_heads(features, proposals, transformed_images.image_sizes, y)
        self.model.rpn.eval()
        self.model.roi_heads.eval()
        loss_dict = {}
        loss_dict.update(proposals_losses)
        loss_dict.update(detector_losses)
        loss = sum(loss for loss in loss_dict.values())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: torch.tensor, batch_idx: int) -> None:
        if len(batch[0]) == 0:
            return None
        else:
            images, labels = batch
        self.model.eval()
        preds = self.model(images)
        calculate_preds = FasterCNNLightning.calculate_preds(preds, labels)
        self.log('IoU', calculate_preds[0])
        self.log('MSE', calculate_preds[1])
        report_dict = classification_report(*self.clean_labels(preds, labels), output_dict=True)
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    self.log(f'{label}_{metric}', value)
            else:
                self.log(label, metrics)

    @staticmethod
    def compute_iou(pred_boxes: torch.tensor, gt_boxes: torch.tensor) -> torch.tensor:
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return torch.tensor(0.0)
        return box_iou(pred_boxes, gt_boxes)

    @staticmethod
    def calculate_preds(preds: list[dict], labels: list[dict]) -> tuple[float, float]:
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

    @staticmethod
    def clean_labels(preds: list[dict], labels: list[dict]) -> tuple[list, list]:
        best_idx = [np.argmax(pred['scores'].detach().cpu()) if len(pred['scores']) > 0 else 0 for pred in preds]
        pred_labels = [pred['labels'][idx].detach().cpu() for pred, idx in zip(preds, best_idx)]
        origin_label = [label['labels'].detach().cpu() for label in labels]
        p_l = []
        o_l = []
        for p, o in zip(pred_labels, origin_label):
            if p.numpy().size > 1:
                p_l.append(p[0])
            else:
                p_l.append(p)
            if o.numpy().size > 1:
                o_l.append(o[0])
            else:
                o_l.append(o)
        OL_clean = [int(t.item()) for t in o_l]
        PL_clean = [int(t.item()) for t in p_l]
        return OL_clean, PL_clean
