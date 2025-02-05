import hydra
import torch
from omegaconf import DictConfig


class PretrainedModel:
    def __init__(self, cfg: DictConfig) -> None:
        self.model = hydra.utils.instantiate(cfg.models.params.model)
        self.freeze_layers(cfg.models.params.frozen_layers)
        self._setup_classifier(cfg.models.params.classifier, cfg.models.params.num_classes)
        self.unfreeze_layers(cfg.models.params.trainable_layers)

    def _setup_classifier(self, classifier, num_classes: int = 10):
        if hasattr(self.model, 'roi_heads'):
            setattr(self.model, 'roi_heads.box_predictor', classifier)
        elif hasattr(self.model, classifier):
            setattr(self.model, classifier, torch.nn.Linear(getattr(self.model, classifier).in_features, num_classes))

    def freeze_layers(self, layer: int) -> None:
        for idx, param in enumerate(self.model.parameters()):
            if idx < layer:
                param.requires_grad = False

    def unfreeze_layers(self, trainable_layers: list[str]) -> None:
        if self.model.__str__().startswith('FasterRCNN'):
            for layer_name in trainable_layers:
                if hasattr(self.model.backbone.body, layer_name):
                    for param in getattr(self.model.backbone.body, layer_name).parameters():
                        param.requires_grad = True
        else:
            for layer_name in trainable_layers:
                if hasattr(self.model, layer_name):
                    for param in getattr(self.model, layer_name).parameters():
                        param.requires_grad = True

    @staticmethod
    def get_model(cfg: DictConfig) -> torch.nn.Module:
        return PretrainedModel(cfg).model
