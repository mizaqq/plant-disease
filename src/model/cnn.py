import hydra
import torch
import torch.nn.functional as F


class Convolutional(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super(Convolutional, self).__init__()
        self.layers = torch.nn.ModuleDict({k: hydra.utils.instantiate(v) for k, v in cfg.models.params.layers.items()})
        self.pool_indices = cfg.models.params.get("pool_indices", [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, (_, layer) in enumerate(self.layers.items()):
            if isinstance(layer, torch.nn.Conv2d):
                x = F.relu(layer(x))
                if idx in self.pool_indices:
                    x = F.max_pool2d(x, 2)
            elif isinstance(layer, torch.nn.Linear):
                if x.dim() > 2:
                    x = x.view(-1, self.num_flat_features(x))
                x = F.relu(layer(x))
            elif isinstance(layer, torch.nn.Dropout):
                x = layer(x)
        return F.softmax(x, dim=1)

    @staticmethod
    def num_flat_features(x: torch.Tensor) -> int:
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
