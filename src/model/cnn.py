import torch
import torch.nn.functional as F
import hydra

class Convolutional(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super(Convolutional, self).__init__()
        layers = {k:hydra.utils.instantiate(v) for k,v in cfg.models.params.layers.items()}
        for key in layers: 
            setattr(self, key, layers[key])
             
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x
    @staticmethod
    def num_flat_features(x: torch.Tensor) -> int:
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
