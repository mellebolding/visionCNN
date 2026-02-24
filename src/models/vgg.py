"""VGG models with configurable normalization and Global Average Pooling."""
import math
import torch.nn as nn
from .norms import get_norm_layer, NoNorm, WSConv2d


# VGG-11 config: 8 conv layers
VGG11_CFG = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
# VGG-16 config: 13 conv layers
VGG16_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VGG(nn.Module):
    """VGG network with configurable normalization and GAP head.

    Uses AdaptiveAvgPool2d instead of the original massive FC layers,
    making the model much more parameter-efficient.
    """

    def __init__(self, layer_cfg, num_classes=100, norm_layer=nn.BatchNorm2d,
                 use_ws=False):
        super().__init__()
        self._use_ws = use_ws
        self._norm_layer = norm_layer
        self.features = self._make_layers(layer_cfg, norm_layer)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layers(self, cfg, norm_layer):
        layers = []
        in_channels = 3
        conv_cls = WSConv2d if self._use_ws else nn.Conv2d
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    conv_cls(in_channels, v, kernel_size=3, padding=1),
                    norm_layer(v),
                    nn.ReLU(inplace=True),
                ])
                in_channels = v
        return nn.Sequential(*layers)

    def _init_weights(self):
        num_convs = sum(1 for m in self.modules() if isinstance(m, (nn.Conv2d, WSConv2d)))
        is_nonorm = isinstance(self._norm_layer(1), NoNorm)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, WSConv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                # Scale weights for NoNorm to prevent activation explosion
                if is_nonorm:
                    m.weight.data.mul_(1.0 / math.sqrt(num_convs))
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def vgg_small(num_classes=100, norm_layer="batchnorm", **kwargs):
    """VGG-11 with GAP head."""
    use_ws = norm_layer.lower() == "nonorm_ws"
    return VGG(VGG11_CFG, num_classes=num_classes,
               norm_layer=get_norm_layer(norm_layer), use_ws=use_ws)


def vgg_medium(num_classes=100, norm_layer="batchnorm", **kwargs):
    """VGG-16 with GAP head."""
    use_ws = norm_layer.lower() == "nonorm_ws"
    return VGG(VGG16_CFG, num_classes=num_classes,
               norm_layer=get_norm_layer(norm_layer), use_ws=use_ws)
