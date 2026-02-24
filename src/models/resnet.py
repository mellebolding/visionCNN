"""Custom ResNet models with configurable normalization.

Standard ImageNet ResNet architecture (7x7 stem, 4 stages, GAP + FC head)
but with swappable normalization layers for the normalization comparison experiment.
"""
import torch.nn as nn
from .norms import get_norm_layer, NoNorm, WSConv2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d,
                 downsample=None, conv_cls=nn.Conv2d):
        super().__init__()
        self.conv1 = conv_cls(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_cls(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet for ImageNet with configurable normalization."""

    def __init__(self, block, layers, num_classes=100, norm_layer=nn.BatchNorm2d,
                 use_ws=False):
        super().__init__()
        self.in_planes = 64
        self.norm_layer = norm_layer
        self._conv_cls = WSConv2d if use_ws else nn.Conv2d

        # Stem
        self.conv1 = self._conv_cls(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                self._conv_cls(self.in_planes, planes * block.expansion, 1,
                               stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, self.norm_layer, downsample,
                        conv_cls=self._conv_cls)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, norm_layer=self.norm_layer,
                                conv_cls=self._conv_cls))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, WSConv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-init last conv in each residual block for stability without norm
        # (Fixup-style: makes residual branch start near zero)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                if isinstance(m.bn2, NoNorm):
                    nn.init.zeros_(m.conv2.weight)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def resnet_small(num_classes=100, norm_layer="batchnorm", **kwargs):
    """ResNet-18 with configurable normalization."""
    use_ws = norm_layer.lower() == "nonorm_ws"
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                  norm_layer=get_norm_layer(norm_layer), use_ws=use_ws)


def resnet_medium(num_classes=100, norm_layer="batchnorm", **kwargs):
    """ResNet-34 with configurable normalization."""
    use_ws = norm_layer.lower() == "nonorm_ws"
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                  norm_layer=get_norm_layer(norm_layer), use_ws=use_ws)
