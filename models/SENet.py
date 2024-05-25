import torch.nn as nn
from torchvision import models

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEResNet50(nn.Module):
    def __init__(self, num_classes=1000, freeze_layers_except_last=False, layers_to_freeze=None):
        super(SEResNet50, self).__init__()
        self.num_classes = num_classes
        self.freeze_layers_except_last = freeze_layers_except_last
        self.layers_to_freeze = layers_to_freeze if layers_to_freeze is not None else []
        
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')  # Load pre-trained ResNet-50
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)  # Modify the fully connected layer to match the number of classes

        print(freeze_layers_except_last)
        if self.freeze_layers_except_last:
            self.freeze_model_layers()
            self.set_last_layer_trainable()
        elif self.layers_to_freeze:
            self.freeze_specific_layers()
        else:
            print("no freeze")

        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.frozen_params = self.total_params - self.trainable_params

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1, reduction=8):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, reduction))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=reduction))

        return nn.Sequential(*layers)

    def freeze_model_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def set_last_layer_trainable(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def freeze_specific_layers(self):
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in self.layers_to_freeze):
                param.requires_grad = False

    def get_params_info(self):
        params_info = {
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
            'frozen_params': self.frozen_params,
            # 'layers': []
        }
        
        # for name, param in self.named_parameters():
        #     params_info['layers'].append({
        #         'name': name,
        #         'requires_grad': param.requires_grad,
        #         'num_params': param.numel()
        #     })
        
        return params_info

