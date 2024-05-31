import torch
from torch import nn, optim
from torchvision import models

class CustomSqueezeNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(CustomSqueezeNet, self).__init__()
        self.squeezenet = models.squeezenet1_1(pretrained=True)
        
        # Freeze all layers initially
        for param in self.squeezenet.parameters():
            param.requires_grad = False
        
        # Add dropout layers
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(len(self.squeezenet.features))])
        
        # Replace the classifier with a new one
        self.squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.num_classes = num_classes

    def forward(self, x):
        for i, layer in enumerate(self.squeezenet.features):
            x = layer(x)
            if self._has_frozen_parameters(layer):
                x = self.dropout_layers[i](x)
        
        x = self.squeezenet.classifier(x)
        x = x.view(x.size(0), self.num_classes)
        return x
    
    def _has_frozen_parameters(self, layer):
        return any(param.requires_grad == False for param in layer.parameters())
    
    def unfreeze_layer(self, index):
        if index < 14:
            layer = self.squeezenet.features[index]
            for param in layer.parameters():
                param.requires_grad = True