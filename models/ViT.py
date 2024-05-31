import timm
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, num_classes=1000):
        super(ViT, self).__init__()
        self.model_name = 'vit_base_patch16_224'
        self.num_classes = num_classes
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=num_classes)

        # Define the last layer for classification
        self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)

    def forward(self, x):
        # Pass input through the pre-trained model
        output, intermediates = self.model.forward_intermediates(x)
        return output, intermediates


