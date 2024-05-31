from transformers import ViTForImageClassification
import torch.nn as nn

class ViTFineTuner(nn.Module):
    def __init__(self, num_classes, freeze_layers=True, num_frozen_blocks=6):
        super(ViTFineTuner, self).__init__()
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        
        # Initialize the new classifier layer
        nn.init.xavier_uniform_(self.model.classifier.weight)
        if self.model.classifier.bias is not None:
            nn.init.zeros_(self.model.classifier.bias)
        
        if freeze_layers:
            # Freeze the embedding layer and the first few transformer blocks
            for param in self.model.vit.embeddings.parameters():
                param.requires_grad = False
            for param in self.model.vit.encoder.layer[:num_frozen_blocks].parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x).logits