import timm
import torch.nn as nn


class ViT:
    def __init__(self, num_classes=1000, freeze_layers_except_last=False, layers_to_freeze=None):
        self.model_name = 'vit_base_patch16_224'
        self.num_classes = num_classes
        self.freeze_layers_except_last = freeze_layers_except_last
        self.layers_to_freeze = layers_to_freeze if layers_to_freeze is not None else []
        self.model = timm.create_model(self.model_name, pretrained=True)
        
        if self.freeze_layers_except_last:
            self.freeze_model_layers()
            self.set_last_layer_trainable()
        elif self.layers_to_freeze:
            self.freeze_specific_layers()

        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.frozen_params = self.total_params - self.trainable_params

    def freeze_model_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def set_last_layer_trainable(self):
        self.model.head = nn.Linear(self.model.head.in_features, self.num_classes)
        for param in self.model.head.parameters():
            param.requires_grad = True

    def freeze_specific_layers(self):
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in self.layers_to_freeze):
                param.requires_grad = False

    def get_params_info(self):
        params_info = {
            'total_params': self.total_params,
            'trainable_params': self.trainable_params,
            'frozen_params': self.frozen_params,
            'layers': []
        }
        
        for name, param in self.model.named_parameters():
            params_info['layers'].append({
                'name': name,
                'requires_grad': param.requires_grad,
                'num_params': param.numel()
            })
        
        return params_info