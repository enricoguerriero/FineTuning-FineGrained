import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import train_model, evaluate_model
from utils.utils import get_data_loaders, save_model_info, save_model
import time
import os
import torchvision
import torchvision.models as models


# Load config file
with open('FineTuning-FineGrained/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Get parameters
model_name = str(config['model_name'])
dataset = str(config['data_dir'])
data_dir = 'datasets/' + dataset
batch_size = int(config['batch_size'])
num_epochs = int(config['num_epochs'])
dropout_rate = float(config['dropout_rate'])
learning_rate = float(config['learning_rate'])
momentum = float(config['momentum'])
weight_decay = float(config['weight_decay'])
criteria = str(config['criteria'])
optimz = str(config['optimizer'])
scheduler_step_size = int(config['scheduler_step_size'])
scheduler_gamma = float(config['scheduler_gamma'])
patience = int(config['patience'])
resize = int(config['resize'])
crop_size = int(config['crop_size'])
mean = list(config['mean'])
std = list(config['std'])
#freeze_layers_except_last = bool(config['freeze_layers_except_last'])
# layers_to_freeze = list(config['layers_to_freeze'])

# Some order and other variables
unfreeze = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(os.listdir(data_dir + '/train'))
if model_name == 'seresnet':
    from models.SENet import SEResNet50
    model = SEResNet50(num_classes=num_classes, dropout_prob = dropout_rate).to(device)
elif model_name == 'vit':
    from models.ViT import ViT
    model = ViT(num_classes=num_classes).to(device)
elif model_name == 'efficientnet':
    model = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    # Freeze the first layer
    first_layer = model.features[0]
    for param in first_layer.parameters():
        param.requires_grad = False
    # Modify the classifier for the desired number of output classes
    in_features = model.classifier[1].in_features  # Access the in_features of the second layer of the classifier
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)
elif model_name == 'seresnet2':
    from models.SENet2 import CustomSqueezeNet
    model = CustomSqueezeNet(num_classes).to(device)
    unfreeze = True 
elif model_name == 'vit2':
    from models.ViT2 import ViTFineTuner
    model = ViTFineTuner(num_classes).to(device)
    unfreeze = True
else:
    raise ValueError("Model not found")
if criteria == 'cross_entropy':
    criterion = nn.CrossEntropyLoss()
else:
    raise ValueError("Criterion not found")
if optimz == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimz == 'sgd':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
elif optimz == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
else:
    raise ValueError("Optimizer not found")
file_name = f'{model_name}_{dataset}_{num_epochs}e_bs{batch_size}_lr{learning_rate}_dr{dropout_rate}_c{criteria}_o{optimz}_sg{scheduler_gamma}_sss{scheduler_step_size}.pth'

print("Parameters loaded")

print("\nRun info:")
print("Model name: ", model_name)
print("Dataset: ", dataset)
print("Batch size: ", batch_size)
print("Number of epochs: ", num_epochs)
print("Number of classes: ", num_classes)
print("Learning Rate: ", learning_rate)
print("Dropout Rate: ", dropout_rate)
print("Weight Decay: ", weight_decay)
print("Momentum: ", momentum)
print("Criterion: ", criteria)
print("Optimizer: ", optimz)
print("Scheduler Step Size: ", scheduler_step_size)
print("Scheduler Gamma: ", scheduler_gamma)
print("Patience: ", patience)
print("Resize: ", resize)
print("Crop size: ", crop_size)
print("Mean: ", mean)
print("Std: ", std)
# print("Frozen layers: ", layers_to_freeze)

# print("\nModel info:\n", model.get_params_info())

# Load data
train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size, resize, crop_size, mean, std)

print("\nData loaded")

start = time.time()
print("\nStart training!\n")

# Train model
model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler_gamma, scheduler_step_size, 
            num_epochs, device, patience, model_name, file_name, unfreeze)

train_time = time.time() - start
print(f"\nEnd training!\nTraining time: {train_time} seconds")

# Evaluate model
accuracy = evaluate_model(model, test_loader, criterion = None, device = device)

print('Test Accuracy: {:.4f}'.format(accuracy))

# Save model
save_model(model, file_name)
# Save model info
save_model_info(model_name = file_name, train_time = train_time, test_accuracy = accuracy)
