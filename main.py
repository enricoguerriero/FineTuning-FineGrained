import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import train_model, evaluate_model
from utils.utils import get_data_loaders, save_model_info
import time
import os


# Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Get parameters
model_name = config['model_name']
data_dir = config['data_dir']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
dropout_rate = config['dropout_rate']
learning_rate = config['learning_rate']
momentum = config['momentum']
weight_decay = config['weight_decay']
criteria = config['criteria']
optimz = config['optimizer']
scheduler_step_size = config['scheduler_step_size']
scheduler_gamma = config['scheduler_gamma']
patience = config['patience']
resize = config['resize']
crop_size = config['crop_size']
mean = config['mean']
std = config['std']
freeze_layers_except_last = config['freeze_layers_except_last']
layers_to_freeze = config['layers_to_freeze']

# Some order and other variables
num_classes = len(os.listdir(data_dir + '/train'))
if model_name == 'resnet':
    from models.SENet import SEResNet50
    model = SEResNet50(num_classes=num_classes, freeze_layers_except_last = freeze_layers_except_last, layers_to_freeze = layers_to_freeze)
elif model_name == 'vit':
    from models.ViT import ViT
    model = ViT(num_classes=num_classes, freeze_layers_except_last = freeze_layers_except_last, layers_to_freeze = layers_to_freeze)
else:
    raise ValueError("Model not found")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if criteria == 'cross_entropy':
    criterion = nn.CrossEntropyLoss()
else:
    raise ValueError("Criterion not found")
if optimz == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
else:
    raise ValueError("Optimizer not found")
file_name = f'{model_name}_{data_dir}_{num_epochs}e_bs{batch_size}_lr{learning_rate}_dr{dropout_rate}_c{criteria}_o{optimz}_sg{scheduler_gamma}_sss{scheduler_step_size}_.pth'


print("Parameters loaded")

print("Model info:\n", model.get_params_info())

# Load data
train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size, resize, crop_size, mean, std)

print("Data loaded")

start = time.time()
print("\nStart training!\n")

# Train model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler_gamma, scheduler_step_size, 
            dropout_rate, num_epochs, device, patience, model_name, file_name)

train_time = time.time() - start
print(f"\nEnd training!\nTraining time: {train_time} seconds")

# Evaluate model
accuracy = evaluate_model(model, test_loader, device)

print('Test Accuracy: {:.4f}'.format(accuracy))

# Save model
torch.save(model.state_dict(), "trained_models/" + file_name)

# Save model info
save_model_info(model_name = model_name, train_time = train_time, test_accuracy = accuracy)