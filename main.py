import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from utils import train_model, evaluate_model
from utils import get_data_loaders
import time


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

# Some order and other variables
if model_name == 'resnet':
    from SENet import SEResNet50
    model = SEResNet50(num_classes=10)
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

print("Parameters loaded")

# Load data
train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size, resize, crop_size, mean, std)

print("Data loaded")

start = time.time()
print("\nStart training!\n")

# Train model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler_gamma, scheduler_step_size, 
            dropout_rate, num_epochs, device, patience, model_name)

print(f"\nEnd training!\nTraining time: {time.time() - start} seconds")

# Evaluate model
accuracy = evaluate_model(model, test_loader, device)

print('Test Accuracy: {:.4f}'.format(accuracy))

# Save model
file_name = f'{model_name}_{data_dir}_{num_epochs}e_bs{batch_size}_lr{learning_rate}_dr{dropout_rate}_c{criteria}_o{optimz}_sg{scheduler_gamma}_sss{scheduler_step_size}_.pth'
torch.save(model.state_dict(), f'{model_name}_model.pth')