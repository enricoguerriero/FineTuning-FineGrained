import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import torch
import json
import torch.nn as nn
import wandb
import torch.optim as optim
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

def save_model_info(model_name, train_time, test_accuracy):
    """
    Save model information to info.txt file.

    Args:
    model (tf.keras.Model): The trained model.
    history (tf.keras.callbacks.History): The history object returned by model.fit().
    filename (str): The name of the file to save the information.
    """
    # Collect model information
    model_info = {
        "Model_name": model_name,
        "train_time": train_time,
        "test_accuracy": test_accuracy
    }

    # Open file in write mode and save information
    with open("info.txt", 'a') as file:
        json.dump(model_info, file, indent=4)
        file.write("\n")

def save_model(model, file_name):
    # Define the directory
    dir_name = "trained_models/"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Save the model's state dictionary to the specified file
    file_path = os.path.join(dir_name, file_name)
    torch.save(model.state_dict(), file_path)


def get_data_loaders(data_dir, batch_size=32,
                     resize=(256, 256), crop=(224, 224),
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def get_data_loaders_2(data_dir, batch_size=32,
                     resize=(256, 256), crop=(224, 224),
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def get_data_loaders_3(data_dir, batch_size=32, resize=(256, 256), crop=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    train_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(crop),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    trainset = datasets.OxfordIIITPet(root=data_dir, split = 'trainval',
                                             download=True, transform=train_transform)
    train_size = int(0.8 * len(trainset))  # 80% for training
    val_size = len(trainset) - train_size  # 20% for validation
    train_set, val_set = random_split(trainset, [train_size, val_size])

    test_set = datasets.OxfordIIITPet(root=data_dir, split = 'test',
                                            download=True, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler_gamma, scheduler_step_size, 
                num_epochs, device, patience, model_name, file_name, dataset_name, unfreeze=False):

    best_val_loss = float('inf')
    best_val_acc = 0.0
    counter = 0
    
    if scheduler_step_size is not None and scheduler_gamma is not None:
        scheduler_bool = True
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    else:
        scheduler_bool = False

    for epoch in range(num_epochs):
        e_model_name = model_name + '_epoch_' + str(epoch+1) + 'data' + dataset_name + '.pt'
        print("\n", '-'*10)
        if scheduler_bool:
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}, Current Learning Rate: {current_lr}')     
        e_start = time.time()

        if unfreeze:
            if epoch < 13:
                model.unfreeze_layer(-(epoch + 1))
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
        
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.sampler)
        train_acc = evaluate_model(model, train_loader, None, device)
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')
        print('Epoch time: ', time.time() - e_start)
        
        wandb.log({
            "epoch":epoch+1,
            "train/loss":epoch_loss,
            "train/accuracy":train_acc,
            "val/loss":val_loss,
            "val/accuracy":val_acc
        })

        # Check for best validation accuracy and save model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/' + e_model_name)
            print(f'Best model saved with val accuracy: {best_val_acc:.4f}')
        
        # Check for best validation loss and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Validation loss did not improve for {patience} epochs. Early stopping...')
                break
        
        if scheduler_bool:
            scheduler.step()

        # save as a checkpoint
        save_model(model, file_name)
        
    print('Training complete. Best validation accuracy: {:.4f}'.format(best_val_acc))

    return model

def evaluate_model(model, data_loader, criterion=None, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    if criterion:
        avg_loss = running_loss / len(data_loader.dataset)
        return avg_loss, accuracy
    return accuracy


def load_exam_test(dir_name, resize, crop, mean, std):

    class TestDataset(Dataset):
        def __init__(self, test_dir, transform=None):
            self.test_dir = test_dir
            self.image_files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
            self.transform = transform

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_name = os.path.join(self.test_dir, self.image_files[idx])
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            return image, self.image_files[idx]
    
    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_dataset = TestDataset("/home/disi/ml/datasets/comp/test", transform = test_transform)

    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    return test_loader