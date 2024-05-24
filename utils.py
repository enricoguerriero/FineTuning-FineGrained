import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import torch


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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler_gamma, scheduler_step_size, 
            dropout_rate, num_epochs, device, patience, model_name):

    best_val_loss = float('inf')
    counter = 0
    if scheduler_step_size is not None and scheduler_gamma is not None:
        scheduler_bool = True
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    else:
        scheduler_bool = False
    if dropout_rate is not None:
        dropout_bool = True
        dropout = torch.nn.Dropout(dropout_rate)
    else:
        dropout_bool = False

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        print("\n", '-'*10)
        if scheduler_bool:
            current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}/{num_epochs}, Current Learning Rate: {current_lr}')     
        e_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if dropout_bool:
                outputs = dropout(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.sampler)
        train_acc = evaluate_model(model, train_loader, None, device)
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        # # Log the loss and accuracy to TensorBoard
        # writer.add_scalar('Loss/Train', epoch_loss, epoch)
        # writer.add_scalar('Loss/Validation', val_loss, epoch)
        # writer.add_scalar('Accuracy/Train', train_acc, epoch)
        # writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}')
        print('Epoch time: ', time.time() - e_start)
        
        # Check for best validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_name)
        if scheduler_bool:
            scheduler.step()
        print('-'*10, "\n")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Validation loss did not improve for {patience} epochs. Early stopping...')
                break
    
    # writer.close()
    print('Training complete. Best validation accuracy: {:.4f}'.format(best_val_acc))

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
