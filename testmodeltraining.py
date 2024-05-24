# Questo ancora non va
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import train_model, evaluate_model


class TestModelTraining(unittest.TestCase):

    def setUp(self):
        # Dummy dataset
        inputs = torch.randn(100, 3, 224, 224)  # 100 samples of 3x224x224 images
        labels = torch.randint(0, 10, (100,))   # 100 labels for 10 classes
        dataset = TensorDataset(inputs, labels)
        
        self.train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Simple model for testing
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 224 * 224, 10)
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

    def test_training(self):
        train_model(self.model, self.train_loader, self.val_loader, self.criterion, self.optimizer, self.scheduler, num_epochs=2, device='cuda')
        self.assertTrue(True)  # If the model trains without error, the test passes

    def test_evaluation(self):
        accuracy = evaluate_model(self.model, self.val_loader, device='cuda')
        self.assertIsInstance(accuracy, float)  # Check if the returned accuracy is a float
