# pip install -q git+https://github.com/huggingface/transformers
# pip install lightning=2.2.1

from transformers import ViTForImageClassification
ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", 
                                          num_labels=10, 
                                          ignore_mismatched_sizes=True
                                          )

#!/usr/bin/python

from torch.nn import functional as F
from torch import optim

from transformers import ViTForImageClassification
import torchmetrics

import lightning as L

class VisionTransformerPretrained(L.LightningModule):
    '''
    Wrapper for the pretrained Vision Transformers
    '''

    def __init__(self, model="google/vit-base-patch16-224", num_classes=1000):

        super().__init__()
        backbone = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", 
                                                             num_labels=10, ignore_mismatched_sizes=True)
        self.backbone = backbone

        # metrics
        self.acc = torchmetrics.Accuracy('multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)

    def step(self, batch):
       '''
       Any step processes batch to return loss and predictions
       '''
       x, y = batch
       prediction = self.backbone(x)
       y_hat = torch.argmax(prediction.logits, dim=-1)

       loss = F.cross_entropy(prediction.logits, y)
       acc = self.acc(y_hat, y)
       
       return loss, acc, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, acc, y_hat, y = self.step(batch)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, y_hat, y = self.step(batch)

        self.log('valid_acc', acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    

#!/usr/bin/env python

import torch
import numpy as np

from sklearn.model_selection import train_test_split

from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

import lightning as L
from torch.utils.data import DataLoader, Subset

class EuroSAT_RGB_DataModule(L.LightningDataModule):
    '''
    Lightning datamodule for the EuroSAT dataset
    '''

    def __init__(self, data_root, batch_size):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size

    def setup(self, stage=None):
        '''
        Setup the dataset - here, train / valid / test all at once
        '''

        # define the transforms
        # - resize to (224, 224) as expected for ViT
        # - scale to [0,1] and transform to float32
        # - normalize with ViT mean/std

        transforms = v2.Compose([v2.ToImage(),
                                 v2.Resize(size=(224,224), interpolation=2),
                                 v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
        
        data = ImageFolder(self.data_root, transform=transforms)
        targets = np.asarray(data.targets)

        train_ix, test_ix = train_test_split(np.arange(len(data.targets)), test_size=5400, stratify=targets)
        train_ix, valid_ix = train_test_split(train_ix, test_size=2700, stratify=targets[train_ix])
                                
        self.train_data = Subset(data, train_ix)
        self.valid_data = Subset(data, valid_ix)
        self.test_data = Subset(data, test_ix)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False)
    
#!/usr/bin/env python

import lightning as L

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from eurosat_module import EuroSAT_RGB_DataModule
from vision_transformer import VisionTransformerPretrained

def main(arg):
    L.seed_everything(1312)

    # setup data
    datamodule = EuroSAT_RGB_DataModule('./data/', batch_size=32)
    datamodule.prepare_data()
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    valid_dataloader = datamodule.valid_dataloader()
    test_dataloader = datamodule.test_dataloader()

    # setup model
    model = VisionTransformerPretrained('google/vit-base-patch16-224', datamodule.num_classes, learning_rate=1e-4)

    # setup callbacks
    early_stopping = EarlyStopping(monitor='valid_acc', patience=6, mode='max')

    # logger
    logger = TensorBoardLogger("tensorboard_logs", name='eurosat_vit')

    # train
    trainer = L.Trainer(devices=1, callbacks=[early_stopping], logger=logger)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # test
    trainer.test(model=model, dataloaders=test_dataloader, verbose=True)


trainer.test(model=model, dataloaders=test_dataloader, verbose=True)