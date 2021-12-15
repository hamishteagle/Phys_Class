import torch
import torch.nn as nn
import torch.nn.functional as F

##Standard NN
class LNet(nn.Module):
    def __init__(self, X_size):
        super(LNet, self).__init__()
        self.dropout10 = nn.AlphaDropout(p=0.1)
        self.dropout20 = nn.AlphaDropout(p=0.2)
        self.dropout30 = nn.AlphaDropout(p=0.3)
        self.linear_input = nn.Linear(X_size, X_size)
        self.linear1 = nn.Linear(X_size, X_size)
        self.linear2 = nn.Linear(X_size, X_size)
        self.linear3 = nn.Linear(X_size, X_size)
        self.linear4 = nn.Linear(X_size, X_size)
        self.linear_output = nn.Linear(X_size, 1)

    def forward(self, data):
        x = data
        x = F.relu(self.linear_input(x))
        x = self.dropout30(x)  ##Dropout 30% of first layer
        x = F.relu(self.linear1(x))
        x = self.dropout30(x)  ##Dropout 30% of second layer
        x = F.relu(self.linear2(x))
        x = self.dropout30(x)  ##Dropout 30% of second layer
        x = F.relu(self.linear3(x))
        x = self.dropout30(x)  ##Dropout 30% of second layer
        x = torch.sigmoid(self.linear_output(x))
        return x


##Standard NN
class Cat(nn.Module):
    def __init__(self, X_size=None, num_classes=None):
        super(Cat, self).__init__()
        multiplier = 4
        self.dropout10 = nn.AlphaDropout(p=0.1)
        self.dropout20 = nn.AlphaDropout(p=0.2)
        self.dropout30 = nn.AlphaDropout(p=0.3)
        self.linear_input = nn.Linear(X_size, X_size * multiplier)
        self.linear1 = nn.Linear(X_size * multiplier, X_size * multiplier)
        self.linear2 = nn.Linear(X_size * multiplier, X_size * multiplier)
        self.linear3 = nn.Linear(X_size * multiplier, X_size * multiplier)
        self.linear4 = nn.Linear(X_size * multiplier, X_size * multiplier)
        self.linear5 = nn.Linear(X_size * multiplier, X_size * multiplier)
        self.linear_output = nn.Linear(X_size * multiplier, num_classes)

    def forward(self, data):
        x = data
        x = F.relu(self.linear_input(x))
        x = F.relu(self.linear1(x))
        # x = self.dropout10(x)##Dropout 30% of second layer
        x = F.relu(self.linear2(x))
        # x = self.dropout10(x)##Dropout 30% of second layer
        x = F.relu(self.linear3(x))
        # x = self.dropout10(x)##Dropout 30% of second layer
        x = F.relu(self.linear4(x))
        # x = self.dropout10(x)##Dropout 30% of second layer
        x = F.relu(self.linear5(x))
        # x = self.dropout10(x)##Dropout 30% of second layer
        x = self.linear_output(x)
        if not self.training:
            x = torch.nn.functional.softmax(
                x
            )  ## When training/calculating losses using CrossEntropyLoss we should feed it raw logits, it applies softmax itself. When applying/getting probabilites, we should switch this on..annoying
        return x


# ##Lightening NN
# from pytorch_lightning.core.lightning import LightningModule
# import numpy as np
# from torch.utils.data.sampler import SubsetRandomSampler
# class CatL(LightningModule):
#     def __init__(self, num_classes=5, learning_rate=1e-4, batch_size=1000, weight_decay=0, split_fraction=0.8):
#         from dataset import bbMeT_NN
#         device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.dataset = bbMeT_NN(root='data', device=device, transform=None)
#         self.lr = learning_rate
#         self.split_fraction = split_fraction
#         self.weight_decay = weight_decay
#         self.batch_size = batch_size
#         X_size = self.dataset.X_data.shape[1]
#         super(CatL, self).__init__()
#         multiplier=4
#         self.dropout10 = nn.AlphaDropout(p=0.1)
#         self.dropout20 = nn.AlphaDropout(p=0.2)
#         self.dropout30 = nn.AlphaDropout(p=0.3)
#         self.linear_input = nn.Linear(X_size, X_size*multiplier)
#         self.linear1 = nn.Linear(X_size*multiplier, X_size*multiplier)
#         self.linear2 = nn.Linear(X_size*multiplier, X_size*multiplier)
#         self.linear3 = nn.Linear(X_size*multiplier, X_size*multiplier)
#         self.linear4 = nn.Linear(X_size*multiplier, X_size*multiplier)
#         self.linear5 = nn.Linear(X_size*multiplier, X_size*multiplier)
#         self.linear_output = nn.Linear(X_size*multiplier, num_classes)

#     def forward(self, x):
#         x = F.relu(self.linear_input(x))
#         x = F.relu(self.linear1(x))
#         #x = self.dropout10(x)##Dropout 30% of second layer
#         x = F.relu(self.linear2(x))
#         #x = self.dropout10(x)##Dropout 30% of second layer
#         x = F.relu(self.linear3(x))
#         #x = self.dropout10(x)##Dropout 30% of second layer
#         x = F.relu(self.linear4(x))
#         #x = self.dropout10(x)##Dropout 30% of second layer
#         x = F.relu(self.linear5(x))
#         #x = self.dropout10(x)##Dropout 30% of second layer
#         x = self.linear_output(x)
#         if not self.training:
#             x = torch.nn.functional.softmax(x)## When training/calculating losses using CrossEntropyLoss we should feed it raw logits, it applies softmax itself. When applying/getting probabilites, we should switch this on..annoying
#         return x

#     def train_dataloader(self):
#         indices = list(range(len(self.dataset)))
#         split = int(np.floor(self.split_fraction * len(self.dataset)))
#         self.train_indices = indices[:split]
#         train_sampler = SubsetRandomSampler(self.train_indices)
#         from torch.utils.data import DataLoader
#         train_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=train_sampler)
#         return train_loader

#     def val_dataloader(self):
#         indices = list(range(len(self.dataset)))
#         split = int(np.floor(self.split_fraction * len(self.dataset)))
#         self.test_indices = indices[split:]
#         test_sampler = SubsetRandomSampler(self.test_indices)
#         from torch.utils.data import DataLoader
#         val_loader = DataLoader(self.dataset, batch_size=self.batch_size, sampler=test_sampler)
#         return val_loader

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.lr , weight_decay=self.weight_decay)

#     def training_step(self, batch, batch_idx):
#         criterion = torch.nn.CrossEntropyLoss()
#         x, y, w = batch
#         logits = self(x.float())
#         loss = criterion(logits,y)
#         self.logger.experiment.add_scalars("losses", {"loss": loss}, global_step=self.global_step)
#         return {'loss': loss}

#     def training_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
#         self.logger.experiment.add_scalars("losses", {"train_loss": avg_loss}, global_step=self.global_step)
#         return {'avg_train_loss': avg_loss}

#     def validation_step(self, batch, batch_idx):
#         criterion = torch.nn.CrossEntropyLoss()
#         x, y, w = batch
#         logits = self(x.float())
#         loss = criterion(logits,y)
#         return {'val_loss': loss}

#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
#         self.logger.experiment.add_scalars("losses", {"val_loss": avg_loss}, global_step=self.global_step)
#         return {'avg_val_loss': avg_loss}

