#!/usr/bin/env python
# coding: utf-8

import sys
import torch


# In[3]:


sys.path.append("../")


# In[4]:


import modules


# In[5]:


import Driver
from dataset import bbMeT_NN
from Driver import logging

msg = logging.getLogger("Train")


# In[6]:


import NN_driver

NN_driver.load_NN_variables()


# In[7]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = bbMeT_NN(root="data", device=device, transform=None)
device


# In[8]:


num_workers = NN_driver.num_workers
batch_size = NN_driver.batch_size
split_fraction = NN_driver.split_fraction
epochs = NN_driver.epochs
lr = NN_driver.lr
weight_decay = NN_driver.weight_decay
class_names = NN_driver.class_names
num_classes = len(NN_driver.class_names)
doReWeight = NN_driver.doReWeight
doWeighted = NN_driver.doWeighted


# In[9]:


# import numpy as np
# from torch.utils.data.sampler import SubsetRandomSampler
# indices = list(range(len(dataset)))
# split = int(np.floor(split_fraction * len(dataset)))
# train_indices,test_indices = indices[:split], indices[split:]
epochs = 1
# train_sampler = SubsetRandomSampler(train_indices)
# test_sampler = SubsetRandomSampler(test_indices)


# In[10]:


model = modules.Cat(X_size=len(dataset.X_data[0]), num_classes=num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
model = model.float()


# In[11]:


import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

indices = list(range(len(dataset)))
split = int(np.floor(split_fraction * len(dataset)))
train_indices, test_indices = indices[:split], indices[split:]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False)
#%load_ext tensorboard
import XGB_board

xmini, ymini, wmini = next(iter(train_loader))
board = XGB_board.board_object(model, xmini=xmini, device=device, output_dir="current_log/")
#%tensorboard --logdir board.save_dir


# In[ ]:


from datetime import datetime
from tqdm import tqdm

for epoch in tqdm(range(epochs)):
    model.train()
    for local_batch, local_labels, local_weights in train_loader:
        local_batch, local_labels, local_weights = local_batch.to(device), local_labels.to(device), local_weights.to(device)
        optimiser.zero_grad()
        if doWeighted:
            criterion = torch.nn.CrossEntropyLoss(weight=ut.get_batch_weights(local_labels, local_weights, (len(class_names)), doMonoSigWgt))
        elif doReWeight:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        out = model(local_batch.float())
        loss = criterion(out, local_labels)
        loss.backward()
        optimiser.step()
    model.eval()
    with torch.no_grad():
        if doWeighted:
            valid_loss = 0
            train_loss = 0
            for xb, yb, wb in test_loader:
                criterion = torch.nn.CrossEntropyLoss(weight=ut.get_batch_weights(yb, wb, (len(class_names)), doMonoSigWgt))
                valid_loss = valid_loss + criterion(model(xb.float()), yb)
            for xb, yb, wb in train_loader:
                criterion = torch.nn.CrossEntropyLoss(weight=ut.get_batch_weights(yb, wb, (len(class_names)), doMonoSigWgt))
                train_loss = train_loss + criterion(model(xb.float()), yb)
        else:
            valid_loss = sum(criterion(model(xb.float()), yb) for xb, yb, wb in test_loader)
            train_loss = sum(criterion(model(xb.float()), yb) for xb, yb, wb in train_loader)
    test_loss = valid_loss / len(test_loader)
    train_loss = train_loss / len(train_loader)
    board.writer.add_scalars("Losses", {"validation_loss/epoch": test_loss, "train_loss/epoch": train_loss,}, epoch)
model.eval()  # Start evaluation
