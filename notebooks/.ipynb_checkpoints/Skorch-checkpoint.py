#!/usr/bin/env python
# coding: utf-8

# Import all our packages and local bits

# In[1]:


import sys
sys.path.append("../")


# In[2]:


import modules
import numpy as np
import torch.nn


# In[3]:


from skorch import NeuralNetClassifier, callbacks


# Gather our data

# In[4]:


X_train = np.load("/user/hteagle/AnalysisDirectory/Rel21/Base.21.2.72/athena/PyAnalysisUtils/ML/NN/data/processed/xgboost-Wh_nominal/X_train.npy")
Y_train = np.load("/user/hteagle/AnalysisDirectory/Rel21/Base.21.2.72/athena/PyAnalysisUtils/ML/NN/data/processed/xgboost-Wh_nominal/Y_train.npy")


# In[5]:


module = modules.Cat(X_size=len(X_train[0]), num_classes=5)


# In[6]:


net = NeuralNetClassifier( module, criterion=torch.nn.CrossEntropyLoss, max_epochs=1,lr=0.1, iterator_train__shuffle=True, callbacks=[callbacks.ProgressBar()], batch_size=10000)


# In[7]:


X=X_train.astype(np.float32)
Y=Y_train.astype(np.int64)


# In[ ]:


net.fit(X,Y)


# In[ ]:




