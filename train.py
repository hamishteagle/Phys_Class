import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import bbMeT_NN
import modules
import numpy as np
import tensorboard_modules
import utils.utils as ut
import shutil
import Driver
import os,imp
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = bbMeT_NN(root='data', device=device, transform=None)
from Driver import logging
msg=logging.getLogger("Train")
##Get the hyperparameters from the NN_driver (config file)
if Driver.par.driver_to_use is not None:
    dpath = os.path.abspath(Driver.par.driver_to_use)
    print("Using driver from "+dpath)
    NN_driver = imp.load_source("NN_driver",dpath)

else: import NN_driver
is_condor=NN_driver.is_condor##must be a better way to do this
num_workers = NN_driver.num_workers
batch_size = NN_driver.batch_size
split_fraction = NN_driver.split_fraction
epochs = NN_driver.epochs
lr = NN_driver.lr
weight_decay=NN_driver.weight_decay
class_names=NN_driver.class_names
num_classes = len(NN_driver.class_names)
##Early stopping
earlyStopping = NN_driver.earlyStopping
chisqr_stopping = NN_driver.do_chisqr_stopping
doMonoSigWgt = NN_driver.doMonoSigWgt


indices = list(range(len(dataset)))
split = int(np.floor(split_fraction * len(dataset)))
train_indices,test_indices = indices[:split], indices[split:]

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)


train_loader = DataLoader(dataset, batch_size=batch_size,sampler=train_sampler, drop_last=True)
test_loader = DataLoader(dataset, batch_size=batch_size,sampler=test_sampler, drop_last=False)

doReWeight=NN_driver.doReWeight
doWeighted=NN_driver.doWeighted

if doReWeight:
    msg.info("Getting class weights")
    class_weights = ut.get_train_test_stats(NN_driver, train_loader, test_loader, train_indices, test_indices, doReWeight)
else:
    class_weights =None

#if doWeighted:

model = modules.Cat(X_size=len(dataset.X_data[0]),num_classes=num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss( weight = class_weights)
optimiser = torch.optim.Adam(model.parameters(), lr=lr , weight_decay=weight_decay)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,patience=3,threshold=0.001,factor=0.5)
model=model.float()

output_dir="runs"
if Driver.par.output_dir is not None:
    output_dir=Driver.par.output_dir
xmini,ymini,wmini = next(iter(train_loader))
board = tensorboard_modules.board_object(model,xmini,output_dir=output_dir)



if chisqr_stopping: do_invars="yes"
elif not is_condor: do_invars = ut.get_do_invars()

else: do_invars=False

if do_invars == "y" or do_invars == "yes" :
    msg.info("Drawing input variable plots...")
    chisqr_inputs = board.draw_train_validation_inputs(train_loader,test_loader)
    msg.info(chisqr_inputs)
    max_input_chisqr = max(chisqr_inputs)
    msg.info("done")



msg.info("Beginning training")
msg.info("Number of batches: "+str(len(train_loader)))
valid_loss_track=100
consistent_increase=0
for epoch in range(epochs):
    model.train()
    for local_batch, local_labels, local_weights in train_loader:
        local_batch, local_labels, local_weights = local_batch.to(device), local_labels.to(device), local_weights.to(device)
        optimiser.zero_grad()
        if doWeighted:
            criterion = torch.nn.CrossEntropyLoss( weight = ut.get_batch_weights(local_labels,local_weights,(len(class_names)),doMonoSigWgt))
        elif doReWeight:
            criterion = torch.nn.CrossEntropyLoss( weight = class_weights)
        out = model(local_batch.float())
        loss = criterion(out, local_labels)
        loss.backward()
        optimiser.step()
    model.eval()
    with torch.no_grad():
        msg.info("Calculating Epoch losses:")
        if doWeighted:
            valid_loss = 0
            train_loss = 0
            for xb, yb, wb in test_loader:
                criterion = torch.nn.CrossEntropyLoss( weight = ut.get_batch_weights(yb,wb,(len(class_names)),doMonoSigWgt))
                valid_loss = valid_loss + criterion(model(xb.float()), yb)
            for xb, yb, wb in train_loader:
                criterion = torch.nn.CrossEntropyLoss( weight = ut.get_batch_weights(yb,wb,(len(class_names)),doMonoSigWgt))
                train_loss = train_loss + criterion(model(xb.float()), yb)
        else:
            valid_loss = sum(criterion(model(xb.float()), yb) for xb, yb, wb in test_loader)
            train_loss = sum(criterion(model(xb.float()), yb) for xb, yb, wb in train_loader)
    test_loss = valid_loss/len(test_loader)
    train_loss = train_loss/len(train_loader)
    #scheduler.step(test_loss)

    board.writer.add_scalars("Losses",{"validation_loss/epoch":test_loss,
                                        "train_loss/epoch":train_loss,},
    epoch
    )
    board.writer.add_scalar("learning_rate",lr, epoch)
    msg.info("Epoch: %i Loss: %f Validation Loss: %f" %(epoch,train_loss,test_loss))
    ##Early stopping
    if test_loss<valid_loss_track:
        valid_loss_track = test_loss
        consistent_increase = 0
    else:
        consistent_increase = consistent_increase + 1
    if consistent_increase>earlyStopping:
        print("Info: Stopping due to validation loss not decreasing for 5 rounds")
        break
    if chisqr_stopping:
        with torch.no_grad():
            chisqr_bkg,chisqr_sig = board.draw_train_validation(train_loader,test_loader,model, draw=False)
            if chisqr_bkg>max_input_chisqr or chisqr_sig>max_input_chisqr:
                print("Info: Stopping due to chisqr_bkg/sig>chisqr_input")
                break
with torch.no_grad():
    board.draw_train_validation(train_loader,test_loader,model)
    #board.add_hparams(batch_size,lr,epochs,n_hidden,split_fraction,"Adam")
board.writer.close()
ut.save_model(model, board.save_dir,train_loader)
if Driver.par.driver_to_use is not None:
    shutil.copyfile(Driver.par.driver_to_use, board.save_dir+"/NN_driver_config.py")
else:
    shutil.copyfile("NN_driver.py", board.save_dir+"/NN_driver_config.py")
shutil.copyfile("modules.py", board.save_dir+"/modules.py")
##Save a copy of the scaler that was used in this training
shutil.copyfile(dataset.scaler_file , os.path.join(board.save_dir,dataset.scaler_file)
