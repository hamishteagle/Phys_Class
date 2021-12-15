import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import inspect
import torch
import signal
import os, sys
import xgboost as xgb
import math
from datetime import datetime
import shutil
import imp

try:
    pau = os.environ["PYANALYSISUTILS"]
except:
    raise KeyError("Setup an environment variable to point to PYANALYSISUTILS! e.g. export PYANALYSISUTILS=<path>")
    exit(0)
sys.path.append(pau)
import Driver
from Driver import logging

msg = logging.getLogger("Utils")

if Driver.par.driver_to_use is not None:
    dpath = os.path.abspath(Driver.par.driver_to_use)
    print("Using driver from " + dpath)
    driver_name = dpath.split("/")[-1].split(".py")[0]
    msg.info("Loading driver file: " + dpath)
    NN_driver = imp.load_source(driver_name, dpath)
else:
    import NN_driver


def hist_errorbars(data, xerrs=True, weights=None, *args, **kwargs):
    """Plot a histogram with error bars. Accepts any kwarg accepted by either numpy.histogram or pyplot.errorbar"""
    # pop off normed kwarg, since we want to handle it specially
    norm = False
    if "normed" in kwargs.keys():
        norm = kwargs.pop("normed")

    # retrieve the kwargs for numpy.histogram
    histkwargs = {}
    for key, value in kwargs.items():
        if key in inspect.getargspec(np.histogram).args:
            histkwargs[key] = value

    if weights is None:
        histvals, binedges = np.histogram(data, **histkwargs)
    else:
        histvals, binedges = np.histogram(data, weights=weights, **histkwargs)
    yerrs = np.sqrt(histvals)

    if norm:
        nevents = float(sum(histvals))
        binwidth = binedges[1] - binedges[0]
        histvals = histvals / nevents / binwidth
        yerrs = yerrs / nevents / binwidth

    bincenters = (binedges[1:] + binedges[:-1]) / 2

    if xerrs:
        xerrs = (binedges[1] - binedges[0]) / 2
    else:
        xerrs = None

    # retrieve the kwargs for errorbar
    ebkwargs = {}
    for key, value in kwargs.items():
        if key not in inspect.getargspec(np.histogram).args and key != "log":
            ebkwargs[key] = value
    out = plt.errorbar(bincenters, histvals, yerrs, xerrs, fmt=".", **ebkwargs)

    if "log" in kwargs.keys():
        if kwargs["log"]:
            plt.yscale("log")

    if "range" in kwargs.keys():
        plt.xlim(*kwargs["range"])

    return out


def save_model(model, save_dir, train_loader):
    """Save the model and run a tracing module to convert to c++ API"""
    torch.save(model.state_dict(), save_dir + "/model.pth")
    example_event, example_label, example_weight = next(iter(train_loader))
    traced_script_module = torch.jit.trace(model.float(), example_event.float())
    traced_script_module.save(save_dir + "/model.pt")


def get_train_test_stats(NN_driver, train_loader, test_loader, train_indices, test_indices, doReWeight):
    train_stats = [0] * (len(NN_driver.class_numbers) + 1)
    for xvec, yvec, wvec in train_loader:
        unq, stats = torch.unique(yvec, return_counts=True)
        for j, i in enumerate(unq):
            train_stats[i] += stats[j].item()

    test_stats = [0] * (len(NN_driver.class_numbers) + 1)
    for xvec, yvec, wvec in test_loader:
        unq, stats = torch.unique(yvec, return_counts=True)
        for j, i in enumerate(unq):
            test_stats[i] += stats[j].item()
    print("Training stats: %s" % (train_stats))
    reWeights = []
    if doReWeight:
        for n in train_stats:
            reWeights.append(float(len(train_indices)) / float(n))

        print("Info: Number of training events: " + str(len(train_indices)))
        for i, class_name in enumerate(NN_driver.class_numbers):
            print("Class: " + NN_driver.class_names[i] + " : " + str(train_stats[i]) + ", reweight = " + str(reWeights[i]))
        print("Info: Number of validation events: " + str(len(test_indices)))
        for i, class_name in enumerate(NN_driver.class_numbers):
            print("Class: " + NN_driver.class_names[i] + " : " + str(test_stats[i]) + ", reweight = " + str(reWeights[i]))
        return torch.FloatTensor(reWeights)
    else:
        return None


def get_batch_weights(labels, weights, n_classes, doMonoSigWgt):
    # Average the weights in the batch for each class (the NN can only take per class weights)
    avg_weights = {}
    for i in range(n_classes):
        avg_weights[i] = []
    for label, weight in zip(labels, weights):
        lab = label.item()
        # replace this signal weight LumiWeight with some central value (500_200 = 8.948e-06)
        if doMonoSigWgt:
            if lab == 0:
                weight[1] = 8.948e-06
        full_weight = torch.prod(weight, 0)
        try:
            avg_weights[lab].append(full_weight)
        except KeyError:
            avg_weights[lab] = [full_weight]
    avg_weights = [torch.Tensor(avg_weights[w]) for w in avg_weights]
    means = [torch.mean(w) for w in avg_weights]
    return torch.Tensor(means)


def raise_timeout(signum, frame):
    raise IOError


def get_do_invars():
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(10)
    try:
        do_invars = input("Do you want to draw input vars?  ")
        if "Y" or "y" in do_invars:
            signal.alarm(0)
            return True
        else:
            signal.alarm(0)
            return False
    except TimeoutError:
        print("Too slow.. not drawing")
        return False


def get_train_test_chisqr(h1, h2):
    assert len(h1) == len(h2), "Error, histogram bins are different length for caclulating the chisqr??"
    chisqr = 0
    for x, y in zip(h1, h2):
        if x > 0 and y > 0:
            chisqr += ((x - y) ** 2) / (x + y)
    return chisqr


def fill_root_hist(hist, X, W):
    assert len(X) == len(W)
    for i, pred in enumerate(X):
        if W is not None:
            hist.Fill(pred, W[i])
        else:
            hist.Fill(pred)


def getZnGlenCowen(s, b, b_err_abs):
    if b == 0:
        b = 0.01
    if b_err_abs == 0:
        b_err_abs = 0.01
    tot = s + b
    b2 = b * b
    b_err2 = b_err_abs * b_err_abs
    b_plus_err2 = b + b_err2
    retval = 0
    try:
        retval = math.sqrt(2 * ((tot) * math.log(tot * b_plus_err2 / (b2 + tot * b_err2)) - b2 / b_err2 * math.log(1 + b_err2 * s / (b * b_plus_err2))))
    except ValueError:
        pass

    return retval


# --------------Functions for loading and processing data---------------------


def pt_to_numpy(loader):
    """Convert a pytorch dataloader into numy arrays for a dmatrix"""
    for batch, labels, weights in loader:
        X = batch.cpu().numpy()
        Y = labels.cpu().numpy()
        W = weights.cpu().numpy()
    return X, Y, W


def get_variables():
    try:
        if NN_driver.drop_variables is not None:
            # _ = drop_variables(X_sam, NN_driver.variables, NN_driver.drop_variables)
            train_variables = [x for x in NN_driver.variables if x not in NN_driver.drop_variables]
    except AttributeError:
        train_variables = NN_driver.variables
    try:
        if NN_driver.constant_variables is not None:
            train_variables = train_variables + NN_driver.constant_variables
    except AttributeError:
        pass
    return train_variables


def load_numpy(dataset, NN_driver, single_sample=None, dMatrix=False):
    if single_sample is not None:
        X_sam, Y_sam, W_sam = dataset.process(single_sample=single_sample)
        X_sam = X_sam.numpy()
        Y_sam = Y_sam.numpy()
        W_sam = W_sam.numpy()
        X_sam = add_constant_variables(X_sam, NN_driver)
        train_variables = get_variables()
        if dMatrix:
            d_sam = xgb.DMatrix(X_sam, label=Y_sam, weight=product_weights(W_sam, NN_driver.do_train_weights), feature_names=train_variables)
            return d_sam, Y_sam, W_sam, X_sam
        else:
            return X_sam, Y_sam, W_sam, d_sam
    else:
        msg.error("Only use this for single samples at the moment")


def product_weights(W, do, balance=False, labels=None):
    if not do:
        return None
    if balance and not isinstance(labels, np.ndarray):
        print("labels: " + str(type(labels)))
        exit("You've requested to balance training weights for the signal class but did not properly provide the labels")

    from sklearn.preprocessing import normalize

    # Multiply the weights we have and normalise them
    ret = []
    ret = np.prod(W, axis=1)
    ret = normalize(ret[:, np.newaxis], axis=0).ravel()
    if balance:
        # sum the weights for the signal (label 0) and the background classes
        sig_tot = np.sum(ret[np.where(labels == 0)])
        bkg_tot = np.sum(ret[np.where(labels != 0)])
        scale = bkg_tot / sig_tot
        print("Scaling signals up by %f" % (scale))
        # scale the signal weights up to match those of the background class
        muls = np.array([1 if x != 0 else (scale) for x in labels])
        ret = np.multiply(ret, muls)
    return ret


def drop_variables(X, variables, dropvariables):
    # Cross-check the variables we are dropping are a subset of variables
    if not all(var in variables for var in dropvariables):
        msg.warn("dropvariables", dropvariables)
        msg.warn("variables", variables)
        msg.error("Cannot remove dropvariables from  ")
    # get the indices of the variables to drop
    train_variables = [x for x in variables if x not in dropvariables]
    drop_indices = []
    for dv in dropvariables:
        drop_indices.append(variables.index(dv))
    X = np.delete(X, drop_indices, 1)
    assert len(X[0] == len(train_variables))
    return X


def add_constant_variables(X, NN_driver):
    try:
        variables = NN_driver.constant_variables
        if len(variables) > 0:
            msg.info("Adding these variables as constants in the inputs:")
            print(variables)
            new_vars = np.zeros([X.shape[0], len(variables)])
            X = np.concatenate((X, new_vars), axis=1)
    except AttributeError:
        pass
    return X


def load_xgboost(name, reco_samples_list=None, truth_samples_list=None, client=None):
    """Load in the training and test dmatrices, first try to find a binary file, then try from .pt then load from root"""
    reco_samples = None
    truth_samples = None
    split_fraction = NN_driver.split_fraction
    variables = NN_driver.variables
    try:
        dropvariables = NN_driver.drop_variables
        msg.info("Dropping these variables from the training: ")
        for var in dropvariables:
            msg.info(var)
    except AttributeError:
        dropvariables = []
    train_variables = get_variables()
    try:

        Y_train = np.load(pau + "/ML/NN/data/processed/xgboost-%s/Y_train.npy" % (name))
        Y_test = np.load(pau + "/ML/NN/data/processed/xgboost-%s/Y_test.npy" % (name))
        X_train = np.load(pau + "/ML/NN/data/processed/xgboost-%s/X_train.npy" % (name))
        X_test = np.load(pau + "/ML/NN/data/processed/xgboost-%s/X_test.npy" % (name))
        W_train = np.load(pau + "/ML/NN/data/processed/xgboost-%s/W_train.npy" % (name))
        W_test = np.load(pau + "/ML/NN/data/processed/xgboost-%s/W_test.npy" % (name))
        if client is not None:
            import dask.array as da

            # Load the data into a daskdmatrix for distributed training
            X_train = da.from_array(X_train, chunks=1e5)
            X_test = da.from_array(X_test, chunks=1e5)
            Y_train = da.from_array(Y_train, chunks=1e5)
            Y_test = da.from_array(Y_test, chunks=1e5)
            W_train = da.from_array(W_train, chunks=1e5)
            W_test = da.from_array(W_test, chunks=1e5)
            dtrain = xgb.dask.DaskDMatrix(client, X_train, Y_train)
            dtest = xgb.dask.DaskDMatrix(client, X_test, Y_test)
        else:
            dtrain = xgb.DMatrix(pau + "/ML/NN/data/processed/xgboost-%s/train.buffer" % (name), feature_names=train_variables)
            dtest = xgb.DMatrix(pau + "/ML/NN/data/processed/xgboost-%s/test.buffer" % (name), feature_names=train_variables)
            # Get the scaler file which should be stored in the directory
            for f in os.listdir(pau + "/ML/NN/data/processed/xgboost-%s/" % (name)):
                if "scaler" in f and f.endswith(".save"):
                    scaler_file = pau + "/ML/NN/data/processed/xgboost-%s/" % (name) + f
    except Exception as e:
        print(e)
        from dataset import bbMeT_NN

        dataset = bbMeT_NN(root="data", device="cpu", transform=None)  # We need the dataset to load signle samples
        # import the torch dataset
        from torch.utils.data.sampler import SubsetRandomSampler
        from torch.utils.data import DataLoader

        indices = list(range(len(dataset)))
        split = int(np.floor(split_fraction * len(dataset)))
        train_indices, test_indices = indices[:split], indices[split:]

        # convert the torch dataset to a full numpy array (may take ages..)
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        # Hack to get the torch dataset as one big numpy array
        num_workers = 1
        batch_size = int(len(train_indices) / num_workers)

        train_start = datetime.now()
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=False)
        X_train, Y_train, W_train = pt_to_numpy(train_loader)
        train_end = datetime.now()
        msg.info("Time for loading training: " + str(train_end - train_start))

        test_start = datetime.now()
        batch_size = int(len(test_indices) / num_workers)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=True)
        X_test, Y_test, W_test = pt_to_numpy(test_loader)
        test_end = datetime.now()
        msg.info("Time for loading testing: " + str(test_end - test_start))

        msg.info("Dropping these variables from the training: " + str(dropvariables))
        msg.info("Stats-> Training: " + str(len(train_indices)) + ", Testing: " + str(len(test_indices)))
        n_signal = np.count_nonzero(Y_train == 0)
        n_bkg = len(Y_train) - n_signal
        msg.info("Stats-> Training: N Signal: " + str(n_signal) + " N Background: " + str(n_bkg))

        # Drop certain variables from the training (if they are already saved in the binaries but we don't want them)
        X_train = drop_variables(X_train, variables, dropvariables)
        X_test = drop_variables(X_test, variables, dropvariables)

        # Add additional constant variables, these are variables which are all 1 (for re-training the model with additional inputs)
        X_train = add_constant_variables(X_train, NN_driver)
        X_test = add_constant_variables(X_test, NN_driver)

        assert len(X_test) == len(Y_test) == len(W_test)
        assert len(X_train) == len(Y_train) == len(W_train)

        dtrain = xgb.DMatrix(X_train, label=Y_train, weight=product_weights(W_train, NN_driver.do_train_weights, balance=NN_driver.balance_weights, labels=Y_train), feature_names=train_variables)
        dtest = xgb.DMatrix(X_test, label=Y_test, weight=product_weights(W_test, NN_driver.do_train_weights, balance=NN_driver.balance_weights, labels=Y_test), feature_names=train_variables)

        msg.info("Training feature names:")
        print(dtrain.feature_names)

        # Save everything to binary so we don't have to process inputs again:
        if not os.path.isdir(pau + "/ML/NN/data/processed/xgboost-%s/" % (name)):
            os.mkdir(pau + "/ML/NN/data/processed/xgboost-%s/" % (name))
        msg.info("Saving binaries to %s/ML/NN/data/processed/xgboost-%s" % (pau, name))
        dtrain.save_binary(pau + "/ML/NN/data/processed/xgboost-%s/train.buffer" % (name))
        dtest.save_binary(pau + "/ML/NN/data/processed/xgboost-%s/test.buffer" % (name))
        np.save(pau + "/ML/NN/data/processed/xgboost-%s/Y_train.npy" % (name), Y_train)
        np.save(pau + "/ML/NN/data/processed/xgboost-%s/Y_test.npy" % (name), Y_test)
        np.save(pau + "/ML/NN/data/processed/xgboost-%s/X_train.npy" % (name), X_train)
        np.save(pau + "/ML/NN/data/processed/xgboost-%s/X_test.npy" % (name), X_test)
        np.save(pau + "/ML/NN/data/processed/xgboost-%s/W_train.npy" % (name), W_train)
        np.save(pau + "/ML/NN/data/processed/xgboost-%s/W_test.npy" % (name), W_test)
        msg.info("Saving scaler file to %s/ML/NN/data/processed/xgboost-%s" % (pau, name))
        shutil.copyfile(dataset.scaler_file, "%s/ML/NN/data/processed/xgboost-%s/" % (pau, name) + dataset.scaler_file.split("/")[-1])
        scaler_file = dataset.scaler_file
    # Get the truth and reco signal sets by loading single samples using dataset.process
    # reco_samples_list  = dataset.getSignals(doReco=True)
    if reco_samples_list or truth_samples_list:
        from dataset import bbMeT_NN

        dataset = bbMeT_NN(root="data", device="cpu", transform=None)  # We need the dataset to load signle samples
        reco_samples = []
        truth_samples = []
        if reco_samples_list:
            reco_samples = [load_numpy(dataset, NN_driver=NN_driver, single_sample=s, dMatrix=True) for s in reco_samples_list]
        if truth_samples_list:
            truth_samples = [load_numpy(dataset, NN_driver=NN_driver, single_sample=s, dMatrix=True) for s in truth_samples_list]

    return dtrain, dtest, X_train, X_test, Y_train, Y_test, W_train, W_test, reco_samples, truth_samples, scaler_file
    # def merge(samples)

