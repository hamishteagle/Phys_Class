from torch.utils.tensorboard import SummaryWriter
from datetime import date, datetime
import time
import torch
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import utils.utils as ut
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import NN_driver as driver


class board_object:
    def __init__(self, model, X_data, device, output_dir="runs", doWeightedTests=True):
        t = time.localtime()
        mytime = datetime.now().strftime("%H:%M:%S.%f")
        #        mytime = time.strftime("%H:%M:%S.%f", t)
        print("New output:" + output_dir + "/" + str(date.today()) + "/" + str(mytime))
        self.save_dir = output_dir + "/" + str(date.today()) + "/" + str(mytime)
        self.writer = SummaryWriter(self.save_dir)
        dummy_input = torch.randn(X_data.shape).to(device)
        model(dummy_input)
        self.writer.add_graph(model, (dummy_input), False)
        self.doWeightedTests = True
        if doWeightedTests is not None:
            self.doWeightedTests = doWeightedTests
        self.bins = 20
        self.device = device

    def draw_train_validation(self, train, test, model, draw=True):
        model.eval()
        train_bkg_list = []
        train_bkg_weights = []
        test_bkg_list = []
        test_bkg_weights = []
        train_sig_list = []
        train_sig_weights = []
        test_sig_list = []
        test_sig_weights = []
        test_labels = []
        test_preds = []
        for xb, yb, wb in test:
            pred = model(xb.float())
            y_onehot = torch.FloatTensor(len(pred[0])).to(device)
            for i, y in enumerate(yb):
                if y.item() == 0:
                    test_sig_list.append(pred[i][0].item())
                    if self.doWeightedTests:
                        test_sig_weights.append(torch.prod(wb[i], 0))
                if y.item() >= 1:
                    test_bkg_list.append(pred[i][0].item())
                    if self.doWeightedTests:
                        test_bkg_weights.append(torch.prod(wb[i], 0))
                y_onehot.zero_()
                y_onehot.scatter_(0, y, 1)
                test_labels.append(y_onehot)
                test_preds.append(pred[i])
        for xb, yb, wb in train:
            pred = model(xb.float())
            for i, y in enumerate(yb):
                if y.item() == 0:
                    train_sig_list.append(pred[i][0].item())
                    if self.doWeightedTests:
                        train_sig_weights.append(torch.prod(wb[i], 0))
                if y.item() >= 1:
                    train_bkg_list.append(pred[i][0].item())
                    if self.doWeightedTests:
                        train_bkg_weights.append(torch.prod(wb[i], 0))

        # print(train_bkg_list)
        if draw:
            plt.figure(len(driver.variables) + 1)
        train_bkg_hist = plt.hist(train_bkg_list, self.bins, facecolor="g", alpha=0.5, label="train_bkg", density=True, range=[0.0, 1.0])
        ut.hist_errorbars(test_bkg_list, xerrs=True, normed=True, bins=self.bins, label="test_bkg", color="green", log="log", range=[0.0, 1.0])
        train_sig_hist = plt.hist(train_sig_list, self.bins, facecolor="r", alpha=0.5, label="train_sig", density=True, range=[0.0, 1.0])
        ut.hist_errorbars(test_sig_list, xerrs=True, normed=True, bins=self.bins, label="test_sig", color="red", log="log", range=[0.0, 1.0])
        if draw:
            plt.legend(loc="upper right")

            plt.yscale("log")
            plt.savefig(self.save_dir + "/tran_validation_hists.png", format="png")  ##Train validation normalised histograms
            plt.clf()
            total_bkg_list = train_bkg_list + test_bkg_list
            total_bkg_weights = train_bkg_weights + test_bkg_weights
            total_sig_list = train_sig_list + test_sig_list
            total_sig_weights = train_sig_weights + test_sig_weights

            total_bkg_hist = plt.hist(total_bkg_list, self.bins, weights=total_bkg_weights, facecolor="g", alpha=0.5, label="total_bkg", density=False, range=[0.0, 1.0])
            total_sig_hist = plt.hist(total_sig_list, self.bins, weights=total_sig_weights, facecolor="r", alpha=0.5, label="total_bkg", density=False, range=[0.0, 1.0])
            plt.yscale("log")
            plt.savefig(self.save_dir + "/weighted_bkg_sig_hists.png", format="png")  ##Weighted total shape histograms.

        plt.close()

        test_bkg_hist = plt.hist(test_bkg_list, self.bins, facecolor="g", alpha=0.5, label="test_bkg", density=True, range=[0.0, 1.0])
        test_sig_hist = plt.hist(test_sig_list, self.bins, facecolor="r", alpha=0.5, label="tet_sig", density=True, range=[0.0, 1.0])

        chisqr_bkg = ut.get_train_test_chisqr(train_bkg_hist[0], test_bkg_hist[0])
        chisqr_sig = ut.get_train_test_chisqr(train_sig_hist[0], test_sig_hist[0])
        print("chisqr_bkg = " + str(chisqr_bkg))
        print("chisqr_sig = " + str(chisqr_sig))
        if not draw:
            return chisqr_bkg, chisqr_sig
        print(len(test_labels[0]))
        print(len(test_preds[0]))
        pr_labels = torch.cat(test_labels)
        pr_preds = torch.cat(test_preds)
        print(pr_labels)
        print(pr_preds)
        self.writer.add_pr_curve("pr_curve", pr_labels, pr_preds, global_step=0, num_thresholds=1000)
        # self.get_roc(np.asarray(test_labels),np.asarray(test_preds))

    def draw_train_validation_inputs(self, train, test):
        train_bkg = {}
        test_bkg = {}
        train_sig = {}
        test_sig = {}
        ##load up the dictionaries with the correct names
        for var in driver.variables:
            train_bkg[var] = []
            test_bkg[var] = []
            train_sig[var] = []
            test_sig[var] = []
        for xb, yb in test:
            for i, y in enumerate(yb):
                if len(xb[i]) != len(driver.variables):
                    exit("The variables in the loader are different to those in the dataset.. exiting")
                for j, var in enumerate(driver.variables):
                    if y.item() == 0:
                        test_sig[var].append(xb[i][j].item())
                    if y.item() == 1:
                        test_bkg[var].append(xb[i][j].item())
        for xb, yb in train:
            for i, y in enumerate(yb):
                if len(xb[i]) != len(driver.variables):
                    exit("The variables in the loader are different to those in the dataset.. exiting")
                for j, var in enumerate(driver.variables):
                    if y.item() == 0:
                        train_sig[var].append(xb[i][j].item())
                    if y.item() == 1:
                        train_bkg[var].append(xb[i][j].item())
        print("Info: Number of training signal events: " + str(len(train_sig[driver.variables[0]])))
        print("Info: Number of training bkg events: " + str(len(train_bkg[driver.variables[0]])))
        print("Info: Number of validation signal events: " + str(len(test_sig[driver.variables[0]])))
        print("Info: Number of validation bkg events: " + str(len(test_bkg[driver.variables[0]])))

        chisqr_inputs = []
        for i, var in enumerate(driver.variables):
            plt.figure(i)
            train_bkg_hist = plt.hist(train_bkg[var], self.bins, facecolor="g", alpha=0.5, label="train_bkg", density=True, range=[0.0, 1.0])
            ut.hist_errorbars(test_bkg[var], xerrs=True, normed=True, bins=self.bins, label="test_bkg", color="green", log="log", range=[0.0, 1.0])
            train_sig_hist = plt.hist(train_sig[var], self.bins, facecolor="r", alpha=0.5, label="train_sig", density=True, range=[0.0, 1.0])
            ut.hist_errorbars(test_sig[var], xerrs=True, normed=True, bins=self.bins, label="test_sig", color="red", log="log", range=[0.0, 1.0])
            plt.legend(loc="upper right")
            # plt.set_title(var)
            plt.yscale("log")
            plt.savefig(self.save_dir + "/train_valid_" + var + ".png", format="png")

            test_bkg_hist = plt.hist(test_bkg[var], self.bins, facecolor="g", alpha=0.5, label="test_bkg", density=True, range=[0.0, 1.0])
            test_sig_hist = plt.hist(test_sig[var], self.bins, facecolor="r", alpha=0.5, label="tet_sig", density=True, range=[0.0, 1.0])

            chisqr_bkg = ut.get_train_test_chisqr(train_bkg_hist[0], test_bkg_hist[0])
            chisqr_sig = ut.get_train_test_chisqr(train_sig_hist[0], test_sig_hist[0])
            print("chisqr_bkg = " + str(chisqr_bkg) + ", variable:" + str(var))
            print("chisqr_sig = " + str(chisqr_sig) + ", variable:" + str(var))
            chisqr_inputs.append(chisqr_bkg)
            chisqr_inputs.append(chisqr_sig)
        plt.close()
        return chisqr_inputs

    def get_roc(self, y_valid_true, y_valid_pred):
        score = roc_auc_score(y_valid_true, y_valid_pred)
        fpr, tpr, _ = roc_curve(y_valid_true, y_valid_pred)
        plt.figure(2)
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(fpr, tpr, label="AUC=" + str(score))
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.legend(loc="best")
        plt.savefig(self.save_dir + "/validation_ROC.png", format="png")
        plt.close()

    def add_hparams(self, batch_size, lr, epochs, split_fraction, optimiser):
        param_dict = {"batch_size": batch_size, "learning_rate": lr, "n_epochs": epochs, "split_fraction": split_fraction, "optimiser": optimiser}
        self.writer.add_hparams(param_dict)
