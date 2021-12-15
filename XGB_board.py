import numpy as np
import utils as ut
import sys, os
import utils.utils as ut
from datetime import date, datetime
import matplotlib.pyplot as plt
import ROOT as R

import xgboost
from sklearn.metrics import roc_curve, roc_auc_score
import math

from tqdm import tqdm

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(dir_path)
import NN_driver
import Driver
from Driver import logging
import modules
import torch
import utils.utils as ut
from torch.utils.tensorboard import SummaryWriter
from PlotMaker import PlotMaker
from Histograms import Histograms, HistogramType
import collections

msg = logging.getLogger("XGB_board")


class board_object:
    def __init__(self, model, output_dir="runs/", device="cpu", xmini=None):
        mytime = datetime.now().strftime("%H:%M:%S.%f")
        print("New output:" + output_dir + str(date.today()) + "/" + str(mytime))
        self.model = model
        self.save_dir = output_dir + str(date.today()) + "/" + str(mytime)
        self.ensure_dir(self.save_dir)
        self.results_file = self.save_dir + "/results.txt"
        self.writer = SummaryWriter(self.save_dir)
        self.bins = 50
        self.doWeighted = False
        if isinstance(model, modules.Cat):
            dummy_input = torch.randn(xmini.shape).to(device)
            model(dummy_input)
            self.writer.add_graph(model, (dummy_input), False)
            self.model_type = "Pytorch"
            print("Got Pytorch model")
            self.msg = logging.getLogger("Pytorch_board")
        else:
            self.model_type = "XGBoost"
            self.msg = logging.getLogger("XGBoost_board")
        R.gStyle.SetOptStat(0)
        R.gStyle.SetOptTitle(R.kFALSE)
        self.bkg_col = R.kGreen + 2
        R.gStyle.SetPalette(R.kSolar)
        self.doWeighted = True

    def ensure_dir(self, d):
        if not os.path.isdir(d):
            os.makedirs(d)

    def get_inputs(self, X_train, X_test, Y_train=None, Y_test=None, W_train=None, W_test=None):
        self.msg.info("Getting input lists")
        if W_train is not None and W_test is not None:
            self.doWeighted = True
        if isinstance(X_train, torch.utils.data.dataloader.DataLoader) and isinstance(X_test, torch.utils.data.dataloader.DataLoader):
            self.test_preds, self.test_labels, self.test_weights, self.test_sig_list, self.test_bkg_list, self.test_sig_weights, self.test_bkg_weights = self.results_torch(X_test)
            self.msg.info("Finished getting test inputs")
            self.train_preds, self.train_labels, self.train_weights, self.train_sig_list, self.train_bkg_list, self.train_sig_weights, self.train_bkg_weights = self.results_torch(X_train)
            self.msg.info("Finished getting train inputs")

        else:
            self.test_preds, self.test_labels, self.test_weights, self.test_sig_list, self.test_bkg_list, self.test_sig_weights, self.test_bkg_weights = self.results_xgb(X_test, Y_test, W_test)
            self.msg.info("Finished getting test inputs")
            self.train_preds, self.train_labels, self.train_weights, self.train_sig_list, self.train_bkg_list, self.train_sig_weights, self.train_bkg_weights = self.results_xgb(X_train, Y_train, W_train)
            self.msg.info("Finished getting train inputs")
        self.total_bkg_list = self.train_bkg_list + self.test_bkg_list
        self.total_bkg_weights = self.train_bkg_weights + self.test_bkg_weights
        self.total_sig_list = self.train_sig_list + self.test_sig_list
        self.total_sig_weights = self.train_sig_weights + self.test_sig_weights

    def results_xgb(self, X, Y, W):
        preds = []
        labels = []
        weights = []
        sig_list = []
        bkg_list = []
        sig_weights = []
        bkg_weights = []
        for i, pred in tqdm(enumerate(self.model.predict(X))):
            # binary classes:
            if isinstance(pred, list):
                if i == 0:
                    self.msg.info("Getting test scores for binary model")
                pred = [pred]
            # multiclass
            if Y[i] == 0:
                sig_list.append(pred[0])
                if self.doWeighted:
                    sig_weights.append(np.prod(W[i], axis=0) * 139000.0)
            if Y[i] > 0:
                bkg_list.append(pred[0])
                if self.doWeighted:
                    bkg_weights.append(np.prod(W[i], axis=0) * 139000.0)
            labels.append(Y[i])
            weights.append(np.prod(W[i]))
            preds.append(pred)
        return preds, labels, weights, sig_list, bkg_list, sig_weights, bkg_weights

    def results_torch(self, loader):
        preds = []
        labels = []
        weights = []
        sig_list = []
        bkg_list = []
        sig_weights = []
        bkg_weights = []
        with torch.no_grad():
            for X, Y, W in tqdm(loader):
                Y = Y.detach().cpu().numpy()
                W = W.detach().cpu().numpy()
                for i, pred in enumerate(self.model(X.float())):
                    # binary classes:
                    if isinstance(pred, list):
                        if i == 0:
                            self.msg.info("Getting test scores for binary model")
                        pred = [pred]
                    # multiclass
                    if Y[i] == 0:
                        sig_list.append(pred[0])
                        if self.doWeighted:
                            sig_weights.append(np.prod(W[i], axis=0) * 139000.0)
                    if Y[i] > 0:
                        bkg_list.append(pred[0])
                        if self.doWeighted:
                            bkg_weights.append(np.prod(W[i], axis=0) * 139000.0)
                    labels.append(Y[i])
                    preds.append(pred.cpu().detach().numpy())
                weights = bkg_weights + sig_weights
                assert len(preds) == len(weights)
        return preds, labels, weights, sig_list, bkg_list, sig_weights, bkg_weights

    def draw_train_validation(self, X_train=None, X_test=None, Y_train=None, Y_test=None, W_train=None, W_test=None, model=None):
        self.msg.info("Drawing training and test")
        self.model = model
        if "total_bkg_list" not in self.__dict__.keys():
            self.get_inputs(X_train, X_test, Y_train, Y_test, W_train, W_test)
        plt.figure(1)
        plt.hist(self.train_bkg_list, self.bins, facecolor="g", alpha=0.5, label="train_bkg", density=True, range=[0.0, 1.0])
        ut.hist_errorbars(self.test_bkg_list, xerrs=True, normed=True, bins=self.bins, label="test_bkg", color="green", range=[0.0, 1.0])
        plt.hist(self.train_sig_list, self.bins, facecolor="r", alpha=0.5, label="train_sig", density=True, range=[0.0, 1.0])
        ut.hist_errorbars(self.test_sig_list, xerrs=True, normed=True, bins=self.bins, label="test_sig", color="red", range=[0.0, 1.0])
        plt.legend(loc="upper right")
        plt.yscale("log")
        plt.savefig(self.save_dir + "/tran_validation_hists.png", format="png")
        plt.clf()

        self.total_bkg_hist = plt.hist(self.total_bkg_list, self.bins, weights=self.total_bkg_weights, facecolor="g", alpha=0.5, label="total_bkg", density=False, range=[0.0, 1.0])
        self.total_sig_hist = plt.hist(self.total_sig_list, self.bins, weights=self.total_sig_weights, facecolor="r", alpha=0.5, label="total_bkg", density=False, range=[0.0, 1.0])
        plt.yscale("log")
        plt.savefig(self.save_dir + "/weighted_bkg_sig_hists.png", format="png")  ##Weighted total shape histograms.
        plt.clf()
        # self.get_roc(np.asarray(test_labels),np.asarray(test_preds))

    def draw_single_samples(self, single_sam_list, name):
        print("Drawing single samples")
        self.total_bkg_hist = plt.hist(self.total_bkg_list, self.bins, weights=self.total_bkg_weights, facecolor="g", alpha=0.5, label="total_bkg", density=False, range=[0.0, 1.0])
        for sam in single_sam_list:
            X_sam, Y_sam, W_sam = sam
            single_sam_list = []
            single_sam_weights = []
            for pred in self.model.predict(X_sam):
                single_sam_list.append(pred[0])
            for i in range(len(W_sam)):
                single_sam_weights.append(np.prod(W_sam[i], axis=0) * 139000.0)

            single_sample_hist = plt.hist(single_sam_list, self.bins, weights=single_sam_weights, alpha=0.5, label="single_sam", density=False, range=[0.0, 1.0], histtype="step")
        plt.yscale("log")
        plt.savefig(self.save_dir + "/weighted_single_sam_hists" + name + ".png", format="png")
        plt.clf()

    def draw_Truth_reco(self, truth_samples, reco_samples):
        truth_group = []
        reco_group = []
        for sam in truth_samples:
            X_sam, Y_sam, W_sam = sam
            truth_sam_list = []
            truth_sam_weights = []
            for pred in self.model.predict(X_sam):
                truth_sam_list.append(pred[0])
            for i in range(len(W_sam)):
                truth_sam_weights.append(np.prod(W_sam[i], axis=0))
            truth_group.append([truth_sam_list, truth_sam_weights])
        for sam in reco_samples:
            X_sam, Y_sam, W_sam = sam
            reco_sam_list = []
            reco_sam_weights = []
            for pred in self.model.predict(X_sam):
                reco_sam_list.append(pred[0])
            for i in range(len(W_sam)):
                reco_sam_weights.append(np.prod(W_sam[i], axis=0))
            reco_group.append([reco_sam_list, reco_sam_weights])
        i = 0
        for truth_tuple, reco_tuple in zip(truth_group, reco_group):
            i = i + 1
            plt.clf()
            truth_hist = plt.hist(truth_tuple[0], self.bins, weights=truth_tuple[1], alpha=0.5, label="truth_sam", density=True, range=[0.0, 1.0], histtype="step")
            reco_hist = plt.hist(reco_tuple[0], self.bins, weights=reco_tuple[1], alpha=0.5, label="reco_sam", density=True, range=[0.0, 1.0], histtype="step")
            ##do the ratio here
            plt.savefig(self.save_dir + "/truth_reco_hists_" + str(i) + ".png", format="png")

    def draw_train_test_root(self):
        self.doWeighted = True
        self.test_bkg_hist = R.TH1F("self.test_bkg_hist", "self.test_bkg_hist", 20, 0, 1)
        self.test_sig_hist = R.TH1F("self.test_sig_hist", "self.test_sig_hist", 20, 0, 1)
        self.train_bkg_hist = R.TH1F("self.train_bkg_hist", "self.train_bkg_hist", 20, 0, 1)
        self.train_sig_hist = R.TH1F("self.train_sig_hist", "self.train_sig_hist", 20, 0, 1)

        root_start = datetime.now()
        xmin = 1
        xmax = 0
        for i, pred in enumerate(self.test_preds):
            if self.test_labels[i] == 0:
                if self.doWeighted:
                    self.test_sig_hist.Fill(pred[0], np.prod(self.test_weights[i], axis=0))
                else:
                    self.test_sig_hist.Fill(pred[0])
            elif self.test_labels[i] > 0:
                if self.doWeighted:
                    self.test_bkg_hist.Fill(pred[0], np.prod(self.test_weights[i], axis=0))
                else:
                    self.test_bkg_hist.Fill(pred[0])
            if pred[0] > xmax:
                xmax = pred[0]
            if pred[0] < xmin:
                xmin = pred[0]
        for i, pred in enumerate(self.train_preds):
            if self.train_labels[i] == 0:
                if self.doWeighted:
                    self.train_sig_hist.Fill(pred[0], np.prod(self.train_weights[i], axis=0))
                else:
                    self.train_sig_hist.Fill(pred[0])
            elif self.train_labels[i] > 0:
                if self.doWeighted:
                    self.train_bkg_hist.Fill(pred[0], np.prod(self.train_weights[i], axis=0))
                else:
                    self.train_bkg_hist.Fill(pred[0])
        ##Determine the range
        for h in [self.test_bkg_hist, self.test_sig_hist, self.train_bkg_hist, self.train_sig_hist]:
            h.GetXaxis().SetRangeUser(xmin, xmax)
        c = R.TCanvas("c", "c", 1100, 900)
        c.cd()
        c.SetLogy()
        col1 = R.kGreen + 2
        col2 = R.kRed + 1
        self.train_bkg_hist.SetFillColorAlpha(col1, 0.5)
        self.train_sig_hist.SetFillColorAlpha(col2, 0.5)
        self.train_bkg_hist.SetLineColor(col1)
        self.train_sig_hist.SetLineColor(col2)
        self.test_bkg_hist.SetLineColor(col1)
        self.test_sig_hist.SetLineColor(col2 + 1)
        self.test_bkg_hist.SetMarkerStyle(R.kFullCircle)
        self.test_sig_hist.SetMarkerStyle(R.kFullCircle)
        self.test_bkg_hist.SetMarkerColor(col1)
        self.test_sig_hist.SetMarkerColor(col2 + 1)
        self.train_bkg_hist.Draw("Hist")
        self.train_sig_hist.Draw("Hist same")
        self.test_bkg_hist.Draw("e same")
        self.test_sig_hist.Draw("e same")
        c.SaveAs(self.save_dir + "/root_hists.png")
        c_n = R.TCanvas("c_n", "c_n", 1100, 900)
        c_n.cd()
        c_n.SetLogy()
        self.train_bkg_hist.Scale(1 / self.train_bkg_hist.Integral())
        self.train_sig_hist.Scale(1 / self.train_sig_hist.Integral())
        self.test_bkg_hist.Scale(1 / self.test_bkg_hist.Integral())
        self.test_sig_hist.Scale(1 / self.test_sig_hist.Integral())

        ##Get the P-value comparing train and test
        pval_bkg = self.train_bkg_hist.Chi2Test(self.test_bkg_hist, "WW")
        self.pval_bkg_KS = self.train_bkg_hist.KolmogorovTest(self.test_bkg_hist)
        pval_sig = self.train_sig_hist.Chi2Test(self.test_sig_hist, "WW")
        self.pval_sig_KS = self.train_sig_hist.KolmogorovTest(self.test_sig_hist)

        self.train_bkg_hist.SetMaximum(1)
        msg.info("PValue background KS: " + str(self.pval_bkg_KS))
        msg.info("PValue signal KS: " + str(self.pval_sig_KS))

        ##Write the PValues to results
        with open(self.results_file, "a+") as results:
            #            results.write("PValue background: "+str(pval_bkg)+"\n")
            #            results.write("PValue signal: "+str(pval_sig)+"\n")
            results.write("PValue background KS: " + str(self.pval_bkg_KS) + "\n")
            results.write("PValue signal KS: " + str(self.pval_sig_KS) + "\n")
        self.train_bkg_hist.Draw("Hist")
        self.train_sig_hist.Draw("Hist same")
        self.test_bkg_hist.Draw("ep same")
        self.test_sig_hist.Draw("ep same")
        c_n.SaveAs(self.save_dir + "/root_norm_hists.png")
        root_end = datetime.now()
        print("Time to plot with ROOT: " + str(root_end - root_start))
        return c_n

    def draw_single_samples_root(self, single_sam_list, names):
        pltm = PlotMaker()
        pltm.setSavePath(self.save_dir)
        collection = collections.OrderedDict()
        collection["preSelection"] = collections.OrderedDict()
        hists = Histograms(varname="Signal_score", selname="preSelection")

        import tools.colours

        cmap = tools.colours.GetColors("bbMeT")
        # set the colorMap
        pltm.setColorMap(cmap)
        for i, signal in enumerate(names):
            i = i + 3
            pltm.addColorMap({signal: i})
        pltm.setHistsLineWidth(3)
        pltm.setLogy()
        print("Drawing single samples")
        # c1 = R.TCanvas("c", "c", 1100, 900)
        # c1.cd()
        # c1.SetLogy()
        self.total_bkg_hist = R.TH1F("SM", "SM", self.bins, 0.5, 1)
        ut.fill_root_hist(self.total_bkg_hist, self.total_bkg_list, self.total_bkg_weights)
        self.total_bkg_hist.SetFillColorAlpha(self.bkg_col, 0.5)
        self.total_bkg_hist.SetLineColorAlpha(self.bkg_col, 0.5)

        hists.add(self.total_bkg_hist, name="SM", htype=HistogramType.BACKGROUND)

        single_hists = []
        for i, sam in enumerate(single_sam_list):
            X_raw, Y, W_raw = sam
            single_hists.append(R.TH1F(names[i], names[i], self.bins, 0.5, 1))
            if self.model_type == "Pytorch":
                W_raw = W_raw.numpy()
                X_pred = [pred[0] for pred in self.model(X_raw.float())]
            else:
                X_pred = [pred[0] for pred in self.model.predict(X_raw)]
            W = [np.prod(W_, axis=0) * 139000 for W_ in W_raw]
            ut.fill_root_hist(single_hists[i], X_pred, W)
            single_hists[i].SetLineWidth(3)
            single_hists[i].SetMarkerSize(0)
            single_hists[i].SetLineColor(pltm.colorMap[names[i]])
            hists.add(single_hists[i], name=names[i], htype=HistogramType.SIGNAL)

        collection["preSelection"]["signal_score"] = hists

        # Get the significance of each of the single samples (last 3 bins)
        for selname, collection in collection.items():
            for varname, hists in collection.items():
                pltm.setupPlot("preSelection_signal_score", isRatio=True)
                hists.sort()
                hists.useSignalLegend()
                hists.stack(ordQCD=False)
                hists.setDrawSM(True)

                # draw histograms onto the top pad
                pltm.drawTopPad(lambda x: hists.draw("hist e"))

                # also draw the legend onto this pad
                pltm.drawTopPad(lambda x: hists.drawLegend())
                # draw some labels

                pltm.drawLabels(lumi=130000, xpos=0.2, size=0.06)
                pltm.drawRatioPad(lambda x: hists.drawRatioSignificance(ref="SM", keys=names, useGlenCowen=True, ymax=4, draw_option="hist", remove_low_stat=5))

                pltm.update()
                # save the plots
                pltm.saveAs([".pdf", ".eps"])

    def get_roc(self, class_="Signal"):
        from sklearn.preprocessing import LabelBinarizer
        from sklearn import metrics

        lb = LabelBinarizer()
        labels = lb.fit_transform(self.test_labels)
        fpr = {}
        tpr = {}
        aucs = {}
        for i in range(len(NN_driver.class_names)):
            y = [label[i] for label in labels]
            x = [score[i] for score in self.test_preds]
            class_name = NN_driver.class_names[i]
            aucs[class_name] = metrics.roc_auc_score(y, x)
            fpr[class_name], tpr[class_name], _ = metrics.roc_curve(y, x)

        curve = metrics.RocCurveDisplay(fpr=fpr[class_], tpr=tpr[class_], roc_auc=aucs[class_name], estimator_name=class_)
        curve.plot()  # We need to plot the curve before we add it to the board
        self.writer.add_figure("Signal_ROC", curve.figure_)
        return aucs[class_], curve

    def calc_signif(self, sam, name, syst_bkg=0.3, nbins=None):
        cumulative = 0
        for bin in range(nbins):
            error_bkg = R.Double(0)
            error_sig = R.Double(0)
            bkg = self.total_bkg_hist.IntegralAndError(self.total_bkg_hist.GetNbinsX() - bin, self.total_bkg_hist.GetNbinsX() + 1, error_bkg)
            sig = sam.IntegralAndError(sam.GetNbinsX() - bin, sam.GetNbinsX() + 1, error_sig)
            total_error = math.sqrt(math.pow(error_bkg, 2) + math.pow(syst_bkg * bkg, 2) + math.pow(error_sig, 2))
            # significance=R.RooStats.NumberCountingUtils.BinomialExpZ(sig,bkg,total_error*bkg)
            significance = ut.getZnGlenCowen(sig, bkg, total_error * bkg)
            msg.debug("Bin:%i, yield: %f, +- %f" % (bin, bkg, error_bkg))
            msg.debug("Bin:%i, yield: %f, +- %f" % (bin, sig, error_sig))
            msg.debug("Bin:%i, signifance: %f" % (bin, significance))
            cumulative = math.sqrt(math.pow(cumulative, 2) + math.pow(significance, 2))
        with open(self.results_file, "a+") as results:
            results.write("Cumulative significance  %s: %f" % (name, cumulative) + "\n")
        msg.info("Cumulative significance (%i bins) %s: %f" % (nbins, name, cumulative))

    def draw_variable_ranking(self):
        print("Drawing variable ranking plots")
        plt.figure(3)
        by_weight = xgboost.plot_importance(self.model)
        plt.tight_layout()
        plt.savefig(self.save_dir + "/weight_feature_importances.png", format="png")
        plt.clf()
        by_gain = xgboost.plot_importance(self.model, importance_type="gain")
        plt.tight_layout()
        plt.savefig(self.save_dir + "/gain_feature_importances.png", format="png")
        plt.clf()
        by_cover = xgboost.plot_importance(self.model, importance_type="cover")
        plt.tight_layout()
        plt.savefig(self.save_dir + "/cover_feature_importances.png", format="png")
        plt.clf()

