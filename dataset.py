import numpy as np
import torch
import random
import imp
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import sys, glob, os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(dir_path)
tools_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tools"))
sys.path.append(tools_path)
our_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))  # Where we are now
sys.path.append(our_path)
try:
    pau = os.environ["PYANALYSISUTILS"]
except:
    raise KeyError("Setup an environment variable to point to PYANALYSISUTILS! e.g. export PYANALYSISUTILS=<path>")
    exit(0)

import Driver

Driver.parse()


if Driver.par.driver_to_use is not None:
    dpath = os.path.abspath(Driver.par.driver_to_use)
    driver_name = dpath.split("/")[-1].split(".py")[0]
    print("Loading driver file: " + dpath)
    NN_driver = imp.load_source(driver_name, dpath)
else:
    import NN_driver


class bbMeT_NN(Dataset):
    def __init__(self, root, device, transform=None):
        self.device = device
        self.transform = transform
        self.root_dir = root
        self.scaler_file = None
        from Driver import logging

        self.stop = None
        self.domini = False
        try:
            self.do_reco_signal = NN_driver.do_reco_signal  ##Find a better way to do this
        except:
            self.do_reco_signal = False
        self.do_high_signal = False
        # import samples.Wh_21_2_116 as self.analysis_samples
        import samples.Wh_21_2_112 as analysis_samples

        self.analysis_samples = analysis_samples

        if "mini" in Driver.regions:
            self.domini = True
            self.stop = int(1e4)  # We will stop at this many events
            Driver.regions.remove("mini")

        self.msg = logging.getLogger("DatasetLoader")
        try:
            self.set_scaler_file()  ##set the scaler file name so it can be saved to the training output
            self.set_data_file()
            self.msg.info("Attempt to load:" + self.data_file)
            self.X_data, self.Y_data, self.W_data = torch.load(self.data_file, map_location=torch.device(device))

        except Exception as e:
            self.X_data, self.Y_data, self.W_data = self.process()

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, index):
        event = self.X_data[index]
        label = self.Y_data[index]
        weight = self.W_data[index]
        return event, label, weight

    def set_scaler_file(self):
        analysis = self.analysis_samples.get_name().replace("/", "")

        self.scaler_file = pau + "/ML/NN/data/processed/scaler_" + analysis + ".save"
        if self.domini:
            self.scaler_file = pau + "/ML/NN/data/processed/scaler_" + analysis + "_mini.save"
        if self.do_reco_signal:
            self.scaler_file = self.scaler_file.replace(".save", "_reco_sig.save")
        if self.do_high_signal:
            self.scaler_file = self.scaler_file.replace(".save", "_highM.save")
        if sys.version_info[0] > 3:
            self.scaler_file = self.scaler_file.replace(".save", "_p3.save")

    def set_data_file(self):
        analysis = self.analysis_samples.get_name().replace("/", "")
        self.data_file = pau + "/ML/NN/data/processed/data_" + analysis + ".pt"
        if self.domini:
            self.data_file = pau + "/ML/NN/data/processed/data_" + analysis + "_mini.pt"
        if self.do_reco_signal:
            self.data_file = self.data_file.replace(".pt", "_reco_sig.pt")
        if self.do_high_signal:
            self.data_file = self.data_file.replace(".pt", "_highM.pt")
        if sys.version_info[0] > 3:
            self.data_file = self.data_file.replace(".pt", "_p3.pt")

    def getSignals(self, doReco=None):
        if doReco is None:
            doReco = self.do_reco_signal
        sig_samples = self.analysis_samples.LoadSignalSamples()
        sig_list = []
        dm = 0  # Need to define these to make them available to eval
        m1 = 0
        m2 = 0
        try:
            expr = raw_input("Input signal group requirments based on dm, m1 or m2:  ")
        except NameError:
            expr = input("Input signal group requirments based on dm, m1 or m2:  ")
        print(expr)
        for sample in sig_samples:
            if ("Truth" in sample and not doReco) or ("Truth" not in sample and doReco):
                m1 = float(sample.split("_")[2])
                m2 = float(sample.split("_")[3])
                dm = m1 - m2
                if eval(expr):
                    sig_list.append(sample)
        # ret = smpls.get(sig_list)
        ret = [key for key in sig_samples.keys() if key in sig_list]
        self.msg.info("Signal samples: " + str(ret))
        # print(sig_list)
        return ret

    def process(self, single_sample=None):

        if single_sample is None:
            self.msg.info("Loading a new Dataset")
        else:
            self.msg.info("Loading a single sample")
        """
        Utility functions for loading data

        """

        import root_numpy as rnp
        import tools
        import selections.Wh_1Lbb as selection_strings
        import joblib

        self.set_data_file()
        self.set_scaler_file()

        variables = NN_driver.variables
        weights = NN_driver.weights
        backgrounds = NN_driver.backgrounds
        if NN_driver.classes is not None:
            classes = NN_driver.classes
            class_numbers = NN_driver.class_numbers
        else:
            classes = [backgrounds]
            class_numbers = [0]
        # backgrounds =["Z"]
        # variables = ["mct2b","nj_good"]
        if single_sample is None:
            smpls = self.analysis_samples.LoadSamples()

        ### Run with -r mini to produce a mini training sample
        selections = tools.getSelections(selection_strings, Driver.regions, None)

        print("Regions:", Driver.regions)
        region = Driver.regions[0]  # Should only be one selection set in Driver (for now)

        data_dict = []
        weight_dict = []
        nominal_tree = "CollectionTree_PFlow_"
        if single_sample is None:
            try:
                signal_samples = NN_driver.signal_samples
                if len(signal_samples) == 0:
                    raise Exception
            except Exception as e:
                signal_samples = self.getSignals()
            self.analysis_samples.LoadSignalSamples(smpls=smpls, keys=signal_samples)

            for signal_key in signal_samples:
                nominal_tree = smpls[signal_key].getNominalTree()
                signal = rnp.tree2array(smpls[signal_key][nominal_tree], branches=variables, selection=list(selections.values())[0], stop=self.stop).view(np.recarray)
                signal_weights = rnp.tree2array(smpls[signal_key][nominal_tree], branches=weights, selection=list(selections.values())[0], stop=self.stop).view(np.recarray)
                data_dict = data_dict + [(0, x) for x in signal]
                weight_dict = weight_dict + [(0, x) for x in signal_weights]
            self.msg.info("Finished loading signals")
            for i, bkg_class in enumerate(classes):
                for background_key in bkg_class:
                    nominal_tree = smpls[background_key].getNominalTree()
                    background = rnp.tree2array(smpls[background_key][nominal_tree], branches=variables, selection=list(selections.values())[0], stop=self.stop).view(np.recarray)
                    background_weights = rnp.tree2array(smpls[background_key][nominal_tree], branches=weights, selection=list(selections.values())[0], stop=self.stop).view(np.recarray)
                    data_dict = data_dict + [(class_numbers[i], x) for x in background]
                    weight_dict = weight_dict + [(class_numbers[i], x) for x in background_weights]
            self.msg.info("Finished loading backgrounds")
        if single_sample is not None:
            smpls = self.analysis_samples.LoadSignalSamples(keys=[single_sample])
            nominal_tree = smpls[single_sample].getNominalTree()
            signal = rnp.tree2array(smpls[single_sample][nominal_tree], branches=variables, selection=list(selections.values())[0], stop=self.stop).view(np.recarray)
            signal_weights = rnp.tree2array(smpls[single_sample][nominal_tree], branches=weights, selection=list(selections.values())[0], stop=self.stop).view(np.recarray)
            data_dict = data_dict + [(0, x) for x in signal]
            weight_dict = weight_dict + [(0, x) for x in signal_weights]
        random.shuffle(data_dict)
        ##Load the data here
        X = []
        W = []
        Y = []
        for label, data in data_dict:
            data = np.array([x for x in data])
            X.append(data)
            Y.append(label)
        for label, weights in weight_dict:
            W.append([x for x in weights])

        assert len(X) == len(W)
        X = np.asarray(X)
        Y = np.asarray(Y)
        W = np.asarray(W)
        ##Normalise the input dataset, if we are just loading up a single sample to test, use the scaler we created for the training set
        if NN_driver.do_scale_inputs:
            if single_sample is None:
                scaler = StandardScaler()
                scaler.fit(X)
            else:
                self.msg.info("Using scaler: " + str(self.scaler_file))
                scaler = joblib.load(self.scaler_file)
            X = scaler.transform(X)
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        W = torch.from_numpy(W)
        if single_sample:
            return X, Y, W
        torch.save((X, Y, W), self.data_file)
        self.msg.info("Saved data to: " + self.data_file)
        ##Save the scaler to import for the application

        if NN_driver.do_scale_inputs:
            joblib.dump(scaler, self.scaler_file)
            self.msg.info("Saved scaler transform file to:" + self.scaler_file)
        return X, Y, W

