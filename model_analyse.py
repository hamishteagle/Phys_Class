import xgboost as xgb
import argparse
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

par = argparse.ArgumentParser("Inputs and what to analyse")
par.add_argument("-i", "--input", dest="input", help="Input serialised XGB model")
args = par.parse_args()

ser_model = args.input
variables = [
    "mct2b",
    "MTjmin",
    "eT_miss",
    "pT_1jet",
    "pT_2jet",
    "pT_3jet",
    "pT_4jet",
    "leadb1",
    "leadb2",
    "pT_1bjet",
    "pT_2bjet",
    "nj_good",
    "metsig",
    "eta_1jet",
    "eta_2jet",
    "eta_3jet",
    "eta_4jet",
    "meff",
    "dRb1b2",
    "Rot_phib1",
    "Rot_phib2",
    "Rot_phij1",
    "Rot_phij2",
    "Rot_phij3",
    "Rot_phij4",
]
labels = [0, 1, 2, 3]
test_set = np.random.uniform(20, 500, size=(100, len(variables)))
test_labels = np.random.uniform(1, 1, size=(100, len(labels)))

bst = xgb.Booster()
bst.load_model(ser_model)
test_dmatrix = xgb.DMatrix(test_set, label=test_labels, feature_names=variables)
pred = bst.predict(test_dmatrix)

xgb.plot_importance(bst, importance_type="gain")
plt.savefig("test.png")

