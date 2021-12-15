variables = [
    "m_CTcorr",
    "ETMiss",
    "ETMissPhi",
    "metsig_New",
    "m_T",
    "m_bb",
    "m_b1l",
    "m_b2l",
    "amT2",
    "pTl1",
    "pTb1",
    "pTb2",
    "pTj1",
    "pTj2",
    "pTj3",
    "b1_quantile",
    "b2_quantile",
    "j1_quantile",
    "j2_quantile",
    "j3_quantile",
    "phil1",
    "phib1",
    "phib2",
    "phij1",
    "phij2",
    "phij3",
    "etal1",
    "etab1",
    "etab2",
    "etaj1",
    "etaj2",
    "etaj3",
    "dRb1b2",
    "dRL1b1",
    "dRL1b2",
]
drop_variables = []
backgrounds = ["ttbar", "st", "W", "Higgs", "diboson", "ttV"]
classes = [["ttbar"], ["st"], ["W"], ["Higgs", "diboson", "ttV"]]
class_numbers = [1, 2, 3, 4]
class_names = ["Signal", "ttbar", "st", "W", "other"]
weights = ["mcEventWeight", "HFScale_pb", "JVTSF", "puWgt", "bJetSF", "muonSF", "electronSF", "muonTriggerSF_fix", "electronTriggerSF_fix", "isttbarMET200_OK", "YearWeight"]
num_workers = 4
do_reco_signal = True
do_scale_inputs = True
balance_weights = False


# Load up the specific variables for each method
def load_NN_variables():
    global batch_size, split_fraction, epochs, lr, doReWeight, doWeighted, doMonoSigWgt, earlyStopping, do_chisqr_stopping, is_condor, weight_decay
    batch_size = 1000
    split_fraction = 0.8  # The fraction of events in the dataset used for training (the rest is used for testing/validation)
    epochs = 30
    lr = 1e-3
    doReWeight = False
    doWeighted = False
    doMonoSigWgt = False
    earlyStopping = 100
    do_chisqr_stopping = False
    is_condor = False
    weight_decay = 0


def load_XGB_variables():
    global split_fraction, lam, alpha, max_depth, gamma, eta, min_child_weight, subsample, colsample_bytree, xgb_early_stopping, n_threads, positive_scale, n_rounds, do_train_weights, doreco_in_test
    split_fraction = 0.8
    lam = 0
    alpha = 0
    max_depth = 8
    gamma = 0
    eta = 0.6
    min_child_weight = 0.0
    subsample = 0.9
    colsample_bytree = 0.8
    xgb_early_stopping = 5
    n_threads = -1
    positive_scale = 1
    n_rounds = 100
    do_train_weights = True
    doreco_in_test = False


def load_XGB_hyperparameters():
    global max_depth, eta, alpha, min_child_weight
    max_depth = [4, 5, 6, 7]
    eta = [0.8, 0.9, 0.99]
    alpha = [0, 1, 1.5]
    min_child_weight = [0.0, 0.2, 0.4]
