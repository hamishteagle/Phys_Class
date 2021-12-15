import torch
import xgboost as xgb
from datetime import datetime
import shutil
import os
import sys
import imp


# Set some paths and import local modules
try:
    pau = os.environ["PYANALYSISUTILS"]
except KeyError:
    raise KeyError("Setup an environment variable to point to PYANALYSISUTILS! e.g. export PYANALYSISUTILS=<path>")
    exit(0)
sys.path.append(pau)

import Driver
from Driver import parser

subparser = parser.add_subparsers(help="XGB_training parser")
train_parser = subparser.add_parser("train")
train_parser.add_argument("-o", "--optimise", action="store_true", dest="doOpt", help="Run hyperparameter optimistation", default=False)
train_parser.add_argument("-e", "--eval_only", action="store_true", dest="eval_only", help="Run evaluation on model specified with pt_model", default=False)
train_parser.add_argument("-n", "--njobs", dest="njobs", type=int, default=5, help="Number of jobs to run with distributed optimisation")
train_parser.add_argument("-d", "--distributed", dest="doDistributed", action="store_true", default=False, help="Run the training using dask distributed")
train_parser.add_argument("--pre_apply", dest="pre_apply", default=None, help="Pre-apply a training and use it as a variable in the current training")
Driver.parse()


# These can be imported once the Driver has parsed the args
import XGB_board
from Driver import logging
import utils.utils as ut


print("XGBoost version: ", xgb.__version__)
msg = logging.getLogger("Trainer")
if Driver.par.driver_to_use is not None:
    dpath = os.path.abspath(Driver.par.driver_to_use)
    print("Using driver from " + dpath)
    driver_name = dpath.split("/")[-1].split(".py")[0]
    msg.info("Loading driver file: " + dpath)
    NN_driver = imp.load_source(driver_name, dpath)
else:
    import NN_driver


def setup_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        msg.info("cuda is available on this machine -> Use gpu resources")
        msg.info("Number of gpus available -> %i " % (torch.cuda.device_count()))

        return device, torch.cuda.device_count()
    else:
        count = os.cpu_count() if os.cpu_count() < 5 else 5
        return device, count


device, n_devices = setup_device()


# Load the XGBoost specific variables
NN_driver.load_XGB_variables()
if Driver.par.doOpt:
    NN_driver.load_XGB_hyperparameters()

param = {
    "lambda": NN_driver.lam,
    "alpha": NN_driver.alpha,
    "max_depth": NN_driver.max_depth,
    "gamma": NN_driver.gamma,
    "eta": NN_driver.eta,
    "objective": "binary:logistic",
    "nthread": NN_driver.n_threads,
    "tree_method": "hist",
    "eval_metric": ["error", "logloss"],
    "subsample": NN_driver.subsample,
    "colsample_bytree": NN_driver.colsample_bytree,
    "seed": 123,
    "random_state": 123,
}
if device == "cuda:0":
    param["tree_method"] = "gpu_hist"

# binary
if len(NN_driver.classes) < 2:
    msg.info("Using binary training")
# multi-class
else:
    msg.info("Using multiclass training with %i classes" % (len(NN_driver.classes)))
    param["num_class"] = len(NN_driver.classes) + 1
    param["objective"] = "multi:softprob"
    param["eval_metric"] = ["merror", "mlogloss"]

# Load up the traing test and separate signal we want to look at
save_name = "Wh_nominal"
if NN_driver.do_reco_signal:
    save_name = save_name + "_reco"
truth_samples_list = None  # ["C1N2_Wh_300.0_150.0", "C1N2_Wh_350.0_200.0", "C1N2_Wh_400.0_250.0"]
reco_samples_list = ["C1N2_Wh_300.0_150.0", "C1N2_Wh_350.0_200.0", "C1N2_WhTruth_350.0_200.0", "C1N2_WhTruth_300.0_150.0"]  # ["C1N2_Wh_300.0_150.0", "C1N2_Wh_350.0_200.0", "C1N2_Wh_400.0_250.0"]  #


def main():
    # For evaluating a pre-trained model
    if Driver.par.eval_only:
        if Driver.par.pt_model is None:
            exit("You asked to evaluate a model (-e, --eval_only) but supplied no model with '--pt_model")
        msg.info("Using script to evaluate pre-trained model")
        bst_xgboost = xgb.Booster()
        model_file = Driver.par.pt_model
        if Driver.par.eval_only:
            bst_xgboost.load_model(model_file)
            msg.info("Loaded model from %s " % (model_file))
    # Train
    res = {}
    xgb_start = datetime.now()
    if Driver.par.doOpt or Driver.par.doDistributed:
        from utils.dask_runner import runner
        import dask_xgboost
        from dask_ml.xgboost import XGBClassifier  # Let's roll with dask's version

        # Set up a dask runner
        xgb_runner = runner(njobs=Driver.par.njobs)
        xgb_runner.get_cluster(local=True)
        # Load the data into a dask dmatrix that can be split amongst the workers
        # dask.config.set(scheduler="single-threaded")
        dtrain, dtest, X_train, X_test, Y_train, Y_test, W_train, W_test, reco_samples, truth_samples, scaler_file = ut.load_xgboost(save_name, reco_samples_list, truth_samples_list, client=xgb_runner.client)
        bst_xgboost = xgb_runner.fit(X_train, Y_train, classifier=XGBClassifier(), params=param, scoring="roc_auc", cv_opt=Driver.par.doOpt)
        xgb_runner.client.close()
        msg.info("Finished training")
        # Load in the normal xgboost inputs for doing the post-training analysis
        dtrain, dtest, X_train, X_test, Y_train, Y_test, W_train, W_test, reco_samples, truth_samples, scaler_file = ut.load_xgboost(save_name, reco_samples_list, truth_samples_list)
    else:
        msg.info("Training parameters:")
        print(param)
        dtrain, dtest, X_train, X_test, Y_train, Y_test, W_train, W_test, reco_samples, truth_samples, scaler_file = ut.load_xgboost(save_name, reco_samples_list, truth_samples_list)

        if not Driver.par.eval_only:
            # Get pre-trained algorithm to continue training
            if Driver.par.pre_apply is not None:
                bst_xgboost = xgb.train(param, dtrain, NN_driver.n_rounds, evals=[(dtrain, "train"), (dtest, "test")], evals_result=res, early_stopping_rounds=NN_driver.xgb_early_stopping, xgb_model=Driver.par.pre_apply)
            else:
                bst_xgboost = xgb.train(param, dtrain, NN_driver.n_rounds, evals=[(dtrain, "train"), (dtest, "test")], evals_result=res, early_stopping_rounds=NN_driver.xgb_early_stopping)
    xgb_end = datetime.now()
    if not Driver.par.eval_only:
        msg.info("Time for training the model: " + str(xgb_end - xgb_start))

    # ---------------Use XGB_board to do the drawing-------------
    if Driver.par.eval_only:
        output_dir = "evals/"
    else:
        output_dir = "runs/"
    if Driver.par.output_dir is not None:
        output_dir = Driver.par.output_dir
    board = XGB_board.board_object(bst_xgboost, output_dir=output_dir)
    # --------------Save the model and relavent files--------------

    bst_xgboost.save_model(board.save_dir + "/xgboost_model.model")
    if not Driver.par.eval_only:
        if Driver.par.driver_to_use is not None:
            shutil.copyfile(dpath, board.save_dir + "/NN_driver_config.py")
        else:
            shutil.copyfile("NN_driver.py", board.save_dir + "/NN_driver_config.py")
    try:
        shutil.copyfile(scaler_file, board.save_dir + "/scaler_file.save")
    except FileNotFoundError:
        msg.warn("Couldn't copy scaler file..%s be aware if you are scaling input variables" % (scaler_file))
    print("Done, output saved to " + board.save_dir)

    board.draw_train_validation(X_train=dtrain, X_test=dtest, Y_train=Y_train, Y_test=Y_test, W_train=W_train, W_test=W_test, model=bst_xgboost)
    board.draw_train_test_root()
    board.draw_variable_ranking()
    if reco_samples is not None:
        board.draw_single_samples_root(reco_samples, reco_samples_list)
        # board.draw_single_samples(reco_samples, "reco")
    # if truth_samples is not None:
    #     board.draw_single_samples(truth_samples, "truth")
    #     board.draw_Truth_reco(truth_samples, reco_samples)
    exit("Got out")


if __name__ == "__main__":
    main()
