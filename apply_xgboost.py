import xgboost as xgb
import numpy as np
import sys, os
import root_numpy as rnp
import ROOT as R
import math
import joblib

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(dir_path)
tools_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tools"))
import Driver

Driver.parse()
import time
import argparse
import imp

try:
    pau = os.environ["PYANALYSISUTILS"]
except:
    raise KeyError(
        "Setup an environment variable to point to PYANALYSISUTILS! e.g. export PYANALYSISUTILS=<path>"
    )
    exit(0)

# Load up the model
model_path = Driver.par.pt_model
model_file = Driver.par.pt_model + "/xgboost_model.model"
csv_input = Driver.par.csv_input
bst = xgb.Booster()
xgb_model = bst.load_model(model_file)
tree = Driver.apply_tree
sys.path.insert(1, os.path.abspath(model_path))

# Load up Driver config
driver_file = Driver.par.pt_model + "/NN_driver_config.py"
dpath = os.path.abspath(driver_file)
print("Using driver from " + dpath)
config = imp.load_source("NN_driver_config", dpath)

branches = config.variables
const_vars = (
    None  # ["b1_quantile","b2_quantile","j1_quantile","j2_quantile","j3_quantile"]
)
print("Variables: ", len(branches))
if config.class_names is not None:
    class_names = config.class_names
else:
    class_names = [""]
print("Loaded up the model fine")
# Load up the data
sample_dir = "/user/hteagle/ttbar_new/"


def calc_new_phis(file, trees):
    phi_branches = ["phi_met", "phi_1jet", "phi_2jet", "phi_3jet", "phi_4jet"]
    pT_branches = ["pT_1bjet", "pT_2bjet", "pT_1jet", "pT_2jet", "pT_3jet", "pT_4jet"]

    def phi_mpi_pi(phi):
        if phi > math.pi:
            return phi - (2 * math.pi)
        else:
            return phi

    # Put metphi in the right coodinate system and calculate
    def calc(metphi, phi):
        # The jets are already in mpi,pi
        if abs(phi) > 4:
            return -99
        return phi_mpi_pi(metphi) - phi

    for t in trees:
        pTs = rnp.root2array(file, t, branches=pT_branches, selection=None).view(
            np.recarray
        )
        phis = rnp.root2array(file, t, branches=phi_branches, selection=None).view(
            np.recarray
        )
        metPhis = rnp.root2array(file, t, branches=["phi_met"], selection=None).view(
            np.recarray
        )
        phi_j1s = rnp.root2array(file, t, branches=["phi_1jet"], selection=None).view(
            np.recarray
        )
        phi_j2s = rnp.root2array(file, t, branches=["phi_2jet"], selection=None).view(
            np.recarray
        )
        phi_j3s = rnp.root2array(file, t, branches=["phi_3jet"], selection=None).view(
            np.recarray
        )
        phi_j4s = rnp.root2array(file, t, branches=["phi_4jet"], selection=None).view(
            np.recarray
        )

        phis = rnp.rec2array(phis)
        phi_b1s = []
        phi_b2s = []
        for event in zip(phis, pTs):
            phi_b1 = -99
            phi_b2 = -99
            # Get the b-jet phis, checking their pT with the jets (for which we have the phi values)
            for i, jet in enumerate(event[1]):
                if event[1][0] == jet and i != 0 and jet > 0:
                    phi_b1 = event[0][i - 1]
                if event[1][1] == jet and i != 1 and jet > 0:
                    phi_b2 = event[0][i - 1]
            phi_b1s.append(phi_b1)
            phi_b2s.append(phi_b2)
        new_phi_b1s = np.array(
            [calc(metPhi.item()[0], jphi) for metPhi, jphi in zip(metPhis, phi_b1s)],
            dtype=[("new_phib1_2ls", np.float32)],
        )
        new_phi_b2s = np.array(
            [calc(metPhi.item()[0], jphi) for metPhi, jphi in zip(metPhis, phi_b2s)],
            dtype=[("new_phib2_2ls", np.float32)],
        )
        new_phi_j1s = np.array(
            [
                calc(metPhi.item()[0], jphi.item()[0])
                for metPhi, jphi in zip(metPhis, phi_j1s)
            ],
            dtype=[("new_phij1_2ls", np.float32)],
        )
        new_phi_j2s = np.array(
            [
                calc(metPhi.item()[0], jphi.item()[0])
                for metPhi, jphi in zip(metPhis, phi_j2s)
            ],
            dtype=[("new_phij2_2ls", np.float32)],
        )
        new_phi_j3s = np.array(
            [
                calc(metPhi.item()[0], jphi.item()[0])
                for metPhi, jphi in zip(metPhis, phi_j3s)
            ],
            dtype=[("new_phij3_2ls", np.float32)],
        )
        new_phi_j4s = np.array(
            [
                calc(metPhi.item()[0], jphi.item()[0])
                for metPhi, jphi in zip(metPhis, phi_j4s)
            ],
            dtype=[("new_phij4_2ls", np.float32)],
        )
        rnp.array2root(new_phi_b1s, file, t)
        rnp.array2root(new_phi_b2s, file, t)
        rnp.array2root(new_phi_j1s, file, t)
        rnp.array2root(new_phi_j2s, file, t)
        rnp.array2root(new_phi_j3s, file, t)
        rnp.array2root(new_phi_j4s, file, t)


# new_tree_name = "XGB_nom"
if Driver.par.pt_model.endswith("/"):
    new_tree_name = Driver.par.pt_model.split("/")[-2].replace(":", "_")
else:
    new_tree_name = Driver.par.pt_model.split("/")[-1].replace(":", "_")

print("Adding trees with name prefix:%s" % (new_tree_name))

start = time.process_time()
# Get the files to run on
file_list = []
if csv_input is not None:
    sample_dir = ""
    import csv

    with open(csv_input) as fp:
        reader = csv.reader(fp, delimiter=",")
        file_list = next(
            reader
        )  # Get the first line of the csv file (should only be one)
else:
    file_list = os.listdir(sample_dir)

for file in file_list:
    print(file)
    # Check the CollectionTree_PFlow_ has entries.skip if not
    # if "CollectionTree_PFlow_" not in rnp.list_trees(sample_dir + file):
    #     continue
    check_file = R.TFile(sample_dir + file)
    # nom = check_file.Get("CollectionTree_PFlow_")
    # if nom.GetEntries() == 0:
    #     print("Tree CollectionTree_PFlow_ is empty for this file: " + str(file))
    #     continue

    if tree == "all":
        trees = rnp.list_trees(sample_dir + file)
    else:
        trees = [tree]

    for t in trees:
        out = {}
        preds = {}

        # Check the tree has entries
        check_tree = check_file.Get(t)
        if check_tree.GetEntries() == 0:
            print("Tree " + str(t) + " is empty for this file" + str(file))
            continue

        # Check we're not overwriting branches
        if any(
            new_tree_name in branch
            for branch in rnp.list_branches(sample_dir + file, t)
        ):
            print("This branch is already in the tree:" + str(t))
            continue

        # ##Calculate rotated phi variables if missing
        # if "new_phib1_2ls" not in rnp.list_branches(sample_dir+file, t):
        #     print("Calculating new phis for: "+str(sample_dir+file)+":"+str(t))
        #     calc_new_phis(sample_dir+file, [t])

        data = rnp.root2array(
            sample_dir + file, t, branches=branches, selection=None
        ).view(np.recarray)
        # Convert to a normal numpy array
        data = rnp.rec2array(data)

        # setting on variable as constant
        if const_vars is not None:
            for const_var in const_vars:
                const_var_index = branches.index(const_var)
                data[:, const_var_index] = 5

        # Normalise the input
        scaler = joblib.load(
            Driver.par.pt_model + "/scaler_file.save"
        )  ##Temp...change me to pick up from the run output
        data = scaler.transform(data)
        ##Evaluate the entire tree in one go...possibly a silly way to do this...could do it in batches
        pred = bst.predict(xgb.DMatrix(data))
        #
        pred = pred.tolist()
        assert len(pred[0]) == len(class_names)
        for i, tclass in enumerate(class_names):
            preds[tclass] = [x[i] for x in pred]
            out[tclass] = np.array(
                preds[tclass], dtype=[(new_tree_name + tclass, np.float32)]
            )
            rnp.array2root(out[tclass], sample_dir + file, t)

end = time.process_time()
print("Done! This took " + str(end - start) + " s")
