import torch
import sys,os,imp
import root_numpy as rnp
import ROOT as R
dir_path=os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
sys.path.append(dir_path)
tools_path=os.path.abspath(os.path.join(os.path.dirname(__file__),'../../tools'))
sys.path.append(tools_path)
import Driver
import time
try:
    pau=os.environ["PYANALYSISUTILS"]
except :
    raise KeyError("Setup an environment variable to point to PYANALYSISUTILS! e.g. export PYANALYSISUTILS=<path>")
    exit(0)


##Load up the model "model.pth" and the config file "NN_driver_config" which are both written to the run directory by train.py
model_path = Driver.par.pt_model
print("Loading modules: "+os.path.abspath(Driver.par.pt_model+"/modules.py"))
modules = imp.load_source("modules", os.path.abspath(Driver.par.pt_model+"/modules.py"))
model_file = Driver.par.pt_model+"/model.pth"
csv_input =Driver.par.csv_input
sys.path.insert(1, os.path.abspath(model_path))

dpath = os.path.abspath(Driver.par.pt_model+"/NN_driver_config.py")
print("Using driver from "+dpath)
config = imp.load_source("NN_driver_config",dpath)
X_size=len(config.variables)
branches = config.variables
if config.class_names is not None:
    class_names = config.class_names
else: class_names=[""]
tree = Driver.apply_tree
##

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = modules.Cat4(X_size=X_size).to(device)
model.load_state_dict(torch.load(model_file))
print("Loaded the model fine")


#Get the files to run on
sample_dir = "/nfs/dust/atlas/user/hteagle/bbMeT_samples/training/"
#sample_dir = "/nfs/dust/atlas/user/hteagle/bbMeT_samples/processed/"
file_list=[]
if csv_input is not None:
    sample_dir=""
    import csv
    with open(csv_input) as fp:
        reader = csv.reader(fp, delimiter=',')
        file_list = next(reader)#Get the first line of the csv file (should only be one)
else: file_list = os.listdir(sample_dir)

import root_numpy as rnp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
print("Applying pt model: "+Driver.par.pt_model)
NN_branch_name="NN_"+Driver.par.pt_model.split("/")[-1].replace(":","_").replace(".","_")
print("NN branch name:"+str(NN_branch_name))
start = time.clock()
with torch.no_grad():
    model=model.float()
    model.eval()
    for file in file_list:
        if "Nominal" not in rnp.list_trees(sample_dir+file):continue
        if tree == "all":
            trees = rnp.list_trees(sample_dir+file)
        else:
            trees = [tree]
        print(file)

        for t in trees:
            out={}
            preds={}

            for branch in rnp.list_branches(sample_dir+file, t):
                if NN_branch_name in branch:
                    print("This NN name: '"+str(NN_branch_name)+"' is already in the tree: %s ..don't overwrite it"%(tree))
                    already_done_tree = True
                else:
                    already_done_tree = False

            if already_done_tree: continue
            #Check the tree has entries
            check_file = R.TFile(sample_dir+file)
            check_tree = check_file.Get(t)
            if check_tree.GetEntries()==0:
                print("Tree "+str(t)+" is empty for this file"+str(file))
                continue
            try:
                data = rnp.root2array(sample_dir+file,t, branches = branches, selection=None).view(np.recarray)
                #Convert to a normal numpy array
                data = rnp.rec2array(data)
                #Normalise the input
                scaler = joblib.load(pau+"/ML/NN/data/processed/scaler_NN_4cat.save")
                data = scaler.transform(data)
                #Convert the input to torch tensor
                data=torch.from_numpy(data)
                ##Evaluate the entire tree in one go...possibly a silly way to do this...could do it in batches
                pred = model(data.float())
                pred = pred.tolist()
                for i,tclass in enumerate(class_names):
                    preds[tclass]=[x[i] for x in pred]
                    out[tclass] = np.array(preds[tclass],dtype=[(NN_branch_name+tclass,np.float32)])
                    rnp.array2root(out[tclass], sample_dir+file, t)
            except IOError as e:

                with open(pau+"/ML/NN/run_dir/logdir/problematic_files.txt","a") as prob_file:
                    prob_file.write(file+"\n")
                print("Had some problems applying to this file!!!!: %s, tree: %s"%(file,t))
                print(e)
                pass
end=time.clock()
print("Done! This took "+str(end-start)+" s")
