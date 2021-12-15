import os
import sys
import argparse
import getpass
import subprocess

# Find PyAnalysisUtils
try:
    pau = os.environ["PYANALYSISUTILS"]
except:
    raise KeyError("Setup an environment variable to point to PYANALYSISUTILS! e.g. export PYANALYSISUTILS=<path>")
    exit(0)

sys.path.append(pau)


def get_slurm():
    sub = subprocess.Popen("sinfo", shell=True, stdout=subprocess.PIPE)
    subprocess_return = sub.stdout.read()
    if subprocess_return != "sinfo: command not found":
        return True
    else:
        return False


def get_condor():
    sub = subprocess.Popen("condor_q", shell=True, stdout=subprocess.PIPE)
    subprocess_return = sub.stdout.read()
    if str(subprocess_return) != "":
        print("return: " + str(subprocess_return))
        try:
            import htcondor
            return True
        except:
            return False
    else:
        return False

global is_slurm, is_condor 
is_slurm = get_slurm()
is_condor = get_condor()

def split_job_files(file_dir, split, doSeparatettbar=False):
    split_file_list = []
    path = os.path.abspath(file_dir)
    file_list = []
    for file in os.listdir(file_dir):
        file_list.append(path + "/" + file)
    for i in range(split):
        split_file_list.append(file_list[(i * len(file_list) // split) : (i + 1) * len(file_list) // split])
    if doSeparatettbar:
        print("Splitting the running for ttbar")
        separate_ttbar_files = []
        for sub in split_file_list:
            for file in sub:
                if "ttbar" in file and "allhad" in file:
                    sub.remove(file)
                    separate_ttbar_files.append([file])
        split_file_list = split_file_list + separate_ttbar_files
    total_files = sum(len(i) for i in split_file_list)
    assert total_files == len(file_list), "Something's gone wrong when splitting up the files to run"
    return split_file_list


def make_application_job(fname="test", options="-t", cmd_template=None, this_split=None, run_directory=None, isttbar=False):

    setup = r"""#!/bin/bash
    source %s/NNsetup.sh
    mkdir -p %s
    cd %s""" % (
        pau,
        run_directory,
        run_directory,
    )
    fcmd = ""
    if not is_slurm:
        fcmd = setup
    fcmd += "\nmkdir -p %s; cd %s" % (fname, fname)
    fcmd += "\n" + cmd_template
    csv_path = os.path.abspath("run_csvs")
    fcmd += "--csv_input %s/%s.csv" % (csv_path, this_split)
    fname = fname + ".sh"
    a = open(fname, "w")
    a.write(fcmd)
    if doSubmit:
        
        if is_condor:
            os.system("chmod u+x " + fname)
            import htcondor

            myjob = htcondor.Submit()
            collector = htcondor.Collector()
            mySchedd = htcondor.Schedd()
            with mySchedd.transaction() as myTransaction:
                myjob["executable"] = fname
                myjob["requirements"] = 'OpSysAndVer == "CentOS7"'
                myjob["request_cpus"] = "1"
                myjob["transfer_executable"] = "True"
                if isttbar:
                    myjob["request_memory"] = "3 GB"
                else:
                    myjob["request_memory"] = "1 GB"
                myjob["universe"] = "vanilla"
                myjob["output"] = fname.replace(".sh", "") + ".out"
                # print myjob["output"]
                myjob["error"] = fname.replace(".sh", "") + ".error"
                myjob["log"] = fname.replace(".sh", "") + ".log"
                myjob["JobBatchName"] = fname.replace(".sh", "")
                myjob.queue(myTransaction)
        elif is_slurm:
            from slurmpy import Slurm
            s = Slurm(fname, {"partition": "compute", "time":"24:00:00", "ntasks": "1", "mem":"5G", "mail-user": "hamish.teagle@cern.ch"})
            s.run(fcmd)


def make_train_job(command_option, train_output, fname="test", options="-t", driver=None, doSubmit=False, run_directory=None, cmd_template=None):

    if cmd_template is None:
        cmd_template = "python " + pau + "/ML/NN/%s.py -B --output_dir %s " % (command_option, train_output)
    setup = r"""#!/bin/bash
source %s/NNsetup.sh
mkdir -p %s
cd %s""" % (
        pau,
        run_directory,
        run_directory,
    )
    fcmd = setup
    fcmd += "\nmkdir -p %s; cd %s" % (fname, fname)
    fcmd += "\ncp %s/ML/NN/modules.py ." % (pau)
    fcmd += "\n" + cmd_template + " -d " + driver
    fname = fname + ".sh"
    a = open(fname, "w")
    a.write(fcmd)
    print("Writing train file: " + fname)
    if doSubmit:
        if is_condor:
            os.system("chmod u+x " + fname)
            import htcondor

            myjob = htcondor.Submit()
            collector = htcondor.Collector()
            mySchedd = htcondor.Schedd()
            batch_name = fname.replace(".sh", "")
            with mySchedd.transaction() as myTransaction:
                myjob["executable"] = fname
                myjob["requirements"] = 'OpSysAndVer == "CentOS7"'
                myjob["request_cpus"] = "1"
                # myjob["request_memory"] = "4"
                myjob["transfer_executable"] = "True"
                myjob["+RequestRuntime"] = "100000"
                myjob["universe"] = "vanilla"
                myjob["output"] = fname.replace(".sh", "") + ".out"
                # print myjob["output"]
                myjob["error"] = fname.replace(".sh", "") + ".error"
                myjob["log"] = fname.replace(".sh", "") + ".log"
                myjob["JobBatchName"] = batch_name
                id = myjob.queue(myTransaction)
        return mySchedd, id
    else: 
        exit("TODO: not setup to run train on slurm batch")


def main():
    parser = argparse.ArgumentParser(description="Script for running ML application and training on condor")
    parser.add_argument("--do", action="store", dest="command_option", type=str, required=True, help="which command to run, train,train_xgboost,apply,apply_xgboost")
    parser.add_argument("--rc", action="store", dest="run_configs", type=str, default="drivers", required=False, help="Directory of drivers to run training with")
    parser.add_argument("--rd", action="store", dest="run_directory", type=str, required=False, default="/nfs/dust/atlas/user/hteagle/ML_runs/", help="Directory where you want the training to run")
    parser.add_argument("--fd", action="store", dest="file_directory", type=str, required=False, default="/nfs/dust/bbMeT_samples/practice/")
    parser.add_argument("--pt_model", action="store", dest="pt_model", type=str, required=False)
    parser.add_argument("--at", action="store", dest="apply_trees", type=str, required=False)
    parser.add_argument("--submit", "-s", dest="doSubmit", action="store_true", default=False)
    parser.add_argument("--train_output", dest="train_output", action="store", default="../runs/", help="Directory to put the output of the training in")
    # ============== Global variables =================
    par = parser.parse_args()
    global pt_model, apply_trees, command_option, run_configs, run_directory, file_directory, train_output, doSubmit
    doSubmit = par.doSubmit
    pt_model = par.pt_model
    apply_trees = par.apply_trees
    command_option = par.command_option
    run_configs = par.run_configs
    run_directory = par.run_directory
    file_directory = par.file_directory
    train_output = par.train_output

    doSeparatettbar = True
    # basic commands to run
    if command_option not in ["train", "train_xgboost", "apply", "apply_xgboost"]:
        exit("Supply a command: 'train,train_xgboost,apply,apply_xgboost'")

    if command_option in ["train", "train_xgboost"]:
        if run_configs is None:
            exit("Need to supply a run config directory for this job")
        cmd_template = "python " + pau + "/ML/NN/%s.py --output_dir %s " % (command_option, train_output)
        if not os.path.exists(run_configs) or len(os.listdir(run_configs)) == 0:
            exit("Couldn't find the driver config files>>exit")
        if not os.path.exists("logdir"):
            os.mkdir("logdir")
        # if len(os.listdir("logdir"))>0: os.system("rm logdir/*")
        for i, driver in enumerate(os.listdir(run_configs)):
            if ".py" in driver and ".pyc" not in driver:
                print("Launching: " + str(driver))
                make_train_job(fname=command_option + "_" + str(i), cmd_template=cmd_template, driver=os.path.abspath(run_configs + "/" + str(driver)))
    elif command_option in ["apply", "apply_xgboost"]:
        cmd_template = "python " + pau + "/ML/NN/%s.py  --pt_model %s --at %s  " % (command_option, os.path.abspath(pt_model), apply_trees)

        # Split the files to be run over
        split = 50
        if len(os.listdir(file_directory)) < split:
            split = len(os.listdir(file_directory))
        file_lists = split_job_files(file_dir=file_directory, split=split, doSeparatettbar=doSeparatettbar)
        # Write the list of files to csv, do this so we can manipulate what's run by hand if we like
        if not os.path.exists("run_csvs"):
            os.mkdir("run_csvs")
        if len(os.listdir("run_csvs")) > 0:
            os.system("rm run_csvs/*")
        if not os.path.exists("logdir"):
            os.mkdir("logdir")
        # if len(os.listdir("logdir"))>0: os.system("rm logdir/*")
        import csv

        for i, sub_list in enumerate(file_lists):
            file_csv = open("run_csvs/" + str(i) + ".csv", "w")
            writer = csv.writer(file_csv)
            writer.writerow(sub_list)
            isttbar = False
            if any("410470" in x for x in sub_list):
                isttbar = True
            file_csv.close()
            make_application_job(fname=command_option + "_" + str(i), cmd_template=cmd_template, this_split=i, run_directory=run_directory, isttbar=isttbar)


if __name__ == "__main__":
    main()
