import os
import sys
import random
from subprocess import Popen
import multiprocessing
import bisect
import json
import imp

try:
    pau=os.environ["PYANALYSISUTILS"]
except :
    raise KeyError("Setup an environment variable to point to PYANALYSISUTILS! e.g. export PYANALYSISUTILS=<path>")
    exit(0)
sys.path.append(pau)
import Driver
from Driver import logging
msg = logging.getLogger("Optimiser")

"""
This is a wrapper function for train_xgboost, the idea is to implement a genetic algorithm for optimising hyperparameters.
"""

# Optimisation hyperparameters
m_iterations = 5


# Basic parameters that are not going to be optimised:


driver_base = """variables=["m_CTcorr", "ETMiss", "ETMissPhi", "metsig_New", "m_T", "m_bb", "m_b1l", "m_b2l", "amT2", "pTl1", "pTb1", "pTb2", "pTj1", "pTj2", "pTj3", "b1_quantile", "b2_quantile", "j1_quantile", "j2_quantile", "j3_quantile", "phil1", "phib1", "phib2", "phij1", "phij2", "phij3", "etal1", "etab1", "etab2", "etaj1", "etaj2", "etaj3", "dRb1b2", "dRL1b1", "dRL1b2"]
backgrounds = ['ttbar','st','W','Higgs','diboson','ttV']
classes = [['ttbar'],['st'],['W'],['Higgs','diboson','ttV']]
class_numbers = [0,1,2,3,4]
class_names=["Signal", "ttbar", "st", "W", "other"]
weights=["mcEventWeight", "HFScale_pb", "JVTSF", "puWgt", "bJetSF", "muonSF", "electronSF", "muonTriggerSF_fix", "electronTriggerSF_fix", "isttbarMET200_OK", "YearWeight"]
num_workers = 4
do_reco_signal=True
do_scale_inputs = True
split_fraction =0.8
doreco_in_test=False
do_train_weights =True
n_threads = -1
positive_scale = 1
xgb_early_stopping = 5
subsample = 0.8
def load_XGB_variables():
"""
def launch(driver_pool, condor_launch=False):
    ncpus = multiprocessing.cpu_count()
    cmds_list = ['python train_xgboost.py -r preSelection -B -d ' + str(driver) for driver in driver_pool]
    if not condor_launch:
        proc = []
        active = []
        while cmds_list or proc:
            while len(proc) < 1 and cmds_list:
                msg.info("Adding new task: %s" % (cmds_list[0]))
                active.append(cmds_list[0])
                proc.append(run(cmds_list[0], log=driver_pool[0].replace(".py", ".log")))
                driver_pool.remove(driver_pool[0])
                cmds_list.remove(cmds_list[0])  # Remove the command from the list if it's been submitted

            # Add another task if there's space
            for p, cmd in zip(proc, active):
                if p.poll() is not None:
                    proc.remove(p)  # Remove the job from the process list when finished
                    active.remove(cmd)
                    print('{} done, status {}'.format(cmd.split('-d')[1], p.returncode))
                    break
    else:
        from condor_launch import make_train_job
        scheds = []
        for driver in driver_pool:
            driver_full = os.path.abspath(driver)
            output_full = os.path.abspath("Optimisation_Drivers") + "/"
            scheds.append(make_train_job("train_xgboost", output_full, fname=driver.replace(".py", ""), driver=driver_full, run_directory=os.getcwd()+"/Optimisation_Runners", cmd_template='python '+pau+'/ML/NN/train_xgboost.py -r preSelection -B --output_dir '+output_full, doSubmit=True))
        while scheds:
            for sched, id in scheds:
                q = sched.query(constraint='ClusterId=?={}'.format(id), attr_list=["ClusterId", "ProcId", "JobStatus", "JobBatchName"])
                try:
                    if q[0].eval('JobStatus') not in [1, 2, 3]:
                        msg.info("Condor Job %s has finished " % (q[0].eval("JobBatchName")))
                        scheds.remove((sched,id))
                        msg.info("%s Condor Jobs left  " %(len(scheds)))
                except IndexError:
                    # the job has already been removed
                    msg.info("Condor Job  has finished ")
                    scheds.remove((sched,id))
                    msg.info("%s Condor Jobs left  " %(len(scheds)))


def run(cmd, log='/dev/null'):
    p = Popen(cmd, stdout=open(log, "w+"), stderr=open(log.replace(".log", ".err"), 'w'), shell=True)
    return p


def clear():
    # Clear the driver list
    if os.path.exists("Optimisation_Drivers/"):
        for d in os.listdir("Optimisation_Drivers/"):
            if "driver" in d:
                os.remove("Optimisation_Drivers/"+d)
    else: os.mkdir("Optimisation_Drivers/")


def write_driver(driver_base, member, name=""):
    """Create a driver for a passed member"""

    driver = "Optimisation_Drivers/driver_"+name+".py"
    if not os.path.exists(driver):
        with open(driver, "w+") as o:
            o.write(driver_base)
            globals = ", " .join([p for p in member])
            o.write("\tglobal "+globals+"\n")
            for p, val in member.items():
                o.write("\t%s=%s \n" % (p, val))
    return driver

def init_pool(n, pool=None, previous_gen=None):

    """Loop through the optimisation parameters and select a random element from the possibilities or create a new pool from the previous_gen"""
    if pool is None:
        pool = {}

    if previous_gen is None:
        gen = '0'
        for i in range(n):
            first = {}
            if gen+"_"+str(i) in pool.keys():
                msg.info("Driver: "+gen+"_"+str(i)+" is already in the pool")
                continue
            for param, choices in opt_params.items():
                first[param] = random.choice(choices)
            pool[gen+"_"+str(i)] = first
        return pool
    else:
        pool = {}
        gen = str(int(list(previous_gen.keys())[0].split("_")[1])+1)
        msg.info("Creating new generation: "+gen)
        sampler = weighted_sampler(previous_gen)
        for i in range(n):
            mother = previous_gen[sampler()]
            father = previous_gen[sampler()]
            child = procreate(mother, father)
            pool[gen+"_"+str(i)] = child
        return pool

def weighted_sampler(pool):
    """Return a random-sample function that picks from seq weighted by weights.
    Taken directly from: https://github.com/aimacode/aima-python/blob/master/utils.py"""
    totals = []
    indivs = [indiv for indiv in pool]

    weights = [pool[indiv]['fitness'] for indiv in indivs]
    for w in weights:
        totals.append(w + totals[-1] if totals else w)

    return lambda: indivs[bisect.bisect(totals, random.uniform(0, totals[-1]))]

def mutate(child):
    prob = 0.02 # 2% probability of a parameter mutating ~20% of children mutate
    for param,choices in opt_params.items():
        if random.random() <= prob:
            msg.info("Random mutation of %s " % (param))
            child[param] = random.choice(choices)

def procreate(mother, father):
    """Simulate sexual recombination of two dna's"""
    n_genes = len(mother)
    cut = random.randrange( 0, n_genes )
    values = list(mother.values())[:cut] + list(father.values())[cut:]
    child={}
    for key,value in zip(list(mother.keys()),values):
        child[key]=value
    mutate(child)
    return child


def get_drop_variables(variables, n=1,):
    ret = [[v]for v in variables]  # Just randomly drop one for now, will have to change this up for training with truth
    return ret

def extract_local_drivers():
    pool={}
    """Extract any drivers that have already been created"""
    for file in os.listdir("Optimisation_Drivers/"):
        first={}
        if not file.endswith(".py"):
            continue
        NN_driver = imp.load_source(file,"Optimisation_Drivers/"+file)
        NN_driver.load_XGB_variables()
        for param in opt_params:
            first[param] = getattr(NN_driver,param)
        pool[file.split(".py")[0].split("driver_")[1]] = first
    return pool


def extract_fitnesses():
    """Extract the fitness of each run from the results file, write this to a map of all the drivers that have been tested"""
    # Get the list of results files
    logs = [f for f in os.listdir("Optimisation_Drivers") if ".out" in f and extract_results_file("Optimisation_Drivers/"+f)]
    missing_logs = [f for f in os.listdir("Optimisation_Drivers") if not os.path.exists("Optimisation_Drivers/"+f.replace(".py",".out")) ]
    results = [extract_results_file("Optimisation_Drivers/"+f) for f in logs]
    print(missing_logs)
    for f in missing_logs:
        msg.warn("Output log  %s for driver file %s is missing, deleting log and driver file " % (f.replace(".py", ".out"), f))
        os.remove("Optimisation_Drivers/"+f)
    # Get the database of fitnesses
    db_file = open("Optimisation_Drivers/opt_db.json", "a+")
    try:
        db = json.load(db_file)
    except json.decoder.JSONDecodeError:
        db = {}
    for l, r in zip(logs, results):
        msg.info("Getting fitness for driver: "+l)
        if os.path.exists(r.replace("\n", "/results.txt")):
            fitness = eval_fitness(r.replace("\n", "/results.txt"))
        else:
            msg.warn("Results file %s for log file %s is missing, deleting log and driver file " % (r.replace("\n", "/results.txt"), l))
            os.remove("Optimisation_Drivers/"+l)
            os.remove("Optimisation_Drivers/"+l.replace(".log", ".py"))
            os.remove("Optimisation_Drivers/"+l.replace(".out", ".py"))
            continue
        db[l.replace(".out", "")] = fitness
    json.dump(db, db_file)
    db_file.close()
    return db



def eval_fitness(result):
    """Evaluate the fitness of a result using the PValue of the train-test histograms and the cumulative significance
    """
    with open(result) as r:
        sigs = {}  # Dictionary of mC1/N2 and signifs
        # Kill anything with PValue >0.2
        for line in r:
            if "PValue" in line:
                if extract_float(line) < 0.2:
                    msg.info("PValue indicates overfitting: %f" % (extract_float(line)))
                    return 0
            if "significance" in line:
                m = float(line.split("Wh_")[1].split("_")[0])
                sigs[m] = float(line.split(":")[1].strip().rstrip())
    fitness = 0
    for m, sig in sigs.items():
        fitness += (sig*m)
    msg.info("Fitness: %f"%(fitness))
    return fitness

def extract_results_file(file):
    with open(file,"r") as f:
        for line in f:
            if "Done, output saved to" in line:
                r_file = line.split("Done, output saved to ")[1]
                return r_file
        msg.warn("File: %s has no results file!!"%(file))
        return False


def extract_float(line, st=-1):
    """Split the line by spaces until we get a float."""
    attmpt = line.split()[st].strip().rstrip()
    try:
        ret = float(attmpt)
        return ret
    except Exception:
        extract_float(line, st-1)


variables = ["m_CTcorr", "ETMiss", "ETMissPhi", "metsig_New", "m_T", "m_bb", "m_b1l", "m_b2l", "amT2", "pTl1", "pTb1", "pTb2", "pTj1", "pTj2", "pTj3", "b1_quantile", "b2_quantile", "j1_quantile", "j2_quantile", "j3_quantile", "phil1", "phib1", "phib2", "phij1", "phij2", "phij3", "etal1", "etab1", "etab2", "etaj1", "etaj2", "etaj3", "dRb1b2", "dRL1b1", "dRL1b2"]
# XGB variables to optimise and their possible ranges
global opt_params
opt_params = {}
opt_params["drop_variables"] = get_drop_variables(variables, n=1)  # Drop up to 5 variables
opt_params["lam"] = [a*0.1 for a in range(0, 20)]
opt_params["alpha"] = [a*0.1 for a in range(0, 10)]
opt_params["max_depth"] = [a for a in range(4, 10)]
opt_params["gamma"] = [a*0.1 for a in range(0, 5)]
opt_params["eta"] = [a*0.001 for a in range(995, 1000)]
opt_params["min_child_weight"] = [a*0.1 for a in range(0, 5)]
opt_params["colsample_bytree"] = [a*0.1 for a in range(0, 9)]
opt_params["n_rounds"] = [100,150,200]


def main():
    # Extract any results that have already been run
    pool = extract_local_drivers()
    pool = init_pool(100, pool=pool)


    for gen in range(m_iterations):
        pool_drivers = []
        # Write out the drivers we have created
        for number, member in pool.items():
            pool_drivers.append(write_driver(driver_base, member, name=str(number)))
        launch(pool_drivers, condor_launch=True)
        results = extract_fitnesses()

        # Get the results for this pool
        pool = {k: v for k, v in pool.items() if "driver_"+k in results}
        results = {k: v for k, v in results.items() if k.split("driver_")[1] in pool}
        print(results)
        # Append the results to the current pool
        sorted_results = []
        for driver, result in zip(pool, results):
            pool[driver]['fitness'] = results[result]
            sorted_results.append(results[result])

        sorted_results.sort(reverse=True)
        msg.info("Top 3 scores for generation %i :%f,%f,%f" % (gen, sorted_results[0], sorted_results[1], sorted_results[2]))

        # Build a new pool from the previous generation, weighted by the fitness
        pool = init_pool(50, previous_gen=pool)


if __name__ == '__main__':
    main()
