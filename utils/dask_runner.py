"""
This is where we attempt to run training or hyperparameter optimisation on the batch with dask
"""
from dask_ml.model_selection import RandomizedSearchCV as daskCV
from datetime import datetime
from dask.distributed import Client
import Driver
from Driver import logging
import os


class runner:
    def __init__(self, njobs=None):
        self.msg = logging.getLogger("Runner")
        self.n_devices = njobs

    def fit(self, X, Y, classifier, params=None, scoring=None, cv_opt=False):
        """This is where we actually run the fit"""
        self.classifier = classifier
        self.scoring = scoring
        self.params = params
        self.params["eval_metric"] = params.pop("eval_metric")[0]
        self.params["learning_rate"] = params.pop("eta")  # This has a different name wrt XGB
        # Set the classifier parameters
        self.clf_params = {}
        for name, par in self.params.items():
            if not isinstance(par, list):
                self.clf_params[name] = par
        self.msg.info("Setting these parameters in the classifier:")
        print(self.clf_params)
        # Set the parameter lists to scan
        self.hyperparameters = {}
        for name, par in self.params.items():
            if isinstance(par, list):
                self.hyperparameters[name] = par
        self.msg.info("Optimising over these hypeparameters:")
        print(self.hyperparameters)
        self.classifier.set_params(**self.clf_params)
        if self.n_devices is None:
            self.n_devices = os.cpu_count() if os.cpu_count() < 10 else 10  # If we haven't configured a cluster we'll run locally, get the devices available
        self.msg.info("Starting hyperparameter optimisation with %i devices" % (self.n_devices))
        start = datetime.now()
        # If the client is setup, run with the client otherwise run locally
        if cv_opt:
            self.cv = daskCV(self.classifier, self.hyperparameters, n_jobs=self.n_devices, cv=2, scheduler=self.client, scoring=self.scoring,)
            bst = self.cv.fit(X, Y)
            self.results = self.cv.cv_results_
            self.msg.info("Best score: %f " % (self.cv.best_score_))
            self.msg.info("Best params:")
            print(self.cv.best_params_)
        else:
            bst = self.classifier.fit(X, Y)
        end = datetime.now()
        self.msg.info("Time for dask: " + str(end - start))
        return bst

    def get_cluster(self, local=False):
        from dask_jobqueue import SLURMCluster
        from dask_jobqueue import HTCondorCluster

        if local:
            self.client = Client()
        else:
            self.cluster = SLURMCluster(queue="short", cores=8, memory="2GB", walltime="00:30:00")
            self.msg.info("Setup Slurm cluster, requested %i jobs" % (self.n_devices))
            self.cluster.scale(self.n_devices)
            ##Temp try this
            self.client = Client("dcache-dot1.desy.de:8001")
            # self.client = Client(self.cluster)
            self.client.wait_for_workers()  # Wait for the client to find all the workers before sending jobs to them
            print(self.cluster.job_script())
        print(self.client.scheduler_info()["services"])

