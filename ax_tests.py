from ax.plot.pareto_frontier import plot_pareto_frontier
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
import itertools
import Surface_confined_inference as sci
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import os
from ax.utils.notebook.plotting import init_notebook_plotting, render
from pathlib import Path
from submitit import AutoExecutor
import sys
loc="/home/henryll/Documents/Experimental_data/Jamie/set1/"
loc="/users/hll537/Experimental_data/jamie/set1"
files =os.listdir(loc)
frequencies=[2.98, 3.99, 7.99]
run=int(sys.argv[1])

input_params={
    "2.98":{
    "E_start":-0.0987952,
    "E_reverse":0.501985,   
    "omega" :2.92436,
    "phase": 6.28319,
    "delta_E" : 0.198667,
    "v":0.0111908
    },
    "3.99":{
    "E_start":-0.0987803,
    "E_reverse":0.502042,   
    "omega" :3.94881,
    "phase": 0.00316304,
    "delta_E" : 0.198673,
    "v":0.0223832
    },
    "7.99":{
    "E_start":-0.0988314,
    "E_reverse": 0.501972,   
    "omega" :3.9395,
    "phase": 6.28319,
    "delta_E" : 0.198673,
    "v": 0.0111913
    },
}

strfreqs=[str(x) for x in frequencies]
parameters=["E0_mean","E0_std","k0","gamma","Ru", "Cdl","alpha","CdlE1","CdlE2","CdlE3"]
class FTV_evaluation:
    def __init__(self, frequencies, parameters, dataloc, input_params):
      
        strfreqs=[str(x) for x in frequencies]
        classes={}
        files=os.listdir(dataloc)
        dec_amount=32
        for i in range(0, len(frequencies)):
            

            experiment_key=strfreqs[i]
            experiment_params={
                                "Temp":278,
                                "area":0.036,
                                "N_elec":1,
                                "Surface_coverage":1e-10}
                
            for key in input_params[strfreqs[i]].keys():

                experiment_params[key]=input_params[strfreqs[i]][key]
            classes[experiment_key]={
                "class":sci.RunSingleExperimentMCMC("FTACV",
                                experiment_params,
                                problem="inverse",
                                normalise_parameters=True
                                )
                } 
            dummy_zero_class=sci.RunSingleExperimentMCMC("FTACV",
                                experiment_params,
                                problem="forwards",
                                normalise_parameters=False
                                )
            classes[experiment_key]["class"].boundaries={
                "E0":[0, 0.5],
                "E0_mean":[-0.5, -0.3],
                "E0_std":[1e-3, 0.1],
                "k0":[0.1, 5e3],
                "alpha":[0.4, 0.6],
                "Ru":[0.1, 5e3],
                "gamma":[1e-11, 1e-9],
                "Cdl":[0,1e-3],
                "CdlE1":[-0.1, 0.1],
                "CdlE2":[-0.05, 0.05],
                "CdlE3":[-0.01, 0.01],
                "alpha":[0.4, 0.6]
                }
                   
            file=[x for x in files if strfreqs[i] in x and "current" in x][0]
            data=np.loadtxt(os.path.join(dataloc,file))
            time=data[::dec_amount,0]
            current=data[::dec_amount,1]
            del data
            classes[experiment_key]["class"].dispersion_bins=[30]
            classes[experiment_key]["class"].optim_list=parameters
            classes[experiment_key]["data"]=classes[experiment_key]["class"].nondim_i(current)
            classes[experiment_key]["times"]=classes[experiment_key]["class"].nondim_t(time)
           
            classes[experiment_key]["class"].GH_quadrature=True
            classes[experiment_key]["class"].dispersion_bins=[30]
            classes[experiment_key]["class"].Fourier_fitting=True
            classes[experiment_key]["class"].Fourier_window="hanning"
            classes[experiment_key]["class"].top_hat_width=0.25
            classes[experiment_key]["class"].Fourier_function="abs"
            classes[experiment_key]["class"].Fourier_harmonics=list(range(4, 10))
            classes[experiment_key]["FT"]=classes[experiment_key]["class"].experiment_top_hat(classes[experiment_key]["times"],  classes[experiment_key]["data"] )
            dummy_zero_class.dispersion_bins=[1]
            dummy_zero_class.optim_list=parameters
            worst_case=dummy_zero_class.simulate([0.25, 1e-5, 100, 7e-11,100, 1.8e-4, 0.5, 0, 0, 0], classes[experiment_key]["times"])             
            ft_worst_case=classes[experiment_key]["class"].experiment_top_hat(classes[experiment_key]["times"], worst_case)

            classes[experiment_key]["zero_point"]=sci._utils.RMSE(worst_case, classes[experiment_key]["data"])
            classes[experiment_key]["zero_point_ft"]=sci._utils.RMSE(ft_worst_case, classes[experiment_key]["FT"])
        self.classes=classes
        self.parameter_names=parameters
        self.freqs=frequencies
    def evaluate(self, parameters):
        
        
        
        return_dict={}
        for key in self.classes.keys():
            values=[parameters.get(x) for x in self.parameter_names]
            #values=[x for x in parameters]
            data=self.classes[key]["data"]


            sim=self.classes[key]["class"].simulate(values, self.classes[key]["times"])
            FT=self.classes[key]["class"].experiment_top_hat(self.classes[key]["times"], sim)
            return_dict[key+"_ts"]=sci._utils.RMSE(sim, data)
            return_dict[key+"_ft"]=sci._utils.RMSE(FT, self.classes[key]["FT"])
        return return_dict
simclass=FTV_evaluation(frequencies, parameters, loc, input_params)
ax_client = AxClient()

#
param_arg=[
        {
            "name": parameters[x],
            "type": "range",
            "value_type":"float",
            "bounds": [0.0, 1.0],
        }
        for x in range(0,len(parameters))
    ]
objectives={}
print_tresh={}
for i in range(0, len(strfreqs)):

    objectives[strfreqs[i]+"_ts"]=ObjectiveProperties(minimize=True, threshold=simclass.classes[strfreqs[i]]["zero_point"])
    objectives[strfreqs[i]+"_ft"]=ObjectiveProperties(minimize=True, threshold=simclass.classes[strfreqs[i]]["zero_point_ft"])
print(print_tresh)

ax_client.create_experiment(
    name="FTV_experiment",
    parameters=param_arg,
    objectives=objectives,
    overwrite_existing_experiment=True,
    is_test=True,

)
paralell=ax_client.get_max_parallelism()
non_para_iterations=paralell[0][0]
directory=os.getcwd()
executor = AutoExecutor(folder=os.path.join(directory, "tmp_tests")) 
executor.update_parameters(timeout_min=60) # Timeout of the slurm job. Not including slurm scheduling delay.

executor.update_parameters(cpus_per_task=2)
executor.update_parameters(slurm_partition="nodes")
executor.update_parameters(slurm_job_name="mo_test")
executor.update_parameters(slurm_account="chem-electro-2024")
executor.update_parameters(mem_gb=2)
objectives = ax_client.experiment.optimization_config.objective.objectives
all_metrics=[objectives[x].metric for x in range(0, len(objectives))]
all_keys=[vars(x)["_name"] for x in all_metrics]
metric_dict=dict(zip(all_keys, all_metrics))
combinations=list(itertools.combinations(all_keys, 2))
print(combinations)
def save_current_front(input_dictionary):
    
    
    metrics=input_dictionary["metrics"]

    
    obj1=input_dictionary["combinations"][0]
    obj2=input_dictionary["combinations"][1]
    frontier = compute_posterior_pareto_frontier(
        experiment=input_dictionary["experiment"],
        #data=ax_client.experiment.fetch_data(),
        primary_objective=metrics[obj1],
        secondary_objective=metrics[obj2],
        absolute_metrics=[obj1, obj2],
        num_points=50,
    )

    np.save("frontier_results/set_{2}/iteration_{3}/{0}_{1}".format(obj1, obj2, input_dictionary["run"], input_dictionary["iteration"]), {"frontier":frontier})
Path(os.path.join(directory, "frontier_results","set_{0}".format(run))).mkdir(parents=True, exist_ok=True)
    
for i in range(200):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
   
    ax_client.complete_trial(trial_index=trial_index, raw_data=simclass.evaluate(parameters))
    

    #print("pre_saving")
    np.save("frontier_results/set_{1}/exp_iteration_{0}.npy".format(i, run), {"saved_frontier":ax_client})
    if i>non_para_iterations:
        Path(os.path.join(directory, "frontier_results","set_{0}".format(run), "iteration_{0}".format(i))).mkdir(parents=True, exist_ok=True)
        with executor.batch():
            for j in range(0, len(combinations)):
                save_dict={}
                save_dict["metrics"]=metric_dict
                save_dict["combinations"]=combinations[j]
                save_dict["experiment"]=ax_client.experiment
                save_dict["run"]=run
                save_dict["iteration"]=i
                executor.submit(save_current_front, save_dict)
    #print("Saving")



