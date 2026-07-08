Banana boxes
======

- Summary: Evaluate approaches to calculate the total drug weight in a batch  
- Developed by: Netherlands Forensics Institute, team Evidence Evaluation & Statistics, 2026  
- Publication: submitted for publication (DOI-link will be added when available)

## General background

### Situation being modelled
- A batch of cardboard banana boxes has been seized. The bottoms of the boxes were impregnated with cocaine.  
- The entire batch consists of 1000 boxes, of which 300 have been analysed for cocaine presence.  
- From 5 cocaine-containing boxes, the weights of the bottoms have been measured (including cocaine).  
- For 10 cardboard bottoms, the cocaine concentrations have been measured using 3 repeats per bottom.  
- An estimate is required for the total weight of cocaine in the batch, together with an uncertainty interval. 

### Modelling approaches

This repository currently supports the evaluation of five modelling approaches:
- An existing frequentist approach, using weight and drug-presence measurements, based on a tool from ENFSI.
- An existing frequentist approach, using weight and drug-presence measurements, based on a tool from SWGDRUG.
- An existing frequentist approach, using weight and drug-presence measurements, based on Alberink _et al._ (2010).
- An existing frequentist approach, using weight and concentration measurements, based on Alberink _et al._ (2016).
- A novel Bayesian approach, using weight, drug-presence and concentration measurements, and using MCMC-simulations.

## Using the repository
### Setting things up
- Python version: 3.12
- Packages used: listed in `requirements.txt` (including version numbers)
- Configuration: the `input_configurations` folder contains all files hat define the simulations:
  - `default_case.yaml`: default values for all simulation settings and parameters 
  - `local_adjustments.yaml`: values in here will override the values of all simulations being done.  
  An example `local_adjustments.yaml-example` is provided, it can be renamed to `local_adjustments.yaml`.
  - To reproduce specific simulation results, change the `random_seed:` from empty/none to a 'date/time-stamp' number.  
  This date/time-stamp can be found in the folder name of that specific simulation, for example `20260415035613`.

### Running simulations
There are two ways to run the simulations: via the Python command line, or using a file with a series of OS-commands.
#### Python command line
- Start in `main.py`
- By default a single simulation run is done, based on the values in `default_case.yaml` and `local_adjustments.yaml`.
- Several optional input arguments are supported:
  - `-no_default_run`: Do not run the default simulation defined in `default_case.yaml`.
  - `-run_single_yaml`, in combination with `-yaml_file` and `path/to/file.yaml`:  
  Run the simulation specified in `file.yaml`, located in the folder `path/to`.  
  - `-run_all_yaml_files_in_folder`, in combination with `-folder_yaml_files` and `path/to/folder`:  
  Run the simulations specified by all yaml-files located in the folder `path/to/folder`.
  - `-run_method_comparison`:  
  Run the simulations specified by all yaml-files located in the folder `input_configurations/method_comparison`.
  - `-run_sensitivity_analysis`:  
  Run the simulations specified by all yaml-files located in the folder `input_configurations/sensitivity_analysis`.
  - `-create_overview_results`, in combination with `-results_dir` and `path/to/folder`:  
  Make an overview table of all simulations results located in the sub-folders of the folder `path/to/folder`.
- The four `-run_...` arguments should not be combined; use only one of them at a time.
  - If any of them is used, `-results_dir` is overwritten with the output folder of that run-argument.
- Some example usages:
  - To do the sensitivity analysis including default run and summarising overview:  
  `python main.py -run_sensitivity_analysis -create_overview_results`
  - To do the method comparison without default run and including summarising overview:  
  `python main.py -no_default_run -run_method_comparison -create_overview_results`
  - To make a summarising overview of a selection previously run simulation results:  
  `python main.py -no_default_run -create_overview_results -results_dir results/my_result_selection/`

#### Series of OS-commands
- Automatically clearing pytensor cache 
  - When running a single yaml-file, the 'simulation time per run' slowly increases from the first to the last run.
  - This behaviour is caused by the growing size of the pytensor cache, related to the MCMC-calculations.
  - By default, this cache is cleared before the start of a simulation (`clear_cache: True`).
  - Clearing the cache is only possible before Python starts its calculation, not during the calculations.
  - To automatically run many simulation-yamls while clearing the cache in between, a series of OS-commands can be used.
- Examples files are included for two common types of Operating System (OS): 
  - Windows: `run_all_yamls.bat-example`
  - Linux: `run_all_yamls.sh-example`
- To use such a file, do the following:
  - Copy the relevant file, and remove `-example` from the file extension
  - Update some settings in these files, for example:
    - Location of the Python interpreter to use
    - Location of folder(s) with simulation yaml-files to run
  - Execute the `.bat` or `.sh` file

### Description of other folders and files

- `tests`: Folder with several files that contain multiple tests, to verify the implementations of the modelling approaches.  
  The random nature of the MCMC-calculations may cause the tests in `test_baysian.py` to fail, depending on hardware.
- `boundary_methods.py`: Functions to calculate the relevant percentiles, based on weights, presences, and concentrations.
  - `calculate_percentiles_enfsi`: Using ENFSI approach: lower bound, best estimate, and upper bound.
  - `calculate_percentile_swgdrug`: Using SWGDRUG approach: lower bound.
  - `calculate_percentiles_alberink2010`: Using approach from Alberink _et al._ (2010): lower bound, best estimate, and upper bound
  - `calculate_percentiles_alberink2016`: Using approach from Alberink _et al._ (2016): lower bound, best estimate, and upper bound
  - `calculate_percentiles_bayesian`: Using novel Bayesian approach: lower bound, best estimate (median), upper bound.
- `simulation_steps`: Functions used to execute distinct parts of a simulation.
  - `process_cfg`: Process a configuration file (`file.yaml`) into input variables used for a simulation.
  - `run_single_simulation`: Do a single simulation run: generate ground truth, sample from it, apply methods.
  - `aggregate_simulation_results`: Summarise the results of a simulation, by aggregating over its runs.
  - `save_aggregated_results`: Save the aggregated results as csv-file including rounding, do some MCMC-checks.
- `utils`: Functions to perform various supporting tasks.
  - `lower_bound_hyper`: Determine lower bound for number of drug-containing units, using a hypergeometric distribution.
  - `draw_normal_mcmc_samples`: Draw samples from a normal distribution using MCMC-simulations.
  - `get_dirs_per_measurement_types`: Make lists of sub-folders with results, grouped by measurements types present.
  - `create_overview_results`: Make an overview table of all simulations results located in a specific folder.
