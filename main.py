import argparse
import os
from pathlib import Path

import confidence
from datetime import datetime
import pandas as pd
from confidence import Configuration
from tqdm import tqdm
import logging

from simulation_steps import process_cfg, run_single_simulation, aggregate_simulation_results, save_aggregated_results
from utils import create_overview_results


def run_simulations(cfg: Configuration, output_folder: str) -> None:
    """
    Perform simulation based on the configuration.

    Parameters
    ----------
    cfg: Configuration
        Configuration of the parameters
    output_folder: str
        Folder for all output
    """

    # suppress logging output for pymc
    logger = logging.getLogger("pymc.sampling")
    logger.propagate = False

    # Process cfg to general simulations inputs.
    output_folder = Path(output_folder)
    rng, methods, mcmc_settings, presence_sample_count, weight_sample_count, concentration_sample_count = (
        process_cfg(cfg, output_folder))

    # Use simulation inputs to execute many simulation runs.
    results_list = []
    for simulation_index in tqdm(range(cfg.simulation_count)):
        # Create and fill a dict with results from this specific simulation run.
        result_dict = run_single_simulation(cfg, output_folder, rng, methods, mcmc_settings,
                                            presence_sample_count, weight_sample_count, concentration_sample_count)
        # Add the results from this simulation run to the results of all simulation runs.
        loop_dict = {'loop': simulation_index}
        results_list.append(loop_dict | result_dict)

    # Save all detailed results in a csv-file; a row per simulation run.
    results_df = pd.DataFrame(results_list)
    results_df_save = results_df.round(2)
    file_name = 'simulation_results-' + cfg.measurement_types + '.csv'
    results_df_save.to_csv(output_folder / file_name, index=False)

    # Gather the aggregated results (mean, std, or quantile) for all methods in a single overview; a method per row.
    aggregated_results_list = []
    for method in methods:
        aggregated_results_dict = aggregate_simulation_results(cfg, method, results_df, presence_sample_count,
                                                               weight_sample_count, concentration_sample_count)
        aggregated_results_list.append(aggregated_results_dict)

    # Convert results overview to dataframe, save to csv, check some diagnostics
    save_aggregated_results(aggregated_results_list, output_folder)

    print('finished!')


if __name__ == '__main__':
    config = confidence.loadf('input_configurations/default_case.yaml',
                              'input_configurations/local_adjustments.yaml')

    parser = argparse.ArgumentParser()
    parser.add_argument('-no_default_run',
                        help='Do not run default experiment with default config settings',
                        action='store_true')
    parser.add_argument('-run_sensitivity_analysis',
                        help='Run experiments with all yaml files in the sensitivity_analysis folder',
                        action='store_true')
    parser.add_argument('-run_method_comparison',
                        help='Run experiments with all yaml files in the method_comparison folder',
                        action='store_true')
    parser.add_argument('-run_all_yaml_files_in_folder',
                        help='Run experiments with all yaml files in given folder_yaml_files',
                        action='store_true')
    parser.add_argument('-folder_yaml_files',
                        help='Folder containing yaml files, used with parameter run_all_yaml_files_in_folder',
                        default='input_configurations/method_comparison')
    parser.add_argument('-create_overview_results',
                        help='Create overview results of all aggregated results in results_dir',
                        action='store_true')
    parser.add_argument('-results_dir',
                        help='Directory where result folders are stored',
                        default='results')
    parser.add_argument('-run_single_yaml',
                        help='Run experiment with configuration of one yaml file. Specify the path to the yaml'
                             'with argument -yaml_file',
                        action='store_true')
    parser.add_argument('-yaml_file',
                        help='Yaml file to run when running a single yaml file')
    args = parser.parse_args()

    output_folder_base = f'{config.output_folder}/{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}'

    if config.clear_cache:
        os.system('pytensor-cache purge')

    if not args.no_default_run:
        # Run the simulation with the default configuration
        output_folder_exp = f'{output_folder_base}/{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}'
        run_simulations(config, output_folder_exp)

    if args.run_sensitivity_analysis:
        for yaml in os.listdir('input_configurations/sensitivity_analysis'):
            config = confidence.loadf('input_configurations/default_case.yaml',
                                      f'input_configurations/sensitivity_analysis/{yaml}',
                                      'input_configurations/local_adjustments.yaml')
            output_folder_exp = f'{output_folder_base}/{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}'
            run_simulations(config, output_folder_exp)
            args.results_dir = output_folder_base

    if args.run_method_comparison:
        for yaml in os.listdir('input_configurations/method_comparison'):
            config = confidence.loadf('input_configurations/default_case.yaml',
                                      f'input_configurations/method_comparison/{yaml}',
                                      'input_configurations/local_adjustments.yaml')
            output_folder_exp = f'{output_folder_base}/{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}'
            run_simulations(config, output_folder_exp)
            args.results_dir = output_folder_base

    if args.run_all_yaml_files_in_folder:
        for yaml in os.listdir(args.folder_yaml_files):
            config = confidence.loadf('input_configurations/default_case.yaml',
                                      f'{args.folder_yaml_files}/{yaml}',
                                      'input_configurations/local_adjustments.yaml')
            output_folder_exp = f'{output_folder_base}/{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}'
            run_simulations(config, output_folder_exp)
            args.results_dir = output_folder_base

    if args.create_overview_results:
        create_overview_results(results_dir=args.results_dir)

    if args.run_single_yaml:
        config = confidence.loadf('input_configurations/default_case.yaml',
                                  f'{args.yaml_file}',
                                  'input_configurations/local_adjustments.yaml')
        output_folder_exp = f'{output_folder_base}/{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}'
        run_simulations(config, output_folder_exp)
        args.results_dir = output_folder_base

