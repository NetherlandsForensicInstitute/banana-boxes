from pathlib import Path

import confidence
from confidence import Configuration
import numpy as np
import pandas as pd

from boundary_methods import (calculate_percentiles_bayesian,
                              calculate_percentiles_enfsi,
                              calculate_percentiles_alberink2010,
                              calculate_percentiles_alberink2016,
                              calculate_percentile_swgdrug,
                              calculate_percentiles_hyper_t)


def process_cfg(cfg: Configuration, output_folder: Path) -> tuple[np.random.Generator, list[str], dict, int, int, int]:
    """
    Process the configuration file into general input variables used for all simulations.

    Parameters
    ----------
    cfg: Configuration
        Configuration of the parameters
    output_folder: Path
        Folder for all output

    Returns
    -------
    tuple[np.random.Generator, list[str], dict, int, int, int]
        np.random.Generator: rng
            Random number generator used to generate ground truth data
        list[str]: methods
            Methods used to calculate best estimates and/or bounds for the total drug weight
        dict: mcmc_settings
            Settings used during MCMC-simulations
        int: presence_sample_count:
            Number of drug-presence samples
        int: weight_sample_count
            Number of unit weight samples
        int: concentration_sample_count
            Number of concentration samples
    """

    # use random seed to make random number generator
    if cfg.random_seed is None:
        # assumes that "%Y-%m-%d_%H-%M-%S" was used as format for the date+time at the end of the output path
        random_seed = int(output_folder.name.replace('-', '').replace('_', ''))
    else:
        random_seed = cfg.random_seed
    rng = np.random.default_rng(random_seed)
    # save config file
    output_folder.mkdir(parents=True, exist_ok=True)
    confidence.dumpf(cfg, f'{output_folder}/config.yaml')

    # Check on the number of boxes and the number of samples
    if any(x > cfg.box_count for x in [cfg.presence_sample_count, cfg.weight_sample_count, cfg.concentration_sample_count]):
        raise ValueError('Number of samples must be less than or equal to the number of boxes.')

    # Simulations; first everything that is the same for every simulation run
    # Define methods to use, based on measurement types included
    if cfg.measurement_types == 'weights_presences_concentrations':
        # Bayesian method only; the others do not support all three measurement types
        methods = ['bayesian']
    elif cfg.measurement_types == 'weights_presences':
        # A comparison between the bayesian method and frequentist methods that use weight and drug-presence
        # measurements only (so no concentration measurements). The relevant methods are: 'enfsi', 'alberink2010',
        # 'swgdrug' and 'hyper_t'.
        methods = ['bayesian', 'enfsi', 'alberink2010', 'swgdrug', 'hyper_t']
    elif cfg.measurement_types == 'weights_concentrations':
        # A comparison between the bayesian method and frequentist methods that use weight and concentration
        # measurements only (so no drug-presence measurements). The relevant method is: 'alberink2016'.
        # Weights of all units are required in this measurement type. Ground truth drug-presences of 0 are used to
        # change the relevant concentrations to a value of 0.
        methods = ['bayesian', 'alberink2016']
    else:
        raise ValueError('Unknown measurement_types specified.')

    # Set and adjust some simulation settings and measurement values, based on measurement type.
    mcmc_settings = dict(cfg.mcmc_settings)
    mcmc_settings['mcmc_rng'] = rng
    if cfg.measurement_types == 'weights_presences':
        mcmc_settings['mcmc_no_concentration'] = True
    if cfg.measurement_types == 'weights_concentrations':
        mcmc_settings['mcmc_no_weight'] = True

    # Get the sample counts
    presence_sample_count = cfg.presence_sample_count
    weight_sample_count = cfg.weight_sample_count
    concentration_sample_count = cfg.concentration_sample_count
    # Adjust some sample counts based on measurement type and methods that are part of it.
    if cfg.measurement_types == 'weights_presences':
        # Methods 'enfsi' and 'alberink2010' require the weight and drug-presence measurements are done on the same
        # samples; the smallest of both counts is used for both of them.
        weight_sample_count = np.min((weight_sample_count, presence_sample_count))
        presence_sample_count = np.min((weight_sample_count, presence_sample_count))
    if cfg.measurement_types == 'weights_concentrations':
        # Weights of all units are required for method 'alberink2016'.
        weight_sample_count = cfg.box_count
        # Concentration and drug-presence measurements are to be combined, use the smallest count for both.
        presence_sample_count = np.min((presence_sample_count, concentration_sample_count))
        concentration_sample_count = np.min((presence_sample_count, concentration_sample_count))

    return rng, methods, mcmc_settings, presence_sample_count, weight_sample_count, concentration_sample_count


def run_single_simulation(cfg: Configuration, output_folder: Path,
                          rng: np.random.Generator, methods: list[str], mcmc_settings: dict,
                          presence_sample_count: int, weight_sample_count: int,
                          concentration_sample_count: int) -> dict:
    """
    Run a single simulation: generate ground truth; sample from it; calculate best estimates and lower/upper bounds for
    the total drug weight, using multiple calculation methods.

    Parameters
    ----------
    cfg: Configuration
        Configuration of the parameters
    output_folder: Path
        Folder for all output
    rng: np.random.Generator
        Random number generator used to generate ground truth data
    methods: list[str]
        Methods used to calculate best estimates and/or bounds for the total drug weight
    mcmc_settings: dict
        Settings used during MCMC-simulations
    presence_sample_count: int
        Number of drug-presence samples
    weight_sample_count: int
        Number of unit weight samples
    concentration_sample_count: int
        Number of concentration samples

    Returns
    -------
    dict
        Collection of the relevant simulations run results.
    """

    # Simulate ground truth for presence of drugs, weight of the bottoms and drug concentrations.
    presence_simulation = rng.binomial(1, cfg.presence_real_p, cfg.box_count)
    weight_simulation = rng.normal(cfg.weight_real_mu, cfg.weight_real_sigma, cfg.box_count)
    if cfg.measurement_types == 'weights_presences':
        # No concentration measurements; set all concentration values at one (100%, full weight counts as drugs).
        concentration_simulation = np.ones(cfg.box_count)
    else:
        concentration_simulation = rng.normal(cfg.concentration_real_mu, cfg.concentration_real_sigma,
                                              cfg.box_count)
    # Calculate total weight in this simulation.
    total_weight_simulation = np.sum(presence_simulation * weight_simulation * concentration_simulation)

    # Draw samples from the simulated ground truth.
    presence_simulation_sample = presence_simulation[:presence_sample_count]
    weight_simulation_sample = weight_simulation[:weight_sample_count]
    if cfg.measurement_types == 'weights_presences':
        concentration_simulation_sample = concentration_simulation[:concentration_sample_count]
    else:
        repeated_concentration_simulation_sample = rng.normal(
            concentration_simulation[:concentration_sample_count],
            cfg.repeated_concentration_real_sigma,
            (cfg.repeated_concentration_sample_count,
             concentration_sample_count))
        # For methods that do not use repeated concentration samples, the mean of the repeats is used as the
        # concentration sample.
        concentration_simulation_sample = repeated_concentration_simulation_sample.mean(axis=0)

    # Initialise results; only include values that change every simulation
    result_dict = {
        'real weight': total_weight_simulation,
        'zeros': presence_sample_count - np.count_nonzero(presence_simulation_sample),
    }

    # Calculate drug weight percentiles using different methods; and add to the results of this simulation run
    if 'bayesian' in methods:
        # Calculate drug weight percentiles using the Bayesian method.
        percentages_bayesian, percentiles_bayesian, diagnostics = calculate_percentiles_bayesian(
            cfg.box_count, presence_simulation_sample, weight_simulation_sample, concentration_simulation_sample,
            cfg.alpha, output_folder, **mcmc_settings)
        result_dict.update({
            f'bayesian {percentages_bayesian[0]}% lower bound': percentiles_bayesian[0],
            'bayesian best estimate': percentiles_bayesian[1],
            f'bayesian {percentages_bayesian[2]}% upper bound': percentiles_bayesian[2],
            'bayesian error best estimate': percentiles_bayesian[1] - total_weight_simulation,
            'bayesian interval width': percentiles_bayesian[2] - percentiles_bayesian[0],
            'in bayesian two-sided interval': int((total_weight_simulation >= percentiles_bayesian[0]) and
                                                  (total_weight_simulation <= percentiles_bayesian[2])),
            'in bayesian one-sided interval': int(total_weight_simulation >= percentiles_bayesian[0])})

    if 'enfsi' in methods:
        # Results from the ENFSI-method
        percentages_enfsi, percentiles_enfsi = calculate_percentiles_enfsi(cfg.box_count,
                                                                           presence_simulation_sample,
                                                                           weight_simulation_sample,
                                                                           cfg.alpha)
        percentiles_enfsi = [i * concentration_simulation_sample[0] for i in percentiles_enfsi]
        result_dict.update({
            f'enfsi {percentages_enfsi[0]}% lower bound': percentiles_enfsi[0],
            'enfsi best estimate': percentiles_enfsi[1],
            f'enfsi {percentages_enfsi[2]}% upper bound': percentiles_enfsi[2],
            'enfsi error best estimate': percentiles_enfsi[1] - total_weight_simulation,
            'enfsi interval width': percentiles_enfsi[2] - percentiles_enfsi[0],
            'in enfsi two-sided interval': int((total_weight_simulation >= percentiles_enfsi[0]) and
                                               (total_weight_simulation <= percentiles_enfsi[2]))
            if not np.isnan(percentiles_enfsi[0]) else np.nan,
            'in enfsi one-sided interval': int(total_weight_simulation >= percentiles_enfsi[0])
            if not np.isnan(percentiles_enfsi[0]) else np.nan})

    if 'alberink2010' in methods:
        # Results from the Alberink2010-method
        percentages_alb2010, percentiles_alb2010 = calculate_percentiles_alberink2010(cfg.box_count,
                                                                                      presence_simulation_sample,
                                                                                      weight_simulation_sample,
                                                                                      cfg.alpha)
        percentiles_alb2010 = [i * concentration_simulation_sample[0] for i in percentiles_alb2010]
        result_dict.update({
            f'alberink2010 {percentages_alb2010[0]}% lower bound': percentiles_alb2010[0],
            'alberink2010 best estimate': percentiles_alb2010[1],
            f'alberink2010 {percentages_alb2010[2]}% upper bound': percentiles_alb2010[2],
            'alberink2010 error best estimate': percentiles_alb2010[1] - total_weight_simulation,
            'alberink2010 interval width': percentiles_alb2010[2] - percentiles_alb2010[0],
            'in alberink2010 two-sided interval': int((total_weight_simulation >= percentiles_alb2010[0]) and
                                                      (total_weight_simulation <= percentiles_alb2010[2]))
            if not np.isnan(percentiles_alb2010[0]) else np.nan,
            'in alberink2010 one-sided interval': int(total_weight_simulation >= percentiles_alb2010[0])
            if not np.isnan(percentiles_alb2010[0]) else np.nan})

    if 'swgdrug' in methods:
        # Results from the SWGDRUG-method
        # Run with alpha/2, such that the lower bound can be compared to two-sided methods.
        percentages_swgdrug, percentiles_swgdrug = calculate_percentile_swgdrug(cfg.box_count,
                                                                                presence_simulation_sample,
                                                                                weight_simulation_sample,
                                                                                cfg.alpha / 2)
        percentiles_swgdrug = [i * concentration_simulation_sample[0] for i in percentiles_swgdrug]
        result_dict.update({
            f'swgdrug {percentages_swgdrug[0]}% lower bound': percentiles_swgdrug[0],
            'in swgdrug one-sided interval': int(total_weight_simulation >= percentiles_swgdrug[0])
            if not np.isnan(percentiles_swgdrug[0]) else np.nan})

    if 'hyper_t' in methods:
        # Results from the hyper_t-method
        percentages_hypert, percentiles_hypert, best_estimate_hypert = calculate_percentiles_hyper_t(
            cfg.box_count, presence_simulation_sample, weight_simulation_sample, cfg.alpha)
        percentiles_hypert = [i * concentration_simulation_sample[0] for i in percentiles_hypert]
        best_estimate_hypert = best_estimate_hypert * concentration_simulation_sample[0]
        result_dict.update({
            f'hyper_t {percentages_hypert[0]}% lower bound': percentiles_hypert[0],
            'hyper_t best estimate': best_estimate_hypert,
            f'hyper_t {percentages_hypert[1]}% upper bound': percentiles_hypert[1],
            'hyper_t error best estimate': best_estimate_hypert - total_weight_simulation,
            'hyper_t interval width': percentiles_hypert[1] - percentiles_hypert[0],
            'in hyper_t two-sided interval': int((total_weight_simulation >= percentiles_hypert[0]) and
                                                 (total_weight_simulation <= percentiles_hypert[1]))
            if not np.isnan(percentiles_hypert[0]) else np.nan,
            'in hyper_t one-sided interval': int(total_weight_simulation >= percentiles_hypert[0])
            if not np.isnan(percentiles_hypert[0]) else np.nan})

    if 'alberink2016' in methods:
        # Results from the Alberink2016-method
        repeated_concentration_simulation_sample = repeated_concentration_simulation_sample * presence_simulation_sample
        percentages_alb_spr, percentiles_alb_spr = calculate_percentiles_alberink2016(cfg.box_count,
                                                                                         weight_simulation_sample,
                                                                                         repeated_concentration_simulation_sample,
                                                                                         cfg.alpha)
        result_dict.update({
            f'alberink2016 {percentages_alb_spr[0]}% lower bound': percentiles_alb_spr[0],
            'alberink2016 best estimate': percentiles_alb_spr[1],
            f'alberink2016 {percentages_alb_spr[2]}% upper bound': percentiles_alb_spr[2],
            'alberink2016 error best estimate': percentiles_alb_spr[1] - total_weight_simulation,
            'alberink2016 interval width': percentiles_alb_spr[2] - percentiles_alb_spr[0],
            'in alberink2016 two-sided interval': int((total_weight_simulation >= percentiles_alb_spr[0]) and
                                                         (total_weight_simulation <= percentiles_alb_spr[2])),
            'in alberink2016 one-sided interval': int(total_weight_simulation >= percentiles_alb_spr[0])
            if not np.isnan(percentiles_alb_spr[0]) else np.nan})

    # Add MCMC-diagnostics to output
    if 'bayesian' in methods:
        extreme_fractions = diagnostics[['fraction_extreme_upper','fraction_extreme_lower']]
        max_fraction = extreme_fractions.max(axis=None)
        if max_fraction > 0:
            max_index = extreme_fractions.max(axis=1) == max_fraction
            max_type = diagnostics['sample_type'].values[max_index][0]
            max_parameter = extreme_fractions.index.values[max_index][0]
            max_side = extreme_fractions.columns.values[extreme_fractions.max(axis=0) == max_fraction][0][-5:]
            max_bound = f'{max_type} {max_parameter} {max_side}'
        else:
            max_bound = ''
        result_dict.update({
            'MCMC max R_hat': diagnostics['r_hat'].max(),
            'MCMC min ESS-bulk': diagnostics['ess_bulk'].min(),
            'MCMC min ESS-tail': diagnostics['ess_tail'].min(),
            'MCMC max fraction extreme values': max_fraction,
            'MCMC max fraction extreme bound': max_bound,
        })

    return result_dict


def aggregate_simulation_results(cfg: Configuration, method: str, results_df: pd.DataFrame, presence_sample_count: int,
                                 weight_sample_count: int, concentration_sample_count: int) -> dict:
    """
    Aggregated the simulation results (mean, std, or quantile) for all methods in a single overview; a method per row.

    Parameters
    ----------
    cfg: Configuration
        Configuration of the parameters
    method: str
        Method used to calculate best estimates and/or bounds for the total drug weight
    results_df: pd.DataFrame
        Dataframe with results from all simulation runs
    presence_sample_count: int
        Number of drug-presence samples
    weight_sample_count: int
        Number of unit weight samples
    concentration_sample_count: int
        Number of concentration samples

    Returns
    -------
    dict
        Collection of aggregated simulations results.
    """

    aggregated_results_dict = {
        'measurements': cfg.measurement_types,
        'method': method,
        'alpha': cfg.alpha,
        'n': cfg.box_count,
        'm_w': weight_sample_count,
        'm_p': presence_sample_count,
        'm_c': concentration_sample_count,
        'presence_real': cfg.presence_real_p,
        'mean_real_weight': results_df.get('real weight').mean(),
        'mean_best_estimate': results_df.get(f'{method} best estimate', np.nan).mean()
        if f'{method} best estimate' in results_df else np.nan,
        'std_best_estimate': results_df.get(f'{method} best estimate', np.nan).std()
        if f'{method} best estimate' in results_df else np.nan,
        'mean_deviation_best_estimate': results_df.get(f'{method} error best estimate', np.nan).mean()
        if f'{method} error best estimate' in results_df else np.nan,
        'std_mean_deviation_best_estimate': results_df.get(f'{method} error best estimate', np.nan).std()
        if f'{method} error best estimate' in results_df else np.nan,
        'mean_width': results_df.get(f'{method} interval width', np.nan).mean()
        if f'{method} interval width' in results_df else np.nan,
        'std_width': results_df.get(f'{method} interval width', np.nan).std()
        if f'{method} interval width' in results_df else np.nan,
        'mean_lower_bound': results_df.get(f'{method} {100 * cfg.alpha / 2}% lower bound', np.nan).mean()
        if f'{method} {100 * cfg.alpha / 2}% lower bound' in results_df else np.nan,
        'std_lower_bound': results_df.get(f'{method} {100 * cfg.alpha / 2}% lower bound', np.nan).std()
        if f'{method} {100 * cfg.alpha / 2}% lower bound' in results_df else np.nan,
        'mean_upper_bound': results_df.get(f'{method} {100 - 100 * cfg.alpha / 2}% upper bound', np.nan).mean()
        if f'{method} {100 - 100 * cfg.alpha / 2}% upper bound' in results_df else np.nan,
        'std_upper_bound': results_df.get(f'{method} {100 - 100 * cfg.alpha / 2}% upper bound', np.nan).std()
        if f'{method} {100 - 100 * cfg.alpha / 2}% upper bound' in results_df else np.nan,
        'relative_bias': (results_df.get(f'{method} error best estimate', np.nan) / results_df.get('real weight')).mean()
        if f'{method} error best estimate' in results_df else np.nan,
        'relative_bw': (results_df.get(f'{method} interval width', np.nan) / results_df.get('real weight')).mean()
        if f'{method} interval width' in results_df else np.nan,
        'coverage, two-sided': results_df[f'in {method} two-sided interval'].mean()
        if f'in {method} two-sided interval' in results_df else np.nan,
        'relative_lb': (results_df.get(f'{method} {100 * cfg.alpha / 2}% lower bound', np.nan) / results_df.get('real weight')).mean()
        if f'{method} {100 * cfg.alpha / 2}% lower bound' in results_df else np.nan,
        'coverage, one-sided': results_df[f'in {method} one-sided interval'].mean(),
        'coverage count': results_df[f'in {method} one-sided interval'].dropna().count(),
        'experiment': cfg.experiment,
        'mean_abs_deviation_best_estimate': results_df.get(f'{method} error best estimate',
                                                           np.nan).abs().mean()
        if f'{method} error best estimate' in results_df else np.nan,
        'std_abs_deviation_best_estimate': results_df.get(f'{method} error best estimate',
                                                          np.nan).abs().std()
        if f'{method} error best estimate' in results_df else np.nan,
        '95%_upper_bound_MCMC_max_Rhat': results_df.get('MCMC max R_hat').quantile(0.95),
        '5%_lower_bound_MCMC_min_ESSbulk': results_df.get('MCMC min ESS-bulk').quantile(0.05),
        '5%_lower_bound_MCMC_min_ESStail': results_df.get('MCMC min ESS-tail').quantile(0.05),
        '95%_pctl_max_extreme_values': results_df.get('MCMC max fraction extreme values').quantile(0.95),
    }

    return aggregated_results_dict


def save_aggregated_results(aggregated_results_list: list[dict], output_folder: Path) -> None:
    """
    Save the aggregated simulation results as csv file, and check some diagnostic values.

    Parameters
    ----------
    aggregated_results_list: list[dict]
        List with collections of aggregated simulations results.
    output_folder: Path
        Folder for all output
    """

    # Save aggregated results, number of digits depends on column
    aggregated_results_df = pd.DataFrame(aggregated_results_list).drop_duplicates()
    aggregated_results_df = aggregated_results_df.round({'mean_real_weight': 2,
           '95%_upper_bound_MCMC_max_Rhat': 3, '5%_lower_bound_MCMC_min_ESSbulk': 0,
           '5%_lower_bound_MCMC_min_ESStail': 0, '95%_pctl_max_extreme_values': 3,
           'mean_abs_deviation_best_estimate': 2,
           'std_abs_deviation_best_estimate': 2, 'mean_deviation_best_estimate': 2,
           'std_mean_deviation_best_estimate': 2, 'mean_width': 2, 'std_width': 2,
           'relative_bias': 3, 'relative_bw': 3, 'relative_lb': 3,
           'mean_lower_bound': 2, 'std_lower_bound': 2, 'coverage': 3, 'mean_upper_bound': 2,
           'std_upper_bound': 2, 'mean_best_estimate': 2, 'std_best_estimate': 2})
    aggregated_results_df.to_csv(output_folder / f'simulation_overview.csv', index=False)

    # Give warnings if certain diagnostics exceed their specified limits
    if aggregated_results_df['95%_upper_bound_MCMC_max_Rhat'].max() > 1.01:
        print('WARNING: at least one 95%-percentile of the maximum R_hat exceeds 1.01.')
    if aggregated_results_df['5%_lower_bound_MCMC_min_ESSbulk'].min() < 400:
        print('WARNING: at least one 5%-percentile of the minimum ESS-bulk is below 400.')
    if aggregated_results_df['5%_lower_bound_MCMC_min_ESStail'].min() < 400:
        print('WARNING: at least one 5%-percentile of the minimum ESS-tail is below 400.')
    if aggregated_results_df['95%_pctl_max_extreme_values'].max() > 0.01:
        print('WARNING: at least one 95%-percentile of the extreme values exceeds 0.01.')


def merge_simulation_results(results_dir: Path):
    """
    Merge the simulations present in multiple csv files into a single csv file with accompanying overview file.
    """

    # Find cfg's with exactly the same contents
    all_configs = results_dir.rglob("config.yaml")
    same_cfgs = {}
    for config_path in all_configs:
        new_cfg = True
        new_cfg_name = config_path.parent
        cfg = confidence.loadf(config_path)
        for cfg_name, cfg_dict in same_cfgs.items():
            if cfg_dict['cfg'] == cfg:
                cfg_dict['name'].append(new_cfg_name)
                new_cfg = False
        if new_cfg:
            same_cfgs[new_cfg_name] = {'cfg': cfg, 'name': [new_cfg_name]}
    for cfg_name, cfg_dict in same_cfgs.items():
        # Read the simulation csv's and merge them into a single csv
        cfg = cfg_dict['cfg']
        results_df = pd.DataFrame([])
        file_name = "simulation_results-" + cfg.measurement_types + ".csv"
        for simulations in cfg_dict['name']:
            results_df = pd.concat([results_df, pd.read_csv(simulations / file_name)])
        # Save it to a new folder, along with the overview csv file
        output_folder = cfg_name.parent.parent / 'merged' / cfg_name.parent.name
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_folder / file_name, index=False)
        # Gather the aggregated results (mean, std, or quantile) for all methods in a single overview; a method per row.
        _, methods, _, presence_sample_count, weight_sample_count, concentration_sample_count = process_cfg(cfg, output_folder)
        aggregated_results_list = []
        for method in methods:
            aggregated_results_dict = aggregate_simulation_results(cfg, method, results_df, presence_sample_count,
                                                                   weight_sample_count, concentration_sample_count)
            aggregated_results_list.append(aggregated_results_dict)
        # Convert results overview to dataframe, save to csv, check some diagnostics
        save_aggregated_results(aggregated_results_list, output_folder)
