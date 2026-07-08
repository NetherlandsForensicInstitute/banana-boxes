import os
import warnings
from collections import defaultdict
from pathlib import Path

import arviz as az
import confidence
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import re
from scipy.stats import hypergeom
import yaml


def lower_bound_hyper(total_count: int,
                      sample_count: int,
                      sample_positive_count: int,
                      alpha: float) -> float:
    """
    Calculates a lower bound for the total number of positive units. This is done by doing hypothesis tests, where
    the hypothesis is tested whether k>=sample_positive_count is plausible for a certain K. Plausible means that the
    probability is greater than alpha. Returns the largest lower bound for K for which the resulting interval is
    equal to or larger than 100*(1 - alpha).

    Parameters
    ----------
    total_count: int
        The total number of units.
    sample_count: int
        The size of the sample.
    sample_positive_count: int
        The number of positive units in the sample.
    alpha: float
        The desired value for alpha.

    Returns
    ----------
    lower_bound: float
        Lower bound for the total number of positive units. The lower bound is the lower bound of a one-sided
        100*(1-alpha)-confidence interval.
    """

    # Calculate the hypergeometric cumulative distribution for positive units 0 to total_count.
    # P(k>=x) >= alpha is equivalent with P(k<=x-1) <= 1-alpha, therefore the input is k = sample_positive_count - 1.
    cumulative_distribution_hyp = np.array([hypergeom.cdf(k=sample_positive_count - 1, M=total_count,
                                                          n=total_positive_count, N=sample_count)
                                            for total_positive_count in range(0, total_count + 1)])

    # Determine where the cumulative probability is smaller than 1 - alpha.
    lower_bound = int(np.min(np.where(cumulative_distribution_hyp <= 1 - alpha)))

    return lower_bound


def draw_normal_mcmc_samples(observed_data: np.ndarray,
                             mu_lower_bound: float,
                             mu_upper_bound: float,
                             sigma_lower_bound: float,
                             sigma_upper_bound: float,
                             sample_type: str,
                             output_folder: Path,
                             mcmc_rng: np.random.Generator = np.random.default_rng(),
                             mcmc_draw_count: int = 1000,
                             mcmc_chain_count: int = 4,
                             mcmc_tune_count: int = 1000,
                             mcmc_save_posteriors: bool = False) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Perform MCMC simulation on observed data, resulting in draws from the posterior distribution(s) of the
    underlying parameters mu and sigma. The observed data is assumed to be normally distributed with parameters
    mu and sigma. The priors for mu and sigma are uniformly distributed with the provided lower and upper bounds.

    Parameters
    ----------
    observed_data: np.ndarray
        Observed data to base the MCMC analysis on.
    mu_lower_bound: float
        Lower bound of the prior distribution for mu.
    mu_upper_bound: float
        Upper bound of the prior distribution for mu.
    sigma_lower_bound: float
        Lower bound of the prior distribution for sigma.
    sigma_upper_bound: float
        Upper bound of the prior distribution for sigma.
    sample_type: str
        Type of samples that were taken, used as an extension to the filename of the posterior distribution
        of the MCMC simulation.
    output_folder: Path
        The path to the output folder.
    mcmc_rng: np.random.Generator
        Numpy random number generator used during MCMC simulations.
    mcmc_draw_count: int = 1000
        The number of samples to draw. Defaults to 1000. The number of tuned samples are discarded
        by default.
    mcmc_chain_count: int = 4
        The number of chains to sample, defaults to 4.
    mcmc_tune_count: int = 1000
        Number of iterations to tune, defaults to 1000.
    mcmc_save_posteriors: bool
        Whether to save a plot of the posterior distributions of the MCMC simulation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, pd.DataFrame]
        np.ndarray
            Array containing draws from the posterior distribution of mu of the MCMC simulation.
        np.ndarray
            Array containing draws from the posterior distribution of sigma of the MCMC simulation.
        pd.DataFrame
            A Dataframe with MCMC-diagnostics (such as R_hat and ESS) for the two model parameters (mu and sigma).
    """

    normal_model = pm.Model()
    with (normal_model):
        # Train the model using the observed data
        prior_mu = pm.Uniform('mu', mu_lower_bound, mu_upper_bound)
        prior_sigma = pm.Uniform('sigma', sigma_lower_bound, sigma_upper_bound)
        pm.Normal('Y', mu=prior_mu, sigma=prior_sigma, observed=observed_data)
        # Sample from the posterior distribution, return convergence and (if indicated) save plot
        normal_trace = pm.sample(mcmc_draw_count, cores=1, chains=mcmc_chain_count, tune=mcmc_tune_count,
                                 random_seed=mcmc_rng, progressbar=False)
        diagnostics = az.summary(normal_trace, var_names=['mu', 'sigma'], kind='diagnostics', round_to=4)
        diagnostics['sample_type'] = sample_type

        trace_count = mcmc_draw_count * mcmc_chain_count
        trace_mu = normal_trace.posterior['mu'].values.flatten()
        lowest_1_pct_mu = mu_lower_bound + 0.01 * (mu_upper_bound - mu_lower_bound)
        fraction_lowest_1_pct_mu = np.count_nonzero(trace_mu < lowest_1_pct_mu)/trace_count
        highest_1_pct_mu = mu_upper_bound - 0.01 * (mu_upper_bound - mu_lower_bound)
        fraction_highest_1_pct_mu = np.count_nonzero(trace_mu > highest_1_pct_mu)/trace_count

        trace_sigma = normal_trace.posterior['sigma'].values.flatten()
        lowest_1_pct_sigma = sigma_lower_bound + 0.01 * (sigma_upper_bound - sigma_lower_bound)
        fraction_lowest_1_pct_sigma = np.count_nonzero(trace_sigma < lowest_1_pct_sigma)/trace_count
        highest_1_pct_sigma = sigma_upper_bound - 0.01 * (sigma_upper_bound - sigma_lower_bound)
        fraction_highest_1_pct_sigma = np.count_nonzero(trace_sigma > highest_1_pct_sigma)/trace_count

        diagnostics['fraction_extreme_upper'] = np.nan
        diagnostics.loc['mu', 'fraction_extreme_upper'] = fraction_highest_1_pct_mu
        diagnostics.loc['sigma', 'fraction_extreme_upper'] = fraction_highest_1_pct_sigma

        diagnostics['fraction_extreme_lower'] = np.nan
        diagnostics.loc['mu', 'fraction_extreme_lower'] = fraction_lowest_1_pct_mu
        diagnostics.loc['sigma', 'fraction_extreme_lower'] = fraction_lowest_1_pct_sigma

        if mcmc_save_posteriors:
            print(diagnostics['r_hat'])
            axes = az.plot_trace(normal_trace)
            plt.savefig(os.path.join(output_folder, "posterior_normal_" + sample_type))
            plt.close()

        del normal_model
        del normal_trace
        gc.collect()

    return trace_mu, trace_sigma, diagnostics


def get_dirs_per_measurement_types(results_dir: Path) -> defaultdict[str, list]:
    """
    Retrieves the paths of the directories in the root_dir containing the results of the simulation per
    measurement type. This is necessary for getting an overview of all results in the results_dir.


    Parameters
    ----------
    results_dir: Path
        Path to the root directory of the result folders to be searched.

    Returns
    -------
    defaultdict[str, list]
        Defaultdict containing a list of paths to the result folders for each measurement type.
    """
    dirs_per_measurement_types = defaultdict(list)
    for config_path in results_dir.rglob("config.yaml"):
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)

            measurement_types = data.get("measurement_types")
            if measurement_types is not None:
                dirs_per_measurement_types[measurement_types].append(config_path.parent)

        except Exception as e:
            print(f"Fout in {config_path}: {e}")

    return dirs_per_measurement_types


def create_overview_results(results_dir: str) -> None:
    """
    Gather all results in folder results_dir and make one overview in the root of that folder.

    Parameters:
    ----------
    results_dir: str
        Folder with all the results that have to be summarized. The overview is saved in the root of that folder.

    Return:
    ----------
    None
    """

    dirs_per_measurement_types = get_dirs_per_measurement_types(Path(results_dir))

    df_list = []
    for measurement_types in dirs_per_measurement_types.keys():
        experiments = []  # list of the experiments that are being done for this value of measurement_types

        for subdir in dirs_per_measurement_types[measurement_types]:
            overview_path = subdir / "simulation_overview.csv"
            if overview_path.exists():
                df = pd.read_csv(overview_path)
                df_list.append(df)
                if df['experiment'][0] == 'default':
                    if 'df_default' in locals():
                        warnings.warn(f'More than one default experiment for measurement_types {measurement_types}.')
                    df_default = df.copy()
                    config_default = confidence.loadf(os.path.join(subdir, 'config.yaml'))
                else:
                    experiments.append(re.search(r'([a-zA-Z_]+)_', df['experiment'][0]).group(1))
        # If a default experiment exists for this measurement_types, add the results for all default values in such a
        # way they can easily be found. It is assumed that only a single default is present per measurement_types
        # in the current results folder.
        if 'df_default' in locals():
            for experiment in set(experiments):
                df_default_experiment = df_default.copy()
                # Get default parameter value for the parameter varied in the experiments
                default_value = config_default[experiment]
                df_default_experiment['experiment'].values[:] = f'{experiment}_{default_value}'
                df_list.append(df_default_experiment)
            del df_default

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.sort_values(by=['measurements', 'method', 'experiment'], inplace=True)
    combined_df.to_csv(results_dir + '/overview_all_results.csv')
