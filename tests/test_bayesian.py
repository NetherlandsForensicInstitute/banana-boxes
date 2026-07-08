import confidence
import pandas as pd
from pathlib import Path
import pytest
import numpy as np

from boundary_methods import calculate_percentiles_bayesian


# Test cases are based on (a subset of) samples drawn from a ground truth population generated using the default
# setting as defined in config.yaml. The sample counts used are: 5 weights, 10 concentrations, and 20 presences.
# The settings that are used for the MCMC-simulations are the same as the defaults in config.yaml.
# To make simulations reproducible, a fixed seed is used within each test for the random number generator.

# Define default input values
data_dir = Path(__file__).parent / 'data'
all_data = pd.read_csv(data_dir / 'bayesian.csv')
all_weights = all_data['weight'].dropna().to_numpy()
weight_sample = all_weights
all_concentrations = all_data['concentration'].dropna().to_numpy()
concentration_sample = all_concentrations
all_presences = all_data['presence'].dropna().to_numpy()
presence_sample = all_presences
box_count = 100
alpha = 0.05
cfg = confidence.loadf(data_dir / 'config_bayesian.yaml')
mcmc_settings = dict(cfg.mcmc_settings)


def test_bayesian_defaults():
    # Use original input values
    mcmc_settings['mcmc_rng'] = np.random.default_rng(42)
    percentages, percentiles, diagnostics = calculate_percentiles_bayesian(
        box_count, presence_sample, weight_sample, concentration_sample, alpha, data_dir, **mcmc_settings)
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(percentiles, [6.17738878, 7.37246005, 8.17048113])


def test_bayesian_box_count():
    # Use a smaller total number of units
    box_count = 50
    mcmc_settings['mcmc_rng'] = np.random.default_rng(42)
    percentages, percentiles, diagnostics = calculate_percentiles_bayesian(
        box_count, presence_sample, weight_sample, concentration_sample, alpha, data_dir, **mcmc_settings)
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(percentiles, [3.15768799, 3.69706135, 4.06244103])


def test_bayesian_presences():
    # Use presences with more zero's
    presence_sample = np.concat((all_presences, np.zeros(5)))
    mcmc_settings['mcmc_rng'] = np.random.default_rng(42)
    percentages, percentiles, diagnostics = calculate_percentiles_bayesian(
        box_count, presence_sample, weight_sample, concentration_sample, alpha, data_dir, **mcmc_settings)
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(percentiles, [4.61888477, 5.96271284, 7.01102698])


def test_bayesian_weights():
    # Use more weight samples
    weight_sample = np.concat((all_weights, all_weights[:3]))
    mcmc_settings['mcmc_rng'] = np.random.default_rng(42)
    percentages, percentiles, diagnostics = calculate_percentiles_bayesian(
        box_count, presence_sample, weight_sample, concentration_sample, alpha, data_dir, **mcmc_settings)
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(percentiles, [6.29144567, 7.42013933, 8.16435755])


def test_bayesian_concentrations():
    # Use less concentration samples
    concentration_sample = all_concentrations[:5]
    mcmc_settings['mcmc_rng'] = np.random.default_rng(42)
    percentages, percentiles, diagnostics = calculate_percentiles_bayesian(
        box_count, presence_sample, weight_sample, concentration_sample, alpha, data_dir, **mcmc_settings)
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(percentiles, [6.27006304, 7.5890561 , 8.56941953])


def test_bayesian_alpha():
    # Use different alpha value
    alpha = 0.1
    mcmc_settings['mcmc_rng'] = np.random.default_rng(42)
    percentages, percentiles, diagnostics = calculate_percentiles_bayesian(
        box_count, presence_sample, weight_sample, concentration_sample, alpha, data_dir, **mcmc_settings)
    assert np.allclose(percentages, [5, 50.0, 95])
    assert np.allclose(percentiles, [6.38790779, 7.37246005, 8.04786843])


def test_bayesian_rng():
    # Use random number generator with different seed
    mcmc_settings['mcmc_rng'] = np.random.default_rng(261)
    percentages, percentiles, diagnostics = calculate_percentiles_bayesian(
        box_count, presence_sample, weight_sample, concentration_sample, alpha, data_dir, **mcmc_settings)
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(percentiles, [6.21380296, 7.38847838, 8.15848056])


pytest.main()
