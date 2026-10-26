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
    assert np.allclose(percentiles, [6.19161259, 7.37947384, 8.19697207])


def test_bayesian_box_count():
    # Use a smaller total number of units
    box_count = 50
    mcmc_settings['mcmc_rng'] = np.random.default_rng(42)
    percentages, percentiles, diagnostics = calculate_percentiles_bayesian(
        box_count, presence_sample, weight_sample, concentration_sample, alpha, data_dir, **mcmc_settings)
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(percentiles, [3.17372699, 3.69821854, 4.07916586])


def test_bayesian_presences():
    # Use presences with more zero's
    presence_sample = np.concat((all_presences, np.zeros(5)))
    mcmc_settings['mcmc_rng'] = np.random.default_rng(42)
    percentages, percentiles, diagnostics = calculate_percentiles_bayesian(
        box_count, presence_sample, weight_sample, concentration_sample, alpha, data_dir, **mcmc_settings)
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(percentiles, [4.61988202, 5.96103225, 7.04079353])


def test_bayesian_weights():
    # Use more weight samples
    weight_sample = np.concat((all_weights, all_weights[:3]))
    mcmc_settings['mcmc_rng'] = np.random.default_rng(42)
    percentages, percentiles, diagnostics = calculate_percentiles_bayesian(
        box_count, presence_sample, weight_sample, concentration_sample, alpha, data_dir, **mcmc_settings)
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(percentiles, [6.26459704, 7.43104685, 8.17879252])


def test_bayesian_concentrations():
    # Use less concentration samples
    concentration_sample = all_concentrations[:5]
    mcmc_settings['mcmc_rng'] = np.random.default_rng(42)
    percentages, percentiles, diagnostics = calculate_percentiles_bayesian(
        box_count, presence_sample, weight_sample, concentration_sample, alpha, data_dir, **mcmc_settings)
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(percentiles, [5.98699971, 7.57265971, 8.98395738])


def test_bayesian_alpha():
    # Use different alpha value
    alpha = 0.1
    mcmc_settings['mcmc_rng'] = np.random.default_rng(42)
    percentages, percentiles, diagnostics = calculate_percentiles_bayesian(
        box_count, presence_sample, weight_sample, concentration_sample, alpha, data_dir, **mcmc_settings)
    assert np.allclose(percentages, [5, 50.0, 95])
    assert np.allclose(percentiles, [6.43540828, 7.37947384, 8.03754582])


def test_bayesian_rng():
    # Use random number generator with different seed
    mcmc_settings['mcmc_rng'] = np.random.default_rng(261)
    percentages, percentiles, diagnostics = calculate_percentiles_bayesian(
        box_count, presence_sample, weight_sample, concentration_sample, alpha, data_dir, **mcmc_settings)
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(percentiles, [6.18985075, 7.36882673, 8.16407394])


pytest.main()
