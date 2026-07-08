import pandas as pd
from pathlib import Path
import pytest
import numpy as np

from boundary_methods import calculate_percentiles_alberink2016


data_dir = Path(__file__).parent / 'data'
data = pd.read_csv(data_dir / 'alberink2016_group1.csv')
weight_sample = data['weight'].to_numpy()
box_count = len(weight_sample)
repeated_concentration_sample = data[['fraction1', 'fraction2']].dropna().to_numpy().transpose()
alpha = 0.05

# Test cases are based on group 1 in Example 2 of the article 'Quantifying Uncertainty in Estimations of the Total
# Weight of Drugs in Groups of Complex Matrices Using the Welch–Satterthwaite Equation', by Alberink et al. (2016).
# The reference values of the default case are verified against the Matlab implementation that was used for the
# publication.


def test_alberink2016_defaults():
    # Use original weight samples, concentration samples, and alpha
    percentages, percentiles = calculate_percentiles_alberink2016(
        box_count, weight_sample, repeated_concentration_sample, alpha)
    assert np.allclose(percentages, [2.5, 50, 97.5])
    assert np.allclose(percentiles, [0.711682, 1.077062, 1.442441])


def test_alberink2016_weights():
    # Use different weight samples
    weight_sample = data['weight'].to_numpy()[:30]
    box_count = len(weight_sample)
    percentages, percentiles = calculate_percentiles_alberink2016(
        box_count, weight_sample, repeated_concentration_sample, alpha)
    assert np.allclose(percentages, [2.5, 50, 97.5])
    assert np.allclose(percentiles, [0.578069, 0.868196, 1.158322])


def test_alberink2016_concentrations():
    # Use different concentration samples
    repeated_concentration_sample = data[['fraction1', 'fraction2']].dropna().to_numpy().transpose()[:, :5]
    percentages, percentiles = calculate_percentiles_alberink2016(
        box_count, weight_sample, repeated_concentration_sample, alpha)
    assert np.allclose(percentages, [2.5, 50, 97.5])
    assert np.allclose(percentiles, [0.690833, 1.236246, 1.781659])


def test_alberink2016_presences():
    # Include a single zero in the concentration samples
    presence_sample = np.ones(repeated_concentration_sample.shape[1])
    presence_sample[0] = 0
    percentages, percentiles = calculate_percentiles_alberink2016(
        box_count, weight_sample, repeated_concentration_sample * presence_sample, alpha)
    assert np.allclose(percentages, [2.5, 50, 97.5])
    assert np.allclose(percentiles, [0.506374, 0.861110, 1.215845])
    # Include all zeros in the concentration samples
    presence_sample = np.zeros(repeated_concentration_sample.shape[1])
    percentages, percentiles = calculate_percentiles_alberink2016(
        box_count, weight_sample, repeated_concentration_sample * presence_sample, alpha)
    assert np.allclose(percentages, [2.5, 50, 97.5])
    assert np.isnan(percentiles[0])
    assert np.allclose(percentiles[1], 0)
    assert np.isnan(percentiles[2])


def test_alberink2016_alpha():
    # Use different alpha
    alpha = 0.1
    percentages, percentiles = calculate_percentiles_alberink2016(
        box_count, weight_sample, repeated_concentration_sample, alpha)
    assert np.allclose(percentages, [5, 50, 95])
    assert np.allclose(percentiles, [0.782878, 1.077062, 1.371245])


pytest.main()
