import pandas as pd
from pathlib import Path
import pytest
import numpy as np
from scipy.stats import norm

from boundary_methods import calculate_percentiles_alberink2010


# Test cases are based on calculation results generated with the Matlab-scripts for the publication from 2010:
# "Comparison of Frequentist Methods for Estimating the Total Weight of Consignments of Drugs".
# It was ensured that the average-term was calculated over the non_zero_samples only, instead of all samples.
# The publication itself does not include a calculation example on the level of weight and presence samples.

data_dir = Path(__file__).parent / 'data'
data = pd.read_csv(data_dir / 'alberink2010.csv')
all_weights = data['weight'].to_numpy()
weight_sample = all_weights
presence_sample = np.where(weight_sample > 0, 1, 0)
box_count = 40
alpha = 0.05
t_term = norm.ppf(1 - alpha / 2)


def test_alberink2010_default():
    # Use original box count, samples and alpha
    percentages, percentiles = calculate_percentiles_alberink2010(box_count, presence_sample, weight_sample, alpha)
    std_term = (percentiles[1] - percentiles[0]) / t_term
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(std_term, 78.8061)


def test_alberink2010_box_count():
    # Different box count
    box_count = 80
    percentages, percentiles = calculate_percentiles_alberink2010(box_count, presence_sample, weight_sample, alpha)
    std_term = (percentiles[1] - percentiles[0]) / t_term
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(std_term, 191.8091)


def test_alberink2010_all_boxes():
    # Different box count: all boxes are sampled
    box_count = 20
    percentages, percentiles = calculate_percentiles_alberink2010(box_count, presence_sample, weight_sample, alpha)
    std_term = (percentiles[1] - percentiles[0]) / t_term
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(std_term, 0)


def test_alberink2010_samples():
    # Different samples; use 15 instead of 20
    weight_sample = all_weights[:15]
    presence_sample = np.where(weight_sample > 0, 1, 0)
    percentages, percentiles = calculate_percentiles_alberink2010(box_count, presence_sample, weight_sample, alpha)
    std_term = (percentiles[1] - percentiles[0]) / t_term
    assert np.allclose(percentages, [2.5, 50.0, 97.5])
    assert np.allclose(std_term, 114.2672)


def test_alberink2010_alpha():
    # Different alpha (same results from Matlab, this does not involve alpha)
    alpha = 0.1
    percentages, percentiles = calculate_percentiles_alberink2010(box_count, presence_sample, weight_sample, alpha)
    t_term = norm.ppf(1 - alpha / 2)
    std_term = (percentiles[1] - percentiles[0]) / t_term
    assert np.allclose(percentages, [5.0, 50.0, 95.0])
    assert np.allclose(std_term, 78.8061)


pytest.main()
