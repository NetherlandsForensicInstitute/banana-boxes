import pandas as pd
from pathlib import Path
import pytest
import numpy as np

from boundary_methods import calculate_percentiles_enfsi


# Test cases are based on Example 1 on page 52 of "Guidelines on Sampling of Illicit Drugs for Qualitative Analysis",
# Second edition (2016), by the ENFSI Drug Working Group. The data files are constructed such that the averages over the
# (non-zero) weights equal 0.26500 and the sample standard deviations equal 0.02300, while obeying the specified number
# of samples (23) and number of drug-free samples (either 0, 1, or 2). Reference values are calculated using the ENFSI
# Excel tool.

data_dir = Path(__file__).parent / 'data'
box_count = 100
alpha = 0.05


def test_0_zeros():
    data = pd.read_csv(data_dir / 'enfsi-0_zeros.csv')
    weight_sample = data['weight'].to_numpy()
    presence_sample = np.where(weight_sample > 0, 1, 0)
    # Use original box count and alpha
    percentages, percentiles = calculate_percentiles_enfsi(box_count, presence_sample, weight_sample, alpha)
    assert np.allclose(percentages, [2.5, 50.0, 97.5], atol=1E-1)
    assert np.allclose(percentiles, [25.627, 26.500, 27.373], atol=1E-3)
    # Use larger box count, so no finite population correction is used
    percentages, percentiles = calculate_percentiles_enfsi(1000, presence_sample, weight_sample, alpha)
    assert np.allclose(percentages, [2.5, 50.0, 97.5], atol=1E-1)
    assert np.allclose(percentiles, [255.054, 265.000, 274.946], atol=1E-3)
    # Use different alpha
    percentages, percentiles = calculate_percentiles_enfsi(box_count, presence_sample, weight_sample, 0.1)
    assert np.allclose(percentages, [5.0, 50.0, 95.0], atol=1E-1)
    assert np.allclose(percentiles, [25.777, 26.500, 27.223], atol=1E-3)


def test_1_zeros():
    data = pd.read_csv(data_dir / 'enfsi-1_zeros.csv')
    weight_sample = data['weight'].to_numpy()
    presence_sample = np.where(weight_sample > 0, 1, 0)
    # Use original box count and alpha
    percentages, percentiles = calculate_percentiles_enfsi(box_count, presence_sample, weight_sample, alpha)
    assert np.allclose(percentages, [2.5, 50.0, 97.5], atol=1E-1)
    assert np.allclose(percentiles, [24.492, 25.348, 26.204], atol=1E-3)


def test_2_zeros():
    data = pd.read_csv(data_dir / 'enfsi-2_zeros.csv')
    weight_sample = data['weight'].to_numpy()
    presence_sample = np.where(weight_sample > 0, 1, 0)
    # Use original box count and alpha
    percentages, percentiles = calculate_percentiles_enfsi(box_count, presence_sample, weight_sample, alpha)
    assert np.allclose(percentages, [2.5, 50.0, 97.5], atol=1E-1)
    assert np.allclose(percentiles, [23.357, 24.196, 25.034], atol=1E-3)
    # Use larger box count and different alpha
    percentages, percentiles = calculate_percentiles_enfsi(1000, presence_sample, weight_sample, 0.1)
    assert np.allclose(percentages, [5.0, 50.0, 95.0], atol=1E-1)
    assert np.allclose(percentiles, [234.053, 241.957, 249.860], atol=1E-3)


pytest.main()
