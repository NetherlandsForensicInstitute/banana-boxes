import pandas as pd
from pathlib import Path
import pytest
import numpy as np

from boundary_methods import calculate_percentiles_hyper_t


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
    percentages, percentiles, best_estimate = calculate_percentiles_hyper_t(box_count, presence_sample, weight_sample, alpha)
    assert np.allclose(percentages, [2.5, 97.5])
    assert np.allclose(percentiles, [21.664618, 27.512214])
    assert np.allclose(best_estimate, 26.500000)
    # Use larger box count, so no finite population correction is used
    percentages, percentiles, best_estimate = calculate_percentiles_hyper_t(1000, presence_sample, weight_sample, alpha)
    assert np.allclose(percentages, [2.5, 97.5])
    assert np.allclose(percentiles, [210.232888, 276.401824])
    assert np.allclose(best_estimate, 265.000000)
    # Use different alpha
    percentages, percentiles, best_estimate = calculate_percentiles_hyper_t(box_count, presence_sample, weight_sample, 0.1)
    assert np.allclose(percentages, [5.0, 95.0])
    assert np.allclose(percentiles, [22.295700, 27.372678])
    assert np.allclose(best_estimate, 26.500000)


def test_1_zeros():
    data = pd.read_csv(data_dir / 'enfsi-1_zeros.csv')
    weight_sample = data['weight'].to_numpy()
    presence_sample = np.where(weight_sample > 0, 1, 0)
    weight_sample = weight_sample[weight_sample > 0]
    # Use original box count and alpha
    percentages, percentiles, best_estimate = calculate_percentiles_hyper_t(box_count, presence_sample, weight_sample, alpha)
    assert np.allclose(percentages, [2.5, 97.5])
    assert np.allclose(percentiles, [19.854495, 27.27006])
    assert np.allclose(best_estimate, 25.347826)


def test_2_zeros():
    data = pd.read_csv(data_dir / 'enfsi-2_zeros.csv')
    weight_sample = data['weight'].to_numpy()
    presence_sample = np.where(weight_sample > 0, 1, 0)
    weight_sample = weight_sample[weight_sample > 0]
    # Use original box count and alpha
    percentages, percentiles, best_estimate = calculate_percentiles_hyper_t(box_count, presence_sample, weight_sample, alpha)
    assert np.allclose(percentages, [2.5, 97.5])
    assert np.allclose(percentiles, [18.301862, 27.029132])
    assert np.allclose(best_estimate, 24.195652)
    # Use larger box count and different alpha
    percentages, percentiles, best_estimate = calculate_percentiles_hyper_t(1000, presence_sample, weight_sample, 0.1)
    assert np.allclose(percentages, [5.0, 95.0])
    assert np.allclose(percentiles, [183.852244, 272.052719])
    assert np.allclose(best_estimate, 241.956522)


pytest.main()
