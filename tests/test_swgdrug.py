import pandas as pd
from pathlib import Path
import pytest
import numpy as np

from boundary_methods import calculate_percentile_swgdrug


data_dir = Path(__file__).parent / 'data'
data = pd.read_csv(data_dir / 'swgdrug_example2.csv')
all_weights = data['weight'].to_numpy()
weight_sample = all_weights[:10]
box_count = 100
alpha = 1 - (0.99**2)
presence_sample = np.ones(6)

# Test cases are based on Example 2 on pages 9 to 14 of "Measurement Uncertainty for Extrapolations of Net Weight and
# Unit Count Supplemental Document SD-6", Revision 1 (2017-07-10), by SWGDRUG. u_w (unexpanded): is set to 1E-10.
# Table 1.3 contains 30 weight samples; Example 2 only uses the first 10 of them (also listed in Table 1.1).

# The SWGDRUG tool requires a 'Minimum Statutory Wt.' as input, resulting in a minimum number of units required to be
# positive (containing drugs) in the population, which is finally translated in a sampled number of units for which it
# needs to be confirmed that all of them are positive. For the current function the 'presence sample' is an input,
# instead of an output. To match calculate_percentile_swgdrug and the SWGDRUG-tool, first the number of units that are
# sampled for drug-presence is fixed (at 6 in this case); all of them are supposed to be positive. Next, it is checked
# in calculate_percentile_swgdrug, given the other inputs (weight samples, total box count, and alpha), how many units
# in the population are then inferred to be positive. Finally, the 'Minimum Statutory Wt.' in the SWGDRUG-tool is set to
# a value that results into the same numbers of positive units (in both the population and the sample).


def test_swgdrug_defaults():
    # Use original box count, weight samples and alpha; 'Minimum Statutory Wt.' needed to infer 48 (+): 25.0
    percentages, percentiles = calculate_percentile_swgdrug(box_count, presence_sample, weight_sample, alpha)
    assert np.allclose(percentages, [1.99])
    assert np.allclose(percentiles, [25.255226])


def test_swgdrug_zero_presence():
    # Use sample with a 0; should result into nan as output, because the method cannot deal with this
    presence_sample = np.zeros(1)
    percentages, percentiles = calculate_percentile_swgdrug(box_count, presence_sample, weight_sample, alpha)
    assert percentiles == [np.nan]


def test_swgdrug_box_count():
    # Different total number of boxes; 'Minimum Statutory Wt.' needed to infer 25 (+): 13.0
    box_count = 50
    percentages, percentiles = calculate_percentile_swgdrug(box_count, presence_sample, weight_sample, alpha)
    assert np.allclose(percentages, [1.99])
    assert np.allclose(percentiles, [13.153768])


def test_swgdrug_weight_samples():
    # Different number of weight samples used; 'Minimum Statutory Wt.' needed to infer 48 (+): 23.5
    weight_sample = all_weights[:5]
    percentages, percentiles = calculate_percentile_swgdrug(box_count, presence_sample, weight_sample, alpha)
    assert np.allclose(percentages, [1.99])
    assert np.allclose(percentiles, [23.599848])


def test_swgdrug_alpha():
    # Different alpha; 'Minimum Statutory Wt.' needed to infer 56 (+): 29.5
    alpha = 0.05
    percentages, percentiles = calculate_percentile_swgdrug(box_count, presence_sample, weight_sample, alpha)
    assert np.allclose(percentages, [5])
    assert np.allclose(percentiles, [29.730346])


pytest.main()
