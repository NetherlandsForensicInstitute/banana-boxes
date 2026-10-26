from utils import lower_bound_hyper, upper_bound_hyper


def test_lower_upper_bound_hyper():
    total_count = 100
    sample_count = 10
    positive_sample_count = 8
    alpha = 0.05
    lower_bound = lower_bound_hyper(total_count, sample_count, positive_sample_count, alpha)
    upper_bound = upper_bound_hyper(total_count, sample_count, positive_sample_count, alpha)
    assert lower_bound == 51
    assert upper_bound == 95


def test_lower_bound_hyper_no_uncertainty():
    total_count = 12
    sample_count = 12
    positive_sample_count = 5
    alpha = 0.05
    lower_bound = lower_bound_hyper(total_count, sample_count, positive_sample_count, alpha)
    upper_bound = upper_bound_hyper(total_count, sample_count, positive_sample_count, alpha)
    assert lower_bound == upper_bound == 5


def test_lower_bound_hyper_symmetry():
    total_count = 100
    sample_count = 10
    positive_sample_count = 5
    alpha = 0.05
    lower_bound = lower_bound_hyper(total_count, sample_count, positive_sample_count, alpha)
    upper_bound = upper_bound_hyper(total_count, sample_count, positive_sample_count, alpha)
    assert upper_bound == total_count - lower_bound
