from utils import lower_bound_hyper


def test_lower_bound_hyper():
    total_count = 100
    sample_count = 10
    positive_sample_count = 8
    alpha = 0.05
    lower_bound = lower_bound_hyper(total_count, sample_count, positive_sample_count, alpha)
    assert lower_bound == 51


def test_lower_hyper_no_uncertainty():
    total_count = 12
    sample_count = 12
    positive_sample_count = 5
    alpha = 0.05
    lower_bound = lower_bound_hyper(total_count, sample_count, positive_sample_count, alpha)
    assert lower_bound == 5
