from pathlib import Path

from math import sqrt
import numpy as np
import pandas as pd
from scipy.stats import t, norm

from utils import draw_normal_mcmc_samples, lower_bound_hyper, upper_bound_hyper


def calculate_percentiles_enfsi(box_count: int,
                                presence_sample: np.ndarray,
                                weight_sample: np.ndarray,
                                alpha: float) -> tuple[list[float], list[float]]:
    """
    Calculate the given percentiles of the total drug weight using the ENFSI approach described in "Guidelines on
    Sampling of Illicit Drugs for Qualitative Analysis", Second edition (2016), by the ENFSI Drug Working Group.
    It is based on this publication: https://doi.org/10.1111/j.1556-4029.2009.01256.x

    Assumptions:
    - The units that are weighted are the same as the units where the presence of drugs is determined.
    - The concentration of each unit is 0 or 1.

    Parameters
    ----------
    box_count: int
        The total number of boxes.
    presence_sample: np.ndarray
        A sample indicating the presence of drugs (0's and 1's).
    weight_sample: np.ndarray
        A sample of the weights of the boxes.
    alpha: float
        The level of significance.

    Returns
    -------
    tuple[list[float], list[float]]
        Two lists: a list with the percentages corresponding to the given level of significance, together with
        the best estimate (50 percent). For example, when alpha is 0.1, the percentages returned are [5, 50, 95],
        and a list with the resulting percentiles of the total drug weight corresponding with these percentages.
    """

    assert weight_sample.shape[0] == presence_sample.shape[0], \
        (f"the number of weighted units ({weight_sample.shape[0]}) should be equal to the number of units where the "
         f"presence of drugs is determined ({presence_sample.shape[0]}).")

    sample_count = weight_sample.shape[0]
    drug_free_sample_count = sample_count - np.count_nonzero(presence_sample)
    drug_containing_weight_sample = weight_sample[presence_sample != 0]

    n = box_count
    m = sample_count
    r = drug_free_sample_count
    W_m_r = drug_containing_weight_sample

    # Apply finite population correction if m/n > 0.1.
    if m/n <= 0.1:
        fpc = 1
    else:
        # print("Finite population correction is applied in the ENFSI calculation, "
        #       "since more than 10% of the total number of units is sampled.")
        fpc = sqrt((n-m)/n)

    best_estimate = n * (m - r)/m * np.mean(W_m_r)
    half_bandwidth = n * (m - r) / m * sqrt(np.var(W_m_r, ddof=1) / (m - r)) * t.ppf(1 - alpha / 2, m - r - 1) * fpc
    lower_bound = best_estimate - half_bandwidth
    upper_bound = best_estimate + half_bandwidth

    percentages = [alpha/2 * 100, 50, (1 - alpha/2) * 100]
    percentiles = [lower_bound, best_estimate, upper_bound]

    return percentages, percentiles


def calculate_percentiles_alberink2010(box_count: int,
                                       presence_sample: np.ndarray,
                                       weight_sample: np.ndarray,
                                       alpha: float) -> tuple[list[float], list[float]]:
    """
    Calculate the given percentiles of the total drug weight using the approach described in Alberink et al. (2010):
    "Comparison of Frequentist Methods for Estimating the Total Weight of Consignments of Drugs".
    DOI-link: https://doi.org/10.1111/j.1556-4029.2009.01309.x

    Assumptions:
    - The units that are weighted are the same as the units where the presence of drugs is determined.
    - The concentration of each unit is 0 or 1.

    Note:
    - If there is only one sample containing drugs, the variance cannot be determined and the boundaries returned
    are np.nan.
    - If there are no samples containing drugs, the best estimate is also np.nan.
    - If the best estimate minus the half bandwidth is below zero, the lower bound is set to zero.

    Parameters
    ----------
    box_count: int
        The total number of boxes.
    presence_sample: np.ndarray
        A sample indicating the presence of drugs (0's and 1's).
    weight_sample: np.ndarray
        A sample of the weights of the boxes.
    alpha: float
        The level of significance.

    Returns
    -------
    tuple[list[float], list[float]]
        Two lists: a list with the percentages corresponding to the given level of significance, together with
        the best estimate (50 percent). For example, when alpha is 0.1, the percentages returned are [5, 50, 95],
        and a list with the resulting percentiles of the total drug weight corresponding with these percentages.
    """

    assert weight_sample.shape[0] == presence_sample.shape[0], \
        (f"the number of weighted units ({weight_sample.shape[0]}) should be equal to the number of units where the "
         f"presence of drugs is determined ({presence_sample.shape[0]}).")

    sample_count = weight_sample.shape[0]
    drug_free_sample_count = sample_count - np.count_nonzero(presence_sample)
    drug_containing_weight_sample = weight_sample[presence_sample != 0]

    n = box_count
    m = sample_count
    r = drug_free_sample_count
    W_m_r = drug_containing_weight_sample

    best_estimate = n * (m - r) / m * np.mean(W_m_r)
    half_bandwidth = (n * (m - r) / m * sqrt((np.var(W_m_r, ddof=1) + r / m * np.mean(W_m_r) ** 2) / (m - r)) *
                      norm.ppf(1 - alpha / 2) * sqrt((n - m) / (n - 1)))
    lower_bound = np.max([best_estimate - half_bandwidth, 0])  # make sure lower bound is not less than zero
    upper_bound = best_estimate + half_bandwidth

    percentages = [alpha/2 * 100, 50, (1 - alpha/2) * 100]
    percentiles = [lower_bound, best_estimate, upper_bound]

    return percentages, percentiles


def calculate_percentile_swgdrug(box_count: int,
                                 presence_sample: np.ndarray,
                                 weight_sample: np.ndarray,
                                 alpha: float) -> tuple[list[float], list[float]]:
    """
    Calculate the given lower bound corresponding to the given level of significance of the total drug weight using the
    SWGDRUG approach described in: "Measurement Uncertainty for Extrapolations of Net Weight and Unit Count Supplemental
    Document SD-6", Revision 1 (2017-07-10), by SWGDRUG.

    Presence sample may only contain ones, no drug-free units allowed.

    Parameters
    ----------
    box_count: int
        The total number of boxes.
    presence_sample: np.ndarray
        A sample indicating the presence of drugs (0's and 1's).
    weight_sample: np.ndarray
        A sample of the weights of the boxes.
    alpha: float
        The level of significance.

    Returns
    -------
    tuple[list[float], list[float]]
        Two lists: a list with the percentage corresponding to the given level of significance. For example,
        when alpha is 0.1, the percentage returned is [10], together with a list of the resulting percentile
        of the total drug weight corresponding with this percentage.
    """

    # Only run swgdrug if all sampled units contain drugs. Otherwise, return nan as percentile.
    if not np.all(presence_sample == 1):
        percentage = [alpha * 100]
        percentile = [np.nan]
        return percentage, percentile

    presence_sample_count = presence_sample.shape[0]
    weight_sample_count = weight_sample.shape[0]
    drug_free_sample_count = presence_sample.shape[0] - np.count_nonzero(presence_sample)

    n = box_count
    m_p = presence_sample_count
    m_w = weight_sample_count
    r = drug_free_sample_count

    # The combined confidence level 1 - alpha_c is larger than or equal to the product of the two separate confidence
    # levels: 1 - alpha_c >= (1 - alpha_h) * (1 - alpha_t). We choose alpha_h and alpha_t such that
    # 1 - alpha_c >= 1 - alpha = (1 - alpha_h) * (1 - alpha_t). Now the combined confidence level is at least 1 - alpha.
    alpha_h = alpha_t = 1 - sqrt(1 - alpha)

    # calculate lower bound
    k_lb = lower_bound_hyper(n, m_p, m_p - r, alpha_h)

    lower_bound = k_lb * (np.mean(weight_sample) - sqrt(np.var(weight_sample, ddof=1) / m_w) *
                          t.ppf(1-alpha_t/2, m_w-1))

    percentage = [100 * alpha]
    percentile = [lower_bound]

    return percentage, percentile


def calculate_percentiles_hyper_t(box_count: int,
                                  presence_sample: np.ndarray,
                                  weight_sample: np.ndarray,
                                  alpha: float) -> tuple[list[float], list[float], float]:
    """
    Calculate the percentiles of the total drug weight corresponding to the given level of significance,
    using the hyper-t approach (see article Van der Ham et al.), together with the best estimate

    Parameters
    ----------
    box_count: int
        The total number of boxes.
    presence_sample: np.ndarray
        A sample indicating the presence of drugs (0's and 1's).
    weight_sample: np.ndarray
        A sample of the weights of the boxes.
    alpha: float
        The level of significance.

    Returns
    -------
    tuple[list[float], list[float], float]
        Two lists: a list with the percentage corresponding to the given level of significance, for example,
        when alpha is 0.1, the list [5, 95], a list of the resulting percentiles of the total drug weights
        corresponding with these percentages.
        Float: best estimate of the total drug weight.
    """

    # No weights of zero allowed in the input
    assert np.count_nonzero(weight_sample) == weight_sample.shape[0], "No zeros allowed in weight samples."

    presence_sample_count = presence_sample.shape[0]
    weight_sample_count = weight_sample.shape[0]
    drug_free_sample_count = presence_sample.shape[0] - np.count_nonzero(presence_sample)

    n = box_count
    m_p = presence_sample_count
    m_w = weight_sample_count
    r = drug_free_sample_count

    k_best_estimate = ((m_p-r)/m_p) * n
    # We use a best estimate for k that is not necessary an integer, but which is the most logical value
    best_estimate = k_best_estimate * np.mean(weight_sample)

    # calculate lower and upper bound
    k_lb = lower_bound_hyper(n, m_p, m_p - r, alpha / 4)
    k_ub = upper_bound_hyper(n, m_p, m_p - r, alpha / 4)

    lower_bound = k_lb * (np.mean(weight_sample) - sqrt(np.var(weight_sample, ddof=1) / m_w) * t.ppf(1-alpha/4, m_w-1)
                          * sqrt((n-m_w)/n))
    upper_bound = k_ub * (np.mean(weight_sample) + sqrt(np.var(weight_sample, ddof=1) / m_w) * t.ppf(1-alpha/4, m_w-1)
                          * sqrt((n-m_w)/n))

    percentages = [100 * alpha/2, 100 * (1 - alpha/2)]
    percentiles = [lower_bound, upper_bound]

    return percentages, percentiles, best_estimate


def calculate_percentiles_alberink2016(box_count: int,
                                       weight_sample: np.ndarray,
                                       repeated_concentration_sample: np.ndarray,
                                       alpha: float) -> tuple[list[float], list[float]]:
    """
    Calculate the given percentiles of the total drug weight using the approach suggested by Alberink et al. 2016:
    "Quantifying Uncertainty in Estimations of the Total Weight of Drugs in Groups of Complex Matrices: Using the
    Welch–Satterthwaite Equation".
    DOI-link: https://doi.org/10.1111/1556-4029.13351

    Assumptions:
    - All units are weighted
    - The units in the presence sample are the same as the units in the concentration sample. Therefore, their count
      is equal.
    - There are at least two concentration measurements per sampled item.

    Parameters
    ----------
    box_count: int
        The total number of boxes.
    weight_sample: np.ndarray
        A sample of the weights of the boxes.
    repeated_concentration_sample: np.ndarray
        A repeated sample of the drug concentration of the boxes. Each column (dimension 1) represents a sample, and
        each row (dimension 0) is a different repeat of that sample.
    alpha: float
        The level of significance.

    Returns
    -------
    tuple[list[float], list[float]]
        Two lists: a list with the percentages corresponding to the given level of significance, together with
        the best estimate (50 percent). For example, when alpha is 0.1, the percentages returned are [5, 50, 95],
        accompanied by a list with the resulting percentiles of the total drug weight corresponding with these
        percentages.
    """

    # Check whether there are at least two concentration samples per sampled unit.
    if repeated_concentration_sample.shape[0] <= 1:
        raise ValueError("There should be at least two concentration measurements per sampled item.")

    # Check that all units are weighted.
    weight_sample_count = weight_sample.shape[0]
    assert weight_sample_count == box_count, (f"The number of weighted units ({weight_sample_count}) should be equal "
                                              f"to the total number of boxes {box_count}.")

    c_ij = repeated_concentration_sample
    l_i = c_ij.shape[0]
    m_c = c_ij.shape[1]
    c_i = c_ij.mean(axis=0)
    w_others = weight_sample[m_c:]
    w_rest = np.sum(w_others)  # Total weight of all samples where the concentration is not sampled
    w_i = weight_sample[:m_c]
    c_mean = c_i.mean()

    # best estimate for total drug weight
    best_estimate = np.sum(c_i * w_i) + (c_mean * w_rest)

    # Note: in the paper of Van der Ham et al., var_i = a_i, var_c = b, and df = v.
    # half bandwidth
    std_i = c_ij.std(ddof=1, axis=0)
    var_i = (w_i + w_rest / m_c) ** 2 / l_i * std_i ** 2
    std_c = c_i.std(ddof=1)
    var_c = (w_rest ** 2 / m_c + np.sum(w_others ** 2)) * std_c ** 2
    std_term = np.sqrt(var_c + np.sum(var_i))
    # When all values are equal, there is division by zero, resulting in nan. This is the result we want in that case.
    with np.errstate(divide='ignore', invalid='ignore'):
        df = (var_c + np.sum(var_i)) ** 2 / (var_c ** 2 / (m_c - 1) + np.sum(var_i ** 2 / (l_i - 1)))
    t_term = t.ppf(1 - alpha / 2, df)
    half_bandwidth = std_term * t_term

    lower_bound = best_estimate - half_bandwidth
    upper_bound = best_estimate + half_bandwidth

    percentages = [alpha/2 * 100, 50, (1 - alpha/2) * 100]
    percentiles = [lower_bound, best_estimate, upper_bound]

    return percentages, percentiles


def calculate_percentiles_bayesian(box_count: int,
                                   presence_sample: np.ndarray,
                                   weight_sample: np.ndarray,
                                   concentration_sample: np.ndarray,
                                   alpha: float,
                                   output_folder: Path,
                                   mcmc_priors: dict[str, dict],
                                   mcmc_rng: np.random.Generator = np.random.default_rng(),
                                   mcmc_draw_count: int = 1000,
                                   mcmc_chain_count: int = 4,
                                   mcmc_no_concentration: bool = False,
                                   mcmc_no_weight: bool = False,
                                   **mcmc_kwargs: dict) -> tuple[list[float], list[float], pd.DataFrame]:
    """
    Calculate the percentiles corresponding to the given alpha of the total drug weight using a Bayesian approach.

    Parameters
    ----------
    box_count: int
        The total number of boxes.
    presence_sample: np.ndarray
        A sample indicating the presence of drugs (0's and 1's).
    weight_sample: np.ndarray
        A sample of the weights of the boxes.
    concentration_sample: np.ndarray
        A sample of the concentration of drugs in the boxes.
    alpha: float
        The resulting credibility interval will be a (1-alpha)*100% credibility interval.
    output_folder: Path
        The path to the output folder.
    mcmc_priors: dict[str, dict]
        Two sets (weights & concentrations) of prior distributions for the two MCMC parameters (mu & sigma).
    mcmc_rng: np.random.Generator
        Numpy random number generator used during MCMC simulations.
    mcmc_draw_count: int
        The number of MCMC draws, defaults to 1000.
    mcmc_chain_count: int
        The number of MCMC chains, defaults to 4.
    mcmc_no_concentration: bool
        Whether to skip the concentration simulations or not, defaults to False.
    mcmc_no_weight: bool
        Whether to skip the weight simulations or not, defaults to False.
    **mcmc_kwargs: dict
        Dictionary of MCMC parameters to be used in draw_normal_mcmc_samples
        such as mcmc_tune_count, mcmc_random_seed and mcmc_save_posteriors.

    Returns
    -------
    tuple[list[float], list[float], pd.DataFrame]
        Two lists: a list with the percentages corresponding to the given alpha, together with
            the best estimate (50 percent). For example, when alpha is 0.1, the percentages returned are [5, 50, 95],
            and a list with the resulting percentiles of the total drug weight corresponding with these percentages.
        pd.DataFrame: MCMC-diagnostics (such as R_hat and ESS) for all relevant model parameters (mu's and sigma's).
    """

    # Sample from the posterior distribution of the presence (zeros and ones).
    # The prior distribution is beta(1/2,1/2), the posterior is beta(1/2+positives_sample,1/2+negatives_sample).
    positives_sample = np.sum(presence_sample == 1)
    negatives_sample = np.sum(presence_sample == 0)
    presence_trace_p = mcmc_rng.beta(1/2 + positives_sample, 1/2 + negatives_sample, mcmc_draw_count * mcmc_chain_count)
    # Draw presence (0's and 1's) from the MCMC results.
    presence_mcmc = mcmc_rng.binomial(n=1, p=presence_trace_p, size=(box_count, mcmc_chain_count * mcmc_draw_count))

    # Run MCMC simulation for the weights, assumed to be normally distributed with uniform priors for mu and sigma.
    # The boundary values for the priors are based on the observed values.
    diagnostics = pd.DataFrame()
    if not mcmc_no_weight:
        priors = mcmc_priors['weights']
        weight_trace_mu, weight_trace_sigma, weight_diagnostics = draw_normal_mcmc_samples(
            weight_sample, priors['mu'], priors['sigma'],
            'weight', output_folder, mcmc_rng, mcmc_draw_count, mcmc_chain_count, **mcmc_kwargs)
        # Draw weights from the MCMC results.
        weight_mcmc = mcmc_rng.normal(loc=weight_trace_mu, scale=weight_trace_sigma,
                                   size=(box_count, mcmc_chain_count * mcmc_draw_count))
        # Store the MCMC-diagnostics (R_hat, Effective Sample Size, ...)
        diagnostics = pd.concat([diagnostics, weight_diagnostics])
    else:
        weight_mcmc = np.stack((weight_sample,)*(mcmc_chain_count * mcmc_draw_count), axis=-1)

    # Perform a similar MCMC simulation for the concentrations
    if not mcmc_no_concentration:
        priors = mcmc_priors['concentrations']
        concentration_trace_mu, concentration_trace_sigma, concentration_diagnostics = draw_normal_mcmc_samples(
            concentration_sample, priors['mu'], priors['sigma'],
            'concentration', output_folder, mcmc_rng, mcmc_draw_count, mcmc_chain_count, **mcmc_kwargs)
        # Draw concentrations from the MCMC results.
        concentration_mcmc = mcmc_rng.normal(loc=concentration_trace_mu, scale=concentration_trace_sigma,
                                              size=(box_count, mcmc_chain_count * mcmc_draw_count))
        # Store the MCMC-diagnostics (R_hat, Effective Sample Size, ...)
        diagnostics = pd.concat([diagnostics, concentration_diagnostics])
    else:
        concentration_mcmc = np.ones((box_count, mcmc_chain_count * mcmc_draw_count)) * concentration_sample[0]

    # Overwrite the first n results with the accordingly sampled results. This approach works for all possible sample
    # count combinations (for example also if the weights of all units are measured). In specific cases, when it is
    # certain that the samples of different measurement types do (or do not) belong to the same units, this information
    # can also be included here. This can be done by assigning the sampled results to specific units, instead of the
    # first n.
    presence_sample_count = len(presence_sample)
    presence_mcmc[0:presence_sample_count, :] = np.tile(np.expand_dims(presence_sample, 1),
                                                        (1, mcmc_chain_count * mcmc_draw_count))
    weight_sample_count = len(weight_sample)
    weight_mcmc[0:weight_sample_count, :] = np.tile(np.expand_dims(weight_sample, 1),
                                                    mcmc_chain_count * mcmc_draw_count)
    concentration_sample_count = len(concentration_sample)
    concentration_mcmc[0:concentration_sample_count, :] = np.tile(np.expand_dims(concentration_sample, 1),
                                                                  mcmc_chain_count * mcmc_draw_count)

    # Calculate the total weight of the mcmc simulation
    weight_mcmc_total = np.sum(presence_mcmc * weight_mcmc * concentration_mcmc, axis=0)

    # Calculate the percentiles
    percentages = [alpha/2 * 100, 50, (1 - alpha/2) * 100]
    percentiles = np.percentile(weight_mcmc_total, percentages)

    return percentages, percentiles, diagnostics
