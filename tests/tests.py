import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["axes.facecolor"] = "FFFFFF"
rcParams["savefig.facecolor"] = "FFFFFF"
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"

rcParams.update({"figure.autolayout": True})

rcParams["figure.figsize"] = (9, 9)

import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["OMP_PROC_BIND"] = "TRUE"
os.environ["OMP_PLACES"] = "cores"

import seaborn as sns

import numpy as np

from scipy.stats import norm, nbinom

from fast_mutual_information import mi_negative_binomial, mi_normal, mi_negative_binomial_zi

np.random.seed(42)

def nb_mean(true_r, true_p):
    return true_r * ( 1 - true_p) / true_p

def nb_var(true_r, true_p):
    return nb_mean(true_r, true_p) / true_p

def gaussian_copula_negative_binomials(n_samples, r_list, p_list, corr_matrix=None, random_state=None, alpha=0.0):
    """
    Generates samples from a joint distribution of N negative binomial variables using a Gaussian copula.
    Optionally generates zero inflated negative binomials if alpha > 0.

    Parameters:
    - n_samples: Number of samples to generate.
    - r_list: List or array of r parameters for each negative binomial distribution.
    - p_list: List or array of p parameters for each negative binomial distribution.
    - corr_matrix: (Optional) N x N correlation matrix. If None, a random correlation matrix is generated.
    - random_state: Seed for reproducibility.
    - alpha: (Optional) Fraction of zeros for zero inflation (in [0,1]). Default is 0.0 (no zero inflation).

    Returns:
    - samples: An (n_samples, N) array of samples.
    - corr_matrix: The correlation matrix used to generate the samples.
    """

    n_factors = 10

    n_vars = len(r_list)
    if corr_matrix is None:
        # Generate a random positive definite covariance matrix and convert to correlation matrix.
        A = np.random.randn(n_vars, n_factors)
        cov = np.dot(A, A.T)
        d = np.sqrt(np.diag(cov))
        corr_matrix = cov / np.outer(d, d)

    # Step 1: Generate correlated standard normal variables
    mean = np.zeros(n_vars)
    z = np.random.multivariate_normal(mean, corr_matrix, size=n_samples)

    # Step 2: Transform to uniform [0,1] via the standard normal CDF
    u = norm.cdf(z)

    # Step 3: Transform each uniform variable to negative binomial via inverse CDF (PPF)
    samples = np.zeros_like(u)
    for i in range(n_vars):
        if alpha > 0:
            # For each entry, if u < alpha, output 0; otherwise, adjust the uniform to account for zero inflation.
            samples[:, i] = np.where(u[:, i] < alpha,
                                       0,
                                       nbinom.ppf((u[:, i] - alpha) / (1 - alpha), r_list[i], p_list[i]))
        else:
            samples[:, i] = nbinom.ppf(u[:, i], r_list[i], p_list[i])

    # Plot the pdf and samples of the first two distributions

    return samples, corr_matrix

def generate_nb(n_genes = 250, n_samples = 10000, alpha = 0.0):

    r_list = np.random.randint(20, 21, size=n_genes)
    p_list = np.random.uniform(0.3, 0.7, size=n_genes)

    return gaussian_copula_negative_binomials(n_samples, r_list, p_list, alpha = alpha), r_list, p_list

def estimate_mi_nb(samples, means, r_list, alphas, min_bin_content = 20):

    if np.any(alphas > 1E-8):
        mi_nb, _ = mi_negative_binomial_zi(samples.astype(np.int32), means.astype(np.float32), np.array(r_list).astype(np.float32), np.array(alphas).astype(np.float32), min_bin_content)
    else:
        mi_nb, _ = mi_negative_binomial(samples.astype(np.int32), means.astype(np.float32), np.array(r_list).astype(np.float32), min_bin_content)

    for i in range(len(mi_nb)):
        for j in range(i):
            mi_nb[i, j] = mi_nb[j, i]

    for i in range(len(mi_nb)):
        mi_nb[i, i] = 0

    return mi_nb

def get_mi_estimate_nb(n_genes = 250, n_samples = 10000, min_bin_content = 20, alpha = 0.0):

    r_list = np.random.randint(20, 21, size=n_genes)
    p_list = np.random.uniform(0.3, 0.7, size=n_genes)
    alphas = np.full(r_list.shape, alpha)

    samples, corr_matrix = gaussian_copula_negative_binomials(n_samples, r_list, p_list, alpha = alpha)

    means = nb_mean(r_list, p_list)
    variances = nb_var(r_list, p_list)

    mi_nb = estimate_mi_nb(samples, means, r_list, alphas, min_bin_content = min_bin_content)
    mi_analytical = -0.5 * np.log(1 - corr_matrix ** 2)

    for i in range(len(mi_analytical)):
        mi_analytical[i, i] = 0

    return mi_nb, mi_analytical, corr_matrix, samples

def get_mi_estimate_normal(n_genes = 250, n_samples = 10000, std_dev = 10, min_bin_content = 20):

    n_factors = 5
    B = np.random.randn(n_genes, n_factors)

    # Replace the covariance construction with:
    cov = np.dot(B, B.T)

    # Step 3: Convert Σ to a correlation matrix R via normalization:
    d = np.sqrt(np.diag(cov))
    corr_matrix = cov / np.outer(d, d)
    np.fill_diagonal(corr_matrix, 1.0)  # Ensure the diagonal elements are exactly 1

    # Step 4: Draw samples from the multivariate normal distribution N(0, R)
    mean = np.zeros(n_genes)
    samples_n = np.random.multivariate_normal(mean, corr_matrix, size=n_samples)

    samples_n *= std_dev
    # samples_n += 50
    samples_n += np.abs(np.min(samples_n))

    # means = np.zeros(samples_n.shape[1]) + 50
    # means = np.zeros(samples_n.shape[1]) + np.abs(np.min(samples_n))
    # variances = np.ones(samples_n.shape[1]) * std_dev**2
    means = np.mean(samples_n, axis = 0)
    variances = np.var(samples_n, axis = 0)

    mi, _ = mi_normal(samples_n, means, np.sqrt(variances), min_pop = min_bin_content)

    for i in range(n_genes):
        for j in range(i):
            mi[i, j] = mi[j, i]

    for i in range(n_genes):
        mi[i, i] = 0

    mi_analytical = -0.5 * np.log(1 - corr_matrix ** 2)

    return mi, mi_analytical, corr_matrix, samples_n

def test_mi():

    min_bin_content = 50
    n_samples = 10000
    n_genes = 50
    zi = 0.2

    # Quick scatter comparisons

    mi_nb, mi_analytical, nb_corr, samples_nb = get_mi_estimate_nb(min_bin_content=min_bin_content, n_genes = n_genes, n_samples = n_samples, alpha = zi)

    # Correct for zero inflation
    mi_analytical *= (1 - zi) ** 2

    sns.scatterplot(x=mi_nb.flatten(), y=mi_analytical.flatten(), alpha=0.5)
    plt.savefig('mi_nb_vs_mi_analytical.png')
    plt.clf()

    mi, mi_analytical_normal, normal_corr, samples = get_mi_estimate_normal(min_bin_content=min_bin_content, n_genes = n_genes, n_samples = n_samples)

    sns.scatterplot(x=mi.flatten(), y=mi_analytical_normal.flatten(), alpha=0.5)
    plt.savefig('mi_normal_vs_mi_normal_analytical.png')
    plt.clf()

    # Comparison KDEs

    fig, ax = plt.subplots()

    sns.kdeplot(x = mi.flatten(), y = mi_analytical_normal.flatten(), fill = False, levels=3, color = '#19647E', label = 'Normal marginals', linewidths=2)
    sns.kdeplot(x = mi_nb.flatten(), y = mi_analytical.flatten(), fill = False, levels=3, color = '#EE964B', label = 'NB marginals', linewidths=3)

    plt.plot([mi.min(), mi.max()], [mi.min(), mi.max()], color='grey', linestyle='--')

    plt.xlim(0, 0.5)
    plt.ylim(0, 0.5)

    ax.tick_params(axis='both', which='major', labelsize=21)

    plt.xlabel('Estimated MI', fontsize = 24)
    plt.ylabel('Analytical MI', fontsize = 24)

    plt.plot([-1, -1], [-1, -1], color = '#19647E', label = 'Normal')
    plt.plot([-1, -1], [-1, -1], color = '#EE964B', label = 'NB')
    plt.plot([-1, -1], [-1, -1], color = 'black', linestyle = '--', label = 'NB scikit-learn')

    plt.legend(fontsize = 21, loc = 'lower right')

    plt.savefig('mi_kde_comparison.png', dpi = 300, bbox_inches = 'tight')
    plt.savefig('mi_kde_comparison.pdf', dpi = 300, bbox_inches = 'tight')
    plt.clf()

    # Deviation KDE

    fig, ax = plt.subplots()

    sns.kdeplot(x = normal_corr.flatten(), y = mi.flatten() - mi_analytical_normal.flatten(), fill = False, levels=3, color = '#19647E', label = 'Normal marginals', linewidths = 2)

    sns.kdeplot(x = nb_corr.flatten(), y = mi_nb.flatten() - mi_analytical.flatten(), fill = False, levels=3, color = '#EE964B', label = 'NB marginals', linewidths = 3)

    ax.tick_params(axis='both', which='major', labelsize=21)

    plt.ylabel('Estimated - Analytical', fontsize = 24)
    plt.xlabel('Correlation', fontsize = 24)

    plt.plot([-1, 1.0], [0, 0], color='grey', linestyle='--')

    plt.xlim(-1.0, 1.0)
    plt.ylim(-0.04, 0.06)

    plt.savefig('mi_normal_difference.png', dpi = 300, bbox_inches = 'tight')
    plt.savefig('mi_normal_difference.pdf', dpi = 300, bbox_inches = 'tight')
    plt.clf()
