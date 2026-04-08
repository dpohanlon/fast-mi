import unittest
import numpy as np
from scipy import sparse

from scipy.stats import norm, nbinom

from fast_mutual_information import (
    mi_normal,
    mi_negative_binomial,
    mi_negative_binomial_zi,
    mi_normal_sparse,
    mi_negative_binomial_sparse,
    mi_negative_binomial_exposure,
)

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

        A = np.random.randn(n_vars, n_factors)
        cov = np.dot(A, A.T)
        d = np.sqrt(np.diag(cov))
        corr_matrix = cov / np.outer(d, d)

    mean = np.zeros(n_vars)
    z = np.random.multivariate_normal(mean, corr_matrix, size=n_samples)

    u = norm.cdf(z)

    samples = np.zeros_like(u)
    for i in range(n_vars):
        if alpha > 0:

            samples[:, i] = np.where(u[:, i] < alpha,
                                       0,
                                       nbinom.ppf((u[:, i] - alpha) / (1 - alpha), r_list[i], p_list[i]))
        else:
            samples[:, i] = nbinom.ppf(u[:, i], r_list[i], p_list[i])

    return samples, corr_matrix

def gaussian_copula_negative_binomials_with_exposure(n_samples, r_list, mu0_list, exposure, corr_matrix=None, random_state=None, alpha=0.0):
    n_vars = len(r_list)
    if corr_matrix is None:
        n_factors = 10
        A = np.random.randn(n_vars, n_factors)
        cov = np.dot(A, A.T)
        d = np.sqrt(np.diag(cov))
        corr_matrix = cov / np.outer(d, d)

    mean = np.zeros(n_vars)
    z = np.random.multivariate_normal(mean, corr_matrix, size=n_samples)
    u = norm.cdf(z)

    samples = np.zeros_like(u)
    for i in range(n_vars):
        r = r_list[i]
        mu0 = mu0_list[i]
        m_j = mu0 * exposure  # per-sample means
        p_j = r / (m_j + r)
        if alpha > 0:
            uc = np.clip((u[:, i] - alpha) / (1 - alpha), 0, 1)
            draw = np.array([nbinom.ppf(uc[j], r, p_j[j]) for j in range(n_samples)])
            samples[:, i] = np.where(u[:, i] < alpha, 0, draw)
        else:
            samples[:, i] = np.array([nbinom.ppf(u[j, i], r, p_j[j]) for j in range(n_samples)])
    return samples, corr_matrix

def get_normal(n_genes = 250, n_samples = 10000, std_dev = 10, min_bin_content = 20):

    n_factors = 5
    B = np.random.randn(n_genes, n_factors)

    cov = np.dot(B, B.T)

    d = np.sqrt(np.diag(cov))
    corr_matrix = cov / np.outer(d, d)
    np.fill_diagonal(corr_matrix, 1.0)

    mean = np.zeros(n_genes)
    samples_n = np.random.multivariate_normal(mean, corr_matrix, size=n_samples)

    samples_n *= std_dev

    samples_n += np.abs(np.min(samples_n))

    mi_analytical = -0.5 * np.log(1 - corr_matrix ** 2)

    return mi_analytical, corr_matrix, samples_n

def assert_allclose_tolerant(actual, desired, rtol=1e-7, atol=0, max_violations=0, err_msg=''):
    """
    Assert that two arrays are element-wise equal within a tolerance, but only
    fail if the number of violations exceeds a specified maximum.
    """
    # Calculate the mask of elements that violate the tolerance
    violations_mask = np.abs(actual - desired) > (atol + rtol * np.abs(desired))
    num_violations = np.sum(violations_mask)

    if num_violations > max_violations:
        # If the test fails, provide a detailed error message
        max_abs_diff = np.max(np.abs(actual[violations_mask] - desired[violations_mask]))

        full_err_msg = (
            f"{err_msg}\n"
            f"Validation failed: Found {num_violations} violations, which is more than the allowed {max_violations}.\n"
            f"Maximum absolute difference among violations: {max_abs_diff}"
        )
        raise AssertionError(full_err_msg)

class TestMutualInformation(unittest.TestCase):
    """
    Test suite for the fast_mutual_information package.

    This suite validates the MI estimation for both Normal and Negative Binomial
    distributions against their analytical solutions derived from a Gaussian copula.
    It also verifies the consistency between dense and sparse implementations.
    """

    @classmethod
    def setUpClass(cls):
        """
        Generate simulated datasets once for all tests to save time.
        This method is run once before any tests in this class are executed.
        """
        # --- Configuration ---
        cls.N_GENES = 50
        cls.N_SAMPLES = 10000
        cls.MIN_BIN_CONTENT = 50
        cls.ZI_FRACTION = 0.0

        cls.triu_indices = np.triu_indices(cls.N_GENES, k=1)

        print("\nSetting up test data for TestMutualInformation...")

        cls.analytical_mi_normal, cls.corr_normal, cls.samples_normal = get_normal(
            n_genes=cls.N_GENES, n_samples=cls.N_SAMPLES, min_bin_content=cls.MIN_BIN_CONTENT
        )
        cls.means_normal = np.mean(cls.samples_normal, axis=0)
        cls.std_devs_normal = np.std(cls.samples_normal, axis=0)

        # --- Generate Zero-Inflated Negative Binomial Data ---
        cls.r_list_nb = np.random.randint(20, 21, size=cls.N_GENES)
        cls.p_list_nb = np.random.uniform(0.3, 0.7, size=cls.N_GENES)
        cls.alphas_nb = np.full(cls.N_GENES, cls.ZI_FRACTION)

        cls.samples_nb, cls.corr_nb = gaussian_copula_negative_binomials(
            cls.N_SAMPLES, cls.r_list_nb, cls.p_list_nb, alpha=cls.ZI_FRACTION
        )
        cls.means_nb = nb_mean(cls.r_list_nb, cls.p_list_nb)

        analytical_mi_nb = -0.5 * np.log(1 - cls.corr_nb**2)
        np.fill_diagonal(analytical_mi_nb, 0)
        cls.analytical_mi_nb = analytical_mi_nb * (1 - cls.ZI_FRACTION)**2

        print("Test data setup complete.")

    def test_normal_mi_against_analytical(self):
        """
        Tests if the MI estimated for Normal marginals is close to the analytical value.
        """
        print("Running test: Normal MI vs Analytical...")

        # estimated_mi, _ = mi_normal(
        #     self.samples_normal.astype(np.int32).copy(),
        #     self.means_normal,
        #     self.std_devs_normal,
        #     min_pop=self.MIN_BIN_CONTENT
        # )

        estimated_mi, _ = mi_normal(
            self.samples_normal.astype(np.float32).copy(),
            self.means_normal,
            self.std_devs_normal,
            min_pop=self.MIN_BIN_CONTENT
        )

        num_compared_elements = estimated_mi.size // 2
        max_allowed_violations = int(num_compared_elements * 0.05)

        print(num_compared_elements, max_allowed_violations)

        assert_allclose_tolerant(
            estimated_mi[self.triu_indices],
            self.analytical_mi_normal[self.triu_indices],
            rtol=0.1,
            atol=0.1,
            max_violations=max_allowed_violations,
            err_msg="MI estimation for Normal data deviates significantly from analytical values."
        )

    def test_nb_mi_against_analytical(self):
        """
        Tests if the MI for Zero-Inflated NB marginals is close to the analytical value.
        """
        print("Running test: Zero-Inflated NB MI vs Analytical...")

        estimated_mi_nb, _ = mi_negative_binomial_zi(
            self.samples_nb.astype(np.int32).copy(),
            self.means_nb,
            self.r_list_nb.astype(np.float64),
            self.alphas_nb.astype(np.float64),
            min_pop=self.MIN_BIN_CONTENT,
        )

        num_compared_elements = estimated_mi_nb.size // 2
        max_allowed_violations = int(num_compared_elements * 0.05)

        print(num_compared_elements, max_allowed_violations)

        assert_allclose_tolerant(
            estimated_mi_nb[self.triu_indices],
            self.analytical_mi_nb[self.triu_indices],
            rtol=0.1,
            atol=0.1,
            max_violations=max_allowed_violations,
            err_msg="MI estimation for ZINB data deviates significantly from analytical values."
        )

    def test_sparse_vs_dense_normal(self):
        """
        Verifies that the sparse and dense MI functions for Normal data return identical results.
        """
        print("Running test: Sparse vs Dense for Normal MI...")

        dense_mi, _ = mi_normal(
            self.samples_normal.astype(np.int32).copy(),
            self.means_normal,
            self.std_devs_normal,
            min_pop=self.MIN_BIN_CONTENT
        )

        samples_sparse = sparse.csc_matrix(self.samples_normal.astype(np.int32).copy())
        samples_sparse.eliminate_zeros()
        sparse_mi, _ = mi_normal_sparse(
            samples_sparse,
            self.means_normal,
            self.std_devs_normal,
            min_pop=self.MIN_BIN_CONTENT
        )

        np.testing.assert_allclose(
            dense_mi[self.triu_indices],
            sparse_mi[self.triu_indices],
            rtol=0.5,
            atol=1e-3,
            err_msg="Sparse and Dense Normal MI implementations produced different results."
        )

    def test_sparse_vs_dense_nb(self):
        """
        Verifies that the sparse and dense MI functions for non-inflated NB data return identical results.
        Note: The provided C++ interface has a sparse implementation for standard NB, but not ZINB.
        Therefore, we generate a non-zero-inflated dataset for this specific test.
        """
        print("Running test: Sparse vs Dense for standard NB MI...")

        samples_nb_no_zi, _ = gaussian_copula_negative_binomials(
            self.N_SAMPLES, self.r_list_nb, self.p_list_nb, alpha=0.0
        )

        samples_nb_no_zi_s = samples_nb_no_zi[:]

        dense_mi_nb, _ = mi_negative_binomial(
            samples_nb_no_zi.astype(np.int64).copy(),
            self.means_nb,
            self.r_list_nb.astype(np.float32),
            min_pop=self.MIN_BIN_CONTENT,
        )

        samples_nb_sparse = sparse.csc_matrix(samples_nb_no_zi_s.astype(np.int32)).copy()
        samples_nb_sparse.eliminate_zeros()
        sparse_mi_nb, _ = mi_negative_binomial_sparse(
            samples_nb_sparse.astype(np.int32).copy(),
            self.means_nb,
            self.r_list_nb.astype(np.float64),
            min_pop=self.MIN_BIN_CONTENT,
        )

        np.testing.assert_allclose(
            dense_mi_nb[self.triu_indices],
            sparse_mi_nb[self.triu_indices],
            rtol=0.5,
            atol=1e-3,
            err_msg="Sparse and Dense NB MI implementations produced different results."
        )

    def test_nb_exposure_equivalence_when_exposure_one(self):
        n = self.N_SAMPLES
        p = self.N_GENES

        exposure = np.ones(n)
        # Use the same NB data generator you already use
        samples_nb_no_zi, _ = gaussian_copula_negative_binomials(
            n, self.r_list_nb, self.p_list_nb, alpha=0.0
        )

        # Non-exposure call (means as usual)
        dense_mi_nb, _ = mi_negative_binomial(
            samples_nb_no_zi.astype(np.int32).copy(),
            nb_mean(self.r_list_nb, self.p_list_nb),
            self.r_list_nb.astype(np.float64),
            min_pop=self.MIN_BIN_CONTENT,
        )

        # Exposure-aware call with mu0 chosen so that mu_j = mu0 * 1 equals the same means
        mu0 = nb_mean(self.r_list_nb, self.p_list_nb)
        exp_mi_nb, _ = mi_negative_binomial_exposure(
            samples_nb_no_zi.astype(np.int32).copy(),
            mu0.astype(np.float64),
            self.r_list_nb.astype(np.float64),
            exposure.astype(np.float64),
            min_pop=self.MIN_BIN_CONTENT,
        )

        np.testing.assert_allclose(
            dense_mi_nb[self.triu_indices],
            exp_mi_nb[self.triu_indices],
            rtol=1e-6,
            atol=1e-8,
            err_msg="Exposure path (all ones) did not coincide with non-exposure NB MI."
        )

    def test_nb_exposure_reparameterization_invariance(self):
        n = self.N_SAMPLES
        rng = np.random.default_rng(42)

        # Heterogeneous exposure, normalized once (this is fine)
        exposure = rng.lognormal(mean=0.0, sigma=0.6, size=n)
        exposure = exposure / exposure.mean()

        r = self.r_list_nb.astype(np.float64)
        mu0 = nb_mean(r, self.p_list_nb).astype(np.float64)  # baseline with mean(exposure)=1

        # Generate counts with (mu0, exposure)
        samples_nb, _ = gaussian_copula_negative_binomials_with_exposure(
            n_samples=n,
            r_list=r,
            mu0_list=mu0,
            exposure=exposure,
            alpha=0.0
        )

        # MI with (mu0, exposure)
        mi_a, _ = mi_negative_binomial_exposure(
            samples_nb.astype(np.int32).copy(),
            mu0,
            r,
            exposure.astype(np.float64),
            min_pop=self.MIN_BIN_CONTENT,
        )

        # Scale exposure by c and scale mu0 by 1/c
        c = 3.0
        exposure_scaled = exposure * c
        mu0_scaled = mu0 / c

        mi_b, _ = mi_negative_binomial_exposure(
            samples_nb.astype(np.int32).copy(),
            mu0_scaled,
            r,
            exposure_scaled.astype(np.float64),
            min_pop=self.MIN_BIN_CONTENT,
        )

        np.testing.assert_allclose(
            mi_a[self.triu_indices],
            mi_b[self.triu_indices],
            rtol=1e-8,
            atol=1e-10,
            err_msg="Exposure reparameterization invariance (mu0<-mu0/c, exposure<-exposure*c) failed."
        )

if __name__ == '__main__':
    """
    Allows running the tests directly using `python test_mi_estimation.py`.
    """
    unittest.main()
