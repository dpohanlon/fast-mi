import os
import tempfile
import unittest

import numpy as np
from scipy import sparse
from scipy.stats import norm, nbinom

from fast_mutual_information import (
    mi_ml,
    mi_binarised,
    mi_normal,
    mi_negative_binomial,
    mi_negative_binomial_zi,
    mi_zero_inflated_negative_binomial_dump_first_tree,
    mi_negative_binomial_exposure,
    mi_normal_sparse,
    mi_negative_binomial_sparse,
    mi_normal_crossfit,
    mi_negative_binomial_crossfit,
    mi_zero_inflated_negative_binomial_crossfit,
)


def nb_mean(true_r, true_p):
    return true_r * (1 - true_p) / true_p


def nb_var(true_r, true_p):
    return nb_mean(true_r, true_p) / true_p


def gaussian_copula_negative_binomials(
    n_samples,
    r_list,
    p_list,
    corr_matrix=None,
    random_state=None,
    alpha=0.0,
):
    rng = np.random.default_rng(random_state)

    n_vars = len(r_list)
    if corr_matrix is None:
        n_factors = min(10, max(2, n_vars))
        A = rng.standard_normal((n_vars, n_factors))
        cov = A @ A.T
        d = np.sqrt(np.diag(cov))
        corr_matrix = cov / np.outer(d, d)
        np.fill_diagonal(corr_matrix, 1.0)

    z = rng.multivariate_normal(np.zeros(n_vars), corr_matrix, size=n_samples)
    u = norm.cdf(z)

    samples = np.zeros_like(u, dtype=np.int64)
    for i in range(n_vars):
        if alpha > 0:
            uc = np.clip((u[:, i] - alpha) / (1 - alpha), 0.0, 1.0)
            draw = nbinom.ppf(uc, r_list[i], p_list[i])
            samples[:, i] = np.where(u[:, i] < alpha, 0, draw).astype(np.int64)
        else:
            samples[:, i] = nbinom.ppf(u[:, i], r_list[i], p_list[i]).astype(np.int64)

    return samples, corr_matrix


def gaussian_copula_negative_binomials_with_exposure(
    n_samples,
    r_list,
    mu0_list,
    exposure,
    corr_matrix=None,
    random_state=None,
    alpha=0.0,
):
    rng = np.random.default_rng(random_state)

    n_vars = len(r_list)
    if corr_matrix is None:
        n_factors = min(10, max(2, n_vars))
        A = rng.standard_normal((n_vars, n_factors))
        cov = A @ A.T
        d = np.sqrt(np.diag(cov))
        corr_matrix = cov / np.outer(d, d)
        np.fill_diagonal(corr_matrix, 1.0)

    z = rng.multivariate_normal(np.zeros(n_vars), corr_matrix, size=n_samples)
    u = norm.cdf(z)

    samples = np.zeros_like(u, dtype=np.int64)
    for i in range(n_vars):
        r = r_list[i]
        mu0 = mu0_list[i]
        m_j = mu0 * exposure
        p_j = r / (m_j + r)
        if alpha > 0:
            uc = np.clip((u[:, i] - alpha) / (1 - alpha), 0.0, 1.0)
            draw = np.array([nbinom.ppf(uc[j], r, p_j[j]) for j in range(n_samples)], dtype=np.int64)
            samples[:, i] = np.where(u[:, i] < alpha, 0, draw).astype(np.int64)
        else:
            samples[:, i] = np.array([nbinom.ppf(u[j, i], r, p_j[j]) for j in range(n_samples)], dtype=np.int64)

    return samples, corr_matrix


def get_normal(n_genes=24, n_samples=4000, std_dev=10.0, random_state=0):
    rng = np.random.default_rng(random_state)
    n_factors = min(5, max(2, n_genes // 2))
    B = rng.standard_normal((n_genes, n_factors))
    cov = B @ B.T
    d = np.sqrt(np.diag(cov))
    corr_matrix = cov / np.outer(d, d)
    np.fill_diagonal(corr_matrix, 1.0)

    samples_n = rng.multivariate_normal(np.zeros(n_genes), corr_matrix, size=n_samples)
    samples_n = samples_n * std_dev
    samples_n = samples_n + abs(samples_n.min()) + 1.0

    mi_analytical = -0.5 * np.log(np.maximum(1.0 - corr_matrix ** 2, 1e-12))
    np.fill_diagonal(mi_analytical, 0.0)

    return mi_analytical, corr_matrix, samples_n


def make_two_feature_corr_matrix(rho):
    return np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)


def upper_tri_values(x):
    idx = np.triu_indices(x.shape[0], k=1)
    return x[idx]


def assert_allclose_tolerant(actual, desired, rtol=1e-7, atol=0.0, max_violations=0, err_msg=""):
    violations_mask = np.abs(actual - desired) > (atol + rtol * np.abs(desired))
    num_violations = int(np.sum(violations_mask))

    if num_violations > max_violations:
        max_abs_diff = np.max(np.abs(actual[violations_mask] - desired[violations_mask]))
        raise AssertionError(
            f"{err_msg}\n"
            f"Validation failed: found {num_violations} violations, "
            f"allowed {max_violations}. "
            f"Maximum absolute difference among violations: {max_abs_diff}"
        )


class TestMutualInformation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(12345)

        cls.N_GENES = 24
        cls.N_SAMPLES = 4000
        cls.MIN_BIN_CONTENT = 40
        cls.ZI_FRACTION = 0.25
        cls.triu_indices = np.triu_indices(cls.N_GENES, k=1)

        cls.analytical_mi_normal, cls.corr_normal, cls.samples_normal = get_normal(
            n_genes=cls.N_GENES,
            n_samples=cls.N_SAMPLES,
            std_dev=10.0,
            random_state=1,
        )
        cls.means_normal = np.mean(cls.samples_normal, axis=0)
        cls.std_devs_normal = np.std(cls.samples_normal, axis=0, ddof=0)

        cls.r_list_nb = cls.rng.integers(12, 28, size=cls.N_GENES).astype(np.float64)
        cls.p_list_nb = cls.rng.uniform(0.35, 0.7, size=cls.N_GENES)
        cls.alphas_nb = np.full(cls.N_GENES, cls.ZI_FRACTION, dtype=np.float64)

        cls.samples_nb, cls.corr_nb = gaussian_copula_negative_binomials(
            cls.N_SAMPLES,
            cls.r_list_nb,
            cls.p_list_nb,
            random_state=2,
            alpha=0.0,
        )
        cls.samples_zinb, cls.corr_zinb = gaussian_copula_negative_binomials(
            cls.N_SAMPLES,
            cls.r_list_nb,
            cls.p_list_nb,
            random_state=3,
            alpha=cls.ZI_FRACTION,
        )

        cls.means_nb = nb_mean(cls.r_list_nb, cls.p_list_nb).astype(np.float64)
        cls.vars_nb = nb_var(cls.r_list_nb, cls.p_list_nb).astype(np.float64)

        analytical_mi_nb = -0.5 * np.log(np.maximum(1.0 - cls.corr_nb ** 2, 1e-12))
        np.fill_diagonal(analytical_mi_nb, 0.0)
        cls.analytical_mi_nb = analytical_mi_nb

        analytical_mi_zinb = -0.5 * np.log(np.maximum(1.0 - cls.corr_zinb ** 2, 1e-12))
        np.fill_diagonal(analytical_mi_zinb, 0.0)
        cls.analytical_mi_zinb = analytical_mi_zinb * (1 - cls.ZI_FRACTION) ** 2

        cls.exposure = cls.rng.lognormal(mean=0.0, sigma=0.5, size=cls.N_SAMPLES).astype(np.float64)
        cls.exposure /= cls.exposure.mean()

        cls.samples_nb_exposure, cls.corr_nb_exposure = gaussian_copula_negative_binomials_with_exposure(
            n_samples=cls.N_SAMPLES,
            r_list=cls.r_list_nb,
            mu0_list=cls.means_nb,
            exposure=cls.exposure,
            corr_matrix=cls.corr_nb,
            random_state=4,
            alpha=0.0,
        )

    def test_normal_mi_against_analytical(self):
        estimated_mi, _ = mi_normal(
            self.samples_normal.astype(np.float32),
            self.means_normal.astype(np.float32),
            self.std_devs_normal.astype(np.float32),
            min_pop=self.MIN_BIN_CONTENT,
        )

        num_compared_elements = len(upper_tri_values(estimated_mi))
        max_allowed_violations = int(num_compared_elements * 0.08)

        assert_allclose_tolerant(
            upper_tri_values(estimated_mi),
            upper_tri_values(self.analytical_mi_normal),
            rtol=0.12,
            atol=0.12,
            max_violations=max_allowed_violations,
            err_msg="Normal MI deviates materially from analytical Gaussian-copula values.",
        )

    def test_normal_pair_interface(self):
        rho = 0.7
        corr = make_two_feature_corr_matrix(rho)
        rng = np.random.default_rng(10)
        samples = rng.multivariate_normal([0.0, 0.0], corr, size=5000)
        samples = samples * 4.0 + 15.0

        mean1, mean2 = samples.mean(axis=0)
        std1, std2 = samples.std(axis=0, ddof=0)

        mi_pair, chi2_pair = mi_normal(
            float(mean1),
            float(std1),
            float(mean2),
            float(std2),
            samples.astype(np.float32),
            min_pop=50,
        )

        expected = -0.5 * np.log(1 - rho ** 2)
        self.assertGreater(mi_pair, 0.0)
        self.assertAlmostEqual(mi_pair, expected, delta=0.15)
        self.assertGreater(chi2_pair, 0.0)

    def test_normal_dense_float32_vs_int64_inputs(self):
        mi_float, _ = mi_normal(
            self.samples_normal.astype(np.float32),
            self.means_normal.astype(np.float32),
            self.std_devs_normal.astype(np.float32),
            min_pop=self.MIN_BIN_CONTENT,
        )

        quantised = np.rint(self.samples_normal).astype(np.int64)
        mi_int, _ = mi_normal(
            quantised,
            self.means_normal.astype(np.float64),
            self.vars_nb[: self.N_GENES].astype(np.float64),
            min_pop=self.MIN_BIN_CONTENT,
        )

        self.assertEqual(mi_float.shape, mi_int.shape)
        self.assertTrue(np.isfinite(mi_float).all())
        self.assertTrue(np.isfinite(mi_int).all())

    def test_nb_mi_against_analytical(self):
        estimated_mi_nb, _ = mi_negative_binomial(
            self.samples_nb.astype(np.int64),
            self.means_nb.astype(np.float32),
            self.r_list_nb.astype(np.float32),
            min_pop=self.MIN_BIN_CONTENT,
        )

        num_compared_elements = len(upper_tri_values(estimated_mi_nb))
        max_allowed_violations = int(num_compared_elements * 0.08)

        assert_allclose_tolerant(
            upper_tri_values(estimated_mi_nb),
            upper_tri_values(self.analytical_mi_nb),
            rtol=0.15,
            atol=0.12,
            max_violations=max_allowed_violations,
            err_msg="NB MI deviates materially from analytical Gaussian-copula values.",
        )

    def test_zinb_mi_against_analytical(self):
        estimated_mi_zinb, _ = mi_negative_binomial_zi(
            self.samples_zinb.astype(np.int64),
            self.means_nb.astype(np.float32),
            self.r_list_nb.astype(np.float32),
            self.alphas_nb.astype(np.float32),
            min_pop=self.MIN_BIN_CONTENT,
        )

        num_compared_elements = len(upper_tri_values(estimated_mi_zinb))
        max_allowed_violations = int(num_compared_elements * 0.10)

        assert_allclose_tolerant(
            upper_tri_values(estimated_mi_zinb),
            upper_tri_values(self.analytical_mi_zinb),
            rtol=0.2,
            atol=0.15,
            max_violations=max_allowed_violations,
            err_msg="ZINB MI deviates materially from analytical reference values.",
        )

    def test_sparse_vs_dense_normal(self):
        dense_mi, _ = mi_normal(
            self.samples_normal.astype(np.int32),
            self.means_normal.astype(np.float64),
            self.std_devs_normal.astype(np.float64),
            min_pop=self.MIN_BIN_CONTENT,
        )

        samples_sparse = sparse.csc_matrix(np.rint(self.samples_normal).astype(np.int32))
        samples_sparse.eliminate_zeros()
        sparse_mi, _ = mi_normal_sparse(
            samples_sparse,
            self.means_normal.astype(np.float32),
            self.std_devs_normal.astype(np.float32),
            min_pop=self.MIN_BIN_CONTENT,
        )

        np.testing.assert_allclose(
            upper_tri_values(dense_mi),
            upper_tri_values(sparse_mi),
            rtol=0.5,
            atol=1e-3,
            err_msg="Sparse and dense normal MI disagree.",
        )

    def test_sparse_vs_dense_nb(self):
        dense_mi_nb, _ = mi_negative_binomial(
            self.samples_nb.astype(np.int64),
            self.means_nb.astype(np.float32),
            self.r_list_nb.astype(np.float32),
            min_pop=self.MIN_BIN_CONTENT,
        )

        samples_nb_sparse = sparse.csc_matrix(self.samples_nb.astype(np.int32))
        samples_nb_sparse.eliminate_zeros()
        sparse_mi_nb, _ = mi_negative_binomial_sparse(
            samples_nb_sparse,
            self.means_nb.astype(np.float64),
            self.r_list_nb.astype(np.float64),
            min_pop=self.MIN_BIN_CONTENT,
        )

        np.testing.assert_allclose(
            upper_tri_values(dense_mi_nb),
            upper_tri_values(sparse_mi_nb),
            rtol=0.5,
            atol=1e-3,
            err_msg="Sparse and dense NB MI disagree.",
        )

    def test_nb_exposure_equivalence_when_exposure_one(self):
        exposure = np.ones(self.N_SAMPLES, dtype=np.float64)

        dense_mi_nb, _ = mi_negative_binomial(
            self.samples_nb.astype(np.int32),
            self.means_nb.astype(np.float64),
            self.r_list_nb.astype(np.float64),
            min_pop=self.MIN_BIN_CONTENT,
        )

        exp_mi_nb, _ = mi_negative_binomial_exposure(
            self.samples_nb.astype(np.int32),
            self.means_nb.astype(np.float32),
            self.r_list_nb.astype(np.float32),
            exposure.astype(np.float32),
            min_pop=self.MIN_BIN_CONTENT,
        )

        np.testing.assert_allclose(
            upper_tri_values(dense_mi_nb),
            upper_tri_values(exp_mi_nb),
            rtol=1e-6,
            atol=1e-8,
            err_msg="Exposure-aware NB does not match standard NB when exposure is all ones.",
        )

    def test_nb_exposure_reparameterization_invariance(self):
        mi_a, _ = mi_negative_binomial_exposure(
            self.samples_nb_exposure.astype(np.int64),
            self.means_nb.astype(np.float64),
            self.r_list_nb.astype(np.float64),
            self.exposure.astype(np.float64),
            min_pop=self.MIN_BIN_CONTENT,
        )

        c = 3.0
        exposure_scaled = self.exposure * c
        mu0_scaled = self.means_nb / c

        mi_b, _ = mi_negative_binomial_exposure(
            self.samples_nb_exposure.astype(np.int64),
            mu0_scaled.astype(np.float32),
            self.r_list_nb.astype(np.float32),
            exposure_scaled.astype(np.float32),
            min_pop=self.MIN_BIN_CONTENT,
        )

        np.testing.assert_allclose(
            upper_tri_values(mi_a),
            upper_tri_values(mi_b),
            rtol=1e-7,
            atol=1e-8,
            err_msg="Exposure reparameterization invariance failed."
        )

    def test_mi_ml_detects_dependence(self):
        rng = np.random.default_rng(20)
        n = 6000
        x = rng.integers(0, 5, size=n, dtype=np.int32)
        y_ind = rng.integers(0, 5, size=n, dtype=np.int32)
        y_dep = (x + rng.integers(0, 2, size=n, dtype=np.int32)) % 5

        data_ind = np.column_stack([x, y_ind]).astype(np.int64)
        data_dep = np.column_stack([x, y_dep]).astype(np.int64)

        mi_ind = mi_ml(data_ind)[0, 1]
        mi_dep = mi_ml(data_dep)[0, 1]

        self.assertLess(mi_ind, 0.05)
        self.assertGreater(mi_dep, mi_ind + 0.1)

    def test_mi_binarised_detects_dependence(self):
        rng = np.random.default_rng(21)
        n = 8000
        x = rng.integers(0, 2, size=n, dtype=np.int64)
        y_ind = rng.integers(0, 2, size=n, dtype=np.int64)
        flips = rng.random(n) < 0.1
        y_dep = np.where(flips, 1 - x, x)

        mi_ind = mi_binarised(np.column_stack([x, y_ind]))[0, 1]
        mi_dep = mi_binarised(np.column_stack([x, y_dep]))[0, 1]

        self.assertLess(mi_ind, 0.02)
        self.assertGreater(mi_dep, 0.2)

    def test_dump_first_tree_interface(self):
        samples = self.samples_zinb[:, :4].astype(np.int32)
        means = self.means_nb[:4].astype(np.float64)
        concs = self.r_list_nb[:4].astype(np.float64)
        alphas = self.alphas_nb[:4].astype(np.float64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tree.csv")
            out = mi_zero_inflated_negative_binomial_dump_first_tree(
                samples,
                means,
                concs,
                alphas,
                path,
                min_pop=self.MIN_BIN_CONTENT,
            )
            self.assertEqual(out, 0)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_normal_crossfit_chi2_independent_vs_correlated(self):
        n = 5000
        rng = np.random.default_rng(30)

        corr_ind = make_two_feature_corr_matrix(0.0)
        corr_dep = make_two_feature_corr_matrix(0.8)

        x_ind = rng.multivariate_normal([0.0, 0.0], corr_ind, size=n)
        x_dep = rng.multivariate_normal([0.0, 0.0], corr_dep, size=n)

        x_ind = np.rint((x_ind - x_ind.min()) * 3.0 + 4.0).astype(np.int64)
        x_dep = np.rint((x_dep - x_dep.min()) * 3.0 + 4.0).astype(np.int64)

        means_ind = x_ind.mean(axis=0).astype(np.float64)
        stds_ind = x_ind.std(axis=0, ddof=0).astype(np.float64)
        means_dep = x_dep.mean(axis=0).astype(np.float64)
        stds_dep = x_dep.std(axis=0, ddof=0).astype(np.float64)

        mi_ind, chi2_ind, dof_ind = mi_normal_crossfit(
            x_ind,
            means_ind.astype(np.float32),
            stds_ind.astype(np.float32),
            min_pop=50,
            seed=7,
            min_expected=5.0,
        )
        mi_dep, chi2_dep, dof_dep = mi_normal_crossfit(
            x_dep.astype(np.int32),
            means_dep,
            stds_dep,
            min_pop=50,
            seed=7,
            min_expected=5.0,
        )

        red_ind = chi2_ind[0, 1] / max(dof_ind[0, 1], 1.0)
        red_dep = chi2_dep[0, 1] / max(dof_dep[0, 1], 1.0)

        self.assertLess(red_ind, 3.0)
        self.assertGreater(red_dep, red_ind + 5.0)
        self.assertGreater(mi_dep[0, 1], mi_ind[0, 1] + 0.1)

    def test_nb_crossfit_chi2_independent_vs_correlated(self):
        n = 5000
        r = np.array([18.0, 23.0])
        p = np.array([0.45, 0.6])
        means = nb_mean(r, p)

        ind_samples, _ = gaussian_copula_negative_binomials(
            n,
            r,
            p,
            corr_matrix=make_two_feature_corr_matrix(0.0),
            random_state=40,
            alpha=0.0,
        )
        dep_samples, _ = gaussian_copula_negative_binomials(
            n,
            r,
            p,
            corr_matrix=make_two_feature_corr_matrix(0.8),
            random_state=41,
            alpha=0.0,
        )

        mi_ind, chi2_ind, dof_ind = mi_negative_binomial_crossfit(
            ind_samples.astype(np.int64),
            means.astype(np.float32),
            r.astype(np.float32),
            min_pop=50,
            seed=11,
            min_expected=5.0,
            use_empirical_marginals=False,
            empirical_pseudocount=0.5,
        )
        mi_dep, chi2_dep, dof_dep = mi_negative_binomial_crossfit(
            dep_samples.astype(np.int64),
            means.astype(np.float64),
            r.astype(np.float64),
            min_pop=50,
            seed=11,
            min_expected=5.0,
            use_empirical_marginals=False,
            empirical_pseudocount=0.5,
        )

        red_ind = chi2_ind[0, 1] / max(dof_ind[0, 1], 1.0)
        red_dep = chi2_dep[0, 1] / max(dof_dep[0, 1], 1.0)

        self.assertLess(red_ind, 3.0)
        self.assertGreater(red_dep, red_ind + 5.0)
        self.assertGreater(mi_dep[0, 1], mi_ind[0, 1] + 0.05)

    def test_zinb_crossfit_chi2_independent_vs_correlated(self):
        n = 5000
        r = np.array([16.0, 22.0])
        p = np.array([0.4, 0.55])
        means = nb_mean(r, p)
        alphas = np.array([0.3, 0.3], dtype=np.float64)

        ind_samples, _ = gaussian_copula_negative_binomials(
            n,
            r,
            p,
            corr_matrix=make_two_feature_corr_matrix(0.0),
            random_state=50,
            alpha=0.3,
        )
        dep_samples, _ = gaussian_copula_negative_binomials(
            n,
            r,
            p,
            corr_matrix=make_two_feature_corr_matrix(0.8),
            random_state=51,
            alpha=0.3,
        )

        mi_ind, chi2_ind, dof_ind = mi_zero_inflated_negative_binomial_crossfit(
            ind_samples.astype(np.int64),
            means.astype(np.float32),
            r.astype(np.float32),
            alphas.astype(np.float32),
            min_pop=50,
            seed=13,
            min_expected=5.0,
        )
        mi_dep, chi2_dep, dof_dep = mi_zero_inflated_negative_binomial_crossfit(
            dep_samples.astype(np.int64),
            means.astype(np.float64),
            r.astype(np.float64),
            alphas.astype(np.float64),
            min_pop=50,
            seed=13,
            min_expected=5.0,
        )

        red_ind = chi2_ind[0, 1] / max(dof_ind[0, 1], 1.0)
        red_dep = chi2_dep[0, 1] / max(dof_dep[0, 1], 1.0)

        self.assertLess(red_ind, 3.5)
        self.assertGreater(red_dep, red_ind + 4.0)
        self.assertGreater(mi_dep[0, 1], mi_ind[0, 1] + 0.03)

    def test_nb_crossfit_empirical_marginals_consistent_with_parametric(self):
        samples = self.samples_nb[:, :6].astype(np.int64)
        means = self.means_nb[:6].astype(np.float64)
        concs = self.r_list_nb[:6].astype(np.float64)

        mi_fit, chi2_fit, dof_fit = mi_negative_binomial_crossfit(
            samples,
            means,
            concs,
            min_pop=self.MIN_BIN_CONTENT,
            seed=19,
            min_expected=5.0,
            use_empirical_marginals=False,
            empirical_pseudocount=0.5,
        )
        mi_emp, chi2_emp, dof_emp = mi_negative_binomial_crossfit(
            samples.astype(np.int32),
            means.astype(np.float32),
            concs.astype(np.float32),
            min_pop=self.MIN_BIN_CONTENT,
            seed=19,
            min_expected=5.0,
            use_empirical_marginals=True,
            empirical_pseudocount=0.5,
        )

        np.testing.assert_allclose(
            upper_tri_values(mi_fit),
            upper_tri_values(mi_emp),
            rtol=0.25,
            atol=0.08,
            err_msg="Empirical-marginal NB crossfit MI is not consistent with fitted-marginal MI.",
        )

        np.testing.assert_allclose(
            upper_tri_values(chi2_fit),
            upper_tri_values(chi2_emp),
            rtol=0.35,
            atol=15.0,
            err_msg="Empirical-marginal NB crossfit chi2 is not consistent with fitted-marginal chi2.",
        )

        np.testing.assert_allclose(
            upper_tri_values(dof_fit),
            upper_tri_values(dof_emp),
            rtol=0.25,
            atol=5.0,
            err_msg="Empirical-marginal NB crossfit dof is not consistent with fitted-marginal dof.",
        )

    def test_nb_crossfit_empirical_marginals_independent_vs_correlated(self):
        n = 5000
        r = np.array([15.0, 19.0])
        p = np.array([0.42, 0.58])
        means = nb_mean(r, p)

        ind_samples, _ = gaussian_copula_negative_binomials(
            n,
            r,
            p,
            corr_matrix=make_two_feature_corr_matrix(0.0),
            random_state=60,
            alpha=0.0,
        )
        dep_samples, _ = gaussian_copula_negative_binomials(
            n,
            r,
            p,
            corr_matrix=make_two_feature_corr_matrix(0.75),
            random_state=61,
            alpha=0.0,
        )

        mi_ind, chi2_ind, dof_ind = mi_negative_binomial_crossfit(
            ind_samples.astype(np.int64),
            means.astype(np.float64),
            r.astype(np.float64),
            min_pop=50,
            seed=23,
            min_expected=5.0,
            use_empirical_marginals=True,
            empirical_pseudocount=0.5,
        )
        mi_dep, chi2_dep, dof_dep = mi_negative_binomial_crossfit(
            dep_samples.astype(np.int64),
            means.astype(np.float64),
            r.astype(np.float64),
            min_pop=50,
            seed=23,
            min_expected=5.0,
            use_empirical_marginals=True,
            empirical_pseudocount=0.5,
        )

        red_ind = chi2_ind[0, 1] / max(dof_ind[0, 1], 1.0)
        red_dep = chi2_dep[0, 1] / max(dof_dep[0, 1], 1.0)

        self.assertLess(red_ind, 3.0)
        self.assertGreater(red_dep, red_ind + 5.0)
        self.assertGreater(mi_dep[0, 1], mi_ind[0, 1] + 0.05)


if __name__ == "__main__":
    unittest.main()
