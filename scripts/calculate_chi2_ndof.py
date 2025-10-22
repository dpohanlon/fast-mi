import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist
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

from tqdm import tqdm

import fast_mutual_information as fmi


def upper_triangular_values(m):
    idx = np.triu_indices_from(m, k=1)
    return m[idx]


def feasible_feature_count(N, target_features, bytes_per_entry, max_bytes):
    return N
    # if N <= 0:
    #     return 0
    # cap = int(max_bytes // (N * bytes_per_entry))
    # return max(2, min(target_features, cap))


def fit_chi2_ndof(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    if values.size == 0:
        return float("nan")
    df, loc, scale = chi2_dist.fit(values, floc=0.0, fscale=1.0)
    return float(df)


def plot_chi2(values, df, title, out_path):
    values = np.asarray(values)
    xmin = 0.0
    xmax = np.percentile(values, 99.5)
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = values.max() if values.size else 1.0
    xs = np.linspace(xmin, xmax, 400)

    plt.figure()
    plt.hist(values, bins="auto", density=True)
    plt.plot(xs, chi2_dist.pdf(xs, df))
    plt.title(title)
    plt.xlabel("chi2")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    plt.clf()


def plot_mi(values, df, title, out_path):
    values = np.asarray(values)
    xmin = 0.0
    xmax = np.percentile(values, 99.5)
    if not np.isfinite(xmax) or xmax <= 0:
        xmax = values.max() if values.size else 1.0
    xs = np.linspace(xmin, xmax, 400)

    plt.figure()
    plt.hist(values, bins="auto", density=True)
    plt.title(title)
    plt.xlabel("mi")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    plt.clf()


def simulate_normal(N, F, mean, std):
    rng = np.random.default_rng(12345)
    data = rng.normal(loc=mean, scale=std, size=(N, F))
    means = np.full(F, mean, dtype=np.float64)
    stds = np.full(F, std, dtype=np.float64)
    return data, means, stds


def simulate_nb(N, F, mu, k):
    rng = np.random.default_rng(12345)
    p = k / (k + mu)
    data = rng.negative_binomial(n=k, p=p, size=(N, F)).astype(np.int32, copy=False)
    means = np.full(F, float(mu), dtype=np.float64)
    concentrations = np.full(F, float(k), dtype=np.float64)
    return data, means, concentrations


def dynamic_min_pop(N):
    return max(5, min(25, N // 10))


def run(
    out_json="ndof_estimates.json",
    out_dir="chi2_plots",
    N_list=(25, 100, 200, 300, 500, 1000, 2000),  # , 5000, 10000, 50000, 100000),
    target_features=1000,
    normal_mean=0.0,
    normal_std=1.0,
    nb_mu=5.0,
    nb_concentration=10.0,
    max_bytes_normal=800_000_000,
    max_bytes_nb=400_000_000,
):
    out = {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for N in tqdm(N_list):
        resN = {}
        # Normal
        F_norm = feasible_feature_count(
            N, target_features, bytes_per_entry=8, max_bytes=max_bytes_normal
        )
        data_norm, means_norm, stds_norm = simulate_normal(
            N, F_norm, normal_mean, normal_std
        )
        # mp = dynamic_min_pop(N)
        mp = 25
        mi_norm, chi2_norm = fmi.mi_normal(data_norm, means_norm, stds_norm, mp)

        chi2_vals_norm = upper_triangular_values(np.asarray(chi2_norm))
        df_norm = fit_chi2_ndof(chi2_vals_norm)
        plot_chi2(
            chi2_vals_norm,
            df_norm,
            title=f"Normal marginals: N={N}, F={F_norm}, ndof={df_norm:.2f}",
            out_path=out_dir / f"chi2_normal_N{N}_F{F_norm}.png",
        )
        plot_mi(
            upper_triangular_values(np.asarray(mi_norm)),
            df_norm,
            title=f"Normal marginals: N={N}, F={F_norm}, ndof={df_norm:.2f}",
            out_path=out_dir / f"mi_normal_N{N}_F{F_norm}.png",
        )
        resN["normal"] = {
            "ndof": df_norm,
            "n_features": int(F_norm),
            "n_pairs": int(F_norm * (F_norm - 1) // 2),
        }

        # Negative Binomial
        F_nb = feasible_feature_count(
            N, target_features, bytes_per_entry=4, max_bytes=max_bytes_nb
        )
        data_nb, means_nb, conc_nb = simulate_nb(N, F_nb, nb_mu, nb_concentration)
        mi_nb, chi2_nb = fmi.mi_negative_binomial(data_nb, means_nb, conc_nb, mp)
        chi2_vals_nb = upper_triangular_values(np.asarray(chi2_nb))
        df_nb = fit_chi2_ndof(chi2_vals_nb)
        plot_chi2(
            chi2_vals_nb,
            df_nb,
            title=f"NB marginals: N={N}, F={F_nb}, ndof={df_nb:.2f}",
            out_path=out_dir / f"chi2_nb_N{N}_F{F_nb}.png",
        )
        plot_mi(
            upper_triangular_values(np.asarray(mi_nb)),
            df_nb,
            title=f"NB marginals: N={N}, F={F_nb}, ndof={df_nb:.2f}",
            out_path=out_dir / f"mi_nb_N{N}_F{F_nb}.png",
        )
        resN["negative_binomial"] = {
            "ndof": df_nb,
            "n_features": int(F_nb),
            "n_pairs": int(F_nb * (F_nb - 1) // 2),
        }

        out[int(N)] = resN

        del data_norm, mi_norm, chi2_norm, data_nb, mi_nb, chi2_nb

    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    run()
