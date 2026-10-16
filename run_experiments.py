"""Reproduce deterministic tables, coupling simulations and vector figures."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from profiles import corrected_l1, leading, mean_lower_bound, mean_gaps

ROOT = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--reuse-simulation", action="store_true",
                        help="use the supplied CSV instead of running the simulator")
    args = parser.parse_args()
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    if args.reuse_simulation:
        previous = json.loads((results / "metadata.json").read_text())
        if previous["seed"] != args.seed or previous["replicates"] != args.replicates:
            raise ValueError("seed/replicate count differs from stored simulation metadata")
    with (results / "control_means.csv").open("w", newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["n","H"]+[f"g{h}" for h in range(1,7)])
        for n in [100,1000,10000,100000]:
            for H in sorted({1,3,6,max(1,int(.35*math.log(n)))}):
                writer.writerow([n,H]+list(mean_gaps(n,H)[1:])+[0.]*(6-H))
    compiler = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
    if not args.reuse_simulation:
        with tempfile.TemporaryDirectory(prefix="lcd_port_") as tmp:
            executable = Path(tmp) / "simulate"
            subprocess.run(["g++", "-O3", "-std=c++17", "-Wall", "-Wextra",
                            str(ROOT / "simulate.cpp"), "-o", str(executable)], check=True)
            subprocess.run([str(executable), str(args.replicates), str(args.seed),
                            str(results / "coupling_mc.csv"), str(results / "control_means.csv"),
                            str(results / "factor_samples.csv")], check=True)
    with (results / "coupling_mc.csv").open() as f:
        raw_mc = list(csv.DictReader(f))
    if any(int(r["replicates"]) != args.replicates for r in raw_mc):
        raise ValueError("replicate count differs from the supplied simulation")
    enriched = []
    for r in raw_mc:
        n, H = int(r["n"]), int(r["H"])
        p = math.inf if r["p"] == "inf" else int(r["p"])
        lb, first = mean_lower_bound(n, H, p), leading(n, H, p)
        mean, se = lb+float(r["excess_mean"]), float(r["excess_se"])
        enriched.append(dict(n=n, H=H, p=r["p"], replicates=args.replicates,
                             mean_lower_bound=lb, coupling_mean=mean, mc_se=se,
                             raw_coupling_mean=float(r["mean"]),raw_mc_se=float(r["se"]),
                             excess_mean=float(r["excess_mean"]),excess_se=se,
                             leading=first, lower_over_leading=lb/first,
                             coupling_over_leading=mean/first,
                             normalized_mc_se=se/first))
    write_csv(results / "transport_bounds.csv", enriched)

    exact_rows = []
    for n in [10 ** k for k in range(2, 8)]:
        for H in range(1, 13):
            exact, first, second = mean_lower_bound(n, H, 1), leading(n, H), corrected_l1(n, H)
            exact_rows.append(dict(n=n, H=H, exact_l1=exact, leading=first,
                                   corrected=second, leading_error_percent=100*(first/exact-1),
                                   corrected_error_percent=100*(second/exact-1)))
    write_csv(results / "exact_l1.csv", exact_rows)
    table = []
    for r in exact_rows:
        if r["H"] == 3:
            power = round(math.log10(r["n"]))
            table.append(f"$10^{power}$ & {r['exact_l1']:.6f} & {r['leading']:.6f} & "
                         f"{r['corrected']:.6f} & {r['leading_error_percent']:+.2f} & "
                         f"{r['corrected_error_percent']:+.2f} \\\\")
    (results / "Table_exact.tex").write_text("\n".join(table)+"\n")
    plt.rcParams.update({"font.family":"serif", "font.size":9, "axes.titlesize":10,
                         "axes.labelsize":9, "legend.fontsize":8, "pdf.fonttype":42,
                         "ps.fonttype":42, "axes.spines.top":False,
                         "axes.spines.right":False})
    ns = np.unique(np.logspace(2, 7, 160).astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.65), layout="constrained")
    for ax, H in zip(axes, [3, 6]):
        values = np.array([mean_lower_bound(int(n), H, 1) for n in ns])
        first = np.array([leading(int(n), H) for n in ns])
        second = np.array([corrected_l1(int(n), H) for n in ns])
        ax.semilogx(ns, 100*(first/values-1), color="#225e94", label="Leading term")
        ax.semilogx(ns, 100*(second/values-1), color="#b14f25", ls="--", label="With correction")
        ax.axhline(0, color="0.65", lw=0.7)
        ax.set(xlabel=r"LCD size $n$", ylabel="Signed relative error (%)", title=f"$H={H}$")
        ax.grid(alpha=.15); ax.legend(frameon=False)
    fig.savefig(results / "Fig2.pdf"); fig.savefig(results / "Fig2.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.15), layout="constrained")
    for ax, p, title in zip(axes, ["2", "inf"], [r"Euclidean norm ($p=2$)", r"Maximum norm ($p=\infty$)"]):
        for H, color, marker in [(3,"#225e94","o"),(6,"#b14f25","s")]:
            rows = [r for r in enriched if r["p"] == p and r["H"] == H]
            x = [r["n"] for r in rows]
            ax.semilogx(x, [r["lower_over_leading"] for r in rows], ls="--", color=color,
                        label=f"Mean lower bound, $H={H}$")
            ax.errorbar(x, [r["coupling_over_leading"] for r in rows],
                        yerr=[1.96*r["normalized_mc_se"] for r in rows],
                        color=color, marker=marker, ms=3, capsize=2, lw=.9,
                        label=f"Coupling estimate, $H={H}$")
        ax.axhline(1, color="0.45", lw=.8, ls=":")
        ax.set(xlabel=r"LCD size $n$", ylabel="Cost / leading approximation", title=title)
        ax.set_xlim(80,130000)
        ax.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
    fig.savefig(results / "Fig3.pdf"); fig.savefig(results / "Fig3.png", dpi=200)
    plt.close(fig)

    rbins, cbins = np.linspace(0,5,36), np.linspace(0,3,31)
    if not args.reuse_simulation:
        samples = np.genfromtxt(results / "factor_samples.csv", delimiter=",", names=True)
        if len(samples) != args.replicates:
            raise ValueError("factor sample count differs from experiment metadata")
        R = samples["root_weight"] / np.sqrt(samples["n"])
        C = samples["branch_root_weight"] / np.sqrt(samples["n"])
        probabilities = np.r_[np.linspace(.01,.95,95), [.96,.97,.98,.99]]
        theory = 1 / np.sqrt(1-probabilities)-1
        empirical = np.quantile(C/R, probabilities)
        write_csv(results / "relative_factor_quantiles.csv", [
            dict(probability=float(p), theoretical_quantile=float(t), empirical_quantile=float(e))
            for p,t,e in zip(probabilities,theory,empirical)])
        counts,_,_ = np.histogram2d(R,C,bins=(rbins,cbins))
        write_csv(results / "joint_factor_histogram.csv", [
            dict(r_bin=i,c_bin=j,count=int(counts[i,j]))
            for i in range(len(rbins)-1) for j in range(len(cbins)-1)])
    else:
        with (results / "relative_factor_quantiles.csv").open() as f:
            quantiles=list(csv.DictReader(f))
        theory=np.array([float(r["theoretical_quantile"]) for r in quantiles])
        empirical=np.array([float(r["empirical_quantile"]) for r in quantiles])
        counts=np.zeros((len(rbins)-1,len(cbins)-1))
        with (results / "joint_factor_histogram.csv").open() as f:
            for r in csv.DictReader(f):
                counts[int(r["r_bin"]),int(r["c_bin"])]=int(r["count"])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), layout="constrained")
    axes[0].plot(theory, empirical, "o", ms=2.2, color="#225e94", label="Empirical quantiles")
    axes[0].plot([0,9],[0,9], ls="--", color="0.45", lw=.8, label="Limit law")
    axes[0].set(xlabel="Quantiles of the limit $T$", ylabel=r"Quantiles of $c_n/r_n$",
                title="Relative-factor Q–Q plot")
    axes[0].legend(frameon=False, loc="upper left"); axes[0].grid(alpha=.15)
    # Normalize by all paths, including those outside the displayed rectangle.
    density=counts/(args.replicates*np.diff(rbins)[:,None]*np.diff(cbins)[None,:])
    mesh=axes[1].pcolormesh(rbins,cbins,density.T,cmap="Blues",shading="flat",rasterized=True)
    rr,cc=np.meshgrid(np.linspace(0,5,200),np.linspace(0,3,200))
    exact=rr*(rr+cc)*np.exp(-(rr+cc)**2/4)/4
    contours=axes[1].contour(rr,cc,exact,levels=[.05,.15,.25],colors="#b14f25",linewidths=.8)
    axes[1].clabel(contours,inline=True,fontsize=7,fmt="%.2f")
    axes[1].set(xlabel=r"$r_n/\sqrt{n}$",ylabel=r"$c_n/\sqrt{n}$",title="Joint factor density")
    fig.colorbar(mesh,ax=axes[1],label="Empirical density",shrink=.8)
    fig.savefig(results / "Fig4.pdf"); fig.savefig(results / "Fig4.png",dpi=200)
    plt.close(fig)
    log_rows = [r for r in enriched if r["p"] == "inf" and
                r["H"] == max(1, int(.35*math.log(r["n"])))]
    write_csv(results / "logarithmic_window_diagnostic.csv", log_rows)
    metadata = {"seed":args.seed, "replicates":args.replicates,
                "sizes":[100,1000,10000,100000], "engine":"std::mt19937_64",
                "sampler":"64-bit rejection, then modulo; path seed = seed + replication index",
                "python":platform.python_version(), "numpy":np.__version__,
                "scipy":scipy.__version__, "matplotlib":matplotlib.__version__,
                "compiler":compiler, "simulation_reused":args.reuse_simulation,
                "control_variate":"Exact mean of a supporting linear functional; MC estimates the nonnegative residual norm(g)-q dot g",
                "uncertainty":"Pointwise mean +/- 1.96 estimated Monte Carlo standard errors",
                "transport_note":"For p>1 the Monte Carlo mean estimates a coupling upper bound, not the optimal transport distance."}
    (results / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"exact_rows":len(exact_rows), "simulation_rows":len(enriched),
                      "results_directory":str(results)}, indent=2))


if __name__ == "__main__":
    main()
