import os
import json
from glob import glob

import pingouin as pg
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, sem, exponnorm
from scipy.stats import linregress, ttest_1samp, skew, lognorm, wilcoxon


# ============================================================
# Helpers
# ============================================================

def img_to_int(name: str) -> int:
    """
    Robustly extract an integer index from things like:
      '011.png', 'bm_sad_11', 'bm_sad_011.png'
    """
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]       # '011' or 'bm_sad_11'
    token = stem.split("_")[-1]            # '011' or '11'
    return int(token)


def sort_leftnames(cols):
    """Sort a list of column names by their numeric index (using img_to_int)."""
    return sorted(cols, key=img_to_int)


# Expression ordering: sad -> neutral -> happy
EXPR_RANK = {
    "sad": 0,
    "angry": 0,
    "neutral": 1,
    "neu": 1,
    "happy": 2,
    "hap": 2,
}


def parse_identity_expr_idx(label: str):
    """
    Parse a column label of the form 'bm_sad_11' (or similar) into:
      identity='bm', expr='sad', expr_rank, img_idx=11
    """
    base = os.path.basename(label)
    stem = os.path.splitext(base)[0]
    parts = stem.split("_")

    if len(parts) >= 3:
        identity = parts[0]
        expr = parts[1]
        idx_str = parts[-1]
    else:
        identity = parts[0]
        expr = ""
        idx_str = parts[-1]

    expr_rank = EXPR_RANK.get(expr, 1)
    img_idx = int(idx_str)
    return identity, expr, expr_rank, img_idx


def build_symmetric_stimulus_order(cols):
    """
    Given a list of labels (e.g. ['bm_sad_11', 'bm_happy_11', ...]),
    build a symmetric x-axis order:

      Left  side: all SAD stimuli (expr_rank==0), sorted by *decreasing* index.
      Right side: all HAPPY stimuli (expr_rank==2), sorted by *increasing* index.
      Middle: any NEUTRAL / other expressions (if present), sorted by index.

    Returns:
      ordered_cols: reordered column names
      xtick_labels: labels 'identity_expr_index', e.g. 'bm_sad_11'
    """
    parsed = []
    for c in cols:
        identity, expr, expr_rank, img_idx = parse_identity_expr_idx(c)
        parsed.append((identity, expr, expr_rank, img_idx, c))

    sad_items    = [p for p in parsed if p[2] == 0]
    happy_items  = [p for p in parsed if p[2] == 2]
    middle_items = [p for p in parsed if p[2] not in (0, 2)]

    sad_items.sort(key=lambda p: -p[3])      # decreasing index
    happy_items.sort(key=lambda p: p[3])     # increasing index
    middle_items.sort(key=lambda p: p[3])

    ordered = sad_items + middle_items + happy_items

    ordered_cols = []
    xtick_labels = []
    for identity, expr, expr_rank, img_idx, col in ordered:
        ordered_cols.append(col)
        xtick_labels.append(f"{identity}_{expr}_{img_idx}")
    return ordered_cols, xtick_labels


# ============================================================
# Load JSON -> block matrices
# ============================================================

def build_block_matrices(data_dir: str = "./submissions/pairwise"):
    """
    Returns:
      pairwise_mat: rows=user_id, cols='[folder]_[index]', values=rt_ms
                    (block == 'pairwise_early' --> Before PAQ)
      stage2_mat:   rows=user_id, cols='[folder]_[index]', values=rt_ms
                    (block == 'stage2_from_paq' --> After PAQ)

    Example JSON entry:
      {
        "block": "pairwise_early",
        "rt_ms": 2698,
        "folder": "bm_sad",
        "left_name": "011.png",
        ...
      }

    -> column key 'bm_sad_11' with value 2698.
    """
    pairwise_dict = {}
    stage2_dict = {}

    for file in glob(os.path.join(data_dir, "*.json")):
        with open(file, "r") as f:
            data = json.load(f)

        for user_id, entries in data.items():
            pw = {}
            st2 = {}

            for e in entries:
                blk = e.get("block")
                folder = e.get("folder", "")
                left = e.get("left_name")
                rt = e.get("rt_ms", np.nan)

                if not folder or not left:
                    continue

                idx = img_to_int(left)             # 011.png -> 11
                col_key = f"{folder}_{idx}"        # 'bm_sad_11'

                if blk == "pairwise_early":
                    pw[col_key] = rt
                elif blk == "stage2_from_paq":
                    st2[col_key] = rt

            if pw:
                pairwise_dict[user_id] = pw
            if st2:
                stage2_dict[user_id] = st2

    pairwise_mat = pd.DataFrame.from_dict(pairwise_dict, orient="index")
    stage2_mat   = pd.DataFrame.from_dict(stage2_dict, orient="index")

    if not pairwise_mat.empty:
        pairwise_mat = pairwise_mat.reindex(columns=sort_leftnames(pairwise_mat.columns))
    if not stage2_mat.empty:
        stage2_mat = stage2_mat.reindex(columns=sort_leftnames(stage2_mat.columns))

    return pairwise_mat, stage2_mat


# ============================================================
# Plot: RT vs stimuli per user (Before vs After PAQ)
# ============================================================

def plot_user_shared_index_with_identity(pairwise_mat: pd.DataFrame,
                                         stage2_mat: pd.DataFrame,
                                         save_dir: str = "./figures/rt_vs_stimuli"):
    """
    One figure per user.

    For each user:
      - Start from reference faces present in BOTH blocks somewhere.
      - Drop columns where this user has no data in either block.
      - Order remaining columns symmetrically (sad ↓ -> happy ↑).
      - X-ticks: 'identity_expression_index', e.g. 'bm_sad_11'.

    Figures saved as: {save_dir}/{user_id}_RT_vs_stimuli.png
    """
    os.makedirs(save_dir, exist_ok=True)

    users = pairwise_mat.index.intersection(stage2_mat.index)
    global_common_cols = list(set(pairwise_mat.columns) & set(stage2_mat.columns))
    if not global_common_cols:
        print("No common reference faces between pairwise_early and stage2_from_paq.")
        return

    for user_id in users:
        pw_full  = pairwise_mat.loc[user_id].reindex(global_common_cols)
        st2_full = stage2_mat.loc[user_id].reindex(global_common_cols)

        # keep columns where at least one condition has data
        mask = ~(pw_full.isna() & st2_full.isna())
        user_cols = [c for c, keep in zip(global_common_cols, mask) if keep]
        if not user_cols:
            continue

        ordered_cols, xtick_labels = build_symmetric_stimulus_order(user_cols)
        pw_row  = pw_full[ordered_cols]
        st2_row = st2_full[ordered_cols]

        x = np.arange(1, len(ordered_cols) + 1)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(x,
                pw_row.values.astype(float),
                marker='o', linewidth=2,
                color='blue', label='Before PAQ')
        ax.plot(x,
                st2_row.values.astype(float),
                marker='o', linewidth=2,
                color='red', label='After PAQ')

        ax.set_xlabel("Reference face stimuli (identity + expression + index; sad ↓ → happy ↑)")
        ax.set_ylabel("Response Time (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(loc='upper right')
        plt.title(f"User {user_id} — RT vs Reference Stimuli (Before vs After PAQ)")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{user_id}_RT_vs_stimuli.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved RT-vs-stimuli figure for user {user_id}: {save_path}")
        plt.close(fig)


# ============================================================
# Convert matrices to long DataFrame for distributions
# ============================================================

def build_long_dataframe(pairwise_mat: pd.DataFrame,
                         stage2_mat: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a long-form DataFrame with columns:
      user_id, condition ('Before','After'), identity, expression, index, rt_ms
    """
    records = []

    # helper to push from wide matrix to long
    def add_records_from_matrix(mat: pd.DataFrame, condition_label: str):
        for user_id, row in mat.iterrows():
            for col, rt in row.items():
                if pd.isna(rt):
                    continue
                identity, expr, expr_rank, idx = parse_identity_expr_idx(col)
                records.append({
                    "user_id": user_id,
                    "condition": condition_label,
                    "identity": identity,
                    "expression": expr,
                    "index": idx,
                    "rt_ms": float(rt),
                })

    add_records_from_matrix(pairwise_mat, "Before")
    add_records_from_matrix(stage2_mat, "After")

    df = pd.DataFrame.from_records(records)
    return df


# ============================================================
# Summary stats per participant × condition
# ============================================================

def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per user × condition, compute mean, sd, skew of RTs.
    (Aggregates across identities and expressions.)
    """
    summary = (
        df.groupby(["user_id", "condition"])
          .agg(
              mean_rt=("rt_ms", "mean"),
              sd_rt=("rt_ms", "std"),
              skew_rt=("rt_ms", lambda x: skew(x, nan_policy="omit"))
          )
          .reset_index()
    )
    return summary


# ============================================================
# Plot RT distributions
# ============================================================

def plot_rt_kdes_per_user(df: pd.DataFrame,
                          save_dir: str = "./figures/rt_kde_per_user"):
    """
    For each participant, overlay KDEs for Before vs After.
    """
    os.makedirs(save_dir, exist_ok=True)

    for user_id, sub in df.groupby("user_id"):
        if sub["rt_ms"].nunique() <= 1:
            continue

        plt.figure(figsize=(6, 4))
        sns.kdeplot(
            data=sub,
            x="rt_ms",
            hue="condition",
            fill=True,
            common_norm=False,
            alpha=0.5
        )
        plt.title(f"{user_id} — RT distribution (Before vs After PAQ)")
        plt.xlabel("Response Time (ms)")
        plt.ylabel("Density")
        plt.tight_layout()
        path = os.path.join(save_dir, f"{user_id}_RT_kde.png")
        plt.savefig(path, dpi=300)
        print(f"Saved KDE distribution for user {user_id}: {path}")
        plt.close()


def plot_group_lognormal(df: pd.DataFrame,
                         save_dir: str = "./figures/group_lognormal"):
    """
    Group-averaged histogram + log-normal fit per condition.
    """
    os.makedirs(save_dir, exist_ok=True)

    for cond, sub in df.groupby("condition"):
        rt = sub["rt_ms"].dropna()
        rt = rt[rt > 0]  # log-normal needs positive

        if len(rt) < 5:
            continue

        # Fit log-normal
        shape, loc, scale = lognorm.fit(rt, floc=0)

        x = np.linspace(rt.min(), rt.max(), 500)

        plt.figure(figsize=(6, 4))
        plt.hist(rt, bins=30, density=True, alpha=0.4, label="Data")
        plt.plot(x, lognorm.pdf(x, shape, loc, scale), "r-", label="Log-normal fit")
        plt.title(f"Group RT Distribution ({cond})")
        plt.xlabel("Response Time (ms)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(save_dir, f"group_{cond}_lognorm.png")
        plt.savefig(path, dpi=300)
        print(f"Saved group log-normal plot for condition {cond}: {path}")
        plt.close()


# ============================================================
# Optional: trend + group tests (from earlier)
# ============================================================

def analyze_trend_by_leftname(matrix: pd.DataFrame, label: str):
    """
    Per-user regression within a block:
      x = numeric index of the reference image (from column label)
      y = RT
    Tests H0: slope = 0.
    """
    out = []
    for user_id, row in matrix.iterrows():
        y = row.dropna()
        if y.empty:
            continue
        x = np.array([img_to_int(nm) for nm in y.index], dtype=float)

        if len(x) < 3:
            continue

        slope, intercept, r, p, se = linregress(x, y.values.astype(float))
        out.append({
            "user_id": user_id,
            "condition": label,
            "n": len(x),
            "slope": slope,
            "p_value": p,
            "r_value": r,
            "direction": "faster (↓)" if slope < 0 else "slower (↑)"
        })
    return pd.DataFrame(out)


def group_tests(before_df: pd.DataFrame, after_df: pd.DataFrame):
    res = {}
    if not before_df.empty:
        res["before_mean_slope"] = before_df["slope"].mean()
        res["before_ttest"] = ttest_1samp(before_df["slope"], 0.0, nan_policy="omit")
    if not after_df.empty:
        res["after_mean_slope"] = after_df["slope"].mean()
        res["after_ttest"] = ttest_1samp(after_df["slope"], 0.0, nan_policy="omit")

    if not before_df.empty and not after_df.empty:
        common = set(before_df["user_id"]).intersection(set(after_df["user_id"]))
        if len(common) > 0:
            common_list = list(common)
            b = before_df.set_index("user_id").loc[common_list]["slope"]
            a = after_df.set_index("user_id").loc[common_list]["slope"]
            delta = b - a
            res["delta_mean_slope"] = delta.mean()
            res["delta_ttest"] = ttest_1samp(delta, 0.0, nan_policy="omit")
            res["delta_n"] = len(delta)
    return res




def plot_group_descriptive(summary: pd.DataFrame,
                           save_dir: str = "./figures/group_descriptive"):
    """
    Compute group-level descriptive averages and paired t-tests for:
      - mean RT
      - SD of RT
      - skewness of RT
    Produce:
      Figure 3: Mean ± SEM bar (Before vs After)
      Figure 4: SD & Skew bar plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Pivot wide: each metric as column with Before/After
    wide = summary.pivot(index="user_id", columns="condition",
                         values=["mean_rt", "sd_rt", "skew_rt"])
    wide.columns = ['_'.join(col).strip() for col in wide.columns.values]
    wide = wide.dropna()

    metrics = ["mean_rt", "sd_rt", "skew_rt"]
    fig_titles = {
        "mean_rt": "Mean RT ± SEM (Before vs After PAQ)",
        "sd_rt": "Response Time Variance (Before vs After PAQ)",
        "skew_rt": "Response Time Skewness (Before vs After PAQ)"
    }

    # paired t-tests
    for m in metrics:
        pre = wide[f"{m}_Before"]
        post = wide[f"{m}_After"]
        tstat, p = ttest_rel(pre, post)
        print(f"{m}: t={tstat:.3f}, p={p:.4f}")

        # compute group mean ± SEM
        means = [pre.mean(), post.mean()]
        sems  = [sem(pre), sem(post)]

        fig, ax = plt.subplots(figsize=(4, 5))
        ax.bar(["Before", "After"], means, yerr=sems, capsize=8,
               color=["steelblue", "salmon"], alpha=0.8)
        ax.set_ylabel(m.replace("_rt", " (ms)").upper())
        ax.set_title(f"{fig_titles[m]}\n(p = {p:.3g})")
        plt.tight_layout()
        fname = os.path.join(save_dir, f"group_{m}.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"Saved {fname}")

# ============================================================
# (4) Identity × Expression Effects: Repeated Measures ANOVA
# ============================================================

def analyze_identity_expression(df: pd.DataFrame,
                                save_dir: str = "./figures/identity_expression"):
    """
    Run 2-way repeated-measures ANOVA:
      Factors: Condition (Before/After) × Expression (sad/happy)
    Figures:
      Figure 5: Interaction plot (RT vs PAQ × Expression)
      Figure 6: RT per Identity (Before vs After lines)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Keep only sad/happy for clean comparison
    df2 = df[df["expression"].isin(["sad", "happy"])].copy()

    # Average over repetitions per user × identity × expression × condition
    df_mean = (
        df2.groupby(["user_id", "identity", "expression", "condition"], as_index=False)
           .agg(rt_ms=("rt_ms", "mean"))
    )

    # Run 2-way repeated-measures ANOVA
    aov = pg.rm_anova(
        dv="rt_ms",
        within=["condition", "expression"],
        subject="user_id",
        data=df_mean,
        detailed=True
    )
    print("\n=== (4) Identity × Expression: Repeated-Measures ANOVA ===")
    print(aov)

    # Figure 5: Interaction plot (RT vs PAQ for each expression)
    plt.figure(figsize=(6, 5))
    sns.pointplot(
        data=df_mean,
        x="condition", y="rt_ms",
        hue="expression",
        errorbar="se",
        dodge=True,
        markers=["o", "s"],
        capsize=.1
    )
    plt.title("Figure 5 — Interaction: RT vs PAQ × Expression")
    plt.ylabel("Mean RT (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figure5_interaction.png"), dpi=300)
    plt.close()

    # Figure 6: RT per identity, lines connecting pre/post
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_mean,
        x="condition", y="rt_ms",
        hue="identity",
        marker="o"
    )
    plt.title("Figure 6 — RT per Identity (Before vs After)")
    plt.ylabel("Mean RT (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figure6_identity_lines.png"), dpi=300)
    plt.close()

    return aov, df_mean


# ============================================================
# (5) Within- vs Between-Subject Variability
# ============================================================

def analyze_variance_components(df_mean: pd.DataFrame,
                                save_dir: str = "./figures/variance_components"):
    """
    Compute:
      σ²_between = Var_u(mean RT_u)
      σ²_within  = (1/U) * Σ Var_{i,e}(mean RT_{u,i,e})
    Figures:
      Figure 7: Two bars showing variance reduction before→after
    """
    os.makedirs(save_dir, exist_ok=True)

    results = []
    users = df_mean["user_id"].unique()

    for cond in ["Before", "After"]:
        sub = df_mean[df_mean["condition"] == cond]

        # Between-subject variance
        user_means = sub.groupby("user_id")["rt_ms"].mean()
        var_between = np.var(user_means, ddof=1)

        # Within-subject variance: mean over users of variance across identity × expression
        within_list = []
        for u in users:
            u_sub = sub[sub["user_id"] == u]
            if len(u_sub) > 1:
                within_list.append(np.var(u_sub["rt_ms"], ddof=1))
        var_within = np.mean(within_list) if within_list else np.nan

        results.append({
            "condition": cond,
            "var_between": var_between,
            "var_within": var_within
        })

    var_df = pd.DataFrame(results)
    print("\n=== (5) Variance Components ===")
    print(var_df)

    # Compute percentage reduction
    vb_red = 100 * (1 - var_df.loc[var_df["condition"]=="After","var_between"].values[0] /
                         var_df.loc[var_df["condition"]=="Before","var_between"].values[0])
    vw_red = 100 * (1 - var_df.loc[var_df["condition"]=="After","var_within"].values[0] /
                         var_df.loc[var_df["condition"]=="Before","var_within"].values[0])
    print(f"Between-subject variance reduction: {vb_red:.1f}%")
    print(f"Within-subject variance reduction: {vw_red:.1f}%")

    # Figure 7: Bar plot
    fig, ax = plt.subplots(figsize=(5, 5))
    width = 0.35
    x = np.arange(2)
    ax.bar(x - width/2, var_df["var_between"], width, label="Between-subject")
    ax.bar(x + width/2, var_df["var_within"], width, label="Within-subject")
    ax.set_xticks(x)
    ax.set_xticklabels(var_df["condition"])
    ax.set_ylabel("Variance (ms²)")
    ax.set_title(f"Figure 7 — Variance Components\n(↓ {vb_red:.1f}% between, ↓ {vw_red:.1f}% within)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figure7_variance_components.png"), dpi=300)
    plt.close()

    return var_df


# ============================================================
# (6) Response-Time Distribution Fitting: ex-Gaussian
# ============================================================

def fit_exgaussian(rt):
    """
    Fit an ex-Gaussian distribution to a 1D array of RTs (in ms).

    SciPy's exponnorm uses parameters:
        K = tau / sigma, loc = mu, scale = sigma

    We convert back to the more interpretable (mu, sigma, tau):
        mu   : Gaussian mean      (typical decision latency)
        sigma: Gaussian SD        (decision noise)
        tau  : exponential mean   (slow tail / hesitation)
    """
    rt = np.asarray(rt, dtype=float)
    rt = rt[np.isfinite(rt)]
    rt = rt[rt > 0]          # ex-Gaussian is defined on t > 0
    if rt.size < 10:
        raise ValueError("Not enough RT samples to fit ex-Gaussian")

    # MLE fit; allow loc to vary
    K, loc, scale = exponnorm.fit(rt)
    mu = loc
    sigma = scale
    tau = K * sigma
    return mu, sigma, tau, (K, loc, scale)

def analyze_exgaussian(df: pd.DataFrame,
                       save_dir: str = "./figures/exgaussian"):
    """
    (6) Response-Time Distribution Modeling

    - Fit ex-Gaussian per condition (Before / After).
    - Figure 8: KDE density with fitted ex-Gaussian overlays.
    - Figure 9: Bar plots of fitted parameters (mu, sigma, tau).
    """
    os.makedirs(save_dir, exist_ok=True)

    params = {}   # store mu, sigma, tau per condition
    pdf_curves = {}

    # ---- Fit per condition ----
    for cond, sub in df.groupby("condition"):
        rt = sub["rt_ms"].dropna()
        rt = rt[rt > 0]

        mu, sigma, tau, (K, loc, scale) = fit_exgaussian(rt)
        params[cond] = {"mu": mu, "sigma": sigma, "tau": tau}

        x = np.linspace(rt.min(), rt.max(), 500)
        pdf = exponnorm.pdf(x, K, loc=loc, scale=scale)
        pdf_curves[cond] = (x, pdf)

        print(f"[ex-Gaussian] {cond}: mu={mu:.1f}, sigma={sigma:.1f}, tau={tau:.1f}")

    # -------------------------------------------------
    # Figure 8: density curves with ex-Gaussian overlays
    # -------------------------------------------------
    plt.figure(figsize=(7, 5))

    for cond, sub in df.groupby("condition"):
        rt = sub["rt_ms"].dropna()
        rt = rt[rt > 0]
        sns.kdeplot(rt, label=f"{cond} data", fill=False)

        x, pdf = pdf_curves[cond]
        plt.plot(x, pdf, linestyle="--", linewidth=2,
                 label=f"{cond} ex-Gauss fit")

    plt.xlabel("Response Time (ms)")
    plt.ylabel("Density")
    plt.title("Figure 8 — RT Distributions with ex-Gaussian Fits")
    plt.legend()
    plt.tight_layout()
    f8_path = os.path.join(save_dir, "figure8_exgauss_density.png")
    plt.savefig(f8_path, dpi=300)
    plt.close()
    print(f"Saved {f8_path}")

    # -------------------------------------------------
    # Figure 9: parameter bar plots (mu, sigma, tau)
    # -------------------------------------------------
    conds = list(params.keys())
    metrics = ["mu", "sigma", "tau"]
    nice_names = {
        "mu": r"$\mu$ (drift mean)",
        "sigma": r"$\sigma$ (decision variability)",
        "tau": r"$\tau$ (tail / slow responses)"
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
    for ax, m in zip(axes, metrics):
        vals = [params[c][m] for c in conds]
        ax.bar(conds, vals, color=["steelblue", "salmon"])
        ax.set_title(nice_names[m])
        ax.set_ylabel("ms")
    plt.suptitle("Figure 9 — Fitted ex-Gaussian Parameters per Condition")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    f9_path = os.path.join(save_dir, "figure9_exgauss_params.png")
    plt.savefig(f9_path, dpi=300)
    plt.close()
    print(f"Saved {f9_path}")

    return params

# ============================================================
# (7) Summarize per-user mean/variance RT (Before vs After)
# ============================================================
def summarize_rt_by_user(df: pd.DataFrame,
                         save_path: str = "./figures/user_summary_before_after.csv"):
    """
    Compute per-user mean and variance of response time (RT)
    for each condition (Before / After).
    Returns exactly two rows per user.
    """
    # Drop missing data
    df = df.dropna(subset=["rt_ms", "user_id", "condition"])

    # Compute mean and variance per user × condition
    summary = (
        df.groupby(["user_id", "condition"], as_index=False)
          .agg(
              mean_rt=("rt_ms", "mean"),
              var_rt=("rt_ms", "var"),
              n_trials=("rt_ms", "count")
          )
          .sort_values(["user_id", "condition"])
    )

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    summary.to_csv(save_path, index=False)
    print(f"\n--- Saved per-user mean/variance summary to {save_path} ---")
    print(summary)
    return summary

# ============================================================
# Main
# ============================================================

def main():
    # 1) Load data
    pairwise_mat, stage2_mat = build_block_matrices("./submissions/pairwise")

    print("Users with pairwise_early (Before):", len(pairwise_mat))
    print("Users with stage2_from_paq (After):", len(stage2_mat))

    # 2) Per-user RT vs stimuli curves
    plot_user_shared_index_with_identity(pairwise_mat, stage2_mat)

    # 3) Build long DF
    df = build_long_dataframe(pairwise_mat, stage2_mat)
    print("\nLong-form RT dataframe head:")
    print(df.head())

    # 4) Summary stats per participant × condition
    summary = compute_summary_stats(df)
    print("\n--- Per-user summary (mean, SD, skew) ---")
    print(summary.to_string(index=False))

    # 5) KDE per participant
    plot_rt_kdes_per_user(df)

    # 6) Group histograms + log-normal fits
    plot_group_lognormal(df)

    # 7) Optional: trend & group tests (as before)
    before_df = analyze_trend_by_leftname(pairwise_mat, "Before")
    after_df  = analyze_trend_by_leftname(stage2_mat,  "After")
    print("\n--- Per-user trend (Before) ---")
    print(before_df.sort_values("p_value").to_string(index=False) if not before_df.empty else "No data.")
    print("\n--- Per-user trend (After) ---")
    print(after_df.sort_values("p_value").to_string(index=False) if not after_df.empty else "No data.")
    summary_tests = group_tests(before_df, after_df)
    print("\n--- Group-level trend tests ---")
    for k, v in summary_tests.items():
        print(f"{k}: {v}")

    # 8) Simple paired test on mean RT Before vs After
    wide = summary.pivot(index="user_id", columns="condition", values="mean_rt")
    if {"Before", "After"}.issubset(wide.columns):
        pre = wide["Before"].dropna()
        post = wide["After"].loc[pre.index]
        if len(pre) > 0:
            stat, p = wilcoxon(pre, post)
            print(f"\nWilcoxon test on mean RT (Before vs After): stat={stat:.3f}, p={p:.4g}")
            
    # 9) Descriptive group averages (Figures 3–4)
    plot_group_descriptive(summary)
    
    # 10) Identity × Expression ANOVA (Figures 5–6)
    aov, df_mean = analyze_identity_expression(df)
    var_df = analyze_variance_components(df_mean)
    
    # 11) ex-Gaussian fitting (Figures 8–9)
    params_exg = analyze_exgaussian(df)

    # 12) Summarize per-user mean/variance RT
    user_summary = summarize_rt_by_user(df)


if __name__ == "__main__":
    main()
