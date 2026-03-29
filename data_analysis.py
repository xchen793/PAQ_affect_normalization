import os, json
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, ttest_1samp

# ----------------------------
# Helpers
# ----------------------------

def img_to_int(name: str) -> int:
    """
    Robustly extract an integer index from names like:
      '011.png', 'bm_sad_11', 'bm_sad_011.png'
    """
    base = os.path.basename(name)          # e.g. '011.png' or 'bm_sad_11'
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
    stem = os.path.splitext(base)[0]   # strip '.png' if present
    parts = stem.split("_")

    # Expect at least [identity, expr, idx]
    if len(parts) >= 3:
        identity = parts[0]
        expr = parts[1]
        idx_str = parts[-1]
    else:
        # fallback: treat everything as identity, unknown expr, numeric suffix
        identity = parts[0]
        expr = ""
        idx_str = parts[-1]

    expr_rank = EXPR_RANK.get(expr, 1)
    img_idx = int(idx_str)
    return identity, expr, expr_rank, img_idx

def build_symmetric_stimulus_order(cols):
    """
    Given a list of column labels (e.g., ['bm_sad_11', 'bm_happy_11', ...]),
    build a symmetric x-axis order:

      Left  side: all SAD stimuli (expr_rank==0), sorted by *decreasing* index.
      Right side: all HAPPY stimuli (expr_rank==2), sorted by *increasing* index.
      Middle: any NEUTRAL / other expressions (if present), sorted by index.

    Returns:
      ordered_cols: reordered column names
      xtick_labels: human-readable labels (here identical to ordered_cols)
                    e.g. 'bm_sad_11'
    """
    parsed = []
    for c in cols:
        identity, expr, expr_rank, img_idx = parse_identity_expr_idx(c)
        parsed.append((identity, expr, expr_rank, img_idx, c))

    sad_items    = [p for p in parsed if p[2] == 0]      # expr_rank==0
    happy_items  = [p for p in parsed if p[2] == 2]      # expr_rank==2
    middle_items = [p for p in parsed if p[2] not in (0, 2)]

    # sad: decreasing index; happy: increasing index; middle: increasing index
    sad_items.sort(key=lambda p: -p[3])       # -img_idx
    happy_items.sort(key=lambda p: p[3])      #  img_idx
    middle_items.sort(key=lambda p: p[3])

    ordered = sad_items + middle_items + happy_items

    ordered_cols = []
    xtick_labels = []
    for identity, expr, expr_rank, img_idx, col in ordered:
        ordered_cols.append(col)
        # Explicit identity + expression + index, exactly what you asked for
        xtick_labels.append(f"{identity}_{expr}_{img_idx}")

    return ordered_cols, xtick_labels

# ----------------------------
# Load -> Two matrices
# ----------------------------

def build_block_matrices(data_dir: str = "./submissions/pairwise"):
    """
    Returns:
      pairwise_mat: rows=user_id, cols='[folder]_[index]', values=rt_ms
                    (block == 'pairwise_early' --> Before PAQ)
      stage2_mat:   rows=user_id, cols='[folder]_[index]', values=rt_ms
                    (block == 'stage2_from_paq' --> After PAQ)

    Example entry:
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
                blk    = e.get("block")
                folder = e.get("folder", "")
                left   = e.get("left_name")
                rt     = e.get("rt_ms", np.nan)

                if not folder or not left:
                    continue

                idx = img_to_int(left)             # 011.png -> 11
                col_key = f"{folder}_{idx}"        # e.g. 'bm_sad_11'

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

# ----------------------------
# Plotting
# ----------------------------

def plot_user_shared_index_with_identity(pairwise_mat: pd.DataFrame,
                                         stage2_mat: pd.DataFrame,
                                         save_dir: str = "./figures"):
    """
    One figure per user.

    For each user:
      - Take columns that exist in BOTH blocks globally.
      - Drop any columns where this user has no data in either block.
      - Order remaining columns symmetrically:
            sad (decreasing index)  |  happy (increasing index)
      - X-ticks: 'identity_expression_index', e.g. 'bm_sad_11'.

    Figures are saved as {save_dir}/{user_id}_RT_vs_stimuli.png
    """
    os.makedirs(save_dir, exist_ok=True)

    users = pairwise_mat.index.intersection(stage2_mat.index)

    # Global set of reference faces that appear in both blocks somewhere
    global_common_cols = list(set(pairwise_mat.columns) & set(stage2_mat.columns))
    if not global_common_cols:
        print("No common reference faces between pairwise_early and stage2_from_paq.")
        return

    for user_id in users:
        # Take this user's row, restricted to global_common_cols
        pw_full  = pairwise_mat.loc[user_id].reindex(global_common_cols)
        st2_full = stage2_mat.loc[user_id].reindex(global_common_cols)

        # Keep only columns where at least ONE condition has data
        mask = ~(pw_full.isna() & st2_full.isna())
        user_cols = [c for c, keep in zip(global_common_cols, mask) if keep]

        if not user_cols:
            continue  # nothing to plot for this user

        # Build per-user symmetric order and labels
        ordered_cols, xtick_labels = build_symmetric_stimulus_order(user_cols)

        pw_row  = pw_full[ordered_cols]
        st2_row = st2_full[ordered_cols]

        x = np.arange(1, len(ordered_cols) + 1)

        fig, ax = plt.subplots(figsize=(11, 5))

        # Before PAQ (blue)
        ax.plot(
            x,
            pw_row.values.astype(float),
            marker='o', linewidth=2,
            color='blue', label='Before PAQ'
        )

        # After PAQ (red)
        ax.plot(
            x,
            st2_row.values.astype(float),
            marker='o', linewidth=2,
            color='red', label='After PAQ'
        )

        ax.set_xlabel("Reference face stimuli (identity + expression + index; sad ↓ → happy ↑)")
        ax.set_ylabel("Response Time (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.legend(loc='upper right')

        plt.title(f"User {user_id} — RT vs Reference Stimuli (Before vs After PAQ)")
        plt.tight_layout()

        # Save per-user figure
        save_path = os.path.join(save_dir, f"{user_id}_RT_vs_stimuli.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure for user {user_id}: {save_path}")

        plt.close(fig)


# ----------------------------
# Stats (unchanged logic)
# ----------------------------

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

# ----------------------------
# Main
# ----------------------------

def main():
    pairwise_mat, stage2_mat = build_block_matrices("./submissions/pairwise")

    print("Users with pairwise:", len(pairwise_mat))
    print("Users with stage2:", len(stage2_mat))

    print("pairwise matrix (Before PAQ):")
    print(pairwise_mat)
    print("\nstage2 matrix (After PAQ):")
    print(stage2_mat)

    # Plot per user
    plot_user_shared_index_with_identity(pairwise_mat, stage2_mat)

    # Stats
    before_df = analyze_trend_by_leftname(pairwise_mat, "Before PAQ")
    after_df  = analyze_trend_by_leftname(stage2_mat,  "After PAQ")

    print("\n--- Per-user trend (Before) ---")
    print(before_df.sort_values("p_value").to_string(index=False) if not before_df.empty else "No data.")

    print("\n--- Per-user trend (After) ---")
    print(after_df.sort_values("p_value").to_string(index=False) if not after_df.empty else "No data.")

    summary = group_tests(before_df, after_df)
    print("\n--- Group-level tests ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    pairwise_stats = pd.DataFrame({
        "mean_before": pairwise_mat.mean(axis=1, skipna=True),
        "var_before": pairwise_mat.var(axis=1, ddof=1, skipna=True)
    })
    stage2_stats = pd.DataFrame({
        "mean_after": stage2_mat.mean(axis=1, skipna=True),
        "var_after": stage2_mat.var(axis=1, ddof=1, skipna=True)
    })
    combined_stats = pairwise_stats.join(stage2_stats, how="outer")

    print("\n--- Row-wise RT Mean and Variance per User ---")
    print(combined_stats)

if __name__ == "__main__":
    main()
